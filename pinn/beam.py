import os
from pathlib import Path
from typing import List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from tqdm import tqdm
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mlp import MLP

device = 'cuda' if torch.cuda.is_available() else 'device'
print(f"Using {device}")

class PINNSSSBeam:
    def __init__(
      self, 
      net: MLP, 
      lr=1e-4, 
      collocation_ct=10000,
      space_bounds=[-1,1],
      adaptive_resample=True,
      sample_freq=10,
      soft_adapt_weights=True,
      model_prefix="SimplySupportedStaticBeam"
    ) -> None:
        plt.ion()
        self.fig = plt.figure()
        self.net = net.to(device)
        self.loss_fn = nn.MSELoss()
        self.adam = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_history = {"bc": [], "phys": [], "weighted":[]}

        self.s_min: float = space_bounds[0]
        self.s_max: float = space_bounds[1]
        self.s_range = self.s_max - self.s_min

        self.collocation_ct = collocation_ct
        self.adaptive_resample =  adaptive_resample
        self.sample_freq = sample_freq

        self.soft_adapt_weights = soft_adapt_weights
        self.lambda_Temp = 0.1
        self.losses_prev = torch.Tensor([1, 1]).to(device)
        self.weight_phys = 1
        self.weight_bc = 1

        self.it = 0
        self.reporting_frequency = 100
        self.plot_frequency = 100
        self.save_frequency = 100
        self.best_pde_loss = 9999
        self.model_prefix = model_prefix

        self.I = 1
        self.I_fn = lambda pts: -1*(pts-self.s_min)*(pts-self.s_max) + 1
        self.E = 1
        # self.force_fn = lambda pts: 1/(0.1*np.sqrt(2*np.pi)) * torch.exp(-0.5*(pts**2)/(0.1**2))
        self.force_fn = lambda pts: -1/(0.1*np.sqrt(2*np.pi)) * torch.exp(-0.5*(pts**2)/(0.1**2))
        self.force_fn = lambda pts: -3 * torch.sigmoid((pts-0.5)*30)
        self.force_fn = lambda pts: -1 + 0*pts
        self.dim = 1

    def model_path(self, it):
        return Path(os.path.abspath(os.path.dirname(__file__))) / "models" / f"{self.model_prefix}-{it:05d}.pth"
    
    def sample_collocation_adaptive(self):
        initial_ct = self.collocation_ct*10
        pts_rand = torch.rand((initial_ct, self.dim), device=device, requires_grad=True)
        pts = torch.empty_like(pts_rand)
        pts = pts_rand * self.s_range + self.s_min
        results = self.governing_eq(pts)
        r2 = results["r2"]
        e_r2 = torch.mean(r2)
        p_X = (r2 / e_r2) / torch.sum(r2 / e_r2)
        pts_subsample_ix = torch.multinomial(p_X.flatten(), self.collocation_ct, replacement=True)
        pts_subsample = pts[pts_subsample_ix]
        results_subsample = {key: val[pts_subsample_ix] for key,val in results.items()}
        
        return pts_subsample, results_subsample

    def sample_collocation_non_adaptive(self):
        pts_rand = torch.rand((self.collocation_ct,self.dim), device=device, requires_grad=True)
        pts = torch.empty_like(pts_rand)
        pts = pts_rand * self.s_range + self.s_min
        results = self.governing_eq(pts)

        return pts, results
    
    def sample_bcs(self):
        pts = torch.Tensor([
           [self.s_min],
           [self.s_max],
        ]).requires_grad_(True).to(device)

        results = self.governing_eq(pts)
        return pts, results

    def euler_bernoulli(self, pts, w):
        w_x = torch.autograd.grad(
            inputs=pts,
            outputs=w,
            grad_outputs=torch.ones_like(w).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        w_xx = torch.autograd.grad(
            inputs=pts,
            outputs=w_x,
            grad_outputs=torch.ones_like(w_x).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        # EIw_xx = self.E*self.I*w_xx
        EIw_xx = self.E*self.I_fn(pts)*w_xx

        EIw_xxx = torch.autograd.grad(
            inputs=pts,
            outputs=EIw_xx,
            grad_outputs=torch.ones_like(EIw_xx).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        EIw_xxxx = torch.autograd.grad(
            inputs=pts,
            outputs=EIw_xxx,
            grad_outputs=torch.ones_like(EIw_xxx).to(device),
            create_graph=True,
            retain_graph=True
        )[0]


        return {
          "w":          w,
          "w_x":        w_x,
          "w_xx":       w_xx,
          "EIw_xx":   EIw_xx,
          "EIw_xxx":  EIw_xxx,
          "EIw_xxxx": EIw_xxxx,
        }
      
    def governing_eq(self, pts):
        """Predict"""
        w = self.net(pts)
        results = self.euler_bernoulli(pts, w)
        f = self.force_fn(pts)
        residual = results["EIw_xxxx"] - f
        r2 = residual**2
        results["residual"] = residual
        results["r2"] = r2

        return results

    def train_step(self):


        """Reset grads"""
        self.adam.zero_grad()

        """Sample pts"""
        pts_col, results_col = self.sample_collocation_adaptive() if self.adaptive_resample else self.sample_collocation_non_adaptive()
        pts_bc, results_bc = self.sample_bcs()

        loss_physics = torch.mean(results_col["r2"])

        bc_w = results_bc["w"]
        bc_w_xx = results_bc["w_xx"]
        bc_pred = torch.vstack([ bc_w, bc_w_xx ])
        # bc_true = torch.zeros_like(bc_pred)
        loss_bc = torch.mean(bc_pred**2) # simply supported - no vertical displacement, no moment



        """Loss"""
        if self.soft_adapt_weights:
            with torch.no_grad():
                losses = torch.hstack([loss_bc, loss_physics])
                top = torch.exp(self.lambda_Temp * (self.losses_prev - losses))
                lambdas = top / torch.sum(top)
                self.losses_prev = losses
            loss = torch.sum(lambdas * torch.hstack([loss_bc, loss_physics ]))
            loss.backward()
        else:
            loss = self.bc_weight*loss_bc + self.phys_weight*loss_physics 
            loss.backward()

        """Reporting/Saving etc"""
        if self.it %  self.save_frequency == 0:
            torch.save(self.net.state_dict(), self.model_path(self.it))

        if self.it %  self.reporting_frequency == 0:
            print("\n BC / PDE / Weighted", loss_bc.item(), loss_physics.item(), loss.item())

        if self.it %  self.plot_frequency == 0:
            x = torch.linspace(self.s_min,self.s_max,1000,device=device).reshape(-1,1)
            with torch.no_grad():
                w = self.net(x)
            self.fig.clear()
            plt.gca().set_aspect('equal')
            plt.plot(x.flatten().cpu(),w.flatten().cpu())
            plt.ylim([-0.25,0.25])
            plt.plot([self.s_min, self.s_min], [-2, 2], "--", c="gray", lw=0.5)
            plt.plot([self.s_max, self.s_max], [-2, 2], "--", c="gray", lw=0.5)
            plt.plot([self.s_min, self.s_max], [0,0], "--", c="gray", lw=0.5)
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()

        self.it = self.it + 1

        self.loss_history["phys"].append(loss_physics.item())
        self.loss_history["bc"].append(loss_bc.item())
        self.loss_history["weighted"].append(loss.item())
        return loss
    
    def train(self, n_epochs=500, reporting_frequency=None, plot_frequency=None, phys_weight=None, bc_weight=None):

        """Update Params"""
        if phys_weight is not None:
          self.phys_weight = phys_weight
        if bc_weight is not None:
          self.bc_weight = bc_weight
        if reporting_frequency is not None:
          self.reporting_frequency = reporting_frequency
        if plot_frequency is not None:
          self.plot_frequency = plot_frequency

        for it in tqdm(range(n_epochs)):
            loss = self.adam.step(self.train_step) 
    
if __name__ == "__main__":

  MODEL_PREFIX = "SimplySupportedStaticBeam"
  ADAPTIVE_SAMPLING = False
  ADAPTIVE_WEIGHTING = False
  space_bounds = [-1,1]

  pinn = PINNSSSBeam(
      net=MLP(input_dim=1, output_dim=1, hidden_layer_ct=4,hidden_dim=256, act=F.tanh, learnable_act="SINGLE"), 
      lr=1e-3,
      collocation_ct=5000,
      space_bounds=space_bounds,
      adaptive_resample=ADAPTIVE_SAMPLING,
      soft_adapt_weights=ADAPTIVE_WEIGHTING,
      model_prefix=MODEL_PREFIX
  )

  for i in range(10):
    print(f"MetaEpoch: {i}")
    if i == 0:
        pinn.adam.param_groups[0]["lr"] = 1e-3
    if i == 1:
        pinn.adam.param_groups[0]["lr"] = 1e-4
    if i == 2:
        pinn.adam.param_groups[0]["lr"] = 5e-5
    if i == 4:
        pinn.adam.param_groups[0]["lr"] = 1e-5
    if i == 6:
        pinn.adam.param_groups[0]["lr"] = 5e-6
    pinn.train(n_epochs=1000, reporting_frequency=100, plot_frequency=10, phys_weight=1, bc_weight=8)
    # pinn.plot_bcs_and_ics(mode='error', ct=5000)
