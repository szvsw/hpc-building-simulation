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
      bc_ct=1000,
      space_bounds=[-1,1],
      I_a_bounds=[0,2],
      adaptive_resample=True,
      sample_freq=10,
      soft_adapt_weights=True,
      model_prefix="SimplySupportedStaticBeam"
    ) -> None:
        plt.ion()
        self.fig, self.axs = plt.subplots(3,1,figsize=(8,8))
        self.net = net.to(device)
        self.loss_fn = nn.MSELoss()
        self.adam = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_history = {"bc": [], "phys": [], "weighted":[]}

        self.s_min: float = space_bounds[0]
        self.s_max: float = space_bounds[1]
        self.s_range: float = self.s_max - self.s_min

        self.I_a_min: float = I_a_bounds[0]
        self.I_a_max: float = I_a_bounds[1]
        self.I_a_range = self.I_a_max - self.I_a_min

        self.collocation_ct = collocation_ct
        self.bc_ct = bc_ct
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
        self.I_fn = lambda pts: -(pts[:,0:1])*(pts[:,1:2]-self.s_min)*(pts[:,1:2]-self.s_max) + 1
        self.E = 1
        # self.force_fn = lambda pts: 1/(0.1*np.sqrt(2*np.pi)) * torch.exp(-0.5*(pts**2)/(0.1**2))
        self.force_fn = lambda pts: -1/(0.1*np.sqrt(2*np.pi)) * torch.exp(-0.5*(pts**2)/(0.1**2))
        self.force_fn = lambda pts: -3 * torch.sigmoid((pts-0.5)*30)
        self.force_fn = lambda pts: -1 + 0*pts[:,1:2]
        self.dim = 2

    def model_path(self, it):
        return Path(os.path.abspath(os.path.dirname(__file__))) / "models" / f"{self.model_prefix}-{it:05d}.pth"
    
    def sample_collocation_adaptive(self):
        initial_ct = self.collocation_ct*10
        pts_rand = torch.rand((initial_ct, self.dim), device=device, requires_grad=True)
        pts = torch.empty_like(pts_rand)
        pts[:,0] = pts_rand[:,0] * self.I_a_range + self.I_a_min
        pts[:,1] = pts_rand[:,1] * self.s_range + self.s_min
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
        pts[:,0] = pts_rand[:,0] * self.I_a_range + self.I_a_min
        pts[:,1] = pts_rand[:,1] * self.s_range + self.s_min
        results = self.governing_eq(pts)

        return pts, results
    
    def sample_bcs(self):
        pts_rand = torch.rand((self.bc_ct, self.dim), device=device, requires_grad=True)
        pts = torch.empty_like(pts_rand)
        pts[:,0] = pts_rand[:,0] * self.I_a_range + self.I_a_min
        half = int(self.bc_ct/2)
        pts[:half,1] = pts_rand[:half,1]*0 + self.s_min
        pts[half:,1] = pts_rand[half:,1]*0 + self.s_max

        results = self.governing_eq(pts)
        return pts, results

    def euler_bernoulli(self, pts, w):
        w_grad = torch.autograd.grad(
            inputs=pts,
            outputs=w,
            grad_outputs=torch.ones_like(w).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        w_x = w_grad[:,1:2]

        w_x_grad = torch.autograd.grad(
            inputs=pts,
            outputs=w_x,
            grad_outputs=torch.ones_like(w_x).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        w_xx = w_x_grad[:,1:2]


        # EIw_xx = self.E*self.I*w_xx
        EIw_xx = self.E*self.I_fn(pts)*w_xx

        EIw_xx_grad = torch.autograd.grad(
            inputs=pts,
            outputs=EIw_xx,
            grad_outputs=torch.ones_like(EIw_xx).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        EIw_xxx = EIw_xx_grad[:,1:2]

        EIw_xxx_grad = torch.autograd.grad(
            inputs=pts,
            outputs=EIw_xxx,
            grad_outputs=torch.ones_like(EIw_xxx).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        EIw_xxxx = EIw_xxx_grad[:,1:2]


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
            n_pts = 1000
            n_alphas = 5
            x = torch.linspace(self.s_min,self.s_max,n_pts,device=device)
            I_a = torch.linspace(self.I_a_min, self.I_a_max, n_alphas, device=device)
            xx, II_a = torch.meshgrid(x,I_a, indexing="ij")
            pts = torch.dstack([ II_a, xx ])
            pts_flat = pts.reshape(-1,2)

            with torch.no_grad():
                w = self.net(pts_flat).reshape(n_pts,n_alphas,1)
                I = self.I_fn(pts_flat).reshape(n_pts,n_alphas,1)
                f = self.force_fn(pts_flat).reshape(n_pts,n_alphas,1)

            for i in range(3):
                self.axs[i].clear()
            self.axs[0].set_aspect('equal')

            for i in range(n_alphas):
                self.axs[0].plot(x.flatten().cpu(),w[:,i,:].flatten().cpu(), label=f"a={I_a[i].item():0.2f}", lw=1)

            for i in range(n_alphas):
                self.axs[1].plot(x.flatten().cpu(),I[:,i,:].flatten().cpu(), label=f"a={I_a[i].item():0.2f}", lw=1)

            for i in range(n_alphas):
                self.axs[2].plot(x.flatten().cpu(),f[:,i,:].flatten().cpu(), label=f"a={I_a[i].item():0.2f}", lw=1)
            
            self.axs[0].set_title("Displacement")
            self.axs[0].set_ylim([-0.5,0.1])
            self.axs[1].set_title("Moment of Inertia")
            self.axs[1].set_ylim([0,3.5])
            self.axs[2].set_title("Applied Force")
            # self.axs[2].set_ylim([-1,1])

            for i in range(3):
                self.axs[i].legend()
                self.axs[i].plot([self.s_min, self.s_min], [-4, 4], "--", c="gray", lw=0.5)
                self.axs[i].plot([self.s_max, self.s_max], [-4, 4], "--", c="gray", lw=0.5)
                self.axs[i].plot([self.s_min, self.s_max], [0,0], "--", c="gray", lw=0.5)
            self.fig.tight_layout()
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
  I_a_bounds = [0,2]
  collocation_ct = 5000
  bc_ct = 1000

  pinn = PINNSSSBeam(
      net=MLP(input_dim=2, output_dim=1, hidden_layer_ct=4,hidden_dim=256, act=F.tanh, learnable_act="SINGLE"), 
      lr=1e-3,
      collocation_ct=collocation_ct,
      bc_ct=bc_ct,
      space_bounds=space_bounds,
      I_a_bounds=I_a_bounds,
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
    pinn.train(n_epochs=1000, reporting_frequency=100, plot_frequency=100, phys_weight=1, bc_weight=8)
    # pinn.plot_bcs_and_ics(mode='error', ct=5000)
