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

class PINN_1D:
    def __init__(
      self, 
      net, 
      lr=1e-4, 
      collocation_ct=500, 
      boundary_ct=100, 
      initial_ct=100, 
      t_bounds=[0,4], 
      space_bounds=[-2*np.pi,2*np.pi],
      boundary_type=["dirchelet", "dirchelet"],
      ic_fn=lambda pts: torch.sin(pts[:,1:2]),
      bc_fn=lambda pts: pts[:,0:1]*0,
      d_fn=lambda pts: pts[:,1:2]*0 + 5,
      adaptive_resample=False
    ) -> None:
        self.net = net.to(device)
        self.t_bounds = t_bounds
        self.space_bounds = space_bounds
        self.collocation_ct = collocation_ct
        self.boundary_ct = boundary_ct
        self.initial_ct = initial_ct
        self.dim = 1 + 1
        
        self.loss_fn = nn.MSELoss()
        self.adam = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.runs = []
        self.loss_history = []
        self.boundary_type = boundary_type

        self.ic_weight=1,
        self.bc_weight=1
        self.phys_weight = 0.1
        self.res_weight = 0.05

        self.ic_fn = ic_fn
        self.bc_fn = bc_fn
        self.d_fn = d_fn

        self.lambda_Temp = 0.1
        self.losses_prev = torch.Tensor([1,1,1]).to(device)

        self.adaptive_resample = adaptive_resample

    def sample_ics(self):
        # torch.manual_seed(0)
        pts = torch.rand((self.initial_ct, self.dim), device=device, requires_grad=False) * (self.space_bounds[1]-self.space_bounds[0]) + self.space_bounds[0]
        pts[:,0] = 0 # time at start

        # Always include edges in ic sample
        pts[0,1] = self.space_bounds[0]
        pts[1,1] = self.space_bounds[1]

        # Gaussian start
        u = self.ic_fn(pts)

        return pts, u
    
    def sample_bcs(self):
        # torch.manual_seed(1)
        half = int(self.boundary_ct/2)
        pts_l = torch.ones((half, self.dim), device=device, requires_grad=False if self.boundary_type[0] != "neumann" else True ) * self.space_bounds[0]
        pts_r = torch.ones((half, self.dim), device=device, requires_grad=False if self.boundary_type[1] != "neumann" else True ) * self.space_bounds[1]

        t = torch.rand(half, device=device, requires_grad=False) * (self.t_bounds[1] - self.t_bounds[0]) + self.t_bounds[0]
        pts_l[:,0] = t
        pts_r[:,0] = t

        periodic_pts = [pts for i,pts in enumerate([pts_l, pts_r]) if self.boundary_type[i] == "periodic"]
        dirchelet_pts = [pts for i,pts in enumerate([pts_l, pts_r]) if self.boundary_type[i] == "dirchelet"]
        neumann_pts = [pts for i,pts in enumerate([pts_l, pts_r]) if self.boundary_type[i] == "neumann"]

        periodic_vals = [pts_r if i == 0 else pts_l for i,pts in enumerate([pts_l, pts_r]) if self.boundary_type[i] == "periodic"]
        dirchelet_vals = [self.bc_fn(pts) for i,pts in enumerate([pts_l, pts_r]) if self.boundary_type[i] == "dirchelet"]
        neumann_vals = [self.bc_fn(pts) for i,pts in enumerate([pts_l, pts_r]) if self.boundary_type[i] == "neumann"]
        bc_true = torch.vstack(periodic_vals+dirchelet_vals+neumann_vals)
        pts = torch.vstack(periodic_pts+dirchelet_pts+neumann_pts)

        return pts, bc_true, periodic_pts, dirchelet_pts, neumann_pts
    
    def sample_collocation(self, adaptive_sample=False):
        # torch.manual_seed(2)
        if adaptive_sample:
          initial_ct = self.collocation_ct*10
          pts_rand = torch.rand((initial_ct, self.dim), device=device, requires_grad=True)
          pts = torch.empty_like(pts_rand)
          pts[:,0] = pts_rand[:,0] * (self.t_bounds[1]- self.t_bounds[0]) + self.t_bounds[0] 
          pts[:,1] = pts_rand[:,1] * (self.space_bounds[1]- self.space_bounds[0]) + self.space_bounds[0] 
          _u, _r, r2, _q = self.governing_eq(pts)
          e_r2 = torch.mean(r2)
          p_X = (r2 / e_r2) / torch.sum(r2 / e_r2)
          pts_subsample_ix = torch.multinomial(p_X.flatten(), self.collocation_ct, replacement=True)
          pts_subsample = pts[pts_subsample_ix]
          
          return pts_subsample
        else:
          pts_rand = torch.rand((self.collocation_ct, self.dim), device=device, requires_grad=True)
          pts = torch.empty_like(pts_rand)
          pts[:,0] = pts_rand[:,0] * (self.t_bounds[1]- self.t_bounds[0]) + self.t_bounds[0] 
          pts[:,1] = pts_rand[:,1] * (self.space_bounds[1]- self.space_bounds[0]) + self.space_bounds[0] 
          return pts

    def governing_eq(self, pts):
        """Predict"""
        u = self.net(pts)

        """Partials"""
        grad = torch.autograd.grad(
            inputs=pts,
            outputs=u,
            grad_outputs=torch.ones_like(u).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        u_t = grad[:,0:1]
        u_x = grad[:,1:2]

        D = self.d_fn(pts)
        Du_x = D * u_x

        Du_x_grad = torch.autograd.grad(
            inputs=pts,
            outputs=Du_x,
            grad_outputs=torch.ones_like(Du_x).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        Du_xx = Du_x_grad[:,1:2]
 
        """Residual"""
        # TODO: fix diffusion for multimaterial
        residual_diffusion = u_t - Du_xx # Heat Equation
        residual = residual_diffusion 
        r2 = residual_diffusion**2 

        """Heat Flux"""
        q = -Du_x

        return u, residual, r2, q

    def train_step(self):

        """Sample pts"""
        pts_ic, u_ic = self.sample_ics()
        pts_col = self.sample_collocation(adaptive_sample=self.adaptive_resample)
        _pts_bc, bc_true, periodic_bc_pts, dirchelet_bc_pts, neumann_bc_pts = self.sample_bcs()

        """Reset grads"""
        self.adam.zero_grad()

        """Initial Conditions"""
        u_ic_pred = self.net(pts_ic)
        loss_u_ic = self.loss_fn(u_ic_pred,u_ic)

        """Boundary Pts"""
        periodic_pred = [self.net(pts) for pts in periodic_bc_pts]
        dirchelet_pred = [self.net(pts) for pts in dirchelet_bc_pts]
        neumann_pred = [q for _u, _res, _r2, q in [self.governing_eq(pts) for pts in neumann_bc_pts]]
        bc_pred = torch.vstack(periodic_pred + dirchelet_pred + neumann_pred)
        loss_u_bc = self.loss_fn(bc_pred, bc_true)

        """Collocation pts"""
        _u , _residual, r2, _q = self.governing_eq(pts_col)
        loss_physics = torch.mean(r2)
        loss_physics_res_penalty = torch.max(r2)#torch.maximum(torch.max(r2),torch.Tensor([10000]).to(device)) #torch.max(r2)

        """Known Data"""
        # known_pts = torch.linspace(*self.t_bounds,7).to(device).requires_grad_(True).reshape(-1,1)
        # u_known_true = self.oscillator_true(known_pts)
        # u_known_pred = self.net(known_pts)
        # loss_known = self.loss_fn(u_known_pred, u_known_true)


        with torch.no_grad():
          losses = torch.hstack([loss_u_ic, loss_u_bc, loss_physics])
          top = torch.exp(self.lambda_Temp * (self.losses_prev - losses))
          lambdas = top / torch.sum(top)
          self.losses_prev = losses


        loss = torch.sum(lambdas * torch.hstack([ loss_u_ic, loss_u_bc, loss_physics ]))
        loss.backward()
        """Loss"""
        # loss = self.ic_weight*loss_u_ic + self.phys_weight*loss_physics + self.bc_weight*loss_u_bc + self.res_weight*loss_physics_res_penalty
        # loss.backward()
        return loss
    
    def train(self, n_epochs=500, reporting_frequency=500, phys_weight=None, res_weight=None, bc_weight=None, ic_weight=None):
        if phys_weight is not None:
          self.phys_weight = phys_weight
        if  res_weight is not None:
          self.res_weight = res_weight
        if bc_weight is not None:
          self.bc_weight = bc_weight
        if ic_weight is not None:
          self.ic_weight = ic_weight
        results = {
            "loss": []
        }
        for it in tqdm(range(n_epochs)):
            loss = self.adam.step(self.train_step) 
            results["loss"].append(loss.item())
            self.loss_history.append(loss.item())
        self.runs.append(results)
    
    def plot_3d(self,res=200):
      with torch.no_grad():
        fig = go.Figure()
        t = torch.linspace(*self.t_bounds,res).to(device)
        x = torch.linspace(*self.space_bounds,res).to(device)
        xx, tt = torch.meshgrid(x,t, indexing="xy")
        X = torch.stack([tt,xx]).T.reshape(-1,2)
        y = self.net(X).reshape(res,res).cpu()
        scatter_plot = go.Surface(
          x=tt.cpu(),
          y=xx.cpu(),
          z=y.T,
        )
        fig.add_trace(scatter_plot)
        fig.update_layout(
          coloraxis=dict(colorscale='plasma'), 
          showlegend=False,
        )

        fig.update_scenes(
          xaxis_title_text='t [s]',  
          yaxis_title_text='x [m]',  
          zaxis_title_text='T [deg C]',
          xaxis_showbackground=False,
          yaxis_showbackground=False,
          zaxis_showbackground=False,
        )
        fig.show()
              
    def plot_config_and_res(self):
      all_pts = []
      known_pts = []
      known_vals = []
      known_preds = []

      old_col_ct = self.collocation_ct
      self.collocation_ct = 1000
      for i in range(50):
          # ICs
          pts_ic, u_ic = self.sample_ics()
          known_pts.append(pts_ic)
          known_vals.append(u_ic)
          known_preds.append(self.net(pts_ic))

          # BCs
          pts_bc, bc_true, periodic_bc_pts, dirchelet_bc_pts, neumann_bc_pts = self.sample_bcs()
          periodic_pred = [self.net(pts) for pts in periodic_bc_pts]
          dirchelet_pred = [self.net(pts) for pts in dirchelet_bc_pts]
          neumann_pred = [q for _u, _res, _r2, q in [self.governing_eq(pts) for pts in neumann_bc_pts]]
          bc_pred = torch.vstack(periodic_pred + dirchelet_pred + neumann_pred)
          known_pts.append(pts_bc)
          known_vals.append(bc_true)
          known_preds.append(bc_pred)

          pts_col = self.sample_collocation()
          all_pts.append(pts_col)
      self.collocation_ct = old_col_ct


      known_pts = torch.vstack(known_pts)
      known_vals = torch.vstack(known_vals)
      known_preds = torch.vstack(known_preds)

      # Collocation pts
      pts = torch.vstack(all_pts)
      _u, _res, r2, q = self.governing_eq(pts)
      d = self.d_fn(pts)
      pts = pts.detach().cpu()
      r2 = r2.detach().cpu()
      q = q.detach().cpu()
      d = d.detach().cpu()
      plt.title("Diffusivity")
      plt.scatter(pts[:,0].flatten(), pts[:,1].flatten(), s=0.5, c=d, cmap="plasma")
      plt.colorbar()
      plt.figure()
      plt.title("PDE Residual")
      plt.scatter(pts[:,0].flatten(), pts[:,1].flatten(), s=0.5, c=r2, cmap="plasma")
      plt.colorbar()

      plt.figure()
      plt.title("q")
      plt.scatter(pts[:,0].flatten(), pts[:,1].flatten(), s=0.5, c=q, cmap="plasma")
      plt.colorbar()

      # Boundary pts
      error = (known_vals - known_preds)**2
      pts = known_pts.detach().cpu()
      error = error.detach().cpu()
      plt.figure()
      plt.title("Boundary Error")
      plt.scatter(pts[:,0].flatten(), pts[:,1].flatten(), s=0.5, c=error, cmap="plasma")
      plt.colorbar()
      plt.show()


class PINN_2D:
    def __init__(
      self, 
      net: MLP, 
      ics,
      bcs,
      lr=1e-4, 
      collocation_ct=500, 
      t_bounds=[0,4], 
      space_bounds=[-2*np.pi,2*np.pi],
      d_fn=lambda pts: pts[:,1:2]*0 + 5,
      adaptive_resample=False,
      soft_adapt_weights=False
    ) -> None:
        self.net = net.to(device)
        self.t_bounds = t_bounds
        self.t_min = self.t_bounds[0]
        self.t_max = self.t_bounds[1]
        self.t_range = self.t_max - self.t_min

        self.space_bounds = space_bounds
        self.x_min = self.space_bounds[0]
        self.x_max = self.space_bounds[1]
        self.x_range = self.x_max - self.x_min

        self.y_min = self.space_bounds[0]
        self.y_max = self.space_bounds[1]
        self.y_range = self.y_max - self.y_min

        self.collocation_ct = collocation_ct
        self.dim = 1 + 2
        
        self.loss_fn = nn.MSELoss()
        self.adam = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.loss_history = []
        self.best_loss = 99999


        self.ic_weight=1,
        self.bc_weight=1
        self.phys_weight = 0.1
        self.res_weight = 0.05

        self.d_fn = d_fn

        self.adaptive_resample =  adaptive_resample
        self.BCs: List[BC] = bcs
        self.ICs: List[BC] = ics
        for bc in bcs+ics:
          bc.parent = self
        
        """ReLoBRaLo"""
        self.lambda_Temp = 0.1
        self.losses_prev = torch.Tensor([1,1,1]).to(device)
        # self.lambda_alpha = 0.999
        # self.lambda_rho = 0.99
        # self.losses_init = torch.Tensor([1,1,1]).to(device)
        # self.lambdas = torch.Tensor([1,1,1]).to(device)
        self.soft_adapt_weights = soft_adapt_weights

        self.it = 0
    
    def sample_ics(self):
      for ic in self.ICs:
        ic.sample(cache=True)

    def sample_bcs(self):
      for bc in self.BCs:
        bc.sample(cache=True)
    
    def loss_ic(self):
      preds = []
      truths = []
      for ic in self.ICs:
        pred = ic.predict(cache=True)
        preds.append(pred)
        truths.append(ic.truth)
        
      pred = torch.vstack(preds)
      truth = torch.vstack(truths)
      return self.loss_fn(pred,truth)

    def loss_bc(self):
      preds = []
      truths = []
      for bc in self.BCs:
        bc_pred = bc.predict(cache=True)
        preds.append(bc_pred)
        truths.append(bc.truth)
        
      pred = torch.vstack(preds)
      truth = torch.vstack(truths)
      return self.loss_fn(pred,truth)
    
    def sample_collocation(self, adaptive_sample=False):
        # torch.manual_seed(2)
        if adaptive_sample:
          initial_ct = self.collocation_ct*10
          pts_rand = torch.rand((initial_ct, self.dim), device=device, requires_grad=True)
          pts = torch.empty_like(pts_rand)
          pts[:,0] = pts_rand[:,0] * self.t_range + self.t_min
          pts[:,1] = pts_rand[:,1] * self.x_range + self.x_min
          pts[:,2] = pts_rand[:,2] * self.y_range + self.y_min
          results = self.governing_eq(pts)
          r2 = results["r2"]
          e_r2 = torch.mean(r2)
          p_X = (r2 / e_r2) / torch.sum(r2 / e_r2)
          pts_subsample_ix = torch.multinomial(p_X.flatten(), self.collocation_ct, replacement=True)
          pts_subsample = pts[pts_subsample_ix]
          
          return pts_subsample
        else:
          pts_rand = torch.rand((self.collocation_ct, self.dim), device=device, requires_grad=True)
          pts = torch.empty_like(pts_rand)
          pts[:,0] = pts_rand[:,0] * self.t_range + self.t_min
          pts[:,1] = pts_rand[:,1] * self.x_range + self.x_min
          pts[:,2] = pts_rand[:,2] * self.y_range + self.y_min
          return pts

    def heat_flux(self, pts):
        """Predict"""
        D = self.d_fn(pts)
        T = self.net(pts)

        """Partials"""
        grad = torch.autograd.grad(
            inputs=pts,
            outputs=T,
            grad_outputs=torch.ones_like(T).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        T_x = grad[:,1:2]
        T_y = grad[:,2:3]
        q_x = -D * T_x
        q_y = -D * T_y
        return {"T": T, "T_x": T_x, "T_y": T_y, "q_x": q_x, "q_y": q_y, "D": D}
      
    def governing_eq(self, pts):
        """Predict"""
        T = self.net(pts)
        D = self.d_fn(pts)

        """Partials"""
        grad = torch.autograd.grad(
            inputs=pts,
            outputs=T,
            grad_outputs=torch.ones_like(T).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        T_t = grad[:,0:1]
        T_x = grad[:,1:2]
        T_y = grad[:,2:3]
        q_x = D*T_x
        q_y = D*T_y

        """Diffusion"""
        DT_xx = torch.autograd.grad(
            inputs=pts,
            outputs=q_x,
            grad_outputs=torch.ones_like(q_x).to(device),
            create_graph=True,
            retain_graph=True
        )[0][:,1:2]

        DT_yy = torch.autograd.grad(
            inputs=pts,
            outputs=q_y,
            grad_outputs=torch.ones_like(q_y).to(device),
            create_graph=True,
            retain_graph=True
        )[0][:,2:3]

        diffusion = DT_xx + DT_yy

        """Advection"""
        v = torch.ones_like(torch.hstack([T,T]))
        # vy = (torch.sigmoid((pts[:,1:2] + 1.05*np.pi)*6) + torch.sigmoid((-pts[:,1:2] + 1.05*np.pi)*6))-1
        v[:,0] = 0
        v[:,1] = 0
        # v[:,1:2] = 2
        vT = v*T
        vxT = vT[:,0:1]
        vyT = vT[:,1:2]

        vT_x= torch.autograd.grad(
            inputs=pts,
            outputs=vxT,
            grad_outputs=torch.ones_like(vxT).to(device),
            create_graph=True,
            retain_graph=True
        )[0][:,1:2]

        vT_y= torch.autograd.grad(
            inputs=pts,
            outputs=vyT,
            grad_outputs=torch.ones_like(vyT).to(device),
            create_graph=True,
            retain_graph=True
        )[0][:,2:3]

        # vT_x = vT_grad[:,1:2]
        # vT_y = vT_grad[:,2:3]

        advection = vT_x + vT_y
        # advection = 2*T_y

        """Residual"""
        residual = T_t - diffusion + advection  # Heat Equation
        r2 = residual**2 

        return {"T": T, "residual": residual, "r2": r2}

    def train_step(self):

        """Sample pts"""
        pts_col = self.sample_collocation(adaptive_sample=self.adaptive_resample)
        self.sample_ics()
        self.sample_bcs()

        """Reset grads"""
        self.adam.zero_grad()

        """Boundaries"""
        loss_bc = self.loss_bc()
        loss_ic = self.loss_ic()


        """Collocation pts"""
        summary = self.governing_eq(pts_col)
        r2 = summary["r2"]
        loss_physics = torch.mean(r2)
        loss_physics_res_penalty = torch.max(r2)

        """Loss"""

        if self.soft_adapt_weights:
          with torch.no_grad():
            losses = torch.hstack([loss_ic, loss_bc, loss_physics])
            top = torch.exp(self.lambda_Temp * (self.losses_prev - losses))
            lambdas = top / torch.sum(top)
            self.losses_prev = losses
            # failed attempt at ReLoBRraLo
            # alpha = self.lambda_alpha
            # rho = torch.bernoulli(torch.tensor(self.lambda_rho))
            # if self.it == 0:
            #   alpha = 1
            #   rho = 1
            # elif self.it == 1:
            #   alpha = 0
            #   rho = 1
            # losses = torch.hstack([loss_ic, loss_bc, loss_physics])
            # lambda_hats_prev = torch.exp(losses / (self.losses_prev*self.lambda_Temp + 1e-3)) 
            # lambda_hats_init = torch.exp(losses / (self.losses_init*self.lambda_Temp + 1e-3)) 
            # n_losses = losses.shape[0]
            # lambda_bal_prev = n_losses * lambda_hats_prev / torch.sum(lambda_hats_prev)
            # lambda_bal_init = n_losses * lambda_hats_init / torch.sum(lambda_hats_init)
            # lambda_hist = rho*self.lambdas - (1-rho)*lambda_bal_init
            # lambdas = torch.minimum(alpha * lambda_hist - (1-alpha) * lambda_bal_prev, 10*torch.ones_like(self.lambdas))
            # self.lambdas = lambdas
            # if self.it == 0:
            #   self.losses_init = losses
            # self.loss_prev = losses


          loss = torch.sum(lambdas * torch.hstack([ loss_ic, loss_bc, loss_physics ]))
          loss.backward()
        else:
          loss = self.ic_weight*loss_ic + self.bc_weight*loss_bc + self.res_weight*loss_physics_res_penalty + self.phys_weight*loss_physics 
          loss.backward()
        if self.it % 100 == 0:
          print("\nIC / BC / PDE", loss_ic.item(), loss_bc.item(), loss_physics.item())
          if loss < self.best_loss:
            self.best_loss = loss
            torch.save(self.net.state_dict(), model_path)
        self.it = self.it + 1
        return loss
    
    def train(self, n_epochs=500, reporting_frequency=500, phys_weight=None, res_weight=None, bc_weight=None, ic_weight=None):
        if phys_weight is not None:
          self.phys_weight = phys_weight
        if  res_weight is not None:
          self.res_weight = res_weight
        if bc_weight is not None:
          self.bc_weight = bc_weight
        if ic_weight is not None:
          self.ic_weight = ic_weight
        for it in tqdm(range(n_epochs)):
            loss = self.adam.step(self.train_step) 
            self.loss_history.append(loss.item())
    
    def plot_bcs_and_ics(self, mode="Truth", ct=1000):
      fig = go.Figure()
      for ic in self.ICs:
        ic.add_truth_to_fig(fig, ct=ct, z=True) if mode == 'Truth' else ic.add_error_to_fig(fig, ct=ct)
      for bc in self.BCs:
        bc.add_truth_to_fig(fig, ct=ct) if mode == 'Truth' else bc.add_error_to_fig(fig, ct=ct)

      fig.update_layout(
        coloraxis=dict(colorscale='plasma'), 
        showlegend=False,
      )

      fig.update_scenes(
        xaxis_title_text='x [m]',  
        yaxis_title_text='y [m]',  
        zaxis_title_text='t [s]',
        xaxis_showbackground=False,
        yaxis_showbackground=False,
        zaxis_showbackground=False,
      )
      fig.show()
  
    
    def plot_3d(self,res=200, t=0):
      with torch.no_grad():
        fig = go.Figure()
        t = torch.ones((res*res,1),device=device)*t
        x = torch.linspace(*self.space_bounds,res).to(device)
        y = torch.linspace(*self.space_bounds,res).to(device)
        xx, yy = torch.meshgrid(x,y, indexing="xy")

        X = torch.stack([xx,yy]).T.reshape(-1,2)
        pts = torch.hstack([t,X])
        z = self.net(pts).reshape(res,res).cpu()
        scatter_plot = go.Surface(
          x=xx.cpu(),
          y=yy.cpu(),
          z=z.T,
        )
        fig.add_trace(scatter_plot)
        # scatter = go.Scatter3d(
        #   x=X[:,0].cpu(),
        #   y=X[:,1].cpu(),
        #   z=z.flatten().cpu(),
        #   mode="markers",
        #   marker=dict(
        #     size=1.5,
        #     color=z.flatten().cpu(),
        #     colorscale="plasma"
        #   )
        # )
        # fig.add_trace(scatter)
        fig.update_layout(
          coloraxis=dict(colorscale='plasma'), 
          showlegend=False,
        )

        fig.update_scenes(
          xaxis_title_text='x [s]',  
          yaxis_title_text='y [m]',  
          zaxis_title_text='T [deg C]',
          xaxis_showbackground=False,
          yaxis_showbackground=False,
          zaxis_showbackground=False,
        )
        fig.show()
    
    def plot_D(self, ct):
      pts = torch.rand((ct,3))*(self.space_bounds[1]-self.space_bounds[0]) + self.space_bounds[0]
      pts[:,0] = 0
      d = self.d_fn(pts).flatten()
      print(torch.max(d))
      print(torch.min(d))
      plt.figure()
      plt.scatter(pts[:,1], pts[:,2], s=1, c=d)
      plt.colorbar()
      plt.show()

    def plot_IC(self, ct):
      pts = torch.rand((ct,3))*(self.space_bounds[1]-self.space_bounds[0]) + self.space_bounds[0]
      pts[:,0] = 0
      u = self.ICs[0].truth_fn(self, pts.to(device))
      print(torch.max(u))
      print(torch.min(u))
      plt.figure()
      plt.scatter(pts[:,1], pts[:,2], s=1, c=u.detach().cpu().flatten())
      plt.colorbar()
      plt.show()

    def plot_frames(self,res=200,n=5):
      t = torch.linspace(*self.t_bounds,n).to(device)
      with torch.no_grad():
        for i in range(n):
          self.plot_3d(res,t[i])

class BC:
  def __init__(self, 
    truth_fn, 
    pred_fn, 
    ct=1000,
    axis_bounds=[[0,4*np.pi], [-4*np.pi, 4*np.pi], [-4*np.pi, -4*np.pi]], 
    requires_grad=False
  ) -> None:
    self.parent: Union[PINN_2D, None] = None
    self.ct = ct
    self.truth_fn = truth_fn
    self.pred_fn = pred_fn
    self.requires_grad = requires_grad
    self.axis_bounds = torch.Tensor(axis_bounds).to(device).requires_grad_(self.requires_grad)
    self.axis_mins = self.axis_bounds[:,0]
    self.axis_ranges = self.axis_bounds[:,1] - self.axis_bounds[:,0]
    self.pts = None
    self.truth = None
    self.pred = None
  
  def sample(self, cache=False, predict=False, ct=None):
    pts = torch.rand((self.ct if ct == None else ct, self.parent.dim), device=device, requires_grad=self.requires_grad)*self.axis_ranges+ self.axis_mins
    
    truth = self.truth_fn(self.parent, pts)
    if cache:
      self.pts = pts
      self.truth = truth
    pred = None
    if predict:
      pred = self.predict(pts, cache=cache)

    return {"pts": pts, "truth": truth, "pred": pred}
  
  def predict(self, pts=None, cache=False):
    pred = self.pred_fn(self.parent, pts ) if pts != None else self.pred_fn(self.parent, self.pts)
    if cache:
      self.pred = pred
    return pred
  
  def add_truth_to_fig(self,fig,ct,z=False):
    result = self.sample(cache=False, predict=False, ct=ct)
    pts = result["pts"].detach().cpu()
    true = result["truth"].detach().cpu()
    scat = go.Scatter3d(
      x=pts[:,1],
      y=pts[:,2],
      z=true if z else pts[:,0],
      mode="markers",
      marker=dict(
        size=2,
        color=true.flatten(),
        colorscale="plasma"
      )
    )
    fig.add_trace(scat)
  
  def add_error_to_fig(self, fig, ct):
    result = self.sample(cache=False, predict=True, ct=ct)
    pts = result["pts"].detach().cpu()
    true = result["truth"].detach()
    pred = result["pred"].detach()
    err = (pred-true)**2
    err = err.cpu()
    scat = go.Scatter3d(
      x=pts[:,1],
      y=pts[:,2],
      z=pts[:,0],
      mode="markers",
      marker=dict(
        size=2,
        color=err.flatten(),
        colorscale="plasma"
      )
    )
    fig.add_trace(scat)
  
def get_d(pts):
  return torch.sqrt(torch.sum(pts[:,1:]**2, axis=1)).reshape(-1,1)

if __name__ == "__main__":

  model_path = Path(os.path.abspath(os.path.dirname(__file__))) / "models" / "last_runs.pth"
  ADAPTIVE_SAMPLING = True
  ADAPTIVE_WEIGHTING = True
  t_bounds = [0, 16*np.pi]
  s_bounds = [-4*np.pi, 4*np.pi]
  d_fn = lambda pts: pts[:,1:2]*0 + 4
  # d_fn = lambda pts: torch.sigmoid(pts[:,1:2]*6)*16 + torch.sigmoid(pts[:,2:3]*6)*8 + 0.2
  # d_fn = lambda pts: (torch.sigmoid((pts[:,1:2] + np.pi)*6) + torch.sigmoid((-pts[:,1:2] + np.pi)*6))*10-10 + 0.3
  d_fn = lambda pts: torch.sigmoid(torch.sqrt(torch.sum(pts[:,1:]**2,axis=1).reshape(-1,1) / (16*np.pi*np.pi)))*6 + 0.2
  d_fn = lambda pts: torch.sigmoid(torch.sin(pts[:,1:2])*torch.sin(pts[:,2:3]))*6 + 0.2

  # bc_l_fn = lambda pinn, pts: 1-torch.abs(pts[:,2:3]/pinn.y_range)*2
  # bc_r_fn = lambda pinn, pts: bc_l_fn(pinn, pts)*-1
  # bc_l_fn = lambda pinn, pts: torch.sin(pts[:,0:1])*4
  # bc_r_fn = lambda pinn, pts: torch.sin(pts[:,0:1]/2)*torch.sin(pts[:,2:3])*4
  # bc_l_pred_fn = lambda pinn, pts: pinn.heat_flux(pts)["q_x"]
  # bc_r_pred_fn = lambda pinn, pts: pinn.heat_flux(pts)["q_x"]
  bc_l_fn = lambda pinn, pts: pts[:,1:2]*0
  bc_r_fn = lambda pinn, pts: pts[:,1:2]*0 
  bc_l_pred_fn = lambda pinn, pts: pinn.heat_flux(pts)["q_x"]
  bc_r_pred_fn = lambda pinn, pts: pinn.heat_flux(pts)["q_x"]
  # bc_u_fn = lambda pinn, pts: torch.sign(pts[:,2:3])*2
  # bc_d_fn = lambda pinn, pts: -torch.sign(pts[:,2:3])*2
  bc_u_fn = lambda pinn, pts: pts[:,1:2]*0 + 2
  bc_d_fn = lambda pinn, pts: pts[:,1:2]*0 + 2
  bc_u_pred_fn = lambda pinn, pts: pinn.heat_flux(pts)["q_y"]
  bc_d_pred_fn = lambda pinn, pts: pinn.heat_flux(pts)["q_y"]
  # def up_down_periodic_bottom(pinn: PINN_2D, pts):
  #   top = pts.clone()
  #   top[:,2:3] = s_bounds[1]
  #   # T = pinn.net(pts)
  #   # T_top = pinn.net(top)
  #   q_y = pinn.heat_flux(pts)["T_y"]
  #   q_y_t = pinn.heat_flux(top)["T_y"]
  #   return (q_y - q_y_t)
  #   # return (T-T_top)

  # def up_down_periodic_top(pinn: PINN_2D, pts):
  #   bot = pts.clone()
  #   bot[:,2:3] = s_bounds[0]
  #   T = pinn.net(pts)
  #   # T_b = pinn.net(bot)
  #   q_y = pinn.heat_flux(pts)["T_y"]
  #   q_y_b = pinn.heat_flux(bot)["T_y"]
  #   return (q_y - q_y_b)
  #   # return (T - T_b)



  gaussian_height = 15
  gaussian_sigma = 5
  ic_fn = lambda pinn, pts:  gaussian_height*(1/(gaussian_sigma*np.sqrt(2*np.pi)))*torch.exp(-(get_d(pts)**2) / (2*gaussian_sigma**2))
  ic_fn = lambda pinn, pts: pts[:,1:2]*0
  # ic_fn = lambda pinn, pts: (2-(torch.sigmoid((pts[:,1:2] + np.pi)*6) + torch.sigmoid((-pts[:,1:2] + np.pi)*6)))*10-5
  ic_pred_fn = lambda pinn, pts: pinn.net(pts)


  bc_d = BC(
    ct=1000,
    truth_fn=bc_d_fn, 
    pred_fn=bc_d_pred_fn,
    axis_bounds=[t_bounds, s_bounds, [s_bounds[0], s_bounds[0]]], 
    requires_grad=True,
  )
  bc_u = BC(
    ct=1000,
    truth_fn=bc_u_fn, 
    pred_fn=bc_u_pred_fn,
    axis_bounds=[t_bounds, s_bounds, [s_bounds[1], s_bounds[1]]], 
    requires_grad=True,
  )
  bc_l = BC(
    ct=1000,
    truth_fn=bc_l_fn,
    pred_fn=bc_l_pred_fn,
    axis_bounds=[t_bounds, [s_bounds[0], s_bounds[0]], s_bounds], 
    requires_grad=True,
  )
  bc_r = BC(
    ct=1000,
    truth_fn=bc_r_fn,
    pred_fn=bc_r_pred_fn,
    axis_bounds=[t_bounds, [s_bounds[1], s_bounds[1]], s_bounds], 
    requires_grad=True,
  )
  ic = BC(
    ct=1000,
    truth_fn=ic_fn,
    pred_fn=ic_pred_fn,
    axis_bounds=[[t_bounds[0], t_bounds[0]], s_bounds, s_bounds], 
    requires_grad=False,
  )
  bcs = [bc_d, bc_u, bc_l, bc_r]
  ics = [ic]

  pinn = PINN_2D(
      net=MLP(input_dim=3, hidden_layer_ct=4,hidden_dim=256, act=F.tanh, learnable_act="SINGLE"), 
      bcs=bcs,
      ics=ics,
      collocation_ct=2500, 
      d_fn=d_fn,
      t_bounds=t_bounds, 
      space_bounds=s_bounds,
      lr=1e-3,
      adaptive_resample=ADAPTIVE_SAMPLING,
      soft_adapt_weights=ADAPTIVE_WEIGHTING
  )
  # pinn.plot_D(ct=10000)
  # pinn.plot_IC(ct=4000)


  for i in range(15):
    print(f"MetaEpoch: {i}")
    if i == 0:
        pinn.adam.param_groups[0]["lr"] = 1e-3
    if i == 1:
        pinn.adam.param_groups[0]["lr"] = 1e-4
    if i == 2:
        pinn.adam.param_groups[0]["lr"] = 5e-5
    if i == 5:
        pinn.adam.param_groups[0]["lr"] = 1e-5
    if i == 7:
        pinn.adam.param_groups[0]["lr"] = 5e-6
    pinn.train(n_epochs=1000, reporting_frequency=100, phys_weight=1, res_weight=0.01, bc_weight=8, ic_weight=4)
