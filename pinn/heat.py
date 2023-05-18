import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            history_size=50,
            tolerance_grad=1e-7, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",   # better numerical stability
        )

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

        u_x_grad = torch.autograd.grad(
            inputs=pts,
            outputs=u_x,
            grad_outputs=torch.ones_like(u_x).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        u_xx = u_x_grad[:,1:2]
 
        """Residual"""
        # TODO: fix diffusion for multimaterial
        D = self.d_fn(pts)
        residual_diffusion = u_t - D * u_xx # Heat Equation
        residual = residual_diffusion 
        r2 = residual_diffusion**2 

        """Heat Flux"""
        q = -self.d_fn(pts) * u_x

        return u, residual, r2, q

    def train_step(self):

        """Sample pts"""
        pts_ic, u_ic = self.sample_ics()
        pts_col = self.sample_collocation(adaptive_sample=self.adaptive_resample)
        _pts_bc, bc_true, periodic_bc_pts, dirchelet_bc_pts, neumann_bc_pts = self.sample_bcs()

        """Reset grads"""
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

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

        """Loss"""
        loss = self.ic_weight*loss_u_ic + self.phys_weight*loss_physics + self.bc_weight*loss_u_bc + self.res_weight*loss_physics_res_penalty
        loss.backward()
        return loss
    
    def train(self, n_epochs=500, mode='Adam', reporting_frequency=500, phys_weight=None, res_weight=None, bc_weight=None, ic_weight=None):
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
            loss = self.adam.step(self.train_step) if mode == "Adam" else self.lbfgs.step(self.train_step)
            results["loss"].append(loss.item())
            self.loss_history.append(loss.item())
            if it % reporting_frequency == 0:
              print(f"Loss: {loss.item()}")
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
      net, 
      lr=1e-4, 
      collocation_ct=500, 
      boundary_ct=100, 
      initial_ct=100, 
      t_bounds=[0,4], 
      space_bounds=[-2*np.pi,2*np.pi],
      boundary_type=["dirchelet", "dirchelet", "dirchelet", "dirchelet"],
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
        self.dim = 1 + 2
        
        self.loss_fn = nn.MSELoss()
        self.adam = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            history_size=50,
            tolerance_grad=1e-7, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",   # better numerical stability
        )

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

        self.adaptive_resample =  adaptive_resample

    def sample_ics(self):
        # torch.manual_seed(0)

        int_pts = torch.rand((self.initial_ct, self.dim), device=device, requires_grad=False) * (self.space_bounds[1]-self.space_bounds[0]) + self.space_bounds[0]
        edge_pts = (torch.sign(2*torch.rand((self.boundary_ct, self.dim), device=device, requires_grad=False)-1)+1)/2 * (self.space_bounds[1]-self.space_bounds[0]) + self.space_bounds[0]
        pts = torch.vstack([int_pts, edge_pts])
        pts[:,0] = 0 # time at start

        # Gaussian start
        u = self.ic_fn(pts)

        return pts, u
    
    def sample_bcs(self):
        # torch.manual_seed(1)
        quarter = int(self.boundary_ct/4)
        pts_l = torch.ones((quarter, self.dim), device=device, requires_grad=False if self.boundary_type[0] != "neumann" else True ) * self.space_bounds[0]
        pts_r = torch.ones((quarter, self.dim), device=device, requires_grad=False if self.boundary_type[1] != "neumann" else True ) * self.space_bounds[1]
        pts_d = torch.ones((quarter, self.dim), device=device, requires_grad=False if self.boundary_type[1] != "neumann" else True ) * self.space_bounds[0]
        pts_u = torch.ones((quarter, self.dim), device=device, requires_grad=False if self.boundary_type[1] != "neumann" else True ) * self.space_bounds[1]
        pts_l[:,2] = torch.rand((quarter), device=device) * (self.space_bounds[1] - self.space_bounds[0]) + self.space_bounds[0] # randomize y
        pts_r[:,2] = torch.rand((quarter), device=device) * (self.space_bounds[1] - self.space_bounds[0]) + self.space_bounds[0] # randomize y
        pts_d[:,1] = torch.rand((quarter), device=device) * (self.space_bounds[1] - self.space_bounds[0]) + self.space_bounds[0] # randomize x
        pts_u[:,1] = torch.rand((quarter), device=device) * (self.space_bounds[1] - self.space_bounds[0]) + self.space_bounds[0] # randomize x

        t = torch.rand(quarter, device=device, requires_grad=False) * (self.t_bounds[1] - self.t_bounds[0]) + self.t_bounds[0]
        pts_l[:,0] = t
        pts_r[:,0] = t
        pts_d[:,0] = t
        pts_u[:,0] = t

        edge_list = [pts_l, pts_r, pts_d, pts_u]
        periodic_pts = [pts for i,pts in enumerate(edge_list) if self.boundary_type[i] == "periodic"]
        dirchelet_pts = [pts for i,pts in enumerate(edge_list) if self.boundary_type[i] == "dirchelet"]
        neumann_pts = [pts for i,pts in enumerate(edge_list) if self.boundary_type[i] == "neumann"]

        periodic_vals = [pts_r if i == 0 else pts_l for i,pts in enumerate(edge_list) if self.boundary_type[i] == "periodic"]
        dirchelet_vals = [self.bc_fn(pts) for i,pts in enumerate(edge_list) if self.boundary_type[i] == "dirchelet"]
        neumann_vals = [self.bc_fn(pts) for i,pts in enumerate(edge_list) if self.boundary_type[i] == "neumann"]
        bc_true = torch.vstack(periodic_vals+dirchelet_vals+neumann_vals)
        pts = torch.vstack(periodic_pts+dirchelet_pts+neumann_pts)

        return pts, bc_true, periodic_pts, dirchelet_pts, neumann_pts, edge_list
    
    def sample_collocation(self, adaptive_sample=False):
        # torch.manual_seed(2)
        if adaptive_sample:
          initial_ct = self.collocation_ct*10
          pts_rand = torch.rand((initial_ct, self.dim), device=device, requires_grad=True)
          pts = torch.empty_like(pts_rand)
          pts[:,0] = pts_rand[:,0] * (self.t_bounds[1]- self.t_bounds[0]) + self.t_bounds[0] 
          pts[:,1] = pts_rand[:,1] * (self.space_bounds[1]- self.space_bounds[0]) + self.space_bounds[0] 
          pts[:,2] = pts_rand[:,2] * (self.space_bounds[1]- self.space_bounds[0]) + self.space_bounds[0] 
          _u, _r, r2, _q, _qx, _qy = self.governing_eq(pts)
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
          pts[:,2] = pts_rand[:,2] * (self.space_bounds[1]- self.space_bounds[0]) + self.space_bounds[0] 
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
        u_y = grad[:,2:3]

        """Diffusion"""
        u_x_grad = torch.autograd.grad(
            inputs=pts,
            outputs=u_x,
            grad_outputs=torch.ones_like(u_x).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        u_y_grad = torch.autograd.grad(
            inputs=pts,
            outputs=u_y,
            grad_outputs=torch.ones_like(u_y).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        u_xx = u_x_grad[:,1:2]
        u_yy = u_y_grad[:,2:3]

        """Advection"""
        v = torch.ones_like(torch.hstack([u,u]))*1.5
        v[:,0] = 0
        vu = v*u
        vux = vu[:,0:1]
        vuy = vu[:,1:2]

        vu_grad_x = torch.autograd.grad(
            inputs=pts,
            outputs=vux,
            grad_outputs=torch.ones_like(vux).to(device),
            create_graph=True,
            retain_graph=True
        )[0]
        vu_grad_y = torch.autograd.grad(
            inputs=pts,
            outputs=vuy,
            grad_outputs=torch.ones_like(vuy).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        vu_x = vu_grad_x[:,1:2]
        vu_y = vu_grad_y[:,2:3]
        advection = vu_x + vu_y

 
        """Residual"""
        # TODO: fix diffusion
        D = self.d_fn(pts)
        residual_diffusion = u_t - D * (u_xx+u_yy) + advection  # Heat Equation
        residual = residual_diffusion 
        r2 = residual_diffusion**2 

        """Heat Flux"""
        qx = -self.d_fn(pts) * u_x
        qy = -self.d_fn(pts) * u_y
        q = torch.hstack([qx,qy])

        return u, residual, r2, q, qx, qy

    def train_step(self):

        """Sample pts"""
        pts_ic, u_ic = self.sample_ics()
        pts_col = self.sample_collocation(adaptive_sample=self.adaptive_resample)
        _pts_bc, bc_true, periodic_bc_pts, dirchelet_bc_pts, neumann_bc_pts, edge_list = self.sample_bcs()

        """Reset grads"""
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

        """Initial Conditions"""
        u_ic_pred = self.net(pts_ic)
        loss_u_ic = self.loss_fn(u_ic_pred,u_ic)

        """Boundary Pts"""
        # periodic_pred = [self.net(pts) for pts in periodic_bc_pts]
        # dirchelet_pred = [self.net(pts) for pts in dirchelet_bc_pts]
        # TODO: handle neumann
        # neumann_pred = [q for _u, _res, _r2, q, qx, qy in [self.governing_eq(pts) for pts in neumann_bc_pts]]
        # bc_pred = torch.vstack(periodic_pred + dirchelet_pred)
        bc_l = torch.ones_like(edge_list[0][:,1:2], device=device, requires_grad=True)*1
        bc_r = torch.ones_like(edge_list[1][:,1:2], device=device, requires_grad=True)*-1
        bc_d = torch.ones_like(edge_list[2][:,2:3], device=device, requires_grad=True)*0
        bc_u = torch.ones_like(edge_list[3][:,2:3], device=device, requires_grad=True)*0
        _u, _res, _r2, _q, qx_l, _qy = self.governing_eq(edge_list[0])
        _u, _res, _r2, _q, qx_r, _qy = self.governing_eq(edge_list[1])
        _u, _res, _r2, _q, _qx, qy_d = self.governing_eq(edge_list[2])
        _u, _res, _r2, _q, _qx, qy_u = self.governing_eq(edge_list[3])
        bc_true = torch.vstack([bc_l,bc_r,bc_d,bc_u])
        bc_pred = torch.vstack([qx_l,qx_r,qy_d,qy_u])
        loss_u_bc = self.loss_fn(bc_pred, bc_true)

        """Collocation pts"""
        _u , _residual, r2, _q, _qx, _qy = self.governing_eq(pts_col)
        loss_physics = torch.mean(r2)
        loss_physics_res_penalty = torch.max(r2)#torch.maximum(torch.max(r2),torch.Tensor([10000]).to(device)) #torch.max(r2)

        """Known Data"""
        # known_pts = torch.linspace(*self.t_bounds,7).to(device).requires_grad_(True).reshape(-1,1)
        # u_known_true = self.oscillator_true(known_pts)
        # u_known_pred = self.net(self.normalize_pts(nown_pts))
        # loss_known = self.loss_fn(u_known_pred, u_known_true)

        """Loss"""
        loss = self.ic_weight*loss_u_ic + self.phys_weight*loss_physics + self.bc_weight*loss_u_bc + self.res_weight*loss_physics_res_penalty
        loss.backward()
        return loss
    
    def train(self, n_epochs=500, mode='Adam', reporting_frequency=500, phys_weight=None, res_weight=None, bc_weight=None, ic_weight=None):
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
            loss = self.adam.step(self.train_step) if mode == "Adam" else self.lbfgs.step(self.train_step)
            results["loss"].append(loss.item())
            self.loss_history.append(loss.item())
            if it % reporting_frequency == 0:
              print(f"Epoch {it:05d} Loss: {loss.item()}")
              if it > 0:
                if loss.item() < self.loss_history[-1]:
                  torch.save(self.net.state_dict(), f"./models/net.pth")
        self.runs.append(results)
    
    def plot_ics(self):
      with torch.no_grad():
        pts, u = self.sample_ics()
        pts = pts.cpu()
        u = u.cpu()
        fig,ax = plt.subplots(1,1)
        ax.scatter(pts[:,1],pts[:,2], s=1, c=u.flatten(), cmap="plasma")
        ax.set_aspect("equal")
        plt.show()

    def plot_bcs(self):
      pts, u, _,_,_, edge_list = self.sample_bcs()
      pts.cpu()
      fig = go.Figure()
      scatter = go.Scatter3d(
        x=pts[:,1].cpu().detach().flatten(),
        y=pts[:,2].cpu().detach().flatten(),
        z=pts[:,0].cpu().detach().flatten(),
        mode="markers",
        marker=dict(
          size=1.5,
          color=u.flatten().cpu(),
          colorscale="plasma",
          colorbar=dict(thickness=20)
        )
      )
      fig.add_trace(scatter)
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
        # t = torch.linspace(*self.t_bounds,res).to(device)
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
    
    def plot_frames(self,res=200,n=5):
      t = torch.linspace(*self.t_bounds,n).to(device)
      with torch.no_grad():
        for i in range(n):
          self.plot_3d(res,t[i])
              
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
          pts_bc, bc_true, periodic_bc_pts, dirchelet_bc_pts, neumann_bc_pts, edge_list = self.sample_bcs()
          periodic_pred = [self.net(pts) for pts in periodic_bc_pts]
          dirchelet_pred = [self.net(pts) for pts in dirchelet_bc_pts]
          neumann_pred = [q for _u, _res, _r2, q, qx, qy in [self.governing_eq(pts) for pts in neumann_bc_pts]]
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
      _u, _res, r2, q, qx, qy = self.governing_eq(pts)
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
