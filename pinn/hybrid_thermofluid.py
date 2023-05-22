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

class PINN_2D:
    def __init__(
      self, 
      net: MLP, 
      ics,
      bcs,
      collocation_pts, 
      lr=1e-4, 
      collocation_ct=500, 
      t_bounds=[0,4], 
      space_bounds=[[-2*np.pi,2*np.pi], [-2*np.pi,2*np.pi]],
      adaptive_resample=False,
      soft_adapt_weights=False,
      Rayleigh=1708,
      Prandtl=0.71,
      Richardson=1,
      Peclet=0.1,
      Reynolds=100,
    ) -> None:
        self.net = net.to(device)
        self.t_bounds = t_bounds
        self.t_min = self.t_bounds[0]
        self.t_max = self.t_bounds[1]
        self.t_range = self.t_max - self.t_min

        self.space_bounds = space_bounds
        self.x_min = self.space_bounds[0][0]
        self.x_max = self.space_bounds[0][1]
        self.x_range = self.x_max - self.x_min

        self.y_min = self.space_bounds[1][0]
        self.y_max = self.space_bounds[1][1]
        self.y_range = self.y_max - self.y_min

        self.dim = 1 + 2
        
        self.loss_fn = nn.MSELoss()
        self.adam = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.loss_history = []

        self.ic_weight=1,
        self.bc_weight=1
        self.phys_weight = 0.1
        self.res_weight = 0.05


        self.adaptive_resample =  adaptive_resample
        self.collocation_pts: List[BC] = collocation_pts
        self.BCs: List[BC] = bcs
        self.ICs: List[BC] = ics

        for bc in self.BCs+self.ICs+self.collocation_pts:
          bc.parent = self
        
        self.soft_adapt_weights = soft_adapt_weights
        self.lambda_Temp = 0.1
        self.losses_prev = torch.Tensor([1, 1, 1]).to(device)

        self.Peclet = Peclet #46 #0.1 # Diffusive Transport vs Advective Transport - Low means more diffusive
        self.Richardson = Richardson #0.1 # Buoyancy Strength
        self.Reynolds = Reynolds #100 # was 200! Turbulence
        # self.Rayleigh = Rayleigh
        # self.Prandtl = Prandtl

        self.it = 0
        self.best_pde_loss = 9999
    
    def sample_col(self):
      for col_pts in self.collocation_pts:
        col_pts.sample(cache=True)

    def sample_bcs(self):
      for bc in self.BCs:
        bc.sample(cache=True)

    def sample_ics(self):
      for ic in self.ICs:
        ic.sample(cache=True)
    
    def loss_physics(self):
      preds = []
      truths = []
      for col in self.collocation_pts:
        col_pred = col.predict(cache=True)
        preds.append(col_pred)
        truths.append(col.truth)
        
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

    def loss_ic(self):
      preds = []
      truths = []
      for ic in self.ICs:
        ic_pred = ic.predict(cache=True)
        preds.append(ic_pred)
        truths.append(ic.truth)
        
      pred = torch.vstack(preds)
      truth = torch.vstack(truths)
      return self.loss_fn(pred,truth)
    
    # def sample_collocation(self, adaptive_sample=False):
    #     # torch.manual_seed(2)
    #     if adaptive_sample:
    #       initial_ct = self.collocation_ct*10
    #       pts_rand = torch.rand((initial_ct, self.dim), device=device, requires_grad=True)
    #       pts = torch.empty_like(pts_rand)
    #       pts[:,0] = pts_rand[:,0] * self.t_range + self.t_min
    #       pts[:,1] = pts_rand[:,1] * self.x_range + self.x_min
    #       pts[:,2] = pts_rand[:,2] * self.y_range + self.y_min
    #       results = self.governing_eq(pts)
    #       r2 = results["r2"]
    #       e_r2 = torch.mean(r2)
    #       p_X = (r2 / e_r2) / torch.sum(r2 / e_r2)
    #       pts_subsample_ix = torch.multinomial(p_X.flatten(), self.collocation_ct, replacement=True)
    #       pts_subsample = pts[pts_subsample_ix]
          
    #       return pts_subsample
    #     else:
    #       pts_rand = torch.rand((self.collocation_ct, self.dim), device=device, requires_grad=True)
    #       pts = torch.empty_like(pts_rand)
    #       pts[:,0] = pts_rand[:,0] * self.t_range + self.t_min
    #       pts[:,1] = pts_rand[:,1] * self.x_range + self.x_min
    #       pts[:,2] = pts_rand[:,2] * self.y_range + self.y_min
    #       return pts

    def heat_flux(self, pts, T):
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
        d = 1/self.Peclet
        q_x = -d*T_x
        q_y = -d*T_y
        # d = 1 / np.sqrt(self.Rayleigh * self.Prandtl)
        # q_x = -d*T_x
        # q_y = -d*T_y
        # d = 1 / np.sqrt(self.Rayleigh)
        # q_x = -d*T_x
        # q_y = -d*T_y
        return {"T": T, "T_t": T_t, "T_x": T_x, "T_y": T_y, "q_x": q_x, "q_y": q_y}

    def diffusion(self, pts, T_x, T_y):
        T_xx = torch.autograd.grad(
            inputs=pts,
            outputs=T_x,
            grad_outputs=torch.ones_like(T_x).to(device),
            create_graph=True,
            retain_graph=True
        )[0][:,1:2]

        T_yy = torch.autograd.grad(
            inputs=pts,
            outputs=T_y,
            grad_outputs=torch.ones_like(T_y).to(device),
            create_graph=True,
            retain_graph=True
        )[0][:,2:3]

        diffusion = (T_xx + T_yy) * 1 / self.Peclet
        # diffusion = (T_xx + T_yy) * 1 / np.sqrt(self.Rayleigh * self.Prandtl)
        # diffusion = (T_xx + T_yy) 
        # diffusion = (T_xx + T_yy) * 1 / np.sqrt(self.Rayleigh)
        return {"diffusion": diffusion, "T_xx": T_xx, "T_yy": T_yy}

    def advection(self, pts, T, T_x, T_y, u, v):
        # uT = u*T
        # vT = v*T

        # uT_x= torch.autograd.grad(
        #     inputs=pts,
        #     outputs=uT,
        #     grad_outputs=torch.ones_like(uT).to(device),
        #     create_graph=True,
        #     retain_graph=True
        # )[0][:,1:2]

        # vT_y= torch.autograd.grad(
        #     inputs=pts,
        #     outputs=vT,
        #     grad_outputs=torch.ones_like(vT).to(device),
        #     create_graph=True,
        #     retain_graph=True
        # )[0][:,2:3]


        # advection = uT_x + vT_y
        advection = u*T_x + v*T_y
        return {"advection": advection}
    
    def pressure(self, pts, p):
        p_grad = torch.autograd.grad(
            inputs=pts,
            outputs=p,
            grad_outputs=torch.ones_like(p).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        p_x = p_grad[:,1:2]
        p_y = p_grad[:,2:3]

        return {"p_x": p_x, "p_y": p_y}
    
    def momentum(self, pts, u, v):
        u_grad = torch.autograd.grad(
            inputs=pts,
            outputs=u,
            grad_outputs=torch.ones_like(u).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        v_grad = torch.autograd.grad(
            inputs=pts,
            outputs=v,
            grad_outputs=torch.ones_like(v).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        u_x = u_grad[:,1:2]
        u_y = u_grad[:,2:3]
        v_x = v_grad[:,1:2]
        v_y = v_grad[:,2:3]


        u_t = u_grad[:,0:1]
        v_t = v_grad[:,0:1]

        uu_x = u*u_x
        vv_y = v*v_y

        """Second Velocity Derivatives"""
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

        v_x_grad = torch.autograd.grad(
            inputs=pts,
            outputs=v_x,
            grad_outputs=torch.ones_like(v_x).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        v_y_grad = torch.autograd.grad(
            inputs=pts,
            outputs=v_y,
            grad_outputs=torch.ones_like(v_y).to(device),
            create_graph=True,
            retain_graph=True
        )[0]
        v_xx = v_x_grad[:,1:2]
        v_yy = v_y_grad[:,2:3]

        return {
          "u_t": u_t, 
          "u_x": u_x, 
          "u_y": u_y, 
          "v_t": v_t, 
          "v_x": v_x, 
          "v_y": v_y, 
          "uu_x": uu_x, 
          "vv_y": vv_y, 
          "u_xx": u_xx, 
          "u_yy": u_yy, 
          "v_xx": v_xx, 
          "v_yy": v_yy
        }

      
    def governing_eq(self, pts):
        """Predict"""
        R = self.net(pts)
        T = R[:,0:1]
        u = R[:,1:2]
        v = R[:,2:3]
        p = R[:,3:4]

        """Heat Derivatives"""
        heat_fluxes = self.heat_flux(pts, T)
        T_t = heat_fluxes["T_t"]
        T_x = heat_fluxes["T_x"]
        T_y = heat_fluxes["T_y"]

        """Diffusion"""
        diffusion = self.diffusion(pts, T_x, T_y)["diffusion"]

        """Advection"""
        advection = self.advection(pts, T, T_x, T_y, u, v)["advection"]

        """Momentum Derivatives"""
        momentum = self.momentum(pts, u, v)
        u_t = momentum["u_t"]
        u_x = momentum["u_x"]
        u_y = momentum["u_y"]
        uu_x = momentum["uu_x"]
        u_xx = momentum["u_xx"]
        u_yy = momentum["u_yy"]
        v_t = momentum["v_t"]
        v_x = momentum["v_x"]
        v_y = momentum["v_y"]
        vv_y = momentum["vv_y"]
        v_xx = momentum["v_xx"]
        v_yy = momentum["v_yy"]

        """Pressure Derivatives"""
        dp = self.pressure(pts, p)
        p_x = dp["p_x"]
        p_y = dp["p_y"]

        """Residuals"""
        residual_ns_x = u_t + uu_x + v*u_y + p_x - 1/self.Reynolds * (u_xx+u_yy) 
        residual_ns_y = v_t + u*v_x + vv_y + p_y - 1/self.Reynolds * (v_xx+v_yy) - self.Richardson * T

        # residual_ns_x = u_t + uu_x + v*u_y + p_x - np.sqrt(self.Prandtl/self.Rayleigh)* (u_xx+u_yy) 
        # residual_ns_y = v_t + u*v_x + vv_y + p_y - np.sqrt(self.Prandtl/self.Rayleigh)* (v_xx+v_yy) - T

        # residual_ns_x = 1/self.Prandtl * (u_t + uu_x + v*u_y) + p_x - (u_xx+u_yy) 
        # residual_ns_y = 1/self.Prandtl * (v_t + u*v_x + vv_y) + p_y - (v_xx+v_yy) - self.Rayleigh * T
        
        # residual_ns_x = u_t + uu_x + v*u_y + p_x - self.Prandtl/np.sqrt(self.Rayleigh)* (u_xx+u_yy) 
        # residual_ns_y = v_t + u*v_x + vv_y + p_y - self.Prandtl/np.sqrt(self.Rayleigh)* (v_xx+v_yy) - self.Prandtl*T

        r2_ns_x = residual_ns_x**2
        r2_ns_y = residual_ns_y**2

        residual_continuity = u_x + v_y # Continuity
        r2_continuity = residual_continuity**2

        residual_diffusion = T_t - diffusion + advection  # Heat Equation
        # residual_diffusion = T_t - diffusion + advection - 1  # Heat Equation
        r2_heat = residual_diffusion**2 
        
        r2 = r2_ns_x + r2_ns_y + r2_continuity + r2_heat

        return {"T": T, "u": u, "v": v, "p":p, "r2": r2}

    def train_step(self):

        """Sample pts"""
        # pts_col = self.sample_collocation(adaptive_sample=self.adaptive_resample)
        self.sample_col()
        self.sample_bcs()
        self.sample_ics()

        """Reset grads"""
        self.adam.zero_grad()

        """Boundaries"""
        loss_bc = self.loss_bc()
        loss_ic = self.loss_ic()
        loss_physics = self.loss_physics()


        """Collocation pts"""
        # summary = self.governing_eq(pts_col)
        # r2 = summary["r2"]
        # loss_physics = torch.mean(r2)
        # loss_physics_res_penalty = torch.max(r2)

        """Loss"""

        if self.soft_adapt_weights:
          with torch.no_grad():
            losses = torch.hstack([loss_ic, loss_bc, loss_physics])
            top = torch.exp(self.lambda_Temp * (self.losses_prev - losses))
            lambdas = top / torch.sum(top)
            self.losses_prev = losses
          loss = torch.sum(lambdas * torch.hstack([loss_ic, loss_bc, loss_physics ]))
          loss.backward()
        else:
          loss = self.ic_weight*loss_ic + self.bc_weight*loss_bc + self.phys_weight*loss_physics 
          loss.backward()
        if self.it % 100 == 0:
          weighted_loss = self.ic_weight*loss_ic + self.bc_weight*loss_bc + self.phys_weight*loss_physics 
          print("\nIC / BC / PDE", loss_ic.item(), loss_bc.item(), loss_physics.item())
          if self.best_pde_loss > loss_physics.item() and self.it > 0:
            # print("Saving state dict...")
            torch.save(self.net.state_dict(), model_path)
            # self.best_pde_loss = loss_physics.item()
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
        ic.add_truth_to_fig(fig, ct=ct, flatten=True) if mode == 'Truth' else ic.add_error_to_fig(fig, ct=ct)
      for bc in self.BCs:
        bc.add_truth_to_fig(fig, ct=ct, flatten=True) if mode == 'Truth' else bc.add_error_to_fig(fig, ct=ct)
      # for col in self.collocation_pts:
      #   col.add_truth_to_fig(fig, ct=ct) if mode == 'Truth' else col.add_error_to_fig(fig, ct=ct)

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
  
  def add_truth_to_fig(self,fig,ct,z=False, flatten=False):
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
        colorscale="plasma",
        colorbar=dict(thickness=20, x=int(np.random.rand()*4) )
      )
    )
    fig.add_trace(scat)
  
class BCCicle(BC):
  def __init__(self, truth_fn, pred_fn, ct=1000, axis_bounds=[[0, 4 * np.pi], [-4 * np.pi, 4 * np.pi], [-4 * np.pi, -4 * np.pi]], requires_grad=False, r=0.2, c=[0.75,0.75], mode='mask') -> None:
    super().__init__(truth_fn, pred_fn, ct, axis_bounds, requires_grad)
    self.r = r
    self.c = torch.Tensor(c).to(device)
    self.mode = mode
  
  def sample(self, cache=False, predict=False, ct=None):
    pts = None
    if self.mode == 'edge':
      thetas = torch.rand((self.ct if ct == None else ct, 1), device=device, requires_grad=self.requires_grad)*2*np.pi
      x = self.r*torch.cos(thetas) + self.c[0]
      y = self.r*torch.sin(thetas) + self.c[1]
      t = torch.rand((self.ct if ct == None else ct, 1), device=device, requires_grad=self.requires_grad)*self.axis_ranges[0] + self.axis_mins[0]
      pts = torch.hstack([t,x,y])
    elif self.mode == 'mask':
      pts_tmp = torch.rand((self.ct if ct == None else ct, self.parent.dim), device=device, requires_grad=self.requires_grad)*self.axis_ranges+ self.axis_mins
      mask_single = (pts_tmp[:,1:2]-self.c[0])**2 + (pts_tmp[:,2:3]-self.c[1])**2 > self.r**2
      mask = mask_single.flatten()
      pts = pts_tmp[mask]

    truth = self.truth_fn(self.parent, pts)
    if cache:
      self.pts = pts
      self.truth = truth
    pred = None
    if predict:
      pred = self.predict(pts, cache=cache)

    return {"pts": pts, "truth": truth, "pred": pred}
  

def get_d(pts):
  return torch.sqrt(torch.sum(pts[:,1:]**2, axis=1)).reshape(-1,1)

t_bounds = [0, 100]
s_bounds = [[-3, 9], [-2,2]]
if __name__ == "__main__":

  model_path = Path(os.path.abspath(os.path.dirname(__file__))) / "models" / "a100-fluid-method.pth"
  ADAPTIVE_SAMPLING = False
  ADAPTIVE_WEIGHTING = True

  # RICHARDSON=1.0
  # PECLET=20
  # REYNOLDS=1000
  RICHARDSON=0
  PECLET=50
  REYNOLDS=300

  PRANDTL = 4.3 
  RAYLEIGH = 1000000

  x_bounds = s_bounds[0]
  y_bounds = s_bounds[1]
  x_min = x_bounds[0]
  x_max = x_bounds[1]
  y_min = y_bounds[0]
  y_max = y_bounds[1]
  x_range = x_max - x_min
  y_range = y_max - y_min


  def no_u_no_qx_edge(pinn: PINN_2D, pts):
    q_x = pts[:,1:2]*0
    u = pts[:,1:2]*0
    return torch.vstack([ q_x, u ])
  
  def no_v_yes_qy_edge(pinn: PINN_2D, pts):
    q_y = pts[:,1:2]*0 + 0.1
    v = pts[:,1:2]*0
    return torch.vstack([ q_y, v ])

  def no_v_hot_edge(pinn: PINN_2D, pts):
    T = pts[:,1:2]*0 + 0.5
    u = pts[:,1:2]*0
    v = pts[:,1:2]*0
    return torch.vstack([ T, v ])
  
  def no_slip_no_y_flux(pinn: PINN_2D, pts):
    q = pts[:,1:2]*0
    u = pts[:,1:2]*0
    v = pts[:,1:2]*0
    return torch.vstack([q,u,v])

  def no_pen_no_y_flux(pinn: PINN_2D, pts):
    q = pts[:,1:2]*0
    v = pts[:,1:2]*0
    return torch.vstack([q,v])

  def no_v_cold_edge(pinn: PINN_2D, pts):
    T = pts[:,1:2]*0 - 0.5
    u = pts[:,1:2]*0
    v = pts[:,1:2]*0
    return torch.vstack([ T, v ])

  def hot_edge_true(pinn: PINN_2D, pts):
    T = pts[:,1:2]*0 + 1 
    u = pts[:,2:3]*0
    v = pts[:,2:3]*0
    return torch.vstack([ T, u, v ])

  def hot_ramped_edge_true(pinn: PINN_2D, pts):
    T =  torch.sigmoid(pts[:,0:1]-6)
    u = pts[:,2:3]*0
    v = pts[:,2:3]*0
    return torch.vstack([ T, u, v ])

  def rightward_flow_inlet_true(pinn: PINN_2D, pts):
    T = pts[:,1:2]*0 # constant temp
    u = pts[:,1:2]*0 +1
    v = pts[:,1:2]*0 # no slip
    return torch.vstack([ T, u, v ])

  def rightward_ramped_flow_inlet_true(pinn: PINN_2D, pts):
    T = pts[:,1:2]*0 # constant temp
    u = torch.sigmoid(pts[:,0:1]-6) # rightward flow
    v = pts[:,1:2]*0 # no slip
    return torch.vstack([ T, u, v ])

  def zero_qxp_outlet_true(pinn: PINN_2D, pts):
    q_x = pts[:,1:2]*0 # no heat flux
    p = pts[:,1:2]*0 # zero pressure
    return torch.vstack([ q_x, p ])

  def zero_Tuv_true(pinn: PINN_2D, pts):
    T = pts[:,1:2]*0
    u = pts[:,2:3]*0
    v = pts[:,2:3]*0
    return torch.vstack([ T, u, v ])

  def top_boundary_true(pinn: PINN_2D, pts):
    T = pts[:,2:3]*0
    u = pts[:,2:3]*0 # no slip
    v = pts[:,2:3]*0 # no penetration
    # T = -torch.sign(pts[:,2:3])*0.5
    # q_y = pts[:,2:3]*0 + 1 # cooled/heated plate
    # return torch.vstack([q_y, u, v])
    return torch.vstack([T, u, v])

  def ic_true(pinn: PINN_2D, pts):
    # T = pts[:,1:2]*0#torch.sqrt(pts[:,1:2]**2 + pts[:,2:3]**2)
    # T = 2*((pts[:,2:3]-pinn.y_min)/pinn.y_range * 2 -1)#torch.sqrt(pts[:,1:2]**2 + pts[:,2:3]**2)
    # T = -0.5*((pts[:,2:3]-pinn.y_min)/pinn.y_range * 2 -1)
    # T = -1*((pts[:,2:3]-pinn.y_min)/pinn.y_range * 2 -1)

    # T = (pts[:,2:3]-y_min)/y_range * (-2) + 1
    T = pts[:,1:2]*0
    u = pts[:,1:2]*0 
    v = pts[:,1:2]*0
    # T = torch.sin(pts[:,1:2]*10)*torch.sin(3.4*pts[:,2:3])*0.1
    # u = torch.sin(pts[:,1:2]*13)*torch.sin(11.2*pts[:,2:3])*0.1
    # v = torch.sin(pts[:,1:2]*7.4)*torch.sin(1.2*pts[:,2:3])*0.1
    return torch.vstack([T, u, v])

  def true_collocation(pinn: PINN_2D, pts):
    return pts[:,1:2]*0 # always 0
  
  def Tuv_prediction(pinn: PINN_2D, pts):
    R = pinn.net(pts)
    T = R[:,0:1]
    u = R[:,1:2] 
    v = R[:,2:3]
    return torch.vstack([T, u, v])

  def Tv_prediction(pinn: PINN_2D, pts):
    R = pinn.net(pts)
    T = R[:,0:1]
    u = R[:,1:2] 
    v = R[:,2:3]
    return torch.vstack([T, v])
  
  def qxuv_prediction(pinn: PINN_2D, pts):
    R = pinn.net(pts)
    T = R[:,0:1]
    u = R[:,1:2]
    v = R[:,2:3]
    fluxes = pinn.heat_flux(pts, T)
    q_x = fluxes["q_x"]
    return torch.vstack([ q_x, u, v ])

  def qxp_prediction(pinn: PINN_2D, pts):
    R = pinn.net(pts)
    T = R[:,0:1]
    p = R[:,3:4]
    fluxes = pinn.heat_flux(pts, T)
    q_x = fluxes["q_x"]
    return torch.vstack([ q_x, p ])

  def qyuv_prediction(pinn: PINN_2D, pts):
    R = pinn.net(pts)
    T = R[:,0:1]
    u = R[:,1:2]
    v = R[:,2:3]
    fluxes = pinn.heat_flux(pts, T)
    q_y = fluxes["q_y"]
    return torch.vstack([q_y, u, v])

  def qyv_prediction(pinn: PINN_2D, pts):
    R = pinn.net(pts)
    T = R[:,0:1]
    u = R[:,1:2]
    v = R[:,2:3]
    fluxes = pinn.heat_flux(pts, T)
    q_y = fluxes["q_y"]
    return torch.vstack([q_y, v])

  def qxu_prediction(pinn: PINN_2D, pts):
    R = pinn.net(pts)
    T = R[:,0:1]
    u = R[:,1:2]
    v = R[:,2:3]
    fluxes = pinn.heat_flux(pts, T)
    q_x = fluxes["q_x"]
    return torch.vstack([q_x, u])

  def predict_collocation(pinn: PINN_2D, pts):
    summary = pinn.governing_eq(pts)
    r2 = summary["r2"]
    return r2

  def make_vk_cyl():
    bc_u = BC(
      ct=1000,
      # truth_fn=no_slip_no_y_flux, 
      # pred_fn=qyuv_prediction,
      truth_fn=no_pen_no_y_flux, 
      pred_fn=qyv_prediction,
      axis_bounds=[t_bounds, x_bounds, [y_max, y_max]], 
      requires_grad=True,
    )
    bc_d = BC(
      ct=1000,
      # truth_fn=no_slip_no_y_flux, 
      # pred_fn=qyuv_prediction,
      truth_fn=no_pen_no_y_flux, 
      pred_fn=qyv_prediction,
      axis_bounds=[t_bounds, x_bounds, [y_min, y_min]], 
      requires_grad=True,
    )
    bc_inlet = BC(
      ct=1000,
      truth_fn=rightward_ramped_flow_inlet_true,
      pred_fn=Tuv_prediction,
      axis_bounds=[t_bounds, [x_min, x_min],y_bounds], 
      requires_grad=True,
    )
    bc_outlet = BC(
      ct=1000,
      truth_fn=zero_qxp_outlet_true,
      pred_fn=qxp_prediction,
      # truth_fn=zero_qxp_outlet_true,
      # pred_fn=qxp_prediction,
      axis_bounds=[t_bounds, [x_max, x_max],y_bounds], 
      requires_grad=True,
    )
    r=0.5
    c=[0,0]
    bc_cyl = BCCicle(
      ct=1000,
      truth_fn=hot_ramped_edge_true,
      pred_fn=Tuv_prediction,
      axis_bounds=[t_bounds, x_bounds, y_bounds], 
      requires_grad=True,
      mode="edge",
      r=r,
      c=c
    )
    col = BCCicle(
      ct=8000,
      truth_fn=true_collocation,
      pred_fn=predict_collocation,
      axis_bounds=[t_bounds, x_bounds, y_bounds], 
      requires_grad=True,
      mode="mask",
      r=r,
      c=c
    )
    ic = BCCicle(
      ct=4000,
      truth_fn=zero_Tuv_true,
      pred_fn=Tuv_prediction,
      # truth_fn=zero_Tuv_true,
      # pred_fn=Tuv_prediction,
      axis_bounds=[[t_bounds[0], t_bounds[0]], x_bounds, y_bounds], 
      requires_grad=True,
      mode="mask",
      r=r,
      c=c
    )
    ics = [ic]
    bcs = [bc_inlet, bc_outlet, bc_u, bc_d, bc_cyl]
    col = [col]
    return ics, bcs, col

  def make_heated_box():
    block_l_edge = x_min + x_range/4
    block_r_edge = block_l_edge + x_range/4
    block_b_edge = y_min
    block_t_edge = y_min + y_range/3

    bc_u = BC(
      ct=100,
      truth_fn=zero_Tuv_true, 
      pred_fn=Tuv_prediction,
      axis_bounds=[t_bounds, x_bounds, [y_max, y_max]], 
      requires_grad=True,
    )
    bc_l = BC(
      ct=100,
      truth_fn=zero_Tuv_true,
      pred_fn=Tuv_prediction,
      axis_bounds=[t_bounds, [x_min, x_min], [y_min+(y_max-y_min)/3, y_max]], 
      requires_grad=True,
    )
    bc_inlet = BC(
      ct=100,
      truth_fn=rightward_flow_inlet_true,
      pred_fn=Tuv_prediction,
      axis_bounds=[t_bounds, [x_min, x_min],[y_min, y_min+(y_max-y_min)/3]], 
      requires_grad=True,
    )
    bc_r = BC(
      ct=100,
      truth_fn=zero_Tuv_true,
      pred_fn=Tuv_prediction,
      axis_bounds=[t_bounds, [x_max, x_max],[y_min, y_min+2*(y_max-y_min)/3]], 
      requires_grad=True,
    )

    bc_outlet = BC(
      ct=100,
      truth_fn=zero_qxp_outlet_true,
      pred_fn=qxp_prediction,
      axis_bounds=[t_bounds, [x_max, x_max],[y_min+2*(y_max-y_min)/3, y_max]], 
      requires_grad=True,
    )
    bc_d_l = BC(
      ct=100,
      truth_fn=zero_Tuv_true, 
      pred_fn=Tuv_prediction,
      axis_bounds=[t_bounds,[x_min, block_l_edge], [y_min, y_min]], 
      requires_grad=True,
    )
    bc_d_r = BC(
      ct=100,
      truth_fn=zero_Tuv_true, 
      pred_fn=Tuv_prediction,
      axis_bounds=[t_bounds, [block_r_edge, x_max], [y_min, y_min]], 
      requires_grad=True,
    )

    box_boundary_t = BC(
      ct=100,
      truth_fn=hot_edge_true,
      pred_fn=Tuv_prediction,
      axis_bounds=[t_bounds, [block_l_edge, block_r_edge], [block_t_edge, block_t_edge]], 
      requires_grad=True,
    )
    box_boundary_l= BC(
      ct=100,
      truth_fn=hot_edge_true,
      pred_fn=Tuv_prediction,
      axis_bounds=[t_bounds, [block_l_edge, block_l_edge], [block_b_edge, block_t_edge]], 
      requires_grad=True,
    )
    box_boundary_r= BC(
      ct=100,
      truth_fn=hot_edge_true,
      pred_fn=Tuv_prediction,
      axis_bounds=[t_bounds, [block_r_edge, block_r_edge], [block_b_edge, block_t_edge]], 
      requires_grad=True,
    )

    collocation_a = BC(
      ct=850,
      truth_fn=true_collocation,
      pred_fn=predict_collocation,
      axis_bounds=[t_bounds, [x_min, block_l_edge], y_bounds], 
      requires_grad=True,
    )
    collocation_b = BC(
      ct=850,
      truth_fn=true_collocation,
      pred_fn=predict_collocation,
      axis_bounds=[t_bounds, [block_l_edge, block_r_edge],[block_t_edge, y_max]], 
      requires_grad=True,
    )
    collocation_c = BC(
      ct=850,
      truth_fn=true_collocation,
      pred_fn=predict_collocation,
      axis_bounds=[t_bounds, [block_r_edge, x_max], y_bounds], 
      requires_grad=True,
    )

    ic_a = BC(
      ct=300,
      truth_fn=ic_true,
      pred_fn=Tuv_prediction,
      axis_bounds=[[t_bounds[0], t_bounds[0]], [x_min, block_l_edge], y_bounds], 
      requires_grad=True,
    )
    ic_b = BC(
      ct=300,
      truth_fn=ic_true,
      pred_fn=Tuv_prediction,
      axis_bounds=[[t_bounds[0], t_bounds[0]], [block_l_edge, block_r_edge],[block_t_edge, y_max]], 
      requires_grad=True,
    )
    ic_c = BC(
      ct=300,
      truth_fn=ic_true,
      pred_fn=Tuv_prediction,
      axis_bounds=[[t_bounds[0], t_bounds[0]], [block_r_edge, x_max], y_bounds], 
      requires_grad=True,
    )


    # DEFINITION for heated block!
    bcs = [bc_d_l, bc_d_r, bc_u, bc_l, bc_r, bc_inlet, bc_outlet, box_boundary_l, box_boundary_r, box_boundary_t]
    ics = [ic_a,ic_b,ic_c]
    collocation_pts = [collocation_a, collocation_b, collocation_c]
    return ics, bcs, collocation_pts


  def make_rbc():
    bc_d = BC(
      ct=500,
      truth_fn=no_v_hot_edge, 
      pred_fn=Tv_prediction,
      axis_bounds=[t_bounds, x_bounds, [y_min, y_min]], 
      requires_grad=True,
    )
    bc_u = BC(
      ct=500,
      truth_fn=no_v_cold_edge, 
      pred_fn=Tv_prediction,
      axis_bounds=[t_bounds, x_bounds, [y_max, y_max]], 
      requires_grad=True,
    )
    bc_l = BC(
      ct=500,
      truth_fn=no_u_no_qx_edge, 
      pred_fn=qxu_prediction,
      axis_bounds=[t_bounds,[x_min,x_min], y_bounds], 
      requires_grad=True,
    )
    bc_r = BC(
      ct=500,
      truth_fn=no_u_no_qx_edge, 
      pred_fn=qxu_prediction,
      axis_bounds=[t_bounds,[x_max, x_max], y_bounds], 
      requires_grad=True,
    )

    ic = BC(
      ct=2000,
      truth_fn=ic_true,
      pred_fn=Tuv_prediction,
      axis_bounds=[[t_bounds[0], t_bounds[0]], x_bounds, y_bounds], 
      requires_grad=True,
    )

    collocation = BC(
      ct=8000,
      truth_fn=true_collocation,
      pred_fn=predict_collocation,
      axis_bounds=[t_bounds, x_bounds, y_bounds], 
      requires_grad=True,
    )

    bcs = [bc_d,bc_u,bc_l,bc_r]
    ics = [ic]
    collocation_pts=[collocation]
    return ics, bcs, collocation_pts
  
  ics, bcs, collocation_pts = make_vk_cyl()

  pinn = PINN_2D(
      net=MLP(input_dim=3, output_dim=4, hidden_layer_ct=4,hidden_dim=256, act=F.tanh, learnable_act="SINGLE"), 
      bcs=bcs,
      ics=ics,
      collocation_pts=collocation_pts,
      t_bounds=t_bounds, 
      space_bounds=s_bounds,
      lr=1e-3,
      adaptive_resample=ADAPTIVE_SAMPLING,
      soft_adapt_weights=ADAPTIVE_WEIGHTING,
      Rayleigh=RAYLEIGH,
      Prandtl=PRANDTL,
      Richardson=RICHARDSON,
      Peclet=PECLET,
      Reynolds=REYNOLDS
  )
  # pinn.plot_bcs_and_ics(ct=200)

  fig = plt.figure()
  for bc in pinn.BCs:
    pts = bc.sample(ct=1000)["pts"].detach().cpu()
    plt.scatter(pts[:,1:2],pts[:,2:3],s=0.5)
  for ic in pinn.ICs:
    pts = ic.sample(ct=1000)["pts"].detach().cpu()
    plt.scatter(pts[:,1:2],pts[:,2:3],s=0.5)
  for col in pinn.collocation_pts:
    pts = col.sample(ct=5000)["pts"].detach().cpu()
    plt.scatter(pts[:,1:2],pts[:,2:3],s=1)
  plt.show()

  # pinn.plot_bcs_and_ics("Truth", 2000)

  for i in range(10):
    print(f"MetaEpoch: {i}")
    if i == 0:
        pinn.adam.param_groups[0]["lr"] = 1e-3
    if i == 1:
        pinn.adam.param_groups[0]["lr"] = 1e-4
    if i == 2:
        pinn.adam.param_groups[0]["lr"] = 5e-5
    if i == 3:
        pinn.adam.param_groups[0]["lr"] = 1e-5
    if i == 4:
        pinn.adam.param_groups[0]["lr"] = 5e-6
    pinn.plot_bcs_and_ics(mode='error', ct=5000)
    fig = go.Figure()
    pinn.collocation_pts[0].add_error_to_fig(fig, 5000)
    fig.show()
    pinn.train(n_epochs=1000, reporting_frequency=100, phys_weight=1, res_weight=0.01, bc_weight=8, ic_weight=4)
    # pinn.plot_bcs_and_ics(mode='error', ct=5000)
