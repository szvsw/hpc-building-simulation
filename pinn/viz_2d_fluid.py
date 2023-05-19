import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import taichi as ti

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mlp import MLP

device = 'cuda' if torch.cuda.is_available() else 'device'
print(f"Using {device}")

X_RES, Y_RES = 300,100

@ti.kernel
def init_pts(pts: ti.template(), x_min: float, x_range: float, y_min: float, y_range: float):
    for x,y in pts:
        x_norm = x / (X_RES-1)
        y_norm = y / (Y_RES-1)
        x_loc = x_norm * x_range + x_min
        y_loc = y_norm * y_range + y_min
        pts[x,y] = ti.Vector((x_loc,y_loc))


def init_colormap(colormap_field, colormap_arr):
        for i, color in enumerate(colormap_arr):
            colormap_field[i] = ti.Vector(color)

@ti.kernel
def update_colors(
    T: ti.template(), 
    u:ti.template(), 
    v:ti.template(), 
    V: ti.template(), 
    colors: ti.template(), 
    colormap_field: ti.template(), 
    T_min: float, 
    T_range: float,
    u_min: float, 
    u_range: float,
    v_min: float, 
    v_range: float,
    V_min: float,
    V_range: float,
):
    for i,j in u:
        # Temperature (LOWER LEFT)
        h = ti.cast(T[i,j] - T_min, ti.float32)/T_range
        
        level = ti.max(ti.min(ti.floor(h*(colormap_field.shape[0]-1)),colormap_field.shape[0]-2), 0)
        colorphase = ti.cast(ti.min(ti.max(h*(colormap_field.shape[0]-1) - level, 0),1), ti.f32)
        level_idx = ti.cast(level, dtype=int)

        colors[i,j].x = ti.cast((colormap_field[level_idx].x * (1-colorphase) + colorphase*colormap_field[level_idx+1].x)/255, ti.float32)
        colors[i,j].y = ti.cast((colormap_field[level_idx].y * (1-colorphase) + colorphase*colormap_field[level_idx+1].y)/255, ti.float32)
        colors[i,j].z = ti.cast((colormap_field[level_idx].z * (1-colorphase) + colorphase*colormap_field[level_idx+1].z)/255, ti.float32)

        # Vorticity (UPPER LEFT)
        h = ti.cast(V[i,j] - V_min, ti.float32)/V_range
        
        level = ti.max(ti.min(ti.floor(h*(colormap_field.shape[0]-1)),colormap_field.shape[0]-2), 0)
        colorphase = ti.cast(ti.min(ti.max(h*(colormap_field.shape[0]-1) - level, 0),1), ti.f32)
        level_idx = ti.cast(level, dtype=int)

        colors[X_RES+i,j].x = ti.cast((colormap_field[level_idx].x * (1-colorphase) + colorphase*colormap_field[level_idx+1].x)/255, ti.float32)
        colors[X_RES+i,j].y = ti.cast((colormap_field[level_idx].y * (1-colorphase) + colorphase*colormap_field[level_idx+1].y)/255, ti.float32)
        colors[X_RES+i,j].z = ti.cast((colormap_field[level_idx].z * (1-colorphase) + colorphase*colormap_field[level_idx+1].z)/255, ti.float32)

        # u-velocity
        h = ti.cast(u[i,j] - u_min, ti.float32)/u_range
        
        level = ti.max(ti.min(ti.floor(h*(colormap_field.shape[0]-1)),colormap_field.shape[0]-2), 0)
        colorphase = ti.cast(ti.min(ti.max(h*(colormap_field.shape[0]-1) - level, 0),1), ti.f32)
        level_idx = ti.cast(level, dtype=int)

        colors[i,Y_RES+j].x = ti.cast((colormap_field[level_idx].x * (1-colorphase) + colorphase*colormap_field[level_idx+1].x)/255, ti.float32)
        colors[i,Y_RES+j].y = ti.cast((colormap_field[level_idx].y * (1-colorphase) + colorphase*colormap_field[level_idx+1].y)/255, ti.float32)
        colors[i,Y_RES+j].z = ti.cast((colormap_field[level_idx].z * (1-colorphase) + colorphase*colormap_field[level_idx+1].z)/255, ti.float32)

        # v-velocity
        h = ti.cast(v[i,j] - v_min, ti.float32)/v_range
        
        level = ti.max(ti.min(ti.floor(h*(colormap_field.shape[0]-1)),colormap_field.shape[0]-2), 0)
        colorphase = ti.cast(ti.min(ti.max(h*(colormap_field.shape[0]-1) - level, 0),1), ti.f32)
        level_idx = ti.cast(level, dtype=int)

        colors[X_RES+i,Y_RES+j].x = ti.cast((colormap_field[level_idx].x * (1-colorphase) + colorphase*colormap_field[level_idx+1].x)/255, ti.float32)
        colors[X_RES+i,Y_RES+j].y = ti.cast((colormap_field[level_idx].y * (1-colorphase) + colorphase*colormap_field[level_idx+1].y)/255, ti.float32)
        colors[X_RES+i,Y_RES+j].z = ti.cast((colormap_field[level_idx].z * (1-colorphase) + colorphase*colormap_field[level_idx+1].z)/255, ti.float32)
        
        # Block out
        if i > 0.25 * X_RES and i < 0.5 * X_RES and j < 0.33 * Y_RES:
            colors[i,j].x = 0.0
            colors[i,j].y = 0.0
            colors[i,j].z = 0.0
            colors[X_RES+i,j].x = 0.0
            colors[X_RES+i,j].y = 0.0
            colors[X_RES+i,j].z = 0.0
            colors[i,Y_RES+j].x = 0.0
            colors[i,Y_RES+j].y = 0.0
            colors[i,Y_RES+j].z = 0.0
            colors[X_RES+i,Y_RES+j].x = 0.0
            colors[X_RES+i,Y_RES+j].y = 0.0
            colors[X_RES+i,Y_RES+j].z = 0.0


colormap = [
    [64,57,144],
    [112,198,162],
    [230, 241, 146],
    [253,219,127],
    [244,109,69],
    [169,23,69]
]
jet = plt.colormaps["jet"]
colormap = jet(np.arange(jet.N))*255

if __name__ == "__main__":
    from hybrid_thermofluid import s_bounds,t_bounds

    model_path =  "2d-simple-neumann2d-5-heated-with-wind-adaptive.pth"
    model_path =  "new-bc-method.pth"
    # model_path =  "heated_blocks_in_enclosure_4x256.pth"
    OUTPUT_DIM=4
    USE_MESH = False
    dt = 0.01

    T_min, T_range = -1,2 # Tmp LL
    u_min, u_range = 0.5,1.5 # Velocity Mag UL
    v_min, v_range = 0,2 # Qmag UR
    V_min, V_range = -2,4 # Vort LR

    T_min, T_range = -1,2 # Tmp LL
    u_min, u_range = 0,1 # Velocity Mag UL
    v_min, v_range = 0,2 # Qmag UR
    V_min, V_range = -2,4 # Vort LR

    # T_min, T_range = 0,1.5 # Tmp LL
    # u_min, u_range = 0,1 # Velocity Mag UL
    # v_min, v_range = 0,2 # Qmag UR
    # V_min, V_range = -1,2 # Vort LR

    T_min, T_range = 0,1 # Tmp LL
    u_min, u_range = -0.5,1 # UL
    v_min, v_range = 0,1 # Qmag UR
    V_min, V_range = -0.8,1.6 # Vort LR

    mesh_height = 2.0

    """Computed"""
    x_bounds = s_bounds[0]
    y_bounds = s_bounds[1]
    x_min = x_bounds[0]
    x_max = x_bounds[1]
    x_range = x_max - x_min
    y_min = y_bounds[0]
    y_max = y_bounds[1]
    y_range = y_max - y_min

    """Net"""
    net = MLP(input_dim=3, output_dim=OUTPUT_DIM, hidden_layer_ct=4,hidden_dim=256, act=F.tanh, learnable_act="SINGLE")
    net_path = Path(os.path.abspath(os.path.dirname(__file__))) / "models" / model_path
    net.load_state_dict(torch.load(net_path))
    net = net.to(device)

    """Fields"""
    ti.init(arch=ti.gpu)
    shape = (X_RES, Y_RES)
    im_shape = (2*X_RES, 2*Y_RES)
    pts_chi = ti.Vector.field(2, ti.f32, shape=shape)
    T_chi = ti.field(ti.f32, shape=shape)
    v_chi = ti.field(ti.f32, shape=shape)
    u_chi = ti.field(ti.f32, shape=shape)
    V_chi = ti.field(ti.f32, shape=shape)
    q_chi = ti.field(ti.f32, shape=shape)
    vel_chi = ti.field(ti.f32, shape=shape)
    colors = ti.Vector.field(3,dtype=ti.float32, shape=im_shape) # image            
    colormap_field = ti.Vector.field(3, dtype=ti.f32, shape=len(colormap))

    init_pts(pts=pts_chi, x_min=x_min, x_range=x_range, y_min=y_min, y_range=y_range)
    init_colormap(colormap_field, colormap)

    """Torch"""
    pts = pts_chi.to_torch().to(device).requires_grad_(True)
    pts = pts.reshape(-1,2)
    t = torch.zeros((pts.shape[0],1), device=device)
    pts = torch.hstack([t,pts])
    t_steps = torch.arange(t_bounds[0], t_bounds[1], dt)


    """Taichi Window"""
    w_scale=4
    window = ti.ui.Window("2D Diffusion", (X_RES*w_scale, Y_RES*w_scale))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 8, 0)
    camera.lookat(0,0,0)
    camera.up(0,1,0)
    camera_radius = 6
    camera_height = 6
    camera_speed = 0.001
    camera_t = 0


    it = 0
    k = 0
    while window.running:
        pts[:,0] = t_steps[it]
        R = net(pts)
        T = R[:,0:1] 
        u = R[:,1:2]
        v = R[:,2:3]
        p = R[:,3:4]
        
        T_grad = torch.autograd.grad(
            inputs=pts,
            outputs=T,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
            retain_graph=True
        )[0]
        T_x = T_grad[:,1:2]
        T_y = T_grad[:,2:3]
        q_mag = torch.sqrt(T_x**2+T_y**2)

        vel_mag = torch.sqrt(u**2 + v**2)
        u_y = torch.autograd.grad(
            inputs=pts,
            outputs=u,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0][:,2:3]
        v_x = torch.autograd.grad(
            inputs=pts,
            outputs=v,
            grad_outputs=torch.ones_like(v),
            create_graph=False,
            retain_graph=False
        )[0][:,1:2]
        vorticity = v_x-u_y
        T_chi.from_torch(T.reshape(shape))
        u_chi.from_torch(u.reshape(shape))
        v_chi.from_torch(v.reshape(shape))
        V_chi.from_torch(vorticity.reshape(shape))
        q_chi.from_torch(q_mag.reshape(shape))
        vel_chi.from_torch(vel_mag.reshape(shape))
        vel_chi.from_torch(-T_x.reshape(shape))
        # vel_chi.from_torch(p.reshape(shape))

        if it % 100 == 0:
            print(f"---{it}---")
            print(f"T:{torch.min(T).item()},{torch.max(T).item()}")
            print(f"u:{torch.min(u).item()},{torch.max(u).item()}")
            print(f"v:{torch.min(v).item()},{torch.max(v).item()}")
            print(f"V:{torch.min(vorticity).item()},{torch.max(vorticity).item()}")
            print(f"vel:{torch.min(vel_mag).item()},{torch.max(vel_mag).item()}")
            print(f"q:{torch.min(q_mag).item()},{torch.max(q_mag).item()}")
            print(f"T_x:{torch.min(T_x).item()},{torch.max(T_x).item()}")
            print(f"T_y:{torch.min(T_y).item()},{torch.max(T_y).item()}")
            print(f"p:{torch.min(p).item()},{torch.max(p).item()}")

        update_colors(
            T=T_chi, # LL
            V=V_chi, # LR
            u=vel_chi, # UL
            v=q_chi, # UR
            colors=colors,
            colormap_field=colormap_field,
            T_min=T_min,
            T_range=T_range,
            u_min=u_min,
            u_range=u_range,
            v_min=v_min,
            v_range=v_range,
            V_min=V_min,
            V_range=V_range,
        )
        canvas.set_image(colors)

        window.show()
        # if (it + 1) % t_steps.shape[0] == 0:
        #     exit()
        # k = k+1
        # window.save_image(f"pinn/videos/{it:05d}.png")
        it = (it + 1) % t_steps.shape[0]
