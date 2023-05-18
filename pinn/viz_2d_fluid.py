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

RES = 256

@ti.kernel
def init_pts(pts: ti.template(), s_min: float, s_range: float):
    for x,y in pts:
        x_norm = x / (RES-1)
        y_norm = y / (RES-1)
        x_loc = x_norm * s_range + s_min
        y_loc = y_norm * s_range + s_min
        pts[x,y] = ti.Vector((x_loc,y_loc))


def init_colormap(colormap_field, colormap_arr):
        for i, color in enumerate(colormap_arr):
            colormap_field[i] = ti.Vector(color)

@ti.kernel
def init_mesh_vertices(vertices: ti.template()):
    mesh_render_size = 4
    for i,j in ti.ndrange(RES, RES):
        vertices[i+RES*j].x = ti.cast(i/(RES-1)*mesh_render_size-mesh_render_size/2, ti.float32)
        vertices[i+RES*j].y = ti.cast(0.0, ti.float32)
        vertices[i+RES*j].z = ti.cast(j/(RES-1)*mesh_render_size-mesh_render_size/2, ti.float32)


@ti.kernel
def init_mesh_indices(indices: ti.template()):
    for i, j in ti.ndrange(RES - 1, RES - 1):
            quad_id = (i * (RES - 1)) + j
            # First triangle of the square
            indices[quad_id * 6 + 0] = i * RES + j
            indices[quad_id * 6 + 1] = (i + 1) * RES + j
            indices[quad_id * 6 + 2] = i * RES + (j + 1)
            # Second triangle of the square
            indices[quad_id * 6 + 3] = (i + 1) * RES + j + 1
            indices[quad_id * 6 + 4] = i * RES + (j + 1)
            indices[quad_id * 6 + 5] = (i + 1) * RES + j

@ti.kernel
def update_mesh_vertices(u: ti.template(), mesh_colors:ti.template(), vertices: ti.template(), colormap_field: ti.template(), u_min: float, u_range: float, mesh_height: float):
    for i,j in u:
        h = ti.cast(u[i,j] - u_min, ti.float32)/u_range
        vertices[i+RES*j].y = h*mesh_height
        
        level = ti.max(ti.min(ti.floor(h*(colormap_field.shape[0]-1)),colormap_field.shape[0]-2), 0)
        colorphase = ti.cast(ti.min(ti.max(h*(colormap_field.shape[0]-1) - level, 0),1), ti.f32)
        level_idx = ti.cast(level, dtype=int)

        mesh_colors[i+RES*j].x = ti.cast((colormap_field[level_idx].x * (1-colorphase) + colorphase*colormap_field[level_idx+1].x)/255, ti.float32)
        mesh_colors[i+RES*j].y = ti.cast((colormap_field[level_idx].y * (1-colorphase) + colorphase*colormap_field[level_idx+1].y)/255, ti.float32)
        mesh_colors[i+RES*j].z = ti.cast((colormap_field[level_idx].z * (1-colorphase) + colorphase*colormap_field[level_idx+1].z)/255, ti.float32)

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

        colors[RES+i,j].x = ti.cast((colormap_field[level_idx].x * (1-colorphase) + colorphase*colormap_field[level_idx+1].x)/255, ti.float32)
        colors[RES+i,j].y = ti.cast((colormap_field[level_idx].y * (1-colorphase) + colorphase*colormap_field[level_idx+1].y)/255, ti.float32)
        colors[RES+i,j].z = ti.cast((colormap_field[level_idx].z * (1-colorphase) + colorphase*colormap_field[level_idx+1].z)/255, ti.float32)

        # u-velocity
        h = ti.cast(u[i,j] - u_min, ti.float32)/u_range
        
        level = ti.max(ti.min(ti.floor(h*(colormap_field.shape[0]-1)),colormap_field.shape[0]-2), 0)
        colorphase = ti.cast(ti.min(ti.max(h*(colormap_field.shape[0]-1) - level, 0),1), ti.f32)
        level_idx = ti.cast(level, dtype=int)

        colors[i,RES+j].x = ti.cast((colormap_field[level_idx].x * (1-colorphase) + colorphase*colormap_field[level_idx+1].x)/255, ti.float32)
        colors[i,RES+j].y = ti.cast((colormap_field[level_idx].y * (1-colorphase) + colorphase*colormap_field[level_idx+1].y)/255, ti.float32)
        colors[i,RES+j].z = ti.cast((colormap_field[level_idx].z * (1-colorphase) + colorphase*colormap_field[level_idx+1].z)/255, ti.float32)

        # v-velocity
        h = ti.cast(v[i,j] - v_min, ti.float32)/v_range
        
        level = ti.max(ti.min(ti.floor(h*(colormap_field.shape[0]-1)),colormap_field.shape[0]-2), 0)
        colorphase = ti.cast(ti.min(ti.max(h*(colormap_field.shape[0]-1) - level, 0),1), ti.f32)
        level_idx = ti.cast(level, dtype=int)

        colors[RES+i,RES+j].x = ti.cast((colormap_field[level_idx].x * (1-colorphase) + colorphase*colormap_field[level_idx+1].x)/255, ti.float32)
        colors[RES+i,RES+j].y = ti.cast((colormap_field[level_idx].y * (1-colorphase) + colorphase*colormap_field[level_idx+1].y)/255, ti.float32)
        colors[RES+i,RES+j].z = ti.cast((colormap_field[level_idx].z * (1-colorphase) + colorphase*colormap_field[level_idx+1].z)/255, ti.float32)


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

    model_path =  "2d-simple-neumann2d-5-heated-with-wind-adaptive.pth"
    model_path =  "new-bc-method.pth"
    OUTPUT_DIM=4
    USE_MESH = False
    dt = 0.01
    t_bounds = [0, 8*np.pi]
    s_bounds = [-4*np.pi, 4*np.pi]
    T_min, T_range = -1,2
    u_min, u_range = -0.01, 0.02
    v_min, v_range = -1.5, 3#-0.5, 1
    V_min, V_range = -0.01, 0.02

    mesh_height = 2.0

    """Computed"""
    s_range = s_bounds[1] - s_bounds[0]
    s_min = s_bounds[0]

    """Net"""
    net = MLP(input_dim=3, output_dim=OUTPUT_DIM, hidden_layer_ct=10,hidden_dim=256, act=F.tanh, learnable_act="SINGLE")
    net_path = Path(os.path.abspath(os.path.dirname(__file__))) / "models" / model_path
    net.load_state_dict(torch.load(net_path))
    net = net.to(device)

    """Fields"""
    ti.init(arch=ti.gpu)
    pts_chi = ti.Vector.field(2, ti.f32, shape=(RES, RES))
    T_chi = ti.field(ti.f32, shape=(RES, RES))
    v_chi = ti.field(ti.f32, shape=(RES, RES))
    u_chi = ti.field(ti.f32, shape=(RES, RES))
    V_chi = ti.field(ti.f32, shape=(RES, RES))
    q_chi = ti.field(ti.f32, shape=(RES, RES))
    vel_chi = ti.field(ti.f32, shape=(RES, RES))
    vertices = ti.Vector.field(3,ti.f32,shape=(RES*RES))
    indices = ti.field(dtype=int, shape=(3*2*(RES-1)**2))
    mesh_colors = ti.Vector.field(3,dtype=ti.float32, shape=(RES**2)) # mesh        
    colors = ti.Vector.field(3,dtype=ti.float32, shape=(2*RES,2*RES)) # image            
    colormap_field = ti.Vector.field(3, dtype=ti.f32, shape=len(colormap))

    init_pts(pts=pts_chi, s_min=s_min, s_range=s_range)
    init_colormap(colormap_field, colormap)
    init_mesh_vertices(vertices)
    init_mesh_indices(indices)
    update_mesh_vertices(u_chi, mesh_colors, vertices, colormap_field, -2, 2, 3.0)

    """Torch"""
    pts = pts_chi.to_torch().to(device).requires_grad_(True)
    pts = pts.reshape(-1,2)
    t = torch.zeros((pts.shape[0],1), device=device)
    pts = torch.hstack([t,pts])
    t_steps = torch.arange(t_bounds[0], t_bounds[1], dt)


    """Taichi Window"""
    window = ti.ui.Window("2D Diffusion", (400,400))
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
        T_chi.from_torch(T.reshape(RES,RES))
        u_chi.from_torch(u.reshape(RES,RES))
        v_chi.from_torch(v.reshape(RES,RES))
        V_chi.from_torch(vorticity.reshape(RES,RES))
        q_chi.from_torch(T_y.reshape(RES,RES))
        # vel_chi.from_torch(vel_mag.reshape(RES,RES))

        if it % 100 == 0:
            print(f"---{it}---")
            print(f"T:{torch.min(T).item()},{torch.max(T).item()}")
            print(f"u:{torch.min(u).item()},{torch.max(u).item()}")
            print(f"v:{torch.min(v).item()},{torch.max(v).item()}")
            print(f"V:{torch.min(vorticity).item()},{torch.max(vorticity).item()}")
            # print(f"vel:{torch.min(vel_mag).item()},{torch.max(vel_mag).item()}")
            print(f"q:{torch.min(q_mag).item()},{torch.max(q_mag).item()}")
            print(f"T_y:{torch.min(T_y).item()},{torch.max(T_y).item()}")

        if USE_MESH:
            camera_t += camera_speed
            camera.position(camera_radius*ti.sin(camera_t), camera_height, camera_radius*ti.cos(camera_t))
            scene.set_camera(camera)
            scene.ambient_light((0.8, 0.8, 0.8))
            scene.point_light(pos=(0.0, 4.5, 0.0), color=(0.2, 0.2, 0.2))

            update_mesh_vertices(u_chi, mesh_colors, vertices, colormap_field, u_min, u_range, mesh_height)
            scene.mesh(vertices, indices, per_vertex_color=mesh_colors)
            canvas.scene(scene)
        else:
            update_colors(
                T=T_chi, # LL
                V=V_chi, # LR
                u=u_chi, # UL
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
        it = (it + 1) % t_steps.shape[0]
