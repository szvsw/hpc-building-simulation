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
def update_mesh_vertices(
    u: ti.template(), 
    mesh_colors:ti.template(), 
    vertices: ti.template(),
    colormap_field: ti.template(), 
    u_min: float, 
    u_range: float, 
    mesh_height: float
):
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
def update_colors(u:ti.template(), q: ti.template(), a: ti.template(), b: ti.template(), colors: ti.template(), colormap_field: ti.template(), u_min: float, u_range: float, q_min: float, q_range: float, a_min: float, a_range: float, b_min: float, b_range: float):
    for i,j in u:
        h = ti.cast(u[i,j] - u_min, ti.float32)/u_range
        
        level = ti.max(ti.min(ti.floor(h*(colormap_field.shape[0]-1)),colormap_field.shape[0]-2), 0)
        colorphase = ti.cast(ti.min(ti.max(h*(colormap_field.shape[0]-1) - level, 0),1), ti.f32)
        level_idx = ti.cast(level, dtype=int)

        colors[i,j].x = ti.cast((colormap_field[level_idx].x * (1-colorphase) + colorphase*colormap_field[level_idx+1].x)/255, ti.float32)
        colors[i,j].y = ti.cast((colormap_field[level_idx].y * (1-colorphase) + colorphase*colormap_field[level_idx+1].y)/255, ti.float32)
        colors[i,j].z = ti.cast((colormap_field[level_idx].z * (1-colorphase) + colorphase*colormap_field[level_idx+1].z)/255, ti.float32)

        h = ti.cast(q[i,j] - q_min, ti.float32)/q_range
        
        level = ti.max(ti.min(ti.floor(h*(colormap_field.shape[0]-1)),colormap_field.shape[0]-2), 0)
        colorphase = ti.cast(ti.min(ti.max(h*(colormap_field.shape[0]-1) - level, 0),1), ti.f32)
        level_idx = ti.cast(level, dtype=int)

        colors[RES+i,j].x = ti.cast((colormap_field[level_idx].x * (1-colorphase) + colorphase*colormap_field[level_idx+1].x)/255, ti.float32)
        colors[RES+i,j].y = ti.cast((colormap_field[level_idx].y * (1-colorphase) + colorphase*colormap_field[level_idx+1].y)/255, ti.float32)
        colors[RES+i,j].z = ti.cast((colormap_field[level_idx].z * (1-colorphase) + colorphase*colormap_field[level_idx+1].z)/255, ti.float32)

        h = ti.cast(a[i,j] - a_min, ti.float32)/a_range
        
        level = ti.max(ti.min(ti.floor(h*(colormap_field.shape[0]-1)),colormap_field.shape[0]-2), 0)
        colorphase = ti.cast(ti.min(ti.max(h*(colormap_field.shape[0]-1) - level, 0),1), ti.f32)
        level_idx = ti.cast(level, dtype=int)

        colors[i,j+RES].x = ti.cast((colormap_field[level_idx].x * (1-colorphase) + colorphase*colormap_field[level_idx+1].x)/255, ti.float32)
        colors[i,j+RES].y = ti.cast((colormap_field[level_idx].y * (1-colorphase) + colorphase*colormap_field[level_idx+1].y)/255, ti.float32)
        colors[i,j+RES].z = ti.cast((colormap_field[level_idx].z * (1-colorphase) + colorphase*colormap_field[level_idx+1].z)/255, ti.float32)

        h = ti.cast(b[i,j] - b_min, ti.float32)/b_range
        
        level = ti.max(ti.min(ti.floor(h*(colormap_field.shape[0]-1)),colormap_field.shape[0]-2), 0)
        colorphase = ti.cast(ti.min(ti.max(h*(colormap_field.shape[0]-1) - level, 0),1), ti.f32)
        level_idx = ti.cast(level, dtype=int)

        colors[RES+i,j+RES].x = ti.cast((colormap_field[level_idx].x * (1-colorphase) + colorphase*colormap_field[level_idx+1].x)/255, ti.float32)
        colors[RES+i,j+RES].y = ti.cast((colormap_field[level_idx].y * (1-colorphase) + colorphase*colormap_field[level_idx+1].y)/255, ti.float32)
        colors[RES+i,j+RES].z = ti.cast((colormap_field[level_idx].z * (1-colorphase) + colorphase*colormap_field[level_idx+1].z)/255, ti.float32)

        # h = ti.abs(ti.cast(self.q[i,j], ti.float32))

        # level = ti.max(ti.min(ti.floor(h*(self.colormap_field.shape[0]-1)),self.colormap_field.shape[0]-2), 0)
        # colorphase = ti.cast(ti.min(ti.max(h*(self.colormap_field.shape[0]-1) - level, 0),1), ti.f32)
        # level_idx = ti.cast(level, dtype=int)

        # self.colors[i,j+self.n].x = ti.cast((self.colormap_field[level_idx].x * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].x)/255, ti.float32)
        # self.colors[i,j+self.n].y = ti.cast((self.colormap_field[level_idx].y * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].y)/255, ti.float32)
        # self.colors[i,j+self.n].z = ti.cast((self.colormap_field[level_idx].z * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].z)/255, ti.float32)
    for i, j in colors:
        # if ti.abs(i-colors.shape[0]/2) < 1 or ti.abs(i-colors.shape[0]/4) < 1 or ti.abs(i-3*colors.shape[0]/4) < 1 or ti.abs(j-colors.shape[1]/2) < 1 or ti.abs(j-colors.shape[1]/4) < 1 or ti.abs(j-3*colors.shape[1]/4) < 1: 
        #     colors[i,j].x = 0
        #     colors[i,j].y = 0
        #     colors[i,j].z = 0
        if ti.abs(i-colors.shape[0]/2) < 3 or ti.abs(j-colors.shape[1]/2) < 3: 
            colors[i,j].x = 0
            colors[i,j].y = 0
            colors[i,j].z = 0

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
    model_path =  "last_runs.pth"
    model_path =  "a100_runs.pth"
    OUTPUT_DIM=1
    USE_MESH = False
    dt = 0.02
    t_bounds = [0, 16*np.pi]
    s_bounds = [-4*np.pi, 4*np.pi]
    T_min, T_range = -5.5, 11   # LL
    q_min, q_range = 0, 0.8 # LR
    qx_min, qx_range = -0.2, 0.4 # UL
    qy_min, qy_range = 0, 0.9 #UR

    mesh_height = 2.0

    """Computed"""
    s_range = s_bounds[1] - s_bounds[0]
    s_min = s_bounds[0]

    """Net"""
    net = MLP(input_dim=3, output_dim=OUTPUT_DIM, hidden_layer_ct=4,hidden_dim=256, act=F.tanh, learnable_act="SINGLE")
    net_path = Path(os.path.abspath(os.path.dirname(__file__))) / "models" / model_path
    net.load_state_dict(torch.load(net_path))
    net = net.to(device)

    """Fields"""
    ti.init(arch=ti.gpu)
    pts_chi = ti.Vector.field(2, ti.f32, shape=(RES, RES))
    u_chi = ti.field(ti.f32, shape=(RES, RES))
    q_chi = ti.field(ti.f32, shape=(RES, RES))
    qx_chi = ti.field(ti.f32, shape=(RES, RES))
    qy_chi = ti.field(ti.f32, shape=(RES, RES))
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
    w_scale=3
    window = ti.ui.Window("2D Diffusion", (RES*w_scale, RES*w_scale))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 8, 0)
    camera.lookat(0,0,0)
    camera.up(0,1,0)
    camera_radius = 6
    camera_height = 6
    camera_speed = 0.001
    camera_t = 4.5


    it = 0
    while window.running:
        pts[:,0] = t_steps[it]
        T = net(pts)
        grad = torch.autograd.grad(
            inputs=pts,
            outputs=T,
            grad_outputs=torch.ones_like(T),
            create_graph=False,
            retain_graph=False
        )[0]
        q = grad[:,1:]
        q = torch.sqrt(torch.sum(q**2, axis=1))
        q_x = -grad[:,1:2]
        q_y = -grad[:,2:3]
        u_chi.from_torch(T.reshape(RES,RES))
        q_chi.from_torch(q.reshape(RES,RES))
        qx_chi.from_torch(q_x.reshape(RES,RES))
        qy_chi.from_torch(q_y.reshape(RES,RES))

        if it % 100 == 0:
            print(f"---{it}---")
            print(f"min/max T: {torch.min(T).item()}, {torch.max(T).item()}")
            print(f"min/max q: {torch.min(q).item()}, {torch.max(q).item()}")
            print(f"min/max qx:{torch.min(q_x).item()}, {torch.max(q_x).item()}")
            print(f"min/max qy:{torch.min(q_y).item()}, {torch.max(q_y).item()}")

        if USE_MESH:
            camera_t += camera_speed
            camera.position(camera_radius*ti.sin(camera_t), camera_height, camera_radius*ti.cos(camera_t))
            scene.set_camera(camera)
            scene.ambient_light((0.8, 0.8, 0.8))
            scene.point_light(pos=(0.0, 4.5, 0.0), color=(0.2, 0.2, 0.2))

            update_mesh_vertices(u_chi, mesh_colors, vertices, colormap_field, T_min, T_range, mesh_height)
            scene.mesh(vertices, indices, per_vertex_color=mesh_colors)
            canvas.scene(scene)
        else:
            update_colors(u_chi, q_chi, qx_chi, qy_chi, colors, colormap_field, T_min, T_range, q_min, q_range, qx_min, qx_range, qy_min, qy_range)
            canvas.set_image(colors)

        window.show()
        if (it + 1) % t_steps.shape[0] == 0:
            exit()
        window.save_image(f"pinn/videos/{it:05d}.png")
        it = (it + 1) % t_steps.shape[0]
