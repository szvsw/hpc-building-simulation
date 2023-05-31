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
import taichi as ti

from mlp import MLP

device = "cuda" if torch.cuda.is_available() else "device"
print(f"Using {device}")


@ti.data_oriented
class PINNSSSBeam:
    def __init__(
        self,
        net: MLP,
        lr=1e-4,
        collocation_ct=10000,
        bc_ct=1000,
        space_bounds=[-1, 1],
        adaptive_resample=True,
        sample_freq=10,
        soft_adapt_weights=True,
        model_prefix="SimplySupportedStaticBeam",
    ) -> None:
        plt.ion()
        self.fig, self.axs = plt.subplots(4, 1, figsize=(12, 9))
        self.net = net.to(device)
        self.loss_fn = nn.MSELoss()
        self.adam = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_history = {"bc": [], "phys": [], "weighted": []}

        self.s_min: float = space_bounds[0]
        self.s_max: float = space_bounds[1]
        self.s_range: float = self.s_max - self.s_min

        # TODO: make an indexing class
        self.loc_start_ix = 0
        self.loc_ct = 1
        self.geo_params_start_ix = self.loc_start_ix + self.loc_ct
        self.geo_params_ct = 6
        self.end_ix = self.geo_params_start_ix + self.geo_params_ct
        self.force_start_ix = self.end_ix  # TODO: placeholder

        self.a_min = -0.2
        self.a_max = 0.2
        self.a_range = self.a_max - self.a_min

        # self.b0 = 0.8
        # self.h0_min = 0.25
        # self.b1_min = 0.33
        # self.h1_min = 0.5

        self.b0 = 10
        self.h0_min = 1
        self.b1_min = 2
        self.h1_min = 3

        self.collocation_ct = collocation_ct
        self.bc_ct = bc_ct
        self.adaptive_resample = adaptive_resample
        self.sample_freq = sample_freq  # TODO: implement periodic resampling

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

        # self.I = 1
        # self.I_fn = lambda pts: -(pts[:,0:1])*(pts[:,1:2]-self.s_min)*(pts[:,1:2]-self.s_max) + 1
        self.E = 0.2
        self.E = 1
        # self.force_fn = lambda pts: 1/(0.1*np.sqrt(2*np.pi)) * torch.exp(-0.5*(pts**2)/(0.1**2))
        # self.force_fn = lambda pts: -1/(0.1*np.sqrt(2*np.pi)) * torch.exp(-0.5*(pts**2)/(0.1**2))
        # self.force_fn = lambda pts: -3 * torch.sigmoid((pts-0.5)*30)
        self.force_fn = lambda x, force_params: -1 + 0 * x
        self.dim = self.loc_ct + self.geo_params_ct

        """TI Viz Fields"""
        self.render_a_h0 = 0
        self.render_a_b1 = 0
        self.render_a_h1 = 0
        self.mesh_pts_per_edge = 1000
        self.beam_edge_ct = 8
        self.beam_long_face_ct = self.beam_edge_ct
        self.tris_per_beam_face = 2 * (self.mesh_pts_per_edge - 1)
        self.tri_ct = self.tris_per_beam_face * self.beam_long_face_ct + 2 * (
            2 + 4
        )  # endcaps
        self.edge_ct = self.tri_ct * 3 / 2  # every edge is counted twice
        self.mesh_pts = ti.Vector.field(
            3, dtype=ti.f32, shape=int(self.beam_edge_ct * self.mesh_pts_per_edge)
        )
        self.mesh_x = ti.field(dtype=ti.f32, shape=(self.mesh_pts_per_edge))
        mesh_x = torch.linspace(
            self.s_min, self.s_max, self.mesh_pts_per_edge, device=device
        )
        self.mesh_x.from_torch(mesh_x)
        self.mesh_indices = ti.field(dtype=int, shape=(self.tri_ct * 3))
        self.init_mesh_indices()
        self.init_mesh_pt_x_vals()
        self.update_mesh_yz_vals(a_h0=0.0,c_h0=0,a_b1=0,c_b1=0,a_h1=0,c_h1=0)

        self.window = ti.ui.Window("Shaped Beam", (1000, 1000))
        self.gui = self.window.get_gui()
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(0, 8, 0)
        self.camera.lookat(0, 0, 0)
        self.camera.up(0, 1, 0)
        self.camera_radius = 3.3
        self.camera_height = 0
        self.camera_speed = 0.001
        self.camera_t = 4.5
        self.light_x = 0
        self.light_y = -5
        self.light_z = 0
    
    def handle_gui(self):
        with self.gui.sub_window("Shaped Beam Parameters", x=0.05, y=0.05, width=0.5, height=0.1) as window:
            old_a_h0 = self.render_a_h0
            old_a_b1 = self.render_a_b1
            old_a_h1 = self.render_a_h1
            self.render_a_h0 = window.slider_float("Top Height Curvature", self.render_a_h0, self.a_min,  self.a_max)
            self.render_a_b1 = window.slider_float("Bottom Width Curvature", self.render_a_b1, self.a_min,  self.a_max)
            self.render_a_h1 = window.slider_float("Bottom Height Curvature", self.render_a_h1, self.a_min,  self.a_max)
            if self.render_a_h0 != old_a_h0 or self.render_a_b1 != old_a_b1 or self.render_a_h1 != old_a_h1: 
                self.update_mesh_yz_vals(self.render_a_h0, 0, self.render_a_b1, 0, self.render_a_h1, 0)
        
        with self.gui.sub_window("Camera Controls", x=0.6, y=0.05, width=0.35, height=0.1) as window:
            self.camera_speed = window.slider_float("Speed", self.camera_speed, 0, 0.01)
            self.camera_radius = window.slider_float("Radius", self.camera_radius, 0.5, 10)
            self.camera_height = window.slider_float("Height", self.camera_height, -3.5, 3.5)

        with self.gui.sub_window("Light Controls", x=0.6, y=0.85, width=0.35, height=0.1) as window:
            self.light_x = window.slider_float("X", self.light_x, -5,5)
            self.light_y = window.slider_float("Y", self.light_y, -5,5)
            self.light_z = window.slider_float("Z", self.light_z, -5,5)

    @ti.kernel
    def init_mesh_pt_x_vals(self):
        for i in self.mesh_pts:
            offset = ti.cast(i % self.mesh_pts_per_edge, int)
            # self.mesh_pts[i].x = offset / self.mesh_pts_per_edge * 3.0
            self.mesh_pts[i].x = self.mesh_x[offset]

    @ti.kernel
    def update_mesh_yz_vals(self, a_h0: float, c_h0: float, a_b1: float, c_b1: float, a_h1: float, c_h1: float):
        # y is up
        # b0 = 0.8
        # h0_min = 0.25
        # b1_min = 0.25
        # h1_min = 0.5
        b0 = self.b0
        h0_min = self.h0_min
        b1_min = self.b1_min
        h1_min = self.h1_min

        for i in range(self.mesh_pts_per_edge):
            x = self.mesh_x[i]

            parabola = (x - self.s_min) * (x - self.s_max)
            h0 = c_h0 + a_h0 * parabola + h0_min
            b1 = c_b1 + a_b1 * parabola + b1_min
            h1 = c_h1 + a_h1 * parabola + h1_min

            lower_left_inner = 0 * self.mesh_pts_per_edge + i
            lower_right_inner = 1 * self.mesh_pts_per_edge + i
            middle_right_inner = 2 * self.mesh_pts_per_edge + i
            middle_right_outer = 3 * self.mesh_pts_per_edge + i
            top_right_outer = 4 * self.mesh_pts_per_edge + i
            top_left_outer = 5 * self.mesh_pts_per_edge + i
            middle_left_outer = 6 * self.mesh_pts_per_edge + i
            middle_left_inner = 7 * self.mesh_pts_per_edge + i

            self.mesh_pts[top_left_outer].y = 0.0
            self.mesh_pts[top_left_outer].z = -b0 / 2.0

            self.mesh_pts[top_right_outer].y = 0.0
            self.mesh_pts[top_right_outer].z = b0 / 2.0

            self.mesh_pts[middle_left_outer].y = -h0
            self.mesh_pts[middle_left_outer].z = -b0 / 2.0

            self.mesh_pts[middle_right_outer].y = -h0
            self.mesh_pts[middle_right_outer].z = b0 / 2.0

            self.mesh_pts[middle_left_inner].y = -h0
            self.mesh_pts[middle_left_inner].z = -b1 / 2.0

            self.mesh_pts[middle_right_inner].y = -h0
            self.mesh_pts[middle_right_inner].z = b1 / 2.0

            self.mesh_pts[lower_left_inner].y = -h0 - h1
            self.mesh_pts[lower_left_inner].z = -b1 / 2.0

            self.mesh_pts[lower_right_inner].y = -h0 - h1
            self.mesh_pts[lower_right_inner].z = b1 / 2.0

    @ti.kernel
    def init_mesh_indices(self):
        """Long faces"""
        for face_id, quad_id in ti.ndrange(
            self.beam_long_face_ct, int(self.tris_per_beam_face / 2)
        ):
            quad_start_pt_ix = face_id * self.mesh_pts_per_edge + quad_id
            quad_adjacent_start_pt_ix = (
                (face_id + 1) % self.beam_edge_ct
            ) * self.mesh_pts_per_edge + quad_id
            tri_id_a = face_id * self.tris_per_beam_face * 3 + quad_id * 6
            tri_id_b = tri_id_a + 3
            self.mesh_indices[tri_id_a + 0] = ti.cast(quad_start_pt_ix, int)
            self.mesh_indices[tri_id_a + 1] = ti.cast(quad_start_pt_ix + 1, int)
            self.mesh_indices[tri_id_a + 2] = ti.cast(quad_adjacent_start_pt_ix, int)
            self.mesh_indices[tri_id_b + 0] = ti.cast(quad_start_pt_ix + 1, int)
            self.mesh_indices[tri_id_b + 1] = ti.cast(quad_adjacent_start_pt_ix, int)
            self.mesh_indices[tri_id_b + 2] = ti.cast(
                quad_adjacent_start_pt_ix + 1, int
            )

        """End caps"""
        for i in range(2):
            lower_left_inner = 0 * self.mesh_pts_per_edge + i * (
                self.mesh_pts_per_edge - 1
            )
            lower_right_inner = 1 * self.mesh_pts_per_edge + i * (
                self.mesh_pts_per_edge - 1
            )
            middle_right_inner = 2 * self.mesh_pts_per_edge + i * (
                self.mesh_pts_per_edge - 1
            )
            middle_right_outer = 3 * self.mesh_pts_per_edge + i * (
                self.mesh_pts_per_edge - 1
            )
            top_right_outer = 4 * self.mesh_pts_per_edge + i * (
                self.mesh_pts_per_edge - 1
            )
            top_left_outer = 5 * self.mesh_pts_per_edge + i * (
                self.mesh_pts_per_edge - 1
            )
            middle_left_outer = 6 * self.mesh_pts_per_edge + i * (
                self.mesh_pts_per_edge - 1
            )
            middle_left_inner = 7 * self.mesh_pts_per_edge + i * (
                self.mesh_pts_per_edge - 1
            )
            start_ix = self.tris_per_beam_face * self.beam_long_face_ct * 3 + i * 6 * 3

            # tri 1
            self.mesh_indices[start_ix + 0] = lower_left_inner
            self.mesh_indices[start_ix + 1] = lower_right_inner
            self.mesh_indices[start_ix + 2] = middle_left_inner

            # tri 2
            self.mesh_indices[start_ix + 3] = lower_right_inner
            self.mesh_indices[start_ix + 4] = middle_left_inner
            self.mesh_indices[start_ix + 5] = middle_right_inner

            # tri 3
            self.mesh_indices[start_ix + 6] = middle_left_outer
            self.mesh_indices[start_ix + 7] = middle_left_inner
            self.mesh_indices[start_ix + 8] = top_left_outer

            # tri 4
            self.mesh_indices[start_ix + 9] = middle_left_inner
            self.mesh_indices[start_ix + 10] = top_left_outer
            self.mesh_indices[start_ix + 11] = middle_right_inner

            # tri 5
            self.mesh_indices[start_ix + 12] = middle_right_inner
            self.mesh_indices[start_ix + 13] = top_left_outer
            self.mesh_indices[start_ix + 14] = top_right_outer

            # tri 6
            self.mesh_indices[start_ix + 15] = middle_right_inner
            self.mesh_indices[start_ix + 16] = middle_right_outer
            self.mesh_indices[start_ix + 17] = top_right_outer

    def render_scene(self):
        self.camera_t += self.camera_speed
        self.camera.position(
            self.camera_radius * ti.sin(self.camera_t),
            self.camera_height,
            self.camera_radius * ti.cos(self.camera_t),
        )
        self.scene.set_camera(self.camera)
        self.scene.point_light(pos=(self.light_x, self.light_y, self.light_z), color=(1,1,1))
        self.scene.mesh(self.mesh_pts, self.mesh_indices, color=(0.5, 0.2, 0.8), two_sided=True)
        self.canvas.scene(self.scene)
        self.window.show()

    def model_path(self, it):
        return (
            Path(os.path.abspath(os.path.dirname(__file__)))
            / "models"
            / f"{self.model_prefix}-{it:05d}.pth"
        )

    def extract_loc_ix(self, data):
        x = data[:, self.loc_start_ix : self.geo_params_start_ix]
        return x

    def extract_geo_params_ix(self, data):
        params = data[:, self.geo_params_start_ix : self.force_start_ix]
        return params

    def extract_force_params_ix(self, data):
        # TODO: placeholder
        return None

    def compute_I(self, x, parameters):
        # TODO: consider better vectorization
        # b0 = 10
        # h0_min = 1  # self.h0_min
        # b1_min = 2  # self.b1_min
        # h1_min = 3  # self.h1_min
        b0 = self.b0
        h0_min = self.h0_min
        b1_min = self.b1_min
        h1_min = self.h1_min

        a_h0 = parameters[:, 0:1]
        c_h0 = parameters[:, 1:2]

        a_b1 = parameters[:, 2:3]
        c_b1 = parameters[:, 3:4]

        a_h1 = parameters[:, 4:5]
        c_h1 = parameters[:, 5:6]

        parabola = (x - self.s_min) * (x - self.s_max)

        b0 = b0
        h0 = c_h0 + a_h0 * parabola + h0_min

        b1 = c_b1 + a_b1 * parabola + b1_min
        h1 = c_h1 + a_h1 * parabola + h1_min

        area0 = b0 * h0
        area1 = b1 * h1
        # hor_com = b0/2
        com_vert_0 = h0 / 2
        com_vert_1 = h0 + h1 / 2
        weighted_areas = com_vert_0 * area0 + com_vert_1 * area1
        total_area = area0 + area1
        coms = weighted_areas / total_area
        i_baseline_0 = b0 * h0**3 / 12
        i_baseline_1 = b1 * h1**3 / 12
        d0 = com_vert_0 - coms
        d1 = com_vert_1 - coms
        i_true = i_baseline_0 + area0 * d0**2 + i_baseline_1 + area1 * d1**2

        # print("area0", f"{torch.min(area0).item():0.3f}", f"{torch.max(area0).item():0.3f}")
        # print("area1", f"{torch.min(area1).item():0.3f}", f"{torch.max(area1).item():0.3f}")
        # print("i", f"{torch.min(i_true).item():0.3f}", f"{torch.max(i_true).item():0.3f}")

        return {
            "I": i_true,
            "h0": h0,
            "b1": b1,
            "h1": h1,
        }

    def sample_pts(self, ct, compute=True):
        pts_rand = torch.rand((ct, self.dim), device=device, requires_grad=True)
        x_rand = self.extract_loc_ix(pts_rand)
        geo_params_rand = self.extract_geo_params_ix(pts_rand)
        x = x_rand * self.s_range + self.s_min
        geo_params = torch.empty_like(geo_params_rand)
        geo_params[:, 0::2] = (
            # geo_params_rand[:, 0::2] * 1 - 0.5
            geo_params_rand[:, 0::2] * self.a_range + self.a_min
        )  # TODO: parameterize these and check bounds - no negs allowed!!
        geo_params[:, 1::2] = geo_params_rand[:, 1::2] * 0 - 0
        pts = torch.hstack([x, geo_params])
        results = self.governing_eq(pts) if compute else None
        return pts, results

    def sample_collocation_adaptive(self):
        initial_ct = self.collocation_ct * 10
        pts, results = self.sample_pts(initial_ct, compute=True)
        r2 = results["r2"]
        e_r2 = torch.mean(r2)
        p_X = (r2 / e_r2) / torch.sum(r2 / e_r2)
        pts_subsample_ix = torch.multinomial(
            p_X.flatten(), self.collocation_ct, replacement=True
        )
        pts_subsample = pts[pts_subsample_ix]
        results_subsample = {key: val[pts_subsample_ix] for key, val in results.items()}

        return pts_subsample, results_subsample

    def sample_collocation_non_adaptive(self):
        pts, results = self.sample_pts(self.collocation_ct, compute=True)
        return pts, results

    def sample_bcs(self):
        pts_rand, _ = self.sample_pts(self.bc_ct, compute=False)
        pts = torch.empty_like(pts_rand)
        half = int(self.bc_ct / 2)
        # force half the points to the left edge
        pts[:half, self.loc_start_ix : self.geo_params_start_ix] = (
            pts_rand[:half, self.loc_start_ix : self.geo_params_start_ix] * 0
            + self.s_min
        )
        # force half the points to the right edge
        pts[half:, self.loc_start_ix : self.geo_params_start_ix] = (
            pts_rand[half:, self.loc_start_ix : self.geo_params_start_ix] * 0
            + self.s_max
        )

        results = self.governing_eq(pts)
        return pts, results

    def euler_bernoulli(self, pts, w):
        x = self.extract_loc_ix(pts)
        geo_params = self.extract_geo_params_ix(pts)
        w_grad = torch.autograd.grad(
            inputs=pts,
            outputs=w,
            grad_outputs=torch.ones_like(w).to(device),
            create_graph=True,
            retain_graph=True,
        )[0]

        w_x = self.extract_loc_ix(w_grad)

        w_x_grad = torch.autograd.grad(
            inputs=pts,
            outputs=w_x,
            grad_outputs=torch.ones_like(w_x).to(device),
            create_graph=True,
            retain_graph=True,
        )[0]

        w_xx = self.extract_loc_ix(w_x_grad)

        I_results = self.compute_I(x, geo_params)
        I = I_results["I"]
        EIw_xx = self.E * I * w_xx

        EIw_xx_grad = torch.autograd.grad(
            inputs=pts,
            outputs=EIw_xx,
            grad_outputs=torch.ones_like(EIw_xx).to(device),
            create_graph=True,
            retain_graph=True,
        )[0]

        EIw_xxx = self.extract_loc_ix(EIw_xx_grad)

        EIw_xxx_grad = torch.autograd.grad(
            inputs=pts,
            outputs=EIw_xxx,
            grad_outputs=torch.ones_like(EIw_xxx).to(device),
            create_graph=True,
            retain_graph=True,
        )[0]

        EIw_xxxx = self.extract_loc_ix(EIw_xxx_grad)

        pde_data = {
            "x": x,
            "w": w,
            "w_x": w_x,
            "w_xx": w_xx,
            "EIw_xx": EIw_xx,
            "EIw_xxx": EIw_xxx,
            "EIw_xxxx": EIw_xxxx,
        }

        return {**I_results, **pde_data}

    def governing_eq(self, pts):
        """Predict"""
        w = self.net(pts)
        results = self.euler_bernoulli(pts, w)
        x = results["x"]
        f = self.force_fn(x=x, force_params=None)
        residual = results["EIw_xxxx"] - f
        r2 = residual**2
        results["residual"] = residual
        results["r2"] = r2

        return results

    def train_step(self):
        """Reset grads"""
        self.adam.zero_grad()

        """Sample pts"""
        pts_col, results_col = (
            self.sample_collocation_adaptive()
            if self.adaptive_resample
            else self.sample_collocation_non_adaptive()
        )
        pts_bc, results_bc = self.sample_bcs()

        loss_physics = torch.mean(results_col["r2"])

        bc_w = results_bc["w"]
        bc_w_xx = results_bc["w_xx"]
        bc_pred = torch.vstack([bc_w, bc_w_xx])
        # bc_true = torch.zeros_like(bc_pred)
        loss_bc = torch.mean(
            bc_pred**2
        )  # simply supported - no vertical displacement, no moment

        """Loss"""
        if self.soft_adapt_weights:
            with torch.no_grad():
                losses = torch.hstack([loss_bc, loss_physics])
                top = torch.exp(self.lambda_Temp * (self.losses_prev - losses))
                lambdas = top / torch.sum(top)
                self.losses_prev = losses
            loss = torch.sum(lambdas * torch.hstack([loss_bc, loss_physics]))
            loss.backward()
        else:
            loss = self.bc_weight * loss_bc + self.phys_weight * loss_physics
            loss.backward()

        """Reporting/Saving etc"""
        if self.it % self.save_frequency == 0:
            torch.save(self.net.state_dict(), self.model_path(self.it))

        if self.it % self.reporting_frequency == 0:
            print(
                "\n BC / PDE / Weighted",
                loss_bc.item(),
                loss_physics.item(),
                loss.item(),
            )

        if self.it % self.plot_frequency == 0:
            for i in range(500):
            # while pinn.window.running:
                self.handle_gui()
                self.render_scene()
            n_pts = self.mesh_pts_per_edge
            n_alphas = 1
            x = torch.linspace(self.s_min, self.s_max, n_pts, device=device).reshape(
                -1, 1
            )
            geo_params = torch.tile(
                torch.Tensor([self.render_a_h0, 0, self.render_a_b1, 0, self.render_a_h1, 0]).to(device), (n_pts, 1)
            )
            pts = torch.hstack([x, geo_params])
            pts_flat = pts.reshape(-1, 7)

            with torch.no_grad():
                w = self.net(pts_flat).reshape(n_pts, n_alphas, 1)
                I_results = self.compute_I(x=x, parameters=geo_params)
                I = I_results["I"].reshape(n_pts, n_alphas, 1)
                h0 = I_results["h0"].reshape(n_pts, n_alphas, 1)
                b1 = I_results["b1"].reshape(n_pts, n_alphas, 1)
                h1 = I_results["h1"].reshape(n_pts, n_alphas, 1)
                f = self.force_fn(x=x, force_params=None).reshape(n_pts, n_alphas, 1)

            for i in range(4):
                self.axs[i].clear()
            self.axs[0].set_aspect("equal")

            for i in range(n_alphas):
                self.axs[0].plot(
                    x.flatten().cpu(), w[:, i, :].flatten().cpu(), label=f"a={i}", lw=1
                )

            for i in range(n_alphas):
                self.axs[1].plot(
                    x.flatten().cpu(), I[:, i, :].flatten().cpu(), label=f"a={i}", lw=1
                )

            for i in range(n_alphas):
                self.axs[2].plot(
                    x.flatten().cpu(), f[:, i, :].flatten().cpu(), label=f"a={i}", lw=1
                )

            for i in range(n_alphas):
                self.axs[3].plot(
                    x.flatten().cpu(), h0[:, i, :].flatten().cpu(), label=f"h0", lw=1
                )
                self.axs[3].plot(
                    x.flatten().cpu(), b1[:, i, :].flatten().cpu(), label=f"b1", lw=1
                )
                self.axs[3].plot(
                    x.flatten().cpu(), h1[:, i, :].flatten().cpu(), label=f"h1", lw=1
                )

            self.axs[0].set_title("Displacement")
            self.axs[0].set_ylim([-0.5, 0.1])
            self.axs[1].set_title("Moment of Inertia")
            self.axs[1].set_ylim([-1, 40])
            self.axs[2].set_title("Applied Force")
            self.axs[2].set_ylim([-5, 5])
            self.axs[3].set_title("Geo Dimensions")
            self.axs[3].set_ylim([-1, 5])

            for i in range(4):
                self.axs[i].legend()
                self.axs[i].plot(
                    [self.s_min, self.s_min], [-100, 100], "--", c="gray", lw=0.5
                )
                self.axs[i].plot(
                    [self.s_max, self.s_max], [-100, 100], "--", c="gray", lw=0.5
                )
                self.axs[i].plot(
                    [self.s_min, self.s_max], [0, 0], "--", c="gray", lw=0.5
                )
            self.fig.tight_layout()
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()

        self.it = self.it + 1

        self.loss_history["phys"].append(loss_physics.item())
        self.loss_history["bc"].append(loss_bc.item())
        self.loss_history["weighted"].append(loss.item())
        return loss

    def train(
        self,
        n_epochs=500,
        reporting_frequency=None,
        plot_frequency=None,
        phys_weight=None,
        bc_weight=None,
    ):
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
    space_bounds = [-1, 1]
    collocation_ct = 10000  # 5000
    bc_ct = 5000  # 1000

    ti.init(arch=ti.gpu, default_fp=ti.f32)
    pinn = PINNSSSBeam(
        net=MLP(
            input_dim=7,
            output_dim=1,
            hidden_layer_ct=6,
            hidden_dim=64,
            act=F.tanh,
            learnable_act="SINGLE",
        ),
        lr=1e-3,
        collocation_ct=collocation_ct,
        bc_ct=bc_ct,
        space_bounds=space_bounds,
        adaptive_resample=ADAPTIVE_SAMPLING,
        soft_adapt_weights=ADAPTIVE_WEIGHTING,
        model_prefix=MODEL_PREFIX,
    )

    for i in range(10):
    # while pinn.window.running:
        pinn.handle_gui()
        pinn.render_scene()

    # for i in range(10):
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
        pinn.train(
            n_epochs=1000,
            reporting_frequency=100,
            plot_frequency=20,
            phys_weight=1,
            bc_weight=8,
        )
        # pinn.plot_bcs_and_ics(mode='error', ct=5000)
