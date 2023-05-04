import time
from tkinter import W

import numpy as np
import taichi as ti
import matplotlib.pyplot as plt


@ti.data_oriented
class Solver:
    def __init__(self, dt, dx, n, D, mat_ids, mat_defs, mat_diffs, boundary_values, colormap, u_min, u_range, updates_per_batch) -> None:

        """Consts"""
        self.u_min = u_min
        self.u_range = u_range
        self.n  = n
        self.dx = dx
        self.L  = dx * (n - 1)
        self.dt = dt
        self.c  = self.dt / (self.dx**2)
        self.updates_per_batch = updates_per_batch
        self.tol = 0.0001

        """Material Dicts"""
        self.mat_ids = mat_ids
        self.mat_defs = mat_defs
        self.mat_diffs = mat_diffs
        self.mat_defs_field = ti.field(dtype=float, shape=mat_defs.shape)
        self.mat_ids_field = ti.field(dtype=int, shape=(n,n))
        self.mat_diffs_field = ti.field(dtype=int, shape=mat_diffs.shape)

        "Time Field"
        self.t = ti.field(dtype=float, shape=())

        """Material Data Fields"""
        self.mat     = ti.field(dtype=ti.i8, shape=(n,n))
        self.k       = ti.field(dtype=float, shape=(n,n))
        self.k_mid   = ti.field(dtype=float, shape=(n,n,4)) # left, right, down, up
        self.k_sum   = ti.field(dtype=float, shape=(n,n))   # sum of k_mid
        self.rho     = ti.field(dtype=float, shape=(n,n))
        self.cp      = ti.field(dtype=float, shape=(n,n))
        self.cv      = ti.field(dtype=float, shape=(n,n))   # self.rho * self.cp
        self.c_cvinv = ti.field(dtype=float, shape=(n,n))   # self.rho * self.cp
        self.denom   = ti.field(dtype=float, shape=(n,n))   # self.rho * self.cp
        self.D       = ti.field(dtype=float, shape=(n,n))   # self.k / self.cv

        """BC Fields"""
        self.boundaries = ti.field(dtype=float, shape=(2,2))

        """Solution Fields"""
        self.u = ti.field(dtype=float, shape=(n,n))
        self.u_next = ti.field(dtype=float, shape=(n,n))
        self.u_tmp = ti.field(dtype=float, shape=(n,n))
        self.q = ti.field(dtype=float, shape=(n,n))

        """Rendering Fields"""
        self.colors   = ti.Vector.field(3, dtype=ti.f32, shape=(2*n, 2*n))
        self.colormap_field = ti.Vector.field(3, dtype=ti.f32, shape=len(colormap))
        



        """Inits"""
        self.t[None] = 0
        self.init_boundary_values(boundary_values)
        self.populate_u0()
        if type(self.mat_ids) is type(np.zeros(2)):
            self.mat_defs_field.from_numpy(self.mat_defs.astype(np.float32))
            self.mat_ids_field.from_numpy(self.mat_ids)
            self.mat_diffs_field.from_numpy(self.mat_diffs.astype(np.float32))
            self.populate_mat_properties()
        else:
            self.populate_D(D)
        self.precompute_coeffs()
        self.init_colormap(colormap)
        self.load_material_colors(D_max=D)
        # self.load_material_colors(D_max=ti.max(self.D))

        self.fig = plt.figure()
        self.X, self.Y = np.meshgrid(np.arange(self.n), np.arange(self.n), indexing="xy")
        d = self.cp.to_numpy().T
        dr = np.roll(d, 1,axis=0)
        du = np.roll(d, 1,axis=1)
        d = np.logical_or(du != d, dr != d)
        self.edges = d*255

    def init_boundary_values(self, boundary_values):
        # NB: this does not init the cells, just the reference values!
        for ver_vs_hor in range(2):
            for a_vs_b in range(2):
                bound = np.array(boundary_values)[ver_vs_hor , a_vs_b]
                if bound >= 0:
                    bound = bound*self.u_range + self.u_min
                else:
                    bound = -1
                self.boundaries[ver_vs_hor, a_vs_b] = bound

    def init_colormap(self, colormap):
        for i, color in enumerate(colormap):
            self.colormap_field[i] = ti.Vector(color)

    @ti.kernel
    def populate_u0(self):
        for k in ti.grouped(self.u):
            self.u[k] = self.u_min + self.u_range * k.x / self.n
            self.u[k] = self.u_min + ti.random()*self.u_range
            self.u[k] = self.u_min 
            # self.u[k] = self.u_min + self.u_range/2 + ti.abs(0.5 - k.x/self.n)*self.u_range - ti.abs(0.5 - k.y/self.n)*self.u_range

            """Prepare for Jacobi/Gauss-Seidel"""
            self.u_next[k] = self.u[k]

    @ti.kernel
    def populate_mat_properties(self):
        print("Populating material properties from mat map")
        # TODO: self.mat_ids_field is a duplicate of self.mat
        for col,row in self.D:
            self.k[col, row]   = self.mat_defs_field[self.mat_ids_field[col, row], 0]
            self.rho[col, row] = self.mat_defs_field[self.mat_ids_field[col, row], 1]
            self.cp[col, row]  = self.mat_defs_field[self.mat_ids_field[col, row], 2]
            self.D[col, row]   = self.k[col,row]/(self.rho[col,row]*self.cp[col,row])#self.mat_diffs_field[self.mat_ids_field[col, row]]
            self.mat[col, row] = self.mat_ids_field[col, row]

            # TODO: move to shared fn
            if col == 0 or row == 0 or col == self.n-1 or row == self.n-1:
                boundary = 0.0
                if col == 0 or col == self.n - 1:
                    boundary = self.boundaries[0, int(col / (self.n-1))]
                if row == 0 or row == self.n - 1:
                    boundary = self.boundaries[1, int(row / (self.n-1))]
                
                if boundary < 0:
                    """Adiabatic"""
                    self.mat[col, row] = ti.cast(-1, ti.i8)

                    self.k[col, row] = 0.000001
                    self.cp[col, row] = 1.0
                    self.rho[col, row] = 1.0
                    self.D[col, row] = 0.0

    @ti.kernel
    def populate_D(self, D: float):
        self.D.fill(D)

        for col, row in self.D:
            if row < 3/8 * self.n:
                self.mat[col, row] = ti.cast(self.mat_ids['outer'], ti.i8)

                self.k[col, row] = self.mat_defs[self.mat_ids['outer'], 0]
                self.cp[col, row] = self.mat_defs[self.mat_ids['outer'], 1]
                self.rho[col, row] = self.mat_defs[self.mat_ids['outer'], 2]
                self.D[col, row] = self.mat_diffs[self.mat_ids['outer']]
            elif row < 4/8 * self.n:
                self.mat[col, row] = ti.cast(self.mat_ids['xps'], ti.i8)

                self.k[col, row] = self.mat_defs[self.mat_ids['xps'], 0]
                self.cp[col, row] = self.mat_defs[self.mat_ids['xps'], 1]
                self.rho[col, row] = self.mat_defs[self.mat_ids['xps'], 2]
                self.D[col, row] = self.mat_diffs[self.mat_ids['xps']]
            elif row < 5/8 * self.n:
                self.mat[col, row] = ti.cast(self.mat_ids['concrete'], ti.i8)

                self.k[col, row] = self.mat_defs[self.mat_ids['concrete'], 0]
                self.cp[col, row] = self.mat_defs[self.mat_ids['concrete'], 1]
                self.rho[col, row] = self.mat_defs[self.mat_ids['concrete'], 2]
                self.D[col, row] = self.mat_diffs[self.mat_ids['concrete']]
            else:
                self.mat[col, row] = ti.cast(self.mat_ids['outer'], ti.i8)

                self.k[col, row] = self.mat_defs[mat_ids['outer'], 0]
                self.cp[col, row] = self.mat_defs[self.mat_ids['outer'], 1]
                self.rho[col, row] = self.mat_defs[self.mat_ids['outer'], 2]
                self.D[col, row] = self.mat_diffs[self.mat_ids['outer']]
            if (row >= 3/8*self.n and row <= 5/8*self.n) and ((col > 6/21*self.n and col < 7/21*self.n) or (col > 10/21*self.n and col < 11/21*self.n) or (col > 14/21*self.n and col < 15/21*self.n)):
                self.mat[col, row] = ti.cast(self.mat_ids['channel'], ti.i8)

                self.k[col, row] = self.mat_defs[self.mat_ids['channel'], 0]
                self.cp[col, row] = self.mat_defs[self.mat_ids['channel'], 1]
                self.rho[col, row] = self.mat_defs[self.mat_ids['channel'], 2]
                self.D[col, row] = self.mat_diffs[self.mat_ids['channel']]

            if col == 0 or row == 0 or col == self.n-1 or row == self.n-1:
                boundary = 0.0
                if col == 0 or col == self.n - 1:
                    boundary = self.boundaries[0, int(col / (self.n-1))]
                if row == 0 or row == self.n - 1:
                    boundary = self.boundaries[1, int(row / (self.n-1))]
                
                if boundary < 0:
                    """Adiabatic"""
                    self.mat[col, row] = ti.cast(-1, ti.i8)

                    self.k[col, row] = 0.000001
                    self.cp[col, row] = 1.0
                    self.rho[col, row] = 1.0
                    self.D[col, row] = 0.0


    @ti.kernel
    def precompute_coeffs(self):
        self.k_mid.fill(-1) # -1 on boundaries for now
        self.k_sum.fill(0)
        for col, row in self.cv:
            self.cv[col, row] = self.rho[col, row]*self.cp[col, row]
            # TODO: should self.dx be mul'd in here?
            # self.c_cvinv[col, row] = self.dx*self.c/self.cv[col, row]
            self.c_cvinv[col, row] = self.c/self.cv[col, row]
            # TODO: migrate to multi-dim final slot for col/row
            k = self.k[col, row]
            if col > 0:
                if col == 1 and self.boundaries[0,0] < 0:
                    self.k_mid[col, row, 0] = 0 # no heat flow from left into right cell
                else:
                    self.k_mid[col, row, 0] = (k + self.k[col - 1, row])/2
                    self.k_sum[col, row] += self.k_mid[col, row, 0]
            if col < self.n - 1:
                if col == self.n-2 and self.boundaries[0,1] < 0:
                    self.k_mid[col, row, 1] = 0 # no heat flow from right into left cell
                else: 
                    self.k_mid[col, row, 1] = (k + self.k[col + 1, row])/2
                    self.k_sum[col, row] += self.k_mid[col, row, 1]
            if row > 0:
                if row == 1 and self.boundaries[1,0] < 0:
                    self.k_mid[col, row, 2] = 0 # no heat flow from down into up cell
                else:
                    self.k_mid[col, row, 2] = (k + self.k[col, row - 1])/2
                    self.k_sum[col, row] += self.k_mid[col, row, 2]
            if row < self.n - 1:
                if row == self.n-2 and self.boundaries[1,1] < 0:
                    self.k_mid[col, row, 3] = 0 # no heat flow from up into down cell
                else:
                    self.k_mid[col, row, 3] = (k + self.k[col, row + 1])/2
                    self.k_sum[col, row] += self.k_mid[col, row, 3]
            self.denom[col, row] = 1 / (1 + self.k_sum[col, row]*self.c_cvinv[col, row])
    @ti.kernel
    def load_material_colors(self, D_max: float):
        # TODO: make mat colors assignable
        mat_colors = ti.Matrix(
            [
                [ti.random(), ti.random(), ti.random()],
                [ti.random(), ti.random(), ti.random()],
                [ti.random(), ti.random(), ti.random()],
                [ti.random(), ti.random(), ti.random()],
                [ti.random(), ti.random(), ti.random()]
            ]
        )

        for col, row in self.D:
            gray_scale = (self.D[col, row] - 0.01*D_max) / (0.99*D_max)
            self.colors[col+self.n, row] = ti.Vector([mat_colors[self.mat[col, row], 0], mat_colors[self.mat[col, row], 1], mat_colors[self.mat[col, row], 2]])
            self.colors[col+self.n,row+self.n] = ti.Vector([gray_scale, gray_scale, gray_scale])

    @ti.kernel
    def check_explicit_cfl(self):
        for i in ti.grouped(self.D):
            if self.D[i]*self.c > 0.25:
                print(f"Warning! CFL is quite high! {self.D[i]*self.c}")
            assert self.D[i]*self.c < 0.25, "The Courant condition is not satisified!"
            

    def explicit_batch(self):
        for i in range(self.updates_per_batch):
            self.explicit_step()
        self.compute_q()

    @ti.kernel
    def explicit_step(self):
        """Step"""
        for col, row in self.u:
            if col == 0 or row == 0 or col == self.n-1 or row == self.n-1:
                self.handle_boundary_explicit(col, row)
            else:
                self.handle_internal_jacobi(col, row)
        # ti.sync()

        """Shift"""
        for node in ti.grouped(self.u_next):
            self.u[node] = self.u_next[node]
        # ti.sync()

    @ti.kernel
    def compute_q(self):
        """Compute Q"""
        self.q.fill(0.0)
        for col, row in self.u:
            hor = 0.0
            ver = 0.0
            # TODO: handle min/max
            if col > 0:
                hor += (self.u[col, row] - self.u[col - 1, row]) * (self.k[col, row] + self.k[col - 1, row])/2
            if col < self.n - 1:
                hor += (self.u[col + 1, row] - self.u[col, row]) * (self.k[col, row] + self.k[col + 1, row])/2
            if row > 0:
                ver += (self.u[col, row] - self.u[col, row - 1]) * (self.k[col, row] + self.k[col, row - 1])/2
            if row < self.n - 1:
                ver += (self.u[col, row + 1] - self.u[col, row]) * (self.k[col, row] + self.k[col, row + 1])/2
            # TODO: precompute some of these consts.
            # self.q[col, row] = ti.log(ti.sqrt(ti.static(0.25/self.dx**2)*((hor**2 + ver**2))))/ti.static(ti.log(175))
            self.q[col, row] = ti.sqrt(ti.static(0.25/self.dx**2)*((hor**2 + ver**2)))/175

    @ti.func
    def handle_boundary_explicit(self, col, row):
        if col == 0 or col == self.n - 1:
            boundary = self.boundaries[0, int(col / (self.n-1))]
            if boundary >= 0: # TODO: create a separate flag
                """Prescriptive"""
                self.u_next[col, row] = boundary
        else:
            boundary = self.boundaries[1, int(row / (self.n-1))]
            if boundary >= 0: # TODO: create a separate flag
                """Prescriptive"""
                self.u_next[col, row] = boundary
    @ti.func 
    def handle_internal_jacobi(self, col, row):
        # k = self.k[col, row]
        # # # TODO: Move these static computations into a precomputed field
        # alpha_l = (self.k[col - 1, row] + k)/2
        # alpha_r = (self.k[col + 1, row] + k)/2
        # alpha_d = (self.k[col, row - 1] + k)/2
        # alpha_u = (self.k[col, row + 1] + k)/2
        k_l = self.k_mid[col, row, 0]
        k_r = self.k_mid[col, row, 1]
        k_d = self.k_mid[col, row, 2]
        k_u = self.k_mid[col, row, 3]

        # alpha = alpha_l + alpha_r + alpha_d + alpha_u
        # k = self.k_sum[col, row]
        hcp = self.c_cvinv[col, row] #self.c / (self.cv[col, row])
        denom = self.denom[col, row] # 1 + k*hcp

        err = 9999.0

        while err > self.tol:
            self.u_tmp[col, row] = (
                self.u[col,row] + (
                      k_l * self.u_next[col-1, row]\
                    + k_r * self.u_next[col+1, row]\
                    + k_d * self.u_next[col, row-1]\
                    + k_u * self.u_next[col, row+1]
                ) * hcp
            ) * denom
            err = ti.abs(self.u_tmp[col, row] - self.u_next[col, row])
            self.u_next[col, row] = self.u_tmp[col, row]
        
    def benchmark_explicit(self, n_tests):
        print("Starting benchmark...")
        solver.explicit_step() # force compilation
        s = time.time()
        for _ in range(n_tests):
            solver.explicit_batch()
        e = time.time()
        b_time = e - s
        step_time = b_time / (n_tests * self.updates_per_batch)
        node_time = b_time / (n_tests * self.updates_per_batch * self.n)
        nodes_per_s = 1/node_time
        steps_in_year = 365*24*60*60/self.dt
        steps_in_week = 7*24*60*60/self.dt
        steps_in_day = 24*60*60/self.dt
        steps_in_hour = 60*60/self.dt
        hour_time = step_time*steps_in_hour
        day_time = step_time*steps_in_day
        week_time = step_time*steps_in_week
        year_time = step_time*steps_in_year
        print(f"\n--- Benchmark Results ---")
        print(f"dt: {self.dt:0.3f}s")
        print(f"dx: {self.dx*100:0.1f}cm")
        print(f"ar: {self.L*self.L}m2")
        print(f"{int(step_time*1e6)}ns/timestep")
        print(f"{nodes_per_s/1e6:0.3f}m nodes/s")
        print(f"{hour_time}s/hour")
        print(f"{day_time}s/day")
        print(f"{week_time}s/week")
        print(f"{year_time/60}min/year")

    @ti.kernel
    def update_colors(self):
        for i,j in self.u:
            h = ti.cast(self.u[i,j] - self.u_min, ti.float32)/self.u_range
            
            level = ti.max(ti.min(ti.floor(h*(self.colormap_field.shape[0]-1)),self.colormap_field.shape[0]-2), 0)
            colorphase = ti.cast(ti.min(ti.max(h*(self.colormap_field.shape[0]-1) - level, 0),1), ti.f32)
            level_idx = ti.cast(level, dtype=int)

            self.colors[i,j].x = ti.cast((self.colormap_field[level_idx].x * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].x)/255, ti.float32)
            self.colors[i,j].y = ti.cast((self.colormap_field[level_idx].y * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].y)/255, ti.float32)
            self.colors[i,j].z = ti.cast((self.colormap_field[level_idx].z * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].z)/255, ti.float32)

            h = ti.abs(ti.cast(self.q[i,j], ti.float32))

            level = ti.max(ti.min(ti.floor(h*(self.colormap_field.shape[0]-1)),self.colormap_field.shape[0]-2), 0)
            colorphase = ti.cast(ti.min(ti.max(h*(self.colormap_field.shape[0]-1) - level, 0),1), ti.f32)
            level_idx = ti.cast(level, dtype=int)

            self.colors[i,j+self.n].x = ti.cast((self.colormap_field[level_idx].x * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].x)/255, ti.float32)
            self.colors[i,j+self.n].y = ti.cast((self.colormap_field[level_idx].y * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].y)/255, ti.float32)
            self.colors[i,j+self.n].z = ti.cast((self.colormap_field[level_idx].z * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].z)/255, ti.float32)
    
    def plot_isotherms(self):
            plt.cla()
            u = self.u.to_numpy()
            u_col = self.colors.to_numpy()[:self.n,:self.n]
            plt.imshow(np.clip(np.swapaxes(u_col, 0,1), 0, 1.0))
            plt.contour(self.X, self.Y, np.swapaxes(u,0,1), colors='black', levels=np.arange(273.15 -5,273.15+21,0.2), linewidths=(1, 0.25, 0.25, 0.25, 0.25))#, linewidths=(0.25, 0.25, 0.25, 1)), linewidths=(0.25, 0.25, 0.25, 0.25, 1))

            ax = plt.gca()
            ax.invert_yaxis()
            plt.axis("off")
            plt.draw()
            plt.pause(0.01)





if __name__ == '__main__':

    """
    Material Defs
    K [W/mK], Cp [J/kgK], rho [kg/m3]
    """

    print(f"--- Material Library ---")
    mat_names = [
        "outer",
        "channel",
        "concrete",
        "xps"
    ]
    mat_defs = np.array([
        [3, 100, 100], 
        [6, 100, 100],
        [2.3, 1000, 2300], 
        [0.039, 1450, 35],
    ])
    mat_diffs = mat_defs[:,0]/(mat_defs[:,1]*mat_defs[:,2])
    mat_ids = {
        name: i for i,name in enumerate(mat_names)
    }
    for name,mat_id in mat_ids.items():
        print(f"\nMaterial: {name}")
        print(f"k:   {mat_defs[mat_id, 0]:0.2f} [W/mK]")
        print(f"Cp:  { int(mat_defs[mat_id, 1]):04d} [J/kgK]")
        print(f"rho: { int(mat_defs[mat_id, 2]):04d} [kg/m3]")
        print(f"D:   {  mat_diffs[mat_id]:0.3e} [m2/s]")

    ti.init(arch=ti.cuda, default_fp=ti.f32)
    D = np.max(mat_diffs) # [m2/s]
    dx = 0.01 # [m]
    # dt = dx**2 / (4*D) # [s]
    dt = 1 # [s]
    print(f"Timestep: {int(dt*1000):01d}ms")
    p = 8
    n = 2**p

    u_range = 25
    u_min = 273.15 + (-5)

    boundary_values = [
        # [-1, -1],
        [1, 1],
        [1, 0]
    ]

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


    solver = Solver(dt, dx, n, D, mat_ids, mat_defs, mat_diffs, boundary_values, colormap, u_min, u_range, updates_per_batch=100)
    # solver.check_explicit_cfl()
    # solver.benchmark_explicit(n_tests=100)

    """
    Render setup
    """
    window_scale_factor = 0.75
    window = ti.ui.Window("2D Diffusion", (int(window_scale_factor*2*n),int(window_scale_factor*2*n)))
    canvas = window.get_canvas()

    it = 0
    t_marker = 1
    
    while window.running:

        solver.update_colors()
        canvas.set_image(solver.colors)

        solver.explicit_batch()
        # window.save_image(f"./week_5_fd_pde/images_4/{it:05d}.png")
        if it%200 == 0:
            solver.plot_isotherms()
        if it*solver.updates_per_batch*solver.dt/(3600*24) > t_marker:
            print(f"Completed day {t_marker}")
            t_marker +=1

        it += 1
        window.show()




