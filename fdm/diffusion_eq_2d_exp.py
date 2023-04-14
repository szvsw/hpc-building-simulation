import time

import numpy as np
import taichi as ti
import matplotlib.pyplot as plt

"""
Material Defs
K [W/mK], Cp [J/kgK], rho [kg/m3]

J s^-1 m^-1 K^-1 J^-1 kg K kg^-1 m3
J J^-1  K^-1 K kg kg^-1  m3 m^-1 s^-1 
m2 m s^-1  
"""
print(f"--- Material Library ---")
mat_names = [
    "outer",
    "channel",
    "concrete",
    "xps"
]
mat_defs = np.array([
    [3, 3000, 1000], 
    [6, 3000, 1000],
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


@ti.data_oriented
class Solver:
    def __init__(self, dt, dx, n, D, boundary_values, colormap, u_min, u_range, updates_per_batch) -> None:

        """Consts"""
        self.u_min = u_min
        self.u_range = u_range
        self.n  = n
        self.dx = dx
        self.L  = dx * (n - 1)
        self.dt = dt
        self.c  = self.dt / self.dx**2
        self.updates_per_batch = updates_per_batch
        self.tol = 0.001

        "Time Field"
        self.t = ti.field(dtype=float, shape=())

        """Material Data Fields"""
        self.mat = ti.field(dtype=ti.i8, shape=(n,n))
        self.k  = ti.field(dtype=float, shape=(n,n))
        self.rho  = ti.field(dtype=float, shape=(n,n))
        self.cp  = ti.field(dtype=float, shape=(n,n))
        self.D  = ti.field(dtype=float, shape=(n,n))

        """BC Fields"""
        self.boundaries = ti.field(dtype=float, shape=(2,2))

        """Solution Fields"""
        self.u = ti.field(dtype=float, shape=(n,n))
        self.u_next = ti.field(dtype=float, shape=(n,n))
        self.u_temp = ti.field(dtype=float, shape=(n,n))
        self.q = ti.field(dtype=float, shape=(n,n))

        """Rendering Fields"""
        self.colors   = ti.Vector.field(3, dtype=ti.f32, shape=(2*n, 2*n))
        self.colormap_field = ti.Vector.field(3, dtype=ti.f32, shape=len(colormap))


        """Inits"""
        self.t[None] = 0
        self.populate_D(D)
        self.init_boundary_values(boundary_values)
        self.init_colormap(colormap)




    def init_boundary_values(self, boundary_values):
        # NB: this does not init the cells, just the reference values!
        for col in range(2):
            for row in range(2):
                bound = np.array(boundary_values)[col, row]
                if bound >= 0:
                    bound = bound*self.u_range + self.u_min
                else:
                    bound = -1
                self.boundaries[col, row] = bound

    def init_colormap(self, colormap):
        for i, color in enumerate(colormap):
            self.colormap_field[i] = ti.Vector(color)
    @ti.kernel
    def populate_D(self, D: float):
        for k in ti.grouped(self.u):
            self.u[k] = self.u_min + self.u_range * k.x / self.n
            self.u[k] = self.u_min + ti.random()*self.u_range
            # self.u[k] = self.u_min + self.u_range/2 + ti.abs(0.5 - k.x/self.n)*self.u_range - ti.abs(0.5 - k.y/self.n)*self.u_range

            self.u_next[k] = self.u[k]
        self.D.fill(D)

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
            if row < 3/8 * self.n:
                self.D[col, row] = 0.5*D

                self.mat[col, row] = ti.cast(mat_ids['outer'], ti.i8)

                self.k[col, row] = mat_defs[mat_ids['outer'], 0]
                self.cp[col, row] = mat_defs[mat_ids['outer'], 1]
                self.rho[col, row] = mat_defs[mat_ids['outer'], 2]
                self.D[col, row] = mat_diffs[mat_ids['outer']]
            elif row < 4/8 * self.n:
                self.D[col, row] = 0.1*D

                self.mat[col, row] = ti.cast(mat_ids['xps'], ti.i8)

                self.k[col, row] = mat_defs[mat_ids['xps'], 0]
                self.cp[col, row] = mat_defs[mat_ids['xps'], 1]
                self.rho[col, row] = mat_defs[mat_ids['xps'], 2]
                self.D[col, row] = mat_diffs[mat_ids['xps']]
            elif row < 5/8 * self.n:
                self.D[col, row] = 0.01*D
                self.D[col, row] = 0.1*D

                self.mat[col, row] = ti.cast(mat_ids['concrete'], ti.i8)

                self.k[col, row] = mat_defs[mat_ids['concrete'], 0]
                self.cp[col, row] = mat_defs[mat_ids['concrete'], 1]
                self.rho[col, row] = mat_defs[mat_ids['concrete'], 2]
                self.D[col, row] = mat_diffs[mat_ids['concrete']]
            else:
                self.D[col, row] = 0.5*D

                self.mat[col, row] = ti.cast(mat_ids['outer'], ti.i8)

                self.k[col, row] = mat_defs[mat_ids['outer'], 0]
                self.cp[col, row] = mat_defs[mat_ids['outer'], 1]
                self.rho[col, row] = mat_defs[mat_ids['outer'], 2]
                self.D[col, row] = mat_diffs[mat_ids['outer']]
            if (row >= 3/8*self.n and row <= 5/8*self.n) and ((col > 6/21*self.n and col < 7/21*self.n) or (col > 10/21*self.n and col < 11/21*self.n) or (col > 14/21*self.n and col < 15/21*self.n)):
                self.D[col, row] = D

                self.mat[col, row] = ti.cast(mat_ids['channel'], ti.i8)

                self.k[col, row] = mat_defs[mat_ids['channel'], 0]
                self.cp[col, row] = mat_defs[mat_ids['channel'], 1]
                self.rho[col, row] = mat_defs[mat_ids['channel'], 2]
                self.D[col, row] = mat_diffs[mat_ids['channel']]
            gray_scale = (self.D[col, row] - 0.01*D) / (0.99*D)
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

    @ti.kernel
    def explicit_step(self):
        """Step"""
        for col, row in self.u:
            if col == 0 or row == 0 or col == self.n-1 or row == self.n-1:
                self.handle_boundary_explicit(col, row)
            else:
                self.handle_internal_jacobi(col, row)
        ti.sync()

        """Shift"""
        for node in ti.grouped(self.u_next):
            self.u[node] = self.u_next[node]
        ti.sync()

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
            self.q[col, row] = ti.log(ti.sqrt(ti.static(0.25/self.dx**2)*((hor**2 + ver**2))))/ti.static(ti.log(175))

    @ti.func
    def handle_boundary_explicit(self, col, row):
        if col == 0 or col == self.n - 1:
            boundary = self.boundaries[0, int(col / (self.n-1))]
            if boundary >= 0: # TODO: create a separate flag
                """Prescriptive"""
                self.u_next[col, row] = boundary
            else:
                """Adiabatic"""
                mult = 1.0
                alpha = self.D[col, row]*self.c
                right = 0.0
                left = 0.0
                up = 0.0
                down = 0.0
                if col > 0:
                    left  = self.u[col - 1, row]
                else:
                    right = self.u[col + 1, row]
                if row > 0:
                    down = self.u[col, row - 1]
                    mult = mult + 1
                if row < self.n-1:
                    up = self.u[col, row + 1]
                    mult = mult = mult + 1
                self.u_next[col, row] = (1 - mult*alpha) * self.u[col, row] + alpha * (left + right + up + down)
        else:
            boundary = self.boundaries[1, int(row / (self.n-1))]
            if boundary >= 0: # TODO: create a separate flag
                """Prescriptive"""
                self.u_next[col, row] = boundary
            else:
                """Adiabatic"""
                mult = 1.0
                alpha = self.D[col, row]*self.c
                right = 0.0
                left = 0.0
                up = 0.0
                down = 0.0
                if row > 0:
                    down = self.u[col, row - 1]
                else:
                    up = self.u[col, row + 1]
                if col > 0:
                    left = self.u[col - 1, row]
                    mult = mult + 1
                if col < self.n-1:
                    right = self.u[col + 1, row]
                    mult = mult = mult + 1
                self.u_next[col, row] = (1 - mult*alpha) * self.u[col, row] + alpha * (left + right + up + down)
    @ti.func 
    def handle_internal_jacobi(self, col, row):
        k = self.k[col, row]
        # TODO: Move these static computations into a precomputed field
        alpha_l = (self.k[col - 1, row] + k)/2
        alpha_r = (self.k[col + 1, row] + k)/2
        alpha_d = (self.k[col, row - 1] + k)/2
        alpha_u = (self.k[col, row + 1] + k)/2
        alpha = alpha_l + alpha_r + alpha_d + alpha_u
        denom = 1 + alpha*self.c
        weight_l = alpha_l * self.c
        weight_r = alpha_r * self.c
        weight_u = alpha_u * self.c
        weight_d = alpha_d * self.c

        err = 9999.0
        jac_it = 0

        while err > self.tol:
            self.u_temp[col, row] = (self.u[col,row] \
                + weight_l * self.u_next[col-1, row]\
                + weight_r * self.u_next[col+1, row]\
                + weight_d * self.u_next[col, row-1]\
                + weight_u * self.u_next[col, row+1]) / denom
            err = ti.abs(self.u_temp[col, row] - self.u_next[col, row])
            self.u_next[col, row] = self.u_temp[col, row]
            jac_it = jac_it+1
        
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





if __name__ == '__main__':
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    D = np.max(mat_diffs) # [m2/s]
    dx = 0.0025 # [m]
    dt = dx**2 / (4*D*32) # [s]
    print(f"Timestep: {int(dt*1000):01d}ms")
    p = 10
    n = 2**p

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
    jet = plt.cm.get_cmap("jet")
    colormap = jet(np.arange(jet.N))*255




    u_range = 25
    u_min = 273.15 + (-5)

    solver = Solver(dt, dx, n, D, boundary_values, colormap, u_min, u_range, updates_per_batch=100)
    solver.check_explicit_cfl()
    # solver.benchmark_explicit(n_tests=100)

    """
    Render setup
    """
    window_scale_factor = 2
    window = ti.ui.Window("2D Diffusion", (window_scale_factor*2*n,window_scale_factor*2*n))
    canvas = window.get_canvas()

    it = 0
    t_marker = 1
    
    fig = plt.figure()
    X, Y = np.meshgrid(np.arange(solver.n), np.arange(solver.n), indexing="xy")
    while window.running:

        solver.update_colors()
        canvas.set_image(solver.colors)

        solver.explicit_batch()
        # window.save_image(f"./week_5_fd_pde/images_4/{it:05d}.png")
        if it%200 == 0:
            plt.cla()
            u = solver.u.to_numpy()
            u_col = solver.colors.to_numpy()[:solver.n,:solver.n]
            plt.imshow(np.clip(np.swapaxes(u_col, 0,1), 0, 1.0))
            plt.contour(X, Y, np.swapaxes(u,0,1), colors='black', levels=np.arange(273.15 -6,273.15+21,0.25), linewidths=(0.25))#, linewidths=(0.25, 0.25, 0.25, 1))
            ax = plt.gca()
            ax.invert_yaxis()
            plt.axis("off")
            plt.draw()
            plt.pause(0.01)
        if it*solver.updates_per_batch*solver.dt/(3600*24) > t_marker:
            print(f"Completed day {t_marker}")
            t_marker +=1

        it += 1
        window.show()




