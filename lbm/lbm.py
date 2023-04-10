import taichi as ti
import numpy as np

"""
Lattice Directions:
8 1 2
7 0 3
6 5 4
"""

c_D2Q9 = ti.Matrix([
    [ 0,  0], # null
    [ 0,  1], # N
    [ 1,  1], # NE
    [ 1,  0], # E
    [ 1, -1], # SE
    [ 0, -1], # S
    [-1, -1], # SW
    [-1,  0], # W
    [-1,  1], # NW
])

w_D2Q9 = ti.Vector([
    4/9,  # null
    1/9,  # N
    1/36, # NE
    1/9,  # E
    1/36, # SE
    1/9,  # S
    1/36, # SW
    1/9,  # W
    1/36, # NW
])

"""Bounce Back Indices"""
ix_bb_D2Q9 = ti.Vector([
    0, # null -> null
    5, # N -> S
    6, # NE -> SW
    7, # E -> W
    8, # SE -> NW
    1, # S -> N
    2, # SW -> NE
    3, # W -> E
    4, # NW -> SE
])



@ti.data_oriented
class LBM_D2Q9_iso:
    def __init__(self, nx, ny, rho0, tau, updates_per_batch=100, im_scale=3, colormap=[[255,0,0],[0,0,255]]) -> None:
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.tauinv = 1.0/tau
        self.mtauinv = -1.0/tau
        self.rho0 = rho0
        self.updates_per_batch = updates_per_batch
        self.im_scale = im_scale

        """
        Lattice Directions:
        8 1 2
        7 0 3
        6 5 4
        self.c: discretized velocities
        self.c2: discretized velocities^2
        self.cxy: discretized velocities component products
        self.w: discretized velocity weights
        self.ix_bb: bounce back indices
        """
        self.c   = ti.Vector.field(2, shape=(9,), dtype=float)
        self.c2  = ti.Vector.field(2, shape=(9,), dtype=float)
        self.cxy = ti.field(shape=(9,), dtype=float)
        self.w = ti.field(shape=(9,), dtype=float)
        self.ix_bb = ti.field(shape=(9,), dtype=int)

        for i in range(self.c.shape[0]):
            self.c[i]     = ti.Vector([ c_D2Q9[i,0],   c_D2Q9[i,1] ])
            self.c2[i]    = ti.Vector([ c_D2Q9[i,0]**2,  c_D2Q9[i,1]**2 ])
            self.cxy[i]   = c_D2Q9[i,0] * c_D2Q9[i,1]
            self.w[i]     = w_D2Q9[i]
            self.ix_bb[i] = ix_bb_D2Q9[i]


        """The lattice"""
        self.f       = ti.field(shape=(self.nx, self.ny, 9), dtype=float)
        self.f_tmp   = ti.field(shape=(self.nx, self.ny, 9), dtype=float)

        """The density field"""
        self.rho     = ti.field(shape=(self.nx, self.ny),    dtype=float)
        
        """The velocity field (and storage for other oft computed values)"""
        self.u       = ti.Vector.field(2, shape=(self.nx, self.ny), dtype=float) # ux, uy
        self.u_cache = ti.Vector.field(4, shape=(self.nx, self.ny),   dtype=float) # ux2, uy2, ux2+uy2, # 1 - 1.5*(ux^2+uy2), ux*uy

        """The solid boundary field"""
        self.b_s     = ti.field(shape=(self.nx, self.ny),    dtype=ti.i8)
        # TODO: make list of non-solid indices to prevent allocating compute within solids, i.e. to avoid if self.b_s[x,y] == 0: gates

        """The canvas color field"""
        self.image = ti.Vector.field(3, shape=(self.nx, self.ny*2), dtype=float)

        """Rendering""" 
        self.window = ti.ui.Window('LBM Isothermal', res = (self.nx*self.im_scale, 2*self.ny*self.im_scale), pos = (150, 150))
        self.canvas = self.window.get_canvas()
        self.colormap_field = ti.Vector.field(3, shape=(len(colormap)),  dtype=ti.f32,  )
        for i,color in enumerate(colormap):
            self.colormap_field[i] = ti.Vector([color[0], color[1], color[2]])

    @ti.kernel
    def init_solid_boundaries(self):
        self.b_s.fill(ti.cast(0,ti.i8))
        for x,y in self.b_s:
            """Add a cylinder"""
            # if (x-self.nx*0.25)**2 + (y-self.ny*0.25)**2 < (self.ny*0.125)**2:
            #         self.b_s[x,y] = ti.cast(1, ti.i8)
            # if (x-self.nx*0.25)**2 + (y-self.ny*0.75)**2 < (self.ny*0.125)**2:
            #         self.b_s[x,y] = ti.cast(1, ti.i8)
            if (x-self.nx*0.25)**2 + (y-self.ny*0.5)**2 < (self.ny*0.0625)**2:
                    self.b_s[x,y] = ti.cast(1, ti.i8)
            # """Add a cube"""
            # if x > self.nx*0.33-10 and x < self.nx*0.333+10 and y > self.ny * 0.66-10 and y < self.ny*0.66+10:
            #     self.b_s[x,y] = ti.cast(1,ti.i8)
            # """Add a cube"""
            # if x > self.nx*0.33-10 and x < self.nx*0.333+10 and y > self.ny * 0.33-10 and y < self.ny*0.33+10:
            #     self.b_s[x,y] = ti.cast(1,ti.i8)

    @ti.kernel
    def init_f(self):
        """Create a velocity density field"""
        self.f.fill(1) # Arbitrary but equal starting densities
        for x,y,i in self.f:
            if self.b_s[x,y] == 0:
                self.f[x,y,i] += ti.randn()*0.01 # random initial perturbation
                if i == 3: # east flow
                    # x_norm = x / (self.nx - 1.0)
                    # self.f[x,y,i] += 2.0 * (1.0 + 0.2 * ti.cos(2.0*np.pi*x_norm*4.0))
                    self.f[x,y,i] = 2.3
        
        """Accumulate the densities per lattice node"""
        self.rho.fill(0)
        for x,y,i in self.f:
            self.rho[x,y] += self.f[x,y,i]
        
        """Normalize the densities to rho0 everywhere"""
        for x,y,i in self.f:
            self.f[x,y,i] *= self.rho0 / self.rho[x,y] 
    
    @ti.kernel
    def compute_macro(self):
        """Accumulate rho and u"""
        self.rho.fill(0)
        self.u.fill(0)
        for x,y,i in self.f:
            self.rho[x,y] += self.f[x,y,i]
            self.u[x,y].x += self.c[i].x * self.f[x,y,i] * (1.0-self.b_s[x,y]) # zeros out the the solid nodes
            self.u[x,y].y += self.c[i].y * self.f[x,y,i] * (1.0-self.b_s[x,y]) # zeros out the solid nodes
        
        """Normalize u by density and compute useful values"""
        for x,y in self.u:
            self.u[x,y].x /= self.rho[x,y]
            self.u[x,y].y /= self.rho[x,y]
            self.u_cache[x,y][0] = self.u[x,y].x ** 2
            self.u_cache[x,y][1] = self.u[x,y].y ** 2
            self.u_cache[x,y][2] = self.u[x,y].x * self.u[x,y].y
            self.u_cache[x,y][3] = self.u_cache[x,y][0] + self.u_cache[x,y][1]

    @ti.kernel
    def collide(self):
        for x,y,i in self.f:
            """Compute equilibrium"""
            f_eq = self.rho[x,y] * self.w[i] * \
                (
                    1.0 + \
                    3.0 * (self.c[i].x * self.u[x,y].x + self.c[i].y * self.u[x,y].y) + \
                    9.0/2.0 * (
                        self.c2[i].x    * self.u_cache[x,y][0] + \
                        2.0 * self.cxy[i] * self.u_cache[x,y][2] + \
                        self.c2[i].y    * self.u_cache[x,y][1]
                    ) - \
                    3.0/2.0 * self.u_cache[x,y][3]
                )
            """Relax"""
            self.f[x,y,i] += self.mtauinv * (self.f[x,y,i] - f_eq)
    
    @ti.kernel
    def propagate(self):
        self.f_tmp.fill(0)
        # """special? edges"""
        # for y in range(self.ny):
        #     self.f[0, y, 2] = self.f[1, y, 2]
        #     self.f[0, y, 3] = self.f[1, y, 3]
        #     self.f[0, y, 4] = self.f[1, y, 4]
        #     self.f[self.nx-1, y, 6] = self.f[self.nx-2, y, 6]
        #     self.f[self.nx-1, y, 7] = self.f[self.nx-2, y, 7]
        #     self.f[self.nx-1, y, 8] = self.f[self.nx-2, y, 8]

        for x,y,i in self.f:
            if self.b_s[x,y] == 0:

                """
                Find the destination node for the current velocity distribution
                -> modulus implies periodic BCs
                """
                x_idx = int(( x + self.c[i].x + self.nx ) % self.nx)
                y_idx = int(( y + self.c[i].y + self.ny ) % self.ny)
                if self.b_s[x_idx, y_idx] == 0:
                    """Propagate"""
                    self.f_tmp[x_idx, y_idx, i] += self.f[x,y,i] # copy the particles at x,y with velocity c_i to the node pointed at by the vel vector.
                else:
                    """Bounce Back"""
                    self.f_tmp[x, y, self.ix_bb[i]] += self.f[x,y,i]
            
        # TODO: consider using f_tmp in macro and collide to prevent needing copy-back step
        for x,y,i in self.f_tmp:
            self.f[x,y,i] = self.f_tmp[x,y,i]
    
    def step(self):
        self.compute_macro()
        self.collide()
        self.propagate()

    def batch_step(self):
        for _ in range(self.updates_per_batch):
            self.step()
    

    @ti.kernel
    def update_im(self):
        for x,y in self.u_cache:
            if self.b_s[x,y % self.ny] == 0:
                """Velocity"""
                h = ti.sqrt(self.u_cache[x,y][3])*7.5
                
                level = ti.max(ti.min(ti.floor(h*(self.colormap_field.shape[0]-1)),self.colormap_field.shape[0]-2), 0)
                colorphase = ti.cast(ti.min(ti.max(h*(self.colormap_field.shape[0]-1) - level, 0),1), ti.f32)
                level_idx = ti.cast(level, dtype=int)

                self.image[x,y].r = ti.cast((self.colormap_field[level_idx].x * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].x)/255, ti.float32)
                self.image[x,y].g = ti.cast((self.colormap_field[level_idx].y * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].y)/255, ti.float32)
                self.image[x,y].b = ti.cast((self.colormap_field[level_idx].z * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].z)/255, ti.float32)

                """Vorticity"""
                left  = self.u[ (x - 1 + self.nx) % self.nx, y].y
                right = self.u[ (x + 1 + self.nx) % self.nx, y].y
                down  = self.u[ x, (y - 1 + self.ny) % self.ny].x
                up    = self.u[ x, (y + 1 + self.ny) % self.ny].x
                vorticity = ((right-left)-(up-down))
                h = vorticity*50 + 0.5
                level = ti.max(ti.min(ti.floor(h*(self.colormap_field.shape[0]-1)),self.colormap_field.shape[0]-2), 0)
                colorphase = ti.cast(ti.min(ti.max(h*(self.colormap_field.shape[0]-1) - level, 0),1), ti.f32)
                level_idx = ti.cast(level, dtype=int)

                self.image[x,y+self.ny].r = ti.cast((self.colormap_field[level_idx].x * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].x)/255, ti.float32)
                self.image[x,y+self.ny].g = ti.cast((self.colormap_field[level_idx].y * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].y)/255, ti.float32)
                self.image[x,y+self.ny].b = ti.cast((self.colormap_field[level_idx].z * (1-colorphase) + colorphase*self.colormap_field[level_idx+1].z)/255, ti.float32)
            else:
                self.image[x,y] = ti.Vector([ 0.0, 0.0, 0.0])
                self.image[x,y+self.ny] = ti.Vector([ 0.0, 0.0, 0.0])
    
    def render(self):
        self.canvas.set_image(self.image)
        self.window.show()

if __name__ == '__main__':
    ti.init(arch=ti.cuda, default_fp=ti.f32, debug=True)

    # Simulation parameters
    Nx                = 400 # resolution x-dir
    Ny                = 200 # resolution y-dir
    # Re                = 400
    # Max_U             = 2
    # L                 = 0.15*Ny # cylinder rad
    # v                 = Max_U * L / Re # compute kinematic viscosity from Re
    # tau               = 1.0 / (3.0 * v + 0.5) # compute relaxation according to BGK
    tau               = 0.6
    rho0              = 200 # average density
    updates_per_batch = 20
    im_scale = 4
    colormap = [
        [255,0,0],
        [255,255,255],
        [0,0,255]
    ]


    solver = LBM_D2Q9_iso(Nx, Ny, rho0, tau, updates_per_batch, im_scale, colormap)

    solver.init_solid_boundaries()
    solver.init_f()
    
    import time
    while solver.window.running:
        # time.sleep(0.1)
        solver.batch_step()
        solver.update_im()
        solver.render()


	


