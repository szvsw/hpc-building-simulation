import numpy as np
import taichi as ti
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify

from diffusion_eq_2d_exp import Solver

app = Flask(__name__)

# TODO: add dx, dt to config
# TODO: add boundary config
# TODO: add uinit config options
@app.route('/', methods=["GET","POST"])
def index():
    """Extract sim config"""
    data = request.get_json()
    defs = data["defs"]
    map = data["map"]

    mat_ids = np.array(map)
    n = mat_ids.shape[0] 
    m = mat_ids.shape[1]

    try:
        assert n == m, "Currently only supports square domain"
    except:
        return jsonify({"res": "failure"})

    """Build mat def array"""
    mat_defs = np.zeros(shape=(len(defs),3))
    for i,defn in enumerate(defs):
        mat_defs[i,0] = defn["k"]
        mat_defs[i,1] = defn["rho"]
        mat_defs[i,2] = defn["c_p"]
    print("Mat Defs:")
    print(mat_defs)
    mat_diffs = mat_defs[:,0]/(mat_defs[:,1]*mat_defs[:,2])
    print("mat diffs")
    print(mat_diffs)

    ti.init(arch=ti.cuda, default_fp=ti.f32)
    D = np.max(mat_diffs) # [m2/s]
    print(f"max d: {D}")
    dx = 0.01 # [m]
    dt = 1 # [s]
    print(f"Timestep: {int(dt*1000):01d}ms")
    # p = 8
    # n = 2**p

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
        # if it%200 == 0:
        #     solver.plot_isotherms()
        if it*solver.updates_per_batch*solver.dt/(3600*24) > t_marker:
            print(f"Completed day {t_marker}")
            t_marker +=1

        it += 1
        window.show()





    

    return jsonify({"res": "success"})

app.run(host='0.0.0.0', port=5000)