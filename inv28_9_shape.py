# df = 0.2 nf = 31 Q-like FoM + intensity 分母 npa.sum LD_MMA load cavity a
import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import nlopt
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import os, time

# ===  ===
savedir = "inv28_9data_shape"
data_v = f"{savedir}/data/changev"
data_w = f"{savedir}/data/weight"
dir_path  = f"{savedir}/figure"
file_dir  = f"{savedir}/change"
os.makedirs(data_v, exist_ok=True)
os.makedirs(data_w, exist_ok=True)
os.makedirs(dir_path, exist_ok=True)
os.makedirs(file_dir, exist_ok=True)


# Global parameters
mp.verbosity(1)
Air = mp.Medium(index=1)
Si = mp.Medium(index=3.48)
resolution = 250

dx = 2.4  # design_region_x_width = 1    
dy = 0.8  # design_region_y_width = 1   
pml_size = 0.8
air_size = 0.3
Sx = dx + 2*pml_size
Sy = dy + 2*pml_size + 2*air_size
cell_size = mp.Vector3(Sx, Sy)
# ===  ===
Nx = int(dx * resolution) + 1 
Ny = int(dy * resolution) + 1


# === cavity ===
cavity_design = np.load("st_05_8192_179.npy")
cavity_design = cavity_design.reshape((Nx, Ny))

# === Mapping parameter ===
minimum_length = 0.01
eta_i = 0.5
eta_e = 0.55
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
pml_layers = [mp.PML(thickness=pml_size)]

# ===  ===
wavelength = 1.55
fcen = 1 / 1.55
fwidth = 0.2 
nfreq = 51
freq = np.linspace(fcen - fwidth/2,fcen + fwidth/2, nfreq)
source = [mp.Source(mp.GaussianSource(fcen, fwidth=fwidth), component=mp.Ey,
                    center=mp.Vector3(-Sx/2+pml_size+0.8, 0), size=mp.Vector3(0, dy + 2 * air_size))]

# === ===
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), Air, Si, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(dx, dy))
)

# ===  ===
geometry = [
    mp.Block(center=mp.Vector3(), size=mp.Vector3(Sx,dy), material=Si),
    mp.Block(center=design_region.center, size=design_region.size, material=design_variables)
    ]
sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air,
    extra_materials=[Si],
    resolution=resolution,    
)



# === FOM ===
def J(fields1):
    """
    Q-like FOM = log(I_center / mean(I_side))
    """
    # epsilon = 1e-12
    field1 = npa.abs(fields1) ** 2                 # shape: (nf, Nx, Ny)
    intensity = npa.mean(field1, axis=(1, 2))      # shape: (nf,)
    center_idx = len(intensity) // 2               # fcen index
    I_center = intensity[center_idx]               # scalar

    # I_side → scalar
    I_side = npa.sum(npa.concatenate([intensity[:center_idx], intensity[center_idx + 1:]]))

    return npa.log(I_center / (I_side )) + npa.log(I_center)

monitor_vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0.01,0.01))
FourierFields = mpa.FourierFields(sim, monitor_vol, mp.Ey, yee_grid=True)

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J],
    objective_arguments=[FourierFields],
    design_regions=[design_region],
    frequencies=freq,
    decimation_factor = 1,
    maximum_run_time=200,
)

# Saving the structure before optimization
def safe_plot2D(filename, output_plane):
    plt.figure(figsize=(5, 4))
    opt.plot2D(True, output_plane=output_plane)
    plt.title('Optimization Setting')
    plt.tight_layout()
    plt.savefig(filename, transparent=True)
    plt.close()
    print(f"plot2D structure of setting !")

if mp.am_master():
    safe_plot2D(f'{dir_path}/plot_structure.png', 
                mp.Volume(center=design_region.center, size=mp.Vector3(Sx, Sy, 0)))

# === Mapping Function ===
def mapping(x, eta, beta ):
    x2d = x.reshape((Nx, Ny))
    filtered = mpa.conic_filter(x2d, filter_radius, dx, dy, resolution)
        # Only used the smoothed projection if prompted.
    # if use_smoothed_projection:
    projected = mpa.smoothed_projection(filtered, beta=beta, eta=eta, resolution=resolution)
    # else:
    #     projected = mpa.tanh_projection(x, beta=beta, eta=eta)
    final1 = (npa.flipud(projected) + projected) / 2  # left-right symmetry
    final = (npa.fliplr(final1) + final1) / 2  # up-down symmetry
    return final.flatten()

# # === Gradient Mask ===
# mask = np.ones((Nx, Ny))
# mask[Nxc:Nxc + cNx, :] = 0
# freeze_mask = mask.flatten()

# def masked_tensor_jacobian_product(mapping_func, freeze_mask):
#     def wrapped(x, eta, beta,cavity,vec):
#         grad = tensor_jacobian_product(mapping_func, 0)(x, eta, beta,cavity,vec)
#         return grad * freeze_mask
#     return wrapped

# === optimization main code ===
n = Nx * Ny
x = np.ones((n,)) * 0.5
x = x.reshape(Nx,Ny)
x= cavity_design
x = x.flatten()
# algorithm = nlopt.LD_MMA
algorithm = nlopt.LD_CCSAQ
lb = np.zeros((n,))
ub = np.ones((n,))
cur_beta = 4
beta_scale = 2
update_factor = 20
num_betas = 12
ftol = 1e-5
evaluation_history = []
grad_history = []
cur_iter = [0]

def f(v, gradient, beta):
    print(f"Iteration {cur_iter[0]+1:03d}")
    weights = mapping(v, eta_i, beta)
    fval, dJ_du = opt([weights])

    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(v, eta_i, beta,npa.sum(dJ_du,axis=1)) / nfreq
        grad_history.append(gradient.copy)
    evaluation_history.append(np.real(fval))
    np.save(f"{data_v}/st_05_{beta}_{cur_iter[0]+1:03d}.npy", v)
    np.save(f"{data_w}/st_05_{beta}_{cur_iter[0]+1:03d}.npy", weights)

    if mp.am_master():
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        opt.update_design([weights])
        opt.plot2D(False, ax=ax, output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(dx, dy)))
        ax.axis("off")
        plt.savefig(f"{file_dir}/s_change{cur_iter[0]+1:03d}.png", transparent=True)
        plt.close()

    cur_iter[0] += 1
    return np.real(fval)

# === start  opt ===
start = time.time()
for _ in range(num_betas):
    solver = nlopt.opt(algorithm, n )
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
    solver.set_maxeval(update_factor)
    solver.set_ftol_rel(ftol)
    x[:] = solver.optimize(x)
    cur_beta *= beta_scale
end = time.time()

print(f"Optimization finished in {end - start:.2f}s")

# ===  ===
final_design = mapping(x, eta_i, cur_beta/beta_scale)
# final_design = (np.sign(final_design - 0.5) + 1) / 2

np.save(f"{savedir}/data/final_design.npy", final_design)
np.save(f"{savedir}/data/eval_history.npy", np.array(evaluation_history))
np.save(f"{savedir}/data/grad_history.npy", np.array(grad_history))
# === ===
def plot_final():
    opt.update_design([final_design])
    plt.figure()
    opt.plot2D(False, output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(dx, dy)))
    plt.savefig(f"{dir_path}/final_design.png")
    plt.close()

def plot_history():
    plt.figure()
    for i in range(1, len(evaluation_history) // update_factor + 1):
        plt.axvline(x=i * update_factor, color='purple', linestyle='--')
    plt.plot(evaluation_history, "o-")
    plt.xlabel("Iteration")
    plt.ylabel("FOM")
    plt.grid(True)
    plt.savefig(f"{dir_path}/fom_history.png")
    plt.close()

if mp.am_master():
    plot_final()
    plot_history()
