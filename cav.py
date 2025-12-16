import meep as mp
import meep.adjoint as mpa
from meep.materials import Au, Ag 
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import nlopt
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import time
import os

# Setting path
data_v = "cavdata/data/changev"
data_w = "cavdata/data/weight"
dir_path  = "cavdata/figure"
file_dir  = "cavdata/change"
os.makedirs(data_v, exist_ok=True)
os.makedirs(data_w, exist_ok=True)
os.makedirs(dir_path, exist_ok=True)
os.makedirs(file_dir, exist_ok=True)

# Global parameters
mp.verbosity(1)
Air = mp.Medium(index=1)
Si = mp.Medium(index = 3.48)
resolution = 250 
dx = 0.8  #design_region_x_width = 1    
dy = 0.8  #design_region_y_width = 1   
pml_size = 0.8
air_size = 0.3
Sx = dx
Sy = dy + 2*pml_size + 2*air_size
cell_size = mp.Vector3(Sx, Sy)


# Mapping parameters
minimum_length = 0.02
eta_i = 0.5
eta_e = 0.55
eta_d = 1 - eta_e
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
design_region_resolution = int(resolution)
pml_layers = [mp.PML(thickness=pml_size,direction=mp.Y)]

# Design frequency
wavelengths = np.array([1.55])
frequencies = np.array([1 / 1.55])
fcen = 1. / 1.55
width = 0.2
fwidth = width * fcen
source_center = mp.Vector3(-Sx/2,0) 
src_size = dy + 0.1
source_size = mp.Vector3(0,src_size) 
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]

# Design weight
Nx = int(design_region_resolution * dx) +1 
Ny = int(design_region_resolution * dy) +1
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), Air, Si, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(0,0,0),
        size=mp.Vector3(dx, dy,0),
    ),
)

# binarize function 
def mapping(x, eta, beta):

    # filter
    filtered_field = mpa.conic_filter(
        x,
        filter_radius,
        dx,
        dy,
        design_region_resolution,
    )

    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)

    projected_field = (npa.fliplr(projected_field) + projected_field)/ 2  # up-down symmetry   
    
    projected_field = (npa.flipud(projected_field) + projected_field)/ 2  # left-right symmetry
    
    
    # interpolate to actual materials
    return projected_field.flatten()

# FOM 
def J(fields1):
    field1 = npa.sum(npa.abs(fields1) ** 2, axis=(1, 2))
    return npa.mean(field1)
# Setting geometry
geometry = [      
    mp.Block(center=design_region.center, size=design_region.size, material=design_variables),
]
# Setting simultion
kpoint = mp.Vector3(0,0,0)
sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air,
    k_point=kpoint,
    resolution=resolution,
    extra_materials=[Si],
)

# Setting FOM position 
monitor_position = mp.Vector3(0, 0, 0)
monitor_size1 = mp.Vector3(0.02, 0.02,0)
# monitor_size2 = mp.Vector3(0.01, dy, 0)
FourierFields1 = mpa.FourierFields(sim, mp.Volume(center=monitor_position, size=monitor_size1), mp.Ey, yee_grid=True)
# FourierFields2 = mpa.FourierFields(sim, mp.Volume(center=monitor_position, size=monitor_size2), mp.Ey, yee_grid=True)
ob_list = [FourierFields1]
# ob_list = [FourierFields1,FourierFields2]


opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J],
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    decimation_factor = 1,
    maximum_run_time = 100,
)

print("setting is complete!")

# Saving the structure before opt. 
def safe_plot2D(title, filename, output_plane):
    plt.figure(figsize=(5, 5))
    plt.title(title)
    opt.plot2D(True, output_plane=output_plane)
    plt.tight_layout()
    plt.savefig(filename, transparent=True)
    plt.close()
    print(f"plot2D structure of {title} !")

if mp.am_master():
    safe_plot2D('Structure', f'{dir_path}/plot_structure.png', 
            mp.Volume(center=design_region.center, size=mp.Vector3(Sx, Sy, 0)))


# Optimization parameters
algorithm = nlopt.LD_MMA
n = Nx * Ny  
x = np.ones((n,)) * 0.5
lb = np.zeros((n,))
ub = np.ones((n,))
cur_beta = 4
beta_scale = 2
update_factor = 16  # num. iteration per  epoch
num_betas = 12 # total epochs
ftol = 1e-5

evaluation_history = []
cur_iter = [0]

def f(v, gradient, beta):    
    print(f"Current iteration: {cur_iter[0] + 1:03d}")
    weights = mapping(v, eta_i, beta)
    f0, dJ_du = opt([weights])  
    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(v, eta_i, beta, dJ_du)
    evaluation_history.append(np.real(f0))
    np.save(f"{data_v}/st_05_{beta}_{cur_iter[0]+1:03d}.npy", v)
    np.save(f"{data_w}/st_05_{beta}_{cur_iter[0]+1:03d}.npy", weights)
    
    if mp.am_master():
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        opt.plot2D(
        False,
        ax=ax,
        plot_sources_flag=False,
        plot_monitors_flag=False,
        plot_boundaries_flag=False,
          output_plane = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(dx, dy, 0))
        )
        ax.axis("off")
        plt.savefig(f'{file_dir}/s_change{cur_iter[0]+1 :03d}.png',transparent=True)
        # plt.show()
        plt.close()
        
    
    cur_iter[0] += 1
    return np.real(f0)

start_time = time.time()
for iters in range(num_betas):
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
    solver.set_maxeval(update_factor)
    solver.set_ftol_rel(ftol)
    x[:] = solver.optimize(x)
    cur_beta *= beta_scale
end_time = time.time()
print(f"Optimization completed in {end_time - start_time:.4f} seconds")
print("Optimization is complete.")

# Save final data
file_path = "cavdata/data"
final_design = mapping(x, eta_i, cur_beta/beta_scale)
np.save(f"{file_path}/final_design.npy", final_design)
np.save(f"{file_path}/eval_history.npy", np.array(evaluation_history))
print(f"Final results saved to {file_path}")



# Plot final design
def plot_final(design):
    plt.figure()
    ax = plt.gca()
    opt.update_design([design])
    opt.plot2D(
        False,
        ax=ax,
        plot_sources_flag=False,
        plot_monitors_flag=False,
        plot_boundaries_flag=True,
        output_plane = mp.Volume(center=design_region.center, size=mp.Vector3(Sx, Sy, 0))
    )
    plt.savefig(f'{dir_path}/final_st.png',transparent=True)
    plt.close()
    print('plot the final_st.png')



# Function to plot optimization history
def plot_history(evaluation_history):
    plt.figure()
    # Add vertical dashed lines at intervals of num_betas
    for i in range(1, len(evaluation_history)//update_factor+1):
        plt.axvline(x=update_factor * i, color='purple', linestyle='--')
    plt.plot(evaluation_history, "o-")
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("FOM")
    plt.savefig(f'{dir_path}/fom_change.png',transparent=True)
    plt.close()
    print('plot the fom_change.png')

if mp.am_master():
    plot_final(final_design)
    plot_history(evaluation_history)
