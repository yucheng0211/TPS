# %%
import numpy as np 
import meep as mp 
import matplotlib.pyplot as plt
import meep.adjoint as mpa
import os
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%
dir_path = 'post-sim-test3'
os.makedirs(dir_path, exist_ok=True)

# %%
design = np.load("data/final_design.npy")


# %%
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
# === 設計區尺寸與 cavity 區 ===
Nx = int(dx * resolution) + 1 
Ny = int(dy * resolution) + 1

design_2d = design.reshape(Nx,Ny)

# === Mapping 參數 ===
minimum_length = 0.02
eta_i = 0.5
eta_e = 0.55
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
pml_layers = [mp.PML(thickness=pml_size)]

# %%
# Design frequency
wavelengths = np.array([1.55])
frequencies = np.array([1 / 1.55])
fcen = 1. / 1.55
width = 0.2
fwidth = width * fcen
source_center = mp.Vector3(-Sx/2+pml_size+0.8,0) 
src_size = dy 
source_size = mp.Vector3(0,src_size) 
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth,is_integrated=True)
source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
pml_layers = [mp.PML(thickness=pml_size)]

# %%

design_variables = mp.MaterialGrid(
mp.Vector3(Nx, Ny), 
medium1=Air,
medium2=Si,
weights=design_2d,
grid_type="U_MEAN"
)

design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(0,0,0),
        size=mp.Vector3(dx, dy,0),
    ),
)
geometry = [ 
    mp.Block(center=mp.Vector3(), size=mp.Vector3(Sx,dy), material=Si),    
]

sim0 = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air,
    resolution=resolution,
    extra_materials=[Si],
)
if mp.am_master():
    plt.figure(figsize=(5,4))
    ax = plt.gca()
    sim0.plot2D(
        ax=ax,
        plot_sources_flag=True,
        plot_monitors_flag=True,
        plot_boundaries_flag=True,
    )
    plt.tight_layout()
    plt.savefig(f'{dir_path}/geo.png',transparent=True)
    plt.close()

# %%
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
sim0.change_sources(source)

dft_fields = sim0.add_dft_fields([mp.Dy,mp.Ey],
                                fcen,0,1,
                                center=mp.Vector3(),
                                size=mp.Vector3(dx,dy,0),
                                )
sim0.run(until=100)
Ey0 = sim0.get_dft_array(dft_fields, mp.Ey, 0)
amplitude_0 = 10 * np.log10(np.abs(Ey0) ** 2 )  # log-scale intensity
amplitude2_0 = np.abs(Ey0) ** 2
[x, y, z, w] = sim0.get_array_metadata(dft_cell=dft_fields)
np.savez(f"{dir_path}/fields_ref.npz",
         Ey0=Ey0,
         amplitude2_0=amplitude2_0, # |ey_0|^2
         x=x,
         y=y,
         fcen=fcen,
         wavelength=1.55)

# %%
plt.title(r'DFT spectrum for $\lambda$ = 1.55 $\mu$m')
plt.pcolormesh(
    x,
    y,
    np.transpose(amplitude2_0),
    cmap="jet",
    shading="gouraud",
    vmin=0,
    vmax=np.max(amplitude2_0),
)
plt.gca().set_aspect("equal")
plt.xlabel("x (μm)")
plt.ylabel("y (μm)")

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cax=cax, label='enhancement')
plt.tight_layout()
plt.savefig(f'{dir_path}/dft_rel0.png',transparent = True)
plt.show()
plt.close()

# %%
geometry = [
    mp.Block(center=mp.Vector3(), size=mp.Vector3(Sx, dy), material=Si),
    mp.Block(center=mp.Vector3(), size=design_region.size, material=design_variables),
]

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air,
    resolution=resolution,
    extra_materials=[Si],
)

if mp.am_master():
    plt.figure(figsize=(5,4))
    ax = plt.gca()
    sim.plot2D(
        ax=ax,
        plot_sources_flag=True,
        plot_monitors_flag=True,
        plot_boundaries_flag=True,
    )
    plt.tight_layout()
    plt.savefig(f'{dir_path}/geowst.png',transparent=True)
    plt.close()

# %%
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
sim.change_sources(source)
dft_fields = sim.add_dft_fields([mp.Dy,mp.Ey],
                                fcen,0,1,
                                center=mp.Vector3(),
                                size=mp.Vector3(dx,dy,0),
                                )
sim.run(until_after_sources=100)
Ey = sim.get_dft_array(dft_fields, mp.Ey, 0)
amplitude = 10 * np.log10(np.abs(Ey) ** 2 )  # log-scale intensity
amplitude2 = np.abs(Ey) ** 2
[x, y, z, w] = sim.get_array_metadata(dft_cell=dft_fields)
np.savez(f"{dir_path}/fields_struct.npz",
         Ey=Ey,
         amplitude2=amplitude2,
         x=x,
         y=y,
         fcen=fcen,
         wavelength=1.55)

# %%
plt.title('DFT spectrum for $\lambda$ = 1.55 $\mu$m')
plt.pcolormesh(
    x,
    y,
    np.transpose(amplitude),
    cmap="jet",
    shading="gouraud",
    vmin=0,
    vmax=np.max(amplitude),
)

plt.gca().set_aspect("equal")
plt.xlabel("x (μm)")
plt.ylabel("y (μm)")

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cax=cax, label='intensity (db)')
plt.tight_layout()
plt.savefig(f'{dir_path}/dft_struct_dB.png',transparent = True)
plt.show()
plt.close()
#%%

plt.pcolormesh(
    x,
    y,
    np.transpose(amplitude2),
    cmap="jet",
    shading="gouraud",
    vmin=0,
    vmax=np.max(amplitude2),
)
plt.title(r'DFT spectrum for $\lambda$ = 1.55 $\mu$m')
plt.gca().set_aspect("equal")
plt.xlabel("x (μm)")
plt.ylabel("y (μm)")

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cax=cax, label='intensity')
plt.tight_layout()
plt.savefig(f'{dir_path}/dft_struct.png',transparent = True)
plt.show()
plt.close()
# plot enhancement
rel = amplitude2/np.mean(amplitude2_0)
fig, ax = plt.subplots()

# --- plot the field intensity ---
pcm = ax.pcolormesh(
    x,
    y,
    np.transpose(rel),
    cmap="jet",
    shading="gouraud",
    vmin=0,
    vmax=np.max(rel),
)

# --- title & labels ---
ax.set_title(r'DFT spectrum for $\lambda$ = 1.55 $\mu$m')
ax.set_aspect("equal")
ax.set_xlabel("x (μm)", fontsize=13)
ax.set_ylabel("y (μm)", fontsize=13)

# --- colorbar setup ---
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(pcm, cax=cax)                   # ✅ create colorbar from the plotted data
cbar.set_label('intensity enhancement', fontsize=11) # ✅ set label font size correctly
cbar.ax.tick_params(labelsize=10)                   # ✅ control tick font size

plt.tight_layout()
plt.savefig(f'{dir_path}/dft_rel.png', transparent=True, dpi=200)
plt.show()
plt.close()







# %%
