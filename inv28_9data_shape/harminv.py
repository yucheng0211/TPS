# %%
import numpy as np 
import meep as mp 
import matplotlib.pyplot as plt
import meep.adjoint as mpa
import os
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

# %%
dir_path = 'post-sim-test2'
os.makedirs(dir_path, exist_ok=True)
result = np.load("data/final_design.npy")

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

# %%
# Design frequency
wavelengths = np.array([1.55])
frequencies = np.array([1 / 1.55])
fcen = 1. / 1.55
width = 0.2
fwidth = width * fcen
source_center = mp.Vector3(-Sx/2+pml_size,0) 
src_size = dy 
source_size = mp.Vector3(0,src_size) 
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth,is_integrated=True)
source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
pml_layers = [mp.PML(thickness=pml_size)]

# %%
design_2d = result.reshape((Nx, Ny))

print(design_2d.shape)
print(design_2d)
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

# %%
src = mp.GaussianSource(frequency=fcen, fwidth=1)
source = [mp.Source(src, component=mp.Ey, size=mp.Vector3(0,0), center=mp.Vector3(0,0))]
sim.change_sources(source)

if mp.am_master():
    sim.plot2D()
    plt.savefig(f"{dir_path}/final_design.png",transparent =True)
    plt.close()

# %%
harminv_pt = mp.Vector3(0, 0)
h = mp.Harminv(mp.Ey, pt=harminv_pt, fcen=fcen, df=1)

# 執行模擬直到模態衰減
sim.run(mp.after_sources(h), until=2000)

# %%
data = [
    {
        "frequency": mode.freq,
        "decay": mode.decay,
        "Q": mode.Q,
        "abs_amplitude": abs(mode.amp),
        "amplitude": mode.amp,
        "error": mode.err,
    }
    for mode in h.modes  # 過濾低 Q 或雜訊模態
]

df = pd.DataFrame(data)
df.to_csv(f"{dir_path}/harminv.csv", index=False)
print(df)


