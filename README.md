# TPS

This repository contains the following code:

- **Main code**
  - `inv28_9shape.py`: Uses shape optimization (adjoint) to deisgn our target structure.
  - The data `st_05_8192_179.npy` is the initial sturucte that use adjoint opt to designed strucutre. 

- **Testing code**
  - `dftscrp.py`: Simulates the designed DFT field.
  - `harminv.py`: Extracts the \(Q\)-value (quality factor).

- **Plot enhancement**
    - In `inv28_9data_shape/post-sim-test2`, the jupter notebook `t.ipynb` is use the data generated  from `dftscrp.py` code.