# TPS

This repository contains the following code:

- **Main code**
  - `inv28_9shape.py`: Performs adjoint-based **shape optimization** to design the target structure.
  - `cav.py`: Generates the initial structure used to start the shape optimization.
  - `st_05_8192_179.npy` is the initial structure produced by `cav.py`.
  - **Input data (initial structure)**
    - When running `cav.py`, this file  `st_05_8192_179.npy` is saved to:
    - `cavdata/data/changev` directiory.
  ### Notes
  - To run `cav.py` and `inv28_9shape.py`, first create the Conda environment from `pmp.yml`, then activate it before executing either script:

        conda env create -f pmp.yaml
  - The commned to run `cav.py` and `inv28_9shape.py`:

        nohup mpirun -np 40 python cav.py &
    or 

        nohup mpirun -np 40 python inv28_9shape.py &

- **Testing code**
  - `dftscrp.py`: Simulates the designed DFT field.
  - `harminv.py`: Extracts the \(Q\)-value (quality factor).

- **Plot enhancement**
    - In `inv28_9data_shape/post-sim-test2`, the jupter notebook `t.ipynb` is use the data generated  from `dftscrp.py` code.

```mermaid
flowchart LR
  A[cav.py] -->|Generate| B[st_05_8192_179.npy]
  B -->|Initialize| C[Run inv28_9shape.py]
  C -->|Output| D[Final design]
  D -->|Evaluate| E[Testing code]
