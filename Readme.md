# WAM Envs

![WAMWipe gif](/gifs/wamwipe.gif?raw=true "WAMWipe-v1")


You should build a virtual env from scratch for this project: Make sure you use python 3.7.

**Activate your environment, then:**

```
pip install tensorflow-cpu 1.15

pip install mujoco-py
```

**install baselines:**
```
git clone https://github.com/openai/baselines.git

cd baselines

pip install -e .
```
**downgrade protobuf:**
```
pip install protobuf==3.20.*

pip install mpi4py
```
**downgrade h5py:**
```
pip install h5py==2.10.0
```
Now you have all the dependencies...

**Install wam_envs in your virtual env:**

cd into the cloned wam_envs directory and run `pip install -e .`

# Usage
See openai [baselines github page](https://github.com/openai/baselines) for more info on baselines usage. 

Some examples using WAMWipe can be found below.

Watch an untrained, 7dof joint velocity agent in WAMWipe-v1:

`mpirun -np 1 python -m wam_envs.run_baselines --num_env=1 --alg=her --env=WAMWipe-v1 --num_timesteps=0 --play --reward_type=sparse --replay_strategy=none`

Watch an untrained, 2dof NHT agent in NHT_WAMWipe-v1 (note, you will need to install NHT as an additional dependency for this, not yet available):

`mpirun -np 1 python -m wam_envs.run_baselines --num_env=1 --alg=her --env=NHT_WAMWipe-v1 --num_timesteps=0 --play --reward_type=sparse --replay_strategy=none --action_dim=3 --NHT_path=/home/kerrick/uAlberta/projects/NHT/map_model/SCL`