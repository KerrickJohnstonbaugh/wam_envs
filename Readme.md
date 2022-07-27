#WAM Envs

![Alt text](/gifs/wamwipe.jpg?raw=true "Optional Title")


You should build a virtual env from scratch for this project:
Make sure you use python 3.7

Activate your environment
pip install tensorflow-cpu 1.15
pip install mujoco-py

install baselines:
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

downgrade protobuf: 
pip install protobuf==3.20.*

pip install mpi4py

downgrade h5py:
pip install h5py==2.10.0


Now you have all the dependencies...
cd into the cloned wam_envs directory and run "pip install -e ."
