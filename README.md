# CSC412NormalizingFlows
## Setup
```
git clone --recurse-submodules https://github.com/levmckinney/NormalizingFlowPolicies.git
conda create --name cells python=3.8
conda activate flows
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -e .
pip install -e pybullet-gym
```
## Run experiments
```
python scripts/train.py --config configs/flow.yml  --name my_experiment --samples 3 --logdir ray_results
```