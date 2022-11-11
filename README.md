# Single-Life Reinforcement Learning

This code supplements the following paper: 

> [You Only Live Once: Single-Life Reinforcement Learning](https://arxiv.org/pdf/2210.08863.pdf)


## Instructions

Install [MuJoCo](http://www.mujoco.org/) from [here](https://github.com/openai/mujoco-py).

Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```
Install dependencies:
```sh
conda env create -f conda_env.yml
conda activate slrl
```


## Sample commands

As a starting point, the environments are provided in `envs/` with the corresponding prior data and pretrained models (including Q functions) for the Pointmass and Cheetah environments in `data/`.

### Run Q-Weighted Adversarial Learning (QWALE) in the Pointmass environment
```
python train.py use_discrim=True rl_pretraining=True q_weights=True online_steps=200000 env_name=pointmass 
```
### Run SAC fine-tuning in the Pointmass environment
```
python train.py use_discrim=False rl_pretraining=True q_weights=False online_steps=200000 env_name=pointmass
```

Note that due to the randomness in the distribution shift of single-life trials, runs may have a large variance, so running many seeds is often needed to evaluate a method. 


## Citation

```
@article{chen2022you,
  title={You Only Live Once: Single-Life Reinforcement Learning},
  author={Chen, Annie S and Sharma, Archit and Levine, Sergey and Finn, Chelsea},
  journal={Neural Information Processing Systems},
  year={2022}
}
```
