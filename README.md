# Lifelike Agility and Play in Quadrupedal Robots
This repository contains all the source codes and dataset for "Lifelike Agility and Play in Quadrupedal Robots using Reinforcement Learning and Generative Pre-trained Models", published in Nature Machine Intelligence, 2024 ([link](https://www.nature.com/articles/s42256-024-00861-3), [arxiv](https://arxiv.org/abs/2308.15143)). Please refer to the official [project page](https://tencent-roboticsx.github.io/lifelike-agility-and-play/) for a general introduction of this work. 

## Setup
To run the codes, you will need Python3.6 or Python3.7 to install tensorflow 1.15.0, and the [TLeague](https://github.com/tencent-ailab/tleague) and [TPolicies](https://github.com/tencent-ailab/TPolicies) repositories, which are developed for distributed multi-agent RL.
For more details, please refer to the TLeague [paper](https://arxiv.org/abs/2011.12895) or its [project page](https://github.com/tencent-ailab/tleague_projpage).

Please follow these steps to setup the environments:

```sh
git clone https://github.com/tencent-ailab/TLeague.git
git clone https://github.com/tencent-ailab/TPolicies.git
cd TLeague
pip install -e .
cd ..
cd TPolicies
pip install -e .
cd ..
cd lifelike
pip install -e .
```

To test the simulation scenarios, you can simply try the following scripts.

PMC for tracking tasks:

```sh
python test_scripts/pritimitive_level/test_primitive_level_env.py
```

EPMC for traversing tasks:

```sh
python test_scripts/environmental_level/test_environmental_level_env.py
```

SEPMC for the Chase Tag Game:

```sh
python test_scripts/strategic_level/test_strategic_level_env.py
```

The training scripts are provided in .sh files in the `train_scripts' folder. The TLeague training pipeline goes with four modules: model_pool, league_mgr, learner and actor, each of which should be run in an independent terminal. The model_pool holds all the trained or training models, the league_mgr manages the learner and actor tasks, the learner optimizes the current model and the actor runs the agent-environment interaction and generates data. Please refer to the TLeague [paper](https://arxiv.org/abs/2011.12895) for more details of these modules.

To train PMC:

```sh
cd train_scripts
```

Open Terminal 1 and run
```sh
bash example_pmc_train.sh model_pool
```

Open Terminal 2 and run
```sh
bash example_pmc_train.sh league_mgr
```

Open Terminal 3 and run
```sh
bash example_pmc_train.sh learner
```

Open Terminal 4 and run
```sh
bash example_pmc_train.sh actor
```

To train EPMC and SEPMC, you can simply follow the steps of PMC and just replace `example_pmc_train.sh' with 'example_epmc_train.sh' and 'example_sepmc_train.sh'. Note that you can launch multiple actors in a distributed manner to fast generate data samples.

## Motion Capture Data
The motion capture data is obtained from a medium-sized Labrador Retriever. The motions include walking, running, jumping, playing, and sitting. The original data is located in `data/raw_mocap_data`. For tracking with a quadrupedal robot, we retargeted the data and generated a mirrored version, which are located in `data/mocap_data`.

## Citation

If you find the codes and dataset in this repo useful for your research, please cite the paper:
```
@article{han2024lifelike,
      title={Lifelike Agility and Play in Quadrupedal Robots using Reinforcement Learning and Generative Pre-trained Models}, 
      author={Lei Han and Qingxu Zhu and Jiapeng Sheng and Chong Zhang and Tingguang Li and Yizheng Zhang and He Zhang and Yuzhen Liu and Cheng Zhou and Rui Zhao and Jie Li and Yufeng Zhang and Rui Wang and Wanchao Chi and Xiong Li and Yonghui Zhu and Lingzhu Xiang and Xiao Teng and Zhengyou Zhang},
      year={2024},
      journal={Nature Machine Intelligence},
      publisher={Nature Publishing Group UK London},
      volume={7},
      doi = {10.1038/s42256-024-00861-3},
      url = {https://www.nature.com/articles/s42256-024-00861-3},
}

```
## Disclaimer
 
This is not an officially supported Tencent product. The code and data in this repository are for research purpose only. No representation or warranty whatsoever, expressed or implied, is made as to its accuracy, reliability or completeness. We assume no liability and are not responsible for any misuse or damage caused by the code and data. Your use of the code and data are subject to applicable laws and your use of them is at your own risk.
