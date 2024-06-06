# ODQM

---
Estimating data quality is crucial problem in Offline Reinforcement Learning. Unlike in Online RL, data, collected from
some agent has a strong impact on agent final performance.
**ODQM** or __Offline Data Quality Measurement__ provides methods for estimating provided data quallity 


---
## Setup
If you want to use D4RL offline envs, install it following instructions from [official repository](https://github.com/Farama-Foundation/D4RL)

First, you needed to clone this repository:

```commandline
git clone https://github.com/zen0no/odqm
```

Then install requirements:
```commandline
pip install -r requirements.txt
```

---
## Usage

To estimate your data, firstly you need to create directory structure like in `sample_data`
```bash
sample_data
├── actions.npy
├── dones.npy
├── rewards.npy
└── states.npy


```

Next, replace data name from `config/sample_config.yaml` to your data directory and run:
```commandline
python3 train.py config/sample.yaml
```

Optionally, you can write your own yaml config and provide path to it as an argument.