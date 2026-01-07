
# Reinforcement Learning–Based Performance Anomaly Detection

This project implements a reinforcement learning (RL) approach to detect, predict, and mitigate performance degradation in physical servers.

The system:
- Learns from real-time performance metrics (CPU, memory, latency).
- Detects anomalies proactively.
- Uses PyTorch to train an RL agent that interacts with a custom server environment.

## Dataset

This project uses CPU utilization traces from PlanetLab (CloudSim) to train and evaluate the anomaly detection models.

#### Important Note
The dataset is **not included in this repository** due to licensing restrictions. You must download it separately to run the experiments.

#### Downloading the Data

1. Visit the official PlanetLab trace repository:  
   [https://github.com/beloglazov/planetlab-workload-traces](https://github.com/beloglazov/planetlab-workload-traces)

2. Download the `.txt` files corresponding to the dates used in this project.

3. Place all downloaded `.txt` CPU trace files into the following folder in this repository:

```text
data/planetlab/
```

After this, all training and evaluation scripts will automatically load the CPU traces from `data/planetlab/`.

## RL Problem Definition
Each agent learns to classify whether the next time window is likely to show performance degradation based on CPU trends of one server.

At each timestep t, State (observation) is an sliding window of CPU usage values, ex. last 12 intervals = last 1 hour
→ state = [cpu[t-11], ..., cpu[t]]

Since this is anomaly detection, action space includes:

```text
0 = normal
1 = anomaly (performance degradation predicted)
```

Reward function is defined as follows:

```text
If agent predicts anomaly correctly → +1

If wrong → -1

If predicts normal correctly → +0.5

False alarm → -0.5
```

## Project Structure
```text
performance-anomaly-detection/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── src/
│ ├── environment.py # Custom RL environment for server performance
│ ├── agent.py # PyTorch RL model
│ └── train.py # Training loop
│
├── data/
└── notebooks/

```

## Installing dependencies

```text
pip install -r requirements.txt
```

## Running the Project

Train the RL agent:

```text
python src/train.py
```

Modify configuration inside `train.py` as needed.


## References
- PlanetLab workload traces: [https://github.com/beloglazov/planetlab-workload-traces](https://github.com/beloglazov/planetlab-workload-traces)

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.
