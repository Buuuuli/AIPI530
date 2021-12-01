># Offline Deep Reinforcement Learning 

This repository contains the final project of AIPI 530 taht uses the d3rlpy

![](logo.png)

>## Installation

We can directly install the package by the PyPI, Anaconda and Docker
```bash
$ pip install d3rlpy

$ conda install -c conda-forge d3rlpy

$ docker run -it --gpus all --name d3rlpy takuseno/d3rlpy:latest bash
```


>## Getting Started
Offline Instruction

First, we will Setup rendering dependencies for Google Colaboratory.
```bash
!pip install gym pyvirtualdisplay > /dev/null 2>&1
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
```
Then, install the d3rlpy
```bash
!pip install d3rlpy
```

Then, choose a game, here we use the cartpole
```bash
from d3rlpy.datasets import get_cartpole

# get CartPole dataset
dataset, env = get_cartpole()

# we could also choose the following dateset
#from d3rlpy.datasets import get_cartpole # CartPole-v0 dataset
#from d3rlpy.datasets import get_pendulum # Pendulum-v0 dataset
#from d3rlpy.datasets import get_pybullet # PyBullet task datasets
#from d3rlpy.datasets import get_atari    # Atari 2600 task datasets
#from d3rlpy.datasets import get_d4rl     # D4RL datasets

```

Then, setup our RL algorithm
```bash
from d3rlpy.algos import DiscreteCQL
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split

# setup CQL algorithm
cql = DiscreteCQL(use_gpu=False)

# split train and test episodes
train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# start training
cql.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs=1,
        scorers={
            'environment': evaluate_on_environment(env), # evaluate with CartPol-v0 environment
            'advantage': discounted_sum_of_advantage_scorer, # smaller is better
            'td_error': td_error_scorer, # smaller is better
            'value_scale': average_value_estimation_scorer # smaller is better
        })
```

If we want, we could prepare the visualization of our game

```bash
import glob
import io
import base64

from gym.wrappers import Monitor
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display

# start virtual display
display = Display(visible=0, size=(1400, 900))
display.start()

# play recorded video
def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''
            <video alt="test" autoplay loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
            </video>'''.format(encoded.decode('ascii'))))
    else: 
        print("Could not find video")
        
        
        
#Record video!
# wrap Monitor wrapper
env = Monitor(env, './video', force=True)

# evaluate
evaluate_on_environment(env)(cql)

show_video()
```


>## An Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nXXYVp0oEExxSGWvscic887sPOFp1l7e?usp=sharing)




