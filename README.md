# Condition Video
Official Implementation of ConditionVideo.

ConditionVideo: Training-Free Condition-Guided Text-to-Video Generation

[Bo Peng](https://pengbo807.github.io), Xinyuan Chen, Yaohui Wang, Chaochao Lu, Yu Qiao

Our model generates realistic dynamic videos from random noise or given scene videos based on given conditions. Currently, we support openpose keypoint, canny, depth and segment condition. See [Project Page](https://pengbo807.github.io/conditionvideo.github.io/) for more information.

|canny|segment|depth|
|:-:|:-:|:-:|
|<img src="videos/0-0-road at night, oil painting style.gif" width="200"><br> a dog, comicbook style |<img src="videos/jellyfish.gif" width="200"><br> a red jellyfish, pastel colours.|<img src="videos/1-0-a horse under a blue sky.gif" width="200"><br> a horse under a blue sky.|

|pose|customized pose|
|:-:|:-:|
|<img src="videos/62-53-The Astronaut, brown background.gif" width="200"><br> The Astronaut, brown background|<img src="videos/1-2-18-ironman in the sea.gif" width="300"><br> ironman in the sea|
## Setup
To install the environments, use:
```
conda create -n tune-control python=3.10
```
check cuda version then install the corresponding pytorch package, note that we need pytorch==2.0.0
```
pip install -r requirements.txt
conda install xformers -c xformers
```
You may also need to download model checkpoints manually from hugging-face.
## Usage
To run the code, use

```
accelerate launch --num_processes 1 conditionvideo.py --config="configs//config.yaml"
```
for video generation, change the configuration in `config.yaml` for different generation settings.
## Citation
```
```
