# 16362-final-project-nerf

This project is based on the PyTorch implementation of [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch). My personal additions mostly comprise of 1. debugging to get code working on my machine (Google Colab T4 and A100), 2. changing hyperparameters such as total number of iterations and frequency of printing results, and 3. adding plotting code to get graphed results of losses, PSNRs, LPIPS, SSIMS, etc.

## Installation

```
git clone https://github.com/kateyslee/16362-final-project-nerf
cd 16362-final-project-nerf
pip install -r requirements.txt
```

## How To Run?

### Quick Start

Download data for two example datasets: `lego` and `fern`
```
bash download_example_data.sh
```

To train a low-res `fern` NeRF:
```
python run_nerf.py --config configs/fern.txt
```
You can find the results in `logs/fern_test`.

## Method

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://tancik.com/nerf)  
 [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)\*<sup>1</sup>,
 [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/)\*<sup>1</sup>,
 [Matthew Tancik](http://tancik.com/)\*<sup>1</sup>,
 [Jonathan T. Barron](http://jonbarron.info/)<sup>2</sup>,
 [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/)<sup>3</sup>,
 [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)<sup>1</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>Google Research, <sup>3</sup>UC San Diego  
  \*denotes equal contribution  
  
<img src='imgs/pipeline.jpg'/>

> A neural radiance field is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views


## Citation
Kudos to the author of original [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch) implementation :)
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```
