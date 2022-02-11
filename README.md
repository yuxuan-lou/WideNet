# WideNet: An Implementation based on [Colossal-AI](https://www.colossalai.org/)

<p align="center">
  <img src="https://github.com/yuxuan-lou/WideNet/blob/main/IMG/IMG/model_overview.png" width="800">
</p>


This is the re-implementation of model WideNet from paper [Go Wider Instead of Deeper](https://arxiv.org/abs/2107.11817):
```
@misc{xue2021wider,
      title={Go Wider Instead of Deeper}, 
      author={Fuzhao Xue and Ziji Shi and Futao Wei and Yuxuan Lou and Yong Liu and Yang You},
      year={2021},
      eprint={2107.11817},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

* The original implementation was in TensorFlow+TPU. This re-implementation is in Colossal-AI+GPU.
* This repo applies [DALI](https://github.com/NVIDIA/DALI) datalodaer for ImageNet-1k dataset in TFRecord form

## Experiment Settings
### CV
<p align="center">
  <img src="https://github.com/yuxuan-lou/WideNet/blob/main/IMG/IMG/pretrain_setting.png" width="300">
</p>

### NLP
<p align="center">
  <img src="https://github.com/yuxuan-lou/WideNet/blob/main/IMG/IMG/pretrain_nlp.png" width="300">
</p>

## Results
### CV
<p align="center">
  <img src="https://github.com/yuxuan-lou/WideNet/blob/main/IMG/IMG/cv_performance.png" width="300">
</p>

### NLP
<p align="center">
  <img src="https://github.com/yuxuan-lou/WideNet/blob/main/IMG/IMG/nlp_performance.png" width="600">
</p>

## [Colossal-AI](https://www.colossalai.org/)

Colossal-AI is an integrated large-scale model training system with efficient parallelization techniques.

* [Site](https://www.colossalai.org/)
* [Github](https://github.com/hpcaitech/ColossalAI)
* [Examples](https://github.com/hpcaitech/ColossalAI-Examples)
