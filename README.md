# WideNet: An Implementation based on [Colossal-AI](https://www.colossalai.org/)
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
