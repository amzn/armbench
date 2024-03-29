
![armbench](./images/armbench_logo.png)

ARMBench is a large-scale benchmark dataset for perception and manipulation challenges in a robotic pick-and-place setting. The dataset is collected in an Amazon warehouse and captures a wide variety of objects and configurations. It comprises images and videos for different stages of robotic manipulation including picking, transferring, and placing with high-quality annotations.

Currently, the dataset provides data annotations for three main computer vision tasks: Object segmentation, Object Identification, and Defect Detection on images and videos.


This repository contains sample code to load and train models using the ARMBench dataset described in this paper:

[ARMBench: An Object-centric Benchmark Dataset for Robotic Manipulation](https://arxiv.org/abs/2303.16382)

Presented at [ICRA 2023](https://www.icra2023.org/)


# Segmentation

# Identification

# Defect Detection


If you find this code useful in your research then please cite
```
@misc{mitash2023armbench,
      title={ARMBench: An Object-centric Benchmark Dataset for Robotic Manipulation}, 
      author={Chaitanya Mitash and Fan Wang and Shiyang Lu and Vikedo Terhuja and Tyler Garaas and Felipe Polido and Manikantan Nambi},
      year={2023},
      eprint={2303.16382},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```