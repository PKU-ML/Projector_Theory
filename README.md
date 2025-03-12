# Projection Head is Secretly an Information Bottleneck
This repository includes a PyTorch implementation of the ICLR 2025 paper [Projection Head is Secretly an Information Bottleneck]() authored by [Zhuo ouyang*](), [Kaiwen Hu*](https://kaotty.github.io/), [Qi Zhang*](), [Yifei Wang](https://yifeiwang77.com/) and [Yisen Wang](https://yisenwang.github.io/).

## Abstract
Recently, contrastive learning has risen to be a promising paradigm for extracting meaningful data representations. Among various special designs, adding a projection head on top of the encoder during training and removing it for downstream tasks has proven to significantly enhance the performance of contrastive learning. However, despite its empirical success, the underlying mechanism of the projection head remains under-explored. In this paper, we develop an in-depth theoretical understanding of the projection head from the information-theoretic perspective. By establishing the theoretical guarantees on the downstream performance of the features before the projector, we reveal that an effective projector should act as an information bottleneck, filtering out the information irrelevant to the contrastive objective. Based on theoretical insights, we introduce modifications to projectors with training and structural regularizations. Empirically, our methods exhibit consistent improvement in the downstream performance across various real-world datasets, including CIFAR-10, CIFAR-100, and ImageNet-100. We believe our theoretical understanding on the role of the projection head will inspire more principled and advanced designs in this field.

## Instructions
All experiments are conducted with one or two NVIDIA RTX 3090 GPU(s). We mainly conduct the following experiments on CIFAR-10, CIFAR-100, and ImageNet-100. In order to prepare for the experiments, you can run the following command.
```bash
pip install -r requirements.txt
```

Next, be sure to set the directories before you run the expriments. This is done in the `scripts` folder. For example, if you want to pretrain with SimCLR on CIFAR-100, you may open `scripts/pretrain/cifar/simclr.yaml`, where you can set the dataset to `cifar100`, and training path and validation path to your directories. You may also change the hyper parameters here.

We provide code for SimCLR and Barlow Twins on CIFAR-10, CIFAR-100, and ImageNet-100. In order to conduct the pretraining experiments, you can run the following command.
```bash
python main_pretrain.py --config-path scripts/pretrain/{dataset}/ --config-name {config-name}
```
You may select the training dataset from `{cifar, imagenet-100}`. To specifically choose CIFAR-10 or CIFAR-100, you can edit the dataset in the `scripts/pretrain/cifar/.yaml` files. And if you wish to run the experiment using a specific framework, such as SimCLR, you can set the config name to `simclr.yaml`.
<!-- You may set the config name to the one corresponding to the framework under this path. For instance, if you want to run the SimCLR experiments, you can set the config name to `simclr.yaml`. -->
If you want to test the downstream performance of the model, you can save the checkpoint and conduct the linear probing downstream test using the following command.
```bash
python main_linear.py --config-path scripts/linear/{dataset}/ --config-name {config-name}
```
Ensure that you use the same dataset and framework as those employed in the pretraining experiment.

## Utility of Specific Methods
In our paper, we provide three regularization methods: training regularization (controlled by parameter `lmbd`), discrete projector (controlled by parameter `point_num`), and sparse autoencoder (enabled by flag `sparse_autoencoder`). If you wish to explore the sparse autoencoder method under a specific framework, such as SimCLR, you can set the `sparse_autoencoder` to `True` in `simclr.yaml`. By adjusting `lmbd` and `point_num`, you can activate training regularization and the discrete projector, either individually or in combined configurations.
<!-- Alternatively, you are free to use other autoencoder implementations of your choice. By incorporating the `self.sparse_autoencoder` function, the sparse autoencoder can serve as an effective projector. -->


## Citing This Work
If you find the work useful, please cite the accompanying paper:
```
@article{ouyang2025projection,
  title={Projection Head is Secretly an Information Bottleneck},
  author={Ouyang, Zhuo and Hu, Kaiwen and Zhang, Qi and Wang, Yifei and Wang, Yisen},
  journal={arXiv preprint arXiv:2503.00507},
  year={2025}
}
```

## Acknowledgement
Our code follows the official implementation of solo-learn (https://github.com/vturrisi/solo-learn).
