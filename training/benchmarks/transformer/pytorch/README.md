## 1. 模型信息

- 模型介绍

Transformer是一种神经机器翻译（NMT）模型，它使用注意力机制来提高训练速度和整体准确性。Transformer模型最初在Attention Is All You Need中被介绍，并在Scaling Neural Machine Translation中得到了改进。此实现基于Facebook建立在PyTorch之上的Fairseq NLP工具包中的优化实现。

- 模型代码来源

https://github.com/NVIDIA/DeepLearningExamples.git

目录：DeepLearningExamples/PyTorch/Translation/Transformer



## 2. 数据集

预处理后的数据集

- 数据集下载方法

```
wget https://klx-pytorch-work-bd.bd.bcebos.com/training/chenrui22_transformer/wmt14_en_de_joined_dict.tgz\?authorization\=bce-auth-v1/ec795a89fecd11edbe7251bf2e294682/2023-05-30T09%3A39%3A50Z/300/host/b6bd80c6b45804b345c9f91fc8cd08951cf68e24a5f64cda8ab54b9d0741d598\&x-bce-security-token\=ZjkyZmQ2YmQxZTQ3NDcyNjk0ZTg1ZjYyYjlkZjNjODB8AAAAAAYHAADqdnCNSQeeHzE9W3bbjCxdmMvhyFDXa4on2CUsc6%2B/OO0MNcxxRUWmXIcwUWodNUY9Lh7nn0xC4/fM5ROOGRNkm6uZCa8g71JhwVyt7dXOWZ5/l1ZdyZxp9dq2pMMBq07YlRuLaew7oMBMrTgZYPf79NtuQQXLrnWegtJw24oRieFzlwYvxiLyXls54mMf7R%2Bxp4LffolSTxebJohhK4Ecg6nA1o5oBD3ejf1r/ovk1q0yiFscXQTo/V//vJOk0uPQHhluOmdMCWNgbMEDKs43jzIdBstbhVMdnXsp7WsBrwLpqzv911%2B6w7soOxDydvKhzfXcmPFcyVPYobiJeoxY/rxYSGHIf42RLDPp4rrxFxEi9wKkPJglBgX3laRKC2eEvc8Nm%2B%2BTWJArrWhN2czsSi%2BLtnavywKJKtpUvhvUnS0zWoAe8t/1k/KvQo656PId1aLkQTaPQXOIz0DvNTEn9ezrtggdIBrkhKovgXpyvUFOvEraQRuAV5oFGuNJxVyi8zzyjDFkSSTbCRjMSKsta8PN%2BRVObRUzrRwcaIukzFFF%2BTk2zJd7/79rp17RP8Y3%2BOBnfBrHXIn%2B5DVVotfgkLw7/GJVYbH4uadfWP2dhoxbBkTkG2NpwpmqbJYmY2o%3D -O wmt14_en_de_joined_dict.tgz
```



## 3. 模型checkpoint

- 下载地址

  ```
  wget https://klx-pytorch-work-bd.bd.bcebos.com/training/chenrui22_transformer/checkpoint_best.pt\?authorization\=bce-auth-v1/74821e66ffa311eda9d5b78b0b417fe1/2023-05-31T11%3A08%3A21Z/300/host/30fbc1eae84014d78724860ce651166cac54b34b108b228c8e007949adbe7ef1\&x-bce-security-token\=ZjkyZmQ2YmQxZTQ3NDcyNjk0ZTg1ZjYyYjlkZjNjODB8AAAAAAYHAAAKFpHddC3fwlzR3x/rxqqnw7fH%2BUBw5WjFDlXNTVFd8scUM8I8d9h2UjYvRZgkn2ZImFqLfbvY%2BusojlUvCafnUngGEFeUIjamtPVr0cQeNJGIQ5NYx5nE6f7pp%2BCN1U0gF/bay9OZYQsvcPU2rgMbKKyhaz6noWPI2K7AQLgcDUuhh1zGTD7Lz51brbIteS1ftG1TXu5ghiCZLd2G%2BI1VWZG1mIBRk8XAMKNcR7kbf%2B3xkhKM4h/pbk1rvMWGAcLDwUepEkj/PKhQbTX8WkLI3bJe4hXecBEKAoytPR/hTatXjk7%2BXJ3l61lelyWYjtQmx0iDNu1iXDG8GjcxEA3qT7aLmE4g8obbV1E%2BuBEGICx8XKrznhaQCk3ejcjznhsbdPVgP2M0B68WhxX2lPb0o47SHVfxrvd60puvd5JKz8bBBNWxnNE01wR8FqP9I1p1Tvfjap%2BSF7qOgL10sgpznY/5buk4yyqb40ZzWONCKW8Kv5PyxXU8rR/lwUdt2hYNuHeXLNXQGToZiFL3BN4vq96KXrCGMBomQC1csHPapFcEGX0vfdpl29N5vHN6zGIB6GMuKkmv74F15w%2BJX4fltY5cjz7415Drq%2BjHLvx2FiEa2vwHolRG74c3Uo7MGl0%3D -O checkpoint_best.pt
  ```



## 4. 框架与芯片支持情况说明

- 目前FlagPerf提供 &lt;Framework&gt; 的实现.
- 目前已适配本模型的芯片如下：

|              | *Pytorch* | *Paddle* | *TensorFlow2* |
| ------------ | --------- | -------- | ------------- |
| *Nvidia GPU* | *✅*       | *N/A*      | *N/A*         |
| *Kunlun XPU* | *✅*       | *N/A*    | *N/A*         |

## 4. 运行命令

```shell
cd training/benchmarks/transformer/pytorch
python setup.py develop
python -m torch.distributed.launch --nproc_per_node 2 run_pretraining.py \
    --vendor=nvidia \
    --data_dir=../../../data/wmt14_en_de_joined_dict/ \
    --extern_config_dir=../../../nvidia/transformer-pytorch/config \
    --extern_config_file=config_A100x1x8.py
```