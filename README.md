# Multi-Passage-BERT

This is a simple implementation of [Multi-Passage BERT](https://arxiv.org/abs/1908.08167)(not the same, but similar).

Tested with [DuReaderV2 dataset](https://github.com/baidu/DuReader). Using the squad evaluation script, we got:

```
"AVERAGE": "20.993"
"F1": "30.572"
"EM": "11.414"
```

Emm, not good.

## Demo & Blog

Chinese Version:

A simple demo can be found here: [AiSK](https://nlp-romance.aidigger.com/openqa)

A brief intro of openqa can be found in my blog: [OpenQA](https://xueyouluo.github.io/openqa/)

## Requirements

- tensorflow-gpu == 1.15 or 1.14
- tqdm
- horovod(optional)

## Features

Since we have 5 documents in one training example, we can't set the batch size too large. We use:

- Mix-precision training to speed up the training
- Gradient accumulation to make the batch size larger
- Distribute training, actually we only use one server with 2 2080Ti GPUs.

> If you have only one GPU, mix-precision + gradient accumulation works

> Note: If you want to train with distribute training, you should install horovod, the best way to get horovod is to use the [Nvidia docker](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags), we use the one with tag 19.10-py3. 


## How to run

- download Dureader dataset, and unzip

- run preprocess/preprocess.sh script to preprocess the dataset

- run preprocess/convert_dureader_to_squad.py to convert to dataset to squad-like dataset

- run run_mpb.sh to train the model

- run run_predict.sh to predict with the model

- run squad_evalute.py to get the evaluation results


## Postscript

Actually in our real project, we don't use multi-passage bert. We choose to use one MRC model + one answers ranker model, because we can train and optimize these two models separately.

This code is used for practicing. I don't have enough time to test or improve it. Some codes are copied from my jupyter notebook, maybe you need to fix some errors to run the code😂.

## References

- [ACL2018-DuReader](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2018-DuReader)
    - Data preprocess
- [Dureader-Bert](https://github.com/basketballandlearn/Dureader-Bert)
    - Data preprocess, predict
- [CMRC2018-Baselines](https://github.com/ymcui/cmrc2018/tree/master/baseline)
    - training
- [NVIDIA-BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)
    - Distribute training
    - AMP
    - Gradient Accumulation
- [OpenQA](https://mp.weixin.qq.com/s/IN-xzbrjjV2XgrGLPS5wRw)
    - Model part