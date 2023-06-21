# few-shot-adversarial-robustness
## Code for ACL'23 Findings paper on 'Adversarial Robustness of Prompt-based Few-Shot Learning for Natural Language Understanding'

**Authors**: [Venkata Prabhakara Sarath Nookala]()\*<sup>1</sup>, [Gaurav Verma](https://gaurav22verma.github.io/)\*<sup>1</sup>, [Subhabrata Mukherjee](https://www.microsoft.com/en-us/research/people/submukhe/)<sup>2</sup>, and [Srijan Kumar](https://faculty.cc.gatech.edu/~srijan/)<sup>1</sup>  
**Affiliations**: <sup>1</sup>Georgia Institute of Technology, <sup>2</sup>Microsoft Research

**Paper (pdf)**: [arXiv](https://arxiv.org/abs/2306.11066)  
**Poster (pdf**): [coming soon]()  

## Quick links

* [Overview](#overview)
* [Requirements](#requirements)
* [Prepare the data](#prepare-the-data)
* [Run the experiments](#run-experiments)
* [Citation](#citation)


## Overview

In this work we evaluate the adversarial robustness of several prompt-based few-shot learning approaches for natural language understanding tasks [paper](paper link).

We leveraged the below open source implementations to reproduce the results presented in the paper
* [LM-BFF](https://github.com/princeton-nlp/LM-BFF/) for classic finetuning, LM-BFF experiments
* [FewNLU](https://github.com/THUDM/FewNLU) for PET, iPET experiments

## Requirements

The two open source implementations detailed in the above section have different dependency packages with specific version requirements listed in the "requirements.txt" files within the individual folders `./LM-BFF` and `./FewNLU`. Please install them using the following command before running the specific experiment

```
pip install -r requirements.txt
```

**NOTE**: Different versions of packages (like `pytorch`, `transformers`, etc.) may lead to different results from the paper. However, the trend should still hold no matter what versions of packages you use.

## Prepare the data

We followed the [LM-BFF](https://github.com/princeton-nlp/LM-BFF/)'s methodology to preprocess the GLUE dataset and also integrated the AdvGLUE dataset preprocessing to enable testing on AdvGLUE benchmark seamlessly

Please download the original dataset from [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar) and extract the files to `./data/original`, or run the following commands:

```bash
cd data
bash download_dataset.sh
```

#### advGLUE testing 

Download the raw advGLUE evaluation dataset from [here](https://adversarialglue.github.io/dataset/dev.zip) and extract the files into `./data/advGLUE` folder. we preprocess the dev.json file from advGLUE dataset to generate the advGLUE dev data files along with the GLUE dev data files(in the same format and location) 

Use the following command (in the root directory) to add the advGLUE dev files to respective tasks in `./data/origin` folder

```bash
python utils/generate_advglue_data.py
```
**Note**: This will add `dev_adv.tsv` files along with `dev.tsv` files to individual task folders 

Then use the following command (in the root directory) to generate the few-shot data we need:

```bash
python utils/generate_k_shot_data.py
```

See `data/utils/generate_k_shot_data.py` for more options. For results in the paper, we use the default options: we take `K=16` and take 5 different seeds of 13, 21, 42, 87, 100. The few-shot data will be generated to `data/k-shot`. In the directory of each dataset, there will be folders named as `$K-$SEED` indicating different dataset samples.

**NOTE**: The above data preparation step is common for all the experiments and data preparation steps inside the individual experiment folders can be safely ignored.

## Run experiments

All our experiments were conducted using a single GPU with 48GB RAM (NVIDIA Quadro RTX 8000). Please note that using different accelerator and distributed training configurations may lead to different results from the paper. However, the overall trend should remain the same. 

To eliminate the need for an extensive hyper-parameter search, for each of the prompting methods, unless otherwise stated, we use the same set of hyperparametersas recommended in Gao et al. (2020); most notably, batch size of 8, learning rate set to 10−5, and max sequence length of 256

To reproduce the results for classic-finetuning and LM-BFF experiments, please refer to the detailed instructions in the `README.md` inside the `./LM-BFF` folder. Below is an example command to run the LM-BFF's evaluation on SST-2 task.

```bash
for seed in 13 21 42 87 100
do
   TAG=LM-BFF \
   TYPE=prompt-demo \
   TASK=SST-2 \
   BS=8 \
   LR=1e-5 \
   SEED=$seed \
   MODEL=roberta-large \
   bash run_experiment.sh 
done

python tools/gather_result.py --condition "{'tag': 'LM-BFF', 'task_name': 'sst-2', 'few_shot_type': 'prompt-demo'}"
```

To reproduce the results for iPET and PET experiments, please refer to the detailed instructions in the `README.md` inside the `./FewNLU` folder. Below are example commands to run the evaluation of iPET and PET respectively on qqp task
```
bash scripts/search_ipetmlm_multisplit.sh qqp 0 albert ../data/k-shot/QQP/64-13 fewnlu/result/QQP/ipet

bash scripts/search_petmlm_multisplit.sh qqp 0 albert ../data/k-shot/QQP/64-13 fewnlu/result/QQP/pet
```
## Citation

Please cite our paper if you use our evaluation experiments in your work:

```bibtex
@inproceedings{nookala2023adversarial,
   title={Adversarial Robustness of Prompt-based Few-Shot Learning for Natural Language Understanding,
   author={Nookala, Venkata Prabhakara Sarath and Verma, Gaurav and Mukherjee, Subhabrata and Kumar, Srijan},
   booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
   year={2023}
}
```
