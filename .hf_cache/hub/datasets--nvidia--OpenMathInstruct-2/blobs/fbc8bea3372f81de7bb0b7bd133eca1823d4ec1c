---
language:
- en
license: cc-by-4.0
size_categories:
- 10M<n<100M
task_categories:
- question-answering
- text-generation
pretty_name: OpenMathInstruct-2
dataset_info:
  features:
  - name: problem
    dtype: string
  - name: generated_solution
    dtype: string
  - name: expected_answer
    dtype: string
  - name: problem_source
    dtype: string
  splits:
  - name: train_1M
    num_bytes: 1350383003
    num_examples: 1000000
  - name: train_2M
    num_bytes: 2760009675
    num_examples: 2000000
  - name: train_5M
    num_bytes: 6546496157
    num_examples: 5000000
  - name: train
    num_bytes: 15558412976
    num_examples: 13972791
  download_size: 20208929853
  dataset_size: 26215301811
tags:
- math
- nvidia
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: train_1M
    path: data/train_1M-*
  - split: train_2M
    path: data/train_2M-*
  - split: train_5M
    path: data/train_5M-*
---


# OpenMathInstruct-2

OpenMathInstruct-2 is a math instruction tuning dataset with 14M problem-solution pairs 
generated using the [Llama3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) model.

The training set problems of [GSM8K](https://github.com/openai/grade-school-math)
and [MATH](https://github.com/hendrycks/math) are used for constructing the dataset in the following ways: 
- *Solution augmentation*: Generating chain-of-thought solutions for training set problems in GSM8K and MATH. 
- *Problem-Solution augmentation*: Generating new problems, followed by solutions for these new problems.      

<p>
 <img src="SFT Data Diagram 1.jpg" width="75%" title="Composition of OpenMathInstruct-2">
</p>

OpenMathInstruct-2 dataset contains the following fields:

- **problem**: Original problem from either the GSM8K or MATH training set or augmented problem from these training sets.
- **generated_solution**: Synthetically generated solution.
- **expected_answer**: For problems in the training set, it is the ground-truth answer provided in the datasets. **For augmented problems, it is the majority-voting answer.**
- **problem_source**: Whether the problem is taken directly from GSM8K or MATH or is an augmented version derived from either dataset.

  
<p>
 <img src="scaling_plot.jpg" width="40%" title="Scaling Curve">
</p>

We also release the 1M, 2M, and 5M, *fair-downsampled* versions of the entire training set corresponding to points in the above scaling plot. 
These splits are referred to as **train_1M**, **train_2M**, and **train_5M**. 
To use these subsets, just specify one of these subsets as split while downloading the data:
```python
from datasets import load_dataset

# Download only the 1M training split
dataset = load_dataset('nvidia/OpenMathInstruct-2', split='train_1M', streaming=True)
```  

To download the entire training set and to convert it into the jsonl format, use the following code snippet.
This might take 20-30 minutes (or more depending on your network connection) and will use ~20Gb of RAM.
```python
import json

from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset('nvidia/OpenMathInstruct-2', split='train')

print("Converting dataset to jsonl format")
output_file = "openmathinstruct2.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for item in tqdm(dataset):
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Conversion complete. Output saved as {output_file}")
```

Apart from the dataset, we also release the [contamination explorer](https://huggingface.co/spaces/nvidia/OpenMathInstruct-2-explorer) for looking at problems 
in the OpenMathInstruct-2 dataset that are similar to the [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MATH](https://github.com/hendrycks/math), 
[AMC 2023](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation/data/amc23), [AIME 2024](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I), 
and [Omni-MATH](https://huggingface.co/datasets/KbsdJames/Omni-MATH) test set problems. 

See our [paper](https://arxiv.org/abs/2410.01560) to learn more details!


### Note 
The released dataset doesn't filter out extremely long questions. After the dataset release, we found that 564 questions (roughly 0.1%) were longer than 1024 Llama tokens. 
We experimented with removing these questions and didn't see a performance drop (in fact, we observed a minor bump). Dropping these questions, helps with memory as well. 
So we would recommend, filtering out extremely long questions. We have updated the data preparation commands in our [Github documentation](https://nvidia.github.io/NeMo-Skills/openmathinstruct2/dataset/#converting-to-sft-format).   


## OpenMath2 models

To demonstrate the quality of this dataset, we release a series of OpenMath2 models trained on this data.

| Model | GSM8K | MATH | AMC 2023 | AIME 2024 | Omni-MATH |
|:---|:---:|:---:|:---:|:---:|:---:|
| Llama3.1-8B-Instruct | 84.5 | 51.9 | 9/40 | 2/30 | 12.7 |
| OpenMath2-Llama3.1-8B ([nemo](https://huggingface.co/nvidia/OpenMath2-Llama3.1-8B-nemo) \| [HF](https://huggingface.co/nvidia/OpenMath2-Llama3.1-8B)) | 91.7 | 67.8 | 16/40 | 3/30 | 22.0 |
| + majority@256 | 94.1 | 76.1 | 23/40 | 3/30 | 24.6 |
| Llama3.1-70B-Instruct | 95.8 | 67.9 | 19/40 | 6/30 | 19.0 |
| OpenMath2-Llama3.1-70B ([nemo](https://huggingface.co/nvidia/OpenMath2-Llama3.1-70B-nemo) \| [HF](https://huggingface.co/nvidia/OpenMath2-Llama3.1-70B)) | 94.9 | 71.9 | 20/40 | 4/30 | 23.1 |
| + majority@256 | 96.0 | 79.6 | 24/40 | 6/30 | 27.6 |

The pipeline we used to produce the data and models is fully open-sourced!

- [Code](https://github.com/NVIDIA/NeMo-Skills)
- [Models](https://huggingface.co/collections/nvidia/openmath-2-66fb142317d86400783d2c7b)
- [Dataset](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)

## Reproducing our results

We provide [all instructions](https://nvidia.github.io/NeMo-Skills/openmathinstruct2/)
to fully reproduce our results, including data generation.

## Citation

If you find our work useful, please consider citing us!

```bibtex
@article{toshniwal2024openmath2,
  title   = {OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data},
  author  = {Shubham Toshniwal and Wei Du and Ivan Moshkov and  Branislav Kisacanin and Alexan Ayrapetyan and Igor Gitman},
  year    = {2024},
  journal = {arXiv preprint arXiv:2410.01560}
}
```