# ANAH: Analytical Annotation of Hallucinations in Large Language Models

[![license](https://img.shields.io/github/license/InternLM/opencompass.svg)](./LICENSE)

This is the repo for our paper: [ANAH: Analytical Annotation of Hallucinations in Large Language Models](). The repo contains:

+ The [data](#🤗-HuggingFace-Model-&-Dataset#Dataset) for training and evaluating the LLM which consists of ~12k sentence-level hallucination annotations.
+ The [model](#🤗-HuggingFace-Model-&-Dataset) for annotating the hallucination.
+ The [code](#🏗️-Evaluation) for evaluating the LLM.


## ✨ Introduction


ANAH is a bilingual dataset that offers ANalytical Annotation of Hallucinations in LLMs within Generative Question Answering.

Each answer sentence in our dataset undergoes rigorous annotation, involving the retrieval of a reference fragment, the judgment of the hallucination type, and the correction of hallucinated content.

ANAH consists of ~12k sentence-level annotations for ~4.3k LLM responses covering over 700 topics, constructed by a human-in-the-loop pipeline.


<p align="center">
  <img src="docs/figure/teaser.jpg" height="550">
</p>


## 🚀 What's New

## 🤗 HuggingFace Model & Dataset

### Dataset

The ANAH dataset is available on Huggingface dataset hub.

<div style="text-align: center;">
  <table border="1" align="center">
    <tr>
      <th>Dataset</th>
      <th>Huggingface Repo</th>
    </tr>
    <tr>
      <td align="center">ANAH</td>
      <td align="center"><a href="">Dataset Link</a></td>
    </tr>
  </table>
</div>



### Model

ANAH can be used for training hallucination annotators. 

We have trained the annotators based on InternLM2 series models.

The 7B & 20B annotator models are available on Huggingface & OpenXLab model hub.

<div style="text-align: center;">
  <table border="1" align="center">
    <tr>
      <th>Model</th>
      <th>Huggingface Repo</th>
      <th>OpenXLab Repo</th>
    </tr>
    <tr>
      <td align="center">ANAH-7B</td>
      <td align="center"><a href="">Model Link</a></td>
      <td align="center"><a href="">Model Link</a></td>
    </tr>
    <tr>
      <td align="center">ANAH-20B</td>
      <td align="center"><a href="">Model Link</a></td>
      <td align="center"><a href="">Model Link</a></td>
    </tr>
  </table>
</div>


The models follow the conversation format of InternLM2-chat, with the template protocol as:

```python
dict(role='user', begin='<|im_start|>user\n', end='<|im_end|>\n'),
dict(role='assistant', begin='<|im_start|>assistant\n', end='<|im_end|>\n'),
```

## 🏗️ ️Evaluation

ANAH can be used for evaluating the current open-source and close-source LLMs' ability to generate fine-grained hallucination annotation.

### 1. Environment Setup

We recommend you use `Python 3.10` and `Pytorch 1.13.1`.

```bash
conda create --name anah python=3.10.13
conda activate anah
pip install -r requirements.txt
```

### 2. Inference and Evaluation

We now support the evaluation of the InternLM2, Llama2, Qwen, and Baichuan2 series of open-source models.

We recommend you download the huggingface model to your local path and replace the `{your_hf_model_path}` to that path.

Our evaluations are conducted on NVIDIA A100 GPUs, and OOM may occur on other types of machines.

```bash
python -u ./eval/eval.py \
    --model_type {your_model_type} \ 
    --server_addr {your_hf_model_path} \
    --json_path {test_set_path} \
    --output_path {your_inference_results_path} \
    --eval_sorce_path {your_evaluation_result_path} \
```

## ❤️ Acknowledgements

ANAH is built with [InternLM](https://github.com/InternLM/InternLM) and [LMDeploy](https://github.com/InternLM/lagent). Thanks for their awesome work!

## 🖊️ Citation

If you find this project useful in your research, please consider citing:
```

```

## 💳 License

This project is released under the Apache 2.0 [license](./LICENSE).