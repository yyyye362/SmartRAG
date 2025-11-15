# SmartRAG: Learning When and What to Retrieve for Function-Level Code Generation


## ✨ Introduction

SmartRAG is an adaptive RAG framework that decides when to retrieve and what to retrieve for code generation. By combining a lightweight retrieval trigger with an execution-guided knowledge selector, it avoids unnecessary retrieval and injects only useful Knowledge—significantly improving executability with lower inference cost.

![./assets/images/ac_overview.png](./assets/images/SmartRAG.jpg)



## 🛠️ Preparations
The code requires some dependencies as specified in requirements.txt. Please follow the relevant libraries to install or run:
```bash
pip install -r requirements.txt
```

## 📚 Datasets

We use two major datasets in our work: APPS and LeetCode Dataset.

APPS
Download link & instructions:
https://github.com/hendrycks/apps
 provides the full dataset (~1.3 GB). 

Also available via Hugging Face:

from datasets import load_dataset
ds = load_dataset("codeparrot/apps")
``` :contentReference[oaicite:3]{index=3}  
We use the *train* split as the training set, and the *test* split as the evaluation set.

LeetCode Dataset
The curated version “LeetCodeDataset” released recently supports robust evaluation and fine-tuning of code models. 
Download via → [data](data/LeetCode) folder..
We similarly use its training portion for fine-tuning, and its test portion for evaluation.



## 🤗 Model
We fine-tune on four models:
  - [Qwen2.5-Coder (1.5B)](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
  - [Qwen2.5-Coder (3B)](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)
  - [DeepSeek-Coder (6.7B)](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)
  - [Qwen2.5-Coder (7B)](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)



## 🧲 Extracting
Two skeletons can be extracted by the following code:
<pre>
python SMS.py
python SSS.py
python TAS.py
</pre>


## 🏋️ Finetuning

First, fine-tune the base model on the code of the APPS+EFFI dataset and the corresponding natural language description of the APPS dataset by running the following code:
<pre>
python train_base_model.py
</pre>
Then, fine-tune the base model in a multi-task framework :
<pre>
python train_mask_model.py
python train_skeleton_model.py
python train_total_model.py
</pre>


## ✨ Generating

Generate candidate codes for different fine-tuning methods:
<pre>
python generate_base.py
python generate_mask.py
python generate_skeleton.py
python generate_total.py
</pre>

## 📊 Evaluate

You can run "test_one_solution.sh" to evaluate the functional correctness and efficiency of the generated code:
<pre>
cd evaluate/metric
bash test_one_solution.sh
cd evaluate/metric_time
bash test_one_solution.sh
</pre>
