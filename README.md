# NetflixGPT

NetflixGPT is a multi-language text generator based on the GPT-2 model. It is designed to generate Netflix show descriptions in both English and Chinese. The project provides training, inference, and evaluation scripts with modular structures for both languages.

## Directory Structure

```
.
├── data/                           # Directory for storing datasets
├── eval.ipynb                      # Notebook for model evaluation
├── Infernece_en.ipynb              # Notebook for English inference
├── Infernece_zhcn.ipynb            # Notebook for Chinese inference
├── NetflixGPT-chinese/             # Files related to the Chinese GPT model
├── NetflixGPT-english/             # Files related to the English GPT model
├── Train_en.ipynb                  # Notebook for training the English model
├── Train_zhcn.ipynb                # Notebook for training the Chinese model
└── README.md                       # Project documentation
```

---

## Features

- **Multi-language Support**: Includes separate workflows for English and Chinese GPT models.
- **Modular Design**: Model files for English and Chinese are separated into `NetflixGPT-english` and `NetflixGPT-chinese` directories for clarity.
- **Reproducibility**: Training, inference, and evaluation workflows are provided through Jupyter Notebooks for easy replication.

---

## Dataset

The dataset used in this project is sourced from [Kaggle - Netflix Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows).

### Dataset Description

The dataset contains information about Netflix shows, including titles and descriptions. It has been adapted for this project to support both English and Chinese text generation tasks.

---

## Getting Started

### 1. Install Dependencies

Ensure Python 3.8 or later is installed, and use the following command to install required dependencies:

```bash
pip install -r requirements.txt
```

**`requirements.txt` Example:**

```text
transformers
torch
tqdm
gradio
nltk
rouge-score
```

### 2. Prepare Data

Place your dataset files in the `data/` directory. The dataset should be formatted as follows:

- **English Dataset**: Titles and descriptions in English.
- **Chinese Dataset**: Titles and descriptions in Chinese.

Example:

**English Dataset (CSV):**
```csv
title,description
Stranger Things,A group of kids discover supernatural events in their town.
Breaking Bad,A high school teacher turns to cooking meth after a cancer diagnosis.
```

**Chinese Dataset (CSV):**
```csv
標題,描述
精神病特工,一名特工陷入了神秘的精神謎團。
追星女孩,一個女孩踏上了追逐明星夢想的旅程。
```

---

### 3. Training

#### Train the English Model

Run `Train_en.ipynb` to fine-tune the English GPT model with your dataset. You can adjust training parameters like `num_train_epochs` and `batch_size` as needed.

#### Train the Chinese Model

Run `Train_zhcn.ipynb` to fine-tune the Chinese GPT model using the Chinese dataset.

---

### 4. Inference

#### English Inference

Run `Infernece_en.ipynb` to load the trained English model and generate show descriptions by providing a title as input.

#### Chinese Inference

Run `Infernece_zhcn.ipynb` to load the trained Chinese model and generate show descriptions in Chinese.

---

### 5. Evaluation

Run `eval.ipynb` to evaluate the model's performance using metrics such as BLEU and ROUGE. The notebook includes pre-defined functions for comparing generated and reference descriptions.

---

## Parameters

You can adjust the following parameters for fine-tuning and inference:

- **`max_length`**: Controls the maximum length of the generated text, including the input.
- **`top_k`**: Limits the number of highest-probability tokens to consider during generation.
- **`temperature`**: Adjusts the randomness of text generation; lower values make the output more deterministic.
- **`num_train_epochs`**: Specifies the number of training epochs.

---
