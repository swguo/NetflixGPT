Here is a formal and detailed `README.md` for your NetflixGPT project:

---

# NetflixGPT: A Netflix Title Description Generator

NetflixGPT is a text generation project powered by GPT-2, fine-tuned to generate descriptive summaries for Netflix titles. The project includes modules for training, inference, evaluation, and interactive demonstration, enabling users to explore the capabilities of a fine-tuned language model.

This repository provides scripts for model training, text generation, evaluation using BLEU and ROUGE metrics, and an interactive Gradio-based application for live text generation.

---

## Project Features

- **Fine-tuned GPT-2 Model**: Custom-trained on Netflix title and description data.
- **Multiple Generation Modes**:
  - **Greedy decoding** for deterministic text generation.
  - **Beam search** for diverse and optimized output.
- **Evaluation Metrics**: Comprehensive assessment using BLEU and ROUGE scores.
- **Interactive Demo**: A Gradio interface to explore text generation with real-time results.

---

## Directory Structure

```plaintext
.
├── data/                     # Directory for dataset files
│   ├── netflix_zhcn.csv      # Training dataset (Netflix titles and descriptions)
│   ├── netflix_test_zhcn.csv # Test dataset
├── NetflixGPT-chinese/       # Directory for trained model outputs
├── main.py                   # Script for model training
├── inference.py              # Script for generating descriptions
├── eval.py                   # Script for evaluating generated descriptions
├── demo.py                   # Script for interactive demo using Gradio
├── requirements.txt          # List of dependencies
├── README.md                 # Project documentation
```

---

## How to Use

### 1. Installation

#### Prerequisites

Ensure you have Python 3.8 or higher installed.

#### Clone the Repository

```bash
git clone https://github.com/your-username/netflix-gpt.git
cd netflix-gpt
```

#### Install Dependencies

Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**:

```plaintext
torch
transformers
gradio
nltk
rouge-score
```

---

### 2. Dataset Preparation

The dataset must contain Netflix titles and their corresponding descriptions. The repository assumes a dataset structure like this:

```csv
title,content
精神病特工,一位特工在驚悚的旅程中揭開神秘謎團。
追星女孩,一位女孩為追逐她的偶像踏上了夢幻的旅程。
```

Ensure your dataset is saved as `data/netflix_zhcn.csv`. You can split it into training and test sets manually or use `main.py` for automatic handling.

---

### 3. Scripts

#### **a. Model Training (`main.py`)**

Fine-tune the GPT-2 model on your Netflix dataset. Customize training parameters through command-line arguments.

##### Usage:

```bash
python main.py --data_path 'data/netflix_zhcn.csv' \
--output_dir './NetflixGPT-chinese' \
--num_train_epochs 5 \
--batch_size 16
```

##### Arguments:

- `--data_path`: Path to the input dataset (CSV file).
- `--output_dir`: Directory to save the trained model.
- `--num_train_epochs`: Number of epochs for training (default: 3).
- `--batch_size`: Batch size for training and evaluation (default: 8).

---

#### **b. Inference (`inference.py`)**

Generate descriptions for Netflix titles using the fine-tuned model. Choose between greedy decoding or beam search.

##### Usage:

```bash
python inference.py --test_path 'data/netflix_test_zhcn.csv' \
--model_path './NetflixGPT-chinese' \
--output_path 'infr_result.csv' \
--strategy 'greedy' \ 
--num 100
```

##### Arguments:

- `--test_path`: Path to the test dataset (CSV file).
- `--model_path`: Path to the trained model directory.
- `--output_path`: File to save the generated descriptions.
- `--strategy`: Text generation strategy. Choose between:
  - `greedy`: Generate text deterministically by selecting the most probable next token at each step.
  - `beam`: Use beam search for more diverse and optimized text generation.
- `--num`: Number of examples to process from the test dataset. If not provided, all examples in the dataset will be processed.
- `--num_beams`: Number of beams for beam search (only applicable in `beam` mode).

---

#### **c. Evaluation (`eval.py`)**

Evaluate the generated descriptions against the ground truth using BLEU and ROUGE metrics.
```bash
pip install jieba nltk rouge-score
```
##### Usage:

```bash
python eval.py --generated_path 'infr_result.csv' \
--test_path 'data/netflix_test_zhcn.csv' \
--num  100 \
--lang 'zh'
```

##### Arguments:

- `--generated_path`: Path to the file containing generated descriptions.
- `--test_path`: Path to the test dataset (CSV file).
- `--num`: Number of test examples to evaluate. If not provided, all examples in the dataset will be evaluated.
- `--lang`: Language of the sentences to evaluate:
  - `zh`: For Chinese (uses Jieba for tokenization).
  - `en`: For English (uses whitespace-based tokenization).
##### Output:

- **BLEU Scores**: BLEU-1, BLEU-2, BLEU-3, BLEU-4
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L

---

#### **d. Interactive Demo (`demo.py`)**

Explore the model's capabilities through a user-friendly Gradio interface. Select titles and generation modes, and view the generated descriptions in real-time.

```bash
pip install gradio
```
##### Usage:

```bash
python demo.py
```

##### Features:

- Dropdown menu for title selection (e.g., `精神病特工`, `追星女孩`, `非法女人`).
- Radio buttons for choosing generation mode (`greedy` or `beam search`).

**Example Interaction**:

- **Input**:
  - Title: `精神病特工`
  - Mode: `greedy` or `beam`
- **Output**:
  ```
  A secret agent uncovers shocking truths while navigating the dangers of espionage.
  ```

---

## Future Improvements

1. Support for multilingual datasets.
2. Add functionality to input custom titles in the Gradio demo.
3. Allow real-time adjustment of generation parameters (e.g., temperature, top-k, top-p).