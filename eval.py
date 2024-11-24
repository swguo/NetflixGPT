import argparse
import pandas as pd
import jieba
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm

def tokenize_text(text, lang):
    """
    根据指定语言对文本进行分词。
    """
    if lang == "zh":
        return list(jieba.cut(text))  # 中文分词
    elif lang == "en":
        return text.split()  # 英文分词
    else:
        raise ValueError("Unsupported language. Please use 'zh' for Chinese or 'en' for English.")

def evaluate(predictions_path, test_path, num, lang):
    # 加载数据
    if num is None:
        predictions = pd.read_csv(predictions_path)
        test_data = pd.read_csv(test_path)
    else:        
        predictions = pd.read_csv(predictions_path).iloc[:num]
        test_data = pd.read_csv(test_path).iloc[:num]

    # 初始化 BLEU 和 ROUGE 评估器
    bleu_scores = {"BLEU-1": [], "BLEU-2": [], "BLEU-3": [], "BLEU-4": []}
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=(lang == "en"))
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    # 遍历测试数据，逐一计算分数
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating"):
        reference = row['content']  # 参考文本
        prediction = predictions[predictions['title'] == row['title']]['generated_description'].values[0]

        # 分词
        reference_tokens = [tokenize_text(reference, lang)]
        prediction_tokens = tokenize_text(prediction, lang)

        # 计算 BLEU 分数
        bleu_scores["BLEU-1"].append(sentence_bleu(reference_tokens, prediction_tokens, weights=(1, 0, 0, 0)))
        bleu_scores["BLEU-2"].append(sentence_bleu(reference_tokens, prediction_tokens, weights=(0.5, 0.5, 0, 0)))
        bleu_scores["BLEU-3"].append(sentence_bleu(reference_tokens, prediction_tokens, weights=(0.33, 0.33, 0.33, 0)))
        bleu_scores["BLEU-4"].append(sentence_bleu(reference_tokens, prediction_tokens, weights=(0.25, 0.25, 0.25, 0.25)))

        # 计算 ROUGE 分数
        rouge_result = rouge_scorer_instance.score(reference, prediction)
        rouge_scores['rouge1'].append(rouge_result['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(rouge_result['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(rouge_result['rougeL'].fmeasure)

    # 计算平均分
    avg_bleu = {k: sum(v) / len(v) for k, v in bleu_scores.items()}
    avg_rouge = {k: sum(v) / len(v) for k, v in rouge_scores.items()}

    # 输出结果
    print(f"Evaluation for {lang.upper()} sentences:")
    print("Average BLEU Scores:", avg_bleu)
    print("Average ROUGE Scores:", avg_rouge)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the generated descriptions on a test dataset.")

    parser.add_argument("--generated_path", type=str, required=True, help="Path to the file containing generated descriptions.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--num", type=int, default=None, help="Number of test examples to evaluate.")
    parser.add_argument("--lang", type=str, choices=["zh", "en"], required=True, help="Language of the sentences ('zh' for Chinese, 'en' for English).")
    args = parser.parse_args()
    
    evaluate(args.generated_path, args.test_path, args.num, args.lang)
