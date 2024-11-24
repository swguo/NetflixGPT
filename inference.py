import argparse
import pandas as pd
import torch
from model import load_model_and_tokenizer
from transformers import GPT2LMHeadModel
from tqdm.auto import tqdm


def generate_description(model, tokenizer, title, strategy="greedy", max_length=128, num_beams=3, device="cpu"):
    input_text = f"標題:{title}描述:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    if strategy == "beam":
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id
        )
    else:  # Greedy decoding
        output = model.generate(
            input_ids,
            max_length=max_length,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            early_stopping=True
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text = generated_text.replace(" ","")
    return generated_text.replace(input_text, "").strip()

def inference(test_path, model_path, output_path, strategy, num, num_beams, max_len):

    # Detect if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, tokenizer = load_model_and_tokenizer()
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    if num == None:
        test_data = pd.read_csv(test_path)
    else:
        test_data = pd.read_csv(test_path).iloc[:num]
        
    results = []
    for _, row in tqdm(test_data.iterrows(),total=len(test_data)):
        title = row['title']
        description = row['content']
        generated = generate_description(model, tokenizer, title, strategy=strategy, max_length=max_len, num_beams=num_beams,device=device)
        results.append({"title": title,"description":description, "generated_description": generated})
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, encoding="utf_8_sig", index=False)

    # Display first 5 results
    print(result_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a test dataset.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--max_len", type=int, default=128, help="Max token length.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated results.")
    parser.add_argument("--strategy", type=str, choices=["greedy", "beam"], default="greedy", help="Generation strategy: 'greedy' or 'beam'.")
    parser.add_argument("--num", type=int, default=None, help="Number of test")
    parser.add_argument("--num_beams", type=int, default=3, help="Number of beams for beam search (only used if strategy is 'beam').")

    args = parser.parse_args()
    inference(args.test_path, args.model_path, args.output_path, args.strategy, args.num, args.num_beams, args.max_len)
