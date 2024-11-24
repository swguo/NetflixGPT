import argparse
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datamodel import prepare_data
from model import load_model_and_tokenizer

def train_model(data_path, output_dir, num_train_epochs, batch_size, max_len):
    # Load data
    datasets = prepare_data(data_path, output_dir)
    model, tokenizer = load_model_and_tokenizer()

    # Tokenize dataset
    def tokenize_function(example):
        text = f"<|startoftext|> 標題:{example['title']} <|sep|>描述:{example['content']}"
        tokens = tokenizer(text, max_length=max_len, truncation=True, padding="max_length")
        input_ids = tokens['input_ids']
        sep_index = input_ids.index(tokenizer.convert_tokens_to_ids("<|sep|>"))
        labels = input_ids.copy()
        for i in range(sep_index + 1):
            labels[i] = -100
        tokens['labels'] = [label if label != tokenizer.pad_token_id else -100 for label in labels]
        return tokens

    tokenized_datasets = datasets.map(tokenize_function, remove_columns=["title", "content"])
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,        
        run_name="NetflixGPT",  # 指定運行名稱
        report_to="none",  # 禁用所有第三方報告工具，包括 wandb        
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        save_total_limit=1,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Train and save model
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT-2 model on a dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--max_len", type=int, default=128, help="Max token length.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation.")
    args = parser.parse_args()
    train_model(args.data_path, args.output_dir, args.num_train_epochs, args.batch_size, args.max_len)
