from transformers import GPT2LMHeadModel, BertTokenizerFast

PRETRAINMODEL = "ckiplab/gpt2-base-chinese"  # Pretrained model path

def load_model_and_tokenizer():
    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINMODEL)
    tokenizer.add_special_tokens({"bos_token": "<|startoftext|>", "eos_token": "<|endoftext|>", "sep_token": "<|sep|>"})
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model and resize embeddings
    model = GPT2LMHeadModel.from_pretrained(PRETRAINMODEL)
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer
