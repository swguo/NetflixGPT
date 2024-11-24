import torch
from transformers import BertTokenizerFast, GPT2LMHeadModel
import gradio as gr
import asyncio

# 設定裝置為 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入訓練好的模型和 tokenizer
model_path = "./NetflixGPT-chinese"  # 修改為你訓練模型的儲存路徑
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.to(device)
model.eval()


# Define the description generation function with mode selection
def generate_description(title, mode="greedy", max_length=512, temperature=0.7, top_k=50, top_p=0.3):
    # Prepare input text
    input_text = f"標題: {title} 描述:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate text based on the selected mode
    with torch.no_grad():
        if mode == "beam":
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,  # Control randomness
                top_k=top_k,              # Limit to top-k words                
                do_sample=True,           # 啟用隨機抽樣
                top_p=top_p,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        else:  # Default to greedy decoding
            output = model.generate(
                input_ids,
                max_length=max_length,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )

    # Decode the generated description
    generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

# Define async function for progressive display
async def display_text(title, mode, temperature, top_k, top_p):
    description = generate_description(title, mode=mode, temperature=temperature, top_k=top_k, top_p=top_p)
    displayed_text = ""
    for char in description:
        displayed_text += char
        await asyncio.sleep(0.05)  # Adjust speed of character display
        yield displayed_text

iface = gr.Interface(
    fn=display_text,
    inputs=[
        gr.Dropdown(choices=["精神病特工", "追星女孩", "非法女人"], label="Select a Title"),
        gr.Radio(choices=["greedy", "beam"], value="greedy", label="Generation Mode"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, step=1, value=50, label="Top-K"),
        gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.3, label="Top-P"),
    ],
    outputs=gr.Textbox(label="Generated Description"),
    title="Netflix Title Description Generator",
    description="Select a Netflix title and customize generation parameters to view the model's generated description."
)


# Launch Gradio interface
iface.launch()