from transformers import AutoTokenizer, AutoModelForCausalLM
import os


os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"


# 使用预训练模型（示例使用GPT-2快速验证，正式训练建议换为LLama-2-7b）
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 基础对话函数
def chat(text, history=[]):
    inputs = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split(text)[-1].strip()

# 测试
print(chat("今天心情不太好..."))