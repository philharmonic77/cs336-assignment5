from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

model_name = "Qwen/Qwen2.5-Math-1.5B"
save_path = BASE_DIR / "models" / "Qwen2.5-Math-1.5B"

save_path.mkdir(parents=True, exist_ok=True)

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cpu"  # 只下载，不加载到GPU
)

print("Saving locally...")
tokenizer.save_pretrained(str(save_path))
model.save_pretrained(str(save_path))

print(f"Model saved to {save_path}")
