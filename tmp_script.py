from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LongformerForSequenceClassification, LongformerTokenizer
model_name = "allenai/longformer-base-4096"  # e.g., "meta-llama/Llama-2-7b-chat-hf"
save_dir = ".checkpoints/Longformer/"  # Local directory to save

# Load tokenizer and model
tokenizer = LongformerTokenizer.from_pretrained(model_name)
model = LongformerForSequenceClassification.from_pretrained(model_name)

# Save to disk
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

