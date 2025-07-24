from dataclasses import dataclass, field
import tempfile
from typing import Optional
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, HfArgumentParser
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM

# Example usage:
# python scripts/merge_adapter_weights.py --peft_model_id falcon-180b-lora-fa --output_dir merged-weights --save_tokenizer True

def save_model(model_path_or_id, adapter_path, save_dir, save_tokenizer=True):
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_id,
    )
    merged_model = PeftModel.from_pretrained(
       model,
       adapter_path,
    )
    merged_model = merged_model.merge_and_unload()

    merged_model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="3GB")
  
    # save tokenizer
    if save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
        tokenizer.save_pretrained(save_dir) 

@dataclass
class ScriptArguments:
    model_name_or_path: str
    peft_model_id: str = field(metadata={"help": "model id or path to model"})
    output_dir: Optional[str] = field(default="merged-weights", metadata={"help": "where the merged model should be saved"})
    save_tokenizer: Optional[bool] = field(default=True, metadata={"help": "whether to save the tokenizer"})

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

save_model(args.model_name_or_path, args.peft_model_id, args.output_dir, args.save_tokenizer)