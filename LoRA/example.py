# coding:utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import lora_layer as lora

from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

class SFTDataset(Dataset):
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 data_path: str,
                 load_local: bool=False,
                 max_len: int=256,
                 split_len: str="1%"):
        
        super().__init__()

        self.tokenizer = tokenizer

        if load_local:
            ds = load_dataset("json", data_dir=data_path, split=f"train[:{split_len}]")
        else:
            ds = load_dataset(data_path, split=f"train[:{split_len}]")

        self.max_len = max_len

        self.ds = []
        for d in ds:
            out = self._process_each_data(d)
            if out != None:
                self.ds.append(out)

    def _process_each_data(self, example):
        instruction = example["instruction"].strip()
        input_ = example["input"].strip()
        output = example["output"].strip()

        instruction_msg = [{"role": "user", "content": (instruction + f"\n{input_}" if len(input_) > 0 else instruction)}]
        
        tokenized_instruction = self.tokenizer.apply_chat_template(instruction_msg, tokenize=True, add_generation_prompt=True)
        tokenized_output = self.tokenizer(output + "<<|im_end|>>" + f"{self.tokenizer.eos_token}\n")["input_ids"]

        tokenized_prompt = (tokenized_instruction + tokenized_output)[:self.max_len]
        input_ids = tokenized_prompt
        attention_mask = ([1] * len(tokenized_prompt))[:self.max_len]
        # labels = ([-100 for _ in range(len(tokenized_instruction))] + tokenized_output)[:self.max_len]
        # labels = [-100 for _ in range(len(tokenized_instruction))] + tokenized_output[: (self.max_len - len(tokenized_instruction))]

        labels_no_loss = [-100 for _ in range(len(tokenized_instruction))]
        labels_loss = tokenized_output
        if len(labels_no_loss) < self.max_len:
            labels = labels_no_loss + labels_loss[:(self.max_len - len(labels_no_loss))]
        else:
            print("no way")
            return None
            labels = labels_no_loss[:self.max_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        return self.ds[idx]

def collate_fn(batch: list[dict], tokenizer: AutoTokenizer):
    max_len = max(len(item["input_ids"]) for item in batch)
    bsz = len(batch)
    tensor_input_ids = torch.LongTensor(bsz, max_len).fill_(tokenizer.eos_token_id)
    tensor_attention_mask = torch.LongTensor(bsz, max_len).fill_(0)
    tensor_labels = torch.LongTensor(bsz, max_len).fill_(-100)

    for idx, item in enumerate(batch):
        input_ids = item["input_ids"]
        tensor_input_ids[idx, -len(input_ids):] = torch.LongTensor(input_ids)

        attention_mask = item["attention_mask"]
        tensor_attention_mask[idx, -len(attention_mask):] = torch.LongTensor(attention_mask)

        labels = item["labels"]
        tensor_labels[idx, -len(labels):] = torch.LongTensor(labels)
    
    return {
        "input_ids": tensor_input_ids,
        "attention_mask": tensor_attention_mask,
        "labels": tensor_labels,
    }

# def inference(
#     model,
#     tokenizer,
#     text,
#     max_new_tokens: int=160,
#     do_sample: bool=True,
#     temperature: float=0.3,
#     print_inputs: bool=True,
#     streaming: bool=False,
# ):  

#     device = model.device
#     prompt_msg = [
#         {"role": "user", "content": text}
#     ]
#     prompt = tokenizer.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
#     input_ids = inputs["input_ids"].to(device)
#     im_end_id = tokenizer.encode("<|im_end|>")[0]

#     if print_inputs:
#         print(prompt, end="")
    
#     stop_words = [tokenizer.eos_token_id, im_end_id]
#     generated_tokens = []

#     for _ in range(max_new_tokens):
#         with torch.no_grad():
#             outputs = model(input_ids)
        
#         logits = outputs.logits[:, -1, :]

#         if do_sample:
#             logits = logits / temperature
#             probs = F.softmax(logits, dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1)
#         else:
#             next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
#         if next_token.item() in stop_words:
#             break
#         generated_tokens.append(next_token.item())
        
#         if streaming:
#             yield tokenizer.decode(generated_tokens)
        
#         input_ids = torch.cat([input_ids, next_token], dim=-1)
    
#     generated_text = tokenizer.decode(generated_tokens)
#     return generated_text

def inference_streaming(model,
                        tokenizer,
                        text,
                        system_prompt: str=None,
                        max_new_tokens: int=160,
                        do_sample: bool=True,
                        temperature: float=0.3,
                        print_input: bool=True):
    
    device = model.device
    prompt_msg = []
    if system_prompt:
        prompt_msg = [{"role": "system", "content": system_prompt}]
    prompt_msg.append({"role": "user", "content": text})
    prompt = tokenizer.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    if print_input:
        print(prompt, end="")
    
    stop_words_ids = [tokenizer.encode("<|im_end|>")[0], tokenizer.eos_token_id]

    generated_tokens_id = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            output = model(input_ids)
        cur_logits = output.logits[:, -1, :]

        if do_sample:
            cur_logits = cur_logits / temperature
            cur_probs = F.softmax(cur_logits, dim=-1)
            cur_idx = torch.multinomial(cur_probs, num_samples=1)
        else:
            cur_idx = torch.argmax(cur_logits, dim=-1, keepdim=True)
        
        if cur_idx.item() in stop_words_ids:
            break

        generated_tokens_id.append(cur_idx.item())
        yield tokenizer.decode(generated_tokens_id)

        input_ids = torch.cat((input_ids, cur_idx), dim=-1)

def inference(model,
              tokenizer,
              text,
              system_prompt: str=None,
              max_new_tokens: int=160,
              do_sample: bool=True,
              temperature: float=0.3,
              print_input: bool=True):
    
    device = model.device
    prompt_msg = []
    if system_prompt:
        prompt_msg = [{"role": "system", "content": system_prompt}]
    prompt_msg.append({"role": "user", "content": text})
    prompt = tokenizer.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    
    if print_input:
        print(prompt, end="")
    
    stop_tokens_id = [tokenizer.encode("<|im_end|>")[0], tokenizer.eos_token_id]

    generated_tokens_id = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            output = model(input_ids)
        cur_logits = output.logits[:, -1, :]

        if do_sample:
            cur_logits = cur_logits / temperature
            cur_probs = F.softmax(cur_logits, dim=-1)
            cur_idx = torch.multinomial(cur_probs, num_samples=1)
        else:
            cur_idx = torch.argmax(cur_probs, dim=-1, keepdim=True)
        
        if cur_idx.item() in stop_tokens_id:
            break

        generated_tokens_id.append(cur_idx.item())

        input_ids = torch.cat((input_ids, cur_idx), dim=-1)

    return tokenizer.decode(generated_tokens_id)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device != 'cpu' and torch.cuda.is_bf16_supported() else torch.float32
    print(f'device: {device}\ndtype: {dtype}')

    model_name_or_path = 'Qwen/Qwen1.5-0.5B'

    # 加载原始模型
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype).to(device)

    dataset = SFTDataset(tokenizer, 'bio-nlp-umass/bioinstruct')

    data_loader = DataLoader(dataset, collate_fn=lambda batch : collate_fn(batch, tokenizer))

    lora.replace_linear_with_lora(model, rank=8, alpha=16, dropout=0.0)
    lora.print_trainable_parameters(model)

    lr = 5e-5
    num_epochs = 3
    logging_steps = 5
    max_grad_norm = 1.0
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    total_loss = 0.0
    total_step = 0
    for epoch in range(num_epochs):
        for step, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

            total_step +=1
            if total_step % logging_steps == 0:
                avg_loss = total_loss / total_step
                print(f"Step: {step + 1} / {len(data_loader)}, Loss: {avg_loss:.4f}, Grad Norm: {total_norm:.4f}", flush=True)
    
    lora.unload_lora(model, "adapter.pt")
    
    lora.load_lora(model, "adapter.pt")
    model.eval()
    for test_text in [
        'Describe the process of bacterial conjugation and its significance in the context of antibiotic resistance.',
        'Explain the role of insulin in the body and how insulin resistance affects blood sugar levels.',
        'Provide recommendations for lifestyle changes that can help improve the overall health of a patient with type 2 diabetes.',
    ]:
        print("="*80)
        already_print_text = ""
        for all_text in inference_streaming(model, tokenizer, test_text, do_sample=False):
            to_print = all_text.replace(already_print_text, "")
            print(to_print, end="", flush=True)
            already_print_text = all_text
        # last_text = ""
        # for cur_text in inference_streaming(model, tokenizer, test_text):
        #     to_print = cur_text.replace(last_text, "")
        #     print(to_print, end="", flush=True)
        #     last_text = cur_text
        # print("\n")

        # results = inference(model, tokenizer, test_text)
        # print(results)
        # print("\n")
    
    # lora.merge_lora(model)
    # torch.save(model, "model.pt")

if __name__ == "__main__":
    main()