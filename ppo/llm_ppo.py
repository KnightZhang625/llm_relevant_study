# coding:utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel
from dataclasses import dataclass, fields

@dataclass
class Config:
    rollout_bsz: int = 8                 # bsz for generating samples
    micro_rollout_bsz: int = 4           # micro_bsz for generating samples due to heavy computation
    rollout_n: int = 2                   # for each prompt, generate how many datas

    max_length: int = 500               # maximize length for the input
    max_actions: int = 300              # maximize generated tokens

    game_nums: int = 10
    train_bsz: int = 8
    max_epochs: int = 5               # how many rounds we will use the old experience

    kl_ctl: float = 0.5
    reward_clip: float = 1.0

    gamma: float = 0.95
    gae_lambda: float = 0.95

    policy_ratio_clip = 0.2
    value_clip = 0.2

def compute_approx_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    # log_probs, ref_log_probs: [bsz, max_act]
    log_ratio = log_probs.float() - ref_log_probs.float()
    log_ratio = log_ratio * action_mask
    return log_ratio

def combine_rewards_with_kl(kl, rewards, action_mask, kl_ctl=0.5, reward_clip=1.0) -> torch.Tensor:
    kl = -kl_ctl * kl               # [bsz, max_act]
    ends = action_mask.sum(1) - 1   # [bsz]
    reward_clip = torch.tensor(reward_clip).to(kl.device)
    
    rewards = torch.clamp(rewards, -reward_clip, reward_clip)   # [bsz, 1]
    for i in range(kl.size(0)):
        kl[i, ends[i].item()] = kl[i, ends[i]] + rewards[i, 0]

    return kl

def get_advantages_and_returns(values, rewards, action_mask, gamma=0.95, gae_lambda=0.95):
    
    reversed_advantages = []
    last_gae = 0.0

    values = values * action_mask
    rewards = rewards * action_mask

    for i in reversed(range(values.size(1))):
        next_val = values[:, i + 1] if i < values.size(1) - 1 else 0.0
        delta = rewards[:, i] + gamma * next_val - values[:, i]
        last_gae = delta + gamma * gae_lambda * last_gae
        reversed_advantages.append(last_gae)
    
    advantages = torch.stack(reversed_advantages[::-1], dim=1)  # [bsz, max_act]
    returns = advantages + values

    # If you don’t detach the advantage, then during backprop, the gradients will flow into the critic model,
    # the advantages will be used to calculate the policy loss, and through above, we can find that advantages involves using value from critic model.
    return advantages.detach(), returns

def compute_policy_loss(new_log_probs, old_log_probs, advantages, action_mask, policy_ratio_clip):

    normalized_advantages = (advantages - advantages.mean(-1, keepdim=True)) / (advantages.std(-1, keepdim=True) + 1e-8)
    ratio = (new_log_probs - old_log_probs).exp()       # [bsz, max_actions]
    left_term = ratio * normalized_advantages                      # [bsz, max_actions]
    right_term = torch.clamp(ratio, 1 - policy_ratio_clip, 1 + policy_ratio_clip) * normalized_advantages  # [bsz, max_actions]

    loss = - torch.min(left_term, right_term)   # [bsz, max_actions]
    loss = loss * action_mask
    return (loss.sum(-1) / action_mask.sum(-1)).mean()
    
def compute_value_loss(new_values, old_values, returns, action_mask, value_clip):
    left_term = (new_values - returns) ** 2
    
    clipped_values = old_values + torch.clamp(new_values - old_values, -value_clip, value_clip)
    right_term = (clipped_values - returns) ** 2

    loss = torch.max(left_term, right_term)   # [bsz, max_actions]
    loss = loss * action_mask

    return (loss.sum(-1) / action_mask.sum(-1)).mean()

@dataclass
class SamplesBatch:
    generated_seqs_ids: torch.Tensor
    attention_mask: torch.LongTensor
    action_mask: torch.LongTensor
    num_actions: int
    action_length: torch.Tensor     # length for each data action
    total_length: torch.Tensor      # length for each data (prompt + action)

@dataclass
class ExperienceBatch:
    generated_seqs_ids: torch.Tensor
    num_actions: int                # used for remove all prompts, only keeps actions, despite some data in the batch are padded.

    action_log_probs: torch.Tensor
    values: torch.Tensor

    returns: torch.Tensor
    advantages: torch.Tensor

    attention_mask: torch.Tensor
    action_mask: torch.Tensor

    def __getitem__(self, key):
        return getattr(self, key)

class Buffer(Dataset):
    def __init__(self, limit):
        self.limit = limit
        self.buffer: list[dict] = []
    
    def add_buffer(self, experiences: list[ExperienceBatch]):
        attr_names = [p.name for p in fields(experiences[0])]

        for experience in experiences:
            generated_seq_ids = getattr(experience, attr_names[0])
            num_actions = getattr(experience, attr_names[1])
            action_log_probs = getattr(experience, attr_names[2])
            values = getattr(experience, attr_names[3])
            returns = getattr(experience, attr_names[4])
            advantages = getattr(experience, attr_names[5])
            attention_mask = getattr(experience, attr_names[6])
            action_mask = getattr(experience, attr_names[7])

            for i in range(generated_seq_ids.size(0)):
                per_data = {
                   "generated_seq_ids": generated_seq_ids[i, ...],  # [max_len]
                   "num_actions": num_actions,                      # int
                   "action_log_probs": action_log_probs[i, ...],    # [max_actions]
                   "values": values[i, ...],                        # [max_actions]
                   "returns": returns[i, ...],                      # [max_actions]
                   "advantages": advantages[i, ...],                # [max_actions]
                   "attention_mask": attention_mask[i, ...],        # [max_len]
                   "action_mask": action_mask[i, ...],              # [max_len]
                }
                self.buffer.append(per_data)
                if len(self.buffer) > self.limit:
                    self.buffer = self.buffer[-self.limit:]
    
    def clear(self):
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

def buffer_collate_fn(batch) -> ExperienceBatch:

    generated_seq_ids = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []
    for instance in batch:
        generated_seq_ids.append(instance["generated_seq_ids"])
        action_log_probs.append(instance["action_log_probs"])
        values.append(instance["values"])
        returns.append(instance["returns"])
        advantages.append(instance["advantages"])
        attention_mask.append(instance["attention_mask"])
        action_mask.append(instance["action_mask"])

    return ExperienceBatch(
        generated_seqs_ids=torch.stack(generated_seq_ids, dim=0),   # [bsz, max_len]
        num_actions=batch[0]["num_actions"],                        # int
        action_log_probs=torch.stack(action_log_probs, dim=0),      # [bsz, max_actions]
        values=torch.stack(values, dim=0),                          # [bsz, max_actions]
        returns=torch.stack(returns, dim=0),                        # [bsz, max_actions]
        advantages=torch.stack(advantages, dim=0),                  # [bsz, max_actions]
        attention_mask=torch.stack(attention_mask, dim=0),          # [bsz, max_len]
        action_mask=torch.stack(action_mask, dim=0),                # [bsz, max_actions]
    )


class CriticModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        self.head = nn.Linear(backbone.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, num_actions):
        hidden_state = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        value_model_output = self.head(hidden_state)                # [bsz, seq, 1]
        value = value_model_output.squeeze(-1)[:, -num_actions:]    # [bsz, n_act]
        return value

class PromptsDataset(Dataset):

    def __init__(self, tokenizer: AutoTokenizer, prompts):
        self.datas = []

        for prompt in prompts:
            content = [{"role": "user", "content": prompt}]
            self.datas.append(tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True))
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]

class Agent:
    def __init__(self, 
                 actor_model: AutoModelForCausalLM,
                 actor_tokenizer: AutoTokenizer,
                 critic_model: AutoModel, 
                 reward_model: AutoModelForSequenceClassification, 
                 reward_tokenizer: AutoTokenizer,
                 ref_model: AutoModelForCausalLM,
                 prompts: list[str],
                 config: Config):

        self.actor_model = actor_model
        self.actor_tokenizer = actor_tokenizer
        self.actor_tokenizer.padding_side = "left"
        self.critic_model = critic_model
        self.reward_model = reward_model
        self.reward_model.eval()
        self.reward_tokenizer = reward_tokenizer
        self.ref_model = ref_model
        self.ref_model.eval()

        self.optimizer_actor = torch.optim.Adam(self.actor_model.parameters(), lr=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic_model .parameters(), lr=1e-5)

        self.pad_token_id = self.actor_tokenizer.pad_token_id   # 151643
        self.eos_token_id = self.actor_tokenizer.eos_token_id   # 151645

        self.config = config
        self.device = self.actor_model.device

        promtps_dataset = PromptsDataset(tokenizer=self.actor_tokenizer, prompts=prompts)
        self.prompts_loader = DataLoader(promtps_dataset, batch_size=self.config.rollout_bsz, shuffle=True)

    def generate_samples(self, prompts: DataLoader) -> list[SamplesBatch]:
        expanded_prompts: list[str] = sum([[p] * self.config.rollout_n for p in prompts], [])     # [rollout_bsz * rollout_n]
        self.actor_model.eval()
        
        samples_list: list[SamplesBatch] = []
        for s_idx in range(0, len(expanded_prompts), self.config.micro_rollout_bsz):
            selected_prompts = expanded_prompts[s_idx : s_idx + self.config.micro_rollout_bsz]
            tokenized_prompts = self.actor_tokenizer(selected_prompts, padding='max_length', max_length=self.config.max_length, truncation=True, return_tensors="pt")   # [bsz, max_length]
            generated_seqs_ids = self.actor_model.generate(
                **tokenized_prompts.to(self.device),
                max_new_tokens=self.config.max_actions,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )   # [bsz, max_length + max_actions] -> [pad_id, prompts, ans, eos_id]

            if generated_seqs_ids.size(1) >= self.config.max_length + self.config.max_actions:
                generated_seqs_ids = generated_seqs_ids[:, :self.config.max_length + self.config.max_actions]
            else:
                # [bsz, max_length - max_actions - generated_seqs.size(1)]
                to_pad_tensor = torch.full([generated_seqs_ids.size(0), self.config.max_length + self.config.max_actions - generated_seqs_ids.size(1)], fill_value=self.pad_token_id, device=self.device)
                generated_seqs_ids = torch.cat([generated_seqs_ids, to_pad_tensor], dim=1)
            
            attention_mask = generated_seqs_ids.ne(self.pad_token_id).to(dtype=torch.long)  # [bsz, max_len]
            action = generated_seqs_ids[:, tokenized_prompts["input_ids"].size(1):]         # [bsz, max_actions]
            action_mask = action.ne(self.pad_token_id).to(torch.long)                       # [bsz, max_actions]
            
            samples_list.append(
                SamplesBatch(
                    generated_seqs_ids=generated_seqs_ids,              # [bsz, max_len]
                    attention_mask=attention_mask,                      # [bsz, max_len]
                    action_mask=action_mask,                            # [bsz, max_actions]
                    num_actions=action_mask.size(1),                    # max_actions
                    action_length=action_mask.float().sum(dim=-1),      # [bsz]
                    total_length=attention_mask.float().sum(dim=-1),    # [bsz]
                )
            )
        
        return samples_list
    
    def generate_experiences(self, sample_list: list[SamplesBatch]):
        self.actor_model.eval()
        self.critic_model.eval()

        experiences = []
        for sample in sample_list:
            generated_seqs_ids = sample.generated_seqs_ids      # [bsz, max_len]
            attention_mask = sample.attention_mask              # [bsz, max_len]
            action_mask = sample.action_mask                    # [bsz, max_actions]
            num_actions = sample.num_actions                    # max_actions

            with torch.no_grad():
                # 1. old policy model log_prob
                output = self.actor_model(generated_seqs_ids, attention_mask=attention_mask).logits    # [bsz, max_len, V]
                # log_probs: B C e
                # logits:    A B C e
                log_probs = F.log_softmax(output[:, :-1, :], dim=-1)   # [bsz, max_len - 1, V]
                log_probs_labels = torch.gather(log_probs, dim=-1, index=generated_seqs_ids[:, 1:].unsqueeze(-1))   # [bsz, max_len - 1, 1]
                action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]   # [bsz, max_act]

                # 2. ref model log_prob
                ref_output = self.ref_model(generated_seqs_ids, attention_mask=attention_mask).logits
                ref_log_probs = F.log_softmax(ref_output[:, :-1, :], dim=-1)
                ref_log_probs_labels = torch.gather(ref_log_probs, dim=-1, index=generated_seqs_ids[:, 1:].unsqueeze(-1))
                ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]

                kl_divergence = compute_approx_kl(log_probs=action_log_probs, ref_log_probs=ref_action_log_probs, action_mask=action_mask)  # [bsz, max_act]

                # 3. value
                values = self.critic_model(generated_seqs_ids, attention_mask, num_actions)  # [bsz, max_act]

                # 4. reward
                generated_seqs_tokens = self.actor_tokenizer.batch_decode(generated_seqs_ids, skip_sepcial_tokens=True)
                generated_seqs_tokens = [seqs.replace(self.actor_tokenizer.pad_token, "") for seqs in generated_seqs_tokens]
                tokenized_generated_seqs_tokens = self.reward_tokenizer(generated_seqs_tokens, return_tensors="pt", padding=True)
            
                # rewards = []
                # for seqs in generated_seqs_tokens:
                #     if re.search(r"equals 2|two", seqs):
                #         rewards.append(torch.tensor(1.0))
                #     else:
                #         rewards.append(torch.tensor(0.0))
                # rewards = torch.stack(rewards, dim=0).unsqueeze(-1).to(values.device)
                # for i, seqs in enumerate(generated_seqs_tokens):
                #     print(seqs[-num_actions:])
                #     print(rewards[i])
                # print()
                rewards = self.reward_model(**tokenized_generated_seqs_tokens.to(self.device)).logits   # [bsz, 1]
            
                final_rewards = combine_rewards_with_kl(kl_divergence, rewards, action_mask, kl_ctl=self.config.kl_ctl, reward_clip=self.config.reward_clip)    # [bsz, max_act]

                # 5. get advantages: [bsz, max_act], and returns: [bsz, max_act]
                advantages, returns = get_advantages_and_returns(values, final_rewards, action_mask, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)

            experiences.append(
                ExperienceBatch(
                    generated_seqs_ids=generated_seqs_ids,  # [bsz, max_len]
                    num_actions=num_actions,                # [max_actions]
                    action_log_probs=action_log_probs,      # [bsz, max_actions]
                    values=values,                          # [bsz, max_actions]
                    returns=returns,                        # [bsz, max_actions],
                    advantages=advantages,                  # [bsz, max_actions],
                    attention_mask=attention_mask,          # [bsz, max_len]
                    action_mask=action_mask,                # [bsz, max_actions]
                )
            )
        
        return experiences  # [rollout_bsz * rollout_n]

    def train(self):
        buffer = Buffer(limit=100)
        steps = 0
        for eps in range(self.config.game_nums):
            for prompts in self.prompts_loader:
                samples_list: list[SamplesBatch] = self.generate_samples(prompts)
                experiences: list[ExperienceBatch] = self.generate_experiences(samples_list)
                buffer.add_buffer(experiences)
                experiences_loader = DataLoader(buffer, batch_size=self.config.train_bsz, shuffle=True, collate_fn=buffer_collate_fn)
                for epoch in range(self.config.max_epochs):
                    for batch_exp in experiences_loader:
                        self.train_one_step(batch_exp, steps)
                        steps +=1
                
                buffer.clear()
                torch.cuda.empty_cache()
    
    def train_one_step(self, batch, steps):
        """
            generated_seqs_ids=torch.stack(generated_seq_ids, dim=0),   # [bsz, max_len]
            num_actions=batch[0]["num_actions"],                        # int
            action_log_probs=torch.stack(action_log_probs, dim=0),      # [bsz, max_actions]
            values=torch.stack(values, dim=0),                          # [bsz, max_actions]
            returns=torch.stack(returns, dim=0),                        # [bsz, max_actions]
            advantages=torch.stack(advantages, dim=0),                  # [bsz, max_actions]
            attention_mask=torch.stack(attention_mask, dim=0),          # [bsz, max_len]
            action_mask=torch.stack(action_mask, dim=0),                # [bsz, max_actions]
        """
        
        self.actor_model.train()
        self.critic_model.train()

        generated_seqs_ids = batch["generated_seqs_ids"]
        num_actions = batch["num_actions"]
        old_action_log_probs = batch["action_log_probs"]
        old_values = batch["values"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        attention_mask = batch["attention_mask"]
        action_mask = batch["action_mask"]

        new_logits = self.actor_model(generated_seqs_ids, attention_mask=attention_mask).logits # [bsz, max_len, V]
        new_log_probs = F.log_softmax(new_logits, dim=-1)
        new_tokens_log_probs = torch.gather(new_log_probs[:, :-1, :], dim=-1, index=generated_seqs_ids[:, 1:].unsqueeze(-1))  # [bsz, max_len, 1]
        new_action_log_probs = new_tokens_log_probs.squeeze(-1)[:, -num_actions:]   # [bsz, max_len]
        policy_loss = compute_policy_loss(
            new_log_probs=new_action_log_probs,
            old_log_probs=old_action_log_probs,
            advantages=advantages,
            action_mask=action_mask,
            policy_ratio_clip=self.config.policy_ratio_clip,
        )
        
        # input_ids, attention_mask, num_actions
        new_values = self.critic_model(
            input_ids=generated_seqs_ids,
            attention_mask=attention_mask,
            num_actions=num_actions,
        )   # [bsz, max_actions]
        value_loss = compute_value_loss(
            new_values=new_values,
            old_values=old_values,
            returns=returns,
            action_mask=action_mask,
            value_clip=self.config.value_clip,
        )

        loss = policy_loss + value_loss
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

        print(f"Step: {steps}, Policy loss: {policy_loss.item():.4f} Critic loss: {value_loss.item():.4f}")


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # 策略模型
    actor_model = AutoModelForCausalLM.from_pretrained('/nfs1/jiaxinzhang/models/Qwen2.5-0.5B-Instruct').to(device)
    actor_tokenizer = AutoTokenizer.from_pretrained('/nfs1/jiaxinzhang/models/Qwen2.5-0.5B-Instruct')
    # 参考模型
    ref_model = AutoModelForCausalLM.from_pretrained('/nfs1/jiaxinzhang/models/Qwen2.5-0.5B-Instruct').to(device)
    # 奖励模型
    reward_model = AutoModelForSequenceClassification.from_pretrained('/nfs1/jiaxinzhang/models/reward-model-deberta-v3-large-v2').to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained('/nfs1/jiaxinzhang/models/reward-model-deberta-v3-large-v2')
    # 价值模型
    critic_model = CriticModel(actor_model.base_model).to(device)

    prompt_list = [
        'What is 1 + 1?',
        'In PowerShell, how can I tell if virtualization is disabled in the BIOS?',
        'Why do people prefer swimming in aquariums over swimming pools?',
        'You are a marketing expert. Write 30 Instagram Reels scripts with marketing tips.',
        'You are a marketing expert. Write 30 Instagram Reels scripts with marketing tips.',
        'You are a marketing expert. Write 30 Instagram Reels scripts with marketing tips.',
        'Why are all mirrors rectangular?',
        'In the roots of an infected plant, which can we find — ozone or gold?'
    ]

    content = [{"role": "user", "content": "You are a marketing expert. Write 30 Instagram Reels scripts with marketing tips."}]
    test_1 = actor_tokenizer.apply_chat_template(content, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    out_before_train = actor_model.generate(
        test_1.to(device), 
        max_new_tokens = 100, 
        eos_token_id = actor_tokenizer.eos_token_id, 
        pad_token_id = actor_tokenizer.pad_token_id,
    )
    print(actor_tokenizer.batch_decode(out_before_train, skip_special_tokens=True))

    config = Config()
    agent = Agent(
        actor_model=actor_model,
        actor_tokenizer=actor_tokenizer,
        critic_model=critic_model,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        ref_model=ref_model,
        prompts=prompt_list,
        config=config,
    )
    agent.train()

    out_after_train = agent.actor_model.generate(
        test_1.to(device), 
        max_new_tokens = 100, 
        eos_token_id = actor_tokenizer.eos_token_id, 
        pad_token_id = actor_tokenizer.pad_token_id,
    )
    print(actor_tokenizer.batch_decode(out_after_train, skip_special_tokens=True))