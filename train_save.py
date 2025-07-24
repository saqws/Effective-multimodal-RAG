# -*- coding: utf-8 -*-


import os, math, random
from typing import List, Dict
from torch.optim import AdamW
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig, get_peft_model,
    prepare_model_for_kbit_training   
)
import wandb
import logging
import humanize, psutil
logging.getLogger().setLevel(logging.ERROR)


MODEL_DIR          = "./qwen2_5_omni_7b_4bit"
DATA_PATH          = "/home/rag/COCO"
TASK_TYPE          = "T-I"          
BATCH_SIZE         = 1
NUM_NEG            = 4
MAX_LEN            = 512
LR                 = 2e-4
EPOCHS             = 1
TEMP               = 0.07
VAL_EVERY_EPOCHS   = 2
SAVE_EVERY_EPOCHS  = 5
DEVICE             = "cuda"

SYSTEM_MSG = "You are Qwen, a multilingual multimodal assistant."
SEP        = "[SEP]"

INSTRUCTIONS = [
    "Given a web search query, retrieve relevant passages that answer the query.",
    "Given a question, retrieve questions that are semantically equivalentto the given question.",
    "Given a financial question, retrieve user replies that best answer the question."
]

import loader
# from multimodal_dataset import create_multimodal_dataloader  

def build_collate(proc: Qwen2_5OmniProcessor):
    def make_prompt(text: str) -> str:
        conv = [
            {"role": "system",
            "content": [{"type": "text", "text": SYSTEM_MSG}]},
            {"role": "user",
            "content": [{"type": "text", "text": text}]},
        ]
        prompt = proc.apply_chat_template(
            conv,
            add_generation_prompt=False,
            tokenize=False,
        )
        if isinstance(prompt, list):
            prompt = "".join(prompt)
            return str(prompt)


    def _to_str(x):
        if isinstance(x, list):
            return "".join(x)
        return str(x)

    def collate(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        q_prompts, q_imgs = [], []
        cmp_prompts, cmp_imgs = [], []

        for sample in batch:
            instr = random.choice(INSTRUCTIONS)
            user_msg = f"{instr} {SEP} {sample['input_text'] or ''}"

            q_prompts.append(make_prompt(user_msg))
            q_imgs.append(sample['input_image'])

            c_texts = [sample['pos_text']] + sample['neg_texts'][:NUM_NEG]
            c_imgs  = [sample['pos_image']] + sample['neg_images'][:NUM_NEG]

            for txt in c_texts:
                cmp_prompts.append(make_prompt(_to_str(txt)))
            cmp_imgs.extend(c_imgs)

        q_in = proc(
            text=q_prompts,
            images=q_imgs if any(q_imgs) else None,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        cmp_in = proc(
            text=cmp_prompts,
            images=cmp_imgs if any(cmp_imgs) else None,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {"query": q_in, "cmp": cmp_in}

    return collate

def info_nce(q, cmp_, k, τ=TEMP):
    B = q.size(0)
    q   = F.normalize(q, dim=-1)
    cmp = F.normalize(cmp_.view(B, k+1, -1), dim=-1)

    sims = torch.matmul(cmp, q.unsqueeze(-1)).squeeze(-1)       # [B, k+1]
    sims = torch.nan_to_num(sims)                               
    loss = F.cross_entropy(sims / τ, torch.zeros(B, dtype=torch.long, device=q.device))
    return loss, sims

def recall_f1_at5(sims):
    top5 = sims.topk(5, dim=1).indices
    hit  = (top5 == 0).any(dim=1).float()
    prec = hit / 5.0
    rec  = hit
    f1   = torch.where(hit.bool(), 2*prec*rec/(prec+rec), torch.zeros_like(hit))
    return prec.mean(), rec.mean(), f1.mean()

def eos(hid, ids, pad):
    idx = (ids != pad).sum(-1) - 1
    return hid[torch.arange(hid.size(0), device=hid.device), idx]

class Trainer:
    def __init__(self):
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4")
        self.proc = Qwen2_5OmniProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            MODEL_DIR, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.float16, local_files_only=True)
        self.model = prepare_model_for_kbit_training(self.model,use_gradient_checkpointing=True)
        lora_cfg = LoraConfig(r=3, lora_alpha=10, lora_dropout=0.05,
                              bias="none", task_type="CAUSAL_LM",
                              target_modules=["q_proj","k_proj","v_proj","o_proj",
                                              "gate_proj","up_proj","down_proj"])
        self.model = get_peft_model(self.model, lora_cfg)
        assert any("lora" in n for n, p in self.model.named_parameters()
                   if p.requires_grad), \
            "LoRA layers were not attached — check target_modules list!"
        self.pad = self.proc.tokenizer.pad_token_id
        self.model.train()

        full_loader = loader.create_multimodal_dataloader(
            DATA_PATH, TASK_TYPE, BATCH_SIZE, shuffle=True,
            num_negatives=NUM_NEG, dataset_percents = 0.01
)
        val_len = max(1, int(0.05 * len(full_loader.dataset)))
        train_len = len(full_loader.dataset) - val_len
        ds_tr, ds_val = torch.utils.data.random_split(full_loader.dataset,
                                                      [train_len, val_len])
        collate = build_collate(self.proc)
        self.tr_loader = DataLoader(ds_tr,  BATCH_SIZE, shuffle=True,
                                    num_workers=4, collate_fn=collate)
        self.val_loader= DataLoader(ds_val, BATCH_SIZE, shuffle=False,
                                    num_workers=4, collate_fn=collate)

        self.opt = AdamW(self.model.parameters(), lr=LR)
        steps = len(self.tr_loader)*EPOCHS
        self.sched = get_linear_schedule_with_warmup(self.opt, steps//20, steps)

        # wandb.init(project="gme-qwen-omni7b",
        #            name=f"{TASK_TYPE}_LoRA", config=dict(bs=BATCH_SIZE, k=NUM_NEG))

    def _embed(self, pack):
        out = self.model(**pack, output_hidden_states=True,
                         use_cache=False, return_dict=True)
        return eos(out.hidden_states[-1], pack["input_ids"], self.pad)

    def _epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        torch.set_grad_enabled(train)

        tot_loss = tot_p = tot_r = tot_f1 = 0
        for batch in loader:
            batch = {k:{kk:vv.to(DEVICE) for kk,vv in v.items()} for k,v in batch.items()}
            q   = self._embed(batch["query"])
            cmp = self._embed(batch["cmp"])
            loss, sims = info_nce(q, cmp, NUM_NEG)
            p, r, f1 = recall_f1_at5(sims)

            if train:
                self.opt.zero_grad(); loss.backward()
                self.gpu_report("after backward")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.opt.step(); self.sched.step()
                torch.cuda.empty_cache()
                self.gpu_report("after empty_cache")

            tot_loss += loss.item(); tot_p += p; tot_r += r; tot_f1 += f1
        n = len(loader)
        return tot_loss/n, (tot_p/n).item(), (tot_r/n).item(), (tot_f1/n).item()

    def fit(self):
        for ep in range(1, EPOCHS+1):
            tr_loss, tr_p, tr_r, tr_f1 = self._epoch(self.tr_loader, True)
            log = {"epoch": ep, "loss": tr_loss,
                   "P@5": tr_p, "R@5": tr_r, "F1@5": tr_f1,
                   "lr": self.sched.get_last_lr()[0]}

            if ep % VAL_EVERY_EPOCHS == 0:
                with torch.no_grad():
                    vl_loss, vl_p, vl_r, vl_f1 = self._epoch(self.val_loader, False)
                    torch.cuda.empty_cache()
                log.update({"val_loss": vl_loss,
                            "val_P@5": vl_p, "val_R@5": vl_r, "val_F1@5": vl_f1})
            # wandb.log(log)

            if ep % SAVE_EVERY_EPOCHS == 0:
                self.model.save_pretrained(f"lora_{TASK_TYPE}_ep{ep}")
            
    @staticmethod      
    def gpu_report(tag):
        free, total = torch.cuda.mem_get_info(0)
        used = total - free
        print(f"{tag} | used {humanize.naturalsize(used)} / {humanize.naturalsize(total)}")

if __name__ == "__main__":
    torch.cuda.set_device(0)
    Trainer().fit(); #wandb.finish()
