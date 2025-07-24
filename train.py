
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

torch.cuda.empty_cache()
torch.cuda.ipc_collect()


MODEL_DIR          = "./qwen2_5_omni_7b_4bit"
DATA_PATH          = "/home/rag/data/COCO"
TASK_TYPE          = "T-I"          
BATCH_SIZE         = 3
NUM_NEG            = 8
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
        q_prompts,   q_imgs   = [], []
        pos_prompts, pos_imgs = [], []
        neg_prompts, neg_imgs = [], []
        

        for sample in batch:

            instr    = random.choice(INSTRUCTIONS)
            user_msg = f"{instr} {SEP} {sample['input_text'] or ''}"
            q_prompts.append(make_prompt(user_msg))
            q_imgs.append(sample["input_image"])


            pos_prompts.append(make_prompt(sample["pos_text"]))
            pos_imgs.append(sample["pos_image"])

                
            n_texts = sample['neg_texts'][:NUM_NEG]
            n_imgs  = sample['neg_images'][:NUM_NEG]
            tgt_has_text = 'T' in instr.split('-')[-1]
            if tgt_has_text:
                n_texts = sample['neg_texts'][:NUM_NEG]
            else:
                n_texts = [""] * NUM_NEG

            n_imgs = sample['neg_images'][:NUM_NEG]
            for txt in n_texts:
                neg_prompts.append(make_prompt(_to_str(txt)))
            

        q_in = proc(text=q_prompts,
                    images=q_imgs if any(q_imgs) else None,
                    padding=True, truncation=True, max_length=MAX_LEN,
                    return_tensors="pt")

        pos_in = proc(text=pos_prompts,
                      images=pos_imgs if any(pos_imgs) else None,
                      padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt")

        neg_in = proc(text=neg_prompts,
                      images=neg_imgs if any(neg_imgs) else None,
                      padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt")

        return {"query": q_in, "pos": pos_in, "neg": neg_in}
    return collate

def info_nce_sep(q, pos, neg, k, τ=TEMP):
    
    q   = F.normalize(q,   dim=-1)
    pos = F.normalize(pos, dim=-1)
    neg = F.normalize(neg, dim=-1)

    pos_sim = (pos * q).sum(-1, keepdim=True)                 # (B, 1)
    neg_sim = torch.matmul(neg, q.unsqueeze(-1)).squeeze(-1)  # (B, k)
    logits  = torch.cat([pos_sim, neg_sim], dim=1)            # (B, k+1)

    loss = F.cross_entropy(logits / τ, torch.zeros(q.size(0), dtype=torch.long, device=q.device))
    return loss, logits

# def retrieval_at_k(logits, k=5):
#     topk = logits.topk(k, dim=1).indices
#     hit  = (topk == 0).any(dim=1).float()
#     return hit.mean(), hit.mean(), hit.mean()  

def eos(hidden, ids, pad_id):
    idx = (ids != pad_id).sum(-1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]


def recall_f1_at5(sims):
    top5 = sims.topk(5, dim=1).indices
    hit  = (top5 == 0).any(dim=1).float()
    prec = hit / 5.0
    rec  = hit
    f1   = torch.where(hit.bool(), 2*prec*rec/(prec+rec), torch.zeros_like(hit))
    return prec.mean(), rec.mean(), f1.mean()


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
        lora_cfg = LoraConfig(r=3, lora_alpha=16, lora_dropout=0.05,
                              bias="none", task_type="CAUSAL_LM",
                              target_modules=["q_proj","k_proj","v_proj","o_proj",
                                              "gate_proj","up_proj","down_proj"])
        self.model = get_peft_model(self.model, lora_cfg)
        assert any("lora" in n for n, p in self.model.named_parameters() if p.requires_grad), "LoRA layers were not attached — check target_modules list!"

        self.pad = self.proc.tokenizer.pad_token_id
        self.model.train()

        full_loader = loader.create_multimodal_dataloader(
            DATA_PATH, TASK_TYPE, BATCH_SIZE, shuffle=True,
            num_negatives=NUM_NEG, dataset_percents = 0.001
)
        val_len = max(1, int(0.1 * len(full_loader.dataset)))
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
        out = self.model(**pack, output_hidden_states=False, use_cache=False, return_dict=True)
        return eos(out.last_hidden_state, pack["input_ids"], self.pad)

    def _epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        torch.set_grad_enabled(train)

        tot_loss = tot_p = tot_r = tot_f1 = 0
        for batch in loader:
            batch = {k:{kk:v.to(DEVICE) for kk,v in d.items()} for k,d in batch.items()}
            q   = self._embed(batch["query"])
            pos = self._embed(batch["pos"])
            neg = self._embed(batch["neg"]).view(q.size(0), NUM_NEG, -1)

            loss, logits = info_nce_sep(q, pos, neg, NUM_NEG)
            prec, rec, f1 = recall_f1_at5(logits)
            print(f"loss: {loss:.4f}")

            if train:
                self.opt.zero_grad(); loss.backward()
                self.gpu_report("after backward")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.opt.step(); self.sched.step()
                #torch.cuda.empty_cache()

            tot_loss += loss.item(); tot_p += prec; tot_r += rec; tot_f1 += f1

        n = len(loader)
        return tot_loss/n, (tot_p/n).item(), (tot_r/n).item(), (tot_f1/n).item()

    def fit(self):
        for ep in range(1, EPOCHS+1):
            tr_loss, tr_p, tr_r, tr_f1 = self._epoch(self.tr_loader, True)
            log = {"epoch": ep, "loss": tr_loss,
                   "P@5": tr_p, "R@5": tr_r, "F1@5": tr_f1,
                   "lr": self.sched.get_last_lr()[0]}
            

            if ep % VAL_EVERY == 0:
                with torch.no_grad():
                    vl_loss, vl_p, vl_r, vl_f1 = self._epoch(self.val_loader, False)
                log.update({'val_loss' :vl_loss, 'val_recall5' :vl_r, 'val_f1_5': vl_f1})

            if ep % SAVE_EVERY == 0:
                self.model.save_pretrained(f"lora_{TASK_TYPE}_ep{ep}")
                print("AAA")
            
    @staticmethod      
    def gpu_report(tag):
        free, total = torch.cuda.mem_get_info(0)
        used = total - free
        print(f"{tag} | used {humanize.naturalsize(used)} / {humanize.naturalsize(total)}")

if __name__ == "__main__":
    torch.cuda.set_device(0)
    Trainer().fit(); #wandb.finish()
