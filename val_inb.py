import os, math, random, json, logging, argparse, time
from typing import List, Dict
from torch.optim import AdamW
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import (
    LoraConfig, get_peft_model,
    prepare_model_for_kbit_training, PeftModel  
)
import wandb
import logging
import humanize, psutil
logging.getLogger().setLevel(logging.ERROR)
import json
import new_dataloader          
import sys 

torch.cuda.empty_cache()
torch.cuda.ipc_collect()


parser = argparse.ArgumentParser()
parser.add_argument("--lora_ckpt", type=str, default=None,
                    help="путь к существующему LoRA‑чек‑пойнту (ep3/ и т. д.)")
args = parser.parse_args()

MODEL_DIR          = "./qwen2_5_omni_7b_4bit"
DATA_PATH          = "/home/rag/data/COCO"
TASK_TYPE          = "T-I"          
BATCH_SIZE         = 21
NUM_NEG            = 8
MAX_LEN            = 512
LR                 = 2e-4
EPOCHS             = 12
TEMP               = 0.03
TRAIN_LOG_EVERY    = 0.3
VAL_EVERY_EPOCHS   = 3
SAVE_EVERY_EPOCHS  = 3
DEVICE             = "cuda"
TOP_KS             = (1, 5, 10)
VAL_CHUNK          = 512

SYSTEM_MSG = "You are Qwen, a multilingual multimodal assistant."
SEP        = "[SEP]"

base_dir = "/home/rag/model"
lora_dir = os.path.join(base_dir, f"lora_checkpoints/{TASK_TYPE}")
log_dir = os.path.join(base_dir, f"logs/{TASK_TYPE}")

os.makedirs(lora_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32  = True

INSTRUCTION_SETS = {
    "T-IT": [
        "Find an image that matches the given caption.",
        "Identify the image showcasing the described everyday scene.",
        "Find an image and text describing this text."
    ],
    "IT-T": [
        "Retrieve a fact-based paragraph that provides an answer to the given query about the image.",
        "Retrieve paragraph that provide an answer to the question alongside the image.",
        "Provide a specific decription of the image along with the following question."
    ],
    "T-I": [
        "Find an image that matches the given caption.",
        "Based on the following fashion description, retrieve the best matching image.",
        "Identify the image showcasing the described everyday scene."
    ],
    "I-T": [
        "Find an image caption describing the following image.",
        "Find a caption in the given photo.",
        "Find a description for given photo."
    ],
    "IT-IT": [
        "Obtain illustrated documents that correspond to the inquiry alongside the provided image.",
        "Retrieve an image-description pair that provides evidence for the question of this image.",
        "Fetch illustrated documents relevant to the inquiry and corresponding to the provided image. Acquire an image-description pair that offers evidence for the question associated with this image."
    ],
    "T-T": [
        "Given a claim, retrieve documents that support or refute the claim.",
        "Given a query, retrieve relevant passages that answer the query.",
        "Given a query, retrieve relevant entity descriptions."
    ]
}

INSTRUCTIONS = INSTRUCTION_SETS.get(TASK_TYPE, ["Answer the question."])

# import loader
# from multimodal_dataset import create_multimodal_dataloader  

def build_collate(proc: Qwen2_5OmniProcessor):

    def make_prompt(text: str) -> str:
        conv = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
            {"role": "user",   "content": [{"type": "text", "text": text}]},
        ]
        prompt = proc.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
        return "".join(prompt) if isinstance(prompt, list) else str(prompt)

    def collate(batch):
        q_prompts, q_imgs = [], []
        p_prompts, p_imgs = [], []

        for sample in batch:
            instr = random.choice(INSTRUCTIONS)
            user_msg = f"{instr} {SEP} {sample.get('input_text','')}"
            q_prompts.append(make_prompt(user_msg))
            q_imgs.append(sample["input_image"])

            p_prompts.append(make_prompt(sample["pos_text"]))
            p_imgs.append(sample["pos_image"])

        q_in = proc(text=q_prompts,
                    images=q_imgs if any(q_imgs) else None, 
                    padding="longest", truncation=True, max_length=MAX_LEN,
                    return_tensors="pt")

        p_in = proc(text=p_prompts,
                    images=p_imgs if any(p_imgs) else None,
                    padding="longest", truncation=True, max_length=MAX_LEN,
                    return_tensors="pt")

        return {"query": q_in, "pos": p_in}

    return collate


def info_nce_inbatch(q, p, τ=TEMP):
    q = F.normalize(q, dim=-1)          # (B, D)
    p = F.normalize(p, dim=-1)
    logits = (q @ p.T) / τ             # (B, B)  ← все NEG тут
    labels = torch.arange(q.size(0), device=q.device)
    loss = F.cross_entropy(logits, labels)
    return loss, logits




def eos(hidden, ids, pad_id):
    idx = (ids != pad_id).sum(-1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]


# def recall_f1_at5(sims):
#     top5 = sims.topk(5, dim=1).indices
#     hit  = (top5 == 0).any(dim=1).float()
#     prec = hit / 5.0
#     rec  = hit
#     f1   = torch.where(hit.bool(), 2*prec*rec/(prec+rec), torch.zeros_like(hit))
#     return prec.mean(), rec.mean(), f1.mean()

class Trainer:
    def __init__(self):
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4")
        self.proc = Qwen2_5OmniProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        self.proc.image_processor.size        = {"height": 224, "width": 224}
        self.proc.image_processor.max_pixels  = 224 * 224          # 50 176
        self.proc.image_processor.min_pixels  = 1                  # защитное
        self.proc.image_processor.longest_edge = 224               # для косых вызовов
        self.proc.image_processor.shortest_edge = 224
        self.proc.image_processor.do_resize   = True               # убедимся, что ресайз включён
        self.proc.image_processor.resample    = 3 
        
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            MODEL_DIR, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.float16, local_files_only=True)
        
        self.model = prepare_model_for_kbit_training(self.model,use_gradient_checkpointing=True)
        
        lora_cfg = LoraConfig(r=12, lora_alpha=36, lora_dropout=0.05,
                              bias="none", task_type="CAUSAL_LM",
                              target_modules=["q_proj","k_proj","v_proj","o_proj",
                                              "gate_proj","up_proj","down_proj"])
        
        if args.lora_ckpt and os.path.isdir(args.lora_ckpt):
            self.model = PeftModel.from_pretrained(self.model, args.lora_ckpt,is_trainable=True)
            print(f"LoRA‑адаптер загружен из {args.lora_ckpt}")
        else:
            self.model = get_peft_model(self.model, lora_cfg)
            print("Инициализирован новый LoRA‑адаптер")
        assert any("lora" in n for n, p in self.model.named_parameters() if p.requires_grad), "LoRA layers were not attached — check target_modules list!"

        for name, param in self.model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        
        self.pad = self.proc.tokenizer.pad_token_id
        self.model.train()

        full_loader = new_dataloader.create_multimodal_dataloader(
            DATA_PATH, TASK_TYPE, BATCH_SIZE, shuffle=True,
            num_negatives=NUM_NEG, dataset_percents = 0.001
)
        val_len = max(1, int(0.1 * len(full_loader.dataset)))
        train_len = len(full_loader.dataset) - val_len
        print(val_len, train_len)
        ds_tr, ds_val = torch.utils.data.random_split(full_loader.dataset,[train_len, val_len])
        collate = build_collate(self.proc)
        self.tr_loader = DataLoader(ds_tr,  BATCH_SIZE, shuffle=True,
                                    num_workers=8, collate_fn=collate)
        self.val_loader= DataLoader(ds_val, BATCH_SIZE, shuffle=False,
                                    num_workers=8, collate_fn=collate)

        self.opt = AdamW(self.model.parameters(), lr=LR, weight_decay=1e-4)

        total_steps  = len(self.tr_loader) * EPOCHS
        warmup_steps = int(total_steps * 0.10)
        self.sched = get_cosine_schedule_with_warmup(
            self.opt, warmup_steps, total_steps)

        # wandb.init(project="gme-qwen-omni7b",
        #            name=f"{TASK_TYPE}_LoRA", config=dict(bs=BATCH_SIZE, k=NUM_NEG))

    def _embed(self, pack):
        pack = {k: v.to(DEVICE) for k, v in pack.items()}  
        out = self.model(**pack, output_hidden_states=True,
                         use_cache=False, return_dict=True)
        return eos(out.hidden_states[-1], pack["input_ids"], self.pad)
    
    @staticmethod      
    def gpu_report(tag):
        free, total = torch.cuda.mem_get_info(0)
        used = total - free
        print(f"{tag} | used {humanize.naturalsize(used)} / {humanize.naturalsize(total)}")

    def _train_epoch(self, epoch):
        self.model.train()
        loss_acc = prec_acc = rec_acc = f1_acc = 0.0
        half_pt  = int(len(self.tr_loader) * TRAIN_LOG_EVERY)

        for step, b in enumerate(self.tr_loader, 1):
            
            # import numpy as np  # если ещё не импортировал

            # print(f"\n[Epoch {epoch} | Step {step}]")
            # print(self.proc.image_processor)

            # def tensor_stats(name, tensor):
            #     shape = tuple(tensor.shape)
            #     size_bytes = tensor.element_size() * tensor.numel()
            #     size_mb = size_bytes / (1024 ** 2)
            #     print(f"  {name:<12} | shape: {shape}, dtype: {tensor.dtype}, size: {size_mb:.2f} MB")

            # # Вытаскиваем все тензоры из query, pos, neg
            # for k in ["query", "pos"]:
            #     print(f"  --- {k.upper()} ---")
            #     for name, tensor in b[k].items():
            #         if isinstance(tensor, torch.Tensor):
            #             tensor_stats(name, tensor)

            q = self._embed(b["query"])   # (B, D)
            p = self._embed(b["pos"])   # (B, D)

            loss, logits = info_nce_inbatch(q, p)

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.opt.step(); self.sched.step()
            print('.', end='', flush=True)
  
            loss_acc += loss.item()

            if step == half_pt or step == len(self.tr_loader):
                print(f"EP{epoch} step {step}/{len(self.tr_loader)} | "
                    f"loss {(loss_acc/step):.4f}")

            del q, p, logits; torch.cuda.empty_cache()

        n_steps = len(self.tr_loader)
        return loss_acc / n_steps

            
    @torch.no_grad()
    def _build_val_index(self):
        self.model.eval()
        Q_list, P_list = [], []
        for b in self.val_loader:
            Q_list.append(self._embed(b["query"]).cpu())
            P_list.append(self._embed(b["pos"]).cpu())
            torch.cuda.empty_cache()

        Q = F.normalize(torch.cat(Q_list), p=2, dim=1)          
        P = F.normalize(torch.cat(P_list), p=2, dim=1).half()           
        rel = torch.arange(Q.size(0))                         # (N,)
        return Q, P, rel
        
        
    @torch.no_grad()
    def _metrics(self, Q, P, rel):
        N        = Q.size(0)
        P_gpu    = P.T.to(DEVICE)               # (D, N)
        hits     = {k: [] for k in TOP_KS}
        ranks    = []
        probs    = []

        for s in range(0, N, VAL_CHUNK):
            e     = min(s + VAL_CHUNK, N)
            sims  = Q[s:e].to(DEVICE) @ P_gpu.float()    # (chunk, N)
            soft  = F.softmax(sims.float() / TEMP, 1)
            probs.append(soft[torch.arange(e - s), rel[s:e].to(DEVICE)].cpu())

            order = sims.argsort(1, descending=True).cpu()    # (chunk, N)
            ranks.append(((order == rel[s:e].unsqueeze(1)).nonzero()[:, 1]) + 1)

            for k in TOP_KS:
                hits[k].append((order[:, :k] == rel[s:e].unsqueeze(1)).any(1).float())

            del sims, soft, order; torch.cuda.empty_cache()

        mrr  = (1 / torch.cat(ranks).float()).mean().item()
        prob = torch.cat(probs).mean().item()
        out  = {"MRR": mrr, "Prob": prob}
        for k in TOP_KS:
            out[f"R@{k}"] = torch.cat(hits[k]).mean().item()
        return out
   
    
    def fit(self):
        for ep in range(1,EPOCHS+1):
            tr_loss = self._train_epoch(ep)

            log = {
                "epoch": ep,
                "train_loss": tr_loss,
                "lr": self.sched.get_last_lr()[0],
            }
            # log.update(self.gpu_report("train_"))
            if ep % VAL_EVERY_EPOCHS == 0:
                    Q, D, rel = self._build_val_index()
                    val = self._metrics(Q, D, rel)
                    log.update({f"val_{k}": v for k, v in val.items()})
                    print(f"VAL EP{ep}:", {k: f"{v:.4f}" for k, v in val.items()})
                    del Q, D, rel; torch.cuda.empty_cache()

            if ep % SAVE_EVERY_EPOCHS == 0:
                model_path = os.path.join(lora_dir, f"ep{ep}")
                os.makedirs(model_path, exist_ok=True)
                self.model.save_pretrained(model_path)

                epoch_log_dir = os.path.join(log_dir, f"ep{ep}")
                os.makedirs(epoch_log_dir, exist_ok=True)
                log_path = os.path.join(epoch_log_dir, "log.jsonl")

                with open(log_path, "a") as f:
                    f.write(json.dumps(log) + "\n")
            
if __name__ == "__main__":
    torch.cuda.set_device(0)
    Trainer().fit(); #wandb.finish()