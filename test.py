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
from collections.abc import Mapping
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
BATCH_SIZE         = 15
NUM_NEG            = 8
MAX_LEN            = 512
LR                 = 2e-4
EPOCHS             = 9
TEMP               = 0.09
TRAIN_LOG_EVERY    = 0.05
VAL_EVERY_EPOCHS   = 1
SAVE_EVERY_EPOCHS  = 1
DEVICE             = "cuda"
TOP_KS             = (1, 5, 10)
VAL_CHUNK          = 100

SYSTEM_MSG = "You are Qwen, a multilingual multimodal assistant."
SEP        = "[SEP]"

base_dir = "/home/rag/model"
lora_dir = os.path.join(base_dir, f"lora_checkpoints/{TASK_TYPE}_withn")
log_dir = os.path.join(base_dir, f"logs/{TASK_TYPE}_withn")

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

# ==== DEBUG helpers ====
DEBUG = True
PRINT_FIRST_N = 1  # сколько первых элементов батча печатаем "как есть"

def _tensor_stats(t: torch.Tensor):
    if t.numel() == 0:
        return dict(numel=0, shape=tuple(t.shape), device=str(t.device), dtype=str(t.dtype), requires_grad=t.requires_grad)
    d = {
        "shape": tuple(t.shape),
        "device": str(t.device),
        "dtype": str(t.dtype),
        "requires_grad": t.requires_grad,
        "norm2": float(t.float().norm().item()),
    }
    if t.is_floating_point():
        d.update({
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "mean": float(t.mean().item()),
        })
    else:
        d.update({
            "min": int(t.min().item()),
            "max": int(t.max().item()),
            "mean": None,
        })
    return d

def _print_tensor(name: str, t: torch.Tensor):
    try:
        s = _tensor_stats(t)
    except Exception as e:
        print(f"[T] {name}: <error while reading stats: {e}>")
        return
    print(f"[T] {name}: {s}")

def _print_pack(name: str, pack: Mapping):
    print(f"\n===== PACK: {name} =====")
    for k, v in pack.items():
        if isinstance(v, torch.Tensor):
            _print_tensor(k, v)
        else:
            # transformers BatchEncoding может быть маппингом списков/тензоров
            print(f"[?] {k}: type={type(v)}")
    print("========================\n")

def _print_loss_vars(q, p, n, logits, labels, loss, τ):
    print("\n===== LOSS VARS =====")
    _print_tensor("q", q)
    _print_tensor("p", p)
    _print_tensor("n", n)
    _print_tensor("logits", logits)
    _print_tensor("labels", labels)
    print(f"τ (temp): {τ}")
    print(f"loss: {float(loss.item())}")
    print("=====================\n")
# =======================


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

            # NOTE: у вас neg_prompts длины B*NUM_NEG, но neg_imgs вы не расширяли.
            # Если нужны и картинки, нужно так же заполнять neg_imgs (закомментировано, если не надо):
            # for txt, img in zip(n_texts, n_imgs):
            #     neg_prompts.append(make_prompt(_to_str(txt)))
            #     neg_imgs.append(img)

            # Если изображения не нужны (T-I), то можно оставить images=None, тогда neg_imgs пустой.
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

        # Вернём тексты для отладки
        return {
            "query": q_in,
            "pos": pos_in,
            "neg": neg_in,
            "meta": {
                "q_prompts": q_prompts[:PRINT_FIRST_N],
                "pos_prompts": pos_prompts[:PRINT_FIRST_N],
                "neg_prompts": neg_prompts[:PRINT_FIRST_N],
            }
        }
    return collate

def info_nce_sep(q, pos, neg, k, τ=TEMP):
    q   = F.normalize(q,   dim=-1)
    pos = F.normalize(pos, dim=-1)
    neg = F.normalize(neg, dim=-1)

    pos_sim = (pos * q).sum(-1, keepdim=True)                 # (B, 1)
    neg_sim = torch.matmul(neg, q.unsqueeze(-1)).squeeze(-1)  # (B, k)
    logits  = torch.cat([pos_sim, neg_sim], dim=1)            # (B, k+1)

    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    loss = F.cross_entropy(logits / τ, labels)
    return loss, logits, labels


def eos(hidden, ids, pad_id):
    if isinstance(pad_id, torch.Tensor):
        pad_id = pad_id.to(ids.device)  # переместим на тот же девайс

    idx = (ids != pad_id).sum(-1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]


class Trainer:
    def __init__(self):
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4")

        self.proc = Qwen2_5OmniProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        self.proc.image_processor.size        = {"height": 224, "width": 224}
        self.proc.image_processor.max_pixels  = 224 * 224
        self.proc.image_processor.min_pixels  = 1
        self.proc.image_processor.longest_edge = 224
        self.proc.image_processor.shortest_edge = 224
        self.proc.image_processor.do_resize   = True
        self.proc.image_processor.resample    = 3

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            MODEL_DIR, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.float16, local_files_only=True)

        if DEBUG:
            try:
                print(">>> model embedding device:", self.model.get_input_embeddings().weight.device)
            except Exception as e:
                print(">>> cannot get embedding device:", e)

        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)

        lora_cfg = LoraConfig(r=12, lora_alpha=36, lora_dropout=0.05,
                              bias="none", task_type="CAUSAL_LM",
                              target_modules=["q_proj","k_proj","v_proj","o_proj",
                                              "gate_proj","up_proj","down_proj"])

        if args.lora_ckpt and os.path.isdir(args.lora_ckpt):
            self.model = PeftModel.from_pretrained(self.model, args.lora_ckpt,
                                                   is_trainable=True, config=lora_cfg)
            print(f"LoRA‑адаптер загружен из {args.lora_ckpt}")
        else:
            self.model = get_peft_model(self.model, lora_cfg)
            print("Инициализирован новый LoRA‑адаптер")

        # 🔍 Проверка, что LoRA веса действительно загружены
        print("\n🧪 Проверка одного LoRA-веса:")
        for name, param in self.model.named_parameters():
            if "lora" in name:
                print(f"{name} | mean: {param.data.mean().item():.6f}")
                break

        # ✅ Проверка, что trainable веса присутствуют
        print("\n📌 Trainable параметры:")
        self.model.print_trainable_parameters()

        assert any("lora" in n for n, p in self.model.named_parameters() if p.requires_grad), \
            "LoRA layers were not attached — check target_modules list!"

        for name, param in self.model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

        self.pad = self.proc.tokenizer.pad_token_id
        self.model.train()

        full_loader = new_dataloader.create_multimodal_dataloader(
            DATA_PATH, TASK_TYPE, BATCH_SIZE, shuffle=True,
            num_negatives=NUM_NEG, dataset_percents=0.001
        )
        val_len = max(1, int(0.1 * len(full_loader.dataset)))
        train_len = len(full_loader.dataset) - val_len
        print(val_len, train_len)

        ds_tr, ds_val = torch.utils.data.random_split(full_loader.dataset, [train_len, val_len])
        collate = build_collate(self.proc)
        self.tr_loader = DataLoader(ds_tr,  BATCH_SIZE, shuffle=True,
                                    num_workers=4, collate_fn=collate)
        self.val_loader= DataLoader(ds_val, BATCH_SIZE, shuffle=False,
                                    num_workers=4, collate_fn=collate)

        self.opt = AdamW(self.model.parameters(), lr=LR, weight_decay=1e-4)

        total_steps  = len(self.tr_loader) * EPOCHS
        warmup_steps = int(total_steps * 0.10)
        self.sched = get_cosine_schedule_with_warmup(
            self.opt, warmup_steps, total_steps)

        # wandb.init(project="gme-qwen-omni7b",
        #            name=f"{TASK_TYPE}_LoRA", config=dict(bs=BATCH_SIZE, k=NUM_NEG))

    def _embed(self, pack):
        # Берём девайс, где реально лежат эмбеддинги.
        try:
            input_dev = self.model.get_input_embeddings().weight.device
        except Exception:
            input_dev = next(self.model.parameters()).device

        def move_to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device, non_blocking=True)
            elif isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [move_to_device(v, device) for v in obj]
            elif hasattr(obj, "to"):  # BatchEncoding и т.п.
                return obj.to(device)
            else:
                return obj

        pack = move_to_device(pack, input_dev)

        if DEBUG:
            _print_pack("INPUT_TO_MODEL", pack)
            try:
                print("embed weight device:", self.model.get_input_embeddings().weight.device)
            except Exception as e:
                print("cannot print embed weight device:", e)

        out = self.model(**pack,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True)

        last = eos(out.hidden_states[-1], pack["input_ids"], self.pad)

        if DEBUG:
            _print_tensor("eos(last_hidden)", last)

        return last


    @staticmethod
    def gpu_report(tag):
        free, total = torch.cuda.mem_get_info(0)
        used = total - free
        print(f"{tag} | used {humanize.naturalsize(used)} / {humanize.naturalsize(total)}")

    def _train_epoch(self, epoch):
        self.model.train(); torch.set_grad_enabled(True)
        loss_acc = 0.0
        half_pt = int(len(self.tr_loader) * TRAIN_LOG_EVERY)

        for step, b in enumerate(self.tr_loader, 1):

            if DEBUG and isinstance(b, dict) and "meta" in b:
                print("\n===== RAW TEXTS (first N) =====")
                print("q_prompts[0]:", b["meta"]["q_prompts"][0] if b["meta"]["q_prompts"] else None)
                print("pos_prompts[0]:", b["meta"]["pos_prompts"][0] if b["meta"]["pos_prompts"] else None)
                print("neg_prompts[0]:", b["meta"]["neg_prompts"][0] if b["meta"]["neg_prompts"] else None)
                print("===============================\n")

            q = self._embed(b["query"])
            p = self._embed(b["pos"])
            n = self._embed(b["neg"]).view(q.size(0), NUM_NEG, -1)

            loss, logits, labels = info_nce_sep(q, p, n, NUM_NEG)

            if DEBUG:
                _print_loss_vars(q, p, n, logits, labels, loss, TEMP)

            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.opt.step(); self.sched.step()

            if loss.item() < 1e-8:
                print(f"\n[STOP] Zero loss at epoch {epoch}, step {step}. Saving and breaking.")
                model_path = os.path.join(lora_dir, f"ep{epoch}")
                os.makedirs(model_path, exist_ok=True)
                self.model.save_pretrained(model_path)
                break

            print('.', end='', flush=True)
            loss_acc += loss.item()

            if step == half_pt or step == len(self.tr_loader):
                print(f"EP{epoch} step {step}/{len(self.tr_loader)} | "
                      f"loss {(loss_acc/step):.4f}")

            del q, p, n, logits, labels; torch.cuda.empty_cache()

        n_steps = len(self.tr_loader)
        return loss_acc / n_steps

    @torch.no_grad()
    def _build_val_index(self):
        self.model.eval()
        Q_list,D_list,rel=[],[],[]; cursor=0
        for b in self.val_loader:
            q   = self._embed(b["query"]).cpu()
            pos = self._embed(b["pos"]).cpu()
            neg = self._embed(b["neg"]).cpu()
            B = q.size(0)
            D_list.extend(pos); rel.extend(range(cursor,cursor+B)); cursor+=B
            D_list.extend(neg); cursor+=neg.size(0)
            Q_list.append(q)

            torch.cuda.empty_cache()

        Q = F.normalize(torch.cat(Q_list), p=2, dim=1)
        D = F.normalize(torch.stack(D_list), p=2, dim=1).half()
        return Q,D,torch.tensor(rel)

    @torch.no_grad()
    def _metrics(self,Q,D,rel):
        N=Q.size(0)
        D_gpu= D.T.to(DEVICE)
        hits={k:[] for k in TOP_KS}; ranks=[]; probs=[]

        for s in range(0,N,VAL_CHUNK):
            e = min(s + VAL_CHUNK, N)
            sims = Q[s:e].to(DEVICE) @ D_gpu.float()
            soft = F.softmax(sims.float()/TEMP, 1)
            probs.append(soft[torch.arange(e-s), rel[s:e].to(DEVICE)].cpu())

            order = sims.argsort(1,descending=True).cpu()
            ranks.append(((order==rel[s:e].unsqueeze(1)).nonzero()[:,1])+1)

            for k in TOP_KS:
                hits[k].append((order[:,:k]==rel[s:e].unsqueeze(1)).any(1).float())
            del sims,soft,order; torch.cuda.empty_cache()

        mrr = (1/torch.cat(ranks).float()).mean().item()
        prob = torch.cat(probs).mean().item()
        out = {"MRR":mrr,"Prob":prob}
        for k in TOP_KS:
            r = torch.cat(hits[k]).mean().item()
            out[f"R@{k}"]=r
        return out

    def fit(self):
        for ep in range(1,EPOCHS+1):
            tr_loss = self._train_epoch(ep)

            log = {
                "epoch": ep,
                "train_loss": tr_loss,
                "lr": self.sched.get_last_lr()[0],
            }

            if ep % SAVE_EVERY_EPOCHS == 0:
                model_path = os.path.join(lora_dir, f"ep{ep}")
                os.makedirs(model_path, exist_ok=True)
                self.model.save_pretrained(model_path)

            if ep % VAL_EVERY_EPOCHS == 0:
                Q, D, rel = self._build_val_index()
                val = self._metrics(Q, D, rel)
                log.update({f"val_{k}": v for k, v in val.items()})
                print(f"VAL EP{ep}:", {k: f"{v:.4f}" for k, v in val.items()})
                del Q, D, rel; torch.cuda.empty_cache()
                epoch_log_dir = os.path.join(log_dir, f"ep{ep}")
                os.makedirs(epoch_log_dir, exist_ok=True)
                log_path = os.path.join(epoch_log_dir, "log.jsonl")

                with open(log_path, "a") as f:
                    f.write(json.dumps(log) + "\n")

if __name__ == "__main__":
    torch.cuda.set_device(0)
    Trainer().fit(); # wandb.finish()
