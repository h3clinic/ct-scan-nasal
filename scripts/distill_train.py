"""Fine-tune Qwen2.5-1.5B-Instruct on the checked teacher answers (LoRA), then merge the adapter.

Runs in its own CUDA environment, not the project's:

    python scripts/distill_train.py --data data/distill/examples.jsonl --out data/distill/student

Only the assistant's answer counts toward the loss; the system prompt and the facts are context. The merged
model in <out>/merged is then packed for Ollama (see nasal.distill and the README).
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def encode(tokenizer, messages: list[dict], max_len: int) -> dict | None:
    """Token ids with labels only on the assistant's answer; None when it does not fit."""
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + ids[len(p_ids):]}


def batches(rows: list[dict], size: int, pad: int):
    for i in range(0, len(rows), size):
        chunk = rows[i:i + size]
        n = max(len(r["input_ids"]) for r in chunk)
        ids = torch.tensor([r["input_ids"] + [pad] * (n - len(r["input_ids"])) for r in chunk])
        labels = torch.tensor([r["labels"] + [-100] * (n - len(r["labels"])) for r in chunk])
        mask = torch.tensor([[1] * len(r["input_ids"]) + [0] * (n - len(r["input_ids"])) for r in chunk])
        yield ids, labels, mask


def answer_loss(model, ids, labels, mask, device) -> torch.Tensor:
    """Mean cross-entropy over the answer tokens of one example (batch size 1). The answer is the end of the
    sequence, so only its logits are computed: the full logits in float32 (151k-token vocabulary) are the
    largest allocation and ran a 12 GB GPU out of memory. Same value as passing labels to the model."""
    n = int((labels != -100).sum())
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device), logits_to_keep=n + 1).logits
    return torch.nn.functional.cross_entropy(logits[0, :-1].float(), labels[0, -n:].to(device))


def mean_loss(model, rows, pad, device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for ids, labels, mask in batches(rows, 1, pad):
            total, count = total + answer_loss(model, ids, labels, mask, device).item(), count + 1
    model.train()
    return total / max(count, 1)


def keep_top_level_rope(config_path: Path) -> None:
    """Newer transformers save rope_theta only under rope_parameters. Ollama's converter reads the top-level key;
    without it the model gets a RoPE base of 0 and produces garbage, so copy it up."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rope = config.get("rope_parameters") or {}
    if "rope_theta" not in config and "rope_theta" in rope:
        config["rope_theta"] = rope["rope_theta"]
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/distill/examples.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/distill/student"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=2048)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    rows = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    data = {s: [e for r in rows if r["split"] == s if (e := encode(tokenizer, r["messages"], args.max_len))]
            for s in ("train", "test")}
    print(f"train {len(data['train'])} test {len(data['test'])} (of {len(rows)} rows)", flush=True)

    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(data["train"]) / args.accum) * args.epochs
    schedule = get_cosine_schedule_with_warmup(optimizer, max(1, steps // 20), steps)
    report = {"base": BASE, "train_examples": len(data["train"]), "test_examples": len(data["test"]),
              "test_loss_before": round(mean_loss(model, data["test"], pad, device), 4), "epochs": []}
    print(f"test loss before: {report['test_loss_before']}", flush=True)
    start = time.time()
    for epoch in range(args.epochs):
        order = torch.randperm(len(data["train"]), generator=torch.Generator().manual_seed(epoch)).tolist()
        train = [data["train"][i] for i in order]
        for step, (ids, labels, mask) in enumerate(batches(train, 1, pad), start=1):
            loss = answer_loss(model, ids, labels, mask, device)
            (loss / args.accum).backward()
            if step % args.accum == 0 or step == len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
        test_loss = round(mean_loss(model, data["test"], pad, device), 4)
        report["epochs"].append({"epoch": epoch + 1, "test_loss": test_loss, "minutes": round((time.time() - start) / 60, 1)})
        print(f"epoch {epoch + 1}: test loss {test_loss} ({report['epochs'][-1]['minutes']} min)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out / "merged", safe_serialization=True)
    tokenizer.save_pretrained(args.out / "merged")
    keep_top_level_rope(args.out / "merged" / "config.json")
    (args.out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
