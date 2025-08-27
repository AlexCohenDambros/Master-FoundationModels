#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
Como usar:
  Treinar:
    python main.py --mode train --data dataset.jsonl --input-length 168 --pred-length 24lr  --save-path ./moe_ckpt.pt

  Prever (usar checkpoint salvo):
    python main.py --mode predict --load-path ./moe_ckpt.pt --series 1 2 3 ... --device cuda

Obs: verifique PATH/packaging para que as importações relativas funcionem no seu projeto.
"""

import argparse
import random
import torch

from setup.models import configuration_model, predict_from_checkpoint

# -------------------------
# MAIN: parser + dispatch
# -------------------------
def main():
    p = argparse.ArgumentParser(description="MoE router (foundation models as experts) — main/configuration separated.")
    p.add_argument("--mode", choices=["train", "predict", "demo"], default="demo")
    p.add_argument("--data", type=str, default=None, help="JSONL path (cada linha: {'sequence': [...]})")
    p.add_argument("--input-length", type=int, default=168)
    p.add_argument("--pred-length", type=int, default=24)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--save-path", type=str, default="./moe_router_ckpt.pt")
    p.add_argument("--load-path", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--gating-label-frac", type=float, default=0.2)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--series", nargs="+", type=float, default=None, help="For one-shot prediction (list of floats)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save-path-eval-plot", type=str, default="plot_actual_vs_predicted.png")
    args = p.parse_args()

    # seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == "train":
        if args.data is None:
            raise ValueError("--data obrigatório no modo train")
        configuration_model(args)
    elif args.mode == "predict":
        if args.load_path is None:
            raise ValueError("--load-path obrigatório no modo predict")
        predict_from_checkpoint(args)
    elif args.mode == "demo":
        demo_mode(args)
    else:
        raise ValueError("Modo desconhecido")

if __name__ == "__main__":
    main()