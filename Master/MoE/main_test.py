import argparse
from setup.models.modeling_model_test import train_and_save, predict_from_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--data", type=str, help="Caminho para arquivo .jsonl de treino")
    parser.add_argument("--series", nargs="+", type=float, help="Série temporal para predição")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--use_noise", type=str, default="true")
    parser.add_argument("--norm", type=str, default="std")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default="moe_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.mode == "train":
        if not args.data:
            raise ValueError("Precisa fornecer --data no modo train")
        train_and_save(args.data, args.horizon, args.save_path, top_k=args.top_k, epochs=args.epochs, lr=args.lr, use_noise=args.use_noise, device=args.device)

    elif args.mode == "predict":
        if not args.series:
            raise ValueError("Precisa fornecer --series no modo predict")
        preds = predict_from_model(args.save_path, args.series, top_k=args.top_k, use_noise=args.use_noise, device=args.device)
        print(preds.tolist())


if __name__ == "__main__":
    main()