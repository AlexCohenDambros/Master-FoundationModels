import argparse
from setup.models.configuration_model import train_and_save, predict_from_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--data", type=str, help="Caminho para arquivo .jsonl de treino")
    parser.add_argument("--series", nargs="+", type=float, help="Série temporal para predição")
    parser.add_argument("--context_length", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--save_path", type=str, default="moe_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.mode == "train":
        if not args.data:
            raise ValueError("Precisa fornecer --data no modo train")
        train_and_save(args.data, args.context_length, args.horizon, args.save_path, device=args.device)

    elif args.mode == "predict":
        if not args.series:
            raise ValueError("Precisa fornecer --series no modo predict")
        preds = predict_from_model(args.save_path, args.series, args.context_length, device=args.device)
        print(preds.tolist())


if __name__ == "__main__":
    main()