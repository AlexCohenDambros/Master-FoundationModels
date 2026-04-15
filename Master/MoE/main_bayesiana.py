import argparse
import json
from setup.models.modeling_model_v2 import train_and_save, predict_from_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--data", type=str, help="Path to .jsonl training file")
    parser.add_argument("--series", nargs="+", type=float, help="Time series for prediction")
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
            raise ValueError("Must provide --data in train mode")

        _, val_rmse = train_and_save(
            args.data,
            args.horizon,
            args.save_path,
            top_k=args.top_k,
            epochs=args.epochs,
            lr=args.lr,
            use_noise=args.use_noise,
            norm=args.norm,
            device=args.device,
        )

        # Save metric as JSON alongside the .pt so Optuna can read it
        if val_rmse is not None:
            metrics_path = args.save_path.replace(".pt", "_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({"val_rmse": float(val_rmse)}, f)
            print(f"val_rmse: {val_rmse:.6f}")

    elif args.mode == "predict":
        if not args.series:
            raise ValueError("Must provide --series in predict mode")

        preds = predict_from_model(
            args.save_path,
            args.series,
            horizon=args.horizon,
            top_k=args.top_k,
            use_noise=args.use_noise,
            device=args.device,
        )
        print(preds.tolist())


if __name__ == "__main__":
    main()