import os
import re
import shutil
import pandas as pd
import matplotlib.pyplot as plt


def run_analysis(
    base_dir="trained_models",
    output_dir="analysis_results",
):
    """
    Executa a análise completa dos modelos treinados.
    Remove completamente o diretório de output a cada execução.
    """

    # ======================================================
    # Limpa output
    # ======================================================
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    all_weights_summary = []
    all_epochs_info = []

    # ======================================================
    # Funções auxiliares
    # ======================================================
    def extract_year(folder_name):
        match = re.search(r"(19|20)\d{2}", folder_name)
        return match.group(0) if match else "unknown"

    def analyze_experiment(exp_path, horizon, state, year):
        weights_path = os.path.join(exp_path, "experts_weights_train.csv")
        loss_path = os.path.join(exp_path, "train_loss.csv")

        exp_name = os.path.basename(exp_path)

        # ===============================
        # Experts weights
        # ===============================
        if os.path.exists(weights_path):
            df_w = pd.read_csv(weights_path)

            total_selections = len(df_w)

            summary = (
                df_w.groupby("Learner")
                .agg(
                    selections=("Learner", "count"),
                    avg_weight=("Weight", "mean"),
                )
                .reset_index()
            )

            summary["percentage"] = (
                100.0 * summary["selections"] / total_selections
            )

            summary["horizon"] = horizon
            summary["state"] = state
            summary["year"] = year
            summary["experiment"] = exp_name

            all_weights_summary.append(summary)

        # ===============================
        # Train loss / epochs
        # ===============================
        if os.path.exists(loss_path):
            df_l = pd.read_csv(loss_path)

            num_epochs = len(df_l)
            min_loss = df_l["Train Loss"].min()

            all_epochs_info.append(
                {
                    "horizon": horizon,
                    "state": state,
                    "year": year,
                    "experiment": exp_name,
                    "num_epochs": num_epochs,
                    "min_loss": min_loss,
                }
            )

            # Plot Loss x Epochs
            plot_dir = os.path.join(
                output_dir, horizon, state, year, exp_name
            )
            os.makedirs(plot_dir, exist_ok=True)

            plt.figure()
            plt.plot(df_l["Epoch"], df_l["Train Loss"])
            plt.xlabel("Epoch")
            plt.ylabel("Train Loss")
            plt.title(f"Loss x Epochs\n{exp_name}")
            plt.tight_layout()

            plt.savefig(os.path.join(plot_dir, "loss_curve.png"))
            plt.close()

    # ======================================================
    # Percorre diretórios
    # ======================================================
    for horizon in sorted(os.listdir(base_dir)):
        horizon_path = os.path.join(base_dir, horizon)
        if not os.path.isdir(horizon_path):
            continue

        for state in sorted(os.listdir(horizon_path)):
            state_path = os.path.join(horizon_path, state)
            if not os.path.isdir(state_path):
                continue

            for exp_folder in sorted(os.listdir(state_path)):
                exp_path = os.path.join(state_path, exp_folder)
                if not os.path.isdir(exp_path):
                    continue

                year = extract_year(exp_folder)

                analyze_experiment(
                    exp_path=exp_path,
                    horizon=horizon,
                    state=state,
                    year=year,
                )

    # ======================================================
    # Salva resultados agregados
    # ======================================================
    if all_weights_summary:
        df_weights = pd.concat(all_weights_summary, ignore_index=True)

        df_weights.to_csv(
            os.path.join(output_dir, "experts_weights_summary.csv"),
            index=False,
        )

    if all_epochs_info:
        df_epochs = pd.DataFrame(all_epochs_info)

        # Estatísticas globais
        stats = {
            "mean_epochs": df_epochs["num_epochs"].mean(),
            "min_epochs_experiment": df_epochs.loc[
                df_epochs["num_epochs"].idxmin()
            ].to_dict(),
            "max_epochs_experiment": df_epochs.loc[
                df_epochs["num_epochs"].idxmax()
            ].to_dict(),
        }

        pd.DataFrame([stats]).to_json(
            os.path.join(output_dir, "epochs_global_stats.json"),
            indent=4,
        )

        df_epochs.to_csv(
            os.path.join(output_dir, "epochs_per_experiment.csv"),
            index=False,
        )

    print("✅ Análise concluída com sucesso!")
    print(f"📁 Resultados salvos em: {output_dir}")


if __name__ == "__main__":
    run_analysis()