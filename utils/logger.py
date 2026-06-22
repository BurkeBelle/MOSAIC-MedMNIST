# utils/logger.py
"""
Experiment logger.

Records per-round ACC/AUC for student and teacher, tracks forgetting,
and generates summary files (JSON, CSV, plots).
"""

import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class ExperimentLogger:
    def __init__(self, exp_name, dataset_list, output_dir="./results", config=None):
        self.exp_name = exp_name
        self.dataset_list = dataset_list
        self.datasets_2d = [d for d in dataset_list if "3D" not in d]
        self.datasets_3d = [d for d in dataset_list if "3D" in d]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(output_dir) / f"{exp_name}_{ts}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.data = {
            "config": {"exp_name": exp_name, "datasets": dataset_list,
                       "timestamp": ts, **(config or {})},
            "round_results": [],
        }
        self.best_acc = {d: 0.0 for d in dataset_list}
        print(f"Logger: {self.save_dir}")

    # ------------------------------------------------------------------
    def log_round(self, round_idx, student_results, teacher_results=None):
        accs = {d: student_results.get(d, {}).get("acc", 0) * 100 for d in self.dataset_list}
        forgetting = self._forgetting(accs)
        for d, a in accs.items():
            self.best_acc[d] = max(self.best_acc[d], a)

        rec = {
            "round": round_idx + 1,
            "student_mean_acc": student_results.get("mean_acc", 0) * 100,
            "student_mean_auc": student_results.get("mean_auc", 0),
            "avg_forgetting": forgetting["avg"],
        }
        if teacher_results:
            rec["teacher_mean_acc"] = teacher_results.get("mean_acc", 0) * 100
        for d in self.dataset_list:
            rec[f"{d}_acc"] = accs.get(d, 0)

        self.data["round_results"].append(rec)
        self._save()

    def log_test_results(self, student_results, teacher_results=None,
                         best_round=None, best_val_acc=None):
        self.data["test_results"] = {
            "student": {d: student_results[d] for d in self.dataset_list if d in student_results},
            "teacher": {d: teacher_results[d] for d in self.dataset_list if d in (teacher_results or {})},
            "best_round": (best_round + 1) if best_round is not None else None,
            "best_val_acc": best_val_acc,
            "mean_acc": student_results.get("mean_acc", 0),
            "mean_auc": student_results.get("mean_auc", 0),
        }
        self._save()

    # ------------------------------------------------------------------
    def _forgetting(self, current_accs):
        f = {d: max(0, self.best_acc[d] - current_accs.get(d, 0)) for d in self.dataset_list}
        vals = list(f.values())
        return {"avg": float(np.mean(vals)) if vals else 0, "per_dataset": f}

    def _save(self):
        with open(self.save_dir / "results.json", "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def finalize(self):
        self._save()
        if pd is not None and self.data["round_results"]:
            pd.DataFrame(self.data["round_results"]).to_csv(
                self.save_dir / "round_results.csv", index=False)
        self._plot()
        self._summary()
        print(f"Results saved to {self.save_dir}")

    def _plot(self):
        if plt is None or not self.data["round_results"]:
            return
        rounds = [r["round"] for r in self.data["round_results"]]
        accs = [r["student_mean_acc"] for r in self.data["round_results"]]
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(rounds, accs, "o-", label="Student")
        if "teacher_mean_acc" in self.data["round_results"][0]:
            ax.plot(rounds, [r["teacher_mean_acc"] for r in self.data["round_results"]],
                    "s--", label="Teacher")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean ACC (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(self.save_dir / "acc_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _summary(self):
        path = self.save_dir / "summary.txt"
        with open(path, "w") as f:
            f.write(f"Experiment: {self.exp_name}\n")
            f.write(f"Datasets: {len(self.dataset_list)} "
                    f"({len(self.datasets_2d)} 2D, {len(self.datasets_3d)} 3D)\n\n")
            if "test_results" in self.data:
                tr = self.data["test_results"]
                f.write(f"Test ACC: {tr.get('mean_acc', 0)*100:.2f}%\n")
                f.write(f"Test AUC: {tr.get('mean_auc', 0):.4f}\n")
                if tr.get("best_round"):
                    f.write(f"Best round: {tr['best_round']}\n")