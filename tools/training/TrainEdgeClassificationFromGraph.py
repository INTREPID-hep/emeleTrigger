import argparse
import csv
import json
import os
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.transforms import Compose

from transformations import NormalizeEdgeFeatures, NormalizeNodeFeatures


class EdgeClassifierGNN(nn.Module):
    def __init__(self, in_channels, edge_in_channels, hidden_dim=64, model_type="SAGE", dropout=0.2):
        super().__init__()
        self.model_type = model_type.upper()
        self.dropout = dropout

        if self.model_type == "GCN":
            self.conv1 = GCNConv(in_channels, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
        else:
            self.conv1 = SAGEConv(in_channels, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, hidden_dim)

        edge_mlp_in = 2 * hidden_dim + edge_in_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
            edge_attr = data.edge_attr.float()
            edge_feat = torch.cat([x_src, x_dst, edge_attr], dim=1)
        else:
            edge_feat = torch.cat([x_src, x_dst], dim=1)

        logits = self.edge_mlp(edge_feat).squeeze(-1)
        return logits


class TrainEdgeClassificationFromGraph:
    def __init__(self, **kwargs):
        config_file = kwargs.get("config")
        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as file_handle:
                cfg = yaml.safe_load(file_handle)
            for key, value in cfg.items():
                kwargs[key] = value

        self.graph_path = kwargs.get("graph_path", ".")
        self.graph_name = kwargs.get("graph_name", "l1nano")
        self.out_model_path = kwargs.get("out_model_path", "edge_classification_results")
        self.save_tag = kwargs.get("save_tag", "edge_cls")
        self.batch_size = kwargs.get("batch_size", 64)
        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.epochs = kwargs.get("epochs", 50)
        self.earlystop = kwargs.get("earlystop", 10)
        self.model_type = kwargs.get("model_type", "SAGE")
        self.hidden_dim = kwargs.get("hidden_dim", 64)
        self.dropout = kwargs.get("dropout", 0.2)
        self.normalization = kwargs.get("normalization", "NodesAndEdges")
        self.num_files = kwargs.get("num_files", None)
        self.seed = kwargs.get("seed", 42)
        self.train_ratio = kwargs.get("train_ratio", 0.7)
        self.val_ratio = kwargs.get("val_ratio", 0.15)
        self.threshold = kwargs.get("threshold", 0.5)
        self.device_str = kwargs.get("device", "cuda")
        default_model_path = os.path.join(self.out_model_path, f"edge_model_{self.model_type}_{self.save_tag}.pth")
        model_path_from_args = kwargs.get("model_path")
        self.model_path = model_path_from_args if model_path_from_args else default_model_path
        self.pos_weight_mode = kwargs.get("pos_weight", "auto")

        self.do_train = kwargs.get("do_train", False)
        self.do_validation = kwargs.get("do_validation", False)
        self.do_test = kwargs.get("do_test", False)

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.device_str == "cuda" else "cpu")

        self.transform = self._build_transform(self.normalization)
        self.model = None
        self.optimizer = None
        self.criterion = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self._set_seed(self.seed)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_transform(self, normalization):
        if normalization == "NodesAndEdges":
            return Compose([NormalizeNodeFeatures(), NormalizeEdgeFeatures()])
        if normalization == "Nodes":
            return NormalizeNodeFeatures()
        if normalization == "Edges":
            return NormalizeEdgeFeatures()
        return None

    def _load_graphs(self):
        all_files = os.listdir(self.graph_path)
        graph_files = [
            file_name
            for file_name in all_files
            if (file_name.endswith(".pt") or file_name.endswith(".pkl")) and self.graph_name in file_name
        ]

        graph_files.sort()
        if self.num_files is not None:
            graph_files = graph_files[: self.num_files]

        if not graph_files:
            raise RuntimeError(f"No graph files found in {self.graph_path} matching '{self.graph_name}'")

        graphs_per_file = []
        for graph_file in graph_files:
            full_path = os.path.join(self.graph_path, graph_file)
            graph_obj = torch.load(full_path, map_location="cpu", weights_only=False)
            graphs_per_file.append(graph_obj)

        graphs = sum(graphs_per_file, [])
        return graphs

    def _is_valid_graph(self, graph):
        if not hasattr(graph, "edge_y") or graph.edge_y is None:
            return False
        if graph.edge_index is None or graph.edge_index.size(1) == 0:
            return False
        if graph.edge_y.numel() == 0:
            return False
        if graph.edge_y.shape[0] != graph.edge_index.shape[1]:
            return False

        x_nan = torch.isnan(graph.x).any() if hasattr(graph, "x") and graph.x is not None else True
        edge_nan = False
        if hasattr(graph, "edge_attr") and graph.edge_attr is not None and graph.edge_attr.numel() > 0:
            edge_nan = torch.isnan(graph.edge_attr).any()
        y_nan = torch.isnan(graph.edge_y.float()).any()

        return not (x_nan or edge_nan or y_nan)

    def _prepare_graph(self, graph):
        graph = deepcopy(graph)
        graph.edge_y = graph.edge_y.float().view(-1)

        if self.transform is not None:
            graph = self.transform(graph)

        if graph.edge_y.shape[0] != graph.edge_index.shape[1]:
            return None

        if torch.isnan(graph.x).any():
            return None
        if hasattr(graph, "edge_attr") and graph.edge_attr is not None and graph.edge_attr.numel() > 0:
            if torch.isnan(graph.edge_attr).any():
                return None
        if torch.isnan(graph.edge_y).any():
            return None

        return graph

    def load_data(self):
        raw_graphs = self._load_graphs()
        prepared = []

        for graph in raw_graphs:
            if not self._is_valid_graph(graph):
                continue
            graph_prepared = self._prepare_graph(graph)
            if graph_prepared is not None:
                prepared.append(graph_prepared)

        if len(prepared) < 10:
            raise RuntimeError(f"Not enough valid graphs for training: {len(prepared)}")

        random.Random(self.seed).shuffle(prepared)

        total = len(prepared)
        n_train = max(1, int(total * self.train_ratio))
        n_val = max(1, int(total * self.val_ratio))
        if n_train + n_val >= total:
            n_val = max(1, total - n_train - 1)
        n_test = total - n_train - n_val

        if n_test < 1:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

        train_data = prepared[:n_train]
        val_data = prepared[n_train : n_train + n_val]
        test_data = prepared[n_train + n_val :]

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        print(f"Total graphs: {total}")
        print(f"Train/Val/Test graphs: {len(train_data)}/{len(val_data)}/{len(test_data)}")

    def _compute_pos_weight(self):
        if self.pos_weight_mode != "auto":
            return float(self.pos_weight_mode)

        positives = 0
        negatives = 0
        for batch in self.train_loader:
            labels = batch.edge_y.view(-1)
            positives += int((labels == 1).sum().item())
            negatives += int((labels == 0).sum().item())

        if positives == 0:
            return 1.0
        return max(1.0, negatives / positives)

    def initialize_model(self):
        sample_batch = next(iter(self.train_loader))
        in_channels = sample_batch.x.shape[1]
        edge_in_channels = 0
        if hasattr(sample_batch, "edge_attr") and sample_batch.edge_attr is not None:
            edge_in_channels = sample_batch.edge_attr.shape[1]

        self.model = EdgeClassifierGNN(
            in_channels=in_channels,
            edge_in_channels=edge_in_channels,
            hidden_dim=self.hidden_dim,
            model_type=self.model_type,
            dropout=self.dropout,
        ).to(self.device)

        pos_weight = self._compute_pos_weight()
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        print(f"Device: {self.device}")
        print(f"Model type: {self.model_type}")
        print(f"Positive class weight: {pos_weight:.4f}")

    def _run_epoch(self, loader, training=False, return_outputs=False):
        all_logits = []
        all_labels = []
        total_loss = 0.0

        if training:
            self.model.train()
        else:
            self.model.eval()

        for batch in loader:
            batch = batch.to(self.device)
            labels = batch.edge_y.float().view(-1)

            if training:
                self.optimizer.zero_grad()

            logits = self.model(batch)
            loss = self.criterion(logits, labels)

            if training:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        logits_cat = torch.cat(all_logits).numpy()
        labels_cat = torch.cat(all_labels).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits_cat))
        preds = (probs >= self.threshold).astype(int)

        metrics = self._compute_metrics(labels_cat, preds, probs)
        metrics["loss"] = total_loss / max(1, len(loader))
        if return_outputs:
            return metrics, labels_cat, probs, preds
        return metrics

    def _compute_metrics(self, labels, preds, probs):
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(labels, preds),
            "mcc": matthews_corrcoef(labels, preds),
            "positive_rate": float(np.mean(labels)),
            "predicted_positive_rate": float(np.mean(preds)),
        }

        try:
            metrics["roc_auc"] = roc_auc_score(labels, probs)
        except ValueError:
            metrics["roc_auc"] = float("nan")

        try:
            metrics["pr_auc"] = average_precision_score(labels, probs)
        except ValueError:
            metrics["pr_auc"] = float("nan")

        return metrics

    def _format_metrics(self, prefix, metrics):
        return (
            f"{prefix} loss={metrics['loss']:.4f} "
            f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}"
        )

    def training_loop(self):
        os.makedirs(self.out_model_path, exist_ok=True)

        history = []
        best_val_loss = float("inf")
        best_epoch = -1
        patience = 0

        for epoch in range(1, self.epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, training=True)
            val_metrics = self._run_epoch(self.val_loader, training=False)

            row = {"epoch": epoch}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(row)

            print(self._format_metrics(f"Epoch {epoch:03d} TRAIN", train_metrics))
            print(self._format_metrics(f"Epoch {epoch:03d} VAL  ", val_metrics))

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                patience = 0
                torch.save(self.model.state_dict(), self.model_path)
            else:
                patience += 1

            if patience >= self.earlystop:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
                break

        history_path = os.path.join(self.out_model_path, f"history_{self.save_tag}.csv")
        self._save_history_csv(history, history_path)

        plot_path = os.path.join(self.out_model_path, f"history_{self.save_tag}.png")
        self._plot_history(history, plot_path)

        print(f"Saved best model to: {self.model_path}")
        print(f"Saved training history to: {history_path}")

    def _save_history_csv(self, history, path):
        if not history:
            return
        with open(path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    def _plot_history(self, history, output_path):
        if not history:
            return

        epochs = [row["epoch"] for row in history]
        train_loss = [row["train_loss"] for row in history]
        val_loss = [row["val_loss"] for row in history]
        train_acc = [row["train_accuracy"] for row in history]
        val_acc = [row["val_accuracy"] for row in history]
        train_prec = [row["train_precision"] for row in history]
        val_prec = [row["val_precision"] for row in history]
        train_f1 = [row["train_f1"] for row in history]
        val_f1 = [row["val_f1"] for row in history]

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes = axes.ravel()

        axes[0].plot(epochs, train_loss, label="train_loss")
        axes[0].plot(epochs, val_loss, label="val_loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss vs epoch")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs, train_acc, label="train_accuracy")
        axes[1].plot(epochs, val_acc, label="val_accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy vs epoch")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        axes[2].plot(epochs, train_prec, label="train_precision")
        axes[2].plot(epochs, val_prec, label="val_precision")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Precision")
        axes[2].set_title("Precision vs epoch")
        axes[2].grid(alpha=0.3)
        axes[2].legend()

        axes[3].plot(epochs, train_f1, label="train_f1")
        axes[3].plot(epochs, val_f1, label="val_f1")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("F1")
        axes[3].set_title("F1 vs epoch")
        axes[3].grid(alpha=0.3)
        axes[3].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_roc_curve(self, labels, probs, split_name):
        unique_classes = np.unique(labels)
        if unique_classes.size < 2:
            print(f"Skipping ROC plot for {split_name}: only one class present ({unique_classes})")
            return

        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC curve ({split_name})")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        plt.tight_layout()

        output_path = os.path.join(self.out_model_path, f"roc_{split_name}_{self.save_tag}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_confusion_matrix(self, labels, preds, split_name):
        cm = confusion_matrix(labels.astype(int), preds.astype(int), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Bkg", "Pred Sig"])
        ax.set_yticklabels(["True Bkg", "True Sig"])
        ax.set_title(f"Confusion matrix ({split_name})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

        text = f"TN={tn}  FP={fp}  FN={fn}  TP={tp}"
        ax.text(0.5, -0.12, text, transform=ax.transAxes, ha="center", va="top", fontsize=9)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        output_path = os.path.join(self.out_model_path, f"confusion_matrix_{split_name}_{self.save_tag}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_discriminator_distribution(self, labels, probs, split_name):
        signal_mask = labels.astype(int) == 1
        background_mask = labels.astype(int) == 0

        fig, ax = plt.subplots(figsize=(7, 5))
        bins = np.linspace(0.0, 1.0, 50)

        if np.any(background_mask):
            ax.hist(
                probs[background_mask],
                bins=bins,
                histtype="step",
                density=True,
                linewidth=2,
                label="Background (edge_y=0)",
                color="tab:blue",
            )
        if np.any(signal_mask):
            ax.hist(
                probs[signal_mask],
                bins=bins,
                histtype="step",
                density=True,
                linewidth=2,
                label="Signal (edge_y=1)",
                color="tab:orange",
            )

        ax.set_xlabel("Discriminator output")
        ax.set_ylabel("Density")
        ax.set_title(f"Discriminator distribution ({split_name})")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()

        output_path = os.path.join(self.out_model_path, f"discriminator_{split_name}_{self.save_tag}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def load_trained_model(self):
        if self.model is None:
            self.initialize_model()
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def evaluate_split(self, split_name="test"):
        if split_name == "train":
            loader = self.train_loader
        elif split_name == "val":
            loader = self.val_loader
        else:
            loader = self.test_loader

        metrics, labels, probs, preds = self._run_epoch(loader, training=False, return_outputs=True)
        print(self._format_metrics(f"{split_name.upper():<5}", metrics))

        os.makedirs(self.out_model_path, exist_ok=True)
        metrics_path = os.path.join(self.out_model_path, f"metrics_{split_name}_{self.save_tag}.json")
        with open(metrics_path, "w", encoding="utf-8") as file_handle:
            json.dump(metrics, file_handle, indent=2)
        print(f"Saved {split_name} metrics to: {metrics_path}")

        self._plot_roc_curve(labels, probs, split_name)
        self._plot_confusion_matrix(labels, preds, split_name)
        self._plot_discriminator_distribution(labels, probs, split_name)

        outputs_path = os.path.join(self.out_model_path, f"discriminator_outputs_{split_name}_{self.save_tag}.npz")
        np.savez(outputs_path, labels=labels, probs=probs, preds=preds)
        print(f"Saved discriminator outputs to: {outputs_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train/test edge-classification GNN from graph dataset")
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--graph_path", type=str, default=".", help="Path to graph files")
    parser.add_argument("--graph_name", type=str, default="l1nano", help="Substring to select graph files")
    parser.add_argument("--out_model_path", type=str, default="edge_classification_results")
    parser.add_argument("--save_tag", type=str, default="edge_cls")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--earlystop", type=int, default=10)
    parser.add_argument("--model_type", type=str, default="SAGE", help="SAGE or GCN")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--normalization", type=str, default="NodesAndEdges", help="None, Nodes, Edges, NodesAndEdges")
    parser.add_argument("--num_files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos_weight", type=str, default="auto", help="auto or numeric value")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_validation", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = TrainEdgeClassificationFromGraph(**vars(args))
    trainer.load_data()
    trainer.initialize_model()

    if args.do_train:
        trainer.training_loop()

    if args.do_validation or args.do_test:
        trainer.load_trained_model()

    if args.do_validation:
        trainer.evaluate_split("val")

    if args.do_test:
        trainer.evaluate_split("test")


if __name__ == "__main__":
    main()
