#!/usr/bin/env python3
"""
Inspection and validation script for L1Nano datasets.
Loads a saved .pt dataset and:
  - Prints graph structure summary
  - Plots 6 example graphs with networkx
  - Plots distributions of node and edge features
"""

import torch
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path


def print_dataset_summary(dataset):
    """Print high-level summary of the dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset Summary")
    print(f"{'='*60}")
    print(f"Total graphs: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nFirst graph structure:")
        print(f"  x (node features):     {tuple(sample.x.shape)}")
        print(f"  edge_index:            {tuple(sample.edge_index.shape)}")
        print(f"  edge_attr:             {tuple(sample.edge_attr.shape) if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 'None'}")
        print(f"  edge_y (edge labels):  {tuple(sample.edge_y.shape) if hasattr(sample, 'edge_y') and sample.edge_y is not None else 'None'}")
        print(f"  y (node labels):       {tuple(sample.y.shape) if hasattr(sample, 'y') and sample.y is not None else 'None'}")
        print(f"  num_nodes:             {sample.num_nodes}")
        print(f"  num_edges:             {sample.num_edges}")
        
        if hasattr(sample, 'genpart_features'):
            print(f"  genpart_features:      {tuple(sample.genpart_features.shape)}")
        if hasattr(sample, 'matched_muon_idx'):
            print(f"  matched_muon_idx:      {tuple(sample.matched_muon_idx.shape)}")
        if hasattr(sample, 'stub_deltaR'):
            print(f"  stub_deltaR:           {tuple(sample.stub_deltaR.shape)}")
    
    # Compute dataset statistics
    num_nodes_list = [g.num_nodes for g in dataset]
    num_edges_list = [g.num_edges for g in dataset]
    
    print(f"\nDataset Statistics:")
    print(f"  Nodes per graph:   min={min(num_nodes_list)}, max={max(num_nodes_list)}, mean={np.mean(num_nodes_list):.1f}")
    print(f"  Edges per graph:   min={min(num_edges_list)}, max={max(num_edges_list)}, mean={np.mean(num_edges_list):.1f}")


def plot_example_graphs(dataset, num_examples=6, output_file=None, show_plots=True):
    """Plot multiple example graphs using networkx layout."""
    print(f"\n{'='*60}")
    print(f"Plotting {num_examples} example graphs")
    print(f"{'='*60}")
    
    num_to_plot = min(num_examples, len(dataset))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    for plot_idx in range(num_to_plot):
        data = dataset[plot_idx]
        ax = axes[plot_idx]
        ax.set_facecolor('white')
        ax.set_axis_on()
        #ax.clear()
        #plt.sca(ax)
        
        # Build networkx graph
        G = nx.DiGraph()
        for i in range(data.num_nodes):
            G.add_node(i)
        
        edge_index = data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i])
            dst = int(edge_index[1, i])
            G.add_edge(src, dst)
        
        # Hierarchical layout by tfLayer (feature 0)
        tf_layers = data.x[:, 0].cpu().numpy()
        unique_layers = sorted(set(tf_layers.tolist()))
        nodes_by_layer = {layer: [] for layer in unique_layers}
        for node_id in range(data.num_nodes):
            layer = tf_layers[node_id]
            nodes_by_layer[layer].append(node_id)
        
        pos = {}
        for layer_id, layer in enumerate(unique_layers):
            layer_nodes = nodes_by_layer[layer]
            n_nodes = len(layer_nodes)
            for node_idx, node_id in enumerate(layer_nodes):
                pos[node_id] = (layer_id, (node_idx - n_nodes / 2) * 0.8)
        
        # Node colors by tfLayer (as in TrainL1Nano_v2.ipynb)
        node_colors = tf_layers

        # Node labels show (offeta1, offphi1)
        labels = {}
        for i in range(data.num_nodes):
            offeta = float(data.x[i, 1].item())
            offphi = float(data.x[i, 2].item())
            labels[i] = f"({offeta:.1f},{offphi:.1f})"

        # Draw
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            cmap='Pastel2',
            node_size=260,
            vmin=min(unique_layers),
            vmax=max(unique_layers),
            ax=ax,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            alpha=0.35,
            width=1.2,
            arrows=True,
            arrowsize=8,
            edge_color='gray',
            ax=ax,
        )
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8, font_weight='bold')
        
        ax.set_title(
            f"Grafo {plot_idx}\n{data.num_nodes} nodos, {data.num_edges} edges",
            fontsize=6,
        )
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')
        ax.set_xticks(range(len(unique_layers)))
        ax.set_xticklabels([f"L{int(l)}" for l in unique_layers])
        ax.set_xlabel("tfLayer", fontsize=11)
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=8)

        
    # Hide unused subplots
    for idx in range(num_to_plot, len(axes)):
        axes[idx].axis('off')
        
    plt.suptitle('Example Graphs from Dataset', y=0.995)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved graph examples to: {output_file}")
    if show_plots:
        try:
            plt.show()
        except Exception:
            pass  # Ignore show errors in non-interactive backends
    plt.close()


def plot_feature_distributions(dataset, output_file=None, show_plots=True):
    """Plot distributions of node features, edge attributes, and labels."""
    print(f"\n{'='*60}")
    print(f"Plotting feature distributions")
    print(f"{'='*60}")
    
    # Collect all features across dataset
    all_x = []
    all_edge_attr = []
    all_y = []
    all_edge_y = []
    
    for data in dataset:
        all_x.append(data.x.cpu().numpy())
        if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
            all_edge_attr.append(data.edge_attr.cpu().numpy())
        if hasattr(data, 'y') and data.y is not None:
            all_y.append(data.y.cpu().numpy())
        if hasattr(data, 'edge_y') and data.edge_y is not None and data.edge_y.shape[0] > 0:
            all_edge_y.append(data.edge_y.cpu().numpy())
    
    all_x = np.concatenate(all_x, axis=0) if all_x else np.array([])
    all_edge_attr = np.concatenate(all_edge_attr, axis=0) if all_edge_attr else np.array([])
    all_y = np.concatenate(all_y, axis=0) if all_y else np.array([])
    all_edge_y = np.concatenate(all_edge_y, axis=0) if all_edge_y else np.array([])
    
    # Determine layout
    num_node_features = all_x.shape[1] if all_x.size > 0 else 0
    num_edge_features = all_edge_attr.shape[1] if all_edge_attr.size > 0 else 0
    
    # Calculate number of plots needed
    num_plots = num_node_features + num_edge_features + 2  # +2 for node labels and edge labels
    nrows = (num_plots + 2) // 3  # 3 columns per row
    ncols = 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()
    ax_idx = 0
    
    # Plot node features
    if all_x.size > 0:
        for feat_idx in range(num_node_features):
            ax = axes[ax_idx]
            ax_idx += 1
            ax.hist(all_x[:, feat_idx], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_title(f'Node Feature {feat_idx}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            ax.grid(alpha=0.3)
    
    # Plot edge features
    if all_edge_attr.size > 0:
        for feat_idx in range(num_edge_features):
            ax = axes[ax_idx]
            ax_idx += 1
            ax.hist(all_edge_attr[:, feat_idx], bins=50, alpha=0.7, color='seagreen', edgecolor='black')
            ax.set_title(f'Edge Attr {feat_idx}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            ax.grid(alpha=0.3)
    
    # Plot node labels
    if all_y.size > 0:
        ax = axes[ax_idx]
        ax_idx += 1
        unique_y = np.unique(all_y)
        counts = np.bincount(all_y.astype(int))
        ax.bar(range(len(counts)), counts, alpha=0.7, color='coral', edgecolor='black')
        ax.set_title('Node Labels (y)')
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.grid(alpha=0.3, axis='y')
    
    # Plot edge labels
    if all_edge_y.size > 0:
        ax = axes[ax_idx]
        ax_idx += 1
        unique_edge_y = np.unique(all_edge_y)
        counts = np.bincount(all_edge_y.astype(int))
        ax.bar(range(len(counts)), counts, alpha=0.7, color='darkorange', edgecolor='black')
        ax.set_title('Edge Labels (edge_y)')
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.grid(alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(ax_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Feature Distributions Across Dataset')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved feature distributions to: {output_file}")
    
    if show_plots:
        try:
            plt.show()
        except Exception:
            pass  # Ignore show errors in non-interactive backends
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inspect and validate L1Nano datasets")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the saved dataset .pt file')
    parser.add_argument('--num_examples', type=int, default=6, help='Number of example graphs to plot')
    parser.add_argument('--output_prefix', type=str, default=None, help='Output prefix for saving plots as PNG')
    parser.add_argument('--no_show', action='store_true', help='Do not display plots (useful for non-interactive backends)')
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    print(f"Loading dataset from: {dataset_path}")
    dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)
    
    if isinstance(dataset, dict) and 'graphs' in dataset:
        dataset = dataset['graphs']
    
    if isinstance(dataset, list):
        print(f"✓ Loaded {len(dataset)} graphs")
    else:
        print(f"Dataset type: {type(dataset)}")
    
    # Print summary
    print_dataset_summary(dataset)
    
    # Plot examples
    graphs_file = None
    if args.output_prefix:
        graphs_file = f"{args.output_prefix}_example_graphs.png"
    show_plots = not args.no_show
    plot_example_graphs(dataset, num_examples=args.num_examples, output_file=graphs_file, show_plots=show_plots)
    
    # Plot features
    features_file = None
    if args.output_prefix:
        features_file = f"{args.output_prefix}_feature_distributions.png"
    plot_feature_distributions(dataset, output_file=features_file, show_plots=show_plots)
    
    print(f"\n{'='*60}")
    print("✓ Inspection complete!")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
