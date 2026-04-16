import os
import uproot
import torch
from torch_geometric.data import Dataset, Data
import awkward as ak
import networkx as nx
from operator import xor

from torch_geometric.utils.convert import to_networkx

import numpy as np
import matplotlib.pyplot as plt

import yaml

try:
    from converter import remove_empty_or_nan_graphs
except ImportError:
    def remove_empty_or_nan_graphs(data):
        """Placeholder function if converter is not available"""
        return data if data.x.shape[0] > 0 else None

''' Dataset classes for L1Nano and OMTF data '''

class L1NanoDataset(Dataset):
    def __init__(self, **kwargs):
        """
        PyTorch Dataset para leer eventos de L1Nano ROOT files.
        
        Parámetros:
            root_dir (str): Directorio que contiene los archivos ROOT o ruta a un archivo ROOT.
            tree_name (str): Nombre del árbol (default: "Events").
            dR_threshold (float): Umbral de dR para matching geométrico (default: 0.15).
            max_files (int): Número máximo de archivos a procesar.
            max_events (int): Número máximo de eventos a procesar.
            debug (bool): Modo debug.
            pre_transform (callable): Función de pre-transformación.
            transform (callable): Función de transformación (opcional).
        """
        config_file = kwargs.get("config")
        if config_file is not None:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            for key, value in config.items():
                print(f"Setting {key} from config file: {value}")
                kwargs[key] = value

        self.root_dir      = kwargs.get("root_dir")
        self.tree_name     = kwargs.get("tree_name", "Events")
        
        # Definir features de stubs y GenPart
        self.stub_vars     = kwargs.get("stub_vars", [
            'tfLayer', 'offeta1', 'offphi1'
        ])
        self.genpart_vars  = kwargs.get("genpart_vars", [
            'pt', 'eta', 'phi', 'mass', 'pdgId', 'dXY', 'lXY', 'etaSt2', 'phiSt2'
        ])
        
        self.dR_threshold  = kwargs.get("dR_threshold", 0.15)
        self.task          = kwargs.get("task", "classification")  # classification or regression
        self.max_files     = kwargs.get("max_files", None)
        self.max_events    = kwargs.get("max_events", None)
        self.debug         = kwargs.get("debug", False)
        self.edge_deta_threshold = kwargs.get("edge_deta_threshold", 0.5)
        self.edge_dphi_threshold = kwargs.get("edge_dphi_threshold", 1.0)
        self.pre_transform = kwargs.get("pre_transform")
        self.transform     = kwargs.get("transform")

        if "dataset" in kwargs and kwargs["dataset"] is not None:
            self.dataset = kwargs["dataset"]
        else:
            self.dataset = self.load_data_from_root()

    def add_extra_vars_to_tree(self, arr):
        """
        Adds extra variables to the L1Nano tree.
        For L1Nano, we mostly work with stub and GenPart info directly.
        """
        return arr
    
    def load_data_from_root(self):
        """
        Load events from L1Nano ROOT files and create graph data.
        """
        data_list = []
        files_processed = 0
        events_processed = 0
        events_seen = 0
        events_skipped_no_stubs = 0
        events_skipped_no_edges = 0
        events_skipped_nan = 0
        events_skipped_pretransform = 0

        # Check if root_dir is a directory or a file
        if os.path.isdir(self.root_dir):
            root_files = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith(".root")]
        elif os.path.isfile(self.root_dir) and self.root_dir.endswith(".root"):
            root_files = [self.root_dir]
        else:
            raise ValueError(f"{self.root_dir} is not a valid directory or ROOT file")

        # Define branches to load
        stub_branches = [f"stub_{var}" for var in self.stub_vars]
        genpart_branches = [f"GenPart_{var}" for var in self.genpart_vars]
        all_branches = stub_branches + genpart_branches + ['GenPart_statusFlags']
        
        for root_file in root_files:
            print(f"Processing file: {root_file}")
            if self.max_files is not None and files_processed >= self.max_files:
                break
            
            try:
                file = uproot.open(root_file)
                tree = file[self.tree_name]
                
                # Load data using arrays with 'how=zip'
                events_data = tree.arrays(
                    filter_name=all_branches,
                    how="zip",
                    library="ak"
                )
                
                # Convert to list for iteration
                num_events = len(events_data)
                print(f"  Found {num_events} events in file")
                
                for event_idx in range(num_events):
                    if self.max_events is not None and self.max_events >= 0 and events_processed >= self.max_events:
                        break
                    
                    event = events_data[event_idx]
                    events_seen += 1
                    
                    # Extract stub features
                    stub_features = self._extract_stub_features(event)
                    
                    # Extract GenPart features (muons only)
                    genpart_features = self._extract_genpart_features(event)
                    
                    # Matching labels following TrainL1Nano_v2.ipynb logic
                    stub_labels, stub_deltaR, stub_matched_muon_idx = self._match_stubs_to_genpart(event)
                    
                    # Skip events with no stubs
                    if stub_features.shape[0] == 0:
                        events_skipped_no_stubs += 1
                        continue
                    
                    # Build graph with notebook-like logic (valid labels + edges)
                    graph_data = self._build_graph_for_event(
                        stub_features,
                        stub_labels,
                        stub_deltaR,
                        stub_matched_muon_idx,
                        genpart_features,
                    )

                    if graph_data is None:
                        events_skipped_pretransform += 1
                        continue
                    
                    if graph_data.edge_index.size(1) == 0:
                        events_skipped_no_edges += 1

                    has_nan = (
                        torch.isnan(graph_data.x).any() or
                        (graph_data.edge_attr is not None and torch.isnan(graph_data.edge_attr).any()) or
                        (graph_data.y is not None and torch.is_floating_point(graph_data.y) and torch.isnan(graph_data.y).any())
                    )
                    if has_nan:
                        events_skipped_nan += 1
                    
                    # Apply pre-transform if provided
                    if self.pre_transform is not None:
                        graph_data = self.pre_transform(graph_data)
                        if graph_data is None:
                            events_skipped_pretransform += 1
                            continue
                    
                    data_list.append(graph_data)
                    events_processed += 1
                    
                    if self.debug and events_processed % 100 == 0:
                        print(f"  Processed {events_processed} events")
                
                files_processed += 1
                file.close()
                
            except Exception as e:
                print(f"  Error processing file {root_file}: {e}")
                continue
        
        print(f"\nTotal events loaded: {len(data_list)}")
        if self.debug:
            print("Dataset debug summary:")
            print(f"  events seen              : {events_seen}")
            print(f"  skipped (no stubs)       : {events_skipped_no_stubs}")
            print(f"  skipped (no edges)       : {events_skipped_no_edges}")
            print(f"  skipped (NaN)            : {events_skipped_nan}")
            print(f"  skipped (pre_transform)  : {events_skipped_pretransform}")
        return data_list

    def _extract_stub_features(self, event):
        """
        Extract stub features as tensor [num_stubs, num_features].
        """
        # Access stub arrays directly from event.stub
        num_stubs = len(event.stub.offeta1) if hasattr(event.stub, 'offeta1') else 0
        
        if num_stubs == 0:
            return torch.zeros((0, len(self.stub_vars)), dtype=torch.float32)
        
        features = []
        for feat in self.stub_vars:
            if hasattr(event.stub, feat):
                feat_array = getattr(event.stub, feat)
                feat_tensor = torch.tensor(self._ak_to_numpy_safe(feat_array), dtype=torch.float32)
            else:
                feat_tensor = torch.zeros(num_stubs, dtype=torch.float32)
            features.append(feat_tensor)
        
        stub_tensor = torch.stack(features, dim=1)
        return stub_tensor
    
    def _extract_genpart_features(self, event):
        """
        Extract GenPart features (muons only) as tensor [num_muons, num_features].
        """
        # Filter for muons: abs(pdgId) == 13 and isLastCopy
        pdgIds = event.GenPart.pdgId if hasattr(event.GenPart, 'pdgId') else []
        
        if len(pdgIds) == 0:
            return torch.zeros((0, len(self.genpart_vars)), dtype=torch.float32)
        
        mask_muons = abs(pdgIds) == 13
        
        # Filter by statusFlags (bit 13 = isLastCopy)
        if hasattr(event.GenPart, 'statusFlags'):
            statusFlags = event.GenPart.statusFlags
            mask_lastcopy = (statusFlags & (1 << 13)) != 0
            mask_muons = mask_muons & mask_lastcopy

        if hasattr(event.GenPart, 'pt'):
            mask_muons = mask_muons & (event.GenPart.pt > 1)

        if hasattr(event.GenPart, 'etaSt2'):
            mask_muons = mask_muons & (event.GenPart.etaSt2 > -999)
        
        if ak.sum(mask_muons) == 0:
            return torch.zeros((0, len(self.genpart_vars)), dtype=torch.float32)
        
        features = []
        for feat in self.genpart_vars:
            if hasattr(event.GenPart, feat):
                feat_array = getattr(event.GenPart, feat)[mask_muons]
                feat_tensor = torch.tensor(self._ak_to_numpy_safe(feat_array), dtype=torch.float32)
            else:
                num_muons = ak.sum(mask_muons)
                feat_tensor = torch.zeros(num_muons, dtype=torch.float32)
            features.append(feat_tensor)
        
        genpart_tensor = torch.stack(features, dim=1) if features else torch.zeros((ak.sum(mask_muons), 0), dtype=torch.float32)
        return genpart_tensor
    
    def _match_stubs_to_genpart(self, event):
        """
        Matching logic mimicking TrainL1Nano_v2.ipynb.
        Returns:
            - labels: 1 (matched), 0 (no match), -1 (no muons in event)
            - deltaR: minimum dR for each stub
            - matched indices: index of matched muon in the event-level filtered list
        """
        # Use offeta1/offphi1 as in notebook
        if not hasattr(event.stub, 'offeta1') or not hasattr(event.stub, 'offphi1'):
            num_stubs = len(event.stub.eta1) if hasattr(event.stub, 'eta1') else 0
            return (
                torch.full((num_stubs,), -1, dtype=torch.float32),
                torch.full((num_stubs,), 999.0, dtype=torch.float32),
                torch.full((num_stubs,), -1, dtype=torch.float32),
            )

        stub_eta = self._ak_to_numpy_safe(event.stub.offeta1)
        stub_phi = self._ak_to_numpy_safe(event.stub.offphi1)
        
        num_stubs = len(stub_eta)
        
        # Get gen muons
        pdgIds = event.GenPart.pdgId if hasattr(event.GenPart, 'pdgId') else []
        if len(pdgIds) == 0:
            return (
                torch.full((num_stubs,), -1, dtype=torch.float32),
                torch.full((num_stubs,), 999.0, dtype=torch.float32),
                torch.full((num_stubs,), -1, dtype=torch.float32),
            )
        
        mask_muons = abs(pdgIds) == 13
        if hasattr(event.GenPart, 'statusFlags'):
            statusFlags = event.GenPart.statusFlags
            mask_lastcopy = (statusFlags & (1 << 13)) != 0
            mask_muons = mask_muons & mask_lastcopy

        if hasattr(event.GenPart, 'pt'):
            mask_muons = mask_muons & (event.GenPart.pt > 1)

        if hasattr(event.GenPart, 'etaSt2'):
            mask_muons = mask_muons & (event.GenPart.etaSt2 > -999)
        
        if ak.sum(mask_muons) == 0:
            return (
                torch.full((num_stubs,), -1, dtype=torch.float32),
                torch.full((num_stubs,), 999.0, dtype=torch.float32),
                torch.full((num_stubs,), -1, dtype=torch.float32),
            )
        
        if hasattr(event.GenPart, 'etaSt2') and hasattr(event.GenPart, 'phiSt2'):
            muon_eta = self._ak_to_numpy_safe(event.GenPart.etaSt2[mask_muons])
            muon_phi = self._ak_to_numpy_safe(event.GenPart.phiSt2[mask_muons])
        else:
            muon_eta = self._ak_to_numpy_safe(event.GenPart.eta[mask_muons])
            muon_phi = self._ak_to_numpy_safe(event.GenPart.phi[mask_muons])

        labels = np.full(num_stubs, -1, dtype=np.float32)
        min_deltaR = np.full(num_stubs, 999.0, dtype=np.float32)
        matched_indices = np.full(num_stubs, -1, dtype=np.float32)
        
        for i in range(num_stubs):
            min_dR = float('inf')
            matched_idx = -1
            
            for j in range(len(muon_eta)):
                deta = stub_eta[i] - muon_eta[j]
                dphi = self._deltaphi(stub_phi[i], muon_phi[j])
                
                dR = np.sqrt(deta**2 + dphi**2)
                
                if dR < min_dR:
                    min_dR = dR
                    matched_idx = j
            if min_dR < self.dR_threshold:
                labels[i] = 1.0
                min_deltaR[i] = min_dR
                matched_indices[i] = float(matched_idx)
            else:
                labels[i] = 0.0
                min_deltaR[i] = min_dR

        return (
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(min_deltaR, dtype=torch.float32),
            torch.tensor(matched_indices, dtype=torch.float32),
        )

    def _build_graph_for_event(self, stub_features, stub_labels, stub_deltaR, stub_matched_idx, genpart_features):
        # Keep only stubs with defined label (>=0), as in notebook
        valid_mask = stub_labels >= 0
        if valid_mask.sum() == 0:
            return None

        x_full = torch.cat(
            [
                stub_features,
                stub_labels.unsqueeze(1),
                stub_deltaR.unsqueeze(1),
                stub_matched_idx.unsqueeze(1),
            ],
            dim=1,
        )
        x_full = x_full[valid_mask]

        nodes = x_full[:, :3]  # tfLayer, offeta1, offphi1
        node_labels = x_full[:, 3].clone()  # 1/0 labels
        matched_idx = x_full[:, 5].clone()  # matched muon indices

        edge_index, edge_attr, edge_y = self._create_edges_by_layer(nodes, matched_idx)

        return Data(
            x=nodes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_y=edge_y,
            y=node_labels,
            genpart_features=genpart_features,
            num_nodes=nodes.shape[0],
        )
    
    def _create_edges_by_layer(self, node_features, matched_idx):
        """
        Create edges mimicking TrainL1Nano_v2.ipynb logic:
          - connect between consecutive layers
          - if a node is not connected to next layer, try the subsequent layer
          - connect when abs(deta) < 0.5 and abs(dphi) < 1.0
          - edge label = 1 if both stubs are matched to the same muon index (>=0)
        """
        num_stubs = node_features.shape[0]
        
        if num_stubs == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 2), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
            )
        
        # node features are [tfLayer, offeta1, offphi1]
        tflayer = node_features[:, 0].numpy()
        eta = node_features[:, 1].numpy()
        phi = node_features[:, 2].numpy()
        matched_np = matched_idx.numpy()
        
        edge_index = []
        edge_attr = []
        edge_labels = []
        
        unique_layers = np.unique(tflayer)
        
        # Connect stubs between consecutive layers
        for layer_idx in range(len(unique_layers) - 1):
            current_layer = unique_layers[layer_idx]
            next_layer = unique_layers[layer_idx + 1]
            next_next_layer = unique_layers[layer_idx + 2] if layer_idx + 2 < len(unique_layers) else None
            
            current_indices = np.where(tflayer == current_layer)[0]
            next_indices = np.where(tflayer == next_layer)[0]
            next_next_indices = np.where(tflayer == next_next_layer)[0] if next_next_layer is not None else np.array([], dtype=int)
            
            # Connect stubs in next layer first, then fallback to next-next layer
            for i in current_indices:
                connected_to_next = False
                for j in next_indices:
                    deta = eta[i] - eta[j]
                    dphi = self._deltaphi(phi[i], phi[j])

                    if abs(deta) < self.edge_deta_threshold and abs(dphi) < self.edge_dphi_threshold:
                        edge_index.append([i, j])
                        edge_attr.append([deta, dphi])
                        same_muon = matched_np[i] >= 0 and matched_np[j] >= 0 and matched_np[i] == matched_np[j]
                        edge_labels.append(1.0 if same_muon else 0.0)
                        connected_to_next = True

                if (not connected_to_next) and len(next_next_indices) > 0:
                    for k in next_next_indices:
                        deta = eta[i] - eta[k]
                        dphi = self._deltaphi(phi[i], phi[k])

                        if abs(deta) < self.edge_deta_threshold and abs(dphi) < self.edge_dphi_threshold:
                            edge_index.append([i, k])
                            edge_attr.append([deta, dphi])
                            same_muon = matched_np[i] >= 0 and matched_np[k] >= 0 and matched_np[i] == matched_np[k]
                            edge_labels.append(1.0 if same_muon else 0.0)
        
        if len(edge_index) == 0:
            edge_index_t = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_t = torch.zeros((0, 2), dtype=torch.float32)
            edge_y_t = torch.zeros((0,), dtype=torch.float32)
        else:
            edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
            edge_y_t = torch.tensor(edge_labels, dtype=torch.float32)
        
        return edge_index_t, edge_attr_t, edge_y_t

    @staticmethod
    def _deltaphi(phi1, phi2):
        dphi = phi1 - phi2
        while dphi > np.pi:
            dphi -= 2 * np.pi
        while dphi < -np.pi:
            dphi += 2 * np.pi
        return dphi
        
    @staticmethod
    def _ak_to_numpy_safe(values):
        try:
            return ak.to_numpy(values)
        except Exception:
            return np.asarray(ak.to_list(values), dtype=np.float32)
    

    def len(self):
        return len(self.dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def get(self, idx):
        data = self.dataset[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __getitem__(self, idx):
        return self.get(idx)
    
    def __repr__(self):
        return '({}({})'.format(self.__class__.__name__, len(self))
    
    def __str__(self):
        return '({}({})'.format(self.__class__.__name__, len(self))
    
    def plot_graph(self, idx, filename=None, seed=42):
        data = self.get(idx)
        
        G = to_networkx(data, to_undirected=True)
        labels = {i: int(data.x[i, 3].item()) for i in range(data.x.shape[0])}

        '''G = nx.Graph()
        
        for i, node_feature in enumerate(data.x):
            G.add_node(i, feature=node_feature.tolist())
        
        edge_index = data.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i]) #, weight=data.edge_weight[i].item() if 'edge_weight' in data else 1.0)
        '''
        pos = nx.spring_layout(G,seed=seed)
        nx.draw(G, pos, with_labels=True, labels=labels, node_color='skyblue', node_size=500)
        #nx.draw_networkx_edge_labels(G, pos) # edge_labels=edge_labels)
        
        plt.title(f'Grafo de ejemplo del índice {idx}')
        plt.show()
        if filename is not None:
            plt.savefig(filename)
        plt.close()

    def plot_example_graphs(self, filename=None, seed=42):
        """
        Escanea el dataset y muestra grafos de ejemplo con diferentes números de nodos (de 4 a 10).
        Etiqueta los nodos con la cuarta columna de data.x.
        """
        print('Drawing example graphs into ', filename)
        # draw a figure with 6 subplots

        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.ravel()
        ax = plt.gca()
        ax.margins(0.08)
    
        num_nodes_list = range(3, 11)
        found_graphs = {num_nodes: False for num_nodes in num_nodes_list}

        for index, data in enumerate(self.dataset):
            num_nodes = data.x.shape[0]
            if num_nodes in num_nodes_list and not found_graphs[num_nodes]:
                #G = to_networkx(data, to_undirected=True)

                G = nx.Graph()
                
                # Añadir nodos con etiquetas de la cuarta columna de data.x
                for i, node_feature in enumerate(data.x):
                    G.add_node(i, label=int(node_feature[3].item()))
                
                # Añadir aristas
                edge_index = data.edge_index.numpy()
                for i in range(edge_index.shape[1]):
                    G.add_edge(edge_index[0, i], edge_index[1, i])
                
                # Dibujar el grafo
                pos = nx.spring_layout(G,seed=seed)
                labels = nx.get_node_attributes(G, 'label')
                nx.draw(G, pos, labels=labels, node_color='skyblue', edge_color='gray', ax=axs[num_nodes-3])
                
                # Dibujar etiquetas de los pesos de las aristas
                #edge_labels = nx.get_edge_attributes(G, 'weight')
                #nx.draw_networkx_edge_labels(G, pos, ax=axs[num_nodes-3])
                
                #plt.title(f'Grafo de ejemplo del índice {index} con {num_nodes} nodos')
                #plt.show()
                axs[num_nodes-3].set_title(f"{G}") 
                axs[num_nodes-3].axis("off")

                found_graphs[num_nodes] = True
                if all(found_graphs.values()):
                    break  # Salir del bucle si se han encontrado grafos para todos los números de nodos

        plt.tight_layout()
        plt.show()
        if filename is not None:
            plt.savefig(filename)
        plt.close()  # Cerrar la gráfica automáticamente


    def save_dataset(self, file_path):
        torch.save(self.dataset, file_path)
        print(f"Dataset guardado en {file_path}")

    @staticmethod
    def load_dataset(file_path):
        dataset = torch.load(file_path, weights_only=False)
        print(f"Dataset cargado desde {file_path}")
        return L1NanoDataset(dataset=dataset)

def main():
    import argparse
    from torch_geometric.loader import DataLoader

    parser = argparse.ArgumentParser(description="Load L1Nano ROOT files and create a PyTorch Geometric dataset")
    parser.add_argument('--config', type=str, help='Path to the configuration file with parameters')
    parser.add_argument('--root_dir', type=str, required=True, help='Directory containing the ROOT files or path to a single ROOT file')
    parser.add_argument('--tree_name', type=str, default="Events", help='Name of the tree inside the ROOT files')
    parser.add_argument('--stub_vars', nargs='+', type=str, 
                        default=['tfLayer', 'offeta1', 'offphi1'],
                        help='List of stub variables to extract')
    parser.add_argument('--genpart_vars', nargs='+', type=str,
                        default=['pt', 'eta', 'phi', 'mass', 'pdgId', 'dXY', 'lXY', 'etaSt2', 'phiSt2'],
                        help='List of GenPart variables to extract')
    parser.add_argument('--dR_threshold', type=float, default=0.15, help='dR threshold for stub-muon matching')
    parser.add_argument('--edge_deta_threshold', type=float, default=0.5, help='abs(deta) threshold for edge building')
    parser.add_argument('--edge_dphi_threshold', type=float, default=1.0, help='abs(dphi) threshold for edge building')
    parser.add_argument('--task', type=str, default='classification', help='Task type (classification or regression)')
    parser.add_argument('--plot_example', action='store_true', help='Plot an example graph')
    parser.add_argument('--save_path', type=str, help='Path to save the dataset')
    parser.add_argument('--load_path', type=str, help='Path to load the dataset')
    parser.add_argument('--max_files', type=int, help='Maximum number of files to process')
    parser.add_argument('--max_events', type=int, help='Maximum number of events to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    if args.load_path:
        dataset = L1NanoDataset.load_dataset(args.load_path)
    else:
        pre_transformation = remove_empty_or_nan_graphs
        dataset = L1NanoDataset(pre_transform=pre_transformation, **vars(args))

        if args.save_path:
            dataset.save_dataset(args.save_path)

    if len(dataset) == 0:
        print("Dataset loaded with 0 events")
        print("No se puede crear DataLoader con dataset vacío. Revisa el resumen debug para ver por qué se filtraron los eventos.")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Dataset loaded with {len(dataset)} events")
    print(f"First event: {dataset[0]}")

    if args.plot_example and len(dataset) > 0:
        dataset.plot_graph(idx=0)

if __name__ == "__main__":
    main()
