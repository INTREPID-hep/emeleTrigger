# Trigger with machine learning

## Installation

To generate the proper environment: 
```
virtualenv pyenv  --python=3.11
source <name_of_venv>/bin/activate
pip install .
```
If you have python 3.11 already installed in your system, you can also run `python3.11 -m venv pyenv` instead of the `virtualenv` command above. The list of dependencies can be seen in requirements.txt and installed by runing: `pip install -r requirements.txt`. In Lxplus, this line seems to have all the required dependencies: 
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh
```

## L1Nano Dataset Generation and Inspection

### Overview

The `L1NanoDataset` class processes L1 Trigger NanoAOD ROOT files and converts them into PyTorch Geometric graph datasets suitable for GNN training. Each event is converted into one or more graphs representing the stub-muon matching problem.

### Quick Start

#### 1. Generate a Dataset from L1Nano ROOT Files

Use the provided shell script to convert ROOT files into a `.pt` dataset:

```bash
cd /Users/folgueras/cernbox/L1T/2025_09_GNN_L1Nano/CMSSW_15_1_0_pre6/src/emeleTrigger
./tools/training/run_l1nano_save_dataset.sh <root_file_pattern> <max_events> <max_files> <output_path>
```

**Example:**
```bash
./tools/training/run_l1nano_save_dataset.sh \
  "/Users/folgueras/cernbox/L1T/2025_09_GNN_L1Nano/CMSSW_15_1_0_pre6/src/HTo2LongLivedTo4mu_*.root" \
  5000 \
  1 \
  l1nano_dataset_5k.pt
```

**Parameters:**
- `<root_file_pattern>`: Glob pattern or path to ROOT file(s)
- `<max_events>`: Maximum events to process per file (`-1` for all events)
- `<max_files>`: Maximum number of files to process
- `<output_path>`: Path where the `.pt` dataset will be saved

The script processes events with a progress bar and saves the dataset in PyTorch Geometric format.

#### 2. Inspect the Generated Dataset

Use the inspection script to validate dataset quality and visualize examples:

```bash
./tools/training/run_inspect_dataset.sh <dataset_path> <num_examples> <output_prefix>
```

**Example:**
```bash
./tools/training/run_inspect_dataset.sh \
  l1nano_dataset_5k.pt \
  6 \
  inspection
```

**Parameters:**
- `<dataset_path>`: Path to the `.pt` dataset file
- `<num_examples>`: Number of example graphs to plot (default: 6)
- `<output_prefix>`: Prefix for output PNG files (default: "inspection")

**Outputs:**
- `<output_prefix>_example_graphs.png`: Visualization of example graphs with hierarchical layout by tfLayer
- `<output_prefix>_feature_distributions.png`: Histograms of node features, edge features, and labels
- Console output with dataset statistics (number of graphs, avg nodes/edges, feature dimensions)

#### 3. Advanced Usage: Direct Python Interface

For more control, you can use the Python classes directly:

**Generate Dataset:**
```python
from tools.training.InputDataset import L1NanoDataset

dataset = L1NanoDataset(
    root_dir="/path/to/root/files/*.root",
    max_events=1000,
    max_files=1,
    debug=True  # Shows progress bar
)

# Save dataset
import torch
torch.save(dataset, "my_l1nano_dataset.pt")
```

**Inspect Dataset:**
```python
from tools.training.inspect_dataset import print_dataset_summary, plot_example_graphs, plot_feature_distributions
import torch

dataset = torch.load("my_l1nano_dataset.pt")

# Print statistics
print_dataset_summary(dataset)

# Plot example graphs
plot_example_graphs(dataset, num_examples=6, output_file="graphs.png")

# Plot feature distributions
plot_feature_distributions(dataset, output_file="features.png")
```

#### 4. Visualize Individual Events (Debug Mode)

To inspect a single event with detailed stub/edge information:

```bash
./tools/training/run_l1nano_visualization.sh
```

This script processes one event and creates:
- `stub_and_edge_info_event_0.png`: Distribution plots for stub features and edge attributes
- `graph_event_0.png`: Graph visualization for the first event

#### 5. Check Raw L1Nano Input Variables

To reproduce the variable checks done in the notebook directly from the plotting tools, use:

```bash
python tools/plotter/draw_variables.py \
  --mode l1nano \
  --ifile /path/to/file.root /path/to/file2.root \
  --tree Events \
  --ofolder output/l1nano_inputs/ \
  --max-events 100
```

If more than one ROOT file is passed, the script loads them all, merges the selected events, and produces a single combined set of plots.

This produces:
- `gen_muon_summary.png`: GenPart muon kinematics and displacement summary, using the same selection as the notebook.
- `stub_features_all.png`: All numeric `stub_*` inputs.
- `stub_features_qual1.png`: CSC-only stub inputs (`qual == 1`).
- `stub_features_qual2.png`: RPC-only stub inputs (`qual == 2`).
- `stub_features_qual3_endcap.png`: `qual == 3` stub inputs restricted to endcap stubs.

You can also use the wrapper script:

```bash
./tools/plotter/run_l1nano_input_plots.sh <root_file> <out_dir> <max_events> <tree_name>
```

Example:

```bash
./tools/plotter/run_l1nano_input_plots.sh \
  /path/to/file.root \
  output/l1nano_inputs \
  500 \
  Events
```

### Dataset Structure

Each graph in the dataset is a `torch_geometric.data.Data` object with:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `x` | `[N, 3]` | Node features: `[tfLayer, offeta1, offphi1]` |
| `edge_index` | `[2, E]` | Edge connectivity (COO format) |
| `edge_attr` | `[E, 2]` | Edge features: `[Δη, Δφ]` between connected stubs |
| `y` | `[N]` | Node labels: `1` (matched to muon), `0` (unmatched) |
| `edge_y` | `[E]` | Edge labels: `1` (both nodes matched to same muon), `0` (otherwise) |
| `matched_muon_idx` | `[N]` | Index of matched GenPart muon (`-1` if unmatched) |
| `stub_deltaR` | `[N]` | ΔR distance to matched muon (`-1` if unmatched) |
| `genpart_features` | `[M, 6]` | GenPart muon features: `[pt, etaSt2, phiSt2, charge, pdgId, statusFlags]` |

**Matching Logic:**
- Uses propagated GenPart muons to MB2 station (`GenPart_etaSt2`, `GenPart_phiSt2`)
- Filters: `|pdgId| == 13`, `statusFlags` bit 13 set, `pt > 1 GeV`, `etaSt2 > -999`
- Matching threshold: ΔR < 0.3

**Edge Construction:**
- Connects stubs in consecutive `tfLayer` values
- Rectangular cuts: `|Δη| < 0.5`, `|Δφ| < 1.0`
- Fallback: if no match in next layer, tries next-next layer

### Troubleshooting

**Import Errors:**
Make sure you're in the conda environment with all dependencies:
```bash
conda activate cmsl1t
```

**ROOT File Not Found:**
Verify the file path and that the ROOT file contains the required branches:
```bash
python -c "import uproot; print(uproot.open('file.root')['Events'].keys())"
```

**Empty Dataset:**
Check that events have both stubs and matched GenPart muons. Use `--debug` flag to see detailed progress.

## Graph / Dataset creation (OMTF - please ignore it for now)

An example of its utilization can be seen in `test/test_omtf_dataset.py` or you can also run it interactively: 

```
python tools/training/OMTFDataset.py --root_dir /eos/cms/store/user/folguera/L1TMuon/INTREPID/Dumper_Ntuples_v240725/Dumper_l1omtf_001.root --tree_name simOmtfPhase2Digis/OMTFHitsTree --muon_vars "[muonQOverPt]" --stub_vars "['stubEtaG', 'stubPhiG', 'stubR', 'stubLayer', 'stubType']" --save_path /eos/cms/store/user/folguera/L1TMuon/INTREPID/GraphsDataset_v240725_250128//OmtfDataset_Jan27_muonQOverPt_001.pt
```

### Using lxplus batch system
```
python test/submitJobs_DatasetCreation.py 
```
## Training

An example of use can be seen in `test/test_omtf_training.py` or you can also run it interactively: 

```
python tools/training/TrainModelFromGraph.py --model_type SAGE --hidden_dim 32 --normalization NodesAndEdgesAndOnlySpatial --graph_path /eos/cms/store/user/folguera/L1TMuon/INTREPID/Graphs_v240725_250115/ --out_model_path ./Test_Model_wGraphs_v240725_250115/ --do_train --save_tag SAGE_NodesAndEdgesAndOnlySpatial_Bsize64_lr5e-4_241106_20files_3neighbours --batch_size 1024 --learning_rate 0.001 --num_files 20 --graph_name vix_graph_6Nov_all_muonQOverPt --epochs 10
```
### Using lxplus batch system

This script, ``test/submitJobs_training.py`` needs some pre-configuration on the types of models, input and output folder and graph names. 

```########   YOU ONLY NEED TO FILL THE AREA BELOW   #########
########   customization  area #########
InputFolder = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Graphs_v240725_250115/" # list with all the file directories
queue = "workday"  (8h)  # Other options can be checked in: https://batchdocs.web.cern.ch/local/submit.html#job-flavours
OutputDir = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Model_Graphsv240725_QOverPtRegression_250115/"
WORKDIR = "/afs/cern.ch/user/f/folguera/workdir/INTREPID/tmp/TrainingModel/"
ModelTypes = ['GCN', 'SAGE', 'MPNN']
NormalizationTypes = ['NodesAndEdgesAndOnlySpatial']
InputGraphs = [""] #"3neighbours_muonQOverPt/", "all_connections_muonQOverPt/"]
GraphName = "vix_graph_6Nov_all_muonQOverPt" 

Epochs = 50
########   customization end   #########
```

Once configured, it will automatically generate a tmp workdir to submit all the jobs to the batch system: 

``
python test/submitJobs_training.py 
``