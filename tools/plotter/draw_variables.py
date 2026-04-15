#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot


PI = float(np.pi)
L1NANO_GENPART_BRANCHES = [
    "pt",
    "eta",
    "phi",
    "mass",
    "pdgId",
    "dXY",
    "lXY",
    "vertX",
    "vertY",
    "vertZ",
    "status",
    "statusFlags",
    "etaSt2",
    "phiSt2",
    "etaSt1",
    "phiSt1",
]


def ensure_output_dir(outputfolder):
    Path(outputfolder).mkdir(parents=True, exist_ok=True)


def get_root_module():
    try:
        import ROOT  # pylint: disable=import-error
    except ImportError as exc:
        raise RuntimeError("ROOT is required for OMTF plotting mode") from exc
    ROOT.gROOT.SetBatch(True)
    return ROOT


def draw_single_vars(inputfile, outputfolder, treename, plots, cutname=""):
    assert os.path.isfile(inputfile), print("File is does not exist")

    ROOT = get_root_module()
    root_file = ROOT.TFile(inputfile)
    tree = root_file.Get(treename)

    if plots == "all":
        plot_list = []
        for branch in tree.GetListOfBranches():
            if "hits" in branch.GetName():
                continue
            if "killed" in branch.GetName():
                continue
            plot_list.append(branch.GetName())
    else:
        plot_list = plots.split(",")

    for plot in plot_list:
        canvas = ROOT.TCanvas()
        tree.Draw(plot, cutname)
        canvas.SaveAs(os.path.join(outputfolder, plot + ".png"))


def draw_correlations(inputfile, outputfolder, treename, cutname=""):
    assert os.path.isfile(inputfile), print("File is does not exist")

    ROOT = get_root_module()
    root_file = ROOT.TFile(inputfile)
    tree = root_file.Get(treename)

    ROOT.gStyle.SetPalette(1)
    canvas = ROOT.TCanvas()

    correlations = [
        ("muonEta:muonPhi", "muonEta_vs_muonPhi.png"),
        ("muonEta:muonPt", "muonEta_vs_muonPt.png"),
        ("muonPhi:muonPt", "muonPhi_vs_muonPt.png"),
        ("stubPhi:stubProc", "stubPhi_vs_stubProc.png"),
        ("stubPhi:stubType", "stubPhi_vs_stubType.png"),
        ("stubProc:stubType", "stubProc_vs_stubType.png"),
        ("stubPhi:stubQuality", "stubPhi_vs_stubQuality.png"),
        ("stubProc:omtfProcessor", "stubProc_vs_omtfProcessor.png"),
        ("stubTiming:stubQuality", "stubTiming_vs_stubQuality.png"),
        ("stubTiming:stubType", "stubTiming_vs_stubType.png"),
    ]

    for expression, filename in correlations:
        tree.Draw(expression, cutname, "colz")
        canvas.SaveAs(os.path.join(outputfolder, filename))


def deltaphi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > PI:
        dphi -= 2 * PI
    while dphi < -PI:
        dphi += 2 * PI
    return dphi


def flatten_numeric(values):
    try:
        flat = ak.flatten(values, axis=None)
    except Exception:
        flat = values

    try:
        array = ak.to_numpy(flat)
    except Exception:
        array = np.asarray(ak.to_list(flat))

    if array.size == 0:
        return np.array([])

    if not np.issubdtype(array.dtype, np.number):
        return np.array([])

    return array[np.isfinite(array)]


def build_genmuon_mask(events):
    mask = abs(events.GenPart.pdgId) == 13
    if hasattr(events.GenPart, "statusFlags"):
        mask = mask & ((events.GenPart.statusFlags & (1 << 13)) != 0)
    if hasattr(events.GenPart, "pt"):
        mask = mask & (events.GenPart.pt > 1)
    if hasattr(events.GenPart, "etaSt2"):
        mask = mask & (events.GenPart.etaSt2 > -999)
    return mask


def plot_gen_muon_summary(events, outputfolder):
    if not hasattr(events, "GenPart"):
        print("[WARNING] No GenPart collection found in Events tree")
        return

    mask_allmuons = build_genmuon_mask(events)
    n_muons_per_event = ak.num(events.GenPart.pdgId[mask_allmuons])

    muon_pt = flatten_numeric(events.GenPart.pt[mask_allmuons])
    muon_eta = flatten_numeric(events.GenPart.eta[mask_allmuons])
    muon_phi = flatten_numeric(events.GenPart.phi[mask_allmuons])
    muon_dxy = flatten_numeric(events.GenPart.dXY[mask_allmuons])
    muon_lxy = flatten_numeric(events.GenPart.lXY[mask_allmuons])
    muon_vx = flatten_numeric(events.GenPart.vertX[mask_allmuons])
    muon_vy = flatten_numeric(events.GenPart.vertY[mask_allmuons])
    muon_vz = flatten_numeric(events.GenPart.vertZ[mask_allmuons])

    total_muons = len(muon_pt)
    if total_muons == 0:
        print("[WARNING] No generated muons passed the L1Nano selection")
        return

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle("Generated Muons Analysis", fontsize=16)
    axes = axes.flatten()

    n_muons_np = ak.to_numpy(n_muons_per_event)
    axes[0].hist(
        n_muons_np,
        bins=range(0, int(np.max(n_muons_np)) + 2),
        alpha=0.7,
        edgecolor="black",
        color="steelblue",
    )
    axes[0].set_xlabel("Number of muons per event", fontsize=14)
    axes[0].set_ylabel("Number of events", fontsize=14)
    axes[0].set_title("Number of Generated Muons per Event", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    plot_specs = [
        (axes[1], muon_pt, 50, (0, 100), "orange", "pT [GeV]", "Generated Muon pT Distribution"),
        (axes[2], muon_eta, 50, None, "green", "eta", "Generated Muon eta Distribution"),
        (axes[3], muon_phi, 50, (-3.5, 3.5), "purple", "phi [rad]", "Generated Muon phi Distribution"),
        (axes[4], muon_dxy, 50, (-1000, 1000), "brown", "dXY [cm]", "Generated Muon dXY Distribution"),
        (axes[5], muon_lxy, 50, (0, 2000), "teal", "lXY [cm]", "Generated Muon lXY Distribution"),
        (axes[6], muon_vx, 50, (-1000, 1000), "coral", "Vertex X [cm]", "Generated Muon Vertex X Distribution"),
        (axes[7], muon_vy, 50, (-1000, 1000), "gold", "Vertex Y [cm]", "Generated Muon Vertex Y Distribution"),
        (axes[8], muon_vz, 50, (-3000, 3000), "limegreen", "Vertex Z [cm]", "Generated Muon Vertex Z Distribution"),
    ]

    for axis, values, bins, hist_range, color, xlabel, title in plot_specs:
        hist_kwargs = {
            "bins": bins,
            "alpha": 0.7,
            "edgecolor": "black",
            "color": color,
        }
        if hist_range is not None:
            hist_kwargs["range"] = hist_range
        axis.hist(values, **hist_kwargs)
        axis.set_xlabel(xlabel, fontsize=14)
        axis.set_ylabel("Number of muons", fontsize=14)
        axis.set_title(title, fontsize=14)
        axis.grid(True, alpha=0.3)

    plt.tight_layout()
    output = os.path.join(outputfolder, "gen_muon_summary.png")
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Generated Muons Statistics ===")
    print(f"Total events: {len(n_muons_np)}")
    print(f"Events with muons: {int(np.sum(n_muons_np > 0))} ({100 * np.mean(n_muons_np > 0):.2f}%)")
    print(f"Total muons: {total_muons}")
    print(f"Average muons per event: {np.mean(n_muons_np):.2f}")
    print(f"Muon pT: min={np.min(muon_pt):.2f} GeV, max={np.max(muon_pt):.2f} GeV, mean={np.mean(muon_pt):.2f} GeV")
    print(f"Muon eta: min={np.min(muon_eta):.2f}, max={np.max(muon_eta):.2f}, mean={np.mean(muon_eta):.2f}")
    print(f"Muon phi: min={np.min(muon_phi):.2f} rad, max={np.max(muon_phi):.2f} rad, mean={np.mean(muon_phi):.2f} rad")
    print(f"Muon dXY: min={np.min(muon_dxy):.3f} cm, max={np.max(muon_dxy):.3f} cm, mean={np.mean(muon_dxy):.3f} cm")
    print(f"Muon lXY: min={np.min(muon_lxy):.3f} cm, max={np.max(muon_lxy):.3f} cm, mean={np.mean(muon_lxy):.3f} cm")
    print(f"Saved {output}")


def collect_stub_feature_arrays(stub_collection, mask=None):
    feature_arrays = {}
    plot_features = []

    for feat in ak.fields(stub_collection):
        try:
            values = stub_collection[feat]
            if mask is not None:
                values = values[mask]
            array = flatten_numeric(values)
        except Exception:
            continue

        if array.size == 0:
            continue

        plot_features.append(feat)
        feature_arrays[feat] = array

    return plot_features, feature_arrays


def plot_stub_features(events, outputfolder, qual_value=None, require_endcap=False):
    if not hasattr(events, "stub"):
        print("[WARNING] No stub collection found in Events tree")
        return

    selection_mask = None
    suffix = "all"
    title = "All Available Stub Feature Distributions"
    color = "steelblue"

    if qual_value is not None:
        if not hasattr(events.stub, "qual"):
            print("[WARNING] stub.qual is not available, quality-specific plots skipped")
            return
        selection_mask = events.stub.qual == qual_value
        suffix = f"qual{qual_value}"
        title = f"Stub Feature Distributions for qual == {qual_value}"
        color = "darkorange" if qual_value == 2 else "seagreen"
        if qual_value == 3:
            color = "mediumpurple"
        if require_endcap:
            if not hasattr(events.stub, "isEndcap"):
                print("[WARNING] stub.isEndcap is not available, qual==3 endcap plot skipped")
                return
            selection_mask = selection_mask & events.stub.isEndcap
            suffix = f"qual{qual_value}_endcap"
            title = f"Stub Feature Distributions for qual == {qual_value} and isEndcap"

    plot_features, feature_arrays = collect_stub_feature_arrays(events.stub, selection_mask)
    if not plot_features:
        print(f"[WARNING] No plottable numeric stub features found for selection {suffix}")
        return

    n_feat = len(plot_features)
    ncols = 4
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, feat in enumerate(plot_features):
        axis = axes[idx]
        axis.hist(feature_arrays[feat], bins=50, alpha=0.75, edgecolor="black", color=color)
        axis.set_title(feat, fontsize=11)
        axis.set_xlabel("Value")
        axis.set_ylabel("Counts")
        axis.grid(True, alpha=0.25)

    for idx in range(n_feat, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    output = os.path.join(outputfolder, f"stub_features_{suffix}.png")
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    selected_stubs = len(next(iter(feature_arrays.values()))) if feature_arrays else 0
    print(f"Plotted {n_feat} stub features for selection {suffix} ({selected_stubs} flattened entries).")
    print(f"Saved {output}")


def print_l1nano_summary(events):
    stub_fields = ak.fields(events.stub) if hasattr(events, "stub") else []
    genpart_fields = ak.fields(events.GenPart) if hasattr(events, "GenPart") else []

    print("\n=== L1Nano input summary ===")
    print(f"Events loaded: {len(events)}")
    print(f"stub fields ({len(stub_fields)}): {', '.join(stub_fields)}")
    available_genpart_fields = [field for field in L1NANO_GENPART_BRANCHES if field in genpart_fields]
    print(f"GenPart fields used ({len(available_genpart_fields)}): {', '.join(available_genpart_fields)}")

    if hasattr(events, "stub") and hasattr(events.stub, "qual"):
        qual_values = flatten_numeric(events.stub.qual)
        if qual_values.size > 0:
            unique_quals, counts = np.unique(qual_values.astype(int), return_counts=True)
            print("Stub quality counts:")
            for qual, count in zip(unique_quals, counts):
                print(f"  qual == {qual}: {count}")


def draw_l1nano_variables(inputfile, outputfolder, treename, max_events=None):
    assert os.path.isfile(inputfile), print("File does not exist")

    tree = uproot.open(f"{inputfile}:{treename}")
    events = tree.arrays(
        filter_name=["stub_*", "GenPart_*"],
        how="zip",
        library="ak",
        entry_stop=max_events,
    )

    print_l1nano_summary(events)
    plot_gen_muon_summary(events, outputfolder)
    plot_stub_features(events, outputfolder)
    plot_stub_features(events, outputfolder, qual_value=1)
    plot_stub_features(events, outputfolder, qual_value=2)
    plot_stub_features(events, outputfolder, qual_value=3, require_endcap=True)


def main():
    parser = argparse.ArgumentParser(
        usage="draw_variables.py [options]",
        description="Draw input variables for OMTF or L1Nano studies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-i", "--ifile", dest="inputfile", default="data/omtfAnalysis2.root", help="Input filename")
    parser.add_argument("-o", "--ofolder", dest="output", default="output/omtfAnalysis2/", help="Folder name to store results")
    parser.add_argument("-t", "--tree", dest="tree", default="simOmtfDigis/OMTFHitsTree", help="Tree name")
    parser.add_argument("-p", "--plots", dest="plots", default="all", help="Plots to be made in OMTF mode")
    parser.add_argument("-m", "--mode", dest="mode", choices=["omtf", "l1nano"], default="omtf", help="Input format to inspect")
    parser.add_argument("--max-events", dest="max_events", type=int, default=None, help="Maximum number of events to read in l1nano mode")

    args = parser.parse_args()

    ensure_output_dir(args.output)

    print(f"Running on: {args.inputfile}")
    print(f"Saving result in: {args.output}")
    print(f"Mode: {args.mode}")

    if args.mode == "l1nano":
        if args.tree == parser.get_default("tree"):
            args.tree = "Events"
        draw_l1nano_variables(args.inputfile, args.output, args.tree, args.max_events)
    else:
        draw_single_vars(args.inputfile, args.output, args.tree, args.plots, "muonPropEta!=0&&muonPropPhi!=0")
        draw_correlations(args.inputfile, args.output, args.tree, "muonPropEta!=0&&muonPropPhi!=0")

    print("DONE")


if __name__ == "__main__":
    main()
