#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot


PI = float(np.pi)
L1NANO_GenMuon_BRANCHES = [
    "pt",
    "eta",
    "phi",
    "mass",
    "pdgId",
    "dXY",
    "lXY",
    "vx",
    "vy",
    "vz",
    "status",
    "etaSt2",
    "phiSt2",
    "etaSt1",
    "phiSt1",
]


def ensure_output_dir(outputfolder):
    Path(outputfolder).mkdir(parents=True, exist_ok=True)


def parse_input_files(input_values):
    if isinstance(input_values, str):
        raw_values = [input_values]
    else:
        raw_values = input_values

    input_files = []
    for value in raw_values:
        input_files.extend(part.strip() for part in value.split(",") if part.strip())
    return input_files


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


def build_genmuon_mask(events, relax=False):
    mask = abs(events.GenMuon.pdgId) == 13
    if not relax:
        if hasattr(events.GenMuon, "status"):
            mask = mask & ((events.GenMuon.status & (1 << 13)) != 0)
        if hasattr(events.GenMuon, "pt"):
            mask = mask & (events.GenMuon.pt > 1)
        if hasattr(events.GenMuon, "etaSt2"):
            mask = mask & (events.GenMuon.etaSt2 > -999)
    return mask


def plot_genpart_summary(events, outputfolder):
    """Plot all GenPart particles (no muon/status selection)."""
    if not hasattr(events, "GenPart"):
        print("[WARNING] No GenPart collection found in Events tree")
        return

    gp = events.GenPart
    fields = ak.fields(gp)

    plot_specs_raw = [
        ("pt",    50, (0, 200),     "orange",  "pT [GeV]",    "GenPart pT"),
        ("eta",   50, None,         "green",   "eta",         "GenPart eta"),
        ("phi",   50, (-3.5, 3.5),  "purple",  "phi [rad]",   "GenPart phi"),
        ("mass",  50, (0, 200),     "teal",    "mass [GeV]",  "GenPart mass"),
        ("pdgId", 80, None,         "coral",   "pdgId",       "GenPart pdgId"),
        ("status",50, None,         "steelblue","status",     "GenPart status"),
    ]
    plot_specs = [(f, b, r, c, xl, t) for (f, b, r, c, xl, t) in plot_specs_raw if f in fields]

    if not plot_specs:
        print("[WARNING] GenPart has no plottable fields")
        return

    ncols = 3
    nrows = int(np.ceil(len(plot_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle("GenPart distributions (all particles)", fontsize=16)

    for idx, (feat, bins, hist_range, color, xlabel, title) in enumerate(plot_specs):
        arr = flatten_numeric(gp[feat])
        kw = {"bins": bins, "alpha": 0.7, "edgecolor": "black", "color": color}
        if hist_range is not None:
            kw["range"] = hist_range
        axes[idx].hist(arr, **kw)
        axes[idx].set_xlabel(xlabel, fontsize=12)
        axes[idx].set_ylabel("Counts", fontsize=12)
        axes[idx].set_title(title, fontsize=12)
        axes[idx].grid(True, alpha=0.3)

    for idx in range(len(plot_specs), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    output = os.path.join(outputfolder, "genpart_summary.png")
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def plot_gen_muon_summary(events, outputfolder, relax_selection=False):
    if not hasattr(events, "GenMuon"):
        print("[WARNING] No GenMuon collection found in Events tree")
        return

    mask_allmuons = build_genmuon_mask(events, relax=relax_selection)
    n_muons_per_event = ak.num(events.GenMuon.pdgId[mask_allmuons])

    muon_pt = flatten_numeric(events.GenMuon.pt[mask_allmuons])
    muon_eta = flatten_numeric(events.GenMuon.eta[mask_allmuons])
    muon_phi = flatten_numeric(events.GenMuon.phi[mask_allmuons])
    muon_dxy = flatten_numeric(events.GenMuon.dXY[mask_allmuons])
    muon_lxy = flatten_numeric(events.GenMuon.lXY[mask_allmuons])
    muon_vx = flatten_numeric(events.GenMuon.vx[mask_allmuons])
    muon_vy = flatten_numeric(events.GenMuon.vy[mask_allmuons])
    muon_vz = flatten_numeric(events.GenMuon.vz[mask_allmuons])

    total_muons = len(muon_pt)
    if total_muons == 0:
        sel_label = "relaxed" if relax_selection else "L1Nano"
        print(f"[WARNING] No generated muons passed the {sel_label} selection")
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


def get_stub_quality_values(stub_collection):
    if hasattr(stub_collection, "qual"):
        return stub_collection.qual, "qual"
    if hasattr(stub_collection, "quality"):
        return stub_collection.quality, "quality"
    return None, None


def plot_stub_features(events, outputfolder, quality_value=None):
    if not hasattr(events, "stub"):
        print("[WARNING] No stub collection found in Events tree")
        return

    selection_mask = None
    suffix = "all"
    title = "All Available Stub Feature Distributions"
    color = "steelblue"

    if quality_value is not None:
        quality_values, quality_field = get_stub_quality_values(events.stub)
        if quality_values is None:
            print("[WARNING] Neither stub.qual nor stub.quality is available, quality-specific plots skipped")
            return
        selection_mask = quality_values == quality_value
        suffix = f"quality{quality_value}"
        title = f"Stub Feature Distributions for {quality_field} == {quality_value}"
        if quality_value == 1:
            color = "seagreen"
        elif quality_value == 2:
            color = "darkorange"
        elif quality_value == 3:
            color = "mediumpurple"

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
    if events is None:
        print("\n=== L1Nano input summary ===")
        print("No events loaded")
        return

    stub_fields = ak.fields(events.stub) if hasattr(events, "stub") else []
    GenMuon_fields = ak.fields(events.GenMuon) if hasattr(events, "GenMuon") else []

    print("\n=== L1Nano input summary ===")
    print(f"Events loaded: {len(events)}")
    print(f"stub fields ({len(stub_fields)}): {', '.join(stub_fields)}")
    available_GenMuon_fields = [field for field in L1NANO_GenMuon_BRANCHES if field in GenMuon_fields]
    print(f"GenMuon fields used ({len(available_GenMuon_fields)}): {', '.join(available_GenMuon_fields)}")

    if hasattr(events, "stub"):
        quality_values, quality_field = get_stub_quality_values(events.stub)
        if quality_values is not None:
            qual_values = flatten_numeric(quality_values)
            if qual_values.size > 0:
                unique_quals, counts = np.unique(qual_values.astype(int), return_counts=True)
                print("Stub quality counts:")
                for qual, count in zip(unique_quals, counts):
                    print(f"  {quality_field} == {qual}: {count}")


def normalize_collection_names(events, gen_collection="auto"):
    """Normalize collection names to canonical names for code consistency.

    gen_collection: 'auto' | 'GenMuon' | 'GenPart'
      auto  -> prefer GenMuon; fall back to GenPart mapped as GenMuon
      GenMuon -> use GenMuon only
      GenPart -> keep GenPart as-is (also expose as GenMuon for muon plots)
    """
    if hasattr(events, 'MuonStubTps') and not hasattr(events, 'stub'):
        events['stub'] = events['MuonStubTps']

    if gen_collection in ("auto", "GenMuon"):
        if hasattr(events, 'GenPart') and not hasattr(events, 'GenMuon'):
            events['GenMuon'] = events['GenPart']
    elif gen_collection == "GenPart":
        # Expose GenPart also as GenMuon so muon-summary plots work
        if hasattr(events, 'GenPart') and not hasattr(events, 'GenMuon'):
            events['GenMuon'] = events['GenPart']
    return events


def standardize_l1nano_events(events, include_genpart=False):
    collections = {}

    if hasattr(events, "stub"):
        collections["stub"] = events.stub
    if hasattr(events, "GenMuon"):
        collections["GenMuon"] = events.GenMuon
    if include_genpart and hasattr(events, "GenPart"):
        collections["GenPart"] = events.GenPart

    if not collections:
        return None

    return ak.zip(collections, depth_limit=1)


def load_l1nano_events_from_file(inputfile, treename, entry_stop=None,
                                 gen_collection="auto", include_genpart=False):
    if not os.path.isfile(inputfile):
        print(f"[WARNING] File does not exist, skipping: {inputfile}")
        return None

    tree = uproot.open(f"{inputfile}:{treename}")
    branch_names = tree.keys()
    has_stub = any(name.startswith(("stub_", "MuonStubTps_")) for name in branch_names)
    has_gen = any(name.startswith(("GenMuon_", "GenPart_")) for name in branch_names)

    if not (has_stub and has_gen):
        print("\n=== L1Nano input summary ===")
        print("[WARNING] File does not contain the expected L1Nano branches (stub_*/MuonStubTps_* and GenMuon_*/GenPart_*).")
        print(f"[WARNING] Skipping file: {inputfile}")
        preview = ", ".join(branch_names[:12])
        print(f"[WARNING] Branch preview: {preview}")
        return None

    filter_names = ["stub_*", "MuonStubTps_*"]
    if gen_collection in ("auto", "GenMuon"):
        filter_names += ["GenMuon_*", "GenPart_*"]
    elif gen_collection == "GenPart":
        filter_names += ["GenPart_*"]

    events = tree.arrays(
        filter_name=filter_names,
        how="zip",
        library="ak",
        entry_stop=entry_stop,
    )

    if events is None:
        print("\n=== L1Nano input summary ===")
        print("[WARNING] uproot returned no events for selected branches; skipping file")
        return None

    try:
        loaded_events = len(events)
    except TypeError:
        print("\n=== L1Nano input summary ===")
        print("[WARNING] Loaded event container is not iterable; skipping file")
        return None

    if loaded_events == 0:
        print("\n=== L1Nano input summary ===")
        print("[WARNING] No events loaded after branch filtering; skipping file")
        return None

    events = normalize_collection_names(events, gen_collection=gen_collection)
    events = standardize_l1nano_events(events, include_genpart=include_genpart)
    if events is None:
        print("\n=== L1Nano input summary ===")
        print(f"[WARNING] No compatible collections found after normalization; skipping file: {inputfile}")
        return None

    print(f"[INFO] Loaded {loaded_events} events from {inputfile}")
    return events


def load_l1nano_events(inputfiles, treename, max_events=None,
                       gen_collection="auto", include_genpart=False):
    merged_inputs = []
    total_events = 0

    for inputfile in inputfiles:
        remaining_events = None if max_events is None else max_events - total_events
        if remaining_events is not None and remaining_events <= 0:
            break

        events = load_l1nano_events_from_file(
            inputfile,
            treename,
            entry_stop=remaining_events,
            gen_collection=gen_collection,
            include_genpart=include_genpart,
        )
        if events is None:
            continue

        merged_inputs.append(events)
        total_events += len(events)

    if not merged_inputs:
        return None

    if len(merged_inputs) == 1:
        return merged_inputs[0]

    common_fields = set(ak.fields(merged_inputs[0]))
    for events in merged_inputs[1:]:
        common_fields &= set(ak.fields(events))

    ordered_fields = [field for field in ("stub", "GenMuon", "GenPart") if field in common_fields]
    if not ordered_fields:
        print("[WARNING] No common collections found across input files after normalization")
        return None

    if any(set(ak.fields(events)) != set(ordered_fields) for events in merged_inputs):
        print(f"[WARNING] Input files expose different collections; merging common fields only: {', '.join(ordered_fields)}")
        merged_inputs = [
            ak.zip({field: events[field] for field in ordered_fields}, depth_limit=1)
            for events in merged_inputs
        ]

    print(f"[INFO] Merging {len(merged_inputs)} input files for plotting ({total_events} total events)")
    return ak.concatenate(merged_inputs, axis=0)


def draw_l1nano_variables(inputfiles, outputfolder, treename, max_events=None,
                          gen_collection="auto", relax_gen_selection=False,
                          plot_genpart=False):
    events = load_l1nano_events(
        inputfiles,
        treename,
        max_events=max_events,
        gen_collection=gen_collection,
        include_genpart=plot_genpart,
    )
    if events is None:
        print("\n=== L1Nano input summary ===")
        print("[WARNING] No valid input files could be loaded")
        return False

    print_l1nano_summary(events)
    plot_gen_muon_summary(events, outputfolder, relax_selection=relax_gen_selection)
    if plot_genpart:
        plot_genpart_summary(events, outputfolder)
    plot_stub_features(events, outputfolder)
    plot_stub_features(events, outputfolder, quality_value=1)
    plot_stub_features(events, outputfolder, quality_value=2)
    plot_stub_features(events, outputfolder, quality_value=3)
    return True


def main():
    parser = argparse.ArgumentParser(
        usage="draw_variables.py [options]",
        description="Draw input variables for OMTF or L1Nano studies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i", "--ifile", dest="inputfile", nargs="+", default=["data/omtfAnalysis2.root"],
        help="Input filename(s); pass multiple paths separated by spaces or commas",
    )
    parser.add_argument("-o", "--ofolder", dest="output", default="output/omtfAnalysis2/", help="Folder name to store results")
    parser.add_argument("-t", "--tree", dest="tree", default="simOmtfDigis/OMTFHitsTree", help="Tree name")
    parser.add_argument("-p", "--plots", dest="plots", default="all", help="Plots to be made in OMTF mode")
    parser.add_argument("-m", "--mode", dest="mode", choices=["omtf", "l1nano"], default="omtf", help="Input format to inspect")
    parser.add_argument("--max-events", dest="max_events", type=int, default=None, help="Maximum number of events to read in l1nano mode")
    parser.add_argument(
        "--gen-collection", dest="gen_collection",
        choices=["auto", "GenMuon", "GenPart"], default="auto",
        help="Which generator-level collection to use (auto: prefer GenMuon, fall back to GenPart)",
    )
    parser.add_argument(
        "--relax-gen-selection", dest="relax_gen_selection",
        action="store_true", default=False,
        help="Relax GenMuon selection (skip status-bit / pT / etaSt2 cuts)",
    )
    parser.add_argument(
        "--plot-genpart", dest="plot_genpart",
        action="store_true", default=False,
        help="Also draw a GenPart summary plot (all particles, no muon selection)",
    )

    args = parser.parse_args()
    input_files = parse_input_files(args.inputfile)

    ensure_output_dir(args.output)

    print(f"Running on {len(input_files)} input file(s)")
    for input_file in input_files:
        print(f"  - {input_file}")
    print(f"Saving result in: {args.output}")
    print(f"Mode: {args.mode}")

    if args.mode == "l1nano":
        if args.tree == parser.get_default("tree"):
            args.tree = "Events"
        ok = draw_l1nano_variables(
            input_files, args.output, args.tree, args.max_events,
            gen_collection=args.gen_collection,
            relax_gen_selection=args.relax_gen_selection,
            plot_genpart=args.plot_genpart,
        )
        if not ok:
            raise SystemExit(2)
    else:
        if len(input_files) != 1:
            raise SystemExit("OMTF mode supports exactly one input file")
        draw_single_vars(input_files[0], args.output, args.tree, args.plots, "muonPropEta!=0&&muonPropPhi!=0")
        draw_correlations(input_files[0], args.output, args.tree, "muonPropEta!=0&&muonPropPhi!=0")

    print("DONE")


if __name__ == "__main__":
    main()
