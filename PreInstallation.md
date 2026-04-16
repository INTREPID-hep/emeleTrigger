# Pre-installation guide (CMSSW + L1Nano GenParticlePropagator changes)

This guide sets up a clean `CMSSW_15_1_0_pre4` area, initializes `cmsenv`, downloads the packages required by your branch, and applies your modifications from:

- https://github.com/folguera/cmssw/pull/new/from-CMSSW_15_1_0_pre4_L1NanoGenParticlePropagator

## 1) Create and initialize CMSSW area

```bash
# Recommended architecture for CMSSW_15_1_0_pre4
export SCRAM_ARCH=el9_amd64_gcc12

cmsrel CMSSW_15_1_0_pre4
cd CMSSW_15_1_0_pre4/src
cmsenv

# Initialize CMSSW git helpers
git cms-init
```

## 2) Fetch your branch with modifications

```bash
# Add your fork remote once (ignore error if it already exists)
git remote add folguera https://github.com/folguera/cmssw.git || true

# Fetch your topic branch
git fetch folguera from-CMSSW_15_1_0_pre4_L1NanoGenParticlePropagator
```

## 3) Download only required CMSSW packages touched by your branch

```bash
# Compute CMSSW packages touched by the topic branch and checkout them locally
for pkg in $(git diff --name-only HEAD..folguera/from-CMSSW_15_1_0_pre4_L1NanoGenParticlePropagator | cut -d/ -f1,2 | sort -u); do
	git cms-addpkg "$pkg"
done
```

## 4) Apply your modifications in the local area

```bash
# Create local working branch from your remote branch
git checkout -b from-CMSSW_15_1_0_pre4_L1NanoGenParticlePropagator \
	folguera/from-CMSSW_15_1_0_pre4_L1NanoGenParticlePropagator
```

## 5) Build and validate CMSSW

```bash
scram b -j 8
```

Optional quick checks:

```bash
git status
git log --oneline -n 5
```

## 6) Install Python package for the training tools

From inside the same CMSSW area:

```bash
cd emeleTrigger

# Option A: venv
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install .

# Option B: conda (if preferred)
# conda create -n cmsl1t python=3.12 -y
# conda activate cmsl1t
# python -m pip install -U pip
# python -m pip install .
```

## Notes

- If your shell does not have `cmsrel`, source CMSSW defaults first (e.g. CERN/LXPLUS environment).
- If `git cms-addpkg` is unavailable, ensure `cmsenv` has been executed in `CMSSW_15_1_0_pre4/src`.
- If you already have a local branch with changes, skip branch creation and use `git merge` or `git cherry-pick` as needed.
