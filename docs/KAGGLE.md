# Training on Kaggle (free GPU)

Kaggle gives ~30 GPU hours per week for free (T4 x2 or P100). We upload only a small packed
copy of the data (~320 MB instead of ~1 GB of 13-band GeoTIFFs) and clone this repo inside the
notebook.

## 1. Pack the data (on your Mac)

```bash
cd ~/projects/impactlens
uv run python scripts/pack_for_kaggle.py --kaggle-user YOUR_KAGGLE_USERNAME
```

This writes `build/kaggle/` (gitignored): `oscd.npz`, `sen1floods11.npz`, `dataset-metadata.json`,
and checks each packed file against the raw loaders before it finishes. Re-run it whenever
`BAND_CLIPS` in `preprocess.py` changes: the loaders refuse a stale pack.

## 2. Upload the dataset (keep it private)

Web UI: kaggle.com -> **Datasets** -> **New Dataset** -> drag in `oscd.npz` and `sen1floods11.npz`
-> name it `impactlens-packed` -> visibility **Private** -> **Create**.

Or CLI: create a token at kaggle.com -> your avatar -> **Settings** -> **API** -> *Create New Token*,
follow Kaggle's instructions to install it (never commit it), then:

```bash
uv tool run kaggle datasets create -p build/kaggle
```

## 3. Put the repo on GitHub (public)

The notebook does `git clone`, so the repo must be on GitHub. Check `git status` shows nothing
under `data/` or `build/`, then push when you are ready (a private repo would need a token stored
in Kaggle *Secrets*, never pasted into the notebook).

## 4. Run the notebook

1. kaggle.com -> **Code** -> **New Notebook** -> **File -> Import Notebook** -> upload
   `notebooks/train_kaggle.ipynb`.
2. Right panel: **Accelerator** = GPU T4 x2 (or P100); **Internet** = On (Kaggle asks for phone
   verification once); **Add Input** -> your `impactlens-packed` dataset.
3. Edit `REPO_URL` in the first code cell.
4. **Run All**. Sections 1-3 (clone, install, data check) work today. Section 4 trains once
   `train.py` exists (Phase 3).
5. **Save Version** -> *Save & Run All (Commit)* so outputs persist; download them from the
   notebook's **Output** tab into `models/` (gitignored).
