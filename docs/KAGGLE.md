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
3. `REPO_URL` already points at `github.com/bhavishyabondalapati/impactlens`; the repo must be
   pushed (public) before the clone works.
4. **Run All**. Sections 1-3 clone, install and check the data; section 4 trains both models
   (`--task oscd`, then `--task floods`) into `/kaggle/working/oscd` and `/kaggle/working/floods`.
5. **Save Version** -> *Save & Run All (Commit)* so outputs persist; download them from the
   notebook's **Output** tab into `models/` (gitignored).

## 5. Round 2: update the dataset with the SWIR bands

`sen1floods11.npz` now has 6 bands (blue, green, red, nir, swir16, swir22) and is 327 MB.
`oscd.npz` is unchanged (still 4 bands), so you only need to replace the floods file. The loader
refuses the old 4-band file, so a stale upload fails loudly instead of training on wrong data.

1. `uv run python scripts/pack_for_kaggle.py --only floods` (already done in `build/kaggle/`).
2. kaggle.com -> your `impactlens-packed` dataset -> **New Version** -> upload
   `build/kaggle/sen1floods11.npz` (remove the old one; keep `oscd.npz`) -> **Create**.
   CLI alternative: `uv tool run kaggle datasets version -p build/kaggle -m "6-band floods"`
   (needs `YOUR_KAGGLE_USERNAME` in `dataset-metadata.json` replaced by your username).
3. Push the repo (the notebook clones it), then in the notebook make sure the input is the
   *latest* version of the dataset and **Run All**. Output goes to `/kaggle/working/floods_r2`.
