"""Target-free pairing/splits; frozen image features; separate protected label files."""
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from .common import ROOT, config, fraction, sha, write_json


def rgb_record(filename):
    path = ROOT / "data/images" / filename
    with Image.open(path) as source:
        img = source.convert("RGB")
        digest = hashlib.sha256(str(img.size).encode() + img.tobytes()).hexdigest()
        return {"img_id": filename, "rgb_sha256": digest, "file_sha256": sha(path), "width": img.width, "height": img.height}


def group_components(rows, photo_info):
    parents = {p: p for p in rows.patient_id.unique()}
    def find(p):
        while parents[p] != p:
            parents[p] = parents[parents[p]]; p = parents[p]
        return p
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    seen = {}; lesions = {}
    for row, info in zip(rows.itertuples(), photo_info, strict=True):
        if hasattr(row, 'lesion_id'):
            if row.lesion_id in lesions: union(row.patient_id, lesions[row.lesion_id])
            lesions[row.lesion_id] = row.patient_id
        h = info["rgb_sha256"]
        if h in seen:
            union(row.patient_id, seen[h])
        seen[h] = row.patient_id
    return [find(p) for p in rows.patient_id]


def assign_split(group, c):
    value = fraction(c["split"]["salt"], group)
    if value < c['split']['train']:
        return "tune" if fraction(c["split"]["inner_salt"], group) < c["split"]["inner_tune_fraction_of_training"] else "fit"
    return "validation" if value < c['split']['train'] + c['split']['validation'] else "test"


class ClinicalEncoder:
    def __init__(self, numeric, categorical):
        self.numeric, self.categorical = numeric, categorical

    @staticmethod
    def clean(series):
        return series.fillna("__MISSING__").astype(str).replace({"": "__MISSING__", "UNK": "__MISSING__", "nan": "__MISSING__"})

    def fit(self, table):
        numeric = table[self.numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        median = np.nanmedian(numeric, axis=0); median = np.nan_to_num(median)
        filled = np.where(np.isnan(numeric), median, numeric)
        self.state = {"numeric": self.numeric, "categorical": self.categorical, "median": median.tolist(), "mean": filled.mean(0).tolist(), "scale": np.maximum(filled.std(0), 1e-6).tolist(), "vocab": {key: sorted((set(self.clean(table[key])) | {"__MISSING__"}) - {"__UNKNOWN__"}) + ["__UNKNOWN__"] for key in self.categorical}}
        return self

    def transform(self, table):
        numeric = table[self.numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        missing = np.isnan(numeric)
        filled = np.where(missing, np.array(self.state["median"]), numeric)
        pieces = [(filled - np.array(self.state["mean"])) / np.array(self.state["scale"]), missing.astype(float)]
        for key in self.categorical:
            vocab = self.state["vocab"][key]; index = {v: i for i, v in enumerate(vocab)}
            positions = [index.get(v, index["__UNKNOWN__"]) for v in self.clean(table[key])]
            pieces.append(np.eye(len(vocab))[positions])
        result = np.concatenate(pieces, axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite clinical features")
        return result

    @classmethod
    def from_state(cls, state):
        result = cls(state["numeric"], state["categorical"]); result.state = state
        return result


class Photos(Dataset):
    def __init__(self, names, transform):
        self.names, self.transform = names, transform
    def __len__(self):
        return len(self.names)
    def __getitem__(self, i):
        with Image.open(ROOT / "data/images" / self.names[i]) as img:
            return self.transform(img.convert("RGB"))


def prepare():
    c = config(); columns = ["img_id", "patient_id", "lesion_id"] + c["numeric_features"] + c["categorical_features"]
    source = ROOT / "data/raw/metadata.csv"
    expected = json.loads((ROOT / "docs/sources/dataset-root-files.json").read_text())[0]["content_details"]["sha256_hash"]
    if sha(source) != expected:
        raise ValueError("Official metadata hash changed")
    table = pd.read_csv(source, usecols=columns, dtype=str).sort_values("img_id").reset_index(drop=True)
    if table.img_id.duplicated().any() or table[["img_id", "patient_id", "lesion_id"]].isna().any().any():
        raise ValueError("Missing or duplicate record keys")
    # IDs reused across patients are conservatively grouped, not presumed unrelated.
    if set(table.img_id) != {p.name for p in (ROOT / "data/images").glob("*.png")}:
        raise ValueError("Image/metadata keys do not match exactly")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        info = list(executor.map(rgb_record, table.img_id))
    table["component"] = group_components(table, info)
    table["split"] = [assign_split(g, c) for g in table.component]
    out = ROOT / "data/prepared"
    out.mkdir()  # A failed preparation is preserved; reruns need a new reviewed lineage.
    # Membership is frozen before the diagnostic column is opened for opaque partitioning.
    rows = table[["img_id", "patient_id", "lesion_id", "component", "split"]].to_dict("records")
    write_json(out / "membership.json", {"protocol_sha256": sha(ROOT / "protocol.json"), "source_sha256": sha(source), "rows": rows, "images": info, "target_access_for_membership": False}, exclusive=True)
    table.to_csv(out / "features.csv", index=False)
    # Mechanical source-label partitioning is recorded separately from analytical label access.
    labels = pd.read_csv(source, usecols=["img_id", "diagnostic"], dtype=str).set_index("img_id")
    mapping = {label: i for i, label in enumerate(c["classes"])}
    for split in ["fit", "tune", "validation", "test"]:
        names = table.loc[table.split == split, "img_id"].tolist()
        values = [mapping[labels.loc[name, "diagnostic"]] for name in names]
        write_json(out / f"labels-{split}.json", {"img_ids": names, "labels": values}, exclusive=True)
    write_json(out / "label-ingestion.json", {"raw_rows_partitioned": len(table), "membership_frozen_before_label_partition": True, "analytical_validation_reads": 0, "analytical_test_reads": 0, "not_used_for_split_or_feature_selection": True}, exclusive=True)
    fit = table.split == "fit"
    encoder = ClinicalEncoder(c["numeric_features"], c["categorical_features"]).fit(table.loc[fit])
    clinical = encoder.transform(table); np.save(out / "clinical.npy", clinical)
    write_json(out / "clinical-encoder.json", encoder.state, exclusive=True)
    counts = {s: {"images": int((table.split == s).sum()), "patients": int(table.loc[table.split == s, "patient_id"].nunique()), "components": int(table.loc[table.split == s, "component"].nunique())} for s in ["fit", "tune", "validation", "test"]}
    write_json(out / "counts.json", counts, exclusive=True)
    print(json.dumps({"stage": "membership_frozen", "counts": counts, "clinical_dimensions": clinical.shape[1]}), flush=True)
    torch.set_num_threads(c["budgets"]["threads"])
    weights = ResNet18_Weights.IMAGENET1K_V1
    network = resnet18(weights=weights)
    backbone = torch.nn.Sequential(*list(network.children())[:-2]).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    torch.save(backbone.state_dict(), out / "encoder.pt")
    loader = DataLoader(Photos(table.img_id.tolist(), weights.transforms()), batch_size=16, num_workers=2, shuffle=False)
    features = np.lib.format.open_memmap(out / "image_features.npy", mode="w+", dtype=np.float16, shape=(len(table), 7, 7, 512))
    start = time.monotonic(); pos = 0
    with torch.inference_mode():
        for batch in loader:
            maps = backbone(batch).permute(0, 2, 3, 1).numpy()
            if not np.isfinite(maps).all():
                raise ValueError("Nonfinite frozen image features")
            stored = maps.astype(np.float16)
            if not np.isfinite(stored).all(): raise ValueError('Nonfinite compressed feature cache')
            features[pos:pos + len(batch)] = stored; pos += len(batch)
            if pos % 160 == 0 or pos == len(table):
                features.flush()
                print(json.dumps({"stage": "frozen_image_features", "completed": pos, "total": len(table), "seconds": round(time.monotonic() - start, 1)}), flush=True)
    features.flush()
    if pos != len(table): raise ValueError('Incomplete feature row inventory')
    # Only fit-partition statistics, not tune/validation/test, influence scaling.
    x = np.asarray(features[np.flatnonzero(fit)], dtype=np.float32)
    mean = x.mean((0, 1, 2)); scale = np.maximum(x.std((0, 1, 2)), 1e-6)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0): raise ValueError('Invalid image scaler')
    write_json(out / "image-scaler.json", {"mean": mean.tolist(), "scale": scale.tolist(), "fitted_split": "fit"}, exclusive=True)
    path = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.split("/")[-1]
    write_json(out / "encoder-provenance.json", {"weight_url": weights.url, "weight_sha256": sha(path), "frozen": True, "eval_mode": True, "inference_mode": True, "torch": torch.__version__, "transform": str(weights.transforms()), "cache_dtype": "float16", "row_order_sha256": hashlib.sha256("\n".join(table.img_id).encode()).hexdigest()}, exclusive=True)
    files = {str(path.relative_to(out)): sha(path) for path in sorted(out.iterdir()) if path.is_file()}
    write_json(out / "manifest.json", {"files": files, "data_bytes": sum(p.stat().st_size for p in (ROOT / "data").rglob('*') if p.is_file()), "analytical_validation_reads": 0, "analytical_test_reads": 0}, exclusive=True)
    print(json.dumps({"prepared": str(out), "manifest_sha256": sha(out / "manifest.json")}), flush=True)


if __name__ == "__main__":
    prepare()
