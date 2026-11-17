"""Reviewed CPU experiment. Outer validation/test have separate execute-only gates."""
import argparse
import copy
import json
import math
import os
import platform
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import torch
from .common import ROOT, config, sha, json_digest, write_json
from .data import assign_split
from .models import FusionModel
from .metrics import score, bootstrap

EVENTS = {"run_start", "trial_reserved", "trial_start", "epoch_end", "checkpoint", "trial_complete", "warning", "failure", "terminal"}


def binding():
    files = [ROOT / n for n in ("protocol.json", "pyproject.toml", "uv.lock")]
    for folder in ("src", "vendor", "tests"):
        files += sorted((ROOT / folder).rglob("*.py"))
    files += sorted(p for p in (ROOT / "data/prepared").iterdir() if p.is_file())
    result = {}
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Only regular bound files are accepted")
        result[str(path.relative_to(ROOT))] = sha(path)
    return result


def validate_membership(c):
    out = ROOT / "data/prepared"
    manifest = json.loads((out / "manifest.json").read_text())
    for path, digest in manifest["files"].items():
        if sha(out / path) != digest:
            raise ValueError(f"Prepared artifact drift: {path}")
    membership = json.loads((out / "membership.json").read_text())
    if membership["protocol_sha256"] != sha(ROOT / "protocol.json"):
        raise ValueError("Protocol changed after target-free membership freeze")
    table = pd.read_csv(out / "features.csv", dtype=str)
    if table[["img_id", "patient_id", "lesion_id", "component", "split"]].to_dict("records") != membership["rows"]:
        raise ValueError("Row order or membership drift")
    if any(assign_split(row.component, c) != row.split for row in table.itertuples()):
        raise ValueError("Split does not replay from target-free hashes")
    for key in ("patient_id", "lesion_id", "component"):
        if (table.groupby(key).split.nunique() > 1).any():
            raise ValueError(f"Split leakage: {key}")
    return table


def validate_selection(selection_path, expected_mode, current_binding=None):
    """Validate the complete predecessor before admitting a run or opening labels."""
    from .supervisor import validate_journal
    path = Path(selection_path); run = path.parent
    selected = json.loads(path.read_text()); start = json.loads((run / 'start.json').read_text())
    status = json.loads((run / 'resource-status.json').read_text()); c = config()
    start_sha = sha(run / 'start.json'); p = start['proposal']
    expected = [[m, s] for m in c['models'] for s in (c['seeds'][:1] if expected_mode == 'smoke' else c['seeds'])]
    if selected['mode'] != expected_mode or p['mode'] != expected_mode or p['planned_trials'] != expected or [[t['model'], t['seed']] for t in selected['trials']] != expected:
        raise ValueError('Predecessor has wrong, duplicated or missing trial identities')
    if p['binding'] != (binding() if current_binding is None else current_binding):
        raise ValueError('Predecessor code/protocol/cache/environment binding differs')
    if not status['completed_cleanly'] or status['actions'] or status['returncode'] != 0 or any(obj['start_sha256'] != start_sha for obj in (status, selected)):
        raise ValueError('Predecessor did not complete cleanly with this start')
    if selected['protocol_sha256'] != sha(ROOT / 'protocol.json') or selected['label_reads'] != {'fit': 1, 'tune': 1}:
        raise ValueError('Predecessor protocol/label access mismatch')
    events = validate_journal(run)
    if events[-1]['payload']['selection_sha256'] != sha(path):
        raise ValueError('Selection does not match terminal event')
    for kind in ('trial_reserved', 'trial_start', 'trial_complete'):
        identities = [[e['payload']['model'], e['payload']['seed']] for e in events if e['event'] == kind]
        if identities != expected: raise ValueError('Journal trial identities mismatch')
    for t in selected['trials']:
        epoch_rows = [e['payload'] for e in events if e['event'] == 'epoch_end' and e['payload']['model'] == t['model'] and e['payload']['seed'] == t['seed']]
        if [{k: v for k, v in row.items() if k not in ('model', 'seed')} for row in epoch_rows] != t['curves']:
            raise ValueError('Journal and training curves disagree')
        checkpoint = Path(t['checkpoint'])
        if checkpoint.parent.resolve() != (run / 'checkpoints').resolve() or sha(checkpoint) != t['checkpoint_sha256']:
            raise ValueError('Checkpoint location/hash drift')
        saved = torch.load(checkpoint, weights_only=True)
        if saved['model'] != t['model'] or saved['seed'] != t['seed'] or saved['epoch'] != t['best_epoch'] or saved['epoch'] != t['restored_epoch'] or saved['start_sha256'] != start_sha or saved['protocol_sha256'] != selected['protocol_sha256'] or saved['tune_loss'] != t['best_tune_loss']:
            raise ValueError('Checkpoint semantic identity mismatch')
        best = min(t['curves'], key=lambda row: row['tune_loss'])
        if abs(best['tune_loss'] - saved['tune_loss']) > c['training']['minimum_delta']:
            raise ValueError('Checkpoint differs from recorded best tuning loss')
        if not all(torch.isfinite(weight).all() for weight in saved['weights'].values()):
            raise ValueError('Nonfinite checkpoint parameter')
    return selected, start, status


def proposal(mode, run_name, output, selection=None, smoke=None):
    c = config(); table = validate_membership(c)
    if mode not in ("smoke", "train", "validation", "test"):
        raise ValueError("Unknown execution purpose")
    if not run_name.replace('-', '').isalnum():
        raise ValueError("Run name must be a simple identifier")
    planned = [[m, s] for m in c["models"] for s in (c["seeds"][:1] if mode == "smoke" else c["seeds"])]
    p = {"schema": 1, "created_at": datetime.now(timezone.utc).isoformat(), "mode": mode, "run": f"runs/{run_name}", "cwd": str(ROOT), "nonce": str(uuid.uuid4()), "binding": binding(), "planned_trials": planned, "label_access": {"raw_ingestion": "public source labels mechanically partitioned only after membership frozen", "allowed_analytical_partitions": ["fit", "tune"] if mode in ("smoke", "train") else [mode]}, "budgets": c["budgets"], "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(), "threads": c["budgets"]["threads"]}, "memory_estimate_mib": {"mapped_cache_and_input_batches": 256, "labels_and_clinical": 5, "parameters_gradients_optimizer": 32, "activations_temporaries": 192, "framework_and_python": 768, "predictions_checkpoint_buffers": 64, "safety_margin": 512}, "scientific_result": False, "publication_ready": False}
    p["command"] = [str(ROOT / ".venv/bin/python"), "-m", "src.supervisor", "--proposal", str(Path(output).resolve()), "--review", str(ROOT / "reviews" / f"execute-{run_name}.json")]
    if mode == "train":
        if not smoke:
            raise ValueError("Actual supervised smoke is required")
        s = Path(smoke); result, smoke_start, state = validate_selection(s / 'selection.json', 'smoke', p['binding'])
        if not state["completed_cleanly"] or result["mode"] != "smoke" or len(result["trials"]) != len(c["models"]):
            raise ValueError("Incomplete/failed smoke")
        batch_counts = math.ceil((table.split == 'fit').sum() / c['training']['batch_size']) + math.ceil((table.split == 'tune').sum() / c['training']['batch_size'])
        estimated = sum(t["mean_batch_seconds"] * batch_counts * c["training"]["max_epochs"] * len(c["seeds"]) for t in result["trials"]) * 1.8 + 90
        p["feasibility"] = {"smoke": str(s.resolve()), "selection_sha256": sha(s / "selection.json"), "resource_sha256": sha(s / "resource-status.json"), "estimated_full_run_seconds": estimated, "measured_peak_process_tree_mib": state["peak_rss_mib"], "projection_margin": 1.8, "approved_by_threshold": estimated < c["budgets"]["training_soft_seconds"] and state["peak_rss_mib"] * 1.5 < c["budgets"]["rss_limit_gib"] * 1024}
        if not p["feasibility"]["approved_by_threshold"]:
            raise ValueError(f"Run exceeds measured feasibility budget: {p['feasibility']}")
    if mode in ("validation", "test"):
        if not selection:
            raise ValueError("Frozen selection is required")
        p["selection"] = str(Path(selection).resolve()); p["selection_sha256"] = sha(selection)
        saved, training_start, training_status = validate_selection(selection, 'train', p['binding'])
        if saved["mode"] != "train" or len(saved["trials"]) != len(planned):
            raise ValueError("Cannot evaluate incomplete training")
        p["checkpoint_hashes"] = {t["checkpoint"]: t["checkpoint_sha256"] for t in saved["trials"]}
    write_json(Path(output), p, exclusive=True)
    print(json.dumps({"proposal": str(output), "sha256": sha(output), "mode": mode, "feasibility": p.get("feasibility")}), flush=True)


def verify_review(proposal_path, review_path):
    p = json.loads(Path(proposal_path).read_text()); r = json.loads(Path(review_path).read_text())
    now = datetime.now(timezone.utc)
    if r.get("decision") != "approved" or r.get("scope") != "execute-only" or r.get("purpose") != p["mode"]:
        raise ValueError("Separate scope-bound execution review required")
    if r.get("reviewer") != "/root/oncology_model_review" or r.get("reviewer_tool") != "collaboration":
        raise ValueError("Expected independent review identity is absent")
    if r.get("proposal_sha256") != sha(proposal_path) or r.get("proposal_semantic_sha256") != json_digest(p) or r.get("nonce") != p["nonce"]:
        raise ValueError("Review/proposal/hash/nonce mismatch")
    issued = datetime.fromisoformat(r["reviewed_at"]); expires = datetime.fromisoformat(r["expires_at"])
    if not datetime.fromisoformat(p["created_at"]) <= issued <= now < expires or (expires - issued).total_seconds() > 6 * 3600:
        raise ValueError("Review is stale or has invalid chronology")
    if r.get("command") != p["command"] or r.get("cwd") != str(ROOT):
        raise ValueError("Complete launch command was not independently reviewed")
    if binding() != p["binding"]:
        raise ValueError("Fresh code/config/cache/lock hashes differ from review")
    if p["mode"] == "train":
        f = p["feasibility"]
        if not f["approved_by_threshold"] or sha(Path(f["smoke"]) / "resource-status.json") != f["resource_sha256"] or sha(Path(f["smoke"]) / "selection.json") != f["selection_sha256"]:
            raise ValueError("Feasibility artifact drift")
        validate_selection(Path(f['smoke']) / 'selection.json', 'smoke', p['binding'])
    if p["mode"] in ("validation", "test"):
        if sha(p["selection"]) != p["selection_sha256"]:
            raise ValueError("Frozen selection changed")
        for path, digest in p["checkpoint_hashes"].items():
            if sha(path) != digest:
                raise ValueError("Checkpoint drift")
        validate_selection(p['selection'], 'train', p['binding'])
    return p, r


class Journal:
    def __init__(self, run):
        self.file = (run / "events.jsonl").open("x", buffering=1)
        self.start_sha = sha(run / "start.json"); self.seq = 0; self.previous = None; self.closed = False
    def emit(self, kind, **payload):
        if self.closed or kind not in EVENTS:
            raise ValueError("Invalid or post-terminal event")
        event = {"sequence": self.seq, "timestamp": datetime.now(timezone.utc).isoformat(), "start_sha256": self.start_sha, "previous_sha256": self.previous, "event": kind, "payload": payload, "scientific_result": False}
        self.previous = json_digest(event); event["sha256"] = self.previous
        self.file.write(json.dumps(event, allow_nan=False) + "\n"); self.file.flush(); os.fsync(self.file.fileno()); self.seq += 1
        if kind == "terminal":
            self.closed = True; self.file.close()


class Store:
    def __init__(self, table, allowed, journal):
        self.table, self.allowed, self.journal = table, set(allowed), journal
        self.images = np.load(ROOT / "data/prepared/image_features.npy", mmap_mode="r")
        self.clinical = np.load(ROOT / "data/prepared/clinical.npy")
        state = json.loads((ROOT / "data/prepared/image-scaler.json").read_text())
        self.mean, self.scale = np.array(state["mean"], np.float32), np.array(state["scale"], np.float32)
        self.reads = {}
    def labels(self, part):
        if part not in self.allowed or self.reads.get(part, 0):
            raise ValueError("Forbidden or repeated analytical label read")
        ids = np.flatnonzero(self.table.split == part)
        data = json.loads((ROOT / f"data/prepared/labels-{part}.json").read_text())
        if data["img_ids"] != self.table.iloc[ids].img_id.tolist():
            raise ValueError("Label order mismatch")
        self.reads[part] = 1
        return ids, np.array(data["labels"], dtype=np.int64)
    def tensors(self, ids):
        image = (np.asarray(self.images[ids], dtype=np.float32) - self.mean) / self.scale
        return torch.from_numpy(image), torch.from_numpy(self.clinical[ids])


def batches(ids, batch_size):
    # Balanced chunk sizes preserve all rows and prevent MetaBlock BatchNorm singleton batches.
    return np.array_split(ids, max(1, math.ceil(len(ids) / batch_size)))


def predict(model, store, ids, size):
    model.eval(); parts = []
    with torch.inference_mode():
        for batch in batches(ids, size):
            parts.append(model(*store.tensors(batch)).softmax(1).numpy())
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite prediction")
    return result


def worker(run):
    start = json.loads((run / "start.json").read_text()); p = start["proposal"]; c = config()
    torch.set_num_threads(c["budgets"]["threads"]); torch.use_deterministic_algorithms(True)
    journal = Journal(run); journal.emit("run_start", mode=p["mode"], planned_trials=p["planned_trials"])
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError("External monitor requested drain")))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(InterruptedError("Interrupted")))
    begin = time.monotonic(); store = None
    try:
        table = validate_membership(c)
        store = Store(table, p["label_access"]["allowed_analytical_partitions"], journal)
        if p["mode"] in ("validation", "test"):
            evaluate(run, p, c, store, journal)
            journal.emit("terminal", status="success", label_reads=store.reads)
            return
        fit_ids, fit_y = store.labels("fit"); tune_ids, tune_y = store.labels("tune")
        counts = np.bincount(fit_y, minlength=6)
        if np.any(counts == 0) or len(np.unique(tune_y)) < 6:
            raise ValueError("Missing class in fit/tune; no post-label split search is permitted")
        weights = torch.tensor(len(fit_y) / (6 * counts), dtype=torch.float32)
        if p["mode"] == "smoke":
            fit_ids, fit_y = fit_ids[:64], fit_y[:64]; tune_ids, tune_y = tune_ids[:64], tune_y[:64]
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        trials = []; checkpoints = run / "checkpoints"; checkpoints.mkdir()
        for name, seed in p["planned_trials"]:
            if time.monotonic() - begin > c["budgets"]["training_soft_seconds"]:
                raise TimeoutError("Soft deadline before next trial")
            journal.emit("trial_reserved", model=name, seed=seed, remaining=len(p["planned_trials"]) - len(trials))
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            model = FusionModel(name, store.clinical.shape[1], c)
            optimizer = torch.optim.AdamW(model.parameters(), lr=c["training"]["learning_rate"], weight_decay=c["training"]["weight_decay"])
            journal.emit("trial_start", model=name, seed=seed, parameters=sum(t.numel() for t in model.parameters()))
            best = float('inf'); stale = 0; best_epoch = 0; curves = []; durations = []
            max_epochs = 2 if p["mode"] == "smoke" else c["training"]["max_epochs"]
            checkpoint = checkpoints / f"{name}-{seed}.pt"
            for epoch in range(1, max_epochs + 1):
                tick = time.monotonic(); phases = {k: 0.0 for k in ('slice_transform', 'forward', 'backward', 'optimizer', 'tune')}
                model.train(); train_sum = 0.0; train_weight = 0.0
                order = rng.permutation(len(fit_ids)); total_batches = 0
                for local in batches(order, c["training"]["batch_size"]):
                    t = time.monotonic(); x = store.tensors(fit_ids[local]); y = torch.from_numpy(fit_y[local]); phases['slice_transform'] += time.monotonic() - t
                    optimizer.zero_grad(set_to_none=True)
                    t = time.monotonic(); loss = loss_fn(model(*x), y); phases['forward'] += time.monotonic() - t
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    t = time.monotonic(); loss.backward(); phases['backward'] += time.monotonic() - t
                    if any(not torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None):
                        raise ValueError("Nonfinite gradient")
                    t = time.monotonic(); optimizer.step(); phases['optimizer'] += time.monotonic() - t
                    if any(not torch.isfinite(t).all() for t in model.parameters()):
                        raise ValueError("Nonfinite weights")
                    w = float(weights[y].sum()); train_sum += float(loss.detach()) * w; train_weight += w; total_batches += 1
                t = time.monotonic(); model.eval(); tune_sum = 0.0; tune_weight = 0.0
                with torch.inference_mode():
                    for local in batches(np.arange(len(tune_ids)), c["training"]["batch_size"]):
                        y = torch.from_numpy(tune_y[local]); l = loss_fn(model(*store.tensors(tune_ids[local])), y)
                        if not torch.isfinite(l):
                            raise ValueError("Nonfinite tuning loss")
                        w = float(weights[y].sum()); tune_sum += float(l) * w; tune_weight += w; total_batches += 1
                phases['tune'] = time.monotonic() - t; value = tune_sum / tune_weight
                if value < best - c["training"]["minimum_delta"]:
                    best, best_epoch, stale = value, epoch, 0
                    temp = checkpoint.with_suffix('.tmp')
                    torch.save({"model": name, "seed": seed, "epoch": epoch, "tune_loss": value, "start_sha256": sha(run / 'start.json'), "protocol_sha256": sha(ROOT / 'protocol.json'), "weights": model.state_dict()}, temp)
                    with temp.open('rb') as f: os.fsync(f.fileno())
                    os.replace(temp, checkpoint)
                    directory = os.open(checkpoint.parent, os.O_RDONLY)
                    try: os.fsync(directory)
                    finally: os.close(directory)
                    journal.emit("checkpoint", model=name, seed=seed, epoch=epoch, sha256=sha(checkpoint))
                else:
                    stale += 1
                seconds = time.monotonic() - tick; durations.append(seconds / total_batches)
                row = {"epoch": epoch, "train_loss": train_sum / train_weight, "tune_loss": value, "best_epoch": best_epoch, "stale": stale, "seconds": seconds, "phase_seconds": phases}
                curves.append(row); journal.emit("epoch_end", model=name, seed=seed, **row)
                if epoch % 10 == 0 or epoch == 1:
                    print(json.dumps({"model": name, "seed": seed, "epoch": epoch, "train_loss": round(row['train_loss'], 4), "tune_loss": round(value, 4), "best_epoch": best_epoch, "fits_completed": len(trials), "fits_total": len(p['planned_trials'])}), flush=True)
                if stale >= c['training']['patience']:
                    break
            saved = torch.load(checkpoint, weights_only=True); model.load_state_dict(saved['weights']); model.eval()
            trial = {"model": name, "seed": seed, "best_epoch": best_epoch, "restored_epoch": saved['epoch'], "best_tune_loss": best, "epochs_completed": epoch, "stop_reason": "smoke_limit" if p['mode'] == 'smoke' else ('patience' if stale >= c['training']['patience'] else 'epoch_cap'), "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint), "mean_batch_seconds": float(np.mean(durations)), "curves": curves}
            trials.append(trial)
            if trial['stop_reason'] == 'epoch_cap':
                journal.emit('warning', model=name, seed=seed, reason='epoch cap reached; review convergence before held-out access')
            journal.emit("trial_complete", model=name, seed=seed, checkpoint_sha256=trial['checkpoint_sha256'], restored_epoch=best_epoch)
        if [[t['model'], t['seed']] for t in trials] != p['planned_trials']:
            raise ValueError('Incomplete trial plan')
        selection = {"mode": p['mode'], "start_sha256": sha(run / 'start.json'), "protocol_sha256": sha(ROOT / 'protocol.json'), "trials": trials, "class_weights": weights.tolist(), "label_reads": store.reads, "elapsed_seconds": time.monotonic() - begin, "scientific_result": False, "publication_ready": False}
        write_json(run / 'selection.json', selection, exclusive=True)
        journal.emit('terminal', status='success', label_reads=store.reads, selection_sha256=sha(run / 'selection.json'))
    except BaseException as exc:
        failure = {"error_type": type(exc).__name__, "message": str(exc), "label_reads": store.reads if store else {}, "scientific_result": False, "publication_ready": False}
        write_json(run / 'failure.json', failure, exclusive=True)
        journal.emit('failure', **failure); journal.emit('terminal', status='failure')
        raise


def evaluate(run, p, c, store, journal):
    selected, _, _ = validate_selection(p['selection'], 'train', p['binding'])
    claims = ROOT / 'runs' / 'evaluation-claims'; claims.mkdir(exist_ok=True)
    claim = claims / (p['selection_sha256'] + '-' + p['mode'] + '.json')
    write_json(claim, {'run': str(run), 'start_sha256': sha(run / 'start.json'), 'selection_sha256': p['selection_sha256'], 'partition': p['mode']}, exclusive=True)
    ids, y = store.labels(p['mode']); probabilities = {}; rows = []
    for trial in selected['trials']:
        if sha(trial['checkpoint']) != trial['checkpoint_sha256']:
            raise ValueError('Checkpoint hash drift at replay')
        saved = torch.load(trial['checkpoint'], weights_only=True)
        if saved['model'] != trial['model'] or saved['seed'] != trial['seed'] or saved['epoch'] != trial['best_epoch'] or saved['start_sha256'] != selected['start_sha256']:
            raise ValueError('Checkpoint semantic identity mismatch')
        model = FusionModel(trial['model'], store.clinical.shape[1], c); model.load_state_dict(saved['weights'])
        prob = predict(model, store, ids, c['training']['batch_size'])
        probabilities.setdefault(trial['model'], []).append(prob)
        rows.append({'model': trial['model'], 'seed': trial['seed'], **score(y, prob)})
    groups = store.table.iloc[ids].component.to_numpy()
    results = {'split': p['mode'], 'selection_sha256': p['selection_sha256'], 'n_images': len(ids), 'n_components': len(np.unique(groups)), 'n_patients': int(store.table.iloc[ids].patient_id.nunique()), 'per_seed': rows, 'mean_macro_f1': {name: float(np.mean([r['macro_f1'] for r in rows if r['model'] == name])) for name in c['models']}, 'uncertainty': bootstrap(y, probabilities, groups), 'label_reads': store.reads, 'scientific_result': False, 'publication_ready': False, 'clinical_validation': False}
    np.savez_compressed(run / 'predictions.npz', ids=ids, labels=y, groups=groups, **{name: np.array(prob) for name, prob in probabilities.items()})
    write_json(run / 'results.json', results, exclusive=True)
    print(json.dumps({'split': p['mode'], 'mean_macro_f1': results['mean_macro_f1'], 'uncertainty': results['uncertainty']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest='command', required=True)
    pre = commands.add_parser('prepare'); pre.add_argument('--mode', required=True); pre.add_argument('--run', required=True); pre.add_argument('--out', required=True); pre.add_argument('--selection'); pre.add_argument('--smoke')
    work = commands.add_parser('worker'); work.add_argument('--run', required=True)
    args = parser.parse_args()
    if args.command == 'prepare': proposal(args.mode, args.run, args.out, args.selection, args.smoke)
    else: worker(Path(args.run))
