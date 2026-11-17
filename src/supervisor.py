"""External process-tree monitor. Review, immutable start, live resource samples."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
import psutil
from .common import ROOT, sha, json_digest, write_json
from .run import verify_review


def validate_journal(run):
    previous = None
    events = []
    start_sha = sha(run / 'start.json')
    for n, line in enumerate((run / 'events.jsonl').read_text().splitlines()):
        event = json.loads(line); digest = event.pop('sha256')
        if event['sequence'] != n or event['previous_sha256'] != previous or event['start_sha256'] != start_sha or json_digest(event) != digest:
            raise ValueError('Invalid event chain')
        if events and events[-1]['event'] == 'terminal':
            raise ValueError('Event after terminal')
        previous = digest; events.append(event)
    if not events or events[-1]['event'] != 'terminal' or events[-1]['payload']['status'] != 'success':
        raise ValueError('No unique successful terminal event')
    return events


def supervise(proposal_path, review_path):
    p, r = verify_review(proposal_path, review_path)
    command = [str(ROOT / '.venv/bin/python'), '-m', 'src.supervisor', '--proposal', str(Path(proposal_path).resolve()), '--review', str(Path(review_path).resolve())]
    if command != p['command'] or Path.cwd() != ROOT:
        raise ValueError('Launch differs from reviewed command/cwd')
    run = ROOT / p['run']
    if run.parent != ROOT / 'runs':
        raise ValueError('Unsafe run path')
    budgets = p['budgets']
    if psutil.virtual_memory().available < budgets['rss_limit_gib'] * 2**30:
        raise ValueError('Insufficient available RAM for reviewed budget')
    disk_bytes = sum(f.stat().st_size for f in (ROOT / 'data').rglob('*') if f.is_file())
    if disk_bytes > budgets['disk_limit_gib'] * 2**30 or shutil.disk_usage(ROOT).free < 2**30:
        raise ValueError('Disk budget or free-space preflight failed')
    run.parent.mkdir(exist_ok=True); run.mkdir()
    child_command = [str(ROOT / '.venv/bin/python'), '-m', 'src.run', 'worker', '--run', str(run)]
    start = {'created_at': datetime.now(timezone.utc).isoformat(), 'proposal': p, 'proposal_sha256': sha(proposal_path), 'review': r, 'review_sha256': sha(review_path), 'actual_command': command, 'child_command': child_command, 'cwd': str(ROOT), 'initial_data_bytes': disk_bytes, 'scientific_result': False, 'publication_ready': False}
    try:
        write_json(run / 'start.json', start, exclusive=True)
        start_sha = sha(run / 'start.json')
    except BaseException as exc:
        write_json(run / 'launch-failure.json', {'stage': 'immutable start', 'error': f'{type(exc).__name__}: {exc}', 'completed_cleanly': False}, exclusive=True)
        raise
    env = os.environ.copy()
    env.update({key: str(budgets['threads']) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    begin = time.monotonic(); peak = 0.0; actions = []; error = None; process = None
    soft = budgets['smoke_seconds'] if p['mode'] == 'smoke' else budgets['training_soft_seconds']
    hard = soft + 15 if p['mode'] == 'smoke' else budgets['training_hard_seconds']
    drained = None
    def drain(reason):
        nonlocal drained
        if drained is None and process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM); drained = time.monotonic()
            actions.append({'action': 'SIGTERM', 'reason': reason, 'elapsed_seconds': drained - begin})
    try:
        with (run / 'worker.log').open('x') as log, (run / 'resources.jsonl').open('x', buffering=1) as resources:
            process = subprocess.Popen(child_command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            parent = psutil.Process(process.pid); last_print = 0
            while process.poll() is None:
                tick = time.monotonic(); elapsed = tick - begin
                rss = 0; cpu = 0; processes = [parent]
                try: processes += parent.children(recursive=True)
                except psutil.NoSuchProcess: pass
                for proc in processes:
                    try:
                        rss += proc.memory_info().rss; times = proc.cpu_times(); cpu += times.user + times.system
                    except psutil.NoSuchProcess: pass
                peak = max(peak, rss / 2**20)
                sample = {'timestamp': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': elapsed, 'process_tree_rss_mib': rss / 2**20, 'process_tree_cpu_seconds': cpu, 'process_count': len(processes), 'free_disk_gib': shutil.disk_usage(ROOT).free / 2**30, 'start_sha256': start_sha, 'gpu': 'not used'}
                resources.write(json.dumps(sample) + '\n'); resources.flush(); os.fsync(resources.fileno())
                if elapsed - last_print >= 15 or last_print == 0:
                    print(json.dumps({'heartbeat': p['mode'], 'elapsed_seconds': round(elapsed, 1), 'rss_mib': round(sample['process_tree_rss_mib'], 1)}), flush=True); last_print = elapsed
                if rss > budgets['rss_limit_gib'] * 2**30: drain('process-tree RSS limit')
                if sample['free_disk_gib'] < 1: drain('free-disk limit')
                progress = run / 'events.jsonl'
                idle = time.time() - progress.stat().st_mtime if progress.exists() else elapsed
                if idle > budgets['worker_progress_timeout_seconds']: drain('worker progress stale')
                if elapsed > soft: drain('soft wall-clock deadline')
                if elapsed > hard or (drained and tick - drained > 15):
                    os.killpg(process.pid, signal.SIGKILL); actions.append({'action': 'SIGKILL', 'reason': 'drain grace/hard deadline', 'elapsed_seconds': elapsed}); break
                time.sleep(2)
            code = process.wait(timeout=20)
        if code != 0 or actions:
            raise RuntimeError(f'Worker did not complete cleanly: code={code}, actions={actions}')
        events = validate_journal(run)
        if p['mode'] in ('smoke', 'train'):
            selection = json.loads((run / 'selection.json').read_text())
            if [[t['model'], t['seed']] for t in selection['trials']] != p['planned_trials'] or selection['start_sha256'] != start_sha:
                raise ValueError('Incomplete or unbound selection')
            for trial in selection['trials']:
                if sha(trial['checkpoint']) != trial['checkpoint_sha256']:
                    raise ValueError('Final checkpoint drift')
        else:
            result = json.loads((run / 'results.json').read_text())
            if result['selection_sha256'] != p['selection_sha256'] or result['label_reads'] != {p['mode']: 1}:
                raise ValueError('Held-out result binding/access failure')
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        drain('monitor failure/interruption')
        if process and process.poll() is None:
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(); actions.append({'action': 'SIGKILL', 'reason': 'monitor drain timeout'})
    status = {'completed_cleanly': error is None, 'start_sha256': start_sha, 'peak_rss_mib': peak, 'elapsed_seconds': time.monotonic() - begin, 'returncode': process.returncode if process else None, 'actions': actions, 'error': error, 'scientific_result': False, 'publication_ready': False}
    write_json(run / 'resource-status.json', status, exclusive=True)
    print(json.dumps(status), flush=True)
    if error: raise RuntimeError(error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--proposal', required=True); parser.add_argument('--review', required=True)
    args = parser.parse_args(); supervise(args.proposal, args.review)
