from __future__ import annotations

import argparse
import json
from pathlib import Path


def _tail_lines(path: Path, n: int) -> list[str]:
  if not path.exists():
    return []
  with path.open('rb') as f:
    f.seek(0, 2)
    size = f.tell()
    block = 4096
    data = b''
    while size > 0 and data.count(b'\n') <= n:
      step = min(block, size)
      size -= step
      f.seek(size)
      data = f.read(step) + data
    lines = data.splitlines()[-n:]
  return [ln.decode('utf-8', errors='replace') for ln in lines]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--experiment', default='neuarm_irb2400_tracking')
  ap.add_argument('--last-iters', type=int, default=5)
  args = ap.parse_args()

  repo_root = Path(__file__).resolve().parents[1]
  exp_root = repo_root / 'logs' / 'rsl_rl' / args.experiment
  latest_ptr = exp_root / '_latest.txt'
  if not latest_ptr.exists():
    raise SystemExit(f'No latest pointer: {latest_ptr}')

  run_dir = Path(latest_ptr.read_text(encoding='utf-8').strip())
  print(f'latest_run: {run_dir}')

  meta_path = run_dir / 'run_record.json'
  if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    print('git:', meta.get('git', {}))
    print('args:', meta.get('args', meta.get('extra', {})))

  train_jsonl = run_dir / 'train_metrics.jsonl'
  print(f'train_metrics: {train_jsonl}')
  for ln in _tail_lines(train_jsonl, args.last_iters):
    try:
      rec = json.loads(ln)
      it = rec.get('iter')
      ts = rec.get('total_timesteps')
      sps = rec.get('steps_per_s')
      rmse = rec.get('scalars', {}).get('Metrics/traj/joint_pos_rmse')
      tcp = rec.get('scalars', {}).get('TCP_error_mm')
      print(f'  iter={it} ts={ts} steps/s={sps} joint_rmse={rmse} tcp={tcp}')
    except Exception:
      print('  (bad line)', ln[:120])

  eval_hist = exp_root / '_eval' / 'eval_history.jsonl'
  if eval_hist.exists():
    last = _tail_lines(eval_hist, 1)
    if last:
      ev = json.loads(last[0])
      print('last_eval:', {'ts_utc': ev.get('ts_utc'), 'site': ev.get('site'), 'checkpoint': ev.get('checkpoint')})
      print('  metrics:', ev.get('metrics', {}))


if __name__ == '__main__':
  main()
