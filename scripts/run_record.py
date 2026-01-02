from __future__ import annotations

import io
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Iterator, Optional


def _utc_now_iso() -> str:
  return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _try_git_info(repo_root: Path) -> dict[str, Any]:
  import subprocess

  def run(args: list[str]) -> str:
    try:
      out = subprocess.check_output(args, cwd=str(repo_root), stderr=subprocess.DEVNULL)
      return out.decode("utf-8", errors="replace").strip()
    except Exception:
      return ""

  return {
    "commit": run(["git", "rev-parse", "HEAD"]),
    "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    "dirty": run(["git", "status", "--porcelain"]) != "",
    "remote_origin": run(["git", "remote", "get-url", "origin"]),
  }


class _Tee(io.TextIOBase):
  def __init__(self, *streams: IO[str]):
    self._streams = streams

  def write(self, s: str) -> int:
    for st in self._streams:
      st.write(s)
      st.flush()
    return len(s)

  def flush(self) -> None:
    for st in self._streams:
      st.flush()


@dataclass
class TrainingIterRecord:
  it: int
  it_total: int
  ts_utc: str
  total_timesteps: Optional[int] = None
  steps_per_s: Optional[float] = None
  collection_s: Optional[float] = None
  learning_s: Optional[float] = None
  mean_reward: Optional[float] = None
  mean_ep_len: Optional[float] = None
  mean_action_noise_std: Optional[float] = None
  mean_value_loss: Optional[float] = None
  mean_surrogate_loss: Optional[float] = None
  mean_entropy_loss: Optional[float] = None
  scalars: dict[str, float] = None  # Episode_Reward/*, Metrics/*, Termination/*

  def to_dict(self) -> dict[str, Any]:
    d = {
      "iter": self.it,
      "iter_total": self.it_total,
      "ts_utc": self.ts_utc,
      "total_timesteps": self.total_timesteps,
      "steps_per_s": self.steps_per_s,
      "collection_s": self.collection_s,
      "learning_s": self.learning_s,
      "mean_reward": self.mean_reward,
      "mean_episode_length": self.mean_ep_len,
      "mean_action_noise_std": self.mean_action_noise_std,
      "mean_value_function_loss": self.mean_value_loss,
      "mean_surrogate_loss": self.mean_surrogate_loss,
      "mean_entropy_loss": self.mean_entropy_loss,
      "scalars": self.scalars or {},
    }
    return d


class RslRlStdoutParser:
  """Parse rsl_rl stdout into per-iteration JSONL records.

  This is intentionally lightweight: it doesn't depend on tensorboard.
  """

  _re_iter = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
  _re_comp = re.compile(
    r"Computation:\s+([0-9.]+)\s+steps/s\s+\(collection:\s+([0-9.]+)s,\s+learning\s+([0-9.]+)s\)"
  )
  _re_kv = re.compile(r"^\s*([^:]+):\s*([-+0-9.eE]+)\s*$")
  _re_total_ts = re.compile(r"Total timesteps:\s*(\d+)")

  def __init__(self, jsonl_path: Path):
    self.jsonl_path = jsonl_path
    self._fh = jsonl_path.open("a", encoding="utf-8")
    self._cur: Optional[TrainingIterRecord] = None

  def close(self) -> None:
    self._flush()
    self._fh.close()

  def _flush(self) -> None:
    if self._cur is None:
      return
    self._fh.write(json.dumps(self._cur.to_dict(), ensure_ascii=False) + "\n")
    self._fh.flush()
    self._cur = None

  def process_line(self, line: str) -> None:
    m = self._re_iter.search(line)
    if m:
      # starting a new iteration: flush previous.
      self._flush()
      it = int(m.group(1))
      it_total = int(m.group(2))
      self._cur = TrainingIterRecord(it=it, it_total=it_total, ts_utc=_utc_now_iso(), scalars={})
      return

    if self._cur is None:
      return

    m = self._re_comp.search(line)
    if m:
      self._cur.steps_per_s = float(m.group(1))
      self._cur.collection_s = float(m.group(2))
      self._cur.learning_s = float(m.group(3))
      return

    m = self._re_total_ts.search(line)
    if m:
      self._cur.total_timesteps = int(m.group(1))
      return

    m = self._re_kv.match(line)
    if not m:
      return

    key = m.group(1).strip()
    val = float(m.group(2))

    if key == "Mean reward":
      self._cur.mean_reward = val
    elif key == "Mean episode length":
      self._cur.mean_ep_len = val
    elif key == "Mean action noise std":
      self._cur.mean_action_noise_std = val
    elif key == "Mean value_function loss":
      self._cur.mean_value_loss = val
    elif key == "Mean surrogate loss":
      self._cur.mean_surrogate_loss = val
    elif key == "Mean entropy loss":
      self._cur.mean_entropy_loss = val
    else:
      # Keep all other scalar metrics.
      self._cur.scalars[key] = val


@contextmanager
def tee_stdout_to_file(log_path: Path) -> Iterator[tuple[IO[str], IO[str]]]:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with log_path.open("a", encoding="utf-8") as fh:
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, fh)  # type: ignore[assignment]
    sys.stderr = _Tee(old_err, fh)  # type: ignore[assignment]
    try:
      yield (sys.stdout, sys.stderr)
    finally:
      sys.stdout = old_out
      sys.stderr = old_err


def write_run_metadata(log_dir: Path, *, argv: list[str], extra: Optional[dict[str, Any]] = None) -> Path:
  repo_root = Path(__file__).resolve().parents[1]
  meta = {
    "ts_utc": _utc_now_iso(),
    "cwd": os.getcwd(),
    "argv": argv,
    "repo_root": str(repo_root),
    "git": _try_git_info(repo_root),
  }
  if extra:
    meta.update(extra)

  out_path = log_dir / "run_record.json"
  out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  return out_path


def update_latest_pointer(experiment_root: Path, log_dir: Path) -> None:
  experiment_root.mkdir(parents=True, exist_ok=True)
  (experiment_root / "_latest.txt").write_text(str(log_dir) + "\n", encoding="utf-8")


def load_run_metadata(run_dir: Path) -> Optional[dict[str, Any]]:
  """Load `run_record.json` from a run directory, if present."""
  try:
    p = run_dir / "run_record.json"
    if not p.exists():
      return None
    return json.loads(p.read_text(encoding="utf-8"))
  except Exception:
    return None


def record_eval(
  *,
  eval_log_dir: Path,
  checkpoint: Optional[str],
  site: str,
  metrics: dict[str, Any],
  extra: Optional[dict[str, Any]] = None,
) -> Path:
  eval_log_dir.mkdir(parents=True, exist_ok=True)
  repo_root = Path(__file__).resolve().parents[1]
  rec = {
    "ts_utc": _utc_now_iso(),
    "checkpoint": checkpoint,
    "site": site,
    "metrics": metrics,
    "git": _try_git_info(repo_root),
  }
  if extra:
    rec.update(extra)

  hist = eval_log_dir / "eval_history.jsonl"
  with hist.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
  return hist
