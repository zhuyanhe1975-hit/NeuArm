from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_mjlab_on_path(mjlab_src: str | None = None) -> Path:
  """Ensure mjlab (source checkout) is importable without installation.

  Preference order:
  1) explicit `mjlab_src` argument
  2) `MJLAB_SRC` environment variable
  3) default path `/home/yhzhu/AI/mjlab/src`
  """
  mjlab_src = (
    mjlab_src
    or os.environ.get("MJLAB_SRC")
    or "/home/yhzhu/AI/mjlab/src"
  )
  mjlab_src_path = Path(mjlab_src).expanduser().resolve()
  if not mjlab_src_path.exists():
    raise FileNotFoundError(
      f"mjlab source path not found: {mjlab_src_path}. "
      "Set MJLAB_SRC=/path/to/mjlab/src."
    )

  # Ensure Warp kernel cache is inside the repo (sandbox writable).
  repo_root = Path(__file__).resolve().parents[2]
  warp_cache = repo_root / ".warp_cache"
  os.environ.setdefault("WARP_CACHE_PATH", str(warp_cache))
  warp_cache.mkdir(parents=True, exist_ok=True)

  if str(mjlab_src_path) not in sys.path:
    sys.path.insert(0, str(mjlab_src_path))
  return mjlab_src_path
