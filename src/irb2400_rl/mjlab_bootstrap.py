from __future__ import annotations

import os
import sys
from pathlib import Path


# Silence known Warp deprecation warnings emitted by warp 1.12+ when used
# through mujoco_warp/mjlab. Set NEUARM_SHOW_WARP_DEPRECATIONS=1 to show them.
def _silence_warp_deprecations() -> None:
  """Silence Warp DeprecationWarning prints.

  Warp overrides warnings.showwarning and forces simplefilter("default"), so
  filtering via warnings.filterwarnings() is not reliable. We patch Warp
  internal warn() to ignore DeprecationWarning messages instead.
  """
  if os.getenv("NEUARM_SHOW_WARP_DEPRECATIONS", "0") == "1":
    return

  try:
    import warp._src.utils as _wutils

    _orig_warn = _wutils.warn

    def _warn_no_deprec(message, category=None, stacklevel=1, once=False):
      if category is DeprecationWarning:
        return
      return _orig_warn(message, category, stacklevel=stacklevel, once=once)

    _wutils.warn = _warn_no_deprec
  except Exception:
    # If Warp is unavailable, do nothing.
    return



def ensure_mjlab_on_path(mjlab_src: str | None = None) -> Path:
  """Ensure mjlab (source checkout) is importable without installation.

  Preference order:
  1) explicit `mjlab_src` argument
  2) `MJLAB_SRC` environment variable
  3) default path `/home/yhzhu/AI/mjlab/src`
  """
  _silence_warp_deprecations()
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
