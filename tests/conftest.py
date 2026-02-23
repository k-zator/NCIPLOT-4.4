from pathlib import Path
import sys


def _ensure_scripts_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    scripts_str = str(scripts)
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)


_ensure_scripts_on_path()
