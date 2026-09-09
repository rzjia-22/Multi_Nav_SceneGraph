import ast
from pathlib import Path


def test_all_launch_files_are_valid_python():
    launch_directory = Path(__file__).resolve().parents[1] / "launch"
    launch_files = sorted(launch_directory.glob("*.launch.py"))
    assert {path.name for path in launch_files} == {
        "phase1.launch.py",
        "phase2.launch.py",
        "system.launch.py",
    }
    for path in launch_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
