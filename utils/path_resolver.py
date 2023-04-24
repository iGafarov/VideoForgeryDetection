from pathlib import Path


def absolute_path(relative_path: str):
    base_path = Path(__file__).parent.parent
    return (base_path / relative_path).resolve()