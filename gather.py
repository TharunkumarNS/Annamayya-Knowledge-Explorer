"""Combine selected Python source files into a single review-friendly text file.

Run ``python gather.py`` and choose specific files, all Python files, or files
modified recently. Output is written to ``gathered_code.txt`` in this directory.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "gathered_code.txt"


def python_files(root: Path = PROJECT_ROOT) -> list[Path]:
    """List project Python files, excluding common generated/dependency folders."""
    excluded = {".git", ".venv", "venv", "__pycache__", "site-packages", "node_modules"}
    return sorted(
        path
        for path in root.rglob("*.py")
        if not any(part in excluded for part in path.relative_to(root).parts)
    )


def recently_modified(files: list[Path], since_timestamp: float) -> list[Path]:
    """Select files modified since the current gather session began."""
    return [path for path in files if path.stat().st_mtime >= since_timestamp]


def write_bundle(files: list[Path], output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Write source contents with clear file separators; return output path."""
    resolved_output = output_path.resolve()
    selected = [path for path in files if path.resolve() != resolved_output]
    if not selected:
        raise ValueError("No Python files were selected to gather.")

    sections = [
        "Python source bundle",
        f"Created: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Files included: {len(selected)}",
        "",
    ]
    for path in selected:
        try:
            relative = path.resolve().relative_to(PROJECT_ROOT)
        except ValueError:
            relative = path
        content = path.read_text(encoding="utf-8")
        sections.extend(
            [
                "=" * 88,
                f"FILE: {relative}",
                "=" * 88,
                content.rstrip(),
                "",
            ]
        )
    resolved_output.write_text("\n".join(sections), encoding="utf-8")
    return resolved_output


def _choose_files(files: list[Path]) -> list[Path]:
    if not files:
        return []
    print("Python files in project:")
    for number, path in enumerate(files, start=1):
        print(f"  {number}. {path.relative_to(PROJECT_ROOT)}")
    answer = input("Enter file numbers separated by commas (or 'all'): ").strip().lower()
    if answer == "all":
        return files

    selected: list[Path] = []
    for value in answer.split(","):
        value = value.strip()
        if not value:
            continue
        try:
            index = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid file number: {value!r}") from exc
        if index < 1 or index > len(files):
            raise ValueError(f"File number out of range: {index}")
        path = files[index - 1]
        if path not in selected:
            selected.append(path)
    return selected


def main() -> None:
    """Prompt for the bundle mode, then write the selected source files."""
    session_started = datetime.now().timestamp()
    files = python_files()
    if not files:
        print(f"No Python files found under {PROJECT_ROOT}.")
        return

    print("Gather Python files into one text file")
    print("  1. Select specific files")
    print("  2. Gather all Python files")
    print("  3. Gather files modified since this gather session started")
    choice = input("Choose 1, 2, or 3: ").strip()

    if choice == "1":
        selected = _choose_files(files)
    elif choice == "2":
        selected = files
    elif choice == "3":
        selected = recently_modified(files, session_started)
    else:
        raise ValueError("Choose 1, 2, or 3.")

    if not selected:
        print("No Python files matched that selection. No output file was written.")
        return

    output = write_bundle(selected)
    print(f"Gathered {len(selected)} file(s) into {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not gather Python files: {exc}") from exc
