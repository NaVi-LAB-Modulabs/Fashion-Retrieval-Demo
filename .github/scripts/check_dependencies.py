"""Check that local and deployment dependency declarations stay in sync."""

from collections import Counter
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def declarations(entries):
    # Ignore ordering, whitespace, and comments, but retain duplicate declarations.
    return Counter(value for entry in entries
                   if (value := "".join(entry.partition("#")[0].split())))


def main():
    local = declarations((ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deployed = declarations(project["project"]["dependencies"])
    if local == deployed:
        print("Runtime dependency declarations match.")
        return 0
    print("Runtime dependency declarations differ:", file=sys.stderr)
    for filename, extra in (("requirements.txt", local - deployed),
                            ("pyproject.toml", deployed - local)):
        for entry in sorted(extra.elements()):
            print(f"  Only in {filename}: {entry}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
