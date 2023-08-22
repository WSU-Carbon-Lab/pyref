from pathlib import Path


def tree(path: Path, indent: str = "    ", level: int = 0):
    print(indent * level + f"- {path.name}/")
    for entry in path.iterdir():
        if entry.is_dir():
            tree(entry, indent, level + 1)


if __name__ == "__main__":
    print(tree(Path.cwd()))
