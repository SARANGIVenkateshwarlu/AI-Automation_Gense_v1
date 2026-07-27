from pathlib import Path
import sys


def write_file(path: Path, content: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def create_project(project_name: str = "ingestion_project"):
    base_dir = Path(project_name)

    folders = [
        base_dir / "data",
        base_dir / "database",
        base_dir / "logs",
        base_dir / "src",
        base_dir / "config",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    write_file(base_dir / ".env", "")

    write_file(
        base_dir / "requirements.txt",
        "\n".join([
            "langchain",
            "langgraph",
            "langchain-community",
            "pypdf",
            "unstructured",
            "jq",
        ]) + "\n"
    )

    write_file(
        base_dir / "README.md",
        """# Ingestion Project

## Folder structure
- data/ -> input files
- database/ -> saved processed output
- logs/ -> failed file logs
- src/ -> Python source code
- config/ -> settings

## Run
```bash
python src/main.py
```
"""
    )

    write_file(
        base_dir / "src" / "main.py",
        'print("LangChain + LangGraph ingestion project")\n'
    )

    write_file(
        base_dir / "config" / "settings.py",
        """DATA_DIR = "./data"
DB_DIR = "./database"
LOG_DIR = "./logs"
MAX_RETRIES = 3
"""
    )

    write_file(
        base_dir / "logs" / "failed_files.json",
        "[]\n"
    )

    print(f"Project structure created successfully: {base_dir.resolve()}")
    print("Next steps:")
    print(f"1. cd {project_name}")
    print("2. pip install -r requirements.txt")
    print("3. python src/main.py")


if __name__ == "__main__":
    project_name = sys.argv[1] if len(sys.argv) > 1 else "ingestion_project"
    create_project(project_name)