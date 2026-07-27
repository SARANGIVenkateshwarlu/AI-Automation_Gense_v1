import os
import json
import time
import sqlite3
import importlib.util
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document

# Optional LLM client example
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini")

DATA_DIR = "./data"
DB_PATH = "./database/files.db"
GENERATED_LOADERS_DIR = "./generated_loaders"
ERROR_LOG = "./logs/read_errors.json"
MAX_RETRIES = 2


class IngestState(TypedDict, total=False):
    files: List[str]
    current_index: int
    current_file: str
    extension: str
    docs: List[Document]
    retries: Dict[str, int]
    failed_files: List[Dict[str, Any]]
    processed_files: List[str]
    last_error: str
    use_generated_loader: bool
    generated_loader_path: str
    done: bool


def ensure_dirs():
    Path("./data").mkdir(parents=True, exist_ok=True)
    Path("./database").mkdir(parents=True, exist_ok=True)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    Path(GENERATED_LOADERS_DIR).mkdir(parents=True, exist_ok=True)


def init_sqlite():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ingested_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            file_path TEXT,
            page_content TEXT,
            metadata_json TEXT,
            loaded_by TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def append_error_log(file_path: str, error_msg: str):
    Path(ERROR_LOG).parent.mkdir(parents=True, exist_ok=True)
    if Path(ERROR_LOG).exists():
        with open(ERROR_LOG, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append({
        "file": file_path,
        "error": error_msg,
        "ts": time.time()
    })

    with open(ERROR_LOG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def scan_files(state: IngestState):
    ensure_dirs()
    init_sqlite()

    files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, f))
    ]

    return {
        "files": files,
        "current_index": 0,
        "retries": {},
        "failed_files": [],
        "processed_files": [],
        "done": len(files) == 0,
    }


def pick_next_file(state: IngestState):
    idx = state["current_index"]
    files = state["files"]

    if idx >= len(files):
        return {"done": True}

    current_file = files[idx]
    ext = os.path.splitext(current_file)[1].lower()

    return {
        "current_file": current_file,
        "extension": ext,
        "docs": [],
        "last_error": "",
        "use_generated_loader": False,
        "generated_loader_path": "",
        "done": False,
    }


def try_langchain_loader(state: IngestState):
    file_path = state["current_file"]
    ext = state["extension"]

    try:
        if ext in [".txt", ".py", ".md", ".log"]:
            docs = TextLoader(file_path, encoding="utf-8").load()
        elif ext == ".csv":
            docs = CSVLoader(file_path).load()
        elif ext == ".json":
            docs = JSONLoader(file_path=file_path, jq_schema=".", text_content=False).load()
        elif ext == ".pdf":
            docs = PyPDFLoader(file_path).load()
        elif ext == ".docx":
            docs = UnstructuredWordDocumentLoader(file_path).load()
        else:
            raise ValueError(f"Unsupported standard format: {ext}")

        return {"docs": docs, "last_error": ""}
    except Exception as e:
        err = f"langchain_loader_failed: {str(e)}"
        append_error_log(file_path, err)
        return {"docs": [], "last_error": err}


def generate_loader_code_with_llm(state: IngestState):
    file_path = state["current_file"]
    ext = state["extension"]
    safe_ext = ext.replace(".", "") or "unknown"
    loader_file = os.path.join(GENERATED_LOADERS_DIR, f"loader_{safe_ext}.py")

    # In production, replace this with an LLM call that generates code.
    # LangGraph docs recommend handling recoverable issues via state + loop-back routing. [page:1]
    generated_code = f'''
from langchain_core.documents import Document

def load_file(file_path: str):
    with open(file_path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="ignore")
    if not text.strip():
        raise ValueError("Generated loader could not extract readable text for {ext}")
    return [Document(page_content=text, metadata={{"source": file_path, "loader": "generated_{safe_ext}"}})]
'''.strip()

    with open(loader_file, "w", encoding="utf-8") as f:
        f.write(generated_code)

    return {
        "generated_loader_path": loader_file,
        "use_generated_loader": True
    }


def implement_generated_loader(state: IngestState):
    loader_path = state["generated_loader_path"]
    file_path = state["current_file"]

    try:
        spec = importlib.util.spec_from_file_location("generated_loader_module", loader_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        docs = module.load_file(file_path)
        return {"docs": docs, "last_error": ""}
    except Exception as e:
        err = f"generated_loader_failed: {str(e)}"
        append_error_log(file_path, err)
        return {"docs": [], "last_error": err}


def save_to_sql(state: IngestState):
    docs = state.get("docs", [])
    file_path = state["current_file"]
    loader_used = "generated_loader" if state.get("use_generated_loader") else "langchain_loader"

    if not docs:
        return {"last_error": "No documents available for SQL save"}

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        for d in docs:
            cur.execute("""
                INSERT INTO ingested_files (
                    file_name, file_path, page_content, metadata_json, loaded_by
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                os.path.basename(file_path),
                file_path,
                d.page_content,
                json.dumps(d.metadata, ensure_ascii=False),
                loader_used
            ))

        conn.commit()
        conn.close()

        processed = state.get("processed_files", [])
        processed.append(file_path)

        return {"processed_files": processed, "last_error": ""}
    except Exception as e:
        err = f"sql_save_failed: {str(e)}"
        append_error_log(file_path, err)
        return {"last_error": err}


def handle_result(state: IngestState):
    current_file = state["current_file"]
    retries = state.get("retries", {})
    failed_files = state.get("failed_files", [])
    idx = state["current_index"]
    err = state.get("last_error", "")

    if not err:
        return {"current_index": idx + 1}

    attempts = retries.get(current_file, 0) + 1
    retries[current_file] = attempts

    if attempts < MAX_RETRIES:
        return {"retries": retries}

    failed_files.append({
        "file": current_file,
        "error": err,
        "attempts": attempts
    })
    return {
        "retries": retries,
        "failed_files": failed_files,
        "current_index": idx + 1
    }


def route_after_loader(state: IngestState):
    return "save_to_sql" if state.get("docs") else "generate_loader_code_with_llm"


def route_after_generated_loader(state: IngestState):
    return "save_to_sql" if state.get("docs") else "handle_result"


def route_after_handle(state: IngestState):
    if state.get("done"):
        return END
    if state["current_index"] >= len(state["files"]):
        return END

    current_file = state.get("current_file")
    retries = state.get("retries", {})
    err = state.get("last_error", "")

    if err and retries.get(current_file, 0) < MAX_RETRIES:
        return "try_langchain_loader"

    return "pick_next_file"


builder = StateGraph(IngestState)

builder.add_node("scan_files", scan_files)
builder.add_node("pick_next_file", pick_next_file)
builder.add_node("try_langchain_loader", try_langchain_loader)
builder.add_node("generate_loader_code_with_llm", generate_loader_code_with_llm)
builder.add_node("implement_generated_loader", implement_generated_loader)
builder.add_node("save_to_sql", save_to_sql)
builder.add_node("handle_result", handle_result)

builder.add_edge(START, "scan_files")
builder.add_edge("scan_files", "pick_next_file")
builder.add_conditional_edges("pick_next_file", lambda s: END if s.get("done") else "try_langchain_loader")
builder.add_conditional_edges("try_langchain_loader", route_after_loader)
builder.add_edge("generate_loader_code_with_llm", "implement_generated_loader")
builder.add_conditional_edges("implement_generated_loader", route_after_generated_loader)
builder.add_edge("save_to_sql", "handle_result")
builder.add_conditional_edges("handle_result", route_after_handle)

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({})
    print("Processed:", len(result.get("processed_files", [])))
    print("Failed:", result.get("failed_files", []))