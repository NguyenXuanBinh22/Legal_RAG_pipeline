"""Kaggle: crawl VBPL bằng legal_rag, lưu data/raw vào /kaggle/working để tải về hoặc chạy tiếp.

Đẩy lên bằng `python kaggle/push.py` (xem kaggle/README.md). Cần bật Internet cho notebook.
Chạy tiếp từ lần trước: thêm output của notebook này (hoặc dataset legal-rag-raw-seed) làm
input. Script tự tìm checkpoint.json + docs.jsonl trong /kaggle/input và resume từ đó.
"""

import importlib.util
import json
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
DATA_DIR = Path("/kaggle/working/data")
RAW_DIR = DATA_DIR / "raw"
SRC = Path("/tmp/legal_rag_src")

# Kaggle dừng phiên ở 12h và khi đó KHÔNG lưu output → tự dừng sớm, để thời gian ghi output
TIME_BUDGET_S = 11 * 3600
GRACE_S = 300
CRAWL_ARGS = ["--all", "--workers", "4", "--delay", "0.5"]
# Module legal_rag import ở cấp CLI nhưng image Kaggle có thể chưa có
EXTRA_PACKAGES = {"ijson": "ijson", "faiss": "faiss-cpu", "dotenv": "python-dotenv"}

started = time.monotonic()


def remaining() -> float:
    return TIME_BUDGET_S - (time.monotonic() - started)


def install_code() -> None:
    # Kaggle có thể giữ nguyên file zip hoặc giải nén sẵn khi mount dataset
    shutil.rmtree(SRC, ignore_errors=True)
    for main in INPUT.rglob("legal_rag/__main__.py"):
        shutil.copytree(main.parent, SRC / "legal_rag")
        break
    else:
        archive = next(INPUT.rglob("legal_rag_code.zip"), None)
        if archive is None:
            sys.exit("Không thấy code legal_rag trong /kaggle/input. Thêm dataset legal-rag-code làm input.")
        with zipfile.ZipFile(archive) as z:
            z.extractall(SRC)
    missing = [pkg for mod, pkg in EXTRA_PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def seed_raw_data() -> None:
    """Chép docs.jsonl + checkpoint.json lớn nhất tìm thấy trong input vào /kaggle/working (input chỉ đọc)."""
    if (RAW_DIR / "docs.jsonl").exists():
        return
    candidates = [c.parent for c in INPUT.rglob("checkpoint.json") if (c.parent / "docs.jsonl").exists()]
    if not candidates:
        print("Không có dữ liệu crawl trước đó, bắt đầu từ trang 1.")
        return
    best = max(candidates, key=lambda d: (d / "docs.jsonl").stat().st_size)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("docs.jsonl", "checkpoint.json", "crawl.log"):
        if (best / name).exists():
            shutil.copy2(best / name, RAW_DIR / name)
    ckpt = json.loads((RAW_DIR / "checkpoint.json").read_text(encoding="utf-8"))
    print(f"Resume từ {best}: trang tiếp theo {ckpt.get('next_page')}/{ckpt.get('last_page')}")


def run(*args: str, timeout: float | None = None) -> int:
    cmd = [sys.executable, "-m", "legal_rag", "--data-dir", str(DATA_DIR), *args]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=SRC)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Checkpoint ghi sau mỗi trang, docs.jsonl tự sửa dòng ghi dở → dừng lúc nào cũng an toàn
        print("Hết thời gian của phiên, dừng crawl. Chạy lại notebook để tiếp tục.", flush=True)
        proc.send_signal(signal.SIGINT)
        try:
            return proc.wait(timeout=GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()


def crawl_done() -> bool:
    path = RAW_DIR / "checkpoint.json"
    return path.exists() and bool(json.loads(path.read_text(encoding="utf-8")).get("done"))


def main() -> None:
    install_code()
    seed_raw_data()

    # Kiểm tra mạng trước: Kaggle chạy ở nước ngoài, gateway VBPL có thể chặn theo IP/vùng
    if run("probe", "--n", "1") != 0:
        sys.exit("Probe thất bại: kiểm tra Settings → Internet đã bật chưa, hoặc gateway VBPL chặn IP Kaggle.")

    run("crawl", *CRAWL_ARGS, timeout=remaining())
    if crawl_done() and remaining() > 3600:
        run("crawl", "--retry-failed", "--workers", "4", "--delay", "0.5", timeout=remaining())
        run("reconcile", "--workers", "4", "--delay", "0.5", timeout=remaining())
    run("status")


if __name__ == "__main__":
    main()
