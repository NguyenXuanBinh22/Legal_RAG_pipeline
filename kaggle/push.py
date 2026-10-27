"""Đẩy code legal_rag + notebook crawl lên Kaggle (mọi thứ đều PRIVATE).

    python kaggle/push.py --user <kaggle_username>          # code + notebook (notebook chạy ngay)
    python kaggle/push.py --user <kaggle_username> --seed   # thêm data/raw hiện có để không crawl lại
    python kaggle/push.py --user <kaggle_username> --no-run # chỉ cập nhật dataset code

Chạy lại sau khi sửa code: dataset được tạo version mới, notebook được push lại.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "kaggle" / "build"
CODE_SLUG = "legal-rag-code"
SEED_SLUG = "legal-rag-raw-seed"
KERNEL_SLUG = "legal-rag-crawl"


def kaggle(*args: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    exe = shutil.which("kaggle", path=str(Path(sys.executable).parent)) or shutil.which("kaggle")
    if exe is None:
        sys.exit("Chưa có Kaggle CLI: pip install kaggle")
    if not quiet:
        print("$ kaggle", " ".join(args), flush=True)
    return subprocess.run([exe, *args], check=check, capture_output=quiet, text=True)


def dataset_status(ref: str) -> str | None:
    """'ready' / 'pending' / … hoặc None nếu dataset chưa tồn tại."""
    proc = kaggle("datasets", "status", ref, check=False, quiet=True)
    return proc.stdout.strip().lower() if proc.returncode == 0 else None


def wait_ready(ref: str, timeout: float = 900) -> None:
    # Dataset vừa upload cần vài phút để Kaggle xử lý; push notebook sớm quá thì input bị thiếu
    deadline = time.monotonic() + timeout
    while "ready" not in (status := dataset_status(ref) or ""):
        if time.monotonic() > deadline:
            sys.exit(f"{ref} chưa sẵn sàng sau {timeout:.0f}s (trạng thái: {status}). Chạy lại sau.")
        print(f"  {ref}: {status}, chờ…", flush=True)
        time.sleep(15)


def fresh_dir(name: str) -> Path:
    path = BUILD / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def push_dataset(user: str, slug: str, title: str, folder: Path, message: str) -> None:
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "unknown"}]}, indent=2), encoding="utf-8")
    if dataset_status(f"{user}/{slug}") is not None:
        kaggle("datasets", "version", "-p", str(folder), "-m", message)
    else:
        kaggle("datasets", "create", "-p", str(folder))  # mặc định private
    wait_ready(f"{user}/{slug}")


def build_code(folder: Path) -> None:
    with zipfile.ZipFile(folder / "legal_rag_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted((ROOT / "legal_rag").glob("*.py")):
            z.write(path, f"legal_rag/{path.name}")
        z.write(ROOT / "requirements.txt", "requirements.txt")


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)  # docs.jsonl có thể vài GB: hard link, không tốn thêm đĩa
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", required=True, help="username Kaggle")
    parser.add_argument("--seed", action="store_true", help="upload data/raw/docs.jsonl + checkpoint.json làm điểm xuất phát")
    parser.add_argument("--no-run", action="store_true", help="không push (và chạy) notebook")
    parser.add_argument("-m", "--message", default="update code")
    args = parser.parse_args()

    folder = fresh_dir(CODE_SLUG)
    build_code(folder)
    push_dataset(args.user, CODE_SLUG, "legal-rag-code", folder, args.message)
    sources = [f"{args.user}/{CODE_SLUG}"]

    raw = ROOT / "data" / "raw"
    if args.seed:
        folder = fresh_dir(SEED_SLUG)
        for name in ("docs.jsonl", "checkpoint.json", "crawl.log"):
            if (raw / name).exists():
                link_or_copy(raw / name, folder / name)
        push_dataset(args.user, SEED_SLUG, "legal-rag-raw-seed", folder, "seed từ máy local")
    if args.seed or dataset_status(f"{args.user}/{SEED_SLUG}") is not None:
        sources.append(f"{args.user}/{SEED_SLUG}")

    if args.no_run:
        return 0
    folder = fresh_dir("kernel")
    shutil.copy2(ROOT / "kaggle" / "crawl_kernel.py", folder / "crawl_kernel.py")
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": "crawl_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": sources,
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2), encoding="utf-8")
    kaggle("kernels", "push", "-p", str(folder))
    print(f"\nTheo dõi: https://www.kaggle.com/code/{args.user}/{KERNEL_SLUG}")
    print(f"Trạng thái: kaggle kernels status {args.user}/{KERNEL_SLUG}")
    print(f"Tải output: kaggle kernels output {args.user}/{KERNEL_SLUG} -p data_kaggle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
