# Crawl trên Kaggle

Chạy `crawl` trên notebook Kaggle (CPU, có Internet) thay vì máy local. Dataset và notebook đều **private**.

## Một lần: token Kaggle

1. Kaggle → Settings → Phone verification (bắt buộc để notebook bật được Internet).
2. Settings → API → **Create New Token** → tải `kaggle.json` về `C:\Users\<bạn>\.kaggle\kaggle.json`.
   Nếu Kaggle cấp token dạng `KGAT_…`: lưu vào `C:\Users\<bạn>\.kaggle\access_token`.
3. Kiểm tra: `.\venv\Scripts\kaggle datasets list --mine`

## Đẩy lên và chạy

```powershell
# Có seed: upload data/raw hiện có (~560MB, 1.852 văn bản) để Kaggle crawl tiếp từ trang 187
python kaggle/push.py --user <username> --seed

# Lần sau chỉ sửa code: dataset code lên version mới, notebook được push và chạy lại
python kaggle/push.py --user <username> -m "sửa crawl"
```

`push.py` tạo dataset `legal-rag-code` (zip của `legal_rag/` + `requirements.txt`), tuỳ chọn `legal-rag-raw-seed`,
rồi push notebook `legal-rag-crawl` (`crawl_kernel.py`). Notebook chạy ngay sau khi push:

1. `probe`: dừng sớm nếu Kaggle không gọi được gateway VBPL (Internet tắt, hoặc bị chặn IP nước ngoài).
2. `crawl --all --workers 4 --delay 0.5`, tự dừng sau 11 giờ (Kaggle cắt phiên ở 12 giờ và khi đó không lưu output).
3. Nếu đã hết danh sách và còn hơn 1 giờ: `crawl --retry-failed`, rồi `reconcile`.

```powershell
.\venv\Scripts\kaggle kernels status <username>/legal-rag-crawl
.\venv\Scripts\kaggle kernels output <username>/legal-rag-crawl -p data_kaggle   # data_kaggle/data/raw/...
```

Copy `data_kaggle/data/raw/*` vào `data/raw/` rồi chạy tiếp `preprocess`, `chunk`, `embed` ở máy local.

**Chưa crawl xong trong 1 phiên:** mở notebook trên Kaggle → Add Input → Your Work → output của chính
`legal-rag-crawl` → Save & Run All. Script tìm `checkpoint.json` + `docs.jsonl` lớn nhất trong `/kaggle/input` và resume.

Giới hạn cần biết: output tối đa ~20GB (`docs.jsonl` đầy đủ ~11GB), phiên tối đa 12 giờ.
