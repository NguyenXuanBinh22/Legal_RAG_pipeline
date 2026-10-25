# Legal RAG: Crawl + Preprocess VBPL

Viết lại Phase 1 (crawl) và Phase 2 (preprocess) của `R2AI2026-BUILD-AI-LEGAL-ASSISTANT/pipeline/crawl_preprocess.py`.
Output `data/processed/processed_data.json` giữ đúng định dạng mà `chunk_embedding.py` (Phase 3) đang đọc.

## Chạy từng bước

Chạy mọi lệnh từ thư mục gốc `Legal_RAG/`.

```powershell
# 0. Cài đặt
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# 1. Test offline (không gọi mạng)
pytest

# 2. Gọi thử API: 1 trang danh sách + 1 văn bản, không ghi file
python -m legal_rag probe

# 3. Crawl demo: mặc định chỉ 2 trang × 10 văn bản
python -m legal_rag crawl
python -m legal_rag crawl --max-pages 1 --max-docs 5     # nhỏ hơn nữa

# 4. Xem tiến độ
python -m legal_rag status

# 5. (Khi có lỗi) thử lại trang/văn bản lỗi đã ghi trong checkpoint
python -m legal_rag crawl --retry-failed

# 6. Preprocess: khử trùng, lọc hiệu lực, dựng cây mục lục
python -m legal_rag preprocess
```

Smoke test với API thật (2 request): `$env:RUN_LIVE="1"; pytest -m live`

Chạy lại lệnh crawl sẽ tiếp tục từ `next_page` trong checkpoint. Văn bản đã có trong `docs.jsonl` không bị tải hay ghi lại.

### Crawl toàn bộ (khi đã kiểm tra xong demo)

```powershell
python -m legal_rag crawl --all --workers 4 --delay 0.5
python -m legal_rag crawl --retry-failed
python -m legal_rag reconcile        # liệt kê lại danh sách (page 100) và crawl bù văn bản bị trượt trang
python -m legal_rag preprocess
```

Khoảng 37.673 văn bản: 3.768 request danh sách và 37.673 request detail.

## Dữ liệu

```
data/
├── raw/
│   ├── docs.jsonl        1 dòng = {doc_id, page, fetched_at, list_item, detail}
│   ├── checkpoint.json   next_page, total, last_page, failed_pages, failed_docs
│   └── crawl.log
└── processed/
    ├── processed_data.json   [ {...detail, "metadata": cây mục lục} ]  → đầu vào Phase 3
    ├── error_items.json      văn bản bị loại + lý do
    └── report.json           thống kê: trùng, lý do loại, số node theo cấp, id HTML lặp
```

Để dùng với Phase 3 của repo cũ: copy `data/processed/processed_data.json` vào thư mục gốc repo đó rồi chạy `python pipeline/chunk_embedding.py phase3`.

## Khác với code cũ

| Vấn đề ở `crawl_preprocess.py` | Cách xử lý ở đây |
|---|---|
| `TOTAL_PAGES = 36916`, nhưng API chỉ có 37.673 văn bản (3.768 trang size 10). ~33.000 request thừa | Đọc `data.total` từ response, dừng ở trang cuối |
| Mục lục lấy qua Next.js server action của vbpl.vn với hash `next-action` ghi cứng; hiện bị WAF trả **403** | Bỏ request này. Dựng cây từ class `prov-*` + `id` có sẵn trong HTML của API detail (`structure.py`). Chỉ còn 1 request/văn bản |
| Thiếu dòng `1:` trong response mục lục vẫn `return True` → mất mục lục hàng loạt không báo | Không còn phụ thuộc. Ngoài ra có bộ ngắt: một trang có ≥80% văn bản lỗi thì dừng crawl |
| Detail được ghi trước khi lấy mục lục → retry/resume ghi trùng | Mỗi văn bản là 1 dòng ghi một lần; khử trùng theo `doc_id` khi resume và ở preprocess |
| Trang danh sách lỗi bị ghi đè checkpoint → mất 10 văn bản | `failed_pages` trong checkpoint + `crawl --retry-failed` |
| Checkpoint ghi thường, crash giữa chừng → JSON hỏng | Ghi `.tmp` rồi `os.replace`; `docs.jsonl` tự cắt dòng ghi dở khi mở lại |
| Retry chồng 3 tầng (tới ~12 phút/văn bản khi API sập) | 1 tầng: urllib3 `Retry(3, backoff 1s)` cho 429/5xx, tôn trọng `Retry-After` |
| `data: null` được ghi vào file → Phase 2 crash | Báo lỗi `detail_null`, không ghi |
| Sort theo lượt xem → trượt trang, trùng/sót | Khử trùng + lệnh `reconcile`. (Các sort khác của API trả về tập 173.871 văn bản khác hẳn, nên giữ truy vấn cũ) |
| Phase 2 nạp toàn bộ vào RAM, không khử trùng | Đọc 2 lượt theo offset, bản crawl mới nhất thắng, ghi JSON array dạng stream |

Cây mục lục dựng từ HTML dùng cùng schema node (`id, title, level, ptype, orderIndex, children, isLeaf`). Tiêu đề Khoản/Điểm có dạng `Khoản 2`, `Điểm a`, giống dữ liệu cũ.

**Đã đối chiếu với dữ liệu cũ** (Bộ luật Dân sự 91/2015/QH13, chạy `parse_document` của Phase 3 cũ lên output mới): 1.857/1.857 id phần tử khớp với cây vbpl.vn cũ, 1.767 chunk có `embed_text` giống hệt. Phần lệch đến từ VBPL sửa câu chữ (41), lỗi dòng nối tiếp của Phase 3 (49, xem dưới) và các chunk rỗng kiểu `Điều 256 - Mục 1` mà cây cũ lồng sai (59, cây mới không sinh ra).

### Lưu ý cho Phase 3

HTML của VBPL tách một đơn vị thành nhiều `<p>` **cùng `id`** (dòng sau có `data-merged="true"`), ví dụ `Mục 1` + `NĂNG LỰC PHÁP LUẬT DÂN SỰ…`, hoặc `Điều 292. Biện pháp…` + `Các biện pháp bảo đảm… bao gồm:`. `structure.py` gộp chúng thành một node (`merged_rows` trong `report.json`). Nhưng `chunk_embedding.py` cũ vẫn duyệt từng `<p>`, và `id_to_el`/`node_full_title` giữ phần tử **cuối**. Hệ quả: chunk bị trùng `chunk_id` (17.927 id trong `chunks.db` cũ), tiêu đề Điều bị thay bằng dòng tiếp theo, và phần đầu của Khoản bị tách mất nội dung. Cần sửa ở Phase 3: gộp text của mọi phần tử cùng id theo thứ tự.

## Cấu trúc code

```
legal_rag/
├── config.py        endpoint, truy vấn danh sách, trạng thái hiệu lực được giữ
├── vbpl_client.py   list_page, get_detail; map mọi lỗi thành VbplError(reason)
├── storage.py       JsonlStore (append-only, khử trùng), Checkpoint (ghi nguyên tử)
├── crawl.py         crawl, retry_failed, reconcile
├── structure.py     HTML → cây Phần/Chương/Mục/Điều/Khoản/Điểm
├── preprocess.py    Phase 2
└── __main__.py      CLI
tests/               test offline với client giả lập + 1 smoke test live
```
