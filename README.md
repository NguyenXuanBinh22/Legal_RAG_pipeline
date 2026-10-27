# Legal RAG: Crawl → Preprocess → Chunk → Embed VBPL

Viết lại pipeline dữ liệu của `R2AI2026-BUILD-AI-LEGAL-ASSISTANT/pipeline/`:

| Repo cũ | Ở đây |
|---|---|
| `crawl_preprocess.py` (Phase 1-2) | `crawl`, `preprocess` |
| `chunk_embedding.py` (Phase 3-4) + `optimize_data.py` (Phase 5) | `chunk`, `embed`, `fp16` |

Output cuối (`chunks.db`, `faiss.index`, `faiss_fp16.index`) giữ đúng tên file và schema mà `services/Search.py` của repo cũ đang đọc.

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

# 7. Chunk: processed_data.json → data/index/chunks.db
python -m legal_rag chunk

# 8. Embed: chunks.db → data/index/faiss.index (cần server embedding, vd. LM Studio)
python -m legal_rag embed --max-chunks 200      # demo
python -m legal_rag embed                       # chạy lại = resume tới hết

# 9. Nén index fp16 cho máy ít RAM (dựng lại sau mỗi lần embed)
python -m legal_rag fp16
```

Cấu hình embedding đọc từ biến môi trường hoặc file `.env` ở thư mục gốc, cùng tên biến với repo cũ (`EMBEDDING_BASE_URL`, `EMBEDDING_MODEL_NAME`, `EMBEDDING_API_KEY`). Có thể ghi đè bằng `--base-url`, `--model`. Model khi embed phải là model mà `services/Chat.py` dùng để embed query.

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
├── processed/
│   ├── processed_data.json   [ {...detail, "metadata": cây mục lục} ]  → đầu vào Phase 3
│   ├── error_items.json      văn bản bị loại + lý do
│   └── report.json           thống kê: trùng, lý do loại, số node theo cấp, id HTML lặp
└── index/
    ├── chunks.db             SQLite: chunks(faiss_id, chunk_id, doc_id, doc_num, title, article, clause, content) + docs
    ├── chunk_report.json     số chunk, văn bản bị loại, chunk chỉ có tiêu đề, chunk dài nhất
    ├── chunk_errors.json     văn bản không tạo được chunk + lý do
    ├── faiss.index           float32, IndexIDMap(IndexFlatIP), id = faiss_id. Bản gốc để resume
    ├── embed_skipped.json    faiss_id → lý do (chunk vượt context model, không thử lại)
    └── faiss_fp16.index      bản nén, SearchService ưu tiên đọc
```

Để dùng với app của repo cũ: copy `chunks.db` và `faiss_fp16.index` (thêm `faiss.index` nếu máy dư RAM) từ `data/index/` vào `R2AI2026-BUILD-AI-LEGAL-ASSISTANT/data/`. Các file `chunks.json`, `faiss_id_map.json`, `chunk_map.json`, `doc_index_map.json`, `article_index_map.json`, `indexed_ids.json` không còn cần, vì `SearchService` không đọc chúng.

`faiss_id` là số thứ tự chunk trong `chunks.db`, nên `chunk` từ chối chạy lại khi đã có index (`--force` để chunk lại và xoá index cũ).

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

### Phase 3-5: gộp `chunk_embedding.py` + `optimize_data.py`

Hai file cũ chồng chéo: cùng khai báo đường dẫn, Phase 3 ghi `chunks.json` + 4 file map JSON, rồi `optimize_data.py` đọc lại `chunks.json` + `faiss_id_map.json` chỉ để chép sang `chunks.db` và kiểm tra hai file có khớp thứ tự không. Giờ `chunks.db` là nguồn dữ liệu duy nhất. `chunk` ghi thẳng vào đó, `embed` đọc từ đó, và `faiss_id` khớp với index vì nó chính là khoá chính.

| Vấn đề ở code cũ | Cách xử lý ở đây |
|---|---|
| HTML VBPL tách một đơn vị thành nhiều phần tử **cùng `id`** (dòng nối tiếp `data-merged="true"`, hoặc `prov-content` mang id của Điều/Khoản). Phase 3 duyệt từng phần tử và giữ phần tử cuối → chunk trùng `chunk_id` (17.927 id trong `chunks.db` cũ), tiêu đề Điều bị thay bằng dòng sau | Gộp mọi dòng cùng id theo thứ tự đọc: dòng mang class của cấp đó là tiêu đề, các dòng khác là nội dung (`chunking.py`) |
| Đoạn không id sau Khoản/Điểm (vd. `Trong đó: - P: là…`) bị bỏ, vì chỉ Điều không Khoản mới gom đoạn phía sau | Một quy tắc chung cho mọi đơn vị lá: gom tới khi gặp đơn vị cấu trúc kế tiếp |
| `article` = `Điều\xa01` (NBSP) hoặc rỗng khi tiêu đề bị thay → `SearchService` lọc `== "Điều 1"` trượt | Chuẩn hoá NFC + khoảng trắng; số Điều lấy từ tiêu đề đã gộp (nhận cả `Điều 5a`, `ĐIỀU 5`) |
| Phase 3 nạp cả `processed_data.json` vào RAM | Đọc stream bằng `ijson`, ghi SQLite theo batch, `.tmp` → `os.replace` |
| Phase 4 gọi API **1 chunk/request** (`EMBED_BATCH_SIZE` chỉ là cỡ batch thêm vào index) | Gửi `--batch-size` chunk/request. Batch dính chunk vượt context thì embed lẻ từng chunk |
| Mọi lỗi API đều "skip chunk" → server sập thì phần còn lại bị bỏ hết | Chỉ bỏ qua lỗi vượt context (ghi `embed_skipped.json`). Lỗi khác dừng lần chạy, chạy lại sẽ resume |
| Ghi lại toàn bộ `faiss.index` (~3.2GB) sau mỗi 500 vector; `indexed_ids.json` ghi tách rời, crash giữa 2 lần ghi thì lệch | Lưu theo thời gian (`--save-every`, mặc định 300s) và khi dừng (kể cả lỗi/Ctrl+C). Ghi nguyên tử. Trạng thái resume là chính `id_map` của index |
| `EMBEDDING_DIM = 1024` ghi cứng; `MAX_TOKENS`, `OVERLAP` khai báo nhưng không dùng | Lấy dim từ vector đầu tiên, kiểm tra khớp khi resume. Bỏ hằng số chết |
| `fp16` lỗi khi index < 200 vector | Số query kiểm tra = min(200, n) |

**Đã đối chiếu** trên 70 văn bản trong `data/processed` (chạy `parse_document` cũ lên cùng input): cùng tập 23.242 `chunk_id`, code cũ sinh thừa 257 bản trùng. 21.736 `embed_text` giống hệt (1.791 trong đó chỉ khác ở NFC/khoảng trắng). Trong phần khác: 1.215 chunk có thêm đoạn trước đây bị bỏ, 236 là id bị trùng ở code cũ, 55 là Điều có tiêu đề bị thay. Output đã được nạp thử bằng `SearchService` của repo cũ mà không phải sửa gì.

Còn lại (giữ như code cũ): đoạn dẫn của Điều/Khoản **không phải lá** (vd. `…bao gồm:` trước các Khoản) không vào chunk nào. Chunk vượt context model bị bỏ qua chứ không cắt nhỏ.

## Cấu trúc code

```
legal_rag/
├── config.py        endpoint, truy vấn danh sách, trạng thái hiệu lực được giữ
├── vbpl_client.py   list_page, get_detail; map mọi lỗi thành VbplError(reason)
├── storage.py       JsonlStore (append-only, khử trùng), Checkpoint (ghi nguyên tử)
├── crawl.py         crawl, retry_failed, reconcile
├── structure.py     HTML → cây Phần/Chương/Mục/Điều/Khoản/Điểm
├── preprocess.py    Phase 2
├── chunking.py      Phase 3: cây + HTML → chunk lá, ghi chunks.db
├── chunk_db.py      schema chunks.db (hợp đồng với SearchService), ghi nguyên tử, đọc tuần tự
├── embedding.py     Phase 4: EmbeddingClient (API kiểu OpenAI), run_embed (batch, resume)
├── vector_index.py  FAISS: tạo/đọc/ghi nguyên tử, nén fp16 + kiểm tra recall
└── __main__.py      CLI
tests/               test offline với client giả lập + 1 smoke test live
```
