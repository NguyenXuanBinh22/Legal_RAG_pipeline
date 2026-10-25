-- Kiểm tra chunk_id trùng trong chunks.db (output của pipeline/optimize_data.py, repo cũ).
-- Chạy (chỉ đọc):
--   sqlite3 -readonly G:/Projects/R2AI2026-BUILD-AI-LEGAL-ASSISTANT/data/chunks.db < sql/check_duplicate_chunk_ids.sql
-- Mỗi câu mất ~1-5 giây trên 821.787 dòng (không có index trên chunk_id).

.headers on
.mode box

-- 1. Tổng quan: bao nhiêu chunk_id bị trùng, bao nhiêu dòng thừa
--    (kết quả hiện tại: 17.927 id | 39.261 dòng | 21.334 dòng thừa)
SELECT COUNT(*)   AS so_chunk_id_trung,
       SUM(n)     AS tong_dong,
       SUM(n - 1) AS dong_thua
FROM (SELECT chunk_id, COUNT(*) AS n
      FROM chunks GROUP BY chunk_id HAVING COUNT(*) > 1);

-- 2. Phân bố số lần một chunk_id xuất hiện
SELECT n AS so_lan_xuat_hien, COUNT(*) AS so_chunk_id
FROM (SELECT chunk_id, COUNT(*) AS n
      FROM chunks GROUP BY chunk_id HAVING COUNT(*) > 1)
GROUP BY n ORDER BY n;

-- 3. Bản trùng có cùng văn bản và cùng nội dung không?
--    giong_het = số id mà mọi bản sao cùng doc_id và cùng content
SELECT SUM(so_doc = 1 AND so_noi_dung = 1) AS giong_het,
       SUM(so_doc > 1)                     AS khac_van_ban,
       SUM(so_doc = 1 AND so_noi_dung > 1) AS khac_noi_dung
FROM (SELECT chunk_id,
             COUNT(DISTINCT doc_id)  AS so_doc,
             COUNT(DISTINCT content) AS so_noi_dung
      FROM chunks GROUP BY chunk_id HAVING COUNT(*) > 1);

-- 4. Vị trí các bản sao: liền kề nhau, hay xen kẽ với chunk khác?
--    Liền kề = Phase 3 duyệt từng <p> có cùng id (dòng nối tiếp data-merged="true")
SELECT CASE WHEN max_id - min_id = n - 1 THEN 'liền kề'
            ELSE 'xen kẽ' END AS vi_tri,
       COUNT(*)              AS so_chunk_id,
       SUM(n - 1)            AS dong_thua,
       MAX(max_id - min_id)  AS khoang_cach_lon_nhat
FROM (SELECT chunk_id, COUNT(*) AS n, MIN(faiss_id) AS min_id, MAX(faiss_id) AS max_id
      FROM chunks GROUP BY chunk_id HAVING COUNT(*) > 1)
GROUP BY vi_tri;

-- 5. Theo văn bản: lặp một phần hay mọi chunk đều bị nhân đôi
SELECT CASE WHEN n = 2 * d THEN 'mọi chunk x2'
            WHEN n > d     THEN 'lặp một phần'
            ELSE 'không lặp' END AS loai,
       COUNT(*)       AS so_van_ban,
       SUM(n - d)     AS dong_thua,
       ROUND(AVG(d), 1) AS tb_chunk_moi_van_ban
FROM (SELECT doc_id, COUNT(*) AS n, COUNT(DISTINCT chunk_id) AS d
      FROM chunks GROUP BY doc_id)
GROUP BY loai ORDER BY dong_thua DESC;

-- 6. Có văn bản nào bị ghép 2 lần không? (bản sao của văn bản nằm ở 2 khối faiss_id tách rời)
--    lien_mach = số văn bản có faiss_id liên tục; 0 ở cột bi_tach = không có văn bản bị crawl/ghép lặp
SELECT SUM(hi - lo + 1 = n)  AS lien_mach,
       SUM(hi - lo + 1 <> n) AS bi_tach
FROM (SELECT doc_id, COUNT(*) AS n, MIN(faiss_id) AS lo, MAX(faiss_id) AS hi
      FROM chunks GROUP BY doc_id);

-- 7. Văn bản có nhiều dòng thừa nhất
SELECT c.doc_id,
       d.doc_num,
       COUNT(*) - COUNT(DISTINCT c.chunk_id) AS dong_thua,
       COUNT(*)                              AS tong_chunk,
       substr(d.title, 1, 60)                AS title
FROM chunks c JOIN docs d ON d.doc_id = c.doc_id
GROUP BY c.doc_id
HAVING dong_thua > 0
ORDER BY dong_thua DESC
LIMIT 10;

-- 8. Bản trùng thuộc cấp nào (Điều / Khoản / Điểm)
SELECT CASE WHEN content LIKE '% - Điểm %' THEN 'Điểm'
            WHEN clause IS NOT NULL       THEN 'Khoản'
            WHEN article IS NOT NULL      THEN 'Điều (không Khoản)'
            ELSE 'không có Điều' END AS cap,
       COUNT(*) AS so_dong
FROM chunks
WHERE chunk_id IN (SELECT chunk_id FROM chunks GROUP BY chunk_id HAVING COUNT(*) > 1)
GROUP BY cap ORDER BY so_dong DESC;

-- 9. Xem mẫu: một id trùng liền kề và một id trùng xen kẽ
SELECT faiss_id, chunk_id, doc_num, article, clause, substr(content, 1, 80) AS content
FROM chunks
WHERE chunk_id = (SELECT chunk_id FROM chunks GROUP BY chunk_id
                  HAVING COUNT(*) = 2 AND MAX(faiss_id) - MIN(faiss_id) = 1 LIMIT 1)
ORDER BY faiss_id;

SELECT faiss_id, chunk_id, doc_num, article, clause, substr(content, 1, 80) AS content
FROM chunks
WHERE chunk_id = (SELECT chunk_id FROM chunks GROUP BY chunk_id
                  HAVING MAX(faiss_id) - MIN(faiss_id) <> COUNT(*) - 1 LIMIT 1)
ORDER BY faiss_id;

-- 10. Xem mọi bản sao của MỘT chunk_id cụ thể (thay id cần xem)
SELECT faiss_id, doc_num, article, clause, length(content) AS do_dai, substr(content, 1, 120) AS content
FROM chunks
WHERE chunk_id = '000cd724-df4e-11f0-86bd-1bc61932c0df'
ORDER BY faiss_id;
