# PP2 — Kiểm toán benchmark năng lượng nhà máy xử lý nước thải ETP Melbourne

Bộ hồ sơ này là kết quả thực hiện file `MASTER_PROMPT_WATER_TREATMENT.docx` ngày 12/11/2026: rà soát tài liệu, chọn đề tài, tải và kiểm tra dữ liệu thật, chạy phân tích Python, viết bản thảo bài báo tiếng Anh và các báo cáo kèm theo.

**Đề tài được chọn:** *Auditing a public benchmark for wastewater treatment plant energy prediction: misdated flows, an incomplete target and validation leakage in the Melbourne Eastern Treatment Plant dataset.*

## Kết quả chính (đều tính từ dữ liệu thật)

1. Cột mục tiêu của benchmark chỉ là **điện mua lưới**; điện tự phát từ biogas (29,9 % tổng cấp điện) không có trong file.
2. Hai cột lưu lượng của benchmark là số liệu **2008–2012 gán vào ngày 2014–2019** theo thứ tự dòng. Một quy tắc đơn giản (trung bình các bản ghi giờ của ngày UTC thứ i, bỏ bản ghi 00:00 UTC) tái tạo chính xác 1.382/1.382 giá trị lưu lượng vào và 1.381/1.382 giá trị lưu lượng ra.
3. 90 % giá trị TN và 30 % giá trị BOD là giá trị **điền**, không phải đo.
4. Với file nguyên trạng, **không họ mô hình nào trong năm họ vượt persistence** (RMSE 43,5–44,6 so với 40,7 MWh/ngày).
5. Chỉ sửa ngày của hai cột lưu lượng: RMSE giảm còn 27,8. **Giải thích cùng ngày** (có lưu lượng cùng ngày): 17,8–19,0 MWh/ngày cho tổng điện cấp. **Dự báo trước một ngày** (chỉ dùng thông tin có trước): 31–32 MWh/ngày, chỉ tốt hơn persistence 15–17 %.
6. Lưu lượng xả nước sau xử lý chi phối điện năng: E = 238 + 0,43·Q (MWh/ngày, Q: ML/ngày) cho kết quả ngang gradient boosting.
7. Nội suy spline ×10 trước khi chia tập tạo ra MAPE 1,35 % — đúng bằng mức đã công bố (1,36 %) — và chỉ cần chép dòng trước cũng đạt 1,29 %.
8. Dải Gauss từ phần dư trong mẫu chỉ phủ 78–86 % ở mức danh định 90 %; khoảng hiệu chuẩn trên dữ liệu giữ riêng không phủ thiếu. Với dự báo trước một ngày, khoảng 90 % rộng khoảng ±12 %.
9. Kiểm tra giả định không bán điện lên lưới: điện lưới chưa bao giờ dưới 117 MWh/ngày; thay đổi điện tự phát được bù 0,84 MWh điện lưới cho mỗi MWh (KTC 95 %: 0,77–0,91), tức gần nhưng chưa đúng một-đổi-một.

## Cấu trúc thư mục

| Thư mục / file | Nội dung |
|---|---|
| `reports/01_…docx` | Rà soát nghiên cứu và 10 đề tài tiềm năng |
| `reports/02_…docx` | Kiểm tra trùng lặp, Data Suitability and Novelty Matrix, nguồn dữ liệu |
| `reports/03_…docx` | Research proposal |
| `reports/04_Manuscript_EN.docx` | Bản thảo bài báo tiếng Anh (IMRaD), bản nháp v2; bản sao công khai nằm trong `manuscript/` |
| `reports/05_…docx` | Danh mục tài liệu tham khảo và trạng thái xác minh từng mục |
| `reports/06_…docx` | Đánh giá khả năng công bố và các việc còn thiếu |
| `reports/md/` | Bản Markdown nguồn của sáu tài liệu trên |
| `code/` | Mã Python (`run_all.py` chạy toàn bộ), `requirements.txt` |
| `data/raw/` | 6 file dữ liệu gốc và `SHA256SUMS.txt` |
| `data/processed/` | Bảng dựng lại theo ngày, bảng mô hình, dự báo trên tập test |
| `results/tables/` | 24 bảng kết quả CSV (T0–T17) |
| `results/figures/` | 7 hình, mỗi hình có PNG (300 dpi) và PDF |
| `README.md` | Hướng dẫn tiếng Anh cho reviewer |
| `LICENSE`, `CITATION.cff`, `data/README.md` | Giấy phép mã, cách trích dẫn, nguồn và giấy phép dữ liệu |

## Chạy lại

```
pip install -r code/requirements.txt
cd code
python run_all.py
```

Python 3.13; numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2. Thời gian chạy khoảng 7 phút trên 2 CPU. Bước 1 kiểm tra SHA-256 của dữ liệu trước khi tính.

Dữ liệu trong `data/raw/` đã có sẵn nên không cần mạng. Nguồn gốc:

- `Data-Melbourne_F.csv`: Mendeley Data, https://doi.org/10.17632/pprkvz3vbd.1 (CC BY 4.0).
- Năm file `MWC_ETP_*`: Melbourne Water Open Data Hub, https://data-melbournewater.opendata.arcgis.com , tải toàn bộ bản ghi qua REST ngày 12/11/2026. Hai file lưu lượng theo giờ được lưu ở dạng gọn hai cột (`start_epoch_hour` = số giờ kể từ 1970-01-01 UTC, và lưu lượng kL/s).

## Đẩy lên GitHub

Repo đã được tạo sẵn ở chế độ **riêng tư**: https://github.com/phamnam123lite-svg/etp-energy-benchmark-audit

Cách 1 (kéo thả, không cần Git): mở trang `https://github.com/phamnam123lite-svg/etp-energy-benchmark-audit/upload`, rồi trong `D:\pp2` chọn các mục `code`, `data`, `results`, `manuscript`, `.gitignore`, `CITATION.cff`, `LICENSE`, `README.md`, `README_VI.md`, kéo thả vào trang và bấm **Commit changes**. Không kéo thư mục `reports` (tài liệu làm việc nội bộ) và hai file `push_to_github.bat`, `COMMIT_MESSAGE.txt`.

Cách 2 (cần Git for Windows): bấm đúp `push_to_github.bat`. File `.gitignore` tự loại thư mục `reports`.

Khi sẵn sàng cho reviewer xem: vào Settings của repo, mục Danger Zone, chọn Change visibility, rồi Public.

## Những điều cần biết trước khi dùng

- Bản thảo là **bản nháp chưa qua thẩm định**; chưa liên hệ tác giả bộ dữ liệu gốc. Xem Báo cáo 6, mục 5.
- Ba mục tài liệu tham khảo (hai bài NeurIPS, một bài JMLR) chưa kiểm tra trực tuyến được (Báo cáo 5).
- Mô hình deep learning đã chạy là mạng truyền thẳng 3 lớp ẩn (MLP của scikit-learn) và boosting là GBM cổ điển; LSTM/GRU, XGBoost, LightGBM, SHAP không có trong môi trường và không cài thêm được.
- Tra trích dẫn ngược làm trên OpenAlex (232 công bố, lọc theo tiêu đề), không phải Scopus/WoS vì không truy cập được.
- Tuyên bố sử dụng AI, CRediT, nguồn tài trợ và xung đột lợi ích trong bản thảo cần tác giả tự rà soát và hoàn tất.
- Giấy phép: CC BY 4.0 cho file Mendeley và cho các bảng điện, lưu lượng, chất lượng nước vào của Melbourne Water (theo DataVic); bảng chất lượng nước ra chưa xem được giấy phép.
