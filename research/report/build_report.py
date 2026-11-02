from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os
FIG = os.path.join(os.path.dirname(__file__), "fig")
doc = Document()
for s in doc.sections: s.left_margin = s.right_margin = Cm(2.2); s.top_margin = s.bottom_margin = Cm(2.0)
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
def H(t, l=1): doc.add_heading(t, level=l)
def P(t, bold=False, italic=False):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold = bold; r.italic = italic; p.paragraph_format.space_after = Pt(6); return p
def B(items):
    for t in items:
        p = doc.add_paragraph(style="List Bullet"); p.add_run(t); p.paragraph_format.space_after = Pt(2)
def F(name, cap, w=16.0):
    doc.add_picture(os.path.join(FIG, name + ".png"), width=Cm(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph(); r = c.add_run(cap); r.italic = True; r.font.size = Pt(9.5); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
def T(rows, header, widths=None):
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Light Grid Accent 1"
    for i, h in enumerate(header): t.rows[0].cells[i].text = h
    for r in rows:
        cells = t.add_row().cells
        for i, x in enumerate(r): cells[i].text = str(x)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs: r.font.size = Pt(9)
    doc.add_paragraph()

tt = doc.add_heading("Báo cáo tiến độ nghiên cứu: từ SPADE đến câu hỏi nghiên cứu tiếp theo", 0)
P("Người viết: nghiên cứu sinh (Claude). Người nhận: thầy/anh. Ngày: 02/11/2026. Mục đích: trình bày đầy đủ đã làm gì, kết quả nào đáng tin và kết quả nào chưa, rồi cùng thảo luận hướng đi. Mọi con số trong báo cáo đều lấy từ file kết quả trong repo; chỗ nào tôi chưa tự kiểm chứng lại sẽ ghi rõ.", italic=True)

H("0. Tóm tắt điều hành (đọc trang này trước)")
B(["SPADE (bài đã đăng ở Array) học đồ thị nhân quả phi tuyến mà mỗi cạnh là một hàm spline vẽ ra được. Bài đã được nhận; chúng ta không sửa bài.",
   "Sau khi bài được nhận, tôi làm 3 đợt: (a) một đợt 'mở rộng' làm nhanh X1–X5, (b) tổng quan tài liệu dạng scoping review, (c) 5 pilot song song cho 5 khoảng trống tìm được trong tài liệu.",
   "Đợt (a) cho vài con số hứa hẹn nhưng KHÔNG đạt chuẩn là kết quả nghiên cứu: không có thiết kế bài toán trước, một họ bộ sinh dữ liệu, và sai số giữa các lần chạy (khoảng 0.04–0.06 AUROC) lớn hơn nhiều mức cải thiện tôi báo cáo (+0.014). Thầy đã nhận xét đúng ở chỗ này.",
   "Đợt (b) cho bản đồ tài liệu (khoảng 130 bài, đã xác minh qua web nhưng chưa đọc toàn văn) và 5 khoảng trống.",
   "Đợt (c) cho tín hiệu: KHÔNG pilot nào đứng vững thành một paper riêng. Có hai pilot (G1, G2) cùng chỉ vào một đối tượng, đường cong học được trên mỗi cạnh, và một pilot (G4) cho sân chơi đánh giá sạch.",
   "Phát hiện đáng chú ý nhất: group-lasso của SPADE làm biên độ đường cong co còn khoảng một nửa (hệ số 0.48 ở d=20); adaptive weighting khôi phục lại (0.95) mà gần như không hại việc chọn cạnh. Đây là hạt nhân khoa học thật nhất tôi tìm được, nhưng mới ở mức pilot.",
   "Đề xuất: gộp G1 + G2 và dùng G4 làm giao thức đánh giá thành một nghiên cứu về 'thiên lệch biên độ và độ trung thực hình dạng của hàm cạnh thưa'. Mục 6 trình bày các phương án và rủi ro; mục 7 là các câu hỏi cần thầy quyết định."])
F("f1_timeline", "Hình 1. Các giai đoạn công việc. Giai đoạn 3 (cam) làm nhanh, chưa đủ chặt; giai đoạn 4-5 là làm lại có hệ thống.")

H("1. Nền tảng: SPADE là gì và đã công bố được gì")
P("Bài toán: cho dữ liệu nhiều biến, tìm xem biến nào gây ra biến nào, và gây ra theo hình dạng nào (đường thẳng, bão hòa, ngưỡng, chu kỳ). SPADE giả định mô hình nhiễu cộng: mỗi biến kết quả bằng tổng các hàm một biến của các biến cha cộng nhiễu. Mỗi hàm là một spline (B-spline) học được, nên vẽ ra được. Hai thành phần kỹ thuật: (1) group-lasso trên hệ số của từng cạnh để cắt cạnh; (2) ràng buộc acyclicity kiểu NOTEARS để đồ thị không có vòng.")
F("f2_spade", "Hình 2. Sơ đồ SPADE: mỗi biến chỉ nhận thông tin qua các hàm một biến của biến cha (information bottleneck). Điểm này làm cho đồ thị đọc được và làm cấu trúc xác định được.")
P("Kết quả chính của bài đã đăng (đồ thị DAG tức thời phi tuyến, n=1500, d=6/10/20): AUROC 0.947 / 0.926 / 0.915. SPADE vượt hẳn các phương pháp gradient kiến trúc cố định (DAGMA, GraN-DAG, NOTEARS-MLP, GOLEM) và vượt SCORE ở mọi độ rộng. Với NoGAM thì kết quả hỗn hợp: SPADE thắng ở d=6 và 20, thua ở d=10 (0.926 so với 0.955). SPADE huấn luyện nhanh hơn NoGAM 275–930 lần. Trên chuỗi thời gian có độ trễ: hoàn hảo trên đồ thị tuyến tính (0.999), khoảng 0.93 trên phi tuyến; các phương pháp dựa trên kiểm định độc lập (PCMCI, VAR-LiNGAM) vẫn dẫn ở phi tuyến. Dự báo: MSE 0.029, không phải tốt nhất.")
F("f3_published", "Hình 3. Bài đã đăng: AUROC theo phương pháp và độ rộng d. SPADE (cột đầu tiên) đứng đầu hoặc gần đầu; NoGAM là đối thủ đáng kể duy nhất.")
P("Ở Revision 2 (theo nhận xét của Reviewer 1) chúng ta thêm dữ liệu thật phi tài chính, chấm theo kiến thức miền (BK-AUROC: mức tách được cạnh chắc chắn đúng khỏi cạnh bất khả thi). Kết quả: bike theo ngày, SPADE tốt nhất (0.810, DAGMA-nonlinear 0.703, chưa phân biệt được thống kê). Bike theo giờ và Beijing PM2.5 trên mẫu 2000 dòng, SPADE chỉ hòa với NOTEARS-linear; trên toàn chuỗi (17,303 và 41,543 dòng) SPADE dẫn (0.688 so với 0.622; 0.742 so với 0.658).")
F("f4_realworld", "Hình 4. BK-AUROC trên ba bộ dữ liệu thật (đường chấm: ngẫu nhiên 0.5). Trên mẫu nhỏ ở dữ liệu giờ SPADE không thắng rõ; lợi thế đến khi dùng toàn chuỗi.")

H("2. Đợt 'mở rộng' làm nhanh (X1–X5): làm gì, kết quả, và vì sao chưa phải kết quả")
P("Ngay sau khi bài được nhận, tôi chạy năm việc song song để 'sửa điểm yếu của SPADE'. Kết quả cụ thể:")
T([["X1", "Sparsity: adaptive group-lasso + trung bình 5 lần fit trên nửa mẫu (SPADE-AB)", "Trên seed mới (45 lần chạy) AUROC +0.014, p=0.004; khoảng cách với NoGAM ở d=10 co từ 0.063 còn 0.015"],
   ["X2", "Dữ liệu thật có ground truth: Sachs (11 protein, 18 cạnh)", "SPADE-AB AUROC 0.664, các phương pháp khác ≤0.57; nhưng chỉ trúng khoảng 6/18 cạnh"],
   ["X3", "Đo độ trung thực hình dạng (R² đường cong vs hàm thật)", "Spline thô gợn nên phân loại hình dạng đúng chỉ 12–25%; refit 2 giai đoạn có phạt độ trơn đúng 93–94%; MLP cùng dung lượng chỉ tìm 3–4 cạnh phi tuyến mạnh so với 15–16 của SPADE"],
   ["X4", "Quy mô: λ_g = min(0.01, 0.2/d) + sàng lọc cha ứng viên", "AUROC 0.96 / 0.98 / 0.98 ở d=50/100/200; λ cố định 0.01 sụp xuống 0.58–0.68 ở d≥100. Chỉ so được với baseline tuyến tính"],
   ["X5", "Dự báo nhanh hơn", "Dự đoán 'giá trị cuối' (naive) MSE 0.0177 tốt hơn mọi mô hình trong bảng của bài đã đăng; SPADE-fast bằng naive, chạy vài giây thay vì vài trăm giây"]],
  ["Mã", "Việc làm", "Kết quả"])
F("f5_extension_x1x4", "Hình 5. Trái/giữa: X1, chọn biến thể trên seed validation rồi kiểm trên seed mới. Phải: X4, λ cố định sụp ở d lớn; quy tắc theo d giữ AUROC cao (so với baseline tuyến tính duy nhất chạy được).")
F("f6_x2x3x5", "Hình 6. X2 (Sachs), X3 (hình dạng của cạnh phi tuyến mạnh), X5 (dự báo: naive vàng, thấp hơn mọi mô hình học).", w=16.5)
P("Vì sao tôi KHÔNG coi đây là kết quả nghiên cứu:", bold=True)
B(["Không có bài toán và giả thuyết được phát biểu trước. Tôi chạy 5 việc rồi báo cái nào đẹp.",
   "Một họ bộ sinh dữ liệu. Pilot G4 sau đó cho thấy bộ sinh này sortable mạnh (0.91–0.95), tức là một baseline chỉ sắp theo phương sai đã gần bằng SPADE.",
   "Sai số giữa các lần chạy. G4 đo: cùng dữ liệu, đổi số luồng hoặc float32/float64 làm AUROC lệch cỡ 0.04–0.06. Chính SPADE gốc của dự án cho d=10 là 0.926 ở môi trường cũ và 0.892 ở môi trường mới. Mức cải thiện +0.014 nằm dưới ngưỡng nhiễu đó nếu không so cặp qua nhiều bộ dữ liệu.",
   "Ở d≥100 chỉ so được với baseline tuyến tính (sai mô hình), nên 0.98 không nói SPADE hơn phương pháp phi tuyến tốt nhất.",
   "Tôi đã mắc lỗi code ở X3 (đọc nhầm cột basis), đã sửa trước khi lấy số; và bắt được một lần gán nhầm NoGAM với bài DAS trong tổng quan tài liệu. Nghĩa là quy trình lúc đó chưa đủ kiểm soát."])
P("Phát hiện phụ đáng ghi nhận, không ảnh hưởng bài đã đăng nhưng ảnh hưởng cách ta đánh giá về sau: baseline naive (last value) tốt hơn mọi mô hình học trong bảng dự báo; vì vậy tôi không đề xuất dự báo làm điểm bán của hướng tiếp theo.")

H("3. Scoping review: bản đồ tài liệu và khoảng trống")
P("Dạng review thầy chọn: scoping review (lập bản đồ khái niệm, bằng chứng, khoảng trống). Bốn nhánh tìm kiếm song song (differentiable DAG learning và sparsity; order-based/score-matching và quy mô; benchmark và dữ liệu thật; interpretability và chuỗi thời gian), mỗi bài được xác nhận qua trang arXiv/nhà xuất bản. Giới hạn: sàng lọc một lượt do AI, chưa có người thứ hai; phần lớn chỉ đọc abstract; nhiều bài 2025–2026 là preprint; một số danh sách tác giả và số trang chưa xác nhận. Mọi tham khảo phải kiểm lại bằng DOI/arXiv trước khi vào bản thảo.")
P("Bức tranh tổng quát:")
B(["Phương pháp học khả vi (NOTEARS, DAGMA, GraN-DAG, DCDI...) và phương pháp theo thứ tự/score-matching (CAM, SCORE, NoGAM, DAS, DiffAN) là hai dòng chính. Chạy ở d lớn đa số là tuyến tính, mạng thần kinh hộp đen hoặc amortized; loại cộng đọc được (CAM, hàm cơ sở) chỉ chạy ở d nhỏ.",
   "Interpretability phần lớn được khẳng định bằng đồ thị. Trừ Dagma-DCE (cường độ cạnh dựa trên đạo hàm), cường độ cạnh là chuẩn của trọng số. Không bài nào kiểm hình dạng hàm học được với ground truth bên trong causal discovery.",
   "Benchmark dễ bị đánh lừa: varsortability (Reisach 2021), R²-sortability (Reisach 2023), NOTEARS nhạy với thang đo (Kaiser & Sipos), cả Sachs có ground truth gây tranh cãi.",
   "Không tìm thấy bài nào ngoài SPADE kết hợp cạnh spline/KAN với ràng buộc acyclicity hoặc tìm kiếm theo thứ tự cho dữ liệu cắt ngang (KAN về nhân quả hiện có chỉ là Granger chuỗi thời gian, mô hình sinh với cấu trúc biết trước, hoặc bản đồ nhận thức mờ)."])
T([["G1", "Thiên lệch co rút/chọn cạnh của phạt thưa cho learner cộng phi tham số", "Có bàn cho NOTEARS tuyến tính và Granger; chưa có nghiên cứu về việc làm méo đường cong; stability selection chưa áp cho learner khả vi"],
   ["G2", "Đánh giá hàm cạnh với ground truth", "Chưa có thước đo; KAN fit biểu thức báo là không ổn định; concurvity mới nêu cho mô hình dự đoán"],
   ["G3", "Khám phá cộng đọc được ở quy mô lớn", "Pruning kiểu CAM ngốn thời gian; recall của DAS khoảng 0.08–0.09 ở d=1000; không có benchmark tĩnh d lớn có ground truth"],
   ["G4", "Hợp lệ của benchmark cho bộ sinh trơn bị chặn", "Chưa có kiểm toán seed/rò rỉ/phiên bản thư viện cho learner thần kinh hoặc spline"],
   ["G5", "Chuỗi thời gian phi dừng", "Đa số xử lý bằng giả định; KAN-Granger mới là preprint, quy mô nhỏ"]],
  ["Mã", "Khoảng trống", "Bằng chứng trong tài liệu"])
P("Tài liệu đầy đủ (kèm danh sách bài cần kiểm tra lại): research/01_literature_review.md trong repo.")

H("4. Năm pilot cho năm khoảng trống")
P("Mỗi pilot do một agent làm trong thư mục riêng, viết protocol trước khi chạy, tách seed validation khỏi seed test, giới hạn 1 luồng CPU. Đây là khảo sát khả thi, không phải kết quả: 10–30 đồ thị, một họ bộ sinh, 20–45 phút mỗi pilot. Số liệu ở dưới do tôi đối chiếu giữa các pilot, chưa chạy lại độc lập.")
H("4.1 G1: biên độ đường cong bị co (đáng chú ý nhất)", 2)
P("Câu hỏi: group-lasso trừ đi một lượng gần như không đổi trên mỗi nhóm hệ số đang hoạt động, vậy đường cong học được có bị 'dẹt' không? Đo bằng hệ số hiệu chỉnh (hồi quy đường cong học được lên đường cong thật; 1 là đúng, nhỏ hơn 1 là bị co) và R² đường cong. Bộ sinh dùng chuẩn hóa từng biến ngay khi sinh, nên không thể khai thác phương sai.")
F("f7_g1", "Hình 7. G1: trái, hệ số hiệu chỉnh biên độ. Group-lasso thường (base) giữ khoảng 0.67 (d=10) và 0.48 (d=20) biên độ thật; adaptive group-lasso đưa về 0.95–0.96. Giữa: R² đường cong tăng theo. Phải: AUPRC cạnh gần như không đổi với adaptive, nhưng refit và pipeline đầy đủ làm giảm.")
B(["Điều đứng vững: adaptive group-lasso sửa biên độ (R² đường cong +0.11 ở d=10, +0.29 ở d=20, tốt hơn ở 20/20 đồ thị); không hại xếp hạng cạnh nhiều (d=10: AUROC −0.005; d=20: +0.003).",
   "Điều không đứng vững: refit không phạt và stability selection làm AUPRC giảm 0.03–0.04; phạt độ trơn P-spline không giúp.",
   "Cảnh báo của chính agent: baseline AUROC đã ≈0.99 nên không còn chỗ để cải thiện việc chọn cạnh; cần thử n nhỏ, mật độ cao, nhiễu lớn."])
H("4.2 G2: thước đo độ trung thực hình dạng", 2)
P("Câu hỏi: nếu ta muốn nói 'đồ thị đọc được', cần một thước đo hình dạng hàm. Agent xây bộ thước (R² sau khi căn chỉnh affine, dấu đạo hàm, 'SF-F1' = cạnh phải vừa được tìm ra vừa đúng hình dạng, độ ổn định bootstrap) cùng bộ sinh có 5 loại cơ chế (tuyến tính, bão hòa, không đơn điệu, ngưỡng, chu kỳ) và một núm điều chỉnh mức phụ thuộc giữa các biến nguyên nhân (concurvity).")
F("f8_g2g3", "Hình 8. Trái: G2, GAM-ridge (hồi quy spline có phạt ridge, không ràng buộc DAG, không sparsity) ngang SPADE trên SF-F1 và hơn một chút AUROC. Phải: G3, SPOT với thứ tự ước lượng (cam) kém xa SPOT với thứ tự đúng (xanh lá).")
B(["Điều có giá trị: 'tìm ra cạnh' và 'đúng hình dạng' khác nhau khá nhiều (khoảng 0.2–0.3 recall); concurvity làm hỏng việc khám phá hơn là làm méo đường cong (AUROC của SPADE giảm 0.948→0.800 khi ρ tăng, nhưng R² đường cong có điều kiện chỉ giảm 0.975→0.943).",
   "Điều đáng lo: xếp hạng phương pháp gần như không đổi so với AUROC; và một baseline cộng đơn giản không cần DAG đã ngang SPADE. Điều này là mối đe dọa trực tiếp cho mọi tuyên bố 'SPADE đọc được hơn'. Giới hạn: GAM-ridge chạy ở d=8, không có ràng buộc DAG nên có thể không còn đúng khi đồ thị lớn hoặc đặc.",
   "Thước phân loại hình dạng (class agreement) trong pilot này và trong X3 của tôi đều nhạy với các gợn ở đuôi; cần bỏ hoặc kiểm chứng."])
H("4.3 G3: khám phá cộng ở d lớn (không đứng vững)", 2)
P("Thuật toán SPOT: ước lượng thứ tự bằng score-matching (Stein), sàng lọc cha ứng viên là các biến đứng trước, rồi fit spline cộng có group-lasso (giải chính xác bằng FISTA, không cần ràng buộc acyclicity). Kết quả: nếu cho thứ tự ĐÚNG, phần sàng lọc và fit spline rất tốt (AUROC 0.93–1.00 tới d=200, 1–4 phút một lần fit). Với thứ tự ước lượng, chỉ khớp 49–64% cạnh (50% là ngẫu nhiên), nên SPOT thua cả cách cũ không chốt thứ tự (SPADE sàng lọc ở X4). Cảnh báo: bộ sinh 'std' của agent này không thật sự chuẩn hóa (sortable 0.76–0.88), mỗi ô chỉ 1 seed test, và các baseline phi tuyến không chạy nổi ở kích thước này.")
H("4.4 G4: kiểm toán benchmark", 2)
F("f9_g4g5", "Hình 9. Trái và giữa: G4. Trên bộ sinh của dự án, baseline chỉ sắp theo phương sai (xám) đã ≈ NOTEARS và gần SPADE (thậm chí hơn SPADE ở d=50). Trên bộ sinh chuẩn hóa từng biến, baseline này sụp về ≈ngẫu nhiên và lợi thế của SPADE hiện rõ. Phải: G5, trên dữ liệu mức (xám) SPADE hỏng khi chuỗi là I(1).")
B(["Bộ sinh của dự án sortable mạnh (0.91–0.95 ở d=6–50). Một baseline sắp theo phương sai rồi hồi quy có phạt đạt AUROC 0.85–0.90, hơn NOTEARS-linear, DAGMA-linear, GOLEM, NOTEARS-MLP ở mọi d, và chỉ kém SPADE 0.04–0.07 (hơn SPADE ở d=50).",
   "Trên bộ sinh chuẩn hóa từng biến, SPADE vẫn hơn NOTEARS-linear từ +0.06 đến +0.32. Tức là 'SPADE tốt' vẫn đúng, nhưng benchmark cũ làm yếu bằng chứng.",
   "Ngưỡng nhiễu thực hiện: mỗi lần chạy trên một bộ dữ liệu có sd khoảng 0.04–0.06 AUROC chỉ do cài đặt (số luồng, float32/float64). Chênh lệch 0.926 so với 0.892 của SPADE giữa hai môi trường nằm trong ngưỡng đó. Một cải thiện nhỏ hơn 0.03–0.05 chỉ đáng tin nếu so cặp qua tối thiểu 30 bộ dữ liệu có lặp seed phương pháp.",
   "Agent tự bắt được một lỗi rò rỉ của chính nó: lần đầu, đối chứng 'thứ tự ngẫu nhiên' vô tình dùng đúng thứ tự thật vì chia sẻ dòng RNG với bộ sinh. Đây đúng là loại rò rỉ mà danh sách kiểm tra cảnh báo."])
H("4.5 G5: chuỗi thời gian phi dừng", 2)
P("Câu hỏi: với chuỗi tích phân (I(1)), spline học trên dữ liệu mức có đọc xu hướng chung thành cạnh không? Kết quả: có; AUPRC của SPADE trên mức thấp hơn trên sai phân 0.35–0.52 (rất chắc). Công thức ECM-SPADE (thêm số hạng mức và kiểm placebo) chỉ hơn sai phân thuần +0.022 AUPRC ở một kịch bản (t≈2.6, không hiệu chỉnh đa phép thử), và bài toán quá dễ (nhiều phương pháp đạt ≈1.0) nên chưa phân biệt được. Các baseline tuyến tính VAR-Lasso, PCMCI trên mức vẫn tốt vì cơ chế bị chặn. Giá trị: khớp với việc đồ thị tài chính trên giá mức trong bài chỉ phản ánh xu hướng chung; nhưng đây là điểm yếu của SPADE đã biết, chưa phải thuật toán mới.")
F("f10_decision", "Hình 10. Bảng quyết định: năm pilot và kết luận.", w=16.5)

H("5. Tổng hợp: biết gì, chưa biết gì, rủi ro")
T([["SPADE gốc làm co biên độ đường cong; adaptive weighting sửa", "G1 (20/20 đồ thị, hai d), một họ bộ sinh", "Tương đối chắc ở mức pilot"],
   ["Benchmark cũ của dự án dễ bị khai thác bằng phương sai", "G4, đo trực tiếp", "Chắc"],
   ["Ngưỡng nhiễu cài đặt ≈0.04–0.06 AUROC mỗi lần chạy", "G4, 30 seed ở d=10, 5 bộ cho phần luồng", "Khá chắc"],
   ["Spline đọc được tốt hơn MLP cạnh trên cạnh phi tuyến mạnh", "X3 và G2 (theo MLP-edge)", "Tương đối, nhưng baseline GAM-ridge còn nguyên"],
   ["SPADE hơn baseline cộng cổ điển (CAM, GAM)", "Chưa kiểm; G2 gợi ý ngang nhau", "Chưa biết"],
   ["Khám phá cộng đọc được ở d≥100 tốt hơn phương pháp phi tuyến khác", "Chưa kiểm (baseline không chạy nổi)", "Chưa biết"],
   ["Công thức xử lý chuỗi I(1) có giá trị thực", "G5, +0.022, t≈2.6", "Chưa chứng minh"]],
  ["Phát biểu", "Bằng chứng", "Độ tin cậy"])
P("Những rủi ro chung cần nói thẳng:", bold=True)
B(["Mọi pilot do agent chạy, nhỏ và nhanh. Cần chạy lại độc lập trước khi tin vào bất cứ phép so sánh nào.",
   "Cùng một nhóm tác giả xây benchmark và phương pháp (rủi ro thiên vị); cần baseline thật: CAM, NAM, DAGMA với đạo hàm (DCE), KAN, GAM backfitting.",
   "Tuyên bố 'không tìm thấy trong tổng quan' chỉ có nghĩa là không thấy trong lần quét này, không phải là bằng chứng về tính mới.",
   "Bài về benchmark và kiểm toán sẽ bị coi là yếu nếu đứng một mình (Reisach và cộng sự đã làm phần lớn)."])

H("6. Các phương án cho bước tiếp theo")
T([["A (đề xuất)", "Gộp G1 + G2, dùng G4 làm giao thức đánh giá: đo và sửa thiên lệch biên độ/hình dạng của hàm cạnh thưa trong causal discovery cộng phi tuyến", "Có hạt nhân rõ (hệ số co 0.48 → 0.95); thước đo mới có người dùng; sân chơi sạch", "Phải thắng hoặc ngang baseline cộng thật (GAM-ridge/CAM); cải thiện chọn cạnh gần như bằng 0; cần lý thuyết cho ước lượng group-adaptive; hiệu ứng có thể chỉ nằm ở d lớn"],
   ["B", "Chỉ làm bài phương pháp luận đánh giá (G4 mở rộng)", "Rẻ, chắc về dữ liệu", "Bị coi là mỏng, ý tưởng đã có"],
   ["C", "Quay lại G3 với bộ ước lượng thứ tự tốt hơn (DiffAN hoặc SCORE chính thức) hoặc thứ tự mềm", "Nếu thành công thì tác động lớn (d hàng trăm, đọc được)", "Rủi ro cao: bước thứ tự là điểm hỏng; baseline phi tuyến tốn thời gian"],
   ["D", "G5 trong chế độ khó (tín hiệu yếu, T ngắn, d lớn, đồng tích hợp phi tuyến)", "Giải thích được kết quả tài chính của bài", "Mới là sửa điểm yếu SPADE; chưa có thuật toán mới"],
   ["E", "Dừng hướng 'sửa SPADE', viết bài khác dựa trên SPADE (ứng dụng quản trị)", "Ít rủi ro khoa học", "Có thể trùng bài đã đăng"]],
  ["Phương án", "Nội dung", "Ưu điểm", "Rủi ro"])
P("Nếu theo A, phác thảo bài toán để thảo luận (sẽ chặt chẽ hóa ở bước 2):", bold=True)
B(["Mô hình: x_e = Σ_j f_je(x_j) + ε, f_je thuộc lớp hàm trơn; mục tiêu kép: (i) khôi phục tập cạnh, (ii) khôi phục hình dạng hàm f_je (không chỉ có/không).",
   "Câu hỏi chính: phạt thưa kiểu group-lasso gây sai số biên độ ở mức nào, phụ thuộc vào n, d, độ mạnh cạnh, concurvity ra sao; và có ước lượng nào sửa mà không đánh đổi chọn cạnh?",
   "Ứng viên thuật toán: adaptive group-lasso (G1 cho thấy sửa biên độ), rồi bước làm trơn chỉ dùng để hiển thị (X3 cho thấy làm trơn không đổi cấu trúc), chọn tập cạnh bằng tiêu chí thông tin hoặc kiểm chéo thay vì top-2d cố định (G1 nghi ngờ chính chỗ này làm hỏng refit).",
   "Thước đo: hệ số hiệu chỉnh biên độ, R² hình dạng sau căn chỉnh, SF-F1, dấu đạo hàm; không dùng phân loại hình dạng 'class agreement' cho đến khi kiểm chứng.",
   "Giao thức: bộ sinh chuẩn hóa từng biến; baseline sortnregress; baseline cộng thật (CAM/GAM, NAM, DAGMA-DCE); ≥30 bộ dữ liệu mỗi ô; so cặp; báo cáo ngưỡng nhiễu cài đặt; seed validation tách khỏi seed test; ghi phiên bản thư viện và dtype."])

H("7. Câu hỏi để thảo luận")
B(["Thầy có đồng ý đổi hướng từ 'sửa SPADE' sang câu hỏi khoa học 'đo và sửa thiên lệch biên độ/hình dạng của hàm cạnh thưa' (phương án A) không? Hay thầy thấy phương án khác đáng hơn?",
   "Đóng góp mong muốn là thuật toán mới, hay thước đo/benchmark mới, hay phân tích lý thuyết? Mỗi loại đòi hỏi chuẩn chứng minh khác nhau.",
   "Baseline cộng cổ điển (GAM-ridge, CAM) ngang SPADE ở pilot G2. Nếu đúng ở thí nghiệm đầy đủ, ta kể câu chuyện nào? ('SPADE không cần thiết cho interpretability' là kết quả âm nhưng thật.)",
   "Thầy muốn nhắm nơi nào (tạp chí, hội nghị)? Điều đó quyết định mức lý thuyết và số thí nghiệm.",
   "Ngân sách tính toán: pilot dùng 1 luồng CPU; thí nghiệm đầy đủ cho A cần cỡ vài ngày CPU hoặc vài giờ GPU. Có dùng được GPU (Colab) không?",
   "Về tác giả và việc có nhắc bài SPADE đã đăng: bài mới xây trên SPADE, cần xác định tác giả và cách trích dẫn.",
   "Dự báo tạm gác hẳn (naive thắng) hay còn muốn một nhánh riêng?"])

H("Phụ lục A. Nơi lưu mọi thứ trong repo (nhánh main)")
B(["research/01_literature_review.md: scoping review.", "research/02_pilot_synthesis.md: tổng hợp pilot.", "research/pilots/G1…G5: mã, protocol, kết quả thô của từng pilot.",
   "experimental_results/x1…x5, sachs_raw.csv: kết quả đợt 'mở rộng' làm nhanh.", "extension/: bản thảo khung đã được thầy nhận xét là chưa phải paper (giữ lại làm tham khảo, không dùng).",
   "research/report/: báo cáo này, mã vẽ hình make_figures.py (mọi hình vẽ từ file kết quả)."])
H("Phụ lục B. Những lỗi quy trình của tôi và cách xử lý")
B(["Gán nhầm NoGAM với bài DAS trong tổng quan tài liệu: bắt được khi đối chiếu hai nhánh tìm kiếm; đã dùng đúng arXiv:2304.03265 cho NoGAM.",
   "Lỗi code X3 (phi = b @ coef lấy cả d cột thay vì một cột): làm R² phình lên; sửa trước khi lấy số.",
   "Metric phân loại hình dạng quá nhạy với gợn: kết quả v1 được lưu riêng (x3_fidelity_v1_raw.csv) để minh bạch.",
   "Chạy nhiều job cạnh tranh CPU làm thời gian đo bị méo; các bảng giờ đã bỏ cột thời gian.",
   "Một lần lệnh pkill làm tắt luôn shell của tôi và hai lần job nền chết khi lượt làm việc kết thúc: đã chuyển sang chờ trong lượt và có chế độ resume."])
doc.save(os.path.join(os.path.dirname(__file__), "bao_cao_tien_do.docx")); print("saved")
