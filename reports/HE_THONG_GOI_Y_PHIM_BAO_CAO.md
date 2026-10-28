# BÁO CÁO KỸ THUẬT TOÀN DIỆN VỀ HỆ THỐNG GỢI Ý PHIM
## MOVIE RECOMMENDATION SYSTEM — DATA MINING & MACHINE LEARNING PROJECT

---

## 1. TỔNG QUAN DỰ ÁN VÀ BÀI TOÁN

Trong kỷ nguyên bùng nổ dữ liệu số, các nền tảng truyền thông đa phương tiện trực tuyến (như Netflix, YouTube, Spotify) phải đối mặt với bài toán quá tải thông tin (*information overload*). Người dùng đứng trước hàng chục ngàn bộ phim nhưng chỉ có thời gian xem một số lượng rất hạn chế. 

Dự án này giải quyết bài toán **Hệ thống Gợi ý Phim cá nhân hóa (Personalized Movie Recommendation System)** trên tập dữ liệu quy mô lớn **The Movies Dataset** (kết hợp giữa **MovieLens 26M** và **TMDB Metadata**).

### Mục tiêu cốt lõi:
1. Xây dựng một quy trình khai phá dữ liệu (*Data Mining Pipeline*) chuẩn mực, chịu tải cao trên 26 triệu dòng dữ liệu.
2. Thiết kế và đối sánh đa dạng các trường phái gợi ý:
   - **Popularity-based** (Phi cá nhân hóa / Baseline)
   - **Collaborative Filtering** (Lọc cộng tác dựa trên phân rã ma trận tiềm ẩn SVD)
   - **Content-Based Filtering** (Lọc dựa trên nội dung truyền thống qua TF-IDF)
   - **Deep Semantic Embeddings** (Nhúng ngữ nghĩa sâu qua mô hình ngôn ngữ Sentence-Transformers)
   - **Hybrid Systems** (Mô hình lai kết hợp trọng số động *Adaptive Alpha*)
   - **Two-Stage Recommender Architecture** (Kiến trúc công nghiệp 2 giai đoạn: Thu hồi ứng viên *Candidate Retrieval* và Tái xếp hạng *Re-ranking* với cây tăng cường gradient *LightGBM LambdaRank* cùng giải thuật đa dạng hóa danh mục *MMR*).
3. Đánh giá khách quan, khoa học bằng giao thức chuẩn công nghiệp (*Negative Sampling Ranking Evaluation*) với các chỉ số **Hit Rate@10** và **NDCG@10**.

---

## 2. BẢN CHẤT DỮ LIỆU & ĐẶC TÍNH THỐNG KÊ (EXPLORATORY DATA ANALYSIS)

Dự án tích hợp dữ liệu từ 5 tệp nguồn chính:
- `ratings.csv`: Chứa 26,024,289 lượt đánh giá từ 270,896 người dùng trên 45,115 bộ phim. Thang điểm từ $0.5$ đến $5.0$ kèm mốc thời gian (*timestamp*).
- `movies_metadata.csv`: Chứa thông tin 45,466 bộ phim (tiêu đề, tóm tắt nội dung *overview*, ngân sách, doanh thu, thời lượng, điểm bình chọn,...).
- `keywords.csv`: Các từ khóa chủ đề dưới định dạng JSON-string gán cho từng phim.
- `credits.csv`: Danh sách diễn viên (*cast*) và đội ngũ sản xuất (*crew* bao gồm đạo diễn).
- `links.csv`: Bảng ánh xạ định danh giữa mã số nội bộ của MovieLens (`movieId`) và The Movie Database (`tmdbId`).

### Các đặc tính dữ liệu quan trọng phát hiện qua EDA:
1. **Độ thưa cực lớn (Extreme Sparsity):**
   $$\text{Sparsity} = 1 - \frac{|R|}{|U| \times |I|} = 1 - \frac{26,024,289}{270,896 \times 45,115} \approx 99.787\%$$
   Chỉ có $0.213\%$ số ô trong ma trận Người dùng - Phim có giá trị quan sát. Điều này gây khó khăn nghiêm trọng cho các thuật toán CF láng giềng truyền thống (k-NN).
2. **Hiện tượng đuôi dài (Long-tail Phenomenon):**
   - $39.9\%$ số phim ($17,988$ phim) có dưới 5 lượt đánh giá.
   - Ngược lại, một số ít phim bom tấn nhận tới hơn 90,000 lượt đánh giá (phim cao nhất đạt 91,921 ratings).
3. **Hiện tượng khởi đầu lạnh (Cold-start Users):**
   - $5.5\%$ người dùng ($14,789$ người) chỉ có dưới 5 lượt tương tác. Đối với nhóm này, thông tin hành vi quá nghèo nàn để mô hình CF học được biểu diễn tiềm ẩn.
4. **Thiên lệch điểm đánh giá (Rating Bias):**
   - Điểm trung bình toàn hệ thống là $\mu = 3.528$ (độ lệch chuẩn $\sigma = 1.065$).
   - $61.9\%$ số đánh giá rơi vào ngưỡng $\ge 3.5$. Do đó, ngưỡng $3.5$ được chọn làm ranh giới xác định một bộ phim là "tích cực / phù hợp" (*relevant*).

---

## 3. LÝ THUYẾT & QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)

### 3.1. Xử lý dữ liệu lớn theo Chunk và định dạng Parquet
Tệp `ratings.csv` có dung lượng thô trên đĩa lớn (~700 MB) và khi nạp đầy đủ vào RAM bằng Pandas dạng văn bản có thể tốn từ 2-3 GB RAM.
- **Kỹ thuật xử lý:** Triển khai nạp theo khối (`chunksize = 1,000,000` dòng/lần).
- **Làm sạch cục bộ & toàn cục:**
  - Loại bỏ các dòng khuyết thiếu `NaN` ở các trường `userId`, `movieId`, `rating`.
  - Kiểm tra tính hợp lệ của thang điểm: loại bỏ ratings $< 0.5$ hoặc $> 5.0$.
  - Loại bỏ các bản ghi trùng lặp (`drop_duplicates` trên cặp khóa `(userId, movieId)`).
  - Ép kiểu dữ liệu tối ưu: `userId` $\to$ `int32`, `movieId` $\to$ `int32`, `rating` $\to$ `float32`.
- **Chuyển đổi lưu trữ sang Apache Parquet:** Parquet sử dụng kỹ thuật lưu trữ theo cột (*columnar storage*) và thuật toán nén Snappy, giúp giảm kích thước file trên đĩa và tăng tốc độ đọc dữ liệu lên **gấp 5-10 lần** so với CSV.

### 3.2. Hợp nhất Metadata & Phân tích chuỗi JSON AST
Metadata của phim trong TMDB chứa các trường lồng ghép dạng JSON (như `[{"id": 28, "name": "Action"}]`).
- Dùng `ast.literal_eval` để trích xuất danh sách tên thể loại (*genres*), từ khóa (*keywords*), top 3 diễn viên chính và đạo diễn (*director*).
- Tạo chuỗi văn bản tổng hợp (*Metadata Soup*):
  $$\text{Soup} = \text{Clean}(\text{Title}) \oplus \text{Clean}(\text{Genres}) \oplus \text{Clean}(\text{Keywords}) \oplus \text{Clean}(\text{Cast}) \oplus \text{Clean}(\text{Director}) \oplus \text{Clean}(\text{Overview})$$
- Ánh xạ an toàn thông qua `links.csv` để chuẩn hóa toàn bộ phim về khóa duy nhất `movieId`.

### 3.3. Kỹ thuật lọc K-Core Graph (K-Core Filtering)
Trong đồ thị hai phía Người dùng - Phim (*Bipartite User-Item Graph*), các đỉnh bậc quá thấp (phim quá ít lượt xem hoặc người dùng chỉ đánh giá 1-2 lần) đóng vai trò như "nhiễu" (*noise*), làm ma trận thưa và suy giảm chất lượng biểu diễn tiềm ẩn.
- **Cơ chế K-Core:** Áp dụng phương pháp lặp đệ quy:
  - Chỉ giữ lại những người dùng có ít nhất $k_{\text{user}} = 20$ tương tác.
  - Chỉ giữ lại những bộ phim có ít nhất $k_{\text{movie}} = 10$ tương tác.
- **Kết quả cải tiến:**
  - Mở rộng từ tập hẹp cũ ($16,705$ phim khi $k_{\text{movie}}=20$) lên danh mục phong phú **$21,158$ bộ phim** ($+26.6\%$ danh mục phim) và bảo toàn **24,773,346 lượt đánh giá**.
  - Đảm bảo mọi đối tượng trong hệ thống đều có đủ mẫu quan sát thống kê mà vẫn duy trì độ phủ phim đa dạng.

### 3.4. Chiến lược phân chia dữ liệu chống rò rỉ (Per-User Temporal Split)
- **Vấn đề rò rỉ dữ liệu (Data Leakage):** Việc phân chia ngẫu nhiên (*Random Split*) vi phạm nghiêm trọng tính nhân quả thời gian trong bài toán gợi ý, vì mô hình sẽ dùng dữ liệu tương lai để dự đoán quá khứ.
- **Chiến lược áp dụng:** Phân chia theo dòng thời gian cá nhân hóa từng người dùng (*Per-User Temporal Split*):
  - Với mỗi người dùng, sắp xếp toàn bộ lịch sử đánh giá theo `timestamp` tăng dần.
  - $80\%$ tương tác đầu tiên thuộc tập **Train**.
  - $10\%$ tương tác tiếp theo thuộc tập **Validation** (dùng hiệu chỉnh siêu tham số).
  - $10\%$ tương tác cuối cùng (mới nhất) thuộc tập **Test** (dành riêng cho đánh giá cuối cùng).

---

## 4. LÝ THUYẾT VÀ KIẾN TRÚC CÁC MÔ HÌNH HỆ THỐNG

### 4.1. Mô hình Popularity Baseline (IMDB Weighted Rating)
Dành cho người dùng mới hoàn toàn (giải quyết triệt để vấn đề Cold-start thuần túy). Công thức xếp hạng có hiệu chỉnh Bayes:
$$WR = \left(\frac{v}{v + m}\right) \cdot R + \left(\frac{m}{v + m}\right) \cdot C$$
Trong đó:
- $v$: Số lượt đánh giá của bộ phim (`vote_count`).
- $m$: Ngưỡng lượt đánh giá tối thiểu để đưa vào bảng xếp hạng (`POPULARITY_MIN_RATINGS = 50`).
- $R$: Điểm đánh giá trung bình của bộ phim (`vote_average`).
- $C$: Điểm đánh giá trung bình trên toàn bộ danh mục phim.

### 4.2. Lọc cộng tác: Phân rã ma trận tiềm ẩn (SVD Matrix Factorization)
Mô hình xấp xỉ ma trận xếp hạng $R \in \mathbb{R}^{|U| \times |I|}$ bằng tích của hai ma trận chiều thấp hơn:
$$\hat{r}_{u,i} = \mu + b_u + b_i + p_u^T q_i$$
Trong đó:
- $\mu$: Điểm đánh giá trung bình toàn cục.
- $b_u, b_i$: Độ lệch thiên kiến (*biases*) của người dùng $u$ và bộ phim $i$.
- $p_u \in \mathbb{R}^k$: Vector đặc trưng tiềm ẩn của người dùng $u$ ($k = 50$ chiều).
- $q_i \in \mathbb{R}^k$: Vector đặc trưng tiềm ẩn của bộ phim $i$.

**Hàm mất mát và Tối ưu hóa:**
Mô hình tối thiểu hóa sai số toàn phương có điều chuẩn L2 để chống hiện tượng quá khớp (*Overfitting*):
$$\mathcal{L}_{SVD} = \sum_{(u,i) \in R_{\text{train}}} (r_{u,i} - \hat{r}_{u,i})^2 + \lambda \left( \|p_u\|_2^2 + \|q_i\|_2^2 + b_u^2 + b_i^2 \right)$$
Tối ưu hóa bằng giải thuật **Stochastic Gradient Descent (SGD)** qua 15 epochs, tốc độ học $\gamma = 0.005$, hệ số phạt $\lambda = 0.02$. Mô hình đạt sai số trên tập Validation: $\text{RMSE} = 0.7940$ và $\text{MAE} = 0.5977$ (lưu tại `reports/cf_results.txt`).

### 4.3. Mô hình nội dung ngữ nghĩa sâu (Deep Semantic Embeddings)
Thay vì sử dụng TF-IDF truyền thống chỉ bắt các từ khóa từ vựng trùng khớp rời rạc (*lexical matching*), hệ thống ứng dụng mô hình ngôn ngữ **Sentence-Transformers (`all-MiniLM-L6-v2`)**:
- **Cấu trúc Bi-Encoder:** Mã hóa tiêu đề, thể loại và tóm tắt kịch bản thành một vector ngữ nghĩa cô đọng $e_i \in \mathbb{R}^{384}$.
- **Chuẩn hóa L2:**
  $$e_i \leftarrow \frac{e_i}{\|e_i\|_2}$$
- **Tính toán tương đồng:** Do đã chuẩn hóa vector đơn vị, độ tương đồng Cosine giữa hai bộ phim $i$ và $j$ được tính toán cực nhanh thông qua tích vô hướng:
  $$\text{Sim}(i, j) = e_i \cdot e_j^T$$
- **Dự đoán sở thích người dùng:**
  $$\hat{r}_{\text{semantic}}(u, i) = \frac{\sum_{j \in H_u} r_{u,j} \cdot \max(0, \text{Sim}(i, j))}{\sum_{j \in H_u} \max(0, \text{Sim}(i, j))}$$

### 4.4. Mô hình lai kết hợp trọng số động cá nhân hóa (Adaptive Alpha Hybrid)
Các mô hình Hybrid thông thường cố định trọng số giữa Collaborative Filtering và Content-Based ($\hat{r} = \alpha \cdot \hat{r}_{CF} + (1 - \alpha) \cdot \hat{r}_{CB}$ với $\alpha = 0.7$). Tuy nhiên, cách làm này áp đặt bất công cho người dùng mới.

**Thuật toán Adaptive Alpha đề xuất:**
Hệ thống tự động tính toán trọng số $\alpha_u$ động theo số lượng tương tác $N_u = |H_u|$ của từng người dùng bằng hàm sigmoid phi tuyến:
$$\alpha_u = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \frac{1}{1 + \exp\left(-\frac{N_u - N_{\text{threshold}}}{\tau}\right)}$$
Với các siêu tham số tối ưu:
- $\alpha_{\min} = 0.25$: Dành cho người dùng mới, ưu tiên đặc trưng nội dung ngữ nghĩa và độ phổ biến.
- $\alpha_{\max} = 0.88$: Dành cho người dùng tích cực lâu năm, tận dụng tối đa tín hiệu cộng đồng sắc bén của SVD.
- $N_{\text{threshold}} = 40$: Điểm chuyển giao niềm tin giữa nội dung và hành vi cộng đồng.

---

### 4.5. KIẾN TRÚC CÔNG NGHIỆP HAI GIAI ĐOẠN (TWO-STAGE RECOMMENDER)

#### Giai đoạn 1: Thu hồi ứng viên (Candidate Retrieval / Funneling)
Từ không gian hơn 21,000 bộ phim, hệ thống rút trích ra tập nhỏ gồm $150 \sim 200$ ứng viên tiềm năng chưa xem trong vài mili-giây từ 3 nguồn:
1. **Top 80 phim từ Popularity:** Đảm bảo độ phủ danh mục và chất lượng đại chúng.
2. **Top 80 phim từ Deep Semantic Embeddings:** Tìm kiếm các phim có cốt truyện, thể loại tương đồng nhất với 5 phim người dùng từng cho điểm cao nhất.
3. **Các ứng viên từ SVD:** Quét nhanh qua ma trận nhân tử để bổ sung các phim có tiềm năng đạt rating cao.

#### Giai đoạn 2: Trích xuất đặc trưng & Tái xếp hạng (Feature Extraction & Re-ranking)
Với mỗi cặp người dùng $u$ và phim ứng viên $i$, hệ thống trích xuất **vector 11 chiều** đại diện cho nhiều góc độ:
1. `movie_log_count`: Tần suất đánh giá của phim ($\log(1 + \text{count})$)
2. `genre_match_score`: **Cải tiến A:** Độ tương thích thể loại người dùng yêu thích
3. `diff_svd_movie`: Độ lệch giữa dự đoán SVD và điểm trung bình của bộ phim ($\hat{r}_{SVD} - \bar{r}_i$)
4. `movie_avg_rating`: Điểm đánh giá trung bình của bộ phim
5. `semantic_score`: Điểm tương đồng ngữ nghĩa từ Sentence-Transformers
6. `diff_svd_user`: Độ lệch giữa dự đoán SVD và điểm trung bình của người dùng ($\hat{r}_{SVD} - \bar{r}_u$)
7. `user_log_count`: Mức độ hoạt động của user ($\log(1 + \text{count})$)
8. `user_avg_rating`: Điểm đánh giá trung bình của user
9. `hybrid_score`: Điểm kết hợp thích ứng Adaptive Alpha
10. `svd_score`: Điểm dự đoán lọc cộng tác SVD
11. `user_adaptive_alpha`: Trọng số cá nhân hóa

#### Đặc trưng Đột phá: Genre Affinity Feature (`genre_match_score`)
Xây dựng hồ sơ phân phối thể loại yêu thích của người dùng $P_u$ từ các phim có rating $\ge 3.0$ trong lịch sử:
$$P_u(g) = \frac{\sum_{j \in H_u, g \in G_j} (r_{u,j} - 2.0)}{\sum_{g'} \sum_{j \in H_u, g' \in G_j} (r_{u,j} - 2.0)}$$
Với mỗi phim ứng viên $i$ có tập thể loại $G_i$:
$$\text{genre\_match\_score}(u, i) = \frac{1}{|G_i|} \sum_{g \in G_i} P_u(g)$$

#### Tối ưu hóa thứ hạng danh sách với LightGBM LambdaRank
Thay vì sử dụng hồi quy thông thường (Pointwise Regression tối ưu hóa MSE/MAE), hệ thống chuyển đổi sang **Learning to Rank (LTR)** với mục tiêu **LambdaRank** (`objective="lambdarank"`):
- Trực tiếp tối ưu hóa đại lượng **NDCG** trên từng danh sách truy vấn (*Query Groups*).
- LambdaRank tính toán các gradient cặp (*pairwise gradients*) $\lambda_{ij}$ có trọng số điều chỉnh theo mức độ thay đổi của NDCG khi đổi chỗ hai item $i$ và $j$:
  $$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} |\Delta \text{NDCG}|$$
- Nhờ đó, mô hình tập trung kéo các bộ phim thực sự yêu thích lên vị trí **Top 1, Top 3 và Top 10**.

#### Giải thuật Đa dạng hóa danh sách MMR (Maximal Marginal Relevance)
Ở khâu xuất kết quả cuối cùng, hệ thống tích hợp giải thuật **MMR**:
$$\text{MMR} = \arg\max_{m \in C \setminus S} \left[ \lambda \cdot \text{Score}_{\text{LightGBM}}(u, m) - (1 - \lambda) \cdot \max_{m_j \in S} \text{Sim}_{\text{Semantic}}(m, m_j) \right]$$
Trong đó:
- $C$: Tập ứng viên ban đầu.
- $S$: Tập hợp các phim đã được chọn vào danh sách gợi ý.
- $\lambda = 0.7$: Trọng số cân bằng giữa độ phù hợp (*Relevance*) và độ mới lạ (*Diversity*).

---

## 5. CÁC CHỈ SỐ VÀ GIAO THỨC ĐÁNH GIÁ (EVALUATION METRICS & PROTOCOL)

### 5.1. Giao thức kiểm thử: Negative Sampling Protocol (1 vs 99)
- Với mỗi người dùng trong tập Test, lấy 1 bộ phim người dùng thực tế đã thích ($r_{u,i} \ge 3.5$) làm **Ground Truth Positive**.
- Lấy mẫu ngẫu nhiên 99 bộ phim mà người dùng chưa từng tương tác làm **Unrated Negatives**.
- Đưa toàn bộ 100 bộ phim vào mô hình để dự đoán điểm và xếp hạng từ cao xuống thấp.
- Đánh giá xem bộ phim thực tế nằm ở vị trí nào trong danh sách Top-10 được gợi ý.

### 5.2. Công thức toán học các chỉ số

#### 1. Hit Rate@K (HR@10):
$$\text{HR@K} = \frac{1}{|U_{\text{eval}}|} \sum_{u \in U_{\text{eval}}} \mathbb{I}(\text{Rank}_u \le K)$$

#### 2. Normalized Discounted Cumulative Gain (NDCG@K):
Trong kịch bản 1 Positive ($rel = 1$) và 99 Negatives ($rel = 0$), nếu phim mục tiêu nằm ở vị trí thứ tự $r \le K$ (với chỉ số $r$ tính từ 0):
$$\text{NDCG@K} = \frac{1}{\log_2(r + 2)}$$
Nếu phim mục tiêu không nằm trong Top-K, $\text{NDCG@K} = 0$.

#### 3. Sai số dự đoán điểm: RMSE & MAE
$$\text{RMSE} = \sqrt{\frac{1}{|T|} \sum_{(u,i) \in T} (r_{u,i} - \hat{r}_{u,i})^2}, \quad \text{MAE} = \frac{1}{|T|} \sum_{(u,i) \in T} |r_{u,i} - \hat{r}_{u,i}|$$

---

## 6. KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH CHUYÊN SÂU

### 6.1. Bảng so sánh hiệu năng tổng thể trên tập Test độc lập

| Mô hình | Hit Rate@10 | NDCG@10 | Nhận xét phân tích |
| :--- | :---: | :---: | :--- |
| **Content TF-IDF** | 0.1833 (18.33%) | 0.0871 | Bị giới hạn bởi từ vựng nông, dễ bắt nhầm từ khóa không quan trọng |
| **Semantic (Sentence-BERT)** | 0.1467 (14.67%) | 0.0591 | Nắm bắt cốt truyện rất hay nhưng thiếu hoàn toàn tín hiệu hành vi người dùng |
| **SVD (Collaborative Filtering)** | 0.3967 (39.67%) | 0.2328 | Học rất tốt mối tương quan hành vi tiềm ẩn trên ma trận K-Core |
| **Hybrid Fixed ($\alpha = 0.7$)** | 0.3867 (38.67%) | 0.2239 | Trọng số cứng làm giảm độ chính xác ở nhóm người dùng ít tương tác |
| **Hybrid Adaptive Alpha** | 0.3867 (38.67%) | **0.2436** | Tối ưu hóa điểm NDCG nhờ cơ chế chuyển giao linh hoạt giữa nội dung và CF |
| **Two-Stage LightGBM (LambdaRank + Genre)** | **0.8300 (83.00%)** | **0.5649** | **Đột phá vượt trội: Hit Rate đạt 83% và NDCG đạt 0.5649, vượt xa tất cả các mô hình đơn lẻ** |

### 6.2. Tiến trình cải tiến mô hình qua các mốc phát triển

1. **Hệ thống sơ khai (Chưa lọc K-Core):** Hit Rate@10 chỉ đạt **$2.40\%$**, NDCG@10 là **$0.0250$**.
2. **K-Core (min_movie=20) + Two-Stage cơ bản:** Hit Rate nhảy vọt lên **$80.00\%$** (NDCG $0.5041$) trên $16,705$ bộ phim.
3. **Mở rộng K-Core (min_movie=10) + Tối ưu SVD:** Mở rộng thành công không gian phim lên **$21,158$ bộ phim** ($+4,453$ phim mới), SVD đạt Validation RMSE ấn tượng **$0.7940$** và MAE **$0.5977$** (đo đạc trên 21.158 phim, lưu tại `reports/cf_results.txt`).
4. **Tích hợp Genre Affinity & LambdaRank (Mô hình hiện tại):**
   - Đưa đặc trưng `genre_match_score` vào bộ phân loại.
   - Chuyển mục tiêu sang xếp hạng danh sách LambdaRank.
   - Thiết lập kỷ lục mới: **Hit Rate@10 đạt $83.00\%$** và **NDCG@10 đạt $0.5649$**.

### 6.3. Tầm quan trọng của các đặc trưng trong LightGBM (Feature Importance)

1. `movie_log_count` (696 điểm): Đặc trưng số 1 về độ tin cậy và sự phổ biến của phim.
2. `genre_match_score` (618 điểm): **Đặc trưng Cải tiến A xuất sắc đứng thứ 2 toàn hệ thống**.
3. `diff_svd_movie` (292 điểm): Độ lệch giữa dự đoán SVD và mặt bằng chung của bộ phim.
4. `movie_avg_rating` (274 điểm): Điểm số trung bình đại diện cho chất lượng.
5. `semantic_score` (271 điểm): Đóng góp ngữ nghĩa từ vector nhúng sâu.
6. `diff_svd_user` (244 điểm): Độ lệch sở thích so với mức trung bình của user.
7. `user_log_count` (216 điểm) & `user_avg_rating` (147 điểm): Đặc trưng thói quen người dùng.
8. `hybrid_score` (122 điểm) & `svd_score` (118 điểm): Điểm số ước lượng từ tầng dưới.
9. `user_adaptive_alpha` (2 điểm): Đặc trưng điều hòa.

---

## 7. CẤU TRÚC THƯ MỤC VÀ HỆ THỐNG MÃ NGUỒN

```
anti/
├── config.py                      # Cấu hình đường dẫn, siêu tham số và constants toàn hệ thống
├── requirements.txt               # Danh sách thư viện phụ thuộc
├── README.md                      # Hướng dẫn và tài liệu dự án
│
├── data/                          # Thư mục dữ liệu
│   ├── ratings.csv, movies_metadata.csv, keywords.csv, credits.csv, links.csv
│   └── processed/                 # Dữ liệu dạng Parquet (clean, k-core, train/val/test)
│
├── src/                           # Mã nguồn thực thi pipeline
│   ├── 01_check_data.py           # EDA và kiểm tra dữ liệu
│   ├── 02_preprocess.py           # Tiền xử lý theo chunk, parse JSON AST, nén Parquet
│   ├── 04_prepare_data.py         # Phân chia dữ liệu theo thời gian (Per-user Temporal Split)
│   ├── 05_train_baseline.py       # Popularity Baseline (IMDB Weighted Rating)
│   ├── 06_train_cf.py             # SVD Matrix Factorization
│   ├── 07_train_content.py        # Content-Based TF-IDF
│   ├── 07_train_semantic.py       # Deep Semantic Sentence-Transformers
│   ├── 08_train_hybrid.py         # Hybrid Fixed & Adaptive Alpha
│   ├── 08_train_reranker.py       # Huấn luyện LightGBM LambdaRank + Genre Affinity
│   ├── 09_tune_models.py          # Grid search siêu tham số
│   ├── 10_evaluate.py             # Đánh giá toàn diện 6 mô hình bằng Negative Sampling
│   ├── 11_recommend.py            # CLI xuất gợi ý phim thực tế kèm MMR
│   ├── adaptive_alpha.py          # Module hàm tính trọng số động
│   ├── semantic_embeddings.py     # Module mã hóa và suy luận Sentence-BERT
│   ├── two_stage_recommender.py   # Module Two-Stage Pipeline (Retrieval + Rerank + MMR)
│   └── update_all_reports.py      # Tiện ích đồng bộ tự động báo cáo và đồ thị
│
├── models/                        # Trọng số mô hình (.pkl, .npy, .parquet)
├── outputs/                       # Bảng số liệu đầu ra (.parquet, .txt)
└── reports/                       # Nhật ký huấn luyện và biểu đồ trực quan hóa
```

---

## 8. KẾT LUẬN

Hệ thống đã giải quyết trọn vẹn cả 3 thách thức lớn nhất của hệ thống gợi ý quy mô lớn: **Độ thưa ma trận (99.78%)**, **Khởi đầu lạnh (Cold-start)** và **Tốc độ đáp ứng thời gian thực kết hợp chất lượng cá nhân hóa**.

Nhờ kiến trúc Hai giai đoạn kết hợp **LightGBM LambdaRank**, đặc trưng **Genre Affinity** và bộ lọc **MMR**, hệ thống đạt độ chính xác hàng đầu: **Hit Rate@10 đạt 83.00%** và **NDCG@10 đạt 0.5649**, sẵn sàng cho việc triển khai ứng dụng thực tế.
