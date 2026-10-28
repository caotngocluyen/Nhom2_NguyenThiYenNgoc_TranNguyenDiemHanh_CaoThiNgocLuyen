"""
config.py - Cấu hình toàn bộ dự án Movie Recommendation System

File này chứa tất cả đường dẫn và tham số cấu hình dùng chung.

"""

import os
import sys

# Cấu hình encoding UTF-8 cho Windows console để tránh lỗi in ký tự tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# CẤU HÌNH ĐƯỜNG DẪN (Paths Configuration)


# Thư mục gốc của dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn dữ liệu gốc (raw data - không bao giờ được ghi đè)
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Đường dẫn dữ liệu gốc tạm thời (dữ liệu gốc chưa di chuyển vào raw/)
DATA_DIR = os.path.join(BASE_DIR, "data")

# Đường dẫn dữ liệu đã xử lý
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Đường dẫn lưu mô hình
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Đường dẫn lưu kết quả đầu ra
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Đường dẫn lưu báo cáo và biểu đồ
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")


# TÊN FILE DỮ LIỆU GỐC (Raw Data Files)


# File ratings chính (~26 triệu lượt đánh giá)
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")

# Metadata phim (tên, thể loại, ngày phát hành, mô tả,)
MOVIES_METADATA_FILE = os.path.join(DATA_DIR, "movies_metadata.csv")

# Từ khóa phim (dùng cho Content-Based Filtering)
KEYWORDS_FILE = os.path.join(DATA_DIR, "keywords.csv")

# Diễn viên và đạo diễn
CREDITS_FILE = os.path.join(DATA_DIR, "credits.csv")

# Bảng liên kết ID giữa MovieLens và TMDB
LINKS_FILE = os.path.join(DATA_DIR, "links.csv")


# TÊN FILE DỮ LIỆU ĐÃ XỬ LÝ (Processed Data Files)


# Ratings đã làm sạch (định dạng Parquet để tiết kiệm RAM và đọc nhanh)
RATINGS_CLEAN_FILE = os.path.join(DATA_PROCESSED_DIR, "ratings_clean.parquet")

# Metadata phim đã làm sạch
MOVIES_CLEAN_FILE = os.path.join(DATA_PROCESSED_DIR, "movies_clean.parquet")

# Bảng ánh xạ ID (movieId <-> tmdbId <-> index nội bộ)
ID_MAP_FILE = os.path.join(DATA_PROCESSED_DIR, "id_map.parquet")

# Đặc trưng nội dung phim cho Content-Based (TF-IDF vectors)
CONTENT_FEATURES_FILE = os.path.join(DATA_PROCESSED_DIR, "content_features.npz")

# Từ điển metadata nội dung phim
CONTENT_MOVIES_FILE = os.path.join(DATA_PROCESSED_DIR, "content_movies.parquet")

# Tập train
TRAIN_FILE = os.path.join(DATA_PROCESSED_DIR, "train.parquet")
TRAIN_IMPROVED_FILE = os.path.join(DATA_PROCESSED_DIR, "train_improved.parquet")

# Tập validation (dùng để tuning tham số)
VAL_FILE = os.path.join(DATA_PROCESSED_DIR, "val.parquet")
VAL_IMPROVED_FILE = os.path.join(DATA_PROCESSED_DIR, "val_improved.parquet")

# Tập test (chỉ dùng để đánh giá cuối cùng)
TEST_FILE = os.path.join(DATA_PROCESSED_DIR, "test.parquet")
TEST_IMPROVED_FILE = os.path.join(DATA_PROCESSED_DIR, "test_improved.parquet")

# File mô hình mới (Two-Stage, Semantic, SVD Cải Tiến)
SVD_IMPROVED_MODEL = os.path.join(MODELS_DIR, "svd_improved.pkl")
SEMANTIC_EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "semantic_embeddings.npy")
SEMANTIC_MOVIES_FILE = os.path.join(MODELS_DIR, "semantic_movies.parquet")
SEMANTIC_INDEX_FILE = os.path.join(MODELS_DIR, "semantic_movie_idx.pkl")
RERANKER_MODEL_FILE = os.path.join(MODELS_DIR, "reranker_lgbm.pkl")
TWO_STAGE_METADATA_FILE = os.path.join(MODELS_DIR, "two_stage_metadata.pkl")


# THAM SỐ TIỀN XỬ LÝ (Preprocessing Parameters)


# Khoảng rating hợp lệ (MovieLens dùng thang 0.5 - 5.0)
MIN_RATING = 0.5
MAX_RATING = 5.0

# Ngưỡng rating để coi là "phim phù hợp" (dùng trong Top-N evaluation)
# Phim có rating >= 3.5 được coi là phim mà người dùng thích
RELEVANT_RATING_THRESHOLD = 3.5


# THAM SỐ CHIA TẬP (Split Parameters)


# Tỷ lệ chia: 80% train, 10% validation, 10% test
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# Random seed để tái lập kết quả (reproducibility)
RANDOM_SEED = 42


# THAM SỐ MÔ HÌNH (Model Parameters)


#  Collaborative Filtering (SVD Matrix Factorization) 
# n_factors: số chiều của không gian tiềm ẩn (latent factors)
CF_N_FACTORS = 100
CF_N_EPOCHS  = 20
CF_LR_ALL    = 0.005   # learning rate
CF_REG_ALL   = 0.02    # hệ số regularization (tránh overfitting)

#  Content-Based Filtering 
# Số phim tương tự tối đa khi tính cosine similarity
CONTENT_TOP_N_SIMILAR = 50

# Số đặc trưng TF-IDF tối đa
CONTENT_MAX_FEATURES = 5000

#  Popularity-Based 
# Số lượt đánh giá tối thiểu để phim được coi là "phổ biến"
POPULARITY_MIN_RATINGS = 50

#  ALS (Alternating Least Squares) cho implicit feedback 
ALS_FACTORS    = 50
ALS_ITERATIONS = 15
ALS_REGULARIZATION = 0.01


# THAM SỐ ĐÁNH GIÁ (Evaluation Parameters)


# Top-N recommendations (K = 10 là tiêu chuẩn phổ biến)
TOP_K = 10

# Số người dùng mẫu để đánh giá Top-N (đánh giá toàn bộ 270K users rất chậm)
# Đặt None để đánh giá tất cả
EVAL_SAMPLE_USERS = 10000


# THAM SỐ TUNING (Hyperparameter Tuning)


# Grid search tham số cho SVD
SVD_PARAM_GRID = {
    "n_factors": [50, 100, 150],
    "n_epochs":  [15, 20],
    "lr_all":    [0.003, 0.005],
    "reg_all":   [0.02, 0.05],
}

# Dải alpha để thử nghiệm cho Hybrid (0.0 = thuần CF, 1.0 = thuần Content)
HYBRID_ALPHA_RANGE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Tham số Semantic Embeddings (Sentence-Transformers)
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
SEMANTIC_MAX_SEQ_LENGTH = 96
SEMANTIC_EMBEDDING_DIM = 384

# Tham số Adaptive Alpha
ADAPTIVE_ALPHA_MIN = 0.25
ADAPTIVE_ALPHA_MAX = 0.88
ADAPTIVE_ALPHA_THRESHOLD = 40

# Tham số Two-Stage Recommender (Retrieval + LightGBM Re-ranking)
TWO_STAGE_CANDIDATES = 200
RERANKER_OBJECTIVE = "lambdarank"
RERANKER_LEARNING_RATE = 0.05
RERANKER_NUM_LEAVES = 31
RERANKER_N_ESTIMATORS = 100

# Tham số Đa dạng hóa danh sách gợi ý (MMR - Maximal Marginal Relevance)
MMR_DEFAULT_LAMBDA = 0.7  # Trọng số cân bằng giữa độ phù hợp (0.7) và độ mới lạ (0.3)


# LOGGING


LOG_LEVEL = "INFO"  # Mức độ log: DEBUG, INFO, WARNING, ERROR


# TẠO THƯ MỤC NẾU CHƯA TỒN TẠI


for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(_dir, exist_ok=True)
