"""
05_train_baseline.py - Mô hình cơ sở dựa trên độ phổ biến (Popularity-Based)
MỤC ĐÍCH:
    Xây dựng các mô hình khuyến nghị đơn giản nhất làm baseline để so sánh.
    Các baseline được triển khai:

    1. GLOBAL MEAN BASELINE:
       - Dự đoán rating = trung bình toàn cục (mean của tất cả ratings)
       - Đây là baseline đơn giản nhất có thể
    
    2. POPULARITY-BASED RECOMMENDATION (Đề xuất theo độ phổ biến):
       - Khuyến nghị những phim được đánh giá nhiều nhất VÀ điểm cao nhất
       - Không cá nhân hóa - mọi người đều nhận cùng danh sách
       - Phù hợp cho người dùng mới (cold-start)
    
    3. WEIGHTED RATING BASELINE (Bayesian Average):
       - Kết hợp điểm trung bình và số lượt đánh giá
       - Công thức: WR = (v/(v+m)) * R + (m/(v+m)) * C
         Trong đó: v = số vote của phim, m = vote tối thiểu, 
                   R = rating trung bình phim, C = rating trung bình toàn cục
       - Tránh phim ít lượt nhưng điểm cao 10/10 đứng đầu

ĐẦU VÀO:
    - data/processed/train.parquet
    - data/processed/val.parquet
    - data/processed/movies_clean.parquet (tùy chọn - lấy tên phim)

ĐẦU RA:
    - models/popularity_model.parquet (danh sách phim phổ biến đã tính sẵn)
    - outputs/baseline_recommendations.parquet
    - reports/baseline_metrics.txt

CÁCH CHẠY:
    python src/05_train_baseline.py
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np
import joblib
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.REPORTS_DIR, "05_baseline.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)



# MÔ HÌNH 1: GLOBAL MEAN (Trung bình toàn cục)


class GlobalMeanPredictor:
    """
    Mô hình dự đoán đơn giản nhất: luôn dự đoán = rating trung bình.
    
    Đây là lower bound - mọi mô hình tốt hơn phải đánh bại nó.
    """
    
    def __init__(self):
        self.global_mean = None
    
    def fit(self, df_train):
        """Tính rating trung bình từ tập huấn luyện."""
        self.global_mean = df_train["rating"].mean()
        logger.info(f"GlobalMean fit: mean = {self.global_mean:.4f}")
        return self
    
    def predict(self, df):
        """Dự đoán: tất cả đều bằng global mean."""
        return np.full(len(df), self.global_mean, dtype=np.float32)
    
    def save(self, path):
        joblib.dump(self, path)
        logger.info(f"Đã lưu GlobalMeanPredictor: {path}")
    
    @staticmethod
    def load(path):
        return joblib.load(path)



# MÔ HÌNH 2: POPULARITY-BASED RECOMMENDER


class PopularityRecommender:
    """
    Gợi ý phim dựa trên độ phổ biến (Popularity-Based Filtering).
    
    Cách hoạt động:
    1. Tính Bayesian Average score cho mỗi phim
    2. Xếp hạng phim theo score
    3. Khi có user request -> trả về Top-K phim phổ biến nhất mà user CHƯA xem
    
    Ưu điểm:
    - Đơn giản, nhanh, không cần lịch sử user cụ thể
    - Hiệu quả cho người dùng mới (cold-start)
    
    Nhược điểm:
    - Không cá nhân hóa
    - Ủng hộ phim nổi tiếng, bỏ qua phim ít biết nhưng chất lượng
    """
    
    def __init__(self, min_ratings=config.POPULARITY_MIN_RATINGS):
        """
        Tham số:
            min_ratings: Số lượt đánh giá tối thiểu để phim được xem xét.
                         Phim ít hơn ngưỡng này sẽ bị loại (tránh phim 5 nhưng chỉ 1 người xem)
        """
        self.min_ratings = min_ratings
        self.popularity_scores = None   # DataFrame xếp hạng phim
        self.global_mean = None
        self.user_seen_movies = None    # Dict: userId -> set of movieIds đã xem
    
    def fit(self, df_train):
        """
        Huấn luyện: Tính điểm phổ biến (Bayesian Average) cho mỗi phim.
        
        Công thức Bayesian Average (tương tự IMDB top 250):
        WR = (v / (v + m)) × R + (m / (v + m)) × C
        
        Trong đó:
        - v = số lượt vote của phim này
        - m = ngưỡng vote tối thiểu (hyperparameter)
        - R = rating trung bình của phim này
        - C = rating trung bình của toàn bộ dataset
        """
        logger.info("Fitting PopularityRecommender")

        # Rating trung bình toàn cục C
        C = df_train["rating"].mean()
        self.global_mean = C

        # Tính thống kê theo phim
        movie_stats = df_train.groupby("movieId").agg(
            vote_count=("rating", "count"),
            vote_avg=("rating", "mean")
        ).reset_index()

        # Lọc phim có đủ lượt đánh giá
        movie_stats = movie_stats[movie_stats["vote_count"] >= self.min_ratings].copy()
        logger.info(f"  Phim đủ ngưỡng (>= {self.min_ratings} ratings): {len(movie_stats):,}")

        # Tính Bayesian Average score
        m = self.min_ratings
        v = movie_stats["vote_count"]
        R = movie_stats["vote_avg"]
        movie_stats["bayesian_score"] = (v / (v + m)) * R + (m / (v + m)) * C

        # Sắp xếp theo score giảm dần
        self.popularity_scores = movie_stats.sort_values(
            "bayesian_score", ascending=False
        ).reset_index(drop=True)
        
        # Lưu danh sách phim user đã xem (để loại ra khỏi gợi ý)
        self.user_seen_movies = df_train.groupby("userId")["movieId"].apply(set).to_dict()

        logger.info(f"  Top 5 phim phổ biến nhất:")
        for _, row in self.popularity_scores.head(5).iterrows():
            mid_val = int(row['movieId'])
            vc_val = int(row['vote_count'])
            logger.info(f"    movieId={mid_val:5d}: "
                        f"score={row['bayesian_score']:.3f}, "
                        f"votes={vc_val:,}, "
                        f"avg={row['vote_avg']:.2f}")

        return self
    
    def recommend(self, user_id, n=config.TOP_K, exclude_seen=True):
        """
        Tạo danh sách gợi ý cho một user.
        
        Tham số:
            user_id: ID người dùng
            n: Số phim gợi ý (Top-N)
            exclude_seen: Loại phim user đã xem khỏi danh sách
        
        Trả về:
            DataFrame với cột movieId và bayesian_score
        """
        if self.popularity_scores is None:
            raise RuntimeError("Chưa fit mô hình! Gọi fit() trước.")
        
        candidates = self.popularity_scores.copy()
        
        if exclude_seen and user_id in self.user_seen_movies:
            # Loại phim đã xem
            seen = self.user_seen_movies[user_id]
            candidates = candidates[~candidates["movieId"].isin(seen)]
        
        return candidates.head(n)[["movieId", "bayesian_score"]].copy()
    
    def predict_rating(self, df):
        """
        Dự đoán rating bằng cách dùng điểm phổ biến (bayesian_score).
        Fallback về global mean nếu phim không có trong model.
        """
        score_dict = dict(zip(
            self.popularity_scores["movieId"],
            self.popularity_scores["bayesian_score"]
        ))
        preds = df["movieId"].map(score_dict).fillna(self.global_mean)
        return preds.values.astype(np.float32)
    
    def save(self, path):
        joblib.dump(self, path)
        logger.info(f"Đã lưu PopularityRecommender: {path}")
    
    @staticmethod
    def load(path):
        return joblib.load(path)



# TÍNH METRICS CƠ BẢN


def tinh_rmse_mae(y_true, y_pred):
    """
    Tính RMSE và MAE cho bài toán dự đoán rating.
    
    RMSE (Root Mean Squared Error - Căn bậc hai trung bình bình phương sai số):
    - Penalize mạnh sai số lớn
    - RMSE = sqrt(mean((y_true - y_pred)^2))
    
    MAE (Mean Absolute Error - Trung bình sai số tuyệt đối):
    - Dễ hiểu hơn RMSE
    - MAE = mean(|y_true - y_pred|)
    
    Cả hai: Giá trị càng nhỏ càng tốt.
    """
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    errors = y_true - y_pred
    rmse = np.sqrt(np.mean(errors ** 2))
    mae  = np.mean(np.abs(errors))
    return rmse, mae


def danh_gia_baseline(df_val, mo_hinh_ten, y_pred):
    """In kết quả đánh giá RMSE/MAE cho một mô hình."""
    rmse, mae = tinh_rmse_mae(df_val["rating"].values, y_pred)
    logger.info(f"\n  [{mo_hinh_ten}]")
    logger.info(f"    RMSE: {rmse:.4f}")
    logger.info(f"    MAE:  {mae:.4f}")
    return {"model": mo_hinh_ten, "rmse": rmse, "mae": mae}


def hien_thi_goi_y_vi_du(model, df_movies, n_users=5):
    """Hiển thị ví dụ gợi ý cho một số người dùng."""
    if df_movies is None:
        return

    logger.info(f"\n Ví dụ gợi ý (Top-{config.TOP_K}) ")
    movie_title = dict(zip(df_movies["movieId"], df_movies.get("title", df_movies["movieId"])))

    # Lấy một số userId mẫu
    sample_users = list(model.user_seen_movies.keys())[:n_users]

    for uid in sample_users:
        recs = model.recommend(uid, n=config.TOP_K, exclude_seen=True)
        logger.info(f"\n  User {uid}:")
        for _, row in recs.iterrows():
            title = movie_title.get(row["movieId"], f"Movie {row['movieId']}")
            logger.info(f"    - {title[:50]:<50} (score: {row['bayesian_score']:.3f})")


def main():
    """Hàm chính."""

    # Tải dữ liệu
    logger.info("Tải dữ liệu train và val")
    df_train = pd.read_parquet(config.TRAIN_FILE)
    df_val   = pd.read_parquet(config.VAL_FILE)

    df_movies = None
    if os.path.exists(config.MOVIES_CLEAN_FILE):
        df_movies = pd.read_parquet(config.MOVIES_CLEAN_FILE)

    logger.info(f"  Train: {len(df_train):,} ratings")
    logger.info(f"  Val:   {len(df_val):,} ratings")

    results = []

    
    # Mô hình 1: Global Mean
    
    logger.info("\n Mô hình 1: Global Mean ")
    t0 = time.time()
    gm = GlobalMeanPredictor()
    gm.fit(df_train)
    gm.save(os.path.join(config.MODELS_DIR, "global_mean.pkl"))
    
    pred_gm = gm.predict(df_val)
    metrics_gm = danh_gia_baseline(df_val, "Global Mean", pred_gm)
    metrics_gm["train_time"] = time.time() - t0
    results.append(metrics_gm)

    
    # Mô hình 2: Popularity-Based
    
    logger.info("\n Mô hình 2: Popularity-Based (Bayesian Average) ")
    logger.info("""
    Nguyên lý: Dùng công thức Bayesian Average (tương tự IMDB Top 250)
    để xếp hạng phim theo cả số lượt đánh giá lẫn điểm trung bình.
    Phim có nhiều rating cao sẽ được đề xuất cho mọi người dùng.
    """)
    t0 = time.time()
    pop = PopularityRecommender(min_ratings=config.POPULARITY_MIN_RATINGS)
    pop.fit(df_train)
    pop.save(os.path.join(config.MODELS_DIR, "popularity_model.pkl"))

    pred_pop = pop.predict_rating(df_val)
    metrics_pop = danh_gia_baseline(df_val, "Popularity (Bayesian)", pred_pop)
    metrics_pop["train_time"] = time.time() - t0
    results.append(metrics_pop)

    # Lưu danh sách phim phổ biến
    top_movies_path = os.path.join(config.MODELS_DIR, "top_popular_movies.parquet")
    pop.popularity_scores.to_parquet(top_movies_path, index=False)
    logger.info(f"  Danh sách phim phổ biến lưu tại: {top_movies_path}")

    
    # Hiển thị ví dụ gợi ý
    
    hien_thi_goi_y_vi_du(pop, df_movies, n_users=3)

    
    # Tổng kết và lưu kết quả
    
    logger.info("\n" + "=" * 60)
    logger.info("KẾT QUẢ BASELINE MODELS")
    logger.info("=" * 60)
    logger.info(f"{'Model':<30} {'RMSE':>8} {'MAE':>8} {'Time(s)':>10}")
    logger.info("-" * 60)
    for r in results:
        logger.info(f"  {r['model']:<28} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['train_time']:>10.1f}")

    # Lưu kết quả ra file
    df_results = pd.DataFrame(results)
    df_results.to_parquet(
        os.path.join(config.OUTPUTS_DIR, "baseline_metrics.parquet"), index=False
    )
    
    # Lưu text report
    report_path = os.path.join(config.REPORTS_DIR, "baseline_metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BASELINE MODEL RESULTS\n")
        f.write("=" * 60 + "\n")
        for r in results:
            f.write(f"{r['model']}: RMSE={r['rmse']:.4f}, MAE={r['mae']:.4f}\n")
    
    logger.info(f"\nKết quả lưu tại: {report_path}")
    logger.info(" Bước tiếp theo: chạy src/06_train_cf.py")


if __name__ == "__main__":
    main()
