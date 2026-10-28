"""
train_content_module.py - Module chứa class ContentBasedRecommender
File này được tách riêng để có thể import từ nhiều file khác
(07_train_content.py, 08_train_hybrid.py, 10_evaluate.py)

Không chạy trực tiếp file này. Import từ file khác.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """
    Mô hình gợi ý dựa trên nội dung phim.
    
    Pipeline:
    1. Mỗi phim có chuỗi 'content_soup' = thể loại + từ khóa + mô tả + đạo diễn + diễn viên
    2. TF-IDF vectorizer chuyển chuỗi thành vector số
    3. Tính cosine similarity giữa tất cả cặp phim  ma trận độ tương đồng
    4. Khi gợi ý: tìm phim tương tự nhất với những phim user thích
    
    Ưu điểm:
    - Không cần lịch sử user khác (không có privacy issue)
    - Giải thích được (vì sao gợi ý phim này)
    - Không có cold-start với phim mới (chỉ cần có metadata)
    
    Nhược điểm:
    - Cold-start với USER mới (chưa có lịch sử)
    - Dễ bị "filter bubble" (chỉ gợi ý phim giống những gì đã xem)
    - Phụ thuộc chất lượng metadata
    """

    def __init__(self, max_features=config.CONTENT_MAX_FEATURES):
        self.max_features = max_features
        self.tfidf_matrix = None
        self.vectorizer   = None
        self.movies_df    = None
        self.movie_idx_map = None

    def fit(self, df_movies):
        """
        Xây dựng TF-IDF vectors cho tất cả phim.
        
        Tham số:
            df_movies: DataFrame với cột 'movieId' và 'content_soup'
        """
        import time
        logger.info("Fit TF-IDF vectorizer trên content_soup")
        logger.info(f"  Số phim: {len(df_movies):,}")
        logger.info(f"  max_features TF-IDF: {self.max_features:,}")

        self.movies_df = df_movies.reset_index(drop=True).copy()
        soups = self.movies_df["content_soup"].fillna("").tolist()

        # TF-IDF Vectorizer
        # - ngram_range=(1,2): đơn từ + cặp từ
        # - min_df=2: bỏ qua từ chỉ xuất hiện ở 1 phim
        # - max_df=0.95: bỏ qua từ xuất hiện ở >95% phim
        # - sublinear_tf=True: log(TF+1) thay vì TF (mượt hơn)
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=self.max_features,
            sublinear_tf=True
        )

        t0 = time.time()
        self.tfidf_matrix = self.vectorizer.fit_transform(soups)
        logger.info(f"  TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        logger.info(f"  Thời gian: {time.time()-t0:.1f}s")

        # Ánh xạ movieId  index
        self.movie_idx_map = dict(zip(
            self.movies_df["movieId"].values,
            range(len(self.movies_df))
        ))

        logger.info(f"  Số đặc trưng: {self.tfidf_matrix.shape[1]:,}")
        return self

    def get_similar_movies(self, movie_id, n=config.CONTENT_TOP_N_SIMILAR, exclude_self=True):
        """
        Tìm N phim tương tự nhất với phim cho trước.
        
        Trả về:
            List các (movieId, similarity_score) sắp xếp theo score giảm dần
        """
        if movie_id not in self.movie_idx_map:
            return []

        idx = self.movie_idx_map[movie_id]
        sim_scores = linear_kernel(
            self.tfidf_matrix[idx:idx+1], self.tfidf_matrix
        ).flatten()

        sorted_indices = np.argsort(sim_scores)[::-1]
        if exclude_self:
            sorted_indices = sorted_indices[sorted_indices != idx]

        top_indices = sorted_indices[:n]
        top_scores  = sim_scores[top_indices]
        result_movie_ids = self.movies_df["movieId"].values[top_indices]
        return list(zip(result_movie_ids.tolist(), top_scores.tolist()))

    def recommend_for_user(self, user_history, n=config.TOP_K, exclude_seen=True):
        """
        Tạo Top-N gợi ý cho user dựa trên lịch sử xem phim.
        
        Tham số:
            user_history: Dict {movieId: rating} - lịch sử xem của user
            n: Số phim gợi ý
            exclude_seen: Loại phim đã xem
        
        Trả về:
            List các (movieId, score)
        """
        if not user_history:
            return []

        # Ưu tiên phim user thích (giới hạn tối đa 20 phim đánh giá cao nhất làm seed)
        liked_movies = {mid: r for mid, r in user_history.items()
                        if r >= config.RELEVANT_RATING_THRESHOLD}
        if not liked_movies:
            liked_movies = user_history

        if len(liked_movies) > 20:
            liked_sorted = sorted(liked_movies.items(), key=lambda x: x[1], reverse=True)[:20]
            liked_movies = dict(liked_sorted)

        candidate_scores = {}
        for movie_id, user_rating in liked_movies.items():
            weight = (user_rating - config.MIN_RATING) / (config.MAX_RATING - config.MIN_RATING)
            similar = self.get_similar_movies(movie_id, n=config.CONTENT_TOP_N_SIMILAR)

            for sim_movie_id, sim_score in similar:
                if exclude_seen and sim_movie_id in user_history:
                    continue
                if sim_movie_id not in candidate_scores:
                    candidate_scores[sim_movie_id] = 0
                candidate_scores[sim_movie_id] += weight * sim_score

        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_candidates[:n]

    def predict_rating(self, user_history, movie_id):
        """
        Dự đoán rating của user cho phim dựa trên Content-Based.
        
        Dùng trung bình có trọng số theo cosine similarity (vectorized, siêu nhanh).
        
        Trả về:
            float: Rating dự đoán
        """
        if movie_id not in self.movie_idx_map or not user_history:
            return 3.5

        idx_target = self.movie_idx_map[movie_id]

        # Thu thập các movie index và rating tương ứng
        seen_indices = []
        seen_ratings = []
        for seen_id, seen_r in user_history.items():
            if seen_id in self.movie_idx_map:
                seen_indices.append(self.movie_idx_map[seen_id])
                seen_ratings.append(seen_r)

        if not seen_indices:
            return 3.5

        # Tính similarity cho toàn bộ phim trong history bằng phép nhân ma trận sparse (siêu nhanh)
        target_vec = self.tfidf_matrix[idx_target].T
        sims = self.tfidf_matrix[seen_indices].dot(target_vec).toarray().flatten()

        # Chỉ tính trọng số cho phim có độ tương đồng dương
        pos_mask = sims > 0
        if not np.any(pos_mask):
            return 3.5

        weights = sims[pos_mask]
        ratings = np.array(seen_ratings, dtype=np.float32)[pos_mask]
        total_weight = float(weights.sum())

        if total_weight <= 1e-6:
            return 3.5

        pred = float(np.dot(weights, ratings) / total_weight)
        return float(np.clip(pred, config.MIN_RATING, config.MAX_RATING))

    def save(self, model_dir):
        """Lưu model ra thư mục."""
        joblib.dump(self.vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        sp.save_npz(os.path.join(model_dir, "tfidf_matrix.npz"), self.tfidf_matrix)
        self.movies_df.to_parquet(os.path.join(model_dir, "content_movies.parquet"), index=False)
        joblib.dump(self.movie_idx_map, os.path.join(model_dir, "movie_idx_map.pkl"))
        logger.info(f"Đã lưu Content-Based model vào: {model_dir}")

    @staticmethod
    def load(model_dir):
        """Tải model đã lưu."""
        model = ContentBasedRecommender()
        model.vectorizer    = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        model.tfidf_matrix  = sp.load_npz(os.path.join(model_dir, "tfidf_matrix.npz"))
        model.movies_df     = pd.read_parquet(os.path.join(model_dir, "content_movies.parquet"))
        model.movie_idx_map = joblib.load(os.path.join(model_dir, "movie_idx_map.pkl"))
        return model
