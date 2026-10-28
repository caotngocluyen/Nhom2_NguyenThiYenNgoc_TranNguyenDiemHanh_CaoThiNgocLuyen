import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SemanticContentRecommender:
    """
    Mo hinh goi y dua tren bieu dien ngu nghia sau (Semantic Embeddings).
    Su dung mo hinh ngon ngu all-MiniLM-L6-v2 de ma hoa tieu de, the loai
    va tom tat cot truyen thanh vector 384 chieu chuan hoa L2.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self.embeddings = None
        self.movies_df = None
        self.movie_idx_map = None

    def fit(self, df_movies, batch_size=256, max_seq_length=96):
        """
        Ma hoa toan bo phim thanh vector ngu nghia 384 chieu.
        
        Tham so:
            df_movies: DataFrame chua thong tin phim (movieId, title, overview, genres_text).
            batch_size: Kich thuoc lo khi chay mo hinh encoder.
            max_seq_length: Do dai toi da chuoi token (toi uu toc do va bo nho).
        """
        logger.info("Khoi tao SentenceTransformer: %s", self.model_name)
        self.encoder = SentenceTransformer(self.model_name)
        self.encoder.max_seq_length = max_seq_length

        self.movies_df = df_movies.reset_index(drop=True).copy()
        
        # Tao chuoi van ban ngu nghia ket hop tieu de, the loai va cot truyen
        titles = self.movies_df["title"].fillna("").astype(str)
        genres = self.movies_df["genres_text"].fillna("").astype(str) if "genres_text" in self.movies_df.columns else ""
        overviews = self.movies_df["overview"].fillna("").astype(str) if "overview" in self.movies_df.columns else ""

        texts = []
        for t, g, o in zip(titles, genres, overviews):
            part_genre = f" Genres: {g}." if g else ""
            part_desc = f" Overview: {o}" if o else ""
            texts.append(f"{t}.{part_genre}{part_desc}".strip())

        logger.info("Dang ma hoa %d phim sang vector ngu nghia 384 chieu", len(texts))
        raw_embs = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        self.embeddings = np.asarray(raw_embs, dtype=np.float32)

        # Tao ban do anh xa tu movieId sang chi so dong
        self.movie_idx_map = {
            int(mid): idx for idx, mid in enumerate(self.movies_df["movieId"].values)
        }

        logger.info("Da hoan tat ma hoa. Kich thuoc ma tran vector: %s", str(self.embeddings.shape))
        return self

    def get_similar_movies(self, movie_id, n=10, exclude_self=True):
        """
        Tim n phim co do tuong dong ngu nghia cao nhat voi phim chi dinh.
        
        Tham so:
            movie_id: ID cua bo phim can tim tuong dong.
            n: So luong phim tra ve.
            exclude_self: Bo qua chinh bo phim dau vao neu la True.
            
        Tra ve:
            Danh sach tuple (movie_id, similarity_score, title).
        """
        if int(movie_id) not in self.movie_idx_map:
            return []

        idx = self.movie_idx_map[int(movie_id)]
        target_vec = self.embeddings[idx]

        scores = np.dot(self.embeddings, target_vec)

        if exclude_self:
            scores[idx] = -1.0

        top_indices = np.argsort(scores)[::-1][:n]
        
        results = []
        for i in top_indices:
            mid = int(self.movies_df.iloc[i]["movieId"])
            title = str(self.movies_df.iloc[i]["title"])
            sim = float(scores[i])
            results.append((mid, sim, title))
            
        return results

    def predict_rating(self, user_history, movie_id):
        """
        Du doan diem so cua nguoi dung danh cho mot bo phim dua tren vector ngu nghia.
        Diem du doan la trung binh co trong so cua cac phim user da thich trong lich su.
        
        Tham so:
            user_history: Dict dang {movieId: rating}.
            movie_id: ID bo phim can du doan.
            
        Tra ve:
            float: Diem rating du doan trong khoang [0.5, 5.0].
        """
        if int(movie_id) not in self.movie_idx_map or not user_history:
            return 3.5

        idx_target = self.movie_idx_map[int(movie_id)]
        target_vec = self.embeddings[idx_target]

        seen_indices = []
        seen_ratings = []
        for m_id, r in user_history.items():
            if int(m_id) in self.movie_idx_map:
                seen_indices.append(self.movie_idx_map[int(m_id)])
                seen_ratings.append(float(r))

        if not seen_indices:
            return 3.5

        # Tinh cosine similarity voi cac phim da xem
        sims = np.dot(self.embeddings[seen_indices], target_vec)

        # Chi xet cac phim co tuong dong duong
        pos_mask = sims > 0.05
        if not np.any(pos_mask):
            return 3.5

        weights = sims[pos_mask]
        ratings = np.array(seen_ratings, dtype=np.float32)[pos_mask]
        total_weight = float(np.sum(weights))

        if total_weight <= 1e-6:
            return 3.5

        pred = float(np.dot(weights, ratings) / total_weight)
        return float(np.clip(pred, config.MIN_RATING, config.MAX_RATING))

    def save(self, model_dir):
        """Luu vector va metadata vao thu muc models."""
        os.makedirs(model_dir, exist_ok=True)
        np.save(os.path.join(model_dir, "semantic_embeddings.npy"), self.embeddings)
        self.movies_df.to_parquet(os.path.join(model_dir, "semantic_movies.parquet"), index=False)
        joblib.dump(self.movie_idx_map, os.path.join(model_dir, "semantic_movie_idx.pkl"))
        logger.info("Da luu Semantic Recommender vao thu muc: %s", model_dir)

    @classmethod
    def load(cls, model_dir, model_name="all-MiniLM-L6-v2"):
        """Doc vector va metadata da duoc ma hoa tu truoc."""
        instance = cls(model_name=model_name)
        instance.embeddings = np.load(os.path.join(model_dir, "semantic_embeddings.npy"))
        instance.movies_df = pd.read_parquet(os.path.join(model_dir, "semantic_movies.parquet"))
        instance.movie_idx_map = joblib.load(os.path.join(model_dir, "semantic_movie_idx.pkl"))
        logger.info("Da nap thanh cong Semantic Recommender voi %d phim", len(instance.movies_df))
        return instance
