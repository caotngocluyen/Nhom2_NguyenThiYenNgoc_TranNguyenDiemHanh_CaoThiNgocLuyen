import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.adaptive_alpha import compute_adaptive_alpha
from src.semantic_embeddings import SemanticContentRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TwoStageRecommender:
    """
    He thong goi y hai giai doan (Two-Stage Recommendation Pipeline):
    Giai doan 1 (Retrieval): Thu hoi nhanh khoang 200 ung vien tu SVD, Semantic Content va Popularity.
    Giai doan 2 (Re-ranking): Trich xuat 11 dac trung phi tuyen tinh (bao gom Genre Affinity),
    xep hang lai bang LightGBM (LambdaRank) va da dang hoa MMR (Maximal Marginal Relevance).
    """

    def __init__(self, svd_model=None, semantic_model=None, reranker_model=None):
        self.svd_model = svd_model
        self.semantic_model = semantic_model
        self.reranker_model = reranker_model
        self.movie_stats = {}
        self.user_stats = {}
        self.popular_movies = []
        self.all_movie_ids = []
        self.movie_genres = {}

    def set_stats(self, movie_stats, user_stats, popular_movies, all_movie_ids, movie_genres=None):
        """Cap nhat thong ke tong hop cho phim va nguoi dung."""
        self.movie_stats = movie_stats
        self.user_stats = user_stats
        self.popular_movies = popular_movies
        self.all_movie_ids = all_movie_ids
        if movie_genres is not None:
            self.movie_genres = movie_genres

    def compute_user_genre_profile(self, user_history):
        """
        Thong ke phan phoi the loai ua thich cua user tu lich su danh gia tich cuc (rating >= 3.0).
        Diem rating cang cao thi dong gop trong so cang lon.
        """
        if not self.movie_genres or not user_history:
            return {}

        genre_weights = {}
        for m, r in user_history.items():
            if float(r) >= 3.0:
                gs = self.movie_genres.get(int(m), set())
                w = float(r) - 2.0
                for g in gs:
                    genre_weights[g] = genre_weights.get(g, 0.0) + w

        total_w = sum(genre_weights.values())
        if total_w > 0:
            return {g: w / total_w for g, w in genre_weights.items()}
        return {}

    def retrieve_candidates(self, user_id, user_history, n_candidates=200):
        """
        Giai doan 1: Thu hoi tap ung vien tiem nang tu nhieu nguon bo tro.
        
        Tham so:
            user_id: ID nguoi dung.
            user_history: Dict dang {movieId: rating}.
            n_candidates: Tong so ung vien can thu hoi (mac dinh 200).
            
        Tra ve:
            List danh sach movieId ung vien.
        """
        seen_movies = set(user_history.keys())
        candidates = set()

        # Nguon 1: Top 80 phim tu Popularity (dam bao do phu va pho bien)
        for m in self.popular_movies:
            if m not in seen_movies:
                candidates.add(m)
            if len(candidates) >= 80:
                break

        # Nguon 2: Top 80 phim tu Semantic Content Recommender (dua vao phim user thich)
        if self.semantic_model is not None and user_history:
            liked_movies = [m for m, r in user_history.items() if float(r) >= config.RELEVANT_RATING_THRESHOLD]
            if not liked_movies:
                liked_movies = list(user_history.keys())[:5]
            
            # Lay 5 phim user thich nhat de tim cac phim tuong dong ngu nghia
            for liked_id in liked_movies[:5]:
                sim_list = self.semantic_model.get_similar_movies(liked_id, n=20, exclude_self=True)
                for sim_mid, _, _ in sim_list:
                    if sim_mid not in seen_movies:
                        candidates.add(sim_mid)
                    if len(candidates) >= 150:
                        break

        # Nguon 3: Bo sung tu SVD neu chua du so luong ung vien
        if self.svd_model is not None and len(candidates) < n_candidates:
            remaining = [m for m in self.all_movie_ids if m not in seen_movies and m not in candidates]
            if remaining:
                sample_pool = np.random.choice(remaining, min(len(remaining), 300), replace=False)
                svd_scores = [(m, self.svd_model.predict(int(user_id), int(m)).est) for m in sample_pool]
                svd_scores.sort(key=lambda x: x[1], reverse=True)
                for m, _ in svd_scores:
                    candidates.add(m)
                    if len(candidates) >= n_candidates:
                        break

        return list(candidates)[:n_candidates]

    def extract_features(self, user_id, candidate_ids, user_history):
        """
        Giai doan 2: Trich xuat 11 dac trung phi tuyen cho tung cap nguoi dung va phim ung vien.
        Bao gom: SVD, Semantic, Thong ke Movie/User, Adaptive Alpha, Do lech va Genre Affinity.
        """
        n_interactions = len(user_history)
        u_stats = self.user_stats.get(user_id, {"avg_rating": 3.53, "count": n_interactions})
        user_avg = float(u_stats["avg_rating"])
        user_cnt = float(np.log1p(max(1, u_stats["count"])))
        alpha_u = compute_adaptive_alpha(n_interactions, min_alpha=0.25, max_alpha=0.88, threshold=40)

        # Tinh profile the loai cua user
        u_genre_profile = self.compute_user_genre_profile(user_history)

        feature_rows = []
        for mid in candidate_ids:
            # 1. Dac trung SVD
            svd_pred = float(self.svd_model.predict(int(user_id), int(mid)).est) if self.svd_model else 3.5

            # 2. Dac trung Semantic
            sem_pred = float(self.semantic_model.predict_rating(user_history, mid)) if self.semantic_model else 3.5

            # 3. Thong ke ve bo phim
            m_stats = self.movie_stats.get(mid, {"avg_rating": 3.2, "count": 10})
            movie_avg = float(m_stats["avg_rating"])
            movie_cnt = float(np.log1p(m_stats["count"]))

            # 4. Dac trung tuong tac phi tuyen
            diff_svd_movie = svd_pred - movie_avg
            diff_svd_user = svd_pred - user_avg
            hybrid_approx = alpha_u * svd_pred + (1.0 - alpha_u) * sem_pred

            # 5. Dac trung Trùng khớp Thể loại (Genre Affinity Feature)
            cand_genres = self.movie_genres.get(int(mid), set())
            if cand_genres and u_genre_profile:
                genre_match_score = float(sum(u_genre_profile.get(g, 0.0) for g in cand_genres) / len(cand_genres))
            else:
                genre_match_score = 0.0

            row = [
                svd_pred,
                sem_pred,
                movie_avg,
                movie_cnt,
                user_avg,
                user_cnt,
                alpha_u,
                diff_svd_movie,
                diff_svd_user,
                hybrid_approx,
                genre_match_score
            ]
            feature_rows.append(row)

        return np.asarray(feature_rows, dtype=np.float32)

    def rerank(self, user_id, candidate_ids, user_history, top_k=10, mmr_lambda=config.MMR_DEFAULT_LAMBDA):
        """
        Xep hang lai tap ung vien bang LightGBM (LambdaRank) va da dang hoa danh sach bang MMR.
        
        Tham so:
            user_id: ID nguoi dung.
            candidate_ids: Danh sach movieId can cham diem.
            user_history: Lich su danh gia cua user.
            top_k: So luong phim can xuat.
            mmr_lambda: He so can bang MMR (0.0 < lambda < 1.0).
                        Neu mmr_lambda is None hoac >= 1.0: sap xep theo diem LightGBM giam dan.
        
        Tra ve:
            List tuple (movie_id, predicted_score).
        """
        if not candidate_ids:
            return []

        features = self.extract_features(user_id, candidate_ids, user_history)

        if self.reranker_model is not None:
            scores = self.reranker_model.predict(features)
        else:
            # Fallback ve hybrid_approx (cot thu 10) neu chua nap reranker
            scores = features[:, -2]

        raw_scores = np.asarray(scores, dtype=np.float32)

        # Neu khong dung MMR hoac khong co semantic embeddings thi tra ve theo thu tu diem
        if mmr_lambda is None or mmr_lambda >= 1.0 or self.semantic_model is None or top_k <= 1:
            ranked_indices = np.argsort(raw_scores)[::-1][:top_k]
            return [(candidate_ids[i], float(raw_scores[i])) for i in ranked_indices]

        # Giai thuat Da dang hoa MMR (Maximal Marginal Relevance)
        s_min, s_max = float(raw_scores.min()), float(raw_scores.max())
        norm_scores = (raw_scores - s_min) / (s_max - s_min) if s_max > s_min else np.ones_like(raw_scores)

        # Trich xuat vector ngu nghia 384 chieu cua tap ung vien
        emb_list = []
        valid_embs = []
        for mid in candidate_ids:
            if int(mid) in self.semantic_model.movie_idx_map:
                emb_list.append(self.semantic_model.embeddings[self.semantic_model.movie_idx_map[int(mid)]])
                valid_embs.append(True)
            else:
                emb_list.append(np.zeros(config.SEMANTIC_EMBEDDING_DIM, dtype=np.float32))
                valid_embs.append(False)
        cand_embs = np.vstack(emb_list)

        # Chon item dau tien co diem relevance cao nhat
        first_idx = int(np.argmax(norm_scores))
        selected_indices = [first_idx]
        pool_indices = [i for i in range(len(candidate_ids)) if i != first_idx]

        # Lan luot lua chon cac item tiep theo theo cong thuc MMR
        while len(selected_indices) < min(top_k, len(candidate_ids)) and pool_indices:
            sel_embs = cand_embs[selected_indices]
            best_idx = None
            best_mmr_score = -float("inf")

            for c_idx in pool_indices:
                relevance = norm_scores[c_idx]
                if valid_embs[c_idx]:
                    sim_max = float(np.max(np.dot(sel_embs, cand_embs[c_idx])))
                else:
                    sim_max = 0.0

                mmr_val = mmr_lambda * relevance - (1.0 - mmr_lambda) * sim_max
                if mmr_val > best_mmr_score:
                    best_mmr_score = mmr_val
                    best_idx = c_idx

            if best_idx is None:
                best_idx = pool_indices[0]

            selected_indices.append(best_idx)
            pool_indices.remove(best_idx)

        results = [(candidate_ids[i], float(raw_scores[i])) for i in selected_indices]
        return results
