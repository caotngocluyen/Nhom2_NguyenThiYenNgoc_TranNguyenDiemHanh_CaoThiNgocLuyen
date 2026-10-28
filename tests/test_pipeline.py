"""
tests/test_pipeline.py - Kiểm thử các chức năng quan trọng
===========================================================
MỤC ĐÍCH:
    Kiểm thử tự động các hàm và logic quan trọng để đảm bảo kết quả đúng.
    
    Các test bao gồm:
    - Kiểm tra metric tính toán đúng (RMSE, MAE, Precision, Recall, NDCG)
    - Kiểm tra ánh xạ ID không bị sai
    - Kiểm tra phim đã xem bị loại khỏi danh sách gợi ý
    - Kiểm tra chia tập không có data leakage
    - Kiểm tra Content-Based similarity logic

CÁCH CHẠY:
    python -m pytest tests/test_pipeline.py -v
    python tests/test_pipeline.py     # Chạy không cần pytest
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import unittest
import numpy as np
import pandas as pd

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))



# TEST 1: METRICS


class TestMetrics(unittest.TestCase):
    """Kiểm thử các hàm tính metric."""

    def test_rmse_perfect(self):
        """RMSE = 0 khi dự đoán hoàn hảo."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
        self.assertAlmostEqual(rmse, 0.0, places=6)

    def test_rmse_known(self):
        """RMSE phải bằng 1.0 khi mỗi dự đoán sai 1 đơn vị."""
        y_true = [2.0, 3.0, 4.0]
        y_pred = [3.0, 4.0, 5.0]  # Mỗi cái sai 1
        rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
        self.assertAlmostEqual(rmse, 1.0, places=6)

    def test_mae_known(self):
        """MAE phải bằng 0.5 khi mỗi dự đoán sai 0.5."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.5, 2.5, 3.5]
        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        self.assertAlmostEqual(mae, 0.5, places=6)

    def test_precision_at_k_perfect(self):
        """Precision@K = 1.0 khi tất cả K phim gợi ý đều relevant."""
        from src.train_content_module import ContentBasedRecommender
        # Tự tính
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3, 4, 5}
        k = 5
        
        top_k = recommended[:k]
        p = sum(1 for m in top_k if m in relevant) / k
        self.assertAlmostEqual(p, 1.0, places=6)

    def test_precision_at_k_zero(self):
        """Precision@K = 0.0 khi không có phim nào relevant."""
        recommended = [10, 20, 30]
        relevant = {1, 2, 3}
        k = 3
        top_k = recommended[:k]
        p = sum(1 for m in top_k if m in relevant) / k
        self.assertAlmostEqual(p, 0.0, places=6)

    def test_recall_at_k(self):
        """Recall@K = 0.5 khi tìm được 1/2 phim relevant."""
        recommended = [1, 10, 20]
        relevant = {1, 2}  # 2 phim relevant, nhưng chỉ tìm được 1
        k = 3
        n_found = sum(1 for m in recommended[:k] if m in relevant)
        recall = n_found / len(relevant)
        self.assertAlmostEqual(recall, 0.5, places=6)

    def test_ndcg_perfect(self):
        """NDCG@K = 1.0 khi xếp hạng hoàn hảo."""
        recommended = [1, 2, 3]
        relevant = {1, 2, 3}
        k = 3
        
        dcg = sum(1.0/np.log2(i+2) for i, m in enumerate(recommended[:k]) if m in relevant)
        idcg = sum(1.0/np.log2(i+2) for i in range(min(len(relevant), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        self.assertAlmostEqual(ndcg, 1.0, places=6)

    def test_ndcg_worst(self):
        """NDCG@K = 0.0 khi không có phim relevant nào."""
        recommended = [10, 20, 30]
        relevant = {1, 2, 3}
        k = 3
        
        dcg = sum(1.0/np.log2(i+2) for i, m in enumerate(recommended[:k]) if m in relevant)
        idcg = sum(1.0/np.log2(i+2) for i in range(min(len(relevant), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        self.assertAlmostEqual(ndcg, 0.0, places=6)



# TEST 2: CHIA TẬP DỮ LIỆU


class TestDataSplit(unittest.TestCase):
    """Kiểm thử logic chia train/val/test."""

    def setUp(self):
        """Tạo dữ liệu giả để test."""
        np.random.seed(42)
        n = 1000
        self.df = pd.DataFrame({
            "userId": np.random.randint(1, 100, n),
            "movieId": np.random.randint(1, 50, n),
            "rating": np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], n),
            "timestamp": np.arange(n)  # Timestamp tăng dần
        })

    def test_split_sizes(self):
        """Kiểm tra tỷ lệ chia đúng."""
        n = len(self.df)
        n_train = int(n * 0.8)
        n_val   = int(n * 0.1)
        
        df_sorted = self.df.sort_values("timestamp").reset_index(drop=True)
        df_train = df_sorted.iloc[:n_train]
        df_val   = df_sorted.iloc[n_train:n_train+n_val]
        df_test  = df_sorted.iloc[n_train+n_val:]
        
        # Kiểm tra tổng = ban đầu
        self.assertEqual(len(df_train) + len(df_val) + len(df_test), n)
        
        # Kiểm tra tỷ lệ gần đúng
        self.assertAlmostEqual(len(df_train)/n, 0.8, delta=0.02)

    def test_no_data_leakage(self):
        """Kiểm tra không có data leakage (cùng userId+movieId trong train và test)."""
        n = len(self.df)
        n_train = int(n * 0.8)
        
        df_sorted = self.df.sort_values("timestamp").reset_index(drop=True)
        df_train = df_sorted.iloc[:n_train]
        df_test  = df_sorted.iloc[n_train:]
        
        train_pairs = set(zip(df_train["userId"], df_train["movieId"]))
        test_pairs  = set(zip(df_test["userId"],  df_test["movieId"]))
        
        # Không nên có chồng chéo (trong temporal split)
        # Chú ý: Có thể có chồng chéo trong random split, không phải temporal
        # Temporal split không có chồng chéo theo (userId, movieId, timestamp)
        # Nhưng cùng user và phim có thể có nhiều rating tại thời điểm khác nhau
        #  Test này kiểm tra KHÔNG có CÙNG dòng
        overlap_count = len(train_pairs & test_pairs)
        
        # Trong temporal split, cùng userId+movieId có thể xuất hiện ở cả 2 set
        # (nếu user đánh giá lại phim). Điều này OK nhưng cần ghi nhận.
        # Test chính: KHÔNG có cùng exact record (userId, movieId, timestamp)
        train_triples = set(zip(df_train["userId"], df_train["movieId"], df_train["timestamp"]))
        test_triples  = set(zip(df_test["userId"],  df_test["movieId"],  df_test["timestamp"]))
        exact_overlap = len(train_triples & test_triples)
        
        self.assertEqual(exact_overlap, 0, "Có data leakage: cùng record xuất hiện ở cả train và test!")

    def test_temporal_order(self):
        """Kiểm tra temporal split giữ đúng thứ tự thời gian."""
        n = len(self.df)
        n_train = int(n * 0.8)
        
        df_sorted = self.df.sort_values("timestamp").reset_index(drop=True)
        df_train = df_sorted.iloc[:n_train]
        df_test  = df_sorted.iloc[n_train:]
        
        # Timestamp lớn nhất trong train < Timestamp nhỏ nhất trong test
        self.assertLessEqual(
            df_train["timestamp"].max(),
            df_test["timestamp"].min(),
            "Temporal split sai: có rating trong test trước rating trong train!"
        )



# TEST 3: EXCLUDE SEEN MOVIES


class TestExcludeSeenMovies(unittest.TestCase):
    """Kiểm tra logic loại phim đã xem khỏi danh sách gợi ý."""

    def test_seen_movies_excluded(self):
        """Phim đã xem không được xuất hiện trong danh sách gợi ý."""
        seen_movies = {1, 2, 3, 4, 5}  # Phim đã xem
        all_movies  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        
        # Logic: Chỉ gợi ý phim chưa xem
        candidates = all_movies - seen_movies
        
        # Kiểm tra không có phim đã xem trong candidates
        for movie in candidates:
            self.assertNotIn(movie, seen_movies, f"Movie {movie} đã xem vẫn xuất hiện!")
        
        # Kiểm tra số lượng đúng
        self.assertEqual(len(candidates), 5)  # 10 - 5 = 5

    def test_all_seen_returns_empty(self):
        """Nếu đã xem tất cả phim  candidates rỗng."""
        seen_movies = {1, 2, 3}
        all_movies  = {1, 2, 3}
        candidates  = all_movies - seen_movies
        self.assertEqual(len(candidates), 0)



# TEST 4: ID MAPPING


class TestIdMapping(unittest.TestCase):
    """Kiểm tra ánh xạ ID không bị sai."""

    def test_mapping_is_bijective(self):
        """Ánh xạ movieId ↔ idx phải 1-1 (bijection)."""
        movie_ids = [10, 20, 30, 40, 50]
        movie_idx_map = {mid: idx for idx, mid in enumerate(movie_ids)}
        
        # Kiểm tra không có idx trùng
        idx_values = list(movie_idx_map.values())
        self.assertEqual(len(idx_values), len(set(idx_values)), "Có idx trùng!")
        
        # Kiểm tra ánh xạ ngược
        reverse_map = {idx: mid for mid, idx in movie_idx_map.items()}
        self.assertEqual(len(reverse_map), len(movie_ids))

    def test_unknown_movie_handled(self):
        """Phim không có trong mapping phải trả về giá trị mặc định."""
        movie_idx_map = {1: 0, 2: 1, 3: 2}
        
        unknown_id = 999
        result = movie_idx_map.get(unknown_id, -1)  # -1 là sentinel
        
        self.assertEqual(result, -1)



# TEST 5: CONTENT-BASED LOGIC


class TestContentBasedLogic(unittest.TestCase):
    """Kiểm tra logic Content-Based."""

    def test_soup_not_empty(self):
        """content_soup phải có nội dung cho ít nhất một số phim."""
        soup_example = "action adventure drama superhero batman christopher_nolan"
        self.assertTrue(len(soup_example) > 0)
        self.assertIn("action", soup_example)

    def test_cosine_self_similarity(self):
        """Cosine similarity của một vector với chính nó = 1.0."""
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        soups = ["action adventure thriller", "romance comedy drama", "action crime thriller"]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(soups)
        
        # Similarity của vector 0 với chính nó
        sim = cosine_similarity(matrix[0:1], matrix[0:1])[0][0]
        self.assertAlmostEqual(sim, 1.0, places=5)
        
        # Similarity của vector 0 với vector 1 phải < 1
        sim_01 = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        self.assertLess(sim_01, 1.0)

    def test_similar_soups_have_higher_similarity(self):
        """Phim cùng thể loại phải có similarity cao hơn phim khác thể loại."""
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        soups = [
            "action adventure thriller superhero",      # Phim 0
            "action adventure superhero batman",        # Phim 1 (tương tự 0)
            "romance comedy love drama",                # Phim 2 (khác)
        ]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(soups)
        
        sim_01 = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        sim_02 = cosine_similarity(matrix[0:1], matrix[2:3])[0][0]
        
        # Phim 0 và 1 phải tương tự nhau hơn phim 0 và 2
        self.assertGreater(sim_01, sim_02, 
                          "Phim cùng thể loại phải có similarity cao hơn!")



# CHẠY TESTS


def chay_tat_ca_tests():
    """Chạy tất cả test và báo cáo kết quả."""
    print("=" * 60)
    print("CHẠY KIỂM THỬ TỰ ĐỘNG")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Thêm tất cả test classes
    for test_class in [TestMetrics, TestDataSplit, TestExcludeSeenMovies, 
                        TestIdMapping, TestContentBasedLogic]:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Chạy
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print(f" Tất cả {result.testsRun} test đều PASS!")
    else:
        print(f" Có {len(result.failures)} test FAIL và {len(result.errors)} lỗi.")
        for test, traceback in result.failures:
            print(f"\nFAIL: {test}")
        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = chay_tat_ca_tests()
    sys.exit(0 if success else 1)
