"""Patch 10_evaluate.py to define PopularityRecommender so unpickling works cleanly."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('src/10_evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define PopularityRecommender right after imports
marker = 'from train_content_module import ContentBasedRecommender'

popularity_def = '''from train_content_module import ContentBasedRecommender

class PopularityRecommender:
    """Mô hình Popularity để unpickle và sử dụng."""
    def __init__(self, min_ratings=50):
        self.min_ratings = min_ratings
        self.popularity_scores = None
        self.global_mean = None
        self.user_seen_movies = None

    def recommend(self, user_id, n=10, exclude_seen=True):
        candidates = self.popularity_scores.copy()
        if exclude_seen and self.user_seen_movies and user_id in self.user_seen_movies:
            seen = self.user_seen_movies[user_id]
            candidates = candidates[~candidates["movieId"].isin(seen)]
        return candidates.head(n)[["movieId", "bayesian_score"]].copy()

    def predict_rating(self, df):
        score_dict = dict(zip(
            self.popularity_scores["movieId"],
            self.popularity_scores["bayesian_score"]
        ))
        preds = df["movieId"].map(score_dict).fillna(self.global_mean)
        return preds.values.astype(np.float32)

# Đăng ký vào __main__ để pickle tải đúng
sys.modules['__main__'].PopularityRecommender = PopularityRecommender
'''

if marker in content:
    content = content.replace(marker, popularity_def)
    print("Added PopularityRecommender definition to 10_evaluate.py")
else:
    print("marker not found")

with open('src/10_evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Saved 10_evaluate.py")
