import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def check_ground_truth(user_id=8659):
    df_test = pd.read_parquet(config.TEST_IMPROVED_FILE)
    df_movies = pd.read_parquet(config.MOVIES_CLEAN_FILE, columns=["movieId", "title", "genres_text"])

    user_test = df_test[df_test["userId"] == user_id].merge(df_movies, on="movieId")
    test_dict = dict(zip(user_test["movieId"], user_test["rating"]))
    title_dict = dict(zip(user_test["movieId"], user_test["title"]))

    print("=" * 70)
    print(f"KIEM TRA PHIM GOI Y SO VOI THUC TE NGUOI DUNG: {user_id}")
    print("=" * 70)
    print(f"Tong so phim nguoi dung da xem va danh gia trong tap TEST: {len(user_test)}")
    pos_test = user_test[user_test["rating"] >= config.RELEVANT_RATING_THRESHOLD]
    print(f"So phim thuc te danh gia TICH CUC (rating >= 3.5 - Ground Truth): {len(pos_test)}")
    print(f"Top cac phim nguoi dung thich nhat trong tap TEST (Tuong lai):")
    for _, row in pos_test.sort_values(by="rating", ascending=False).head(10).iterrows():
        print(f"   * {row['rating']:.1f} sao: {row['title']} ({row['genres_text']})")

    # Danh sach goi y tu demo cua ban:
    models_recs = {
        "SVD (CF)": [
            "Marketa Lazarová", "Between the Folds", "Dylan Moran: What It Is",
            "20th Century Women", "High School", "Simon", "The Cove",
            "The Boy and the Beast", "Mommy", "The Gatekeepers"
        ],
        "Semantic Content": [
            "Mr. Woodcock", "Woody Allen: A Documentary", "Rubin and Ed",
            "Woodlawn", "Woodlawn", "The Editor", "Tales From the Script", "The Forest for the Trees"
        ],
        "Hybrid Adaptive (alpha=0.88)": [
            "Marketa Lazarová", "Between the Folds", "Dylan Moran: What It Is",
            "20th Century Women", "High School", "Simon", "The Cove",
            "The Boy and the Beast", "Mommy", "The Gatekeepers"
        ],
        "Two-Stage LightGBM (LambdaRank)": [
            "Saving Private Ryan", "The Dark Knight",
            "Dr. Strangelove or: How I Learned to Stop Worrying",
            "Million Dollar Baby", "Juno", "The Hangover",
            "Slumdog Millionaire", "Django Unchained",
            "Inglourious Basterds", "Shutter Island"
        ],
        "Two-Stage LightGBM + MMR Diversified": [
            "Saving Private Ryan", "The Dark Knight",
            "Million Dollar Baby",
            "Dr. Strangelove or: How I Learned to Stop Worrying",
            "Juno", "The King's Speech", "The Hangover",
            "Django Unchained", "WALL·E", "Slumdog Millionaire"
        ]
    }

    movie_title_to_id = dict(zip(df_movies["title"], df_movies["movieId"]))

    print("\n" + "=" * 70)
    print("DOI CHIEU DANH SACH GOI Y TOP-10 VOI TAP TEST (GROUND TRUTH):")
    print("=" * 70)

    for model_name, rec_titles in models_recs.items():
        print(f"\nMo hinh: {model_name}")
        hits = []
        for rank, title in enumerate(rec_titles, 1):
            mid = movie_title_to_id.get(title)
            # Tim theo tieu de hoac chua trong test
            matching_test = user_test[user_test["title"] == title]
            if len(matching_test) > 0:
                actual_rating = matching_test.iloc[0]["rating"]
                is_hit = actual_rating >= config.RELEVANT_RATING_THRESHOLD
                hits.append((rank, title, actual_rating, is_hit))
                status = "CHUAN XAC (HIT!)" if is_hit else "CO XEM (Rating thap)"
                print(f"  [X] Top {rank:2d}: {title:<45} -> Thuc te: {actual_rating:.1f} sao [{status}]")
            else:
                print(f"  [ ] Top {rank:2d}: {title:<45} -> Chua tung xem trong tap Test")

        if not hits:
            print("  --> Ket qua: 0 phim trung voi tap Test!")
        else:
            hit_count = sum(1 for h in hits if h[3])
            print(f"  --> Tong so phim trung khop dung (rating >= 3.5): {hit_count}/{len(rec_titles)} phim!")

if __name__ == "__main__":
    check_ground_truth(8659)
