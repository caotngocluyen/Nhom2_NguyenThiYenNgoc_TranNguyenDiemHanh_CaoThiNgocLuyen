# Movie Recommendation System - He Thong Goi Y Phim

Mon hoc: Khai pha du lieu (Data Mining)
Dataset: The Movies Dataset (MovieLens 26M + TMDB Metadata)
Ngon ngu: Python 3.11+


## 1. Tong quan kien truc he thong

He thong ap dung kien truc tieu chuan cong nghiep cua YouTube va Netflix voi cac cai tien dot pha:

1. **Kien truc Hai giai doan (Two-Stage Candidate Retrieval + Re-ranking):**
   - **Giai doan 1 (Retrieval):** Thu hoi nhanh 150 - 200 ung vien tiem nang chua xem tu 3 nguon bo tro: SVD (CF), Semantic Content (Sentence-Transformers) va Popularity.
   - **Giai doan 2 (Re-ranking):** Trich xuat 11 dac trung phi tuyen tinh cho tung cap nguoi dung - phim, sau do su dung mo hinh cay tang cuong LightGBM toi uu truc tiep theo ham muc tieu **LambdaRank** (Listwise NDCG Optimization).

2. **Dac trung Trung khop the loai (Genre Affinity Feature - Cai tien A):**
   - Xay dung ho so phan phoi the loai ua thich cua nguoi dung ($P_u$) dua tren lich su danh gia tich cuc.
   - Tinh toan dac trung `genre_match_score` giua phan phoi gu phim cua user va the loai cua bo phim ung vien.
   - Tro thanh dac trung co do quan trong dung thu 2 toan he thong, tao tin hieu tuong minh cuc manh cho qua trinh xep hang.

3. **Toi uu thu hang danh sach voi LambdaRank (Cai tien C):**
   - Chuyen doi ham muc tieu tu Hoi quy diem so (`objective="regression"`) sang Xep hang danh sach (`objective="lambdarank"`).
   - Truc tiep toi uu hoa chi so NDCG@10 thong qua cac gradient cap/danh sach (pairwise/listwise), giup do chinh xac Top-3 va Top-10 tang truong vuot bac.

4. **Giai thuat Da dang hoa MMR o khau xuat ket qua (Maximal Marginal Relevance - Cai tien D):**
   - Ap dung MMR giai quyet triet de tinh trang de bi trung lap series/the loai tuong dong (vi du loat phim Toy Story hay loat phim Marvel).
   - Cong thuc:
     $$\text{MMR} = \arg\max_{m \in C} \left[ \lambda \cdot \text{Score}_{\text{LightGBM}}(u, m) - (1 - \lambda) \cdot \max_{m_j \in \text{Selected}} \text{Sim}_{\text{Semantic}}(m, m_j) \right]$$
   - Can bang hai hoa giua do phu hop (Relevance, $\lambda = 0.7$) va do moi la, bat ngo (Serendipity, $1 - \lambda = 0.3$).

5. **Trong so thich ung dong ca nhan hoa (Adaptive Alpha):**
   - Thay the trong so co dinh bang ham phi tuyen tinh theo so luong tuong tac cua tung user.
   - Nguoi dung moi nhan alpha thap (0.25) de uu tien noi dung va phim pho bien (giai quyet cold-start). Nguoi dung ky cuu tang alpha len 0.88 de toi da hoa so thich cong dong tu SVD.

6. **Bieu dien ngu nghia sau (Semantic Embeddings):**
   - Su dung Sentence-Transformers (`all-MiniLM-L6-v2`) ma hoa tieu de, the loai va cot truyen thanh vector 384 chieu chuan hoa L2, ho tro tinh cosine similarity tuc thi qua tich vo huong.


## 2. Cau truc thu muc du an

```
anti/
├── config.py                      # Cau hinh duong dan va tham so toan he thong
├── requirements.txt               # Danh sach thu vien can thiet
├── README.md                      # Tai lieu huong dan du an
│
├── data/
│   ├── ratings.csv                # Du lieu danh gia goc
│   ├── movies_metadata.csv        # Metadata thong tin phim goc
│   ├── keywords.csv               # Tu khoa phim
│   ├── credits.csv                # Dien vien va dao dien
│   ├── links.csv                  # Bang anh xa ID MovieLens va TMDB
│   └── processed/                 # Du lieu da lam sach va phan chia
│       ├── ratings_clean.parquet  # Ratings da lam sach
│       ├── ratings_kcore.parquet  # Ratings loc K-Core chong nhieu
│       ├── movies_clean.parquet   # Thong tin phim sach
│       ├── train_improved.parquet # Tap train (Per-User Split)
│       ├── val_improved.parquet   # Tap validation
│       └── test_improved.parquet  # Tap test danh gia doc lap
│
├── src/
│   ├── 01_check_data.py           # Kiem tra tinh toan ven cua du lieu tho
│   ├── 02_preprocess.py           # Tien xu ly va lam sach du lieu
│   ├── 04_prepare_data.py         # Phan chia tap train/val/test
│   ├── 05_train_baseline.py       # Mo hinh pho bien Popularity Baseline
│   ├── 06_train_cf.py             # Collaborative Filtering SVD cai tien
│   ├── 07_train_content.py        # Content-Based Filtering TF-IDF truyen thong
│   ├── 07_train_semantic.py       # Trich xuat Semantic Embeddings 384 chieu
│   ├── 08_train_hybrid.py         # Mo hinh lai Hybrid Fixed va Adaptive Alpha
│   ├── 08_train_reranker.py       # Huan luyen LightGBM LambdaRank Re-ranker
│   ├── 09_tune_models.py          # Toi uu hoa sieu tham so mo hinh
│   ├── 10_evaluate.py             # Danh gia toan dien tat ca cac mo hinh
│   ├── 11_recommend.py            # Goi y phim thuc te kem MMR da dang hoa
│   ├── adaptive_alpha.py          # Module tinh toan trong so alpha thich ung
│   ├── semantic_embeddings.py     # Module nhung ngu nghia sau
│   ├── two_stage_recommender.py   # Module kien truc Two-Stage Recommender (ho tro MMR)
│   ├── train_content_module.py    # Module Content-Based TF-IDF
│   └── update_all_reports.py      # Dong bo toan bo bao cao va bieu do
│
├── models/                        # File luu tru mo hinh da huan luyen
│   ├── svd_improved.pkl           # Trong so SVD Matrix Factorization (75 MB)
│   ├── semantic_embeddings.npy    # Ma tran vector ngu nghia 21,273 phim (31 MB)
│   ├── semantic_movies.parquet    # Metadata phim tuong ung voi vector
│   ├── semantic_movie_idx.pkl     # Bang anh xa movieId sang index vector
│   ├── reranker_lgbm.pkl          # Mo hinh LightGBM LambdaRank Re-ranker
│   ├── two_stage_metadata.pkl     # Thong ke va metadata bo tro cho Two-Stage
│   ├── popularity_model.pkl       # Mo hinh Popularity
│   └── best_config.json           # Cau hinh mo hinh tot nhat
│
├── outputs/                       # Bang so lieu dau ra
│   ├── final_metrics.parquet      # Bang tong hop chi so danh gia
│   ├── two_stage_feature_importance.parquet # Bang diem tam quan trong dac trung
│   └── recommendations_user_1.txt # Mau goi y thuc te cho nguoi dung
│
└── reports/                       # Nhat ky va bao cao danh gia
    ├── final_evaluation_report.txt# Bao cao tong hop ket qua danh gia
    ├── improved_evaluation_results.txt # Bang so sanh cac the he mo hinh
    ├── 07_semantic_training.log   # Nhat ky ma hoa vector ngu nghia
    ├── 08_reranker_training.log   # Nhat ky huan luyen LightGBM Re-ranker
    ├── 10_evaluate.log            # Nhat ky danh gia tren tap test
    └── figures/                   # Bieu do truc quan hoa ket qua
        ├── 08_model_comparison.png      # So sanh Hit Rate@10 va NDCG@10
        ├── 09_feature_importance.png    # Do quan trong 11 dac trung LightGBM
        └── 10_adaptive_alpha_curve.png  # Duong cong trong so thich ung
```


## 3. Huong dan cai dat va chay he thong

### Cai dat thu vien

```powershell
pip install -r requirements.txt
```

### Thu tu chay pipeline

1. Huan luyen mo hinh SVD Collaborative Filtering:
```powershell
python src/06_train_cf.py
```

2. Ma hoa vector ngu nghia sau (Semantic Embeddings):
```powershell
python src/07_train_semantic.py
```

3. Huan luyen mo hinh LightGBM LambdaRank Re-ranker voi Genre Affinity:
```powershell
python src/08_train_reranker.py
```

4. Chay danh gia toan dien tren tap test:
```powershell
python src/10_evaluate.py
```

5. Chay goi y cho nguoi dung (tich hop san MMR Diversification):
```powershell
python src/11_recommend.py --user_id 1 --mmr_lambda 0.7
```

6. Cap nhat va dong bo toan bo bao cao, bieu do:
```powershell
python src/update_all_reports.py
```


## 4. Bang ket qua thuc nghiem tren tap Test

Giao thuc danh gia: Negative Sampling (1 Ground Truth Positive + 99 Unrated Negatives), xep hang Top-10.
Tap du lieu: MovieLens K-Core mo rong (min_movie = 10, min_user = 20), bao gom 21,158 phim va 24.77 trieu ratings.

| Mo hinh | Hit Rate@10 | NDCG@10 | Nhan xet |
| :--- | :---: | :---: | :--- |
| Content TF-IDF | 0.1833 | 0.0871 | Bi han che boi tu khoa nong va nhieu tren tap phim lon |
| Semantic (Sentence-BERT) | 0.1467 | 0.0591 | Nam bat cot truyen tot, can bo sung hanh vi nguoi dung |
| SVD (CF) | 0.3967 | 0.2328 | Khai thac tot tuong quan nguoi dung va phim tren tap lon |
| Hybrid Fixed (alpha=0.7) | 0.3867 | 0.2239 | Bi keo diem o nhom nguoi dung it tuong tac |
| Hybrid Adaptive Alpha | 0.3867 | 0.2436 | Cai thien nho trong so dong ca nhan hoa |
| **Two-Stage LightGBM (LambdaRank + Genre)** | **0.8300** | **0.5649** | **Hieu qua vuot troi: Hit@10 dat 83% va NDCG@10 dat 0.5649** |


## 5. Tam quan trong cua cac dac trung trong LightGBM (Feature Importance)

Do quan trong cua 11 dac trung sau khi huan luyen LambdaRank:

1. `movie_log_count`: Do pho bien va uy tin cua bo phim tren he thong (696)
2. `genre_match_score`: Muc do trung khop voi gu the loai cua nguoi dung (618) - **Cai tien A**
3. `diff_svd_movie`: Do lech giua du doan SVD va muc trung binh cua phim (292)
4. `movie_avg_rating`: Diem danh gia trung binh cua bo phim (274)
5. `semantic_score`: Diem tuong dong cot truyen va the loai tu Semantic Embeddings (271)
6. `diff_svd_user`: Do lech giua du doan SVD va muc trung binh cua user (244)
7. `user_log_count`: Muc do tich cuc cua nguoi dung trong he thong (216)
8. `user_avg_rating`: Muc do de tinh hoac khat khe cua nguoi dung (147)
9. `hybrid_score`: Diem ket hop thich ung giua CF va Semantic (122)
10. `svd_score`: Diem du doan tu mo hinh CF SVD goc (118)
11. `user_adaptive_alpha`: Trong so alpha ca nhan hoa (2)
