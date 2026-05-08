# Music Recommender System
## Project Overview
The projest target is to estimate a number of distance-based metrics on embeddings of different origins and sizes.

The illustrative experiment is a music recommendation system which uses content-based, collaborative filtering and both of these approaches at once. 

Each approach produces different kinds of embeddings that represent songs/users to then compare performances of different distance' functions applied to them as well as built-in library estimators.

## Embeddings' origins

### Content-based - Raw audio features vectors
Content-based recommender using audio features:                                       
              
1. Load — reads 600k interactions from interactions_audio_v2.parquet (24 cols: user/song ids, play_count, audio features, genre).       
2. Feature engineering — one-hot of genre_root (multi-label via str.get_dummies) and key; z-score scaling of numeric audio features     
(acousticness, danceability, energy, tempo, valence, …) → 49-dim feature space.                                                         
3. Train/test split — per-user 80/20 (489,694 / 110,306, 22,328 test users).                                                      
4. User embeddings — weighted mean of song feature vectors per user, weighted by play_count.
5. Distance functions — cosine_similarity, euclidean_dist, manhattan_dist, chebyshev.                             
6. Top-K recommendation — cosine scores, mask train-seen as ±inf, argpartition for top-10.                                              
7. First comparison table — 8 methods (cos/euclid/manhattan/chebyshev × raw/norm). Best: cosine ≈ P@10 0.0031.                          
8. Second comparison via utils.distance_metrics.evaluate_all — 7 metrics (dot, cos, l2, l1, linf, lp_0.5, maha) × {raw, normalized}                                                                                          
### Collaborative filtering (CF) - Implicit ALS
This method only uses the data with users' activity and creates a latent vector for each user to then compare them using cosine similarity.
1. Load — reads 600k rows (user_id, song_id, play_count) from interactions_audio_v2.1.parquet (~21,781 users).
2. Train/test split — random 80/20 mask on rows; CSR built on train only with confidence weights 1 + 40 * play_count.
3. Train ALS — implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=15).
4. Learn latent embeddings — the model produces two dense matrices:
user_factors ∈ R^(num_users × k)
item_factors ∈ R^(num_items × k)
5. Manual eval @K=10 — loops over test users, calls model.recommend, computes Precision/Recall/NDCG:
  - Precision@10 = 0.0507, Recall@10 = 0.1109, NDCG@10 = 0.0948.
6. Distance-metric comparison — extracts latent embeddings, runs utils.distance_metrics evaluate_all on raw and L2-normalized embeddings across dot, cos, l2, l1, linf, lp_0.5, maha.

### CF + Content-based filtering - LightFM
LightFM hybrid recommender pipeline on an audio interactions dataset:                                                        
1. Load — reads 300k rows from interactions_audio_v2.1.parquet (10,811 users × 14,283 songs).                                           
2. Index & split — encodes user_id/song_id to integer codes, then per-user 80/20 train/test split (244,329 / 55,671).                   
3. Build CSR matrices — binary user×item interaction matrices for train/test.                                                           
4. Build item-feature matrix — each song is represented by
  normalized continuous audio features:
  - danceability,
  - energy,
  - acousticness,
  - tempo,
  - valence,
  - loudness,
  - speechiness,
  - instrumentalness,
  - liveness,
  - duration,
  etc.
  one-hot categorical metadata:
  - genre_root,
  - key,
  - mode,
  - timeSignature.
                                                                       
5. Train LightFM — WARP loss, 32 components, 15 epochs, with item_features.
6. Distance-metric comparison — extracts user embeddings and weighted-sum song embeddings, then evaluates 7 similarity measures (dot,
cos, l2, l1, linf, lp_0.5, maha) in both raw and L2-normalized form via utils.distance_metrics.evaluate_all.

### Distance metrics used

All metrics below are evaluated by `utils.distance_metrics.evaluate_all` on both raw and L2-normalized embeddings. For ranking, similarities are sorted descending and distances ascending; train-seen items are masked out before taking top-K.

- **dot** — inner product `⟨u, v⟩ = Σ uᵢ vᵢ`. Similarity (higher is better). Sensitive to vector magnitude, so popular/longer vectors get boosted; this is the native scoring used by ALS and LightFM during training.
- **cos** — cosine similarity `⟨u, v⟩ / (‖u‖₂ ‖v‖₂)`. Similarity in [-1, 1]. Magnitude-invariant — measures only the angle between vectors. Equivalent to `dot` once both sides are L2-normalized.
- **l2** — Euclidean distance `‖u − v‖₂ = √Σ (uᵢ − vᵢ)²`. Distance (lower is better). Standard geometric distance; on the unit sphere it is monotonically equivalent to cosine, which is why `dot`, `cos`, and `l2` collapse to identical rankings after L2 normalization.
- **l1** — Manhattan / taxicab distance `Σ |uᵢ − vᵢ|`. Distance. More robust to outliers in individual coordinates than `l2` because differences are not squared.
- **linf** — Chebyshev distance `maxᵢ |uᵢ − vᵢ|`. Distance. Determined by the single worst-matching coordinate; tends to be the weakest ranker here because it discards information from all other dimensions.
- **lp_0.5** — fractional Minkowski distance `(Σ |uᵢ − vᵢ|^0.5)^(1/0.5)`. Distance. With p < 1 the unit ball is non-convex and small differences are amplified relative to large ones; sometimes argued to behave better than `l2` in high dimensions, but here it underperforms on raw ALS/LightFM factors.
- **maha** — Mahalanobis distance `√((u − v)ᵀ Σ⁻¹ (u − v))`, where `Σ` is the (diagonal-regularized) covariance of the embedding matrix. Distance. Whitens the space so each principal direction contributes equally — useful when feature scales/variances differ a lot, as in the 49-dim raw audio feature space.

### Results

Top-10 evaluation across embedding sources and distance metrics. Each cell shows Precision@10 / Recall@10 / NDCG@10.

#### Raw embeddings

| Metric | Content (audio, 49d) | ALS k=8 | ALS k=50 | LightFM |
|--------|----------------------|---------|----------|---------|
| dot    | 0.0016 / 0.0038 / 0.0027 | 0.0242 / 0.0501 / 0.0419 | 0.0531 / 0.1163 / 0.0992 | 0.0121 / 0.0231 / 0.0198 |
| cos    | 0.0031 / 0.0080 / 0.0061 | 0.0169 / 0.0403 / 0.0308 | 0.0513 / 0.1170 / 0.1005 | 0.0108 / 0.0208 / 0.0179 |
| l2     | 0.0015 / 0.0039 / 0.0029 | 0.0242 / 0.0501 / 0.0419 | 0.0533 / 0.1166 / 0.0995 | 0.0039 / 0.0072 / 0.0061 |
| l1     | 0.0015 / 0.0043 / 0.0029 | 0.0158 / 0.0323 / 0.0262 | 0.0380 / 0.0854 / 0.0700 | 0.0034 / 0.0059 / 0.0052 |
| linf   | 0.0011 / 0.0026 / 0.0020 | 0.0112 / 0.0238 / 0.0180 | 0.0103 / 0.0222 / 0.0175 | 0.0027 / 0.0046 / 0.0041 |
| lp_0.5 | 0.0015 / 0.0043 / 0.0029 | 0.0060 / 0.0111 / 0.0089 | 0.0103 / 0.0213 / 0.0171 | 0.0028 / 0.0045 / 0.0041 |
| maha   | 0.0021 / 0.0063 / 0.0043 | 0.0230 / 0.0460 / 0.0396 | 0.0452 / 0.0973 / 0.0861 | 0.0042 / 0.0075 / 0.0065 |

#### L2-normalized embeddings

| Metric | Content (audio, 49d) | ALS k=8 | ALS k=50 | LightFM |
|--------|----------------------|---------|----------|---------|
| dot    | 0.0031 / 0.0080 / 0.0061 | 0.0169 / 0.0403 / 0.0308 | 0.0513 / 0.1170 / 0.1005 | 0.0108 / 0.0208 / 0.0179 |
| cos    | 0.0031 / 0.0080 / 0.0061 | 0.0169 / 0.0403 / 0.0308 | 0.0513 / 0.1170 / 0.1005 | 0.0108 / 0.0208 / 0.0179 |
| l2     | 0.0031 / 0.0080 / 0.0061 | 0.0169 / 0.0403 / 0.0308 | 0.0513 / 0.1170 / 0.1005 | 0.0108 / 0.0208 / 0.0179 |
| l1     | 0.0031 / 0.0082 / 0.0061 | 0.0162 / 0.0382 / 0.0295 | 0.0477 / 0.1101 / 0.0939 | 0.0095 / 0.0192 / 0.0161 |
| linf   | 0.0023 / 0.0052 / 0.0042 | 0.0158 / 0.0372 / 0.0287 | 0.0321 / 0.0771 / 0.0651 | 0.0068 / 0.0139 / 0.0115 |
| lp_0.5 | 0.0028 / 0.0073 / 0.0056 | 0.0143 / 0.0339 / 0.0265 | 0.0408 / 0.0970 / 0.0814 | 0.0070 / 0.0141 / 0.0119 |
| maha   | 0.0027 / 0.0076 / 0.0054 | 0.0156 / 0.0352 / 0.0275 | 0.0501 / 0.1104 / 0.0948 | 0.0087 / 0.0172 / 0.0147 |

Users evaluated: Content 22,328 · ALS (k=8, k=50) 33,229 · LightFM 10,247.

Takeaways:
- ALS at k=50 dominates across the board; raw `l2`/`dot` (≈0.053 P@10) edge out cosine, with `maha` close behind.
- Going from k=8 to k=50 roughly doubles all three metrics on ALS — latent capacity matters more than the choice of distance.
- After L2 normalization, `dot`, `cos`, and `l2` collapse to identical scores, as expected on the unit sphere.
- Content-only (audio features) embeddings trail ALS by an order of magnitude — collaborative signal is not recoverable from audio alone.
- LightFM hybrid sits between content-only and ALS k=50 here; raw `dot` is its best single setting (P@10 ≈ 0.0121).

## Datasets
- [Triplets of users' interactions](http://millionsongdataset.com/tasteprofile/)
- [Soundcharts API for fetching songs' audio features](https://developers.soundcharts.com/home)

## Resources
- [Imlicit ALS](https://education.yandex.ru/handbook/ml/article/rekomendacii-na-osnove-matrichnyh-razlozhenij#alternating-least-squares-als)
- [implicit library docs](https://benfred.github.io/implicit/api/models/cpu/als.html)
- [LightFM library docs](https://making.lyst.com/lightfm/docs/lightfm.html)