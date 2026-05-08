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
1. Load — reads 600k rows from interactions_audio_v2.1.parquet (10,811 users × 14,283 songs).                                           
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

### Results

Top-10 evaluation across embedding sources and distance metrics. Each cell shows Precision@10 / Recall@10 / NDCG@10.
k - the size of ALS embeddings

#### Raw embeddings

| Metric | Content (audio, 49d) | ALS k=8 | ALS k=50 | LightFM |
|--------|----------------------|---------|----------|---------|
| dot    | 0.0016 / 0.0038 / 0.0027 | 0.0236 / 0.0481 / 0.0406 | 0.0517 / 0.1122 / 0.0962 | 0.0135 / 0.0266 / 0.0228 |
| cos    | 0.0031 / 0.0080 / 0.0061 | 0.0175 / 0.0391 / 0.0307 | 0.0498 / 0.1126 / 0.0975 | 0.0123 / 0.0255 / 0.0207 |
| l2     | 0.0015 / 0.0039 / 0.0029 | 0.0236 / 0.0481 / 0.0406 | 0.0519 / 0.1128 / 0.0965 | 0.0055 / 0.0112 / 0.0097 |
| l1     | 0.0015 / 0.0043 / 0.0029 | 0.0162 / 0.0324 / 0.0249 | 0.0370 / 0.0830 / 0.0687 | 0.0050 / 0.0100 / 0.0088 |
| linf   | 0.0011 / 0.0026 / 0.0020 | 0.0060 / 0.0115 / 0.0094 | 0.0097 / 0.0186 / 0.0154 | 0.0035 / 0.0071 / 0.0062 |

#### L2-normalized embeddings

| Metric | Content (audio, 49d) | ALS k=8 | ALS k=50 | LightFM |
|--------|----------------------|---------|----------|---------|
| dot    | 0.0031 / 0.0080 / 0.0061 | 0.0175 / 0.0391 / 0.0307 | 0.0498 / 0.1126 / 0.0975 | 0.0123 / 0.0255 / 0.0207 |
| cos    | 0.0031 / 0.0080 / 0.0061 | 0.0175 / 0.0391 / 0.0307 | 0.0498 / 0.1126 / 0.0975 | 0.0123 / 0.0255 / 0.0207 |
| l2     | 0.0031 / 0.0080 / 0.0061 | 0.0175 / 0.0391 / 0.0307 | 0.0489 / 0.1112 / 0.0961 | 0.0123 / 0.0255 / 0.0207 |
| l1     | 0.0031 / 0.0082 / 0.0061 | 0.0167 / 0.0379 / 0.0294 | 0.0450 / 0.1028 / 0.0886 | 0.0105 / 0.0218 / 0.0176 |
| linf   | 0.0023 / 0.0052 / 0.0042 | 0.0162 / 0.0360 / 0.0285 | 0.0309 / 0.0743 / 0.0633 | 0.0073 / 0.0142 / 0.0124 |

Users evaluated: Content 22,328 · ALS (k=8, k=50) 19,882 · LightFM 20,622. `lp_0.5` and `maha` were only run on the content-based audio embeddings.

#### Library built-in evaluators

For reference, the same models evaluated with their library's built-in scorers (no manual top-K loop, no L2 normalization):

| Source | Built-in call | Precision@10 | Recall@10 | NDCG@10 |
|--------|---------------|--------------|-----------|---------|
| ALS k=8  | `model.recommend(...)` (implicit) | 0.0236 | 0.0481 | 0.0406 |
| ALS k=50 | `model.recommend(...)` (implicit) | 0.0517 | 0.1122 | 0.0962 |
| LightFM  | `lightfm.evaluation.precision_at_k` / `recall_at_k` | 0.0279 | 0.0572 | — |

`implicit`'s `model.recommend` ranks by dot product, so its numbers reproduce the "raw dot" rows of the ALS columns above. LightFM's built-in scorers also rank by `model.predict`, but include the learned user/item biases on top of the latent dot product, which is why they outperform the bias-free "raw dot" row for LightFM (0.0279 vs 0.0135). LightFM's helpers don't expose NDCG.

Takeaways:
- ALS at k=50 dominates across the board; raw `l2`/`dot` (≈0.052 P@10) edge out cosine.
- Going from k=8 to k=50 roughly doubles all three metrics on ALS — latent capacity matters more than the choice of distance.
- After L2 normalization, `dot`, `cos`, and `l2` collapse to (near-)identical scores, as expected on the unit sphere.
- Content-only (audio features) embeddings trail ALS by an order of magnitude — collaborative signal is not recoverable from audio alone.
- LightFM hybrid sits between content-only and ALS k=50; raw `dot` is its best single setting (P@10 ≈ 0.0135), and `l2`/`l1` on raw LightFM factors collapse badly because the learned vectors have very uneven magnitudes.

## Datasets
- [Triplets of users' interactions](http://millionsongdataset.com/tasteprofile/)
- [Soundcharts API for fetching songs' audio features](https://developers.soundcharts.com/home)

## Resources
- [Imlicit ALS](https://education.yandex.ru/handbook/ml/article/rekomendacii-na-osnove-matrichnyh-razlozhenij#alternating-least-squares-als)
- [implicit library docs](https://benfred.github.io/implicit/api/models/cpu/als.html)
- [LightFM library docs](https://making.lyst.com/lightfm/docs/lightfm.html)