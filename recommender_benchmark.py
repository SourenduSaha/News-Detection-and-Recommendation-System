
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Evaluation Metrics

def precision_recall_at_k(recommender_fn, test_df, k=5):
    precision_list = []
    recall_list = []

    grouped = test_df.groupby('user_id')

    for user_id, group in grouped:
        actual_clicked = set(group[group['clicked'] == 1]['news_id'])
        if not actual_clicked:
            continue

        try:
            recs = recommender_fn(user_id, top_n=k)
            if isinstance(recs, pd.DataFrame):
                recommended = set(recs['news_id'])
            else:
                continue
        except:
            continue

        hits = len(recommended & actual_clicked)
        precision = hits / k
        recall = hits / len(actual_clicked)

        precision_list.append(precision)
        recall_list.append(recall)

    return {
        'precision@{}'.format(k): sum(precision_list) / len(precision_list),
        'recall@{}'.format(k): sum(recall_list) / len(recall_list)
    }

def auc_score(recommender_fn, test_df):
    auc_scores = []
    grouped = test_df.groupby('user_id')

    for user_id, group in grouped:
        y_true = []
        y_scores = []

        clicked_items = set(group['news_id'][group['clicked'] == 1])
        if not clicked_items:
            continue

        try:
            recs = recommender_fn(user_id, top_n=100)
            if isinstance(recs, pd.DataFrame):
                recommended = list(recs['news_id'])
            else:
                continue
        except:
            continue

        for item in group['news_id']:
            y_true.append(1 if item in clicked_items else 0)
            y_scores.append(1 if item in recommended else 0)

        if len(set(y_true)) < 2:
            continue

        auc = roc_auc_score(y_true, y_scores)
        auc_scores.append(auc)

    return {
        'auc_score': sum(auc_scores) / len(auc_scores)
    }

def coverage(recommender_fn, all_items, user_list, k=5):
    recommended_items = set()
    for user_id in user_list:
        try:
            recs = recommender_fn(user_id, top_n=k)
            if isinstance(recs, pd.DataFrame):
                recommended_items.update(recs['news_id'].tolist())
        except:
            continue
    return {
        'coverage': len(recommended_items) / len(all_items)
    }

def novelty(recommender_fn, train_df, user_list, k=5):
    from collections import Counter
    import numpy as np

    item_popularity = Counter(train_df[train_df['clicked'] == 1]['news_id'])
    total_clicks = sum(item_popularity.values())
    popularity_prob = {item: count / total_clicks for item, count in item_popularity.items()}
    log_popularity = {item: -np.log(prob) for item, prob in popularity_prob.items()}

    novelty_scores = []
    for user_id in user_list:
        try:
            recs = recommender_fn(user_id, top_n=k)
            if isinstance(recs, pd.DataFrame):
                items = recs['news_id'].tolist()
                novelty = np.mean([log_popularity.get(item, 0) for item in items])
                novelty_scores.append(novelty)
        except:
            continue

    return {
        'novelty': sum(novelty_scores) / len(novelty_scores)
    }

# Evaluation Runner

def run_benchmark(name, recommender_fn, test_df, train_df, all_items, users, k=5):
    print(f"\n🔍 Evaluating {name} Recommender")
    results = {}
    results.update(precision_recall_at_k(recommender_fn, test_df, k=k))
    results.update(auc_score(recommender_fn, test_df))
    results.update(coverage(recommender_fn, all_items, users, k=k))
    results.update(novelty(recommender_fn, train_df, users, k=k))
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    return results

# Example Usage
users_to_eval = sampled_df['user_id'].unique()[:500]
all_items = set(user_item_matrix.columns)

run_benchmark("User-Based", recommend_for_users, sampled_df, sampled_df, all_items, users_to_eval)
run_benchmark("Item-Based", recommend_items_for_users, sampled_df, sampled_df, all_items, users_to_eval)
run_benchmark("Hybrid", hybrid_recommendations, sampled_df, sampled_df, all_items, users_to_eval)
