import numpy as np


def _get_gallery_ranking(gallery_labels: np.ndarray,
                         gallery_embeds: np.ndarray,
                         query_embeds: np.ndarray):
    """
    For each query embedding, all gallery embeddings are ranked in descending order
    according to their cosine similarity with the query embedding.

    Parameters:
        gallery_labels (np.ndarray, shape=(N_gallery), dtype=int):
            Labels of all gallery samples.
        gallery_embeds (np.ndarray, shape=(N_gallery, Embed_size), dtype=np.float32):
            Embeddings of all gallery samples.
        query_embeds (np.ndarray, shape=(N_query, Embed_size), dtype=np.float32):
            Embeddings of all query samples.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            gallery_labels_ranked (np.ndarray, shape=(N_query, N_gallery), dtype=int):\
                Labels of ranked gallery samples.

            gallery_scores_ranked (np.ndarray, shape=(N_query, N_gallery), dtype=np.float32):\
                Corresponding scores to the ranked gallery samples. (The scores are in descending order.)
    """

    gallery_idxs_ranked = []
    gallery_scores_ranked = []

    # iterate over batches of queries
    query_batch_size = 1024
    for i in range(0, len(query_embeds), query_batch_size):

        # select batch of queries
        # (query_batch_size, emb_size)
        query_batch = query_embeds[i:i+query_batch_size]

        # computing cosine similarity between query and gallery samples
        #   (gallery and query embeddings are already normalized)
        # (query_batch_size, emb_size) @ (N_gallery, emb_size)^T -> (query_batch_size, N_gallery)
        sims_batch = query_batch @ gallery_embeds.T

        # get indices that sort similarity scores from largest to smallest for each query
        # (query_batch_size, N_gallery)
        idxs = np.argsort(-sims_batch, axis=1)

        # sort the scores
        # (query_batch_size, N_gallery)
        scores = np.take_along_axis(sims_batch, idxs, axis=1)

        gallery_idxs_ranked.append(idxs)
        gallery_scores_ranked.append(scores)

    # stack batches into single array
    # list[(query_batch_size, N_gallery)] -> (N_queries, N_gallery)
    gallery_idxs_ranked = np.vstack(gallery_idxs_ranked)
    # list[(query_batch_size, N_gallery)] -> (N_queries, N_gallery)
    gallery_scores_ranked = np.vstack(gallery_scores_ranked)

    # map indices to actual labels
    # (N_queries, N_gallery)
    gallery_labels_ranked = gallery_labels[gallery_idxs_ranked]

    # (N_queries, N_gallery), (N_queries, N_gallery)
    return gallery_labels_ranked, gallery_scores_ranked


def _pool_gallery_to_class_ranking(gallery_labels_ranked: np.ndarray,
                                   gallery_scores_ranked: np.ndarray):
    """
    Pools (collapses) the ranking of gallery samples
    into a ranking of gallery classes for each query.

    Gallery samples sharing the same label are pooled (grouped) into a single class-level item,
    whose similarity score is the maximum similarity score of the grouped samples.

    Parameters:
        gallery_labels_ranked (np.ndarray, shape=(N_query, N_gallery), dtype=int):
            Labels of ranked gallery samples.

        gallery_scores_ranked (np.ndarray, shape=(N_query, N_gallery), dtype=np.float32):
            Corresponding scores to the ranked gallery samples. (The scores are in descending order.)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            class_labels_ranked (np.ndarray, shape=(N_query, N_labels), dtype=int):\
                Labels of ranked gallery classes.

            class_scores_ranked (np.ndarray, shape=(N_query, N_labels), dtype=np.float32):\
                Corresponding scores to the ranked labels. (The scores are in descending order.)
    """

    N_queries = gallery_scores_ranked.shape[0]
    unique_labels = np.unique(gallery_labels_ranked)
    N_labels = len(unique_labels)

    # each row will contain max pooled score for each label in order of unique_labels
    class_scores = np.full((N_queries, N_labels), -np.inf)

    for i, label in enumerate(unique_labels):
        # create mask for this label across all queries
        mask = (gallery_labels_ranked == label)

        # max pool scores where mask is True
        class_scores[:, i] = np.where(
            # check if the label appears in the row at least once
            mask.any(axis=1),
            np.max(
                np.where(mask, gallery_scores_ranked, -np.inf),
                axis=1),  # max score for the current label row-wise
            -np.inf
        )

    # get labels corresponding to the scores
    # simply repeat row-wise the unique labels as scores are in this order
    # (N_queries, N_labels)
    class_labels = np.tile(unique_labels, (N_queries, 1))

    # get indexes that sort scores in each row in descending order
    # (N_queries, N_labels)
    score_order = np.argsort(-class_scores, axis=1)

    # sort the scores and corresponding lables
    class_scores_ranked = np.take_along_axis(
        class_scores, score_order, axis=1)
    class_labels_ranked = np.take_along_axis(
        class_labels, score_order, axis=1)

    return class_labels_ranked, class_scores_ranked
