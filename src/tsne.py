from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_tsne(file_name, img_encoder, dl, device, logger):
    all_embeddings = []
    all_labels = []
    counter = 0
    with torch.no_grad():
        for images1, images2, labels in dl:
            images1 = images1.to(device)
            embedding = img_encoder(images1)
            all_embeddings.append(embedding.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            counter += labels.shape[0]

    logger.info(f"{len(all_embeddings)}, {all_embeddings[0].shape}")
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(all_embeddings.shape)
    all_labels = np.concatenate(all_labels, axis=0)

    tsne = TSNE(n_components=2, perplexity=30.0,  metric='cosine', verbose=1)
    pos = tsne.fit_transform(all_embeddings)
    logger.info(pos.shape)
    fig, ax = plt.subplots(figsize=(15, 15))

    distinct_labels = list(set(all_labels))
    colors = plt.cm.get_cmap('hsv', len(distinct_labels))
    for i, label_id in enumerate(distinct_labels):
        label_pos = pos[all_labels == label_id]
        ax.scatter(label_pos[:, 0], label_pos[:, 1],
                   c=[colors(i)] * len(label_pos))
    plt.savefig(file_name)
    plt.close('all')
