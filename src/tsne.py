import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def plot_tsne(file_name: str, img_encoder: torch.nn.Module, dl: torch.utils.data.DataLoader, device: torch.device, logger: logging.Logger) -> None:

    """
    Compute embeddings for all samples in dataloader and visualize them with t-SNE.

    Parameters:
        file_name (str): Output path for the saved t-SNE plot.
        img_encoder (torch.nn.Module): Encoder model used to extract embeddings.
        dl (torch.utils.data.DataLoader): Dataloader providing image batches and labels.
        device (torch.device): Device used for computation.
        logger (logging.Logger): Logger for progress information.
    """

    all_embeddings = []
    all_labels = []

    # disable gradient computation because this is only evaluation / visualization
    with torch.no_grad():

        for images_1, images_2, labels in dl:
            # only the first image branch is used for visualization
            # the second branch is part of the dataset pair structure, but it is not needed here
            images_1 = images_1.to(device)

            # convert input images into embedding vectors
            embeddings = img_encoder(images_1)

            # store embeddings and corresponding labels on CPU as NumPy arrays
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    logger.info(f"Collected {len(all_embeddings)} embedding batches, first batch shape: {all_embeddings[0].shape}")

    # merge all batches into one array of shape (N, embedding_dim)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    logger.info(f"All embeddings shape: {all_embeddings.shape}")

    # reduce high-dimensional embeddings to 2D for visualization
    tsne = TSNE(
        n_components=2,
        perplexity=30.0,
        metric="cosine",
        verbose=1
    )
    positions = tsne.fit_transform(all_embeddings)

    logger.info(f"t-SNE output shape: {positions.shape}")

    fig, ax = plt.subplots(figsize=(15, 15))

    # find all distinct class labels to plot each class with a different color
    distinct_labels = list(set(all_labels))
    colors = plt.cm.get_cmap("hsv", len(distinct_labels))

    for index, label_id in enumerate(distinct_labels):
        # select only the 2D points belonging to the current label
        label_positions = positions[all_labels == label_id]

        ax.scatter(
            label_positions[:, 0],
            label_positions[:, 1],
            c=[colors(index)] * len(label_positions)
        )

    plt.savefig(file_name)
    plt.close("all")
