import torch


def pad_patches_collate(batch: list[tuple[torch.Tensor, torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    """
    Custom collate function that pads variable-length patch sequences to the maximum patch count in the batch and produces padding masks.

    When different images produce different numbers of patches (e.g. the grid patcher
    extracts as many fixed-size patches as fit), the default PyTorch collate fails because
    it cannot stack tensors with different first dimensions.

    This collate pads shorter sequences with zero-filled patches and creates boolean masks so the model can ignore padded positions.

    Parameters:
        batch (list[tuple[torch.Tensor, torch.Tensor, int]]): list of samples from the dataset.
            Each sample is a tuple of:
                - image_1_patches (torch.Tensor): patches of first image, shape [N_i, C, H, W]
                - image_2_patches (torch.Tensor): patches of second image, shape [M_i, C, H, W]
                - label (int): cluster / writer ID

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - images_1 (torch.Tensor): padded patches, shape [B, max_patches, C, H, W]
            - images_2 (torch.Tensor): padded patches, shape [B, max_patches, C, H, W]
            - labels (torch.Tensor): writer IDs, shape [B]
            - mask_1 (torch.Tensor): boolean padding mask, shape [B, max_patches].
                True = padded (should be ignored), False = real patch.
            - mask_2 (torch.Tensor): boolean padding mask, shape [B, max_patches].
                True = padded (should be ignored), False = real patch.
    """

    images_1_list, images_2_list, labels_list = zip(*batch)

    # find the maximum patch count across both image branches in the entire batch
    max_patches = max(
        max(img.shape[0] for img in images_1_list),
        max(img.shape[0] for img in images_2_list),
    )

    # get the patch shape (C, H, W) from the first sample - all patches share the same shape
    _, C, H, W = images_1_list[0].shape

    batch_size = len(batch)

    # pre-allocate zero-filled output tensors
    padded_images_1 = torch.zeros(batch_size, max_patches, C, H, W)
    padded_images_2 = torch.zeros(batch_size, max_patches, C, H, W)

    # masks: True where padded (following PyTorch transformer src_key_padding_mask convention)
    mask_1 = torch.ones(batch_size, max_patches, dtype=torch.bool)
    mask_2 = torch.ones(batch_size, max_patches, dtype=torch.bool)

    for i, (img1, img2, _) in enumerate(batch):
        n1 = img1.shape[0]
        n2 = img2.shape[0]

        # copy real patches into the pre-allocated tensors
        padded_images_1[i, :n1] = img1
        padded_images_2[i, :n2] = img2

        # mark real patch positions as False (not padding)
        mask_1[i, :n1] = False
        mask_2[i, :n2] = False

    labels = torch.tensor(labels_list, dtype=torch.long)

    return padded_images_1, padded_images_2, labels, mask_1, mask_2
