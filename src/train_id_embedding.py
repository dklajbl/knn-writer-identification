import argparse
import os
import sys
import time

from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2
import torchvision
import matplotlib.pyplot as plt
import logging

# from safe_gpu.safe_gpu import GPUOwner # UNUSED


from pytorch_metric_learning import losses

from id_dataset import IdDataset
from encoders import Encoder

from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

# from sklearn.manifold import TSNE
from tsne import plot_tsne


logger = logging.getLogger(__name__)


def parseargs():
    parser = argparse.ArgumentParser(
        usage='Trains contrastive self-supervised training on artificial data.')
    parser.add_argument('--gt-file', required=True,
                        help='Text file with a image file name and id o each line.')
    parser.add_argument(
        '--gt-file-tst', help='Testing text file with a image file name and id o each line.')
    parser.add_argument('--lmdb', required=True, help='Path to lmdb DB..')

    parser.add_argument('--width', type=int, default=320)

    parser.add_argument('--start-iteration', default=0, type=int)
    parser.add_argument('--max-iterations', default=50000, type=int)
    parser.add_argument('--view-step', default=50, type=int,
                        help="Number of training iterations between network testing.")

    parser.add_argument('--embed-dim', default=256, type=int,
                        help="Output embedding dimension.")
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--learning-rate', default=0.0002, type=float)
    parser.add_argument('--weight-decay', default=0.01, type=float)

    parser.add_argument('--temperature', default=0.5,
                        type=float, help="Temperature for NTXent loss.")
    parser.add_argument('--out-checkpoints-dir', default='.', type=str)
    parser.add_argument('--show-dir', default='.', type=str)

    parser.add_argument('--eval-on-start', action='store_true')

    parser.add_argument('--logging-level', default='INFO')

    args = parser.parse_args()
    return args


def main():
    args = parseargs()

    log_formatter = logging.Formatter(
        'CONVERT LINES TO JSONL - %(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    logger.info(f'ARGS {args}')

    # gpu_owner = GPUOwner() # UNUSED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.multiprocessing.set_start_method('spawn')

    show_dir_heat_maps_path = os.path.join(args.show_dir, "heat_maps")
    show_dir_tsne_path = os.path.join(args.show_dir, "tsne")
    show_dir_retrieval_path = os.path.join(args.show_dir, "retrieval")
    if not os.path.exists(show_dir_heat_maps_path):
        os.makedirs(show_dir_heat_maps_path)
    if not os.path.exists(show_dir_tsne_path):
        os.makedirs(show_dir_tsne_path)
    if not os.path.exists(show_dir_retrieval_path):
        os.makedirs(show_dir_retrieval_path)
    if not os.path.exists(args.out_checkpoints_dir):
        os.makedirs(args.out_checkpoints_dir)

    img_encoder = Encoder(args.embed_dim).to(device)
    if args.start_iteration > 0:
        checkpoint_path = os.path.join(
            args.out_checkpoints_dir, f'cp-{args.start_iteration:07d}.img.ckpt')
        logger.info(f'Loading image checkpoint {checkpoint_path}')
        img_encoder.load_state_dict(torch.load(
            checkpoint_path, map_location=device))

    ds = IdDataset(args.gt_file, args.lmdb, transform=torchvision.transforms.ToTensor(
    ), augment=True, width=args.width)
    dl = DataLoader(ds, num_workers=8, batch_size=args.batch_size,
                    shuffle=True, drop_last=True)

    if args.gt_file_tst:
        ds_tst = IdDataset(args.gt_file_tst, args.lmdb, transform=torchvision.transforms.ToTensor(), augment=False,
                           restrict_data=True, test=True, width=args.width)
        dl_tst = DataLoader(
            ds_tst, num_workers=0, batch_size=args.batch_size, shuffle=False, drop_last=True)
    else:
        dl_tst = None

    optimizer = torch.optim.AdamW(
        img_encoder.parameters(), lr=args.learning_rate)

    loss_history = [0] * args.start_iteration
    iteration = args.start_iteration
    last_view_iteration = args.start_iteration

    loss_object = losses.NTXentLoss(temperature=args.temperature)

    while True:
        t1 = time.time()
        for images1, images2, labels in dl:
            if iteration == args.start_iteration:
                cv2.imwrite(os.path.join(args.show_dir, 'images1.png'), np.concatenate(
                    images1.permute(0, 2, 3, 1).numpy(), axis=0) * 255)
                cv2.imwrite(os.path.join(args.show_dir, 'images2.png'), np.concatenate(
                    images2.permute(0, 2, 3, 1).numpy(), axis=0) * 255)

            with torch.no_grad():
                images1 = images1.to(device)
                images2 = images2.to(device)
                images = torch.cat([images1, images2], dim=0)
                labels = torch.cat([labels, labels]).to(device)

            optimizer.zero_grad()
            embedding = img_encoder(images)
            loss = loss_object(embedding, labels)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

            if ((iteration % args.view_step == 0 and iteration > args.start_iteration)
                    or (args.eval_on_start and iteration == args.start_iteration)):

                avg_loss = np.mean(loss_history[last_view_iteration:])
                logger.info(
                    f'LOG {iteration} iterations:{iteration-last_view_iteration} '
                    f'loss:{avg_loss:0.6f} time:{time.time()-t1:.1f}s')
                torch.save(img_encoder.state_dict(), os.path.join(
                    args.out_checkpoints_dir, f'cp-{iteration:07d}.img.ckpt'))
                last_view_iteration = iteration
                t1 = time.time()
                fig, ax = plt.subplots()
                distances = torch.mm(
                    embedding, embedding.t()).detach().cpu().numpy()
                heatmap = ax.imshow(distances)
                plt.colorbar(heatmap)
                plt.savefig(os.path.join(show_dir_heat_maps_path,
                            f'cp-{iteration:07d}.png'))
                plt.close('all')
                if dl_tst is not None:
                    auc, mean_auc, fpr, tpr, thr, mean_ap = test_retrieval(
                        os.path.join(
                            show_dir_retrieval_path,
                            f'retrieval-{iteration:07d}.{os.path.basename(args.gt_file_tst)}.png'),
                        img_encoder,
                        dl_tst,
                        device
                    )
                    logger.info(
                        f'TEST {iteration} AUC:{auc:0.6f} MEAN_AUC:{mean_auc:0.6f} MEAN_AP:{mean_ap:0.6f}')
                    plot_tsne(os.path.join(show_dir_tsne_path, f'tsne-{iteration:07d}.tst.png'),
                              img_encoder, dl_tst, device, logger)
            iteration += 1
            if iteration >= args.max_iterations:
                break
        if iteration >= args.max_iterations:
            break


def test_retrieval(file_name, img_encoder, dl, device, query_vis_count=32, result_vis_count=20):
    t_start = time.time()
    all_embeddings = []
    all_labels = []
    all_images = []
    with torch.no_grad():
        for images1, images2, labels in dl:
            images1 = images1.to(device)
            embedding = img_encoder(images1)
            all_embeddings.append(embedding.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_images.append(images1.permute(0, 2, 3, 1).cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_images = np.concatenate(all_images, axis=0) * 255

    similarities = np.dot(all_embeddings, all_embeddings.T)

    logging.info(
        f'SIMILARITIES {similarities.min()} {similarities.max()}, {similarities.shape}')

    collage_ids = np.linspace(
        0, all_images.shape[0] - 1, query_vis_count).astype(np.int64)
    result_collage = []

    scores = []
    labels = []
    ap = []
    auc = []

    for i in range(similarities.shape[0]):
        query_sim = similarities[i]
        query_labels = all_labels[i] == all_labels
        # remove the query image
        query_labels[i] = False
        query_sim[i] = -1e20
        if np.any(query_labels):
            auc.append(roc_auc_score(query_labels, query_sim))
            ap.append(average_precision_score(query_labels, query_sim))

        scores.append(query_sim)
        labels.append(query_labels)

        if i in collage_ids:
            image = cv2.copyMakeBorder(
                all_images[i], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            image = cv2.copyMakeBorder(
                image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            row = [image]
            query_label = all_labels[i]
            best_ids = np.argsort(query_sim)[::-1][:result_vis_count]
            for id in best_ids:
                image = all_images[id]
                if all_labels[id] == query_label:
                    image = cv2.copyMakeBorder(
                        image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 255, 0])
                else:
                    image = cv2.copyMakeBorder(
                        image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 0, 255])
                image = cv2.copyMakeBorder(
                    image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                row.append(image)
            result_collage.append(np.concatenate(row, axis=1))

    if result_collage:
        result_collage = np.concatenate(result_collage, axis=0)
        cv2.imwrite(file_name, result_collage)

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(np.asarray(labels), axis=0)

    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    mean_ap = np.mean(ap)
    mean_auc = np.mean(auc)
    auc = roc_auc_score(labels, scores)
    logger.info(f'TEST TIME: {time.time() - t_start}s')
    return auc, mean_auc, fpr, tpr, thr, mean_ap


if __name__ == '__main__':
    main()
