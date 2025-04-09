import torch
from tqdm import tqdm
from datasets.data_loader import get_loader
from models.crisp_sam2 import CrispSam2
from models.sam2.build_sam import build_sam2
import os
import argparse
from datetime import datetime
import numpy as np
from monai.metrics import compute_meandice, compute_average_surface_distance


def get_test_args_parser():
    parser = argparse.ArgumentParser(description='Test Configuration')

    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--pretrain", type=str, default='/path/to/pretrained_model')
    parser.add_argument("--data_dir", type=str, default='/path/to/data')
    parser.add_argument("--dataset_codes", type=list, default=['0003'])
    parser.add_argument("--patch_size", default=(96, 96, 96), type=tuple)
    parser.add_argument("--spatial_size", default=(32, 512, 512), type=tuple)
    parser.add_argument("--work_dir", type=str, default='./work_dir')
    parser.add_argument("--config_file", type=str, default='./path/to/config_file')
    parser.add_argument("--sam2_ckpt", type=str, default='./path/to/sam2_ckpt')
    # parser.add_argument("--model_id", type=str, default='hf_id')
    parser.add_argument("--clip_text_ckpt", type=str, default='./path/to/clip_text_ckpt')
    parser.add_argument("--clip_image_ckpt", type=str, default='./path/to/clip_image_ckpt')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def test(args):
    sam_model = build_sam2(config_file=args.config_file, checkpoint=args.sam2_ckpt, mode="eval")
    model = CrispSam2(
        image_encoder=sam_model.image_encoder,
        memory_attention=sam_model.memory_attention,
        memory_encoder=sam_model.memory_encoder,
        clip_text_ckpt=args.clip_text_ckpt,
        clip_image_ckpt=args.clip_image_ckpt,
    ).cuda(args.gpu)
    model.eval()

    if args.pretrain:
        checkpoint = torch.load(args.pretrain, map_location=f'cuda:{args.gpu}')
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint from {args.pretrain}")

    test_dataloader = get_loader(args)

    os.makedirs(args.result_dir, exist_ok=True)
    log_file = os.path.join(args.result_dir, f"test_log_{datetime.now().strftime('%Y%m%d-%H%M')}.txt")

    total_dsc = []
    total_nsd = []
    with torch.no_grad(), tqdm(total=len(test_dataloader), desc="Testing") as pbar:
        for batch in test_dataloader:
            image = batch["image"].cuda(args.gpu)
            gt3D = batch["post_label"].cuda(args.gpu)
            organ_name_list = batch['organ_name_list']
            text_list = batch['text']
            patient_id = batch['patient_id'][0]

            pred_masks = []
            for cls_idx in range(len(organ_name_list)):
                labels_cls = gt3D[:, cls_idx]
                organ_name = organ_name_list[cls_idx]
                crisp_masks = []

                for frame_idx in range(image.size()[-1]):
                    output = model(image, text_list, frame_idx)
                    crisp_mask = output['crisp_masks'].sigmoid() > 0.5
                    crisp_masks.append(crisp_mask.cpu().numpy())

                mask_3d = np.stack(crisp_masks, axis=0).squeeze()
                pred_masks.append(mask_3d)
                save_path = os.path.join(args.result_dir, patient_id)
                os.makedirs(save_path, exist_ok=True)
                np.save(os.path.join(save_path, f"{organ_name}_pred.npy"), mask_3d)

                if gt3D is not None:
                    gt_mask = labels_cls.squeeze().cpu().numpy()
                    dsc = compute_meandice(mask_3d, gt_mask, include_background=False)
                    total_dsc.append(dsc)
                    nsd = compute_average_surface_distance(mask_3d, gt_mask, include_background=False)
                    total_nsd.append(nsd)
                    with open(log_file, 'a') as f:
                        f.write(f"Patient {patient_id}, Organ {organ_name}: DSC={dsc:.4f}, NSD={nsd:.4f}\n")

            pbar.update(1)

    if total_dsc and total_nsd:
        mean_dsc = np.mean(total_dsc)
        mean_nsd = np.mean(total_nsd)
        print(f"Test Finished. Mean DSC: {mean_dsc:.4f}, Mean NSD: {mean_nsd:.4f}")
        with open(log_file, 'a') as f:
            f.write(f"Overall Mean DSC: {mean_dsc:.4f}, Overall Mean NSD: {mean_nsd:.4f}\n")
    else:
        print("No ground truth provided, skipped metric calculation.")


def main():
    args = get_test_args_parser()
    print("Test Arguments:", args)
    test(args)


if __name__ == '__main__':
    main()
