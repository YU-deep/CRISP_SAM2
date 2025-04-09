import os
import torch
import argparse
from datetime import datetime
from models.crisp_sam2 import CrispSam2
import torch.multiprocessing as mp
import shutil
from models.components.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.components.loss import CrispSamLoss
from datasets.data_loader import get_loader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models.sam2.build_sam import build_sam2, build_sam2_hf


def set_parse():
    parser = argparse.ArgumentParser()
    # %% set up parser
    parser.add_argument("--pretrain", type=str, default='')
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--dataset_codes", type=list, default=['0003'])

    # config
    parser.add_argument("--patch_size", default=(96, 96, 96), type=tuple)
    parser.add_argument('--work_dir', type=str, default='./work_dir')
    parser.add_argument("--clip_text_ckpt", type=str, default='./path/to/clip_text_ckpt')
    parser.add_argument("--clip_image_ckpt", type=str, default='./path/to/clip_image_ckpt')
    parser.add_argument("--config_file", type=str, default='./path/to/config_file')
    parser.add_argument("--sam2_ckpt", type=str, default='./path/to/sam2_ckpt')
    # parser.add_argument("--model_id", type=str, default='hf_id')
    parser.add_argument('--num_workers', type=int, default=8)

    # dist
    parser.add_argument('--dist', dest='dist', type=bool, default=True,
                        help='distributed segment_anything_training or not')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--init_method', type=str, default="env://")
    parser.add_argument('--bucket_cap_mb', type=int, default=25,
                        help='The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)')

    # key params
    parser.add_argument('--lr_one', type=float, default=1e-4)
    parser.add_argument('--lr_two', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--stage_one_frozen', type=list)
    parser.add_argument('--stage_two_frozen', type=list)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--stage_one', type=int, default=120)
    parser.add_argument('--stage_two', type=int, default=280)
    parser.add_argument('--warmup_epoch', type=int, default=80)
    parser.add_argument('--save_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)

    # parser.add_argument('--stage_two_unfrozen', nargs='*', type=str)

    args = parser.parse_args()
    return args


def train_epoch(args, model, train_dataloader, loss_func, optimizer, scheduler, epoch, rank, gpu, iter_num):
    epoch_loss = 0
    epoch_sam_loss = 0
    epoch_crisp_loss = 0

    epoch_iterator = tqdm(
        train_dataloader, desc=f"[RANK {rank}: GPU {gpu}]", dynamic_ncols=True
    )
    if args.dist:
        train_dataloader.sampler.set_epoch(epoch)
        torch.distributed.barrier()

    for batch in epoch_iterator:
        image, gt3D = batch["image"].cuda(), batch["post_label"].cuda()
        organ_name_list = batch['organ_name_list']
        text_list = batch['text']

        loss_step_avg = 0
        sam_loss_step_avg = 0
        crisp_loss_step_avg = 0
        for cls_idx in range(len(organ_name_list)):
            optimizer.zero_grad()
            labels_cls = gt3D[:, cls_idx]
            for frame_idx in range(image.size()[-1]):
                if torch.sum(labels_cls) == 0:
                    print(f'[RANK {rank}: GPU {gpu}] ITER-{iter_num} --- No object, skip iter')
                    continue

                output = model(image, text_list, frame_idx)
                crisp_mask = output['crisp_masks']
                sam_mask = output['sam_masks']
                ious = output['ious']
                obj_ptr = output['obj_ptr']
                sam_loss, crisp_loss, loss = loss_func(
                    masks_original_pred=sam_mask,
                    masks_refined_pred=crisp_mask,
                    masks_gt=labels_cls,
                    iou_pred=ious,
                    obj_pred=obj_ptr,
                )
                loss_step_avg += loss.item()
                loss.backward()
                optimizer.step()
                print(
                    f'[RANK {rank}: GPU {gpu}] ITER-{iter_num} --- loss {loss.item()}, sam_loss, {sam_loss.item()}, crisp_loss {crisp_loss.item()}')
            iter_num += 1

        loss_step_avg /= len(organ_name_list)
        sam_loss_step_avg /= len(organ_name_list)
        crisp_loss_step_avg /= len(organ_name_list)
        print(
            f'[RANK {rank}: GPU {gpu}] AVG loss {loss_step_avg}, sam_loss, {sam_loss_step_avg}, crisp_loss {crisp_loss_step_avg}')
        if rank == 0:
            args.writer.add_scalar('train_iter/loss', loss_step_avg, iter_num)
            args.writer.add_scalar('train_iter/sam_loss', sam_loss_step_avg, iter_num)
            args.writer.add_scalar('train_iter/crisp_loss', crisp_loss_step_avg, iter_num)

        epoch_loss += loss_step_avg
        epoch_sam_loss += sam_loss_step_avg
        epoch_crisp_loss += crisp_loss_step_avg
    scheduler.step()
    epoch_loss /= len(train_dataloader) + 1e-12
    epoch_sam_loss /= len(train_dataloader) + 1e-12
    epoch_crisp_loss /= len(train_dataloader) + 1e-12
    print(f'{args.model_save_path} ==> [RANK {rank}: GPU {gpu}] ',
          'epoch_loss: {}, sam_loss: {} crisp_loss'.format(epoch_loss, epoch_sam_loss, epoch_crisp_loss))
    if rank == 0:
        args.writer.add_scalar('train/loss', epoch_loss, epoch)
        args.writer.add_scalar('train/sam_loss', epoch_sam_loss, epoch)
        args.writer.add_scalar('train/crisp_loss', epoch_crisp_loss, epoch)
        args.writer.add_scalar('train/lr', scheduler.get_lr(), epoch)
    return epoch_loss, iter_num


def freeze_components(model, component_names):
    for name, param in model.named_parameters():
        for component in component_names:
            if component in name:
                param.requires_grad = False


def unfreeze_components(model, component_names):
    for name, param in model.named_parameters():
        for component in component_names:
            if component in name:
                param.requires_grad = True


def main_worker(gpu, ngpus_per_node, args):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = ngpus_per_node  # args.world_size
    print(f"[Rank {rank}]: Use GPU: {gpu} for segment_anything_training")
    is_main_host = rank == 0
    if is_main_host:
        os.makedirs(args.model_save_path, exist_ok=True)
        shutil.copyfile(__file__, os.path.join(args.model_save_path, args.run_id + '_' + os.path.basename(__file__)))
    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.init_method,
        rank=rank,
        world_size=world_size,
    )
    print('init_process_group finished')

    # sam_model = build_sam2_hf(model_id=args.model_id)
    sam_model = build_sam2(config_file=args.config_file, checkpoint=args.sam2_ckpt,
                           mode="train")  # checkpoint for pretrained sam2
    model = CrispSam2(
        image_encoder=sam_model.image_encoder,
        memory_attention=sam_model.memory_attention,
        memory_encoder=sam_model.memory_encoder,
        clip_text_ckpt=args.clip_text_ckpt,
        clip_image_ckpt=args.clip_image_ckpt,
    ).cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.num_epochs)

    # %% train
    stage1_epochs = args.stage_one
    stage2_epochs = args.stage_two
    num_epochs = args.num_epochs
    warmup_epochs = args.warmup_epoch
    save_epochs = args.save_epoch
    assert stage1_epochs + stage2_epochs == num_epochs, "Total epoches should be the same as the sum of stage ones and stage two"

    lr_one = args.lr_one
    lr_two = args.lr_two

    iter_num = 0

    train_dataloader = get_loader(args)

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(rank, "=> loading checkpoint '{}'".format(args.resume))
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
            scheduler.last_epoch = start_epoch
            print(rank, "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        torch.distributed.barrier()

    if rank == 0:
        args.writer = SummaryWriter(log_dir='./tb_log/' + args.run_id)
        print('Writing Tensorboard logs to ', './tb_log/' + args.run_id)

    # stage one
    for epoch in range(start_epoch, min(start_epoch + stage1_epochs, num_epochs)):
        freeze_components(model, args.stage_one_frozen)
        if epoch < start_epoch + warmup_epochs:  # warm up
            scheduler.step(epoch)
        else:
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=stage1_epochs - warmup_epochs, power=0.9,
                                                              last_epoch=epoch - warmup_epochs)
            optimizer.param_groups[0]['lr'] = lr_one

        with model.join():
            epoch_loss, iter_num = train_epoch(args, model, train_dataloader, optimizer, scheduler, epoch, rank, gpu,
                                               iter_num)

        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
        # save the model checkpoint
        if is_main_host and (epoch + 1) % save_epochs == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.model_save_path, f'medsam_model_e{epoch + 1}.pth'))
        torch.distributed.barrier()

    # stage two
    for epoch in range(start_epoch + stage1_epochs, num_epochs):
        # unfreeze_components(model, args.stage_two_unfrozen)
        freeze_components(model, args.stage_two_frozen)
        optimizer.param_groups[0]['lr'] = lr_two

        with model.join():
            epoch_loss, iter_num = train_epoch(args, model, train_dataloader, optimizer, scheduler, epoch, rank, gpu,
                                               iter_num)

        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
        # save the model checkpoint
        if is_main_host and (epoch + 1) % save_epochs == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.model_save_path, f'medsam_model_e{epoch + 1}.pth'))
        torch.distributed.barrier()


def main():
    # set seeds
    torch.manual_seed(2345)
    torch.cuda.empty_cache()
    args = set_parse()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = os.path.join(args.work_dir, args.run_id)
    args.model_save_path = model_save_path


    ngpus_per_node = torch.cuda.device_count()
    print("Spwaning processces, ngpus_per_node={}".format(ngpus_per_node))
    print(f"=====> project save at {args.model_save_path}")
    mp.spawn(main_worker, nprocs = ngpus_per_node, args=(ngpus_per_node, args))

if __name__ == "__main__":
    main()