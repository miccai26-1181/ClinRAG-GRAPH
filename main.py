import argparse
import json
import logging
import os
import sys
from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from config import ModelConfig, TrainConfig
from model import PCRCenterAgnosticGraphModel
from dataloaders import build_dataloader
from train_utils import run_epoch, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--centers_json",type=str,
        default=json.dumps({
=
        '''
        add your centers here, e.g.
            "ispy1":     {"csv_path": "/path/to/ispy1.csv",
                          "npy_dir":  ""]
        '''
        }),
        help="Multi-center config as JSON string. Each center maps to {csv_path, npy_dir}.",
    )
    
    p.add_argument("--epochs", type=int, default=10==50)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--save_path", type=str, default=None,
                   help="Checkpoint path, e.g. /path/to/model.pt. If None, no checkpoint will be saved.")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=6)

    p.add_argument("--patience", type=int, default=0,
                   help="Early stopping patience on bal_acc_pcr. 0 disables early stopping.")
    p.add_argument("--min_delta", type=float, default=0.0,
                   help="Minimum improvement in bal_acc_pcr to reset patience.")
    return p.parse_args()


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg) 
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(save_dir: str, rank: int) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"train_rank{rank}.log")

    logger = logging.getLogger(f"train_rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False 
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if rank == 0:
        sh = TqdmLoggingHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def _move_tab_to_device(tab: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in tab.items()}


def main():
    args = parse_args()

    # ---- configs ----
    mcfg = ModelConfig()
    tcfg = TrainConfig()
    lw = tcfg.loss_w 
    if args.epochs is not None:
        tcfg.epochs = args.epochs
    if args.batch_size is not None:
        tcfg.batch_size = args.batch_size
    if args.lr is not None:
        tcfg.lr = args.lr
    if args.weight_decay is not None:
        tcfg.weight_decay = args.weight_decay
    if args.save_path is not None:
        tcfg.save_path = args.save_path
    if args.no_amp:
        tcfg.amp = False
    if args.num_workers is not None:
        tcfg.num_workers = args.num_workers

    if tcfg.save_path:
        if tcfg.save_path.endswith(os.sep) or os.path.isdir(tcfg.save_path):
            tcfg.save_path = os.path.join(tcfg.save_path, "model.pt")

        # 如果没有 .pt 后缀，也当成目录/前缀处理
        elif not tcfg.save_path.endswith(".pt"):
            tcfg.save_path = os.path.join(tcfg.save_path, "model.pt")

    # ---- logger dir ----
    default_log_dir = ""

    if tcfg.save_path:
        ckpt_dir = os.path.dirname(tcfg.save_path)
        log_dir = os.path.join(ckpt_dir, "logs")
    else:
        log_dir = default_log_dir

    logger = setup_logger(save_dir=log_dir, rank=rank)
    logger.info(f"[rank {rank}] local_rank={local_rank} device={device}")
    logger.info(f"[paths] save_path={tcfg.save_path}")
    logger.info(f"[paths] log_dir={log_dir}")


    centers_all = json.loads(args.centers_json)

    train_center_names = ["DUKU",  "ispy1", "Zcenter"]
    test_center_names  = ["Qcenter", "Ycenter"]

    center_id_map_train = {name: i for i, name in enumerate(sorted(train_center_names))}

    train_centers = {k: centers_all[k] for k in train_center_names}
    test_centers  = {k: centers_all[k] for k in test_center_names}

    train_loader = build_dataloader(
        centers=train_centers,
        center_id_map=center_id_map_train,
        batch_size=12,
        num_workers=8,
        shuffle=True,
        normalize=False,    #true
        strict=True,
        pin_memory=True,
        drop_last=True
    )
    mcfg.num_centers = len(center_id_map_train)

    # build model
    model = PCRCenterAgnosticGraphModel(mcfg, loss_w=lw).to(device)

    # ---- sanity check: only rank0 ----
    if rank == 0:
        model.eval()
        with torch.no_grad():
            first_batch = None
            for b in train_loader:
                if b is not None:
                    first_batch = b
                    break
            if first_batch is None:
                raise RuntimeError("All batches are empty (all samples skipped). Check dataset roots/phases.")

            x = first_batch["x_dce"]
            tab = first_batch["tab"]
            y = first_batch["y"]
            logger.info(f"[SanityCheck] y unique={torch.unique(y).detach().cpu().tolist()}  min={y.min().item():.4f}  max={y.max().item():.4f}")



            logger.info("[SanityCheck] batch tensors:")
            logger.info(f"  x_dce     : {tuple(x.shape)} {x.dtype}")
            for k, v in tab.items():
                logger.info(f"  tab[{k}]  : {tuple(v.shape)} {v.dtype}")
            logger.info(f"  y         : {tuple(y.shape)} {y.dtype}")

            x = x.to(device)
            tab = _move_tab_to_device(tab, device)

            out = model(x_dce=x, tab=tab,use_adv=True)
            y_logit = out["y_logit"]
            adv_logit = out["adv_logit"]

            logger.info("[SanityCheck] forward outputs:")
            logger.info(f"  y_logit : {tuple(y_logit.shape)} {y_logit.dtype}")
            logger.info(f"  adv_logit : {tuple(adv_logit.shape)} {adv_logit.dtype}")

            assert y_logit.shape == (x.size(0),)
            assert adv_logit.shape == (x.size(0), mcfg.num_centers)

        model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(tcfg.amp and device.type == "cuda"))

    best_bal = -float("inf") 
    no_improve = 0
    last_epoch_ran = 0


    epoch_iter = range(1, tcfg.epochs + 1)
    if rank == 0:
        epoch_iter = tqdm(
            epoch_iter,
            desc="Epoch",
            dynamic_ncols=True,
            mininterval=1,   
            maxinterval=2.0,   
            smoothing=0.1,
        )

    for epoch in epoch_iter:
        if rank == 0:
            logger.info(f"Epoch {epoch}/{tcfg.epochs}")

        # ====== wrap loader with tqdm (rank0 only) ======
        train_loader_epoch = train_loader

        if rank == 0:
            train_loader_epoch = tqdm(
                train_loader,
                desc=f"Train [{epoch}/{tcfg.epochs}]",
                dynamic_ncols=True,
                leave=False,
                mininterval=0.5,
                maxinterval=2.0,
                smoothing=0.1,
            )

        tr = run_epoch(
            model=model,
            loader=train_loader_epoch,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp=(tcfg.amp and device.type == "cuda"),
            train=True,            lambda_adv=lw.adv,
            log_every=tcfg.log_every if rank == 0 else 0,
        )

        if rank == 0 and isinstance(epoch_iter, tqdm):
            epoch_iter.set_postfix({
                "tr_total": f"{tr.get('L_total', 0):.3f}",
                "tr_pcr": f"{tr.get('L_pcr', 0):.3f}",
                "tr_bal": f"{tr.get('bal_acc_pcr', 0):.3f}",
                "tr_rec": f"{tr.get('recall_pcr', 0):.3f}",
            })

        if rank == 0:
            logger.info(
                f"train: L_total={tr.get('L_total', 0):.4f}, L_pcr={tr.get('L_pcr', 0.0):.4f}, bal_acc={tr.get('bal_acc_pcr', 0.0):.4f}, recall={tr.get('recall_pcr', 0.0):.4f}, spec={tr.get('spec_pcr', 0.0):.4f}, acc_pcr(ref)={tr.get('acc_pcr', 0.0):.4f}, pred_pos_rate={tr.get('pred_pos_rate@0.5', 0.0):.4f}"
            )
        last_epoch_ran = epoch


    # save last epoch checkpoint (rank0 only)
    if rank == 0 and tcfg.save_path:
        to_save = model.module if use_ddp else model
        ckpt_dir = os.path.dirname(tcfg.save_path)
        os.makedirs(ckpt_dir, exist_ok=True)
        last_path = os.path.join(ckpt_dir, "last_epoch.pth")
        save_checkpoint(
            to_save,
            optimizer,
            last_path,
            tcfg.epochs,
            extra={"center_id_map_train": center_id_map_train, "model_cfg": mcfg.__dict__, "train_cfg": tcfg.__dict__},
        )
        logger.info(f"saved last checkpoint to {last_path} (epoch={last_epoch_ran})")

if __name__ == "__main__":
    main()