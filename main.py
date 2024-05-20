#!/usr/bin/env python3
import argparse
import torch
import sys
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from lightning_trainer import UnetDACLighting
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from audio_dataset import DictTorchPartedDataset, PinDictTorchPartedDataset

from unet_dac import UnetDAC
import lightning as L

def init_logger():
    logger.remove()
    logger.add(sys.stdout, level='DEBUG')

def parse_args():
    parser = argparse.ArgumentParser(description="DOA model trainer/tester")

    # Add arguments for train mode
    parser.add_argument('--train', action='store_true', help='Enable training mode')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for training (default: 1e-3)')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Max epochs for training (default: 100)')
    parser.add_argument('--early-stopping-patience', type=int, default=3,
                        help='Early stopping patience for training (default: 3)')
    parser.add_argument('--train-pt-prefix', type=str, default='train06r076v3',
                        help='Prefix of train data "[prefix]_[index].pt" files on data dir '
                        '(default: "train06r076v3")')
    parser.add_argument('--val-pt-prefix', type=str, default='validation06r076v3',
                        help='Prefix of validation data "[prefix]_[index].pt" files on data dir '
                        '(default: "validation06r076v3")')

    # Add arguments for test mode
    parser.add_argument('--test', action='store_true', help='Enable testing mode')
    parser.add_argument('--test-pt-prefix', type=str, default='test2r0168v4', help='Prefix of test'
                        ' data "[prefix]_[index].pt" files on data dir (default: "test2r0168v4")')

    # Common args
    parser.add_argument('--checkpoint-path', type=str, default='',
                        help='Path to the model checkpoint for testing')
    parser.add_argument('--data-dir', type=str, default='data_batches',
                        help='Directory of data batches (default: "data_batches")')
    parser.add_argument('--device', type=str, default='cuda', help='Device to load/run the model, should be "cpu" or "cuda" (default: "cuda")')
    
    args = parser.parse_args()

    # Check if at least one mode is selected
    if not (args.train or args.test):
        parser.error('At least one mode (train or test) must be selected.')

    if not args.train and args.test and not args.checkpoint_path:
        parser.error('Test-only requires checkpoint path.')

    return args

def main():
    args = parse_args()
    # 1. Initialise common variables
    # 1.1. Create torch module
    model_name = f"unet_doa_bs{args.batch_size}"
    device = torch.device('cpu' if not torch.cuda.is_available() else args.device)
    logger.info(f"Running on device {device}. torch.cuda.is_available():{torch.cuda.is_available()}")

    learaning_logger = TensorBoardLogger("tb_logs", name=model_name)
    model = UnetDAC().to(device)
    
    criterion = nn.CrossEntropyLoss()

    # 1.2. Load checkpoint if available
    if args.checkpoint_path:
        model_lighting = UnetDACLighting.load_from_checkpoint(args.checkpoint_path, model=model,
                                                              loss_fn=criterion)
        logger.info(f"Loaded model from checkpoint {args.checkpoint_path}")
    elif not args.train and args.test:
        logger.error("No checkpoint path provided for testing.")
        exit(1)
    else:
        model_lighting = UnetDACLighting(model, criterion, lr=args.lr)
        logger.info(f"No checkpoint was provided - training a new model")
    
    # 1.3. Create trainer
    trainer = L.Trainer(max_epochs=args.max_epochs,
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min",
                                                 patience=args.early_stopping_patience)],
                        default_root_dir=model_name,
                        log_every_n_steps=10,
                        logger=learaning_logger)

    # 2. Train
    if args.train:
        logger.info(f"Training mode selected.")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.lr:.1e}")
        
        train_dataset = PinDictTorchPartedDataset(args.data_dir, args.train_pt_prefix ,
                                                  ['samples', 'target'], real_batch_size=64,
                                                  virtual_batch_size=1, device=device)
        validation_dataset = PinDictTorchPartedDataset(args.data_dir, args.val_pt_prefix,
                                                       ['samples', 'target'], real_batch_size=64,
                                                       virtual_batch_size=1, device=device)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=4, persistent_workers=True, prefetch_factor=16)
        valiadtion_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size,
                                           shuffle=False, num_workers=1, persistent_workers=True,
                                           prefetch_factor=16)
        trainer.fit(model_lighting, train_dataloaders=train_dataloader,
                    val_dataloaders=valiadtion_dataloader)

    # 3. Test
    if args.test:
        logger.info("Testing mode selected.")
        test_dataset = PinDictTorchPartedDataset(args.data_dir, args.test_pt_prefix,
                                                 ['samples', 'ref_stft', 'target', 'mixed_signals',
                                                  'perceived_signals', 'radii', 'reverbs'],
                                                 real_batch_size=60, virtual_batch_size=1,
                                                 device=device)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        trainer.test(model_lighting, dataloaders=test_dataloader)


if __name__ == '__main__':
    exit(main())
