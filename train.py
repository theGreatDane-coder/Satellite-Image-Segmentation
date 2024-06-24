import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from diceloss import dice_coef_9cat_loss
from classcount import classcount

torch.autograd.set_detect_anomaly(True)

# Sets for training
train_dir_img = 'data/.toy/train/img_subset/'
train_dir_mask = 'data/.toy/train/masks_subset/'

# Sets for test
test_dir_img = 'data/.toy/train/img_subset/'
test_dir_mask = 'data/.toy/train/masks_subset/'

# Checkpoint path
dir_checkpoint = 'checkpoints/'

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    # Creating the Dataset
    train = BasicDataset(train_dir_img, train_dir_mask, img_scale)
    test = BasicDataset(test_dir_img, test_dir_mask, img_scale)
    n_train = len(train)

    # DataLoader Setup
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # Initializing TensorBoard Writer
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    # Info
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')


    # Optimizer Init
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    
    # An exponential scheduler to decrease LR by 5% in each epoch
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer= optimizer, gamma= 0.95) 

    # Calculate Class Counts, Move to Device and Set Data Type
    weights_classes = torch.from_numpy(classcount(train_loader))
    weights_classes = weights_classes.to(device=device, dtype=torch.float32)

    print("Class Distribution", weights_classes)

    # Init the criterion (weighted CE)
    criterion = nn.CrossEntropyLoss(weight = weights_classes) 
    
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0 # Train loss per epoch

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
            
                imgs = batch['image']
                true_masks = batch['mask']

                # Ensure that the input images have the correct number of channels required by the nn
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32) # Move Images to Device
                mask_type = torch.float32 if net.n_classes == 1 else torch.long # For cross entropy loss

                true_masks = true_masks.to(device=device, dtype=mask_type) # Move True Masks to Device
                 
                masks_pred = net(imgs) # Predict Masks
                
                # Convert the prediction to float32 for avoiding nan in loss calculation
                masks_pred = masks_pred.type(torch.float32)

                # Cross Entropy Loss
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                
                pbar.set_postfix(**{'Epoch Loss': epoch_loss/n_train}) # Update pbar

                # Convert model to full precision for optimization of weights
                net.float()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0]) # Update pbar
                global_step += 1 # Update global step
 

        # Add to the tensorboard trainig loss for every epoch
        writer.add_scalar('Loss/train', epoch_loss/n_train, epoch+1)

        # Add to the tensorboard weights and gradients for every epoch
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch+1)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch+1)

        # Decrease the LR by 5% in each epoch
        scheduler.step()
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch+1)

        # Add to the tensorboard test loss for every epoch
        test_score = eval_net(net, test_loader, device)
        logging.info('Test CE Loss: {}'.format(test_score))
        writer.add_scalar('Loss/test', test_score, epoch+1)
        
        # Create checkpoints every 5 epochs
        if (epoch+1) % 5 == 0:
            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')


# Passing the arguments
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=4e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args() #get args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #init the device
    logging.info(f'Using device {device}')

    # n_channels=3 for RGB images
    # n_classes=7 for each class we have
    net = UNet(n_channels=3, n_classes=7, bilinear=True)
    
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)