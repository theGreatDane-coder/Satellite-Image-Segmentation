import torch
import torch.nn.functional as F
from tqdm import tqdm

def eval_net(net, loader, device):
    net.eval() # Switches the network to evaluation mode
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # Total number of batches in the loader
    tot = 0 # Accumulate the total loss across all batches

    with tqdm(total=n_val, desc='Validation round', unit='batch',disable = True, leave=True) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            tot += F.cross_entropy(mask_pred, true_masks).item()

            pbar.update()
    net.train()
    return tot/n_val