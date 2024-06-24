import argparse
import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
import matplotlib.pyplot as plt
import glob
from torch.utils.data import DataLoader


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval() # Sets the network in evaluation mode

    # Preprocess the image and convert it to a tensor
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0) # Adds a batch dimension
    img = img.to(device=device, dtype=torch.float32) # Moves the image tensor to the device

    with torch.no_grad():
        # Forward pass through the network
        output = net(img) 

        # Applies Softmax across the classes dimension for multi-class segmentation
        probs = F.softmax(output, dim=1)
        
        probs = probs.squeeze(0) # Removes the batch dimension

        # Getting the height and width of the probability map
        height, width = probs.shape[1], probs.shape[2]

        # Mapping probability maps to RGB values for visualization
        mapping = {0: (0  , 255, 255),     #urban_land
                   1: (255, 255, 0  ),     #agriculture
                   2: (255, 0  , 255),     #rangeland
                   3: (0  , 255, 0  ),     #forest_land
                   4: (0  , 0  , 255),     #water
                   5: (255, 255, 255),     #barren_land
                   6: (0  , 0  , 0  )}     #unknown
        
        # Finds the class with the highest probability for each pixel
        class_idx = torch.argmax(probs, dim=0) 

        # Initializes an image tensor to store RGB values 
        image = torch.zeros(height, width, 3, dtype=torch.uint8)

        # Assigns RGB values based on the class index
        for k in mapping:
          idx = (class_idx == torch.tensor(k, dtype=torch.uint8))
          validx = (idx == 1)
          image[validx,:] = torch.tensor(mapping[k], dtype=torch.uint8)

        # Rearranges the dimensions to match the expected format (channels, height, width)         
        image = image.permute(2, 0, 1)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(), # Converts tensor to PIL image
                transforms.Resize(full_img.size[1]), # Resizes image to original dimensions
                transforms.ToTensor() # Converts PIL image back to tensor
            ]
        )

        image = image.permute(1,2,0)
        image = image.squeeze().cpu().numpy()

    return image, class_idx # Applies the transformations and returns the result

def preprocess_mask(pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        torch.set_printoptions(edgeitems=10)
        return img_trans

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()

def RGB_2_class_idx(mask_to_be_converted):
    mapping = {(0  , 255, 255): 0,     #urban_land
                (255, 255, 0  ): 1,    #agriculture
                (255, 0  , 255): 2,    #rangeland
                (0  , 255, 0  ): 3,      #forest_land
                (0  , 0  , 255): 4,      #water
                (255, 255, 255):5,     #barren_land
                (0  , 0  , 0  ):6}           #unknown

    temp = np.array(mask_to_be_converted)
    temp = np.where(temp>=128, 255, 0)
    class_mask=torch.from_numpy(temp)
    h, w = mask_to_be_converted.shape[2], mask_to_be_converted.shape[3]
    img_no=mask_to_be_converted.shape[0]
    mask_out = torch.zeros(img_no,h, w, dtype=torch.long)

    
    for j in range (img_no):
      class_index=0
      for k in mapping:
        idx = (class_mask[j,:,:,:] == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        validx = (idx.sum(0) == 3)
        temp=mask_out[j,:,:]      
        temp[validx]=class_index
        mask_out[j,:,:]=temp
        class_index+=1
    
    return mask_out

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def compute_iou(predicted, actual,num_calsses):
        intersection = 0
        union = 0
        iou=np.zeros((num_calsses-1),dtype=float)
        for k in range(num_calsses-1):
            a = (predicted == k).int()
            b = (actual == k).int()
            intersection = torch.sum(torch.mul(a, b))
            union        = torch.sum(((a + b) > 0).int())
            iou[k]=intersection/union
        mean_iou=(1/(num_calsses-1))*np.sum(iou)
        return mean_iou

if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    
    img_scale=args.scale
    dir_mask='path for masked test data'

    gt_list=[]
    mask_dirs=sorted( filter( os.path.isfile, glob.glob(dir_mask+'*') ) )
    
    for filename in mask_dirs:

        mask = Image.open(filename)
        resized_mask=preprocess_mask(mask, img_scale)      
        mask=np.asarray(resized_mask)     
        gt_list.append(mask)     
    
    gt_tensor=torch.Tensor(gt_list)
    net = UNet(n_channels=3, n_classes=7)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    first= True
    for pattern in args.input:
        # Expand wildcard patterns into actual filenames
        for fn in glob.glob(pattern):

            logging.info("\nPredicting image {} ...".format(fn))

            img = Image.open(fn)
            
            #Splitting input directory from the file name
            name=fn.split('\\')[-1]
            #Removing the file extension
            name=name.split('.')[0]
            seg, mask_indices = predict_img(net=net,
                            full_img=img,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device)
            
            if first:
                seg_array=mask_indices.unsqueeze(0)
                first = False
            else:
                seg_array=torch.cat((seg_array,mask_indices.unsqueeze(0)),0)

            if args.viz:
                im = Image.fromarray(seg)
                im.save(str(args.output[0])+'pred_'+name+'.jpeg')


    #Metric Evaluation 
    gt_tensor=RGB_2_class_idx(gt_tensor)
    
    seg_array = seg_array.to(device)
    gt_tensor = gt_tensor.to(device)
    
    num_classes=7
    total_IoU = compute_iou(seg_array, gt_tensor,num_classes)
    
    print("IoU Value:",total_IoU)