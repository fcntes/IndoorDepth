# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

import networks
from layers import disp_to_depth


def parse_args():
    parser = argparse.ArgumentParser(description='Inference on one Single Image.')
    parser.add_argument('--image_path', type=str,
                        help='path to a test image',
                        required=True)
    parser.add_argument("--load_weights_folder",
                        type=str,
                        help="name of model to load",
                        required=True)
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()

def prepare_model_for_test(args, device):
    model_path = args.load_weights_folder
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    decoder_path = os.path.join(model_path, "depth.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    decoder_dict = torch.load(decoder_path, map_location=device)

    encoder = networks.ResnetEncoder(18, False)
    decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, 
        scales=range(1),num_output_channels=3, use_skips=True, PixelCoorModu=True
    )

    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    decoder.load_state_dict(decoder_dict)
    
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    
    return encoder, decoder, encoder_dict['height'], encoder_dict['width']

def inference(args):

    device = torch.device("cpu")
    
    encoder, decoder, thisH, thisW = prepare_model_for_test(args, device)
    image_path = args.image_path
    print("-> Inferencing on image ", image_path)

    with torch.no_grad():
        # Load image and preprocess

        input_image = pil.open(image_path).convert('RGB')
        extension = image_path.split('.')[-1]
        original_width, original_height = input_image.size

        input_image = input_image.crop((16, 16, 640-16, 480-16))
        name_crop = image_path.replace('.'+extension, '_crop.png')
        input_image.save(name_crop)

        crop_width, crop_height = input_image.size


        input_image = input_image.resize((thisW, thisH), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)


        # Norm_pix_coords
        fx = 5.1885790117450188e+02 / (original_width - 2*16)
        fy = 5.1946961112127485e+02 / (original_height - 2*16)

        cx = (3.2558244941119034e+02 -16) / (original_width - 2*16)
        cy = (2.5373616633400465e+02 -16) / (original_height - 2*16)

        feed_height = thisH 
        feed_width = thisW

        Us, Vs = np.meshgrid(np.linspace(0, feed_width - 1, feed_width, dtype=np.float32),
                            np.linspace(0, feed_height - 1, feed_height, dtype=np.float32),
                            indexing='xy')
        Us /= feed_width
        Vs /= feed_height
        Ones = np.ones([feed_height, feed_width], dtype=np.float32)
        norm_pix_coords = np.stack(((Us - cx) / fx, (Vs - cy) / fy, Ones), axis=0)
        norm_pix_coords = torch.from_numpy(norm_pix_coords).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)
        features_tmp = encoder(input_image)
        outputs = decoder(features_tmp, norm_pix_coords)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (crop_height, crop_width), mode="bilinear", align_corners=False)

        # Saving numpy file
        name_dest_npy = image_path.replace('.'+extension, '_depth.npy') 
        print("-> Saving depth npy to ", name_dest_npy)
        scaled_disp, _ = disp_to_depth(disp, 0.1, 10)
        np.save(name_dest_npy, scaled_disp.cpu().numpy())

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        name_dest_im = image_path.replace('.'+extension, '_depth.png')
        print("-> Saving depth png to ", name_dest_im)
        im.save(name_dest_im)

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    inference(args)
