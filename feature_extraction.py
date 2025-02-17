import os
from pathlib import Path
import yaml
import shutil

from data import LANSFileDataset
import numpy as np
from dlup.data.dataset import WsiAnnotations, TiledWsiDataset, TilingMode
from dlup import SlideImage
import torch
from torch.utils.data import DataLoader, TensorDataset
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
import argparse
from models import ConvStem
from PIL import Image, ImageDraw


def extract_rois(wsi, wsa, margin=1000):
    """ Extracts ROIs from a whole slide annotation. Should be able to extract multiple ROI's from one annotation xml.
    Todo: Verify that rois don't cross the bounds of the slide. Is the margin necessary?

    Parameters:
        wsi: WsiImage
        wsa: WsiAnnotation
        margin: amount of space (in pixels) between the ROI and each of the borders of the image

    Returns:
        rois: list of rois in tuple format
    """
    for k, v in wsa._annotations.items():

        if k.label == 'biopsy-outlines':
            bboxes = v.bounding_boxes
            bb_len = np.asarray(bboxes).shape[0]
            rois = []

            for i in range(bb_len):
                box = bboxes[i]
                x_sum, y_sum = tuple([a + b for a, b in zip(box[0], box[1])])

                # get the slide bounds
                x_min, x_max = wsi.slide_bounds[0][0], wsi.slide_bounds[1][0]
                y_min, y_max = wsi.slide_bounds[0][1], wsi.slide_bounds[1][1]
                x, y, x_t, y_t = box[0][0], box[0][1], box[1][0], box[1][1]

                # check and fix boundaries: dx # how much space there is left within
                dx_right = x_max - x_sum
                dy_bottom = y_max - y_sum

                if x < x_min:
                    x_t = x_t + x
                    x = x_min

                if y < y_min:
                    y_t = y_t + y
                    y = y_min

                if dx_right < margin:
                    x_t = x_t + dx_right - margin

                if dy_bottom < margin:
                    y_t = y_t + dy_bottom - margin

                # make the resized ROI
                box_new = tuple((tuple((x, y)), tuple((x_t, y_t))))
                rois_np = np.asarray(box_new).astype('int')
                rois.append(tuple((tuple((rois_np[0, 0], rois_np[0, 1])), tuple((rois_np[1, 0], rois_np[1, 1])))))

            return rois


def extract_features(sample, model_name, target_mpp=1, tile_size=(224, 224), tile_overlap=(0, 0), batch_size=64,
                     mask_threshold=0.2):
    wsi_path, wsa_path, wsm_path, block_id = sample
    print('Processing: {}, file: {}'.format(block_id, wsi_path))

    # open image and annotation file
    wsi = SlideImage.from_file_path(wsi_path)
    wsa = WsiAnnotations.from_asap_xml(wsa_path)
    wsm = SlideImage.from_file_path(wsm_path)

    # make the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(
        model_name=model_name,
        embed_layer=ConvStem,  # defined above
        pretrained=True,
        num_classes=None
    ).eval().to(device)

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # extract rois + dataset
    rois = extract_rois(wsi, wsa)
    dataset = TiledWsiDataset.from_standard_tiling(wsi_path,
                                                   mpp=target_mpp,
                                                   tile_size=tile_size,
                                                   tile_overlap=tile_overlap,
                                                   mask_threshold=mask_threshold,
                                                   tile_mode=TilingMode.skip,
                                                   rois=rois,
                                                   mask=wsm)
    # extract coordinates and features
    coords = [np.array(d["coordinates"]) for d in dataset]
    patches_tensor = torch.stack([transforms(d['image'].convert('RGB')) for d in dataset])
    tensor_dataset = TensorDataset(patches_tensor)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        feats = torch.cat([model(x[0].to(device)) for x in dataloader],
                          dim=0)  # (batch_size, num_channels, img_size, img_size)
        print('Features shape: {}\n'.format(feats.shape))

    # generate a thumbnail for verification
    scaled_region_view = wsi.get_scaled_view(wsi.get_scaling(target_mpp))
    thumb = Image.new("RGBA", tuple(scaled_region_view.size), (255, 255, 255, 255))

    for d in dataset:
        tile = d["image"]
        coords = np.array(d["coordinates"])
        box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))
        thumb.paste(tile, box)
        draw = ImageDraw.Draw(thumb)
        draw.rectangle(box, outline="red")

    return feats, coords, thumb  # feats = (batch_size, num_features) shaped tensor


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/data/archief/AMC-data/Barrett/LANS/')
    parser.add_argument("--config_file", type=str,
                        default='/home/mbotros/code/lans_weaksupervised/configs/extract_config.yaml')
    args = parser.parse_args()

    # make file dataset
    lans_file_dataset = LANSFileDataset(Path(args.data_path))

    # load config for extraction
    with open(args.config_file) as file:
        config = yaml.safe_load(file)

    # store the config in the output folder
    os.makedirs(args.output_path, exist_ok=True)
    shutil.copy(args.config_file, args.output_path)

    # start extraction
    for sample in lans_file_dataset:
        features, coordinates, thumbnail = extract_features(sample,
                                                            model_name=config['model']['name'],
                                                            target_mpp=config['data']['target_mpp'],
                                                            tile_size=config['data']['tile_size'],
                                                            tile_overlap=config['data']['tile_overlap'],
                                                            batch_size=config['data']['batch_size'])

        # store the features and coordinates
        block_id = sample[-1]
        coord_file = os.path.join(args.output_path, block_id + '-coords.npy')
        feat_file = os.path.join(args.output_path, block_id + '-features.pt')
        thumb_file = os.path.join(args.output_path, block_id + '-thumb.png')
        thumbnail.save(thumb_file, optimize=True, quality=95)
        np.save(file=coord_file, arr=coordinates)
        torch.save(obj=features, f=feat_file)
