from pathlib import Path
import pandas as pd
import os
import torch
import numpy as np


def filter_image_files(image_files):

    # get all blocks identifiers
    block_ids = sorted(list(set([str(f).split('/')[-1].split('HE')[0] for f in image_files])))
    block_identifiers = []
    files = []

    for block_id in block_ids:
        # find all files with this block id
        file = [f for f in image_files if block_id in str(f)][0] # select the first file (1) HE.tiff or (2) HE_1.tiff if not (1) not there
        block_identifiers.append((block_id[:-1]))
        files.append(file)

    return block_identifiers, files


class LANSFileDataset:
    """
    A file dataset class used to extract features from a set of WSIs.
    """

    def __init__(self, data_dir):

        # location of data
        self.data_dir = data_dir

        # load the images
        self.image_files = [f for f in sorted(data_dir.rglob("*.tiff")) if 'HE' in str(f) and 'tm' not in str(f)]

        # only take the first file for each block identifier (otherwise we process so much)
        self.block_identifiers, self.image_files = filter_image_files(self.image_files)

        # get the corresponding annotation and mask files
        self.annotation_files = [str(f).split('.')[0] + '.xml' for f in self.image_files]
        self.mask_files = [str(f).split('.')[0] + '_tm.tiff' for f in self.image_files]

        print('Selected {} H&E image files!'.format(len(self.image_files)))
        # check if all annotations and masks are present before continue
        for f in self.annotation_files:
            assert (Path(f).exists()), 'Annotation missing: {}'.format(f)
        for f in self.mask_files:
            assert (Path(f).exists()), 'Mask missing: {}'.format(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        return self.image_files[item], self.annotation_files[item], self.mask_files[item], self.block_identifiers[item]


class BagDataset:
    """
    A PyTorch Dataset class for loading pre-extracted feature representations of WSIs
    along with their corresponding labels and spatial coordinates. Each WSI (bag) consists of multiple
    instances (patch-level features).
    """
    def __init__(self, features_dir, label_file, binary=False):
        """
         Initializes the BagDataset object by loading the features, coordinates, and labels from the
         specified directory and label file.

         Parameters:
         ----------
         features_dir : str
             Directory containing `.pt` and `.npy` files with WSI features and patch coordinates, respectively.
         label_file : str
             Path to the CSV file containing labels.
        binary : bool
             Whether to consider a binary setting (nondysplastic vs dysplastic)
        """

        # all files and all labels
        self.feature_files = sorted([f for f in os.listdir(features_dir) if '.pt' in f])
        self.coord_files = sorted([f for f in os.listdir(features_dir) if '.npy' in f])
        self.labels = pd.read_csv(label_file)

        # filter out where we don't have both a file and a label
        block_id_files = [f.split('-features')[0] for f in self.feature_files]
        self.block_ids = [x for x in self.labels['block id'] if x in block_id_files]

        # open the files & load the grades
        self.features = [torch.load(os.path.join(features_dir, b + '-features.pt')) for b in self.block_ids]
        self.coordinates = [np.load(os.path.join(features_dir, b + '-coords.npy')) for b in self.block_ids]

        # convert grades to tensors
        self.labels = [
            torch.tensor(self.labels[self.labels['block id'] == b]['dx'].values[0], dtype=torch.long) for b in self.block_ids
        ]

        if binary:
            self.labels = [torch.tensor(0 if label < 1 else 1, dtype=torch.float64).unsqueeze(0) for label in self.labels]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns:
        - features: Tensor of shape (num_patches, feature_dim).
        - label: Tensor containing the label for the WSI.
        - coordinates: Numpy array of shape (num_patches, 2)
        """
        features = self.features[idx]
        label = self.labels[idx]
        coordinates = self.coordinates[idx]
        block_id = self.block_ids[idx]

        return features, label, coordinates, block_id

