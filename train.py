import argparse
from models import AttentionMIL
import seaborn as sns
import wandb
import yaml
import os
import pandas as pd
from matplotlib import pyplot as plt
from data import BagDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryPrecision, BinaryRecall, ConfusionMatrix
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def train(args):
    """
    - Binary pipeline
    (*) At the end of training run on validation set and store:
    (*)   * ROC plot
    Multi-class pipeline
    (*)
    (*)
     - Data prep
    (*) Split: (1) make a file for training (till RL-0950) that marks the SKMS cases to be removed
    (*) Split: (2) split on case level
    (*) Add cases: RL-0927 till RL-0950 (also extract tissue masks => do for the whole dataset while at it)
    (*) Use Ylva's label file instead
    - Feature extraction
    (*) Extract other features with FOMO models: (1) Conch, (2) Virchow2, (3) Prov-GigaPath
    (*) Try other spacings: 0.5 mpp and 2 mpp
    """

    dataset = BagDataset(
        features_dir=args.features_dir,
        label_file=args.label_file,
        binary=True)

    print('Total length dataset: {}'.format(len(dataset)))
    labels = np.array(dataset.labels)
    _, counts = np.unique(labels, return_counts=True)
    print('non-dysplastic: {}, dysplastic: {}'.format(counts[0], counts[1]))
    print('Using features from: {}'.format(args.features_dir))

    # load config for extraction
    with open(os.path.join(args.features_dir, 'extract_config.yaml')) as file:
        feat_extraction_config = yaml.safe_load(file)

    # load the feature extraction config that was used
    config = {'feature_extraction_config': feat_extraction_config,
              'nr_epochs': args.nr_epochs,
              'batch_size': args.batch_size,
              'lr': args.lr,
              'wd': args.wd,
              'k_folds': args.k_folds}

    print('Running {} folds total'.format(args.k_folds))
    for fold, train_loader, val_loader in get_dataloaders(dataset, k_folds=args.k_folds, batch_size=args.batch_size):

        fold_dir = os.path.join(args.exp_dir, '{}_fold_{}'.format(args.run_name, fold))
        os.makedirs(fold_dir, exist_ok=True)

        print('Starting fold {}'.format(fold))
        run = wandb.init(project='WeakBE-Net_binary_CTransPath',
                         id='{}_fold_{}'.format(args.run_name, fold),
                         name='{}_fold_{}'.format(args.run_name, fold),
                         config=config,
                         dir=fold_dir)

        wandb_logger = WandbLogger(log_model=True)
        checkpoint_callback = ModelCheckpoint(dirpath=fold_dir,
                                              save_top_k=1,
                                              monitor='val_loss',
                                              mode='min',
                                              filename='best_model')

        model = BinaryMILModel(feature_dim=feat_extraction_config['model']['feature_dim'],
                               lr=args.lr,
                               wd=args.wd,
                               run_dir=fold_dir)

        trainer = pl.Trainer(max_epochs=args.nr_epochs,
                             devices=[0],
                             logger=wandb_logger,
                             log_every_n_steps=1,
                             callbacks=[checkpoint_callback])

        trainer.fit(model, train_loader, val_loader)

        # Load best model and validate
        best_model_path = checkpoint_callback.best_model_path
        print('Best model for fold {} saved at: {}'.format(fold, best_model_path))
        best_model = BinaryMILModel.load_from_checkpoint(best_model_path,
                                                         feature_dim=feat_extraction_config['model']['feature_dim'],
                                                         run_dir=fold_dir)

        best_model.final_validation = True
        trainer.validate(best_model, dataloaders=val_loader)
        run.finish()


class BinaryMILModel(pl.LightningModule):
    """ Implements a standard MIL model for binary classification.
    """

    def __init__(self, feature_dim=1000, hidden_dim=1024, lr=1e-5, wd=1e-4, output_dim=1, run_dir=None):
        super(BinaryMILModel, self).__init__()
        self.model = AttentionMIL(feature_dim, hidden_dim, output_dim=output_dim)
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.wd = wd

        # Metrics
        self.accuracy = BinaryAccuracy()
        self.auroc = BinaryAUROC()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()

        # Final validation rounds, store predictions and confusion matrix
        self.conf_matrix = ConfusionMatrix(num_classes=2, task="binary")
        self.val_probs = []
        self.val_preds = []
        self.val_labels = []
        self.val_block_ids = []
        self.final_validation = False
        self.run_dir = run_dir

    def forward(self, bag_features, mask):
        logits, attn_weights = self.model(bag_features, mask)
        return logits.squeeze(0), attn_weights

    def compute_loss(self, logits, labels, masks):
        """ Computes the loss only for valid bags (having at least one non-padded instance).
        logits: (batch_size)
        labels: (batch_size)
        masks: (batch_size, bag_size)
        """
        # only consider bags with valid instances
        assert (len(logits) > 0)
        valid_bags = masks.sum(dim=1) > 0
        valid_predictions = logits[valid_bags]
        valid_labels = labels[valid_bags]
        loss = self.criterion(valid_predictions, valid_labels)
        return loss

    def training_step(self, batch, batch_idx):
        bag_features, masks, labels, _, _ = batch
        assert (len(bag_features) > 0)
        logits, _ = self(bag_features, masks)
        assert (len(logits) > 0)
        loss = self.compute_loss(logits, labels.float(), masks)

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.accuracy(logits, labels), on_step=True, on_epoch=True)
        self.log('train_auc', self.auroc(logits, labels), on_step=True, on_epoch=True)
        self.log('train_precision', self.precision(logits, labels), on_step=True, on_epoch=True)
        self.log('train_recall', self.recall(logits, labels), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        bag_features, masks, labels, _, block_ids = batch
        logits, _ = self(bag_features, masks)
        loss = self.compute_loss(logits, labels.float(), masks)

        # Store predictions if final validation
        if self.final_validation:
            probs = torch.sigmoid(logits)
            preds = probs > 0.5
            self.val_probs.append(probs)
            self.val_preds.append(preds)
            self.val_labels.append(labels)
            self.val_block_ids.append(block_ids)
            self.log('final_val_acc', self.accuracy(logits, labels), on_epoch=True)
            self.log('final_val_auc', self.auroc(logits, labels), on_epoch=True)
        else:
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_acc', self.accuracy(logits, labels), prog_bar=True, on_epoch=True)
            self.log('val_auc', self.auroc(logits, labels), prog_bar=True, on_epoch=True)
            self.log('val_precision', self.precision(logits, labels), prog_bar=True, on_epoch=True)
            self.log('val_recall', self.recall(logits, labels), prog_bar=True, on_epoch=True)

        return loss

    def compute_confusion_matrix(self, preds, labels):
        """Generate and log confusion matrix"""
        conf_mat = self.conf_matrix(preds, labels).cpu().numpy()
        plt.figure(figsize=(5, 5))
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Dysplastic", "Dysplastic"],
                    yticklabels=["Non-Dysplastic", "Dysplastic"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Validation Confusion Matrix")
        wandb.log({"Confusion Matrix": wandb.Image(plt)})
        plt.close()

    def on_validation_epoch_end(self):
        """Compute confusion matrix only at final validation round"""
        if self.final_validation:
            preds = torch.cat(self.val_preds)
            probs = torch.cat(self.val_probs)
            labels = torch.cat(self.val_labels)
            block_ids = [item for tup in self.val_block_ids for item in tup]
            print('Final evaluation on: {} samples'.format(len(preds)))
            self.compute_confusion_matrix(preds, labels)

            # store prediction results as csv: to-do add probabilities
            results_df = pd.DataFrame({'block_id': block_ids,
                                       'label': labels.cpu().numpy(),
                                       'pred': preds.cpu().numpy(),
                                       'prob': probs.cpu().numpy()})
            print(results_df)
            results_save_path = os.path.join(self.run_dir, 'results.csv')
            print('Saving results to: {}'.format(results_save_path))
            results_df.to_csv(results_save_path, index=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)


def collate_fn(batch):
    """ Makes sure batches are the same length by padding. A mask is used to keep track of padded instances.

    Args:
        batch: (n_feat, n_labels, n_coords, n_block_ids)
    """
    features, labels, coords, block_ids = zip(*batch)
    max_patches = max(f.shape[0] for f in features)

    padded_features = []
    masks = []
    for f in features:
        pad_size = max_patches - f.shape[0]
        padded = F.pad(f, (0, 0, 0, pad_size))
        mask = torch.cat([torch.ones(f.shape[0]), torch.zeros(pad_size)])
        padded_features.append(padded)
        masks.append(mask)

    return torch.stack(padded_features), torch.stack(masks), torch.tensor(labels), coords, block_ids


def get_dataloaders(dataset, k_folds=5, batch_size=4):
    """ Splits the data for KFold cross validation
    todo: assure split on case level: how?

    Args:
        dataset:
        k_folds:
        batch_size:
    Returns:
         fold:
         train_loader:
         val_loader:
    """
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        fold = fold + 1
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        print("Size train_subset {}".format(len(train_subset)))
        print("Size val_subset {}".format(len(val_subset)))

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        yield fold, train_loader, val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='baseline_t', help="the name of this experiment")
    parser.add_argument("--nr_epochs", type=int, default=1500, help="the number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="the size of mini batches")
    parser.add_argument("--lr", type=float, default=1e-5, help="initial the learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay (L2)")
    parser.add_argument("--k_folds", type=int, default=5, help="number of folds")
    parser.add_argument("--exp_dir", type=str,
                        default='/home/mbotros/experiments/lans_weaklysupervised/')
    parser.add_argument("--features_dir", type=str,
                        default='/data/archief/AMC-data/Barrett/LANS_features/CTransPath')
    parser.add_argument("--label_file", type=str,
                        default='/data/archief/AMC-data/Barrett/LANS/lans_consensus_no_ind_nbde=0_lgd=1_hgd=2.csv')
    parser.add_argument("--wandb_key", type=str, help="key for logging to weights and biases")
    parser.add_argument("--test", type=bool, help="whether to also test", default=True)
    args = parser.parse_args()

    train(args)
