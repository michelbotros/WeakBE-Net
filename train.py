import argparse
from models import AttentionMIL
import seaborn as sns
import wandb
import yaml
import os
import pandas as pd
from matplotlib import pyplot as plt
from data import BagDataset, get_dataloaders
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryPrecision, BinaryRecall, ConfusionMatrix
from torchmetrics.classification import MulticlassAccuracy
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def train(args):
    """
    - Binary pipeline
    (*) At the end of training run on validation set and store:
    (*)   * ROC plot
     - Data prep
    (*) Split: (1) make a file for training (till RL-0950) that marks the SKMS cases to be removed
    (*) Split: (2) split on case level
    (*) Add cases: RL-0927 till RL-0950 (also extract tissue masks => do for the whole dataset while at it)
    (*) Use Ylva's label file instead
    - Feature extraction
    (*) Extract other features with FOMO models: (1) Conch, (2) Virchow2
    (*) Try other spacings: 0.5 mpp and 2 mpp
    """
    dataset = BagDataset(
        features_dir=args.features_dir,
        label_file=args.label_file,
        binary=args.binary)

    print('Total length dataset: {}'.format(len(dataset)))
    labels = np.array(dataset.labels)
    print('Label counts: {}'.format(np.unique(labels, return_counts=True)))
    print('Using features from: {}'.format(args.features_dir))

    # load config for extraction
    with open(os.path.join(args.features_dir, 'extract_config.yaml')) as file:
        feat_extraction_config = yaml.safe_load(file)

    print('Running {} folds total'.format(args.k_folds))
    for fold, train_loader, val_loader, class_weights in get_dataloaders(dataset, k_folds=args.k_folds, batch_size=args.batch_size):
        fold_dir = os.path.join(args.exp_dir, '{}_fold_{}'.format(args.run_name, fold))
        os.makedirs(fold_dir, exist_ok=True)

        # load the feature extraction config that was used
        config = {'feature_extraction_config': feat_extraction_config,
                  'hidden_dim': args.hidden_dim,
                  'nr_epochs': args.nr_epochs,
                  'batch_size': args.batch_size,
                  'lr': args.lr,
                  'wd': args.wd,
                  'k_folds': args.k_folds,
                  'drop out': args.drop_out,
                  'class weights:': class_weights.numpy()}

        print('Starting fold {}'.format(fold))
        run = wandb.init(project=args.project_name,
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

        model = MILModel(feature_dim=feat_extraction_config['model']['feature_dim'],
                         hidden_dim=args.hidden_dim,
                         lr=args.lr,
                         wd=args.wd,
                         drop_out=args.drop_out,
                         run_dir=fold_dir,
                         binary=args.binary,
                         class_weights=class_weights)

        trainer = pl.Trainer(max_epochs=args.nr_epochs,
                             devices=[0],
                             logger=wandb_logger,
                             log_every_n_steps=1,
                             callbacks=[checkpoint_callback])

        trainer.fit(model, train_loader, val_loader)

        # Load best model and validate
        best_model_path = checkpoint_callback.best_model_path
        print('Best model for fold {} saved at: {}'.format(fold, best_model_path))
        best_model = MILModel.load_from_checkpoint(best_model_path,
                                                   feature_dim=feat_extraction_config['model']['feature_dim'],
                                                   run_dir=fold_dir,
                                                   binary=args.binary)

        best_model.final_validation = True
        trainer.validate(best_model, dataloaders=val_loader)
        run.finish()


class MILModel(pl.LightningModule):
    """ Implements a standard MIL model for classification.
    """

    def __init__(self,
                 feature_dim=512,
                 hidden_dim=512,
                 lr=1e-5,
                 wd=1e-4,
                 drop_out=0.2,
                 class_weights=None,
                 binary=False,
                 run_dir=None):

        super(MILModel, self).__init__()
        self.output_dim = 1 if binary else 3
        self.model = AttentionMIL(feature_dim, hidden_dim, output_dim=self.output_dim, drop_out=drop_out)
        self.lr = lr
        self.wd = wd

        # Use class weights for multi class, binary was almost equal already
        self.criterion = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss(weight=class_weights)

        # Metrics + Confusion Matrix
        if binary:
            self.binary_accuracy = BinaryAccuracy()
            self.binary_auroc = BinaryAUROC()
            self.binary_precision = BinaryPrecision()
            self.binary_recall = BinaryRecall()
            self.metrics = {'accuracy': self.binary_accuracy,
                            'auc': self.binary_auroc,
                            'precision': self.binary_precision,
                            'recall': self.binary_recall}
            self.conf_matrix = ConfusionMatrix(num_classes=2, task="binary")
            self.class_labels = ['Non-Dysplastic', 'Dysplastic']
        else:
            self.multi_class_accuracy = MulticlassAccuracy(num_classes=self.output_dim)
            self.metrics = {'accuracy': self.multi_class_accuracy}
            self.conf_matrix = ConfusionMatrix(num_classes=3, task="multiclass")
            self.class_labels = ['NDBE', 'LGD', 'HGD']

        # for final validation round: store predictions
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
        valid_labels = valid_labels if self.output_dim > 1 else valid_labels.float()
        loss = self.criterion(valid_predictions, valid_labels)
        return loss

    def training_step(self, batch, batch_idx):
        bag_features, masks, labels, _, _ = batch
        assert (len(bag_features) > 0)
        logits, _ = self(bag_features, masks)
        assert (len(logits) > 0)
        labels = labels if self.output_dim > 1 else labels.float()
        loss = self.compute_loss(logits, labels, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        for name, metric in self.metrics.items():
            self.log('train_{}'.format(name), metric(logits, labels), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        bag_features, masks, labels, _, block_ids = batch
        logits, _ = self(bag_features, masks)
        loss = self.compute_loss(logits, labels, masks)

        # Store predictions if final validation
        if self.final_validation:

            if self.output_dim == 1:               # binary
                probs = torch.sigmoid(logits)
                preds = probs > 0.5
            else:                                  # multiclass
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

            self.val_probs.append(probs)
            self.val_preds.append(preds)
            self.val_labels.append(labels)
            self.val_block_ids.append(block_ids)

            for name, metric in self.metrics.items():
                self.log('final_val_{}'.format(name), metric(logits, labels), on_epoch=True)
        else:
            self.log('val_loss', loss, prog_bar=True)
            for name, metric in self.metrics.items():
                self.log('val_{}'.format(name), metric(logits, labels), prog_bar=True, on_epoch=True)

        return loss

    def compute_confusion_matrix(self, preds, labels):
        """Generate and log confusion matrix"""
        conf_mat = self.conf_matrix(preds, labels).cpu().numpy()
        plt.figure(figsize=(5, 5))
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_labels, yticklabels=self.class_labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Validation Confusion Matrix")
        wandb.log({"Confusion Matrix": wandb.Image(plt)})
        plt.close()

    def on_validation_epoch_end(self):
        """Compute confusion matrix only at final validation round"""
        if self.final_validation:
            preds = torch.cat(self.val_preds)
            probs = torch.cat(self.val_probs).cpu().numpy()
            labels = torch.cat(self.val_labels)
            block_ids = [item for tup in self.val_block_ids for item in tup]
            print('Final evaluation on: {} samples'.format(len(preds)))
            self.compute_confusion_matrix(preds, labels)

            # store prediction results as csv: to-do add probabilities
            results_df = pd.DataFrame({'block_id': block_ids,
                                       'label': labels.cpu().numpy(),
                                       'pred': preds.cpu().numpy()})
            if self.output_dim == 1:
                results_df['prob'] = probs
            else:
                results_df['prob_nd'] = probs[:, 0]
                results_df['prob_lgd'] = probs[:, 1]
                results_df['prob_hgd'] = probs[:, 2]

            print(results_df)
            results_save_path = os.path.join(self.run_dir, 'results.csv')
            print('Saving results to: {}'.format(results_save_path))
            results_df.to_csv(results_save_path, index=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='baseline_CONCH', help="the name of this experiment")
    parser.add_argument("--project_name", type=str, default='WeakBE-Net_multiclass', help="the name of this project")
    parser.add_argument("--binary", type=bool, default=False, help="whether to run in binary setup")
    parser.add_argument("--nr_epochs", type=int, default=2500, help="the number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="the size of mini batches")
    parser.add_argument("--hidden_dim", type=int, default=512, help="hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-5, help="initial the learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay (L2)")
    parser.add_argument("--drop_out", type=float, default=0.0, help="drop out rate")
    parser.add_argument("--k_folds", type=int, default=5, help="number of folds")
    parser.add_argument("--exp_dir", type=str,
                        default='/home/mbotros/experiments/lans_weaklysupervised/')
    parser.add_argument("--features_dir", type=str,
                        default='/data/archief/AMC-data/Barrett/LANS_features/CONCH')
    parser.add_argument("--label_file", type=str,
                        default='/data/archief/AMC-data/Barrett/LANS/lans_consensus_no_ind_nbde=0_lgd=1_hgd=2.csv')
    parser.add_argument("--wandb_key", type=str, help="key for logging to weights and biases")
    parser.add_argument("--test", type=bool, help="whether to also test", default=True)
    args = parser.parse_args()

    train(args)
