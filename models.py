import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers.helpers import to_2tuple


def get_activation_function(activation_name: str):
    """Return the corresponding activation function from a string name.

    Args:
    - activation_name: Name of the activation function

    Returns:
    - activation_function: Activation function
    """
    if activation_name is None:
        return None
    try:
        return getattr(nn, activation_name)()
    except AttributeError:
        raise ValueError(f"Unsupported activation function: {activation_name}")


def build_encoder(n_feats: int, z_dim: int, encoder_layers: int, activation_function: str, dropout_p: float = 0.1,
                  use_batchnorm: bool = True):
    """Builds an encoder with a specified number of layers and activation functions.

    Args:
    - n_feats: Number of input features
    - z_dim: Dimension of the latent space
    - encoder_layers: Number of layers in the encoder
    - activation_function: Activation function to use in the encoder
    - dropout_p: Dropout probability
    - use_batchnorm: Whether to use batch normalization in the encoder

    Returns:
    - encoder: Encoder network

    Linear > BN > Activation > Dropout
    """
    layers = []
    for i in range(encoder_layers):
        in_features = n_feats if i == 0 else z_dim
        layers.append(nn.Linear(in_features, z_dim))
        if activation_function is not None:
            layers.append(get_activation_function(activation_function))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(z_dim))
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
    return nn.Sequential(*layers)


def initialize_weights(module):
    """
    Initialize the weights of the model using Xavier initialization for linear layers and constant initialization
    for batch normalization layers.

    Args:
    - module: The model to initialize

    Returns:
    - None
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class clam_mil(nn.Module):
    """
    Clusering-constrained Attention MIL (CLAM-MIL) model. The model uses a learnable attention mechanism to
    aggregate instance embeddings into slide-level representations. The model also includes a clustering module
    to predict instance-level cluster assignments. For each output class the model has a separate attention
    mechanism and classifier.

    Args:
    - n_feats: Number of input features
    - n_out: Number of output classes
    - z_dim: Dimension of the latent space
    - dropout_p: Dropout probability
    - activation_function: Activation function to use in the encoder
    - encoder_layers: Number of layers in the encoder

    Methods:
    - forward: Forward pass through the model
    - cluster_patches: Cluster patches into clusters
    - calculate_attention: Calculate the attention weights for each instance
    - initialize_weights: Initialize the weights of the model

    Returns:
    - scores: Predicted class scores for in the bag
    - attention_weights: Attention weights for each instance (optional)
    """

    def __init__(self, n_feats: int, n_out: int, z_dim: int = 256, dropout_p: float = 0.1, activation_function='ReLU',
                 encoder_layers=1):
        super(clam_mil, self).__init__()
        self.n_out = n_out
        self.encoder = build_encoder(n_feats, z_dim, encoder_layers, activation_function, dropout_p, True)
        self.attention_U = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Sigmoid()
        )
        self.attention_V = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh()
        )
        self.attention_branches = nn.ModuleList([nn.Linear(z_dim, 1) for _ in range(n_out)])
        self.classifiers = nn.ModuleList([nn.Linear(z_dim, 1) for _ in range(n_out)])
        self.instance_cluster = nn.ModuleList([nn.Linear(z_dim, 2) for _ in range(n_out)])
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, bags, return_attention=False):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        slide_level_representations = []
        attention_weights_list = []

        for i in range(self.n_out):
            attention_U_output = self.attention_U(embeddings)
            attention_V_output = self.attention_V(embeddings)
            attention_scores = self.attention_branches[i](attention_U_output * attention_V_output).softmax(dim=1)
            attention_weights_list.append(attention_scores)
            slide_level_representation = torch.sum(attention_scores * embeddings, dim=1)
            slide_level_representations.append(slide_level_representation)

        slide_level_representations = torch.stack(slide_level_representations, dim=1)
        scores = []
        for i in range(self.n_out):
            score = self.classifiers[i](slide_level_representations[:, i, :])
            scores.append(score)
        scores = torch.cat(scores, dim=1)

        # Stack the attention weights and then take the mean along the first dimension (across branches)
        attention_weights = torch.stack(attention_weights_list, dim=0)  # Shape: (n_out, batch_size, n_patches, 1)
        attention_weights = torch.mean(attention_weights, dim=0)  # Shape: (batch_size, n_patches, 1)

        if return_attention:
            return scores, attention_weights
        else:
            return scores

    def cluster_patches(self, bags):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        cluster_predictions = []
        for i in range(self.n_out):
            cluster_pred = self.instance_cluster[i](embeddings)
            cluster_predictions.append(cluster_pred)
        return cluster_predictions

    def calculate_attention(self, bags, lens=None, apply_softmax=False):
        batch_size, n_patches, n_feats = bags.shape
        embeddings = self.encoder(bags.view(-1, n_feats))
        embeddings = embeddings.view(batch_size, n_patches, -1)
        attention_weights_list = []

        for i in range(self.n_out):
            attention_U_output = self.attention_U(embeddings)
            attention_V_output = self.attention_V(embeddings)
            attention_scores = self.attention_branches[i](attention_U_output * attention_V_output).softmax(dim=1)
            attention_weights_list.append(attention_scores)

        attention_weights = torch.mean(torch.stack(attention_weights_list), dim=0)
        if apply_softmax:
            attention_weights = F.softmax(attention_weights, dim=1)
        return attention_weights


class AttentionMIL(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, drop_out=0.2):
        super(AttentionMIL, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(feature_dim, output_dim)

    def forward(self, bag, mask):
        """ Classic AttentionMIL, using masking to disregard padded instances.
        Args:
            bag: (batch_size, bag_size, feature_dim)
            mask: (batch_size, bag_size)
        Returns:
            logits: (batch_size)
            attn_weights: (batch_size, bag_size)
        """
        attn_weights = self.attention(bag).squeeze()
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))  # mask out padded instances
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted_features = torch.sum(bag * attn_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(weighted_features).squeeze()
        return logits, attn_weights


class ConvStem(nn.Module):
    """Custom Patch Embed Layer.

  Adapted from https://github.com/Xiyue-Wang/TransPath/blob/main/ctran.py#L6-L44
  """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, **kwargs):
        super().__init__()

        # Check input constraints
        assert patch_size == 4, "Patch size must be 4"
        assert embed_dim % 8 == 0, "Embedding dimension must be a multiple of 8"

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Create stem network
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        # Apply normalization layer (if provided)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # Check input image size
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.norm(x)
        return x
