## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import math

# import emlp.nn.pytorch as emlpnn
# from emlp.reps import T, V
# from emlp.groups import SO
import pdb
from locs.models.egnn import E_GCL, EGNN_vel
from locs.models.gvp import GVP
from torch_geometric.nn import GATConv
from torch_geometric.nn.models import GAT
from typing import Optional
from locs.models.my_egnn import EGNN_rot

from locs.utils.geometry import construct_3d_basis_from_2_vectors, multiply_matrices

import rff


def gram_schmidt(vel, acc):
    """Gram-Schmidt orthogonalization"""
    # normalize
    e1 = F.normalize(vel, dim=-1, eps=1e-8)
    # orthogonalize
    u2 = acc - torch.sum(e1 * acc, dim=-1, keepdim=True) * e1
    # normalize
    e2 = F.normalize(u2, dim=-1, eps=1e-8)
    # cross product
    e3 = torch.cross(e1, e2)

    frame1 = torch.stack([e1, e2, e3], dim=-1)
    return frame1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class FrameTransformer(nn.Module):
    def __init__(
        self,
        params,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_features - Number of features each node has (e.g. 4 for 2D coordinates [p_x,p_y,v_x,v_y]
                           and 6 for 3D coordinates [p_x,p_y,p_z,v_x,v_y,v_z]])
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.embed_dim = params.get("localizer_embedding_size", 128)
        self.hidden_dim = params.get("localizer_hidden_size", 256)
        self.num_features = params.get("localizer_n_in_dims", 6)
        self.n_out_dims = params.get("localizer_n_out_dims", 6)
        self.num_heads = params.get("localizer_n_heads", 4)
        self.num_layers = params.get("localizer_n_layers", 2)
        self.trajectory_size = params.get("window_size", 50)
        self.dropout = params.get("localizer_dropout", 0.0)

        # Layers/Networks
        self.input_layer = nn.Linear(2, self.embed_dim)
        self.transformer = nn.Sequential(
            *[
                AttentionBlock(
                    self.embed_dim,
                    self.hidden_dim,
                    self.num_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 2 * self.trajectory_size),
        )
        self.dropout = nn.Dropout(self.dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.positional_encoding = PositionalEncoding(
            d_model=self.embed_dim, max_len=1 + self.trajectory_size
        )

        self.gvp_linear = nn.Linear(2 * self.trajectory_size, 2, bias=False)

    def forward(self, x):
        # Preprocess input
        # The input layer should be in the form of [B, N, T, num_features], where B is the batch size,
        # N is the number of nodes, T is the number of time steps, and num_features is the number of
        # positional and velocity features. We need to reshape the input to be in the form of
        # [B*N, T, num_features] so that we can apply the Transformer to each node independently.
        # Reshape input from [B,N,T,num_features] to be [B*N, T, num_features]
        B, N, T, F = x.shape
        norm_x = x.unflatten(-1, (2, 3)).norm(dim=-1)
        norm_x = norm_x.reshape(B * N, T, 2)
        B2, _, _ = norm_x.shape
        norm_x = self.input_layer(norm_x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B2, 1, 1)
        norm_x = torch.cat([cls_token, norm_x], dim=1)
        norm_x = self.positional_encoding(norm_x)
        # Apply Transforrmer
        norm_x = self.dropout(norm_x)
        norm_x = norm_x.transpose(0, 1)
        norm_x = self.transformer(norm_x)

        # Perform classification prediction
        cls = norm_x[0]
        out = self.mlp_head(cls).reshape(B, N, 2 * self.trajectory_size).unsqueeze(-1)
        # 2*T
        x = x.unflatten(-1, (2, 3)).transpose(1, 2).flatten(2, 3)

        # B,N,2*T,3
        y = out * x
        y = self.gvp_linear(y.transpose(-1, -2)).transpose(-1, -2)
        # Now y is B,N,2T,3
        # The output is a 6D representation of two vectors, which by using Gram-Schmidt orthogonalization
        # we can construct an SO(3) rotation matrix and a translation vector.
        v1, v2 = y[..., 0, :], y[..., 1, :]

        R = construct_3d_basis_from_2_vectors(v1, v2)

        return R


class FrameTransformer2(nn.Module):
    def __init__(
        self,
        params,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_features - Number of features each node has (e.g. 4 for 2D coordinates [p_x,p_y,v_x,v_y]
                           and 6 for 3D coordinates [p_x,p_y,p_z,v_x,v_y,v_z]])
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.embed_dim = params.get("localizer_embedding_size", 128)
        self.hidden_dim = params.get("localizer_hidden_size", 256)
        self.num_features = params.get("localizer_n_in_dims", 6)
        self.n_out_dims = params.get("localizer_n_out_dims", 6)
        self.num_heads = params.get("localizer_n_heads", 4)
        self.num_layers = params.get("localizer_n_layers", 2)
        self.trajectory_size = params.get("window_size", 50)
        self.dropout = params.get("localizer_dropout", 0.0)
        self.num_objects = params.get("localizer_n_objects", 5)
        # Layers/Networks
        self.input_layer = nn.Linear(2 * self.trajectory_size, self.embed_dim)
        self.transformer = nn.Sequential(
            *[
                AttentionBlock(
                    self.embed_dim,
                    self.hidden_dim,
                    self.num_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 2 * self.trajectory_size),
        )
        self.dropout = nn.Dropout(self.dropout)

        # Parameters/Embeddings
        # self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_objects, self.embed_dim)
        )

        self.gvp_linear = nn.Linear(2 * self.trajectory_size, 2, bias=False)

    def forward(self, x):
        # Preprocess input
        # The input layer should be in the form of [B, N, T, num_features], where B is the batch size,
        # N is the number of nodes, T is the number of time steps, and num_features is the number of
        # positional and velocity features. We need to reshape the input to be in the form of
        # [B*N, T, num_features] so that we can apply the Transformer to each node independently.
        # Reshape input from [B,N,T,num_features] to be [B*N, T, num_features]
        B, N, T, F = x.shape
        norm_x = x.unflatten(-1, (2, 3)).norm(dim=-1)
        norm_x = norm_x.reshape(B, N, T * 2)
        norm_x = self.input_layer(norm_x)
        # norm_x = norm_x + self.pos_embedding
        # Apply Transforrmer
        norm_x = self.dropout(norm_x)
        norm_x = norm_x.transpose(0, 1)
        norm_x = self.transformer(norm_x)

        # Perform classification prediction
        # B,N,2T,3
        out = self.mlp_head(norm_x).transpose(0, 1).unsqueeze(-1)

        x = x.unflatten(-1, (2, 3)).transpose(1, 2).flatten(2, 3)

        # B,N,2*T,3
        y = out * x
        y = self.gvp_linear(y.transpose(-1, -2)).transpose(-1, -2)
        # Now y is B,N,2T,3
        # The output is a 6D representation of two vectors, which by using Gram-Schmidt orthogonalization
        # we can construct an SO(3) rotation matrix and a translation vector.
        v1, v2 = y[..., 0, :], y[..., 1, :]

        # R = construct_3d_basis_from_2_vectors(v1, v2)

        # The output is a 6D representation of two vectors, which by using Gram-Schmidt orthogonalization
        # we can construct an SO(3) rotation matrix and a translation vector.
        # v1, v2 = out[..., :3], out[..., 3:]
        R = construct_3d_basis_from_2_vectors(v1, v2)
        # print('R', R.shape)
        # Now the So(3) rotation matrix is in the form of [B*N, 3, 3], and we need to convert it to
        # [B, N, 3, 3] so that we can apply it to the input features.
        # R = R.reshape(-1, N, 3, 3)
        # print(R.shape)
        return R


class FrameMLP(nn.Module):
    def __init__(self, params):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
            num_features - Number of features each node has (e.g. 4 for 2D coordinates [p_x,p_y,v_x,v_y]
                           and 6 for 3D coordinates [p_x,p_y,p_z,v_x,v_y,v_z]])
            n_out_dims - Output of MLP is 6D representation of two vectors, which by using Gram-Schmidt orthogonalization
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.embed_dim = params.get("localizer_embedding_size", 128)
        self.hidden_dim = params.get("localizer_hidden_size", 256)
        self.num_features = params.get("localizer_n_in_dims", 6)
        self.trajectory_size = params.get("window_size", 50)
        self.dropout = params.get("localizer_dropout", 0.0)
        self.n_layers = params.get("localizer_n_layers", 2)
        self.n_out_dims = params.get("localizer_n_out_dims", 6)

        # Layers/Networks
        # Define an MLP to pass the transformed features through num_layer times and use GeLU as the activation function
        self.mlp = nn.Sequential()

        # Input layer
        self.mlp.add_module(
            "input_layer",
            rff.layers.GaussianEncoding(
                sigma=10.0,
                input_size=2 * self.trajectory_size,
                encoded_size=self.embed_dim,
            ),
            # nn.Linear(2 * self.trajectory_size, self.embed_dim),
        )
        self.mlp.add_module("input_layer_activation", nn.GELU())
        self.mlp.add_module("input_layer_dropout", nn.Dropout(self.dropout))
        # Embedding layer
        self.mlp.add_module(
            "hidden_layer_0",
            nn.Linear(self.embed_dim * 2, self.hidden_dim),
        )
        self.mlp.add_module("hidden_layer_0_activation", nn.GELU())
        self.mlp.add_module("hidden_layer_0_dropout", nn.Dropout(self.dropout))
        # Hidden layers
        for i in range(1, self.n_layers - 1):
            self.mlp.add_module(
                "hidden_layer_{}".format(i), nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            self.mlp.add_module("hidden_layer_{}_activation".format(i), nn.GELU())
            self.mlp.add_module(
                "hidden_layer_{}_dropout".format(i), nn.Dropout(self.dropout)
            )
        # Output layer
        self.mlp.add_module(
            "output_layer", nn.Linear(self.hidden_dim, 2 * self.trajectory_size)
        )

        self.gvp_linear = nn.Linear(2 * self.trajectory_size, 2, bias=False)

    def forward(self, x):
        # Preprocess input
        # The input layer should be in the form of [B, N, T, num_features], where B is the batch size,
        # N is the number of nodes, T is the number of time steps, and num_features is the number of
        # positional and velocity features. We need to reshape the input to be in the form of
        # [B, N, T * num_features].
        # Reshape input from [B,N,T,num_features] to be [B,N, T * num_features]

        B, N, T, F = x.shape
        norm_x = x.unflatten(-1, (2, 3)).norm(dim=-1)
        norm_x = norm_x.reshape(B, N, T * 2)

        out = self.mlp(norm_x).unsqueeze(-1)

        # 2*T
        x = x.unflatten(-1, (2, 3)).transpose(1, 2).flatten(2, 3)

        # B,N,2*T,3
        y = out * x
        y = self.gvp_linear(y.transpose(-1, -2)).transpose(-1, -2)
        # Now y is B,N,2T,3
        # The output is a 6D representation of two vectors, which by using Gram-Schmidt orthogonalization
        # we can construct an SO(3) rotation matrix and a translation vector.
        v1, v2 = y[..., 0, :], y[..., 1, :]

        R = construct_3d_basis_from_2_vectors(v1, v2)

        return R


class FrameResMLP(nn.Module):
    def __init__(self, params):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
            num_features - Number of features each node has (e.g. 4 for 2D coordinates [p_x,p_y,v_x,v_y]
                           and 6 for 3D coordinates [p_x,p_y,p_z,v_x,v_y,v_z]])
            n_out_dims - Output of MLP is 6D representation of two vectors, which by using Gram-Schmidt orthogonalization
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()
        self.embed_dim = params.get("localizer_embedding_size", 128)
        self.hidden_dim = params.get("localizer_hidden_size", 256)
        self.num_features = params.get("localizer_n_in_dims", 6)
        self.trajectory_size = params.get("window_size", 50)
        self.dropout = params.get("localizer_dropout", 0.0)
        self.n_out_dims = 2 * self.trajectory_size

        # Layers/Networks
        # Define an MLP to pass the transformed features through num_layer times and use GeLU as the activation function
        self.mlp = nn.Sequential(
            rff.layers.GaussianEncoding(
                sigma=10.0,
                input_size=2 * self.trajectory_size,
                encoded_size=self.embed_dim // 2,
            ),
            # nn.Linear(2 * self.trajectory_size, self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_out_dims),
        )
        self.gvp_linear = nn.Linear(self.n_out_dims, 2)

    def forward(self, x):
        # Preprocess input
        # The input layer should be in the form of [B, N, T, num_features], where B is the batch size,
        # N is the number of nodes, T is the number of time steps, and num_features is the number of
        # positional and velocity features. We need to reshape the input to be in the form of
        # [B, N, T * num_features].
        # Reshape input from [B,N,T,num_features] to be [B,N, T * num_features]
        B, N, T, F = x.shape

        norm_x = x.unflatten(-1, (2, 3)).norm(dim=-1)

        norm_x = norm_x.reshape(B, N, T * 2)

        out = self.mlp(norm_x).unsqueeze(-1)
        # 2*T
        x = x.unflatten(-1, (2, 3)).transpose(1, 2).flatten(2, 3)
        # B,N,2*T,3
        y = out * x
        y = self.gvp_linear(y.transpose(-1, -2)).transpose(-1, -2)
        # Now y is B,N,2,3

        # The output is a 6D representation of two vectors, which by using Gram-Schmidt orthogonalization
        # we can construct an SO(3) rotation matrix and a translation vector.
        v1, v2 = y[..., 0, :], y[..., 1, :]

        R = construct_3d_basis_from_2_vectors(v1, v2)
        # # out_R
        # out_R = multiply_matrices(R, init_R)
        return R


class FrameGVP(nn.Module):
    def __init__(self, params):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
            num_features - Number of features each node has (e.g. 4 for 2D coordinates [p_x,p_y,v_x,v_y]
                           and 6 for 3D coordinates [p_x,p_y,p_z,v_x,v_y,v_z]])
            n_out_dims - Output of MLP is 6D representation of two vectors, which by using Gram-Schmidt orthogonalization
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()
        self.embed_dim = params.get("localizer_embedding_size", 128)
        self.hidden_dim = params.get("localizer_hidden_size", 256)
        self.num_features = params.get("localizer_n_in_dims", 6)
        self.trajectory_size = params.get("window_size", 50)
        self.dropout = params.get("localizer_dropout", 0.0)
        self.n_out_dims = 2 * self.trajectory_size

        self.GVP = GVP(
            in_dims=(2 * self.trajectory_size, 2 * self.trajectory_size),
            out_dims=(2, 2),
            h_dim=None,
            activations=(F.relu, torch.sigmoid),
            vector_gate=params.get("localizer_vector_gate", False),
        )

    def forward(self, x):
        B, N, T, F = x.shape
        # B,N,2*T,1
        norm_x = x.unflatten(-1, (2, 3)).norm(dim=-1)
        norm_x = norm_x.reshape(B, N, T * 2)
        # B,N,2*T,3
        x = x.unflatten(-1, (2, 3)).transpose(1, 2).flatten(2, 3)
        # Pass these to the GVP
        s, v = self.GVP((norm_x, x))
        # Construct the rotation matrix
        R = construct_3d_basis_from_2_vectors(v[..., 0, :], v[..., 1, :])
        return R


# Define the cross-attention module
class CrossAttention(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_query = torch.nn.Linear(input_size, output_size)
        self.linear_key = torch.nn.Linear(input_size, output_size)
        self.linear_value = torch.nn.Linear(input_size, output_size)

    def forward(self, spatial_embed, temporal_embed):
        # Transform the embeddings
        query = self.linear_query(spatial_embed)
        key = self.linear_key(temporal_embed)
        value = self.linear_value(temporal_embed)

        # Compute attention weights
        scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        return output


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        params,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_features - Number of features each node has (e.g. 4 for 2D coordinates [p_x,p_y,v_x,v_y]
                           and 6 for 3D coordinates [p_x,p_y,p_z,v_x,v_y,v_z]])
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.embed_dim = params.get("localizer_embedding_size", 128)
        self.hidden_dim = params.get("localizer_hidden_size", 256)
        self.num_features = params.get("localizer_n_in_dims", 6)
        self.n_out_dims = params.get("localizer_n_out_dims", 6)
        self.num_heads = params.get("localizer_n_heads", 4)
        self.num_layers = params.get("localizer_n_layers", 2)
        self.trajectory_size = params.get("window_size", 50)
        self.dropout_prob = params.get("localizer_dropout", 0.0)

        # Layers/Networks
        # self.input_layer = nn.Linear(2, self.embed_dim)

        self.input_layer = rff.layers.GaussianEncoding(
            sigma=10.0, input_size=2, encoded_size=self.embed_dim // 2
        )
        self.transformer = nn.Sequential(
            *[
                AttentionBlock(
                    self.embed_dim,
                    self.hidden_dim,
                    self.num_heads,
                    dropout=self.dropout_prob,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.dropout = nn.Dropout(self.dropout_prob)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def forward(self, x):
        B, N, T, F = x.shape
        norm_x = x.unflatten(-1, (2, 3)).norm(dim=-1)
        norm_x = norm_x.reshape(B * N, T, 2)
        B2, _, _ = norm_x.shape
        norm_x = self.input_layer(norm_x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B2, 1, 1)
        norm_x = torch.cat([cls_token, norm_x], dim=1)
        # Apply Transforrmer
        norm_x = self.dropout(norm_x)
        norm_x = norm_x.transpose(0, 1)
        norm_x = self.transformer(norm_x)

        cls = norm_x[0]
        out = self.mlp_head(cls).reshape(B, N, self.embed_dim)
        return out


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        params,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_features - Number of features each node has (e.g. 4 for 2D coordinates [p_x,p_y,v_x,v_y]
                           and 6 for 3D coordinates [p_x,p_y,p_z,v_x,v_y,v_z]])
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.embed_dim = params.get("localizer_embedding_size", 128)
        self.hidden_dim = params.get("localizer_hidden_size", 256)
        self.num_features = params.get("localizer_n_in_dims", 6)
        self.n_out_dims = params.get("localizer_n_out_dims", 6)
        self.num_heads = params.get("localizer_n_heads", 4)
        self.num_layers = params.get("localizer_n_layers", 2)
        self.trajectory_size = params.get("window_size", 50)
        self.dropout_prob = params.get("localizer_dropout", 0.0)
        self.num_objects = params.get("localizer_n_objects", 5)
        # Layers/Networks
        self.input_layer = nn.Linear(2 * self.trajectory_size, self.embed_dim)
        self.transformer = nn.Sequential(
            *[
                AttentionBlock(
                    self.embed_dim,
                    self.hidden_dim,
                    self.num_heads,
                    dropout=self.dropout_prob,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        # Preprocess input
        B, N, T, F = x.shape
        # Charge is also a feature
        norm_x = x.unflatten(-1, (2, 3)).norm(dim=-1)
        norm_x = norm_x.reshape(B, N, T * 2)
        norm_x = self.input_layer(norm_x)
        # norm_x = norm_x + self.pos_embedding
        # Apply Transforrmer
        norm_x = self.dropout(norm_x)
        norm_x = norm_x.transpose(0, 1)
        norm_x = self.transformer(norm_x)

        # Perform classification prediction
        # B,N,E
        out = self.mlp_head(norm_x).transpose(0, 1)
        return out


class SpatialGNNEncoder(nn.Module):
    def __init__(
        self,
        params,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_features - Number of features each node has (e.g. 4 for 2D coordinates [p_x,p_y,v_x,v_y]
                           and 6 for 3D coordinates [p_x,p_y,p_z,v_x,v_y,v_z]])
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.embed_dim = params.get("localizer_embedding_size", 128)
        self.hidden_dim = params.get("localizer_hidden_size", 256)
        self.num_features = params.get("localizer_n_in_dims", 6)
        self.num_heads = params.get("localizer_n_heads", 4)
        self.num_layers = params.get("localizer_n_layers", 2)
        self.trajectory_size = params.get("window_size", 50)
        self.dropout_prob = params.get("localizer_dropout", 0.0)
        self.num_objects = params.get("localizer_n_objects", 5)
        self.n_out_dims = 2 * self.trajectory_size

        # Layers/Networks

        self.GNN = GAT(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            hidden_channels=self.hidden_dim,
            dropout=self.dropout_prob,
            num_layers=self.num_layers,
        )

        self.output_layer = nn.Linear(self.embed_dim, self.n_out_dims)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x, edges, edge_attr):
        x = x.view(-1, self.embed_dim)
        edge_index = torch.empty(2, edges[0].shape[0], dtype=torch.long)
        edge_index[0] = edges[0].to(torch.long)
        edge_index[1] = edges[1].to(torch.long)
        # Input is the size of [B, N, H] where B is the batch size, N is the number of nodes, and H is the number of features
        x = self.GNN(x, edge_index.to(x.device))
        x = self.dropout(x)
        x = self.output_layer(x).view(-1, self.num_objects, self.n_out_dims)
        return x


class SpatioTemporalEGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        coords_weight=1.0,
        recurrent=False,
        norm_diff=False,
        tanh=False,
    ):
        super().__init__()
        self.EGNN = EGNN_rot(
            in_node_nf,
            in_edge_nf,
            hidden_nf,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            coords_weight=coords_weight,
            recurrent=recurrent,
            norm_diff=norm_diff,
            tanh=tanh,
        )

        self.device = device

    def get_edges(self, batch_size, n_nodes):
        send_edges, recv_edges = torch.where(~torch.eye(n_nodes, dtype=bool))
        edges = [torch.LongTensor(send_edges), torch.LongTensor(recv_edges)]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows).to(self.device), torch.cat(cols).to(self.device)]
        return edges

    def forward(self, h, edges, rot_rep, edge_attr):
        B, N, NF = h.shape
        h = h.reshape(-1, NF)
        B, E, EF = edge_attr.shape
        edge_attr = edge_attr.reshape(-1, EF)
        edges = self.get_edges(batch_size=B, n_nodes=N)
        x, h = self.EGNN(h, rot_rep, edges, edge_attr)

        return h.reshape(B, N, -1), x.reshape(B, N, 3, 2)


class SpatioTemporalFrame(nn.Module):
    """
    Module that takes as input the node features and outputs a rotation matrix.
    The procedure goes as follows:
        1. Create temporal embedding for each node of the batch using a transformer encoder (rotation invariant)
        2. Create spatial embedding for each node of the batch using a transformer encoder (rotation invariant)
        3. Using the spatiotemporal embedding, perform cross attention between the spatial and temporal embedding
           to produce a two scalar values for each node of the batch [B,N,2] where H is the hidden dimension,
           to multiply with the vector features of each node and create a [B,N,2,3] result.
        4. Using a GVP layer (linear layer without bias) of the form [B,N,2*T,3] -> [B,N,2,3] we create
           two 3D vectors which using Gram-Scmidt orthogonalization we can construct an SO(3) rotation matrix.
    Inputs:
            embed_dim - Dimensionality of the input feature vectors
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
            num_features - Number of features each node has (e.g. 4 for 2D coordinates [p_x,p_y,v_x,v_y]
                           and 6 for 3D coordinates [p_x,p_y,p_z,v_x,v_y,v_z]])
            n_out_dims - Output of MLP is 6D representation of two vectors, which by using Gram-Schmidt orthogonalization
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
    """

    def __init__(self, params):
        super().__init__()
        self.embed_dim = params.get("localizer_embedding_size", 128)
        self.hidden_dim = params.get("localizer_hidden_size", 256)
        self.num_features = params.get("localizer_n_in_dims", 6)
        self.trajectory_size = params.get("window_size", 50)
        self.dropout = params.get("localizer_dropout", 0.0)
        self.n_out_dims = 2 * self.trajectory_size

        self.temporal_encoder = TemporalEncoder(
            params=params,
        )

        self.spatial_encoder = SpatioTemporalEGNN(
            in_node_nf=params.get("localizer_embedding_size", 128),
            in_edge_nf=2,
            hidden_nf=params.get("localizer_hidden_size", 128),
            device="cpu",
            act_fn=nn.SiLU(),
            n_layers=3,
            coords_weight=1.0,
            recurrent=False,
            norm_diff=False,
            tanh=False,
        )

        # self.gvp_linear = nn.Linear(2 * self.trajectory_size, 2, bias=False)

    def forward(self, x, edges, edge_attr, rot_vec):
        # Rotation representation is the velocity vector of last time step and the difference between the last two time steps
        rot_vec = rot_vec.reshape(-1, 3, 2)

        # Temporal embedding
        temporal_embed = self.temporal_encoder(x)
        # Spatial embedding
        spatiotemporal_embed, v = self.spatial_encoder(
            temporal_embed, edges, rot_vec, edge_attr
        )

        # Construct the rotation matrix
        out_R = construct_3d_basis_from_2_vectors(v[..., 0], v[..., 1])
        return out_R, spatiotemporal_embed
