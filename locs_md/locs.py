import pdb
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from locs.models.model_factory import LocalizerFactory

from pytorch3d.transforms import so3_exponential_map, so3_log_map

from locs.training import train_utils
import pytorch_lightning as pl

from locs.utils.geometry import construct_3d_basis_from_2_vectors, multiply_matrices

# Note: This is the model that is used in the paper
"""
General Notes:
dicts are used to store the parameters for each of the modules
params.get('key', default) is used to get the value of a key in the dict
if the key is not present, the default value is returned

"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


"""
Geometric primitives, rotations, cartesian to spherical
"""


def rotation_matrix(ndim, theta, phi=None, psi=None, /):
    """
    theta, phi, psi: yaw, pitch, roll

    NOTE: We assume that each angle is has the shape [dims] x 1
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    if ndim == 2:
        R = torch.stack(
            [
                torch.cat([cos_theta, -sin_theta], -1),
                torch.cat([sin_theta, cos_theta], -1),
            ],
            -2,
        )
        return R
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    R = torch.stack(
        [
            torch.cat([cos_phi * cos_theta, -sin_theta, sin_phi * cos_theta], -1),
            torch.cat([cos_phi * sin_theta, cos_theta, sin_phi * sin_theta], -1),
            torch.cat([-sin_phi, torch.zeros_like(cos_theta), cos_phi], -1),
        ],
        -2,
    )
    return R


def cart_to_n_spherical(x, symmetric_theta=False):
    """Transform Cartesian to n-Spherical Coordinates

    NOTE: Not tested thoroughly for n > 3

    Math convention, theta: azimuth angle, angle in x-y plane

    x: torch.Tensor, [dims] x D
    return rho, theta, phi
    """
    ndim = x.size(-1)

    rho = torch.norm(x, p=2, dim=-1, keepdim=True)

    theta = torch.atan2(x[..., [1]], x[..., [0]])
    if not symmetric_theta:
        theta = theta + (theta < 0).type_as(theta) * (2 * np.pi)

    if ndim == 2:
        return rho, theta

    cum_sqr = (
        rho
        if ndim == 3
        else torch.sqrt(torch.cumsum(torch.flip(x**2, [-1]), dim=-1))[..., 2:]
    )
    EPS = 1e-7
    phi = torch.acos(torch.clamp(x[..., 2:] / (cum_sqr + EPS), min=-1.0, max=1.0))

    return rho, theta, phi


def velocity_to_rotation_matrix(vel):
    num_dims = vel.size(-1)
    orientations = cart_to_n_spherical(vel)[1:]
    R = rotation_matrix(num_dims, *orientations)
    return R


def gram_schmidt(vel, acc):
    """Gram-Schmidt orthogonalization"""
    # normalize
    e1 = F.normalize(vel, dim=-1)
    # orthogonalize
    u2 = acc - torch.sum(e1 * acc, dim=-1, keepdim=True) * e1
    # normalize
    e2 = F.normalize(u2, dim=-1)
    # cross product
    e3 = torch.cross(e1, e2)

    frame1 = torch.stack([e1, e2, e3], dim=-1)
    return frame1


def rotation_matrices_to_quaternions(rotations: torch.Tensor) -> torch.Tensor:
    # Ensure input tensor has the correct shape
    assert rotations.dim() == 4 and rotations.shape[-2:] == (
        3,
        3,
    ), f"Expected tensor of shape [B,N,3,3], got {rotations.shape}"

    # Extract the rotation matrix components
    r11, r12, r13 = rotations[..., 0, 0], rotations[..., 0, 1], rotations[..., 0, 2]
    r21, r22, r23 = rotations[..., 1, 0], rotations[..., 1, 1], rotations[..., 1, 2]
    r31, r32, r33 = rotations[..., 2, 0], rotations[..., 2, 1], rotations[..., 2, 2]

    # Compute the quaternion components
    qw = torch.sqrt(torch.clamp(1.0 + r11 + r22 + r33, min=1e-8)) / 2.0
    qx = torch.sqrt(torch.clamp(1.0 + r11 - r22 - r33, min=1e-8)) / 2.0
    qy = torch.sqrt(torch.clamp(1.0 - r11 + r22 - r33, min=1e-8)) / 2.0
    qz = torch.sqrt(torch.clamp(1.0 - r11 - r22 + r33, min=1e-8)) / 2.0

    # Determine the signs of the quaternion components
    qx = torch.where(r32 - r23 < 0, -qx, qx)
    qy = torch.where(r13 - r31 < 0, -qy, qy)
    qz = torch.where(r21 - r12 < 0, -qz, qz)

    # Combine the quaternion components into a tensor of shape [B,N,4]
    quaternions = torch.stack((qw, qx, qy, qz), dim=-1)

    return quaternions


def rotation_matrix_to_euler(R, num_dims, normalize=True):
    """Convert rotation matrix to euler angles

    In 3 dimensions, we follow the ZYX convention
    NOTE: Use torch.clamp to avoid numerical errors everything has to be in [-1, 1]

    """
    if num_dims == 2:
        euler = torch.atan2(R[..., 1, [0]], R[..., 0, [0]])
    else:
        euler = torch.stack(
            [
                torch.atan2(R[..., 1, 0], R[..., 0, 0]),
                torch.asin(torch.clamp(-R[..., 2, 0], min=-1, max=1)),
                torch.atan2(R[..., 2, 1], R[..., 2, 2]),
            ],
            -1,
        )

    if normalize:
        euler = euler / np.pi
    return euler


def rotate(x, R):
    return torch.einsum("...ij,...j->...i", R, x)


"""
Transformation from global to local coordinates and vice versa
"""


class Localizer(nn.Module):
    def __init__(
        self,
        params,
        num_objects: int = 5,
        num_dims: int = 2,
    ):
        super().__init__()
        self.num_dims = num_dims
        self.num_objects = num_objects
        self.params = params
        self.send_edges, self.recv_edges = torch.where(
            ~torch.eye(self.num_objects, dtype=bool)
        )
        self.window_size = params.get("window_size", 10)

        self.type = params.get("localizer_type", "baseline")
        if self.type != "baseline" and self.type != "velacc":
            self.frame_module = LocalizerFactory.create(self.type, params)

    def set_edge_index(self, send_edges, recv_edges):
        self.send_edges = send_edges
        self.recv_edges = recv_edges

    def sender_receiver_features(self, x):
        batch_range = torch.arange(x.size(0), device=x.device).unsqueeze(-1)

        x_j = x[batch_range, self.send_edges]
        x_i = x[batch_range, self.recv_edges]
        return x_j, x_i

    def charge_to_index(self, charges):
        return ((charges + 1) / 2).long()

    def flatten_and_canonicalize(self, sequence, R):
        vel = sequence[..., self.num_dims : 2 * self.num_dims]
        B, N, T, F = vel.shape
        canon_vel = rotate(vel, R)
        canon_pos = rotate(sequence[..., : self.num_dims], R)
        canon_input = torch.zeros(B, N, T, 2 * F, device=sequence.device)
        canon_input[..., : self.num_dims] = canon_pos
        canon_input[..., self.num_dims : 2 * self.num_dims] = canon_vel
        # Flatten the sequence into B,N,T*(2F)
        return canon_input.reshape(B, N, -1)

    def canonicalize_inputs(self, inputs, charges, sequence):
        # acc = inputs[..., 2*self.num_dims:]
        # R = gram_schmidt(vel, acc)
        # Compute the relative positions of the objects wrt to the position of
        # each object in the sequence at the current time step (last)

        rel_pos = torch.cat(
            [
                sequence[..., : self.num_dims] - sequence[:, [-1], :, : self.num_dims],
                sequence[..., self.num_dims : 2 * self.num_dims],
            ],
            dim=-1,
        )
        node_embeddings = None
        if self.type == "baseline":
            vel = inputs[..., self.num_dims : 2 * self.num_dims]
            R = velocity_to_rotation_matrix(vel)
        elif self.type == "velacc":
            vel = sequence[:, -1, :, self.num_dims : 2 * self.num_dims]
            acc = (
                sequence[:, -1, :, self.num_dims : 2 * self.num_dims]
                - sequence[:, -2, :, self.num_dims : 2 * self.num_dims]
            )
            R = construct_3d_basis_from_2_vectors(vel, acc)
        elif self.type == "resmlp":
            vel = sequence[:, -1, :, self.num_dims : 2 * self.num_dims]
            acc = (
                sequence[:, -1, :, self.num_dims : 2 * self.num_dims]
                - sequence[:, -2, :, self.num_dims : 2 * self.num_dims]
            )
            init_R = construct_3d_basis_from_2_vectors(vel, acc).detach()
            mlp_R = self.frame_module(rel_pos)
            alpha = 0.5
            # R = exp(a * log(R_mlp) + (1-a) * log(R_init)
            R = alpha * so3_log_map(mlp_R) + (1 - alpha) * so3_log_map(init_R)
            R = so3_exponential_map(R)
        elif self.type == "spatio_temporal":
            edges = (self.send_edges, self.recv_edges)
            current_positions = sequence.permute(0, 2, 1, 3)[..., -1, :].detach()
            x_j_t, x_i_t = self.sender_receiver_features(current_positions)
            node_distances = torch.norm(x_j_t - x_i_t, p=2, dim=-1, keepdim=True)
            if charges is not None:
                c_j, c_i = self.sender_receiver_features(charges)
                rel_charges = c_j * c_i
                edge_attr = torch.cat([rel_charges, node_distances], dim=-1)
            else:
                edge_attr = node_distances
            vel_acc = torch.cat(
                [
                    sequence[:, -1, :, self.num_dims : 2 * self.num_dims].unsqueeze(-1),
                    (
                        sequence[:, -1, :, self.num_dims : 2 * self.num_dims]
                        - sequence[:, -2, :, self.num_dims : 2 * self.num_dims]
                    ).unsqueeze(-1),
                ],
                dim=-1,
            )

            R, _ = self.frame_module(rel_pos, edges, edge_attr, vel_acc)
        else:
            R = self.frame_module(rel_pos)

        Rinv = R.transpose(-1, -2)
        canon_in = self.flatten_and_canonicalize(
            rel_pos.permute(0, 2, 1, 3), Rinv.unsqueeze(2)
        )
        if charges is not None:
            canon_inputs = torch.cat([canon_in, charges], dim=-1)
        else:
            canon_inputs = canon_in

        return canon_inputs, R

    def create_edge_attr(self, x, charges, R):
        x = x.permute(0, 2, 1, 3)
        # x_j_t, x_i_t are the features of the sender and receiver at the current
        # timestep which will be used to compute the edge features for each frame
        x_j_t, x_i_t = self.sender_receiver_features(x[..., -1, :])
        x_j, x_i = self.sender_receiver_features(x)
        send_R, recv_R = self.sender_receiver_features(R)

        B, N, T, F = x_j.shape

        # R = gram_schmidt(x_i[..., self.num_dims:2*self.num_dims],
        # x_i[..., 2*self.num_dims:])
        # We approximate orientations via the velocity vector
        # R = velocity_to_rotation_matrix(x_i[..., self.num_dims:2*self.num_dims])
        R_inv = recv_R.transpose(-1, -2)
        # Positions
        relative_positions = x_j[..., : self.num_dims] - x_i_t[
            ..., : self.num_dims
        ].unsqueeze(2)
        # relative_positions = x_j[..., : self.num_dims] - x_i[..., : self.num_dims]
        multiple_R = R_inv.unsqueeze(2).detach()

        rotated_relative_positions = rotate(relative_positions, multiple_R)

        # Orientations
        # send_R = gram_schmidt(x_j[..., self.num_dims:2*self.num_dims],
        # x_j[..., 2*self.num_dims:])
        # send_R = velocity_to_rotation_matrix(x_j[..., self.num_dims:2*self.num_dims])

        rotated_orientations = R_inv @ send_R
        # rotated_euler = rotation_matrix_to_euler(rotated_orientations, self.num_dims)
        rotated_quaternions = rotation_matrices_to_quaternions(rotated_orientations)

        # Rotated relative positions in spherical coordinates
        node_distance = torch.norm(relative_positions, p=2, dim=-1, keepdim=True)
        spherical_relative_positions = torch.cat(
            cart_to_n_spherical(rotated_relative_positions, symmetric_theta=True)[1:],
            -1,
        )
        # Velocities
        rotated_velocities = rotate(
            x_j[..., self.num_dims : 2 * self.num_dims], multiple_R
        )

        edge_attr = torch.cat(
            [
                rotated_relative_positions.reshape(B, N, -1),  # [B, (N*N-1), T*3]
                rotated_quaternions,  # [B, N*(N-1), 4]
                node_distance.reshape(B, N, -1),  # [B, N*(N-1), T*1]
                spherical_relative_positions.reshape(B, N, -1),  # [B, N*(N-1), T*2]
                rotated_velocities.reshape(B, N, -1),  # [B, N*(N-1), T*3]
            ],
            -1,
        )

        if charges is not None:
            c_j, c_i = self.sender_receiver_features(charges)
            # Charges
            relative_charges = c_j * c_i
            edge_attr = torch.cat([edge_attr, relative_charges], dim=-1)
        return edge_attr  # [B, N*(N-1), T*9 + 5]

    def forward(self, x, charges, sequence):
        # self.set_edge_index(*edges)
        # rel_feat is [B, N, 2*T*3+ 1 or 0]
        rel_feat, R = self.canonicalize_inputs(x, charges, sequence)

        # edge_attr is [B, N*(N-1), T*9 + 5 (or 4)]
        edge_attr = self.create_edge_attr(sequence, charges, R)

        batch_range = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        edge_attr = torch.cat([edge_attr, rel_feat[batch_range, self.recv_edges]], -1)
        return rel_feat, R, edge_attr


class Globalizer(nn.Module):
    def __init__(self, num_dims: int = 2):
        super().__init__()
        self.num_dims = num_dims

    def forward(self, x, R):
        return torch.cat([rotate(xi, R) for xi in x.split(self.num_dims, dim=-1)], -1)


"""
LoCS Neural Network
"""


class LoCS(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout_prob,
        num_dims,
        params,
        device="cpu",
        num_layers=4,
    ):
        super().__init__()
        self.params = params
        self.window_size = params.get("window_size", 1)
        self.gnn = GNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout_prob=dropout_prob,
            num_dims=num_dims,
            window_size=self.window_size,
            num_layers=num_layers,
            params=params,
        )
        self.num_dims = num_dims
        self.num_objects = params["num_vars"]
        self.localizer = Localizer(
            self.params,
            self.num_objects,
            self.num_dims,
        )
        self.globalizer = Globalizer(num_dims)
        self.to(device)
        self.send_edges, self.recv_edges = torch.where(
            ~torch.eye(self.num_objects, dtype=bool)
        )

    def forward(self, x, vel, charges, sequence):
        """inputs shape: [batch_size, num_objects, input_size]"""
        x = x[..., -1, :]
        vel = vel[..., -1, :]

        inputs = torch.cat([x, vel], dim=-1)
        edges = self.send_edges, self.recv_edges

        # Global to Local
        rel_feat, Rinv, edge_attr = self.localizer(inputs, charges, sequence)
        # GNN
        pred = self.gnn(rel_feat, edge_attr, edges)
        # Local to Global
        pred = self.globalizer(pred, Rinv)

        # Predict position/velocity difference and integrate
        # outputs = torch.cat([x + pred, pred], dim=-1)
        outputs = x + pred
        return outputs


class GNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout_prob,
        num_dims,
        window_size,
        num_layers=4,
        params=None,
    ):
        super().__init__()
        self.num_dims = num_dims
        out_size = output_size
        self.window_size = window_size
        self.num_orientations = self.num_dims * (self.num_dims - 1) // 2
        # Relative features include: positions, orientations, positions in
        # spherical coordinates, and velocities
        loc_type = params.get("localizer_type", "baseline")

        self.num_relative_features = 2 * self.num_dims * self.window_size + (
            1 if params.get("use_z", False) else 0
        )

        initial_edge_features = (
            2 if params.get("use_z", False) else 1
        )  # It was 2. Now it is 1 because we are using the charge_ij as a feature +1 beacause quaternions
        num_edge_features = (
            self.num_relative_features
            + 9 * self.window_size
            + (5 if params.get("use_z", False) else 4)
        )

        self.gnn = nn.Sequential(
            GNNLayer(
                input_size,
                hidden_size,
                only_edge_attr=True,
                num_edge_features=num_edge_features,
            ),
            *[GNNLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)],
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x, edge_attr, edges):
        """
        inputs shape: [batch_size * num_objects, input_size]
        """
        for layer in self.gnn:
            x, edge_attr = layer(x, edge_attr, edges)

        # Output MLP
        pred = self.out_mlp(x)
        return pred


class GNNLayer(nn.Module):
    def __init__(
        self, input_size, hidden_size, only_edge_attr=False, num_edge_features=0
    ):
        super().__init__()

        # Neural Network Layers
        self.only_edge_attr = only_edge_attr
        num_edge_features = num_edge_features if only_edge_attr else 3 * hidden_size
        self.message_fn = nn.Sequential(
            nn.Linear(num_edge_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )

        self.res = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size
            else nn.Identity()
        )

        self.update_fn = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.SiLU(),
            nn.Linear(2 * hidden_size, hidden_size),
        )

    def forward(self, x, edge_attr, edges):
        """
        inputs shape: [batch_size, num_objects, input_size]
        """
        send_edges, recv_edges = edges
        if not self.only_edge_attr:
            edge_attr = torch.cat(
                [x[:, send_edges, :], x[:, recv_edges, :], edge_attr], dim=-1
            )

        edge_attr = self.message_fn(edge_attr)

        x = (
            self.res(x)
            + scatter(
                edge_attr, recv_edges.to(x.device), dim=1, reduce="mean"
            ).contiguous()
        )

        x = x + self.update_fn(x)

        return x, edge_attr


class LoCSModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        # params["num_dims"] = 2 if "synth" in params["data_path"] else 3
        # self.num_dims = params["num_dims"]
        self.window_size = params.get("window_size", 10)
        self.input_size = self.window_size * (params.get("gnn_n_in_dims", 6)) + (
            1 if params.get("use_z", False) else 0
        )
        self.output_size = self.window_size * params.get("gnn_n_out_dims", 3)
        self.hidden_size = params.get("gnn_hidden_size", 128)
        self.dropout_prob = params.get("gnn_dropout", 0.0)
        self.num_dims = params.get("gnn_n_out_dims", 3)
        self.num_layers = params.get("gnn_n_layers", 4)
        self.params = params
        device = "cuda" if params.get("gpu", False) else "cpu"
        self.model = LoCS(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            dropout_prob=self.dropout_prob,
            num_dims=self.num_dims,
            params=params,
            device=device,
            num_layers=self.num_layers,
        )

        self.loss_fn = nn.MSELoss()
        self.test_outputs = {}
        self.learning_rate = params["lr"]
        # save the hyperparameters
        self.save_hyperparameters()

    # def forward(self, batch, mode="train", teacher_forcing=True):
    #     # if "n_body" in self.params.get("data_path", "charged"):
    #     #     x, vel, edge_attr, charges, target = batch
    #     #     batch_size = x.shape[0]

    #     #     sequence = torch.cat([x, vel], dim=-1)[:, -self.window_size :, :, :]
    #     #     x = x[:, -1, :, :]
    #     #     vel = vel[:, -1, :, :]
    #     #     edges = None
    #     # else:
    #     # the input is a sequence of length 49. We ate going to use the first window_size
    #     # frames to predict the next frame in the sequence and so on until we reach the end

    #     edges = batch["edges"]
    #     charges = batch["charges"]
    #     nodes = None
    #     trajectory_length = batch["inputs"].shape[1]
    #     predicitons = []
    #     for step in range(self.window_size, trajectory_length):
    #         if teacher_forcing or step == self.window_size:
    #             x = batch["inputs"][:, step - 1, :, : self.num_dims].squeeze()
    #             vel = batch["inputs"][
    #                 :, step - 1, :, self.num_dims : 2 * self.num_dims
    #             ].squeeze()
    #             sequence = batch["inputs"][:, step - self.window_size : step, :, :]
    #         else:
    #             x = pred[:, :, : self.num_dims].squeeze()
    #             vel = pred[:, :, self.num_dims : 2 * self.num_dims].squeeze()
    #             sequence = torch.cat([sequence[:, 1:, :, :], pred.unsqueeze(1)], dim=1)

    #         pred = self.model(nodes, x.detach(), edges, vel, charges, sequence)
    #         predicitons.append(pred[..., : self.num_dims])

    #     # pdb.set_trace()
    #     pred = torch.stack(predicitons, dim=1)
    #     target = batch["inputs"][:, self.window_size :, :, : self.num_dims].squeeze()

    #     if "n_body" not in self.params.get("data_path", "charged"):
    #         if mode == "test":
    #             # Access the dataset to unnormalize the predictions
    #             pred = (
    #                 self.trainer.datamodule.test_dataloader().dataset.torch_unnormalize(
    #                     pred
    #                 )
    #             )
    #             target = (
    #                 self.trainer.datamodule.test_dataloader().dataset.torch_unnormalize(
    #                     target
    #                 )
    #             )
    #         elif mode == "val":
    #             pred = (
    #                 self.trainer.datamodule.val_dataloader().dataset.torch_unnormalize(
    #                     pred
    #                 )
    #             )
    #             target = (
    #                 self.trainer.datamodule.val_dataloader().dataset.torch_unnormalize(
    #                     target
    #                 )
    #             )
    #     loss = self.loss_fn(pred, target)

    #     return loss

    def forward(self, batch, mode="train"):
        if "n_body" in self.params.get("data_path", "charged"):
            x, vel, edge_attr, charges, target = batch
            batch_size = x.shape[0]

            sequence = torch.cat([x, vel], dim=-1)[:, -self.window_size :, :, :]
            x = x[:, -1, :, :]
            vel = vel[:, -1, :, :]
            edges = None
        else:
            x = batch["inputs"][:, -1, :, : self.num_dims].squeeze(1)
            vel = batch["inputs"][:, -1, :, self.num_dims : 2 * self.num_dims].squeeze(
                1
            )
            edges = batch["edges"]
            charges = batch["charges"]
            target = batch["target"][..., : self.num_dims].squeeze(1)
            if self.window_size < 10:
                sequence = batch["inputs"][:, -self.window_size :, :, :]
            else:
                sequence = batch["inputs"]
        # Create an empty list and make it a tensor
        nodes = None

        pred = self.model(nodes, x.detach(), edges, vel, charges, sequence)[
            ..., : self.num_dims
        ]

        if "n_body" not in self.params.get("data_path", "charged"):
            if mode == "test":
                # Access the dataset to unnormalize the predictions
                pred = (
                    self.trainer.datamodule.test_dataloader().dataset.torch_unnormalize(
                        pred
                    )
                )
                target = (
                    self.trainer.datamodule.test_dataloader().dataset.torch_unnormalize(
                        target
                    )
                )
            elif mode == "val":
                pred = (
                    self.trainer.datamodule.val_dataloader().dataset.torch_unnormalize(
                        pred
                    )
                )
                target = (
                    self.trainer.datamodule.val_dataloader().dataset.torch_unnormalize(
                        target
                    )
                )
        loss = self.loss_fn(pred, target)

        return loss

    def configure_optimizers(self):
        # The default is Adam, but SGD can be used by setting use_adam=False
        weight_decay = self.params.get("wd", 1e-12)
        if self.params.get("use_adam", False):
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=weight_decay
            )
        else:
            momentum = self.params.get("mom", 0.0)
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        if self.params.get("lr_scheduler", False) is not None:
            training_scheduler = train_utils.build_scheduler(optimizer, self.params)
            return [optimizer], [training_scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def _shared_step(self, batch, batch_idx, mode):
        loss = self.forward(batch, mode=mode)
        if isinstance(batch, list):
            batch_size = batch[0].shape[0]
        else:
            batch_size = batch["inputs"].shape[0]
        results = {"loss": loss, "batch_size": batch_size}
        if mode == "test":
            self.log("test/loss", loss.item(), on_epoch=True, prog_bar=True)
        return results

    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "val")

    def _shared_epoch_end(self, outputs, mode):
        """Accumulates and logs per-model metrics at the end of the epoch"""
        loss = torch.stack([x["loss"] * x["batch_size"] for x in outputs]).sum() / sum(
            [x["batch_size"] for x in outputs]
        )
        self.log(f"{mode}/loss", loss.item(), on_epoch=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "test")


class LoCSModule_Energies(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        # params["num_dims"] = 2 if "synth" in params["data_path"] else 3
        # self.num_dims = params["num_dims"]
        self.window_size = params.get("window_size", 10)
        self.input_size = self.window_size * (params.get("gnn_n_in_dims", 6)) + 1
        self.hidden_size = params.get("gnn_hidden_size", 128)
        self.dropout_prob = params.get("gnn_dropout", 0.0)
        self.num_dims = params.get("gnn_n_out_dims", 3)
        self.num_layers = params.get("gnn_n_layers", 4)
        self.params = params
        device = "cuda" if params.get("gpu", False) else "cpu"
        self.model = LoCS(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            dropout_prob=self.dropout_prob,
            num_dims=self.num_dims,
            params=params,
            device=device,
            num_layers=self.num_layers,
        )

        self.loss_fn = nn.MSELoss()
        self.test_outputs = {}
        self.learning_rate = params["lr"]
        # save the hyperparameters
        self.save_hyperparameters()

    def forward(self, batch, mode="train"):
        x = batch.pos
        atomic_numbers = batch.z
        force = batch.force
        target_energy = batch.y
        batch_size = batch.y.shape[0]

        edges = None

        # Create an empty list and make it a tensor
        nodes = None

        pred = self.model(nodes, x.detach(), edges, force, atomic_numbers)
        loss = self.loss_fn(pred, target_energy)

        return loss

    def configure_optimizers(self):
        # The default is Adam, but SGD can be used by setting use_adam=False
        weight_decay = self.params.get("wd", 1e-12)
        if self.params.get("use_adam", False):
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=weight_decay
            )
        else:
            momentum = self.params.get("mom", 0.0)
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        if self.params.get("lr_scheduler", False) is not None:
            training_scheduler = train_utils.build_scheduler(optimizer, self.params)
            return [optimizer], [training_scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def _shared_step(self, batch, batch_idx, mode):
        loss = self.forward(batch, mode=mode)
        if isinstance(batch, list):
            batch_size = batch[0].shape[0]
        else:
            batch_size = batch["inputs"].shape[0]
        results = {"loss": loss, "batch_size": batch_size}
        if mode == "test":
            self.log("test/loss", loss.item(), on_epoch=True, prog_bar=True)
        return results

    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "val")

    def _shared_epoch_end(self, outputs, mode):
        """Accumulates and logs per-model metrics at the end of the epoch"""
        loss = torch.stack([x["loss"] * x["batch_size"] for x in outputs]).sum() / sum(
            [x["batch_size"] for x in outputs]
        )
        self.log(f"{mode}/loss", loss.item(), on_epoch=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "test")
