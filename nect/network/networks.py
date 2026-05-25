from __future__ import annotations

import copy
import math
from logging import warning

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tinycudann as tcnn
except OSError:
    pass
from typing import TYPE_CHECKING

from nect.network.kplanes import init_grid_param, interpolate_ms_features

if TYPE_CHECKING:
    from nect.config import (
        DenseGridEncoderConfig,
        HashEncoderConfig,
        KPlanesEncoderConfig,
        MLPNetConfig,
        PirateNetConfig,
        TransformerDecoderConfig,
        UNetDecoderConfig,
    )


class HashGrid(nn.Module):
    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        network_config: MLPNetConfig,
    ):
        super().__init__()
        self.include_identity = network_config.include_identity
        if self.include_identity:
            encoding = {
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        **encoding_config.get_encoder_config()
                    },
                    {
                        "n_dims_to_encode": 3,
                        "otype": "Identity"
                    }
                ]
            }
        else:
            encoding = encoding_config.get_encoder_config()
        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=3 + (3 if self.include_identity else 0),
            n_output_dims=1,
            encoding_config=encoding,
            network_config=network_config.get_network_config(),
        )   

    def forward(self, x):
        if self.include_identity:
            inputs = torch.cat([x, x], dim=-1)
        else:
            inputs = x
        out = self.net(inputs)
        return out

class MultiResolutionHashTimeDoubleNetwork(nn.Module):
    def __init__(self, encoding_config: HashEncoderConfig, network_config: MLPNetConfig) -> None:
        super(MultiResolutionHashTimeDoubleNetwork, self).__init__()
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1,
            encoding_config=encoding_config.get_encoder_config(),
            network_config=network_config.get_network_config(),
        )
        self.time_model = tcnn.NetworkWithInputEncoding(
            n_input_dims=4,
            n_output_dims=1,
            encoding_config=encoding_config.get_encoder_config(),
            network_config=network_config.get_network_config(),
        )

    def forward(self, x, t):
        x_out = self.model(x)
        t_input = torch.cat([x, torch.full((x.size(0), 1), t, device=x.device)], dim=1)
        time_out = self.time_model(t_input)
        return x_out + time_out


class MultiResolutionHashTimeDoubleEncoderNetwork(nn.Module):
    def __init__(self, encoding_config: HashEncoderConfig, network_config: MLPNetConfig) -> None:
        super(MultiResolutionHashTimeDoubleEncoderNetwork, self).__init__()
        self.include_adaptive_skip = network_config.include_adaptive_skip
        if network_config.include_adaptive_skip:
            self.encoder_static = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config.get_encoder_config(),
            )
            self.encoder_dynamic = tcnn.Encoding(
                n_input_dims=4,
                encoding_config=encoding_config.get_encoder_config(),
            )
            self.skip_alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
            original_n_hidden_layers = network_config.n_hidden_layers
            network_config.n_hidden_layers = math.floor(original_n_hidden_layers / 2)
            n_neurons = self.encoder_dynamic.n_output_dims + self.encoder_static.n_output_dims
            network_config.n_neurons = n_neurons
            if n_neurons not in [16, 32, 64, 128]:
                network_config.otype = "CutlassMLP"
            original_output_activation = network_config.output_activation
            network_config.output_activation = network_config.activation
            self.net_1 = tcnn.Network(
                n_input_dims=n_neurons,
                n_output_dims=n_neurons,
                network_config=network_config.get_network_config(),
            )
            network_config_2 = copy.deepcopy(network_config)
            network_config_2.n_hidden_layers = original_n_hidden_layers - network_config.n_hidden_layers
            network_config_2.output_activation = original_output_activation
            self.net_2 = tcnn.Network(
                n_input_dims=n_neurons,
                n_output_dims=1,
                network_config=network_config_2.get_network_config(),
                seed=1,
            )

        else:
            encoding = {
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        **encoding_config.get_encoder_config()
                    },
                    {
                        "n_dims_to_encode": 4,
                        **encoding_config.get_encoder_config()
                    },
                ]
            }
            self.net = tcnn.NetworkWithInputEncoding(
                n_input_dims=7,
                n_output_dims=1,
                network_config=network_config.get_network_config(),
                encoding_config=encoding
            )

    def forward(self, x, t):
        t_input = torch.cat([x, torch.full((x.size(0), 1), t, device=x.device)], dim=1)
        if self.include_adaptive_skip:
            static_features = self.encoder_static(x)
            dynamic_features = self.encoder_dynamic(t_input)
            feature_tensor = torch.cat([static_features, dynamic_features], dim=-1)
            out = self.net_1(feature_tensor)
            return self.net_2(out * self.skip_alpha + (1 - self.skip_alpha) * feature_tensor)
        return self.net(torch.cat([x, t_input], dim=-1))


class MultiResolutionHashTimeDoubleEncoderDoubleNetwork(nn.Module):
    def __init__(self, encoding_config, network_config) -> None:
        super(MultiResolutionHashTimeDoubleEncoderDoubleNetwork, self).__init__()
        print(encoding_config.get_encoder_config())
        print(network_config.get_network_config())
        self.static = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=64,
            encoding_config=encoding_config.get_encoder_config(),
            network_config=network_config.get_network_config(),
        )
        self.encoder_dynamic = tcnn.NetworkWithInputEncoding(
            n_input_dims=4,
            n_output_dims=64,
            encoding_config=encoding_config.get_encoder_config(),
            network_config=network_config.get_network_config(),
        )
        self.include_adaptive_skip = network_config.include_adaptive_skip
        if network_config.include_adaptive_skip:
            self.skip_alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
            original_n_hidden_layers = network_config["n_hidden_layers"]
            if original_n_hidden_layers < 2:
                original_n_hidden_layers = 2
                warning("Original number of hidden layers is less than 2, setting to 2")
            network_config["n_hidden_layers"] = max(math.floor(original_n_hidden_layers / 2), 1)
            original_output_activation = network_config["output_activation"]
            network_config["output_activation"] = original_output_activation
            self.net_1 = tcnn.Network(
                n_input_dims=128,
                n_output_dims=128,
                network_config=network_config.get_network_config(),
            )
            network_config_2 = copy.deepcopy(network_config)
            network_config_2["n_hidden_layers"] = original_n_hidden_layers - network_config["n_hidden_layers"]
            self.net_2 = tcnn.Network(
                n_input_dims=128,
                n_output_dims=1,
                network_config=network_config_2,
                seed=1,
            )
        else:
            self.net = tcnn.Network(
                n_input_dims=128,
                n_output_dims=1,
                network_config=network_config.get_network_config(),
            )

    def forward(self, x, t):
        static_features = self.static(x)
        t_input = torch.cat([x, torch.full((x.size(0), 1), t, device=x.device)], dim=1)
        dynamic_features = self.encoder_dynamic(t_input)
        features = torch.cat([static_features, dynamic_features], dim=-1)
        if self.include_adaptive_skip:
            out = self.net_1(features)
            out = self.net_2(out * self.skip_alpha + (1 - self.skip_alpha) * features)
        else:
            out = self.net(features)
        return out


class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GatedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, input1, input2):
        out = F.leaky_relu(self.linear(x))
        return out * input1 + (1 - out) * input2


class AdaptiveBlock(nn.Module):
    def __init__(self, n_input_dims, n_output_dims, alfa_init) -> None:
        super(AdaptiveBlock, self).__init__()
        self.dense1 = GatedLinear(n_input_dims, n_input_dims)
        self.dense2 = GatedLinear(n_input_dims, n_input_dims)
        self.dense3 = nn.Linear(n_input_dims, n_input_dims)
        self.alfa = nn.Parameter(torch.tensor(alfa_init), requires_grad=True)

    def forward(self, x, inpu1, input2):
        dense1 = self.dense1(x, inpu1, input2)
        dense2 = self.dense2(dense1, inpu1, input2)
        dense3 = F.leaky_relu(self.dense3(dense2))
        dense3 = dense3 * self.alfa + (1 - self.alfa) * x
        return dense3


class PirateNetwork(nn.Module):
    def __init__(self, n_input_dims, n_output_dims, n_modules, alfa_init) -> None:
        super(PirateNetwork, self).__init__()
        self.dense1 = nn.Linear(n_input_dims, n_input_dims)
        self.dense2 = nn.Linear(n_input_dims, n_input_dims)
        self.adaptive_block = nn.ModuleList(
            AdaptiveBlock(n_input_dims, n_input_dims, alfa_init) for _ in range(n_modules)
        )
        self.out = nn.Linear(n_input_dims, n_output_dims, bias=True)

    def forward(self, x):
        dense1 = F.leaky_relu(self.dense1(x))
        dense2 = F.leaky_relu(self.dense2(x))
        for block in self.adaptive_block:
            x = block(x, dense1, dense2)
        out = self.out(x)
        if out.max() < 0:
            return out
        return F.relu(out)


class PirateNet(nn.Module):
    def __init__(self, encoding_config: HashEncoderConfig, network_config: PirateNetConfig) -> None:
        super(PirateNet, self).__init__()
        self.hash_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=encoding_config.get_encoder_config(),
        )
        self.net = PirateNetwork(
            n_input_dims=self.hash_encoder.n_output_dims,
            n_output_dims=1,
            n_modules=network_config.n_modules,
            alfa_init=network_config.alfa_init,
        )

    def forward(self, x):
        x_encoded = self.hash_encoder(x)
        out = self.net(x_encoded)
        return out


class DynamicPirateNet(nn.Module):
    def __init__(self, encoding_config: HashEncoderConfig, network_config: PirateNetConfig) -> None:
        super(DynamicPirateNet, self).__init__()
        self.hash_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=encoding_config.get_encoder_config(),
        )
        self.dynamic_encoder = tcnn.Encoding(
            n_input_dims=4,
            encoding_config=encoding_config.get_encoder_config(),
        )
        self.net = PirateNetwork(
            n_input_dims=self.hash_encoder.n_output_dims + self.dynamic_encoder.n_output_dims,
            n_output_dims=1,
            n_modules=network_config.n_modules,
            alfa_init=network_config.alfa_init,
        )

    def forward(self, x, t):
        x_encoded = self.hash_encoder(x)
        t_encoded = self.dynamic_encoder(torch.cat([x, torch.full((x.size(0), 1), t, device=x.device)], dim=1))
        out = self.net(torch.cat([x_encoded, t_encoded], dim=-1))
        return out

class QuadCubesOld(nn.Module):
    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        network_config: MLPNetConfig,
        prior=False,
        concat=True,
    ):
        super().__init__()
        self.concat = concat
        self.scale_factor = 100
        self.static = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=encoding_config.get_encoder_config(),
        )
        self.prior = prior
        if not prior:
            self.xyt = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config.get_encoder_config(),
            )
            self.xzt = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config.get_encoder_config(),
            )
            self.yzt = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config.get_encoder_config(),
            )
        self.include_identity = network_config.include_identity
        additional_parameters = 4 if network_config.include_identity else 0
        self.net = tcnn.Network(
            n_input_dims=self.static.n_output_dims * (4 if self.concat else 1) + additional_parameters,
            n_output_dims=1,
            network_config=network_config.get_network_config(),
        )
        if self.concat is False:
            for encoder in [self.static, self.xyt, self.xzt, self.yzt]:
                for param in encoder.parameters():
                    torch.nn.init.ones_(param.data)

    def forward(self, x, t):
        if self.include_identity and self.concat:
            xyzt_input = torch.cat([x, torch.full((x.size(0), 1), t, device=x.device)], dim=1)
        static_encoded = self.static(x)
        if not self.prior:
            xyt_encoded = self.xyt(
                torch.cat(
                    [x[..., [1, 2]], torch.full((x.size(0), 1), t, device=x.device)],
                    dim=1,
                )
            )
            xzt_encoded = self.xzt(
                torch.cat(
                    [x[..., [0, 2]], torch.full((x.size(0), 1), t, device=x.device)],
                    dim=1,
                )
            )
            yzt_encoded = self.yzt(
                torch.cat(
                    [x[..., [0, 1]], torch.full((x.size(0), 1), t, device=x.device)],
                    dim=1,
                )
            )
        else:
            xyt_encoded = torch.zeros_like(static_encoded)
            xzt_encoded = torch.zeros_like(static_encoded)
            yzt_encoded = torch.zeros_like(static_encoded)
        if self.concat:
            if self.include_identity:
                to_mlp = torch.cat([static_encoded, xyt_encoded, xzt_encoded, yzt_encoded, xyzt_input], dim=-1)
            else:
                to_mlp = torch.cat([static_encoded, xyt_encoded, xzt_encoded, yzt_encoded], dim=-1)
        else:
            if self.include_identity:
                to_mlp = torch.cat([static_encoded * xyt_encoded * xzt_encoded * yzt_encoded, xyzt_input], dim=-1)
            else:
                to_mlp = static_encoded * xyt_encoded * xzt_encoded * yzt_encoded
        out = self.net(to_mlp)
        return out


class QuadCubes(nn.Module):
    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        network_config: MLPNetConfig,
        prior=False,
        concat=True,
    ):
        super().__init__()
        self.concat = concat
        self.include_identity = network_config.include_identity
        if not prior:
            encoding = {
                "otype": "Composite",
                "nested": [
                    {"n_dims_to_encode": 3, **encoding_config.get_encoder_config()},
                    {"n_dims_to_encode": 3, **encoding_config.get_encoder_config()},
                    {"n_dims_to_encode": 3, **encoding_config.get_encoder_config()},
                    {"n_dims_to_encode": 3, **encoding_config.get_encoder_config()}
                ]
            }
            if self.include_identity:
                encoding["nested"].append({"n_dims_to_encode": 4, "otype": "Identity"})

        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=12 + (4 if self.include_identity else 0),
            n_output_dims=1,
            encoding_config=encoding,
            network_config=network_config.get_network_config(),
        )

    def forward(self, zyx, t):
        yxt = torch.cat([zyx[..., [1, 2]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        xzt = torch.cat([zyx[..., [2, 0]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        zyt = torch.cat([zyx[..., [0, 1]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        if self.include_identity:
            zyxt = torch.cat([zyx, torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
            inputs = torch.cat([zyx, yxt, xzt, zyt, zyxt], dim=-1)
        else:
            inputs = torch.cat([zyx, yxt, xzt, zyt], dim=-1)
        out = self.net(inputs)
        return out


class SplitQuadCubes(nn.Module):
    """QuadCubes with separate encoder configs for the spatial (zyx) and temporal (yxt, xzt, zyt) encoders."""

    def __init__(
        self,
        spatial_encoding_config: HashEncoderConfig,
        temporal_encoding_config: HashEncoderConfig,
        network_config: MLPNetConfig,
    ):
        super().__init__()
        self.include_identity = network_config.include_identity
        encoding = {
            "otype": "Composite",
            "nested": [
                {"n_dims_to_encode": 3, **spatial_encoding_config.get_encoder_config()},
                {"n_dims_to_encode": 3, **temporal_encoding_config.get_encoder_config()},
                {"n_dims_to_encode": 3, **temporal_encoding_config.get_encoder_config()},
                {"n_dims_to_encode": 3, **temporal_encoding_config.get_encoder_config()},
            ]
        }
        if self.include_identity:
            encoding["nested"].append({"n_dims_to_encode": 4, "otype": "Identity"})

        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=12 + (4 if self.include_identity else 0),
            n_output_dims=1,
            encoding_config=encoding,
            network_config=network_config.get_network_config(),
        )

    def forward(self, zyx, t):
        yxt = torch.cat([zyx[..., [1, 2]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        xzt = torch.cat([zyx[..., [2, 0]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        zyt = torch.cat([zyx[..., [0, 1]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        if self.include_identity:
            zyxt = torch.cat([zyx, torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
            inputs = torch.cat([zyx, yxt, xzt, zyt, zyxt], dim=-1)
        else:
            inputs = torch.cat([zyx, yxt, xzt, zyt], dim=-1)
        return self.net(inputs)


class HyperCubes(nn.Module):
    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        network_config: MLPNetConfig,
    ):
        super().__init__()
        self.include_identity = network_config.include_identity
        encoding = {
            "otype": "Composite",
            "nested": [
                {
                    "n_dims_to_encode": 3,
                    **encoding_config.get_encoder_config()
                },
                {
                    "n_dims_to_encode": 3,
                    **encoding_config.get_encoder_config()
                },
                {
                    "n_dims_to_encode": 3,
                    **encoding_config.get_encoder_config()
                },
                {
                    "n_dims_to_encode": 3,
                    **encoding_config.get_encoder_config()
                },
                {
                    "n_dims_to_encode": 4,
                    **encoding_config.get_encoder_config()
                }
            ]
        }
        if self.include_identity:
            encoding["nested"].append(
                {
                    "n_dims_to_encode": 4,
                    "otype": "Identity"
                }
            )
        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=16 + (4 if self.include_identity else 0),
            n_output_dims=1,
            encoding_config=encoding,
            network_config=network_config.get_network_config(),
        )

    def forward(self, zyx, t):
        yxt = torch.cat([zyx[..., [1, 2]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        xzt = torch.cat([zyx[..., [2, 0]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        zyt = torch.cat([zyx[..., [0, 1]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        zyxt = torch.cat([zyx, torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        if self.include_identity:
            inputs = torch.cat([zyx, yxt, xzt, zyt, zyxt, zyxt], dim=-1)
        else:
            inputs = torch.cat([zyx, yxt, xzt, zyt, zyxt], dim=-1)
        out = self.net(inputs)
        return out


class KPlanes(nn.Module):
    def __init__(self, encoding_config: KPlanesEncoderConfig, network_config: MLPNetConfig):
        super().__init__()
        self.multiscale_res_multipliers = [1, 2, 4, 8]
        self.concat_features = True
        self.encoding_config = encoding_config

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feature_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            # encoding_config = self.encoding_config.copy()  # Avoids in-place problems
            # Resolution fix: multi-res only on spatial planes
            resolution = [r * res for r in encoding_config.resolution[:3]] + encoding_config.resolution[3:]

            gp = init_grid_param(
                grid_nd=encoding_config.grid_dimensions,
                in_dim=encoding_config.input_coordinate_dim,
                out_dim=encoding_config.output_coordinate_dim,
                reso=resolution,
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]
            self.grids.append(gp)

            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config=network_config.get_network_config(),
            )

    def forward(self, x, t=None):
        if t is not None:
            x = torch.cat([x, torch.full((x.size(0), 1), t, device=x.device)], dim=1)

        features = interpolate_ms_features(
            x,
            ms_grids=self.grids,  # noqa
            grid_dimensions=self.encoding_config.grid_dimensions,
            concat_features=self.concat_features,
            num_levels=None,
        )

        output = self.sigma_net(features)
        return output

class TriCubes(nn.Module):
    """3 pairwise 2D encoders: (x,y), (x,z), (y,z)."""
    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        network_config: MLPNetConfig,
    ):
        super().__init__()
        self.include_identity = network_config.include_identity

        encoding = {
            "otype": "Composite",
            "nested": [
                {"n_dims_to_encode": 2, **encoding_config.get_encoder_config()},  # xy
                {"n_dims_to_encode": 2, **encoding_config.get_encoder_config()},  # xz
                {"n_dims_to_encode": 2, **encoding_config.get_encoder_config()},  # yz
            ],
        }
        if self.include_identity:
            # Identity over original xyz (3 dims) is usually what you want for skip-like behavior.
            encoding["nested"].append({"n_dims_to_encode": 3, "otype": "Identity"})

        n_in = 6 + (3 if self.include_identity else 0)

        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=n_in,
            n_output_dims=1,
            encoding_config=encoding,
            network_config=network_config.get_network_config(),
        )

    def forward(self, x):  # x: (N,3) with cols [x,y,z]
        xy = x[:, [0, 1]]
        xz = x[:, [0, 2]]
        yz = x[:, [1, 2]]
        inputs = torch.cat([xy, xz, yz], dim=-1)  # (N,6)
        if self.include_identity:
            inputs = torch.cat([inputs, x], dim=-1)  # + (N,3)
        return self.net(inputs)


class SexCubes(nn.Module):
    """6 pairwise 2D encoders: (x,y),(x,z),(y,z),(x,t),(z,t),(y,t)."""
    def __init__(self, encoding_config: HashEncoderConfig, network_config: MLPNetConfig):
        super().__init__()
        self.include_identity = network_config.include_identity

        encoding = {
            "otype": "Composite",
            "nested": [
                {"n_dims_to_encode": 2, **encoding_config.get_encoder_config()},  # xy
                {"n_dims_to_encode": 2, **encoding_config.get_encoder_config()},  # xz
                {"n_dims_to_encode": 2, **encoding_config.get_encoder_config()},  # yz
                {"n_dims_to_encode": 2, **encoding_config.get_encoder_config()},  # xt
                {"n_dims_to_encode": 2, **encoding_config.get_encoder_config()},  # zt
                {"n_dims_to_encode": 2, **encoding_config.get_encoder_config()},  # yt
            ],
        }
        if self.include_identity:
            encoding["nested"].append({"n_dims_to_encode": 4, "otype": "Identity"})

        n_in = 12 + (4 if self.include_identity else 0)

        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=n_in,
            n_output_dims=1,
            encoding_config=encoding,
            network_config=network_config.get_network_config(),
        )

    def forward(self, x, t):  
        # x: (N,3), t: scalar float or 0-d/1-d tensor
        # make (N,1) time column
        if not torch.is_tensor(t):
            tcol = torch.full((x.size(0), 1), float(t), device=x.device, dtype=x.dtype)
        else:
            tcol = t.reshape(-1, 1).to(device=x.device, dtype=x.dtype)
            if tcol.size(0) == 1:
                tcol = tcol.expand(x.size(0), 1)

        xy = x[:, [0, 1]]
        xz = x[:, [0, 2]]
        yz = x[:, [1, 2]]
        xt = torch.cat([x[:, [0]], tcol], dim=-1)
        zt = torch.cat([x[:, [2]], tcol], dim=-1)
        yt = torch.cat([x[:, [1]], tcol], dim=-1)

        inputs = torch.cat([xy, xz, yz, xt, zt, yt], dim=-1)  # (N,12)
        if self.include_identity:
            xyzt = torch.cat([x, tcol], dim=-1)                # (N,4)
            inputs = torch.cat([inputs, xyzt], dim=-1)

        return self.net(inputs)

class SingleCube(nn.Module):
    def __init__(self, encoding_config: HashEncoderConfig, network_config: MLPNetConfig):
        super().__init__()
        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=4,
            n_output_dims=1,
            encoding_config=encoding_config.get_encoder_config(),
            network_config=network_config.get_network_config(),
        )

    def forward(self, zyx, t):
        if not torch.is_tensor(t):
            tcol = torch.full((zyx.size(0), 1), float(t), device=zyx.device, dtype=zyx.dtype)
        else:
            tcol = t.reshape(-1, 1).to(device=zyx.device, dtype=zyx.dtype)
            if tcol.size(0) == 1:
                tcol = tcol.expand(zyx.size(0), 1)
        zyxt = torch.cat([zyx, tcol], dim=1)
        return self.net(zyxt)


class CombinedCubes(nn.Module):
    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        network_config: MLPNetConfig,
        n_levels_temporal: int | None = None,
    ):
        super().__init__()
        enc_2d = encoding_config.get_encoder_config_2D()
        if n_levels_temporal is not None:
            enc_2d = {**enc_2d, "n_levels": n_levels_temporal}
        enc_3d = encoding_config.get_encoder_config()
        encoding = {
            "otype": "Composite",
            "nested": [
                {"n_dims_to_encode": 2, **enc_2d},  # z, t
                {"n_dims_to_encode": 2, **enc_2d},  # y, t
                {"n_dims_to_encode": 2, **enc_2d},  # x, t
                {"n_dims_to_encode": 3, **enc_3d},  # z, y, x
            ]
        }

        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=9,
            n_output_dims=1,
            encoding_config=encoding,
            network_config=network_config.get_network_config(),
        )

    def forward(self, zyx, t):
        tcol = torch.full((zyx.size(0), 1), t, device=zyx.device)
        xt = torch.cat([zyx[:, [0]], tcol], dim=-1)
        yt = torch.cat([zyx[:, [1]], tcol], dim=-1)
        zt = torch.cat([zyx[:, [2]], tcol], dim=-1)
        inputs = torch.cat([xt, yt, zt, zyx], dim=-1)
        out = self.net(inputs)
        return out


class MixedCubes(nn.Module):
    """CombinedCubes with collision-free dense 2D grids for the temporal pairs.

    Replaces the three 2D hash-grid encoders in CombinedCubes with dense
    2D grids (no hashing, no collisions).  The 3D spatial encoder (z,y,x)
    keeps the hash grid since a dense 3D grid would be too large.

    Input layout fed to the composite encoder (9 dims total):
        [zt(2), yt(2), xt(2), zyx(3)]
    which matches CombinedCubes exactly — only the encoder type differs.
    """

    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        encoding_config_2d: DenseGridEncoderConfig,
        network_config: MLPNetConfig,
    ):
        super().__init__()
        dense_cfg = encoding_config_2d.get_encoder_config()
        encoding = {
            "otype": "Composite",
            "nested": [
                {"n_dims_to_encode": 2, **dense_cfg},           # zt
                {"n_dims_to_encode": 2, **dense_cfg},           # yt
                {"n_dims_to_encode": 2, **dense_cfg},           # xt
                {"n_dims_to_encode": 3, **encoding_config.get_encoder_config()},  # zyx
            ],
        }
        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=9,
            n_output_dims=1,
            encoding_config=encoding,
            network_config=network_config.get_network_config(),
        )

    def forward(self, zyx, t):
        tcol = torch.full((zyx.size(0), 1), t, device=zyx.device)
        zt = torch.cat([zyx[:, [0]], tcol], dim=-1)
        yt = torch.cat([zyx[:, [1]], tcol], dim=-1)
        xt = torch.cat([zyx[:, [2]], tcol], dim=-1)
        inputs = torch.cat([zt, yt, xt, zyx], dim=-1)
        return self.net(inputs)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch-native decoder variants for the QuadCubes encoder.
#
# Motivation: swap TCNN's fixed MLP for a richer decoder so that smaller
# encoders (fewer levels / lower hash budget) can still produce sharp
# reconstructions.  The encoder itself is still TCNN for speed; only the
# decoder is pure PyTorch, making it trivial to replace later.
#
# Both classes expose the same forward(zyx, t) → [B, 1] interface as the
# existing TCNN models, so they drop straight into the existing trainer.
# ─────────────────────────────────────────────────────────────────────────────


class _QuadEncoder(nn.Module):
    """Four independent TCNN 3D hash-encoders for the QuadCubes decomposition.

    Encodes all six 3D projections of (z,y,x,t):
        enc_zyx : (z, y, x)
        enc_yxt : (y, x, t)
        enc_xzt : (x, z, t)
        enc_zyt : (z, y, t)

    Returns four float32 feature tensors so the downstream PyTorch decoder
    can operate in full precision (TCNN encoding outputs are float16).
    """

    def __init__(self, encoding_config: HashEncoderConfig):
        super().__init__()
        enc_cfg = encoding_config.get_encoder_config()
        self.enc_zyx = tcnn.Encoding(3, enc_cfg)
        self.enc_yxt = tcnn.Encoding(3, enc_cfg)
        self.enc_xzt = tcnn.Encoding(3, enc_cfg)
        self.enc_zyt = tcnn.Encoding(3, enc_cfg)
        self.n_output_dims = self.enc_zyx.n_output_dims  # same for all four

    def forward(self, zyx, t):
        """
        Args:
            zyx: [B, 3] coordinate tensor, columns are (z, y, x) in [0, 1].
            t:   scalar float timestep in [0, 1].

        Returns:
            Tuple of four [B, D] float32 feature tensors.
        """
        tcol = torch.full((zyx.size(0), 1), float(t), device=zyx.device, dtype=zyx.dtype)
        yxt = torch.cat([zyx[:, [1, 2]], tcol], dim=1)
        xzt = torch.cat([zyx[:, [2, 0]], tcol], dim=1)
        zyt = torch.cat([zyx[:, [0, 1]], tcol], dim=1)
        return (
            self.enc_zyx(zyx).float(),
            self.enc_yxt(yxt).float(),
            self.enc_xzt(xzt).float(),
            self.enc_zyt(zyt).float(),
        )


class _SelfAttnBlock(nn.Module):
    """Pre-LN multi-head self-attention block implemented with torch.matmul.

    Deliberately avoids torch.nn.MultiheadAttention (and its use of
    scaled_dot_product_attention) because those dispatch to flash-attention
    CUDA kernels that have a hard grid-size limit on batch*n_heads.  With up
    to 5 M coordinate points and 4 attention heads that limit is exceeded.
    Plain torch.matmul on [B, H, 4, 4] matrices works at any batch size.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, S, d_model]   S = 4 tokens
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head
        N = B * S
        # cuBLASLt on A100 in FP16 requires M (batch rows) to be a multiple of 8.
        # S=4, so N=B*4 is a multiple of 8 only when B is even.  Pad to be safe.
        pad = (-N) % 8

        def _linear_padded(linear, t2d):
            """Apply linear to [N, *] tensor with row-count padded to multiple of 8."""
            if pad:
                t2d = F.pad(t2d, (0, 0, 0, pad))
            out = linear(t2d)
            if pad:
                out = out[:N]
            return out

        # ── Self-attention (pre-LN) ──────────────────────────────────────────
        h = self.norm1(x)
        qkv = _linear_padded(self.qkv, h.reshape(N, D)).reshape(B, S, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)          # each [B, H, S, Dh]

        # [B, H, S, S]  — S=4, so this is always a tiny 4×4 matrix
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)      # [B, H, S, Dh]
        out = out.permute(0, 2, 1, 3)
        x = x + _linear_padded(self.out_proj, out.reshape(N, D)).reshape(B, S, D)

        # ── Feed-forward (pre-LN) ────────────────────────────────────────────
        x = x + _linear_padded(self.ff, self.norm2(x).reshape(N, D)).reshape(B, S, D)
        return x


class QuadCubesTransformer(nn.Module):
    """QuadCubes encoder + Transformer decoder.

    The four encoder feature vectors are treated as a sequence of 4 tokens.
    Self-attention lets each token attend to the others, then the tokens are
    mean-pooled and projected to a scalar output.

    Uses a manual attention implementation (_SelfAttnBlock) to avoid
    PyTorch's scaled_dot_product_attention CUDA kernels, which fail when
    batch_size * n_heads exceeds the hardware grid-size limit.

    Config:
        encoding_config : HashEncoderConfig  (same as QuadCubes)
        decoder_config  : TransformerDecoderConfig
            d_model  – token projection dimension
            n_heads  – number of attention heads (must divide d_model)
            n_layers – number of _SelfAttnBlock layers
            dropout  – dropout probability
    """

    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        decoder_config: TransformerDecoderConfig,
    ):
        super().__init__()
        self.encoder = _QuadEncoder(encoding_config)

        feat_dim_raw = self.encoder.n_output_dims
        d_model = decoder_config.d_model
        n_heads = decoder_config.n_heads
        n_layers = decoder_config.n_layers
        dropout = decoder_config.dropout

        # cuBLASLt on A100 requires the K dimension of any matmul to be a
        # multiple of 8 for Tensor Core operations.  feat_dim = n_levels *
        # n_features_per_level may not satisfy this (e.g. 21*4=84).  Pad to
        # the next multiple of 8 so token_proj never triggers NOT_SUPPORTED.
        self._feat_dim_raw = feat_dim_raw
        self._feat_dim = math.ceil(feat_dim_raw / 8) * 8
        self.token_proj = nn.Linear(self._feat_dim, d_model)

        self.pos_embed = nn.Parameter(torch.zeros(4, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [_SelfAttnBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        self.out_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, zyx, t):
        B = zyx.size(0)
        f_zyx, f_yxt, f_xzt, f_zyt = self.encoder(zyx, t)

        # [B, 4, feat_dim_raw] — pad K to multiple of 8 if needed
        tokens = torch.stack([f_zyx, f_yxt, f_xzt, f_zyt], dim=1)
        if self._feat_dim > self._feat_dim_raw:
            tokens = F.pad(tokens, (0, self._feat_dim - self._feat_dim_raw))

        # Reshape to 2D before token_proj.  cuBLASLt FP16 on A100 requires all
        # of m/n/k to be multiples of 8; B*4 is often not, so pad then trim.
        tokens_2d = tokens.contiguous().reshape(B * 4, -1)
        N = tokens_2d.size(0)
        pad = (-N) % 8  # == (8 - N%8) % 8
        if pad:
            tokens_2d = F.pad(tokens_2d, (0, 0, 0, pad))
        tokens_2d = self.token_proj(tokens_2d)
        if pad:
            tokens_2d = tokens_2d[:N]
        tokens = tokens_2d.reshape(B, 4, -1)
        tokens = tokens + self.pos_embed.unsqueeze(0)  # [B, 4, d_model]

        for block in self.blocks:
            tokens = block(tokens)

        # Mean pool → [B, d_model] → [B, 1]
        return self.out_head(tokens.mean(dim=1))


class QuadCubesUNet(nn.Module):
    """QuadCubes encoder + U-Net style decoder.

    The multi-resolution hash grid naturally produces features at different
    scales (coarse levels → low-frequency, fine levels → high-frequency).
    This decoder exploits that structure:

        Down-path  (coarse → medium → fine, each group fused with previous scale)
        Up-path    (fine → medium → coarse, with skip connections from the down-path)

    This mirrors U-Net's encoder/decoder skip connections, but for a 1D
    feature vector rather than a spatial feature map.

    Config:
        encoding_config : HashEncoderConfig  (same as QuadCubes)
        decoder_config  : UNetDecoderConfig
            hidden_dims   – list of 3 integers [d1, d2, d3]
                            d1 = coarse scale width
                            d2 = medium scale width
                            d3 = bottleneck (fine) width
            levels_coarse – how many hash levels go into the coarse group
            levels_medium – how many hash levels go into the medium group
                            remaining levels form the fine group
            dropout       – (reserved, currently unused in skip-MLP path)
    """

    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        decoder_config: UNetDecoderConfig,
    ):
        super().__init__()
        self.encoder = _QuadEncoder(encoding_config)

        nfpl = encoding_config.n_features_per_level  # features per level
        n_levels = encoding_config.n_levels
        lc = decoder_config.levels_coarse
        lm = decoder_config.levels_medium
        lf = n_levels - lc - lm
        if lf <= 0:
            raise ValueError(
                f"levels_coarse ({lc}) + levels_medium ({lm}) must be < n_levels ({n_levels})"
            )
        self._lc = lc
        self._lm = lm
        self._nfpl = nfpl

        # Dimensions of each scale group across all 4 encoders
        coarse_dim = 4 * lc * nfpl
        medium_dim = 4 * lm * nfpl
        fine_dim   = 4 * lf * nfpl

        d1, d2, d3 = decoder_config.hidden_dims

        # ── Down-path ────────────────────────────────────────────────────────
        # Each step ingests the current-scale features and the previous scale's
        # output (skip from above).
        self.down1 = nn.Sequential(nn.Linear(coarse_dim, d1), nn.ReLU())
        self.down2 = nn.Sequential(nn.Linear(medium_dim + d1, d2), nn.ReLU())
        self.bottleneck = nn.Sequential(nn.Linear(fine_dim + d2, d3), nn.ReLU())

        # ── Up-path (with skip connections from down-path) ────────────────────
        self.up2 = nn.Sequential(nn.Linear(d3 + d2, d2), nn.ReLU())
        self.up1 = nn.Sequential(nn.Linear(d2 + d1, d1), nn.ReLU())

        self.out_head = nn.Linear(d1, 1)

    def _split_levels(self, feat):
        """Split a single encoder's output into (coarse, medium, fine) groups."""
        c = self._lc * self._nfpl
        m = self._lm * self._nfpl
        return feat[:, :c], feat[:, c : c + m], feat[:, c + m :]

    def forward(self, zyx, t):
        f_zyx, f_yxt, f_xzt, f_zyt = self.encoder(zyx, t)

        # Split each encoder's features by resolution scale
        c0, m0, f0 = self._split_levels(f_zyx)
        c1, m1, f1 = self._split_levels(f_yxt)
        c2, m2, f2 = self._split_levels(f_xzt)
        c3, m3, f3 = self._split_levels(f_zyt)

        # Concatenate across encoders at each scale
        coarse = torch.cat([c0, c1, c2, c3], dim=-1)
        medium = torch.cat([m0, m1, m2, m3], dim=-1)
        fine   = torch.cat([f0, f1, f2, f3], dim=-1)

        # Down-path
        e1 = self.down1(coarse)
        e2 = self.down2(torch.cat([medium, e1], dim=-1))
        e3 = self.bottleneck(torch.cat([fine, e2], dim=-1))

        # Up-path with skip connections
        d2 = self.up2(torch.cat([e3, e2], dim=-1))
        d1 = self.up1(torch.cat([d2, e1], dim=-1))

        return self.out_head(d1)


# ─────────────────────────────────────────────────────────────────────────────
# SexCubesKPlanes / CombinedCubesKPlanes / MixedCubesKPlanes
#
# Multiplying complementary plane pairs that together cover all 4 coordinates:
#   f_zy * f_xt  →  z, y, x, t all present
#   f_zx * f_yt  →  z, x, y, t all present
#   f_yx * f_zt  →  y, x, z, t all present
#
# A point only activates a product when both of its 2D projections match, making
# the features far more selective than plain concatenation. The MLP no longer
# has to figure out coordinate coupling on its own — the products encode it
# explicitly. Inspired by K-Planes (Fridovich-Keil et al., 2023).
# ─────────────────────────────────────────────────────────────────────────────

def _str_to_act(name: str) -> nn.Module:
    return {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        "SiLU": nn.SiLU,
        "None": nn.Identity,
    }.get(name, nn.ReLU)()


def _build_mlp(in_dim: int, network_config) -> nn.Module:
    """Build an MLP head, using tcnn.Network for FullyFusedMLP/CutlassMLP."""
    if network_config.otype in ("FullyFusedMLP", "CutlassMLP"):
        return tcnn.Network(in_dim, 1, network_config.get_network_config())
    w = network_config.n_neurons
    layers: list[nn.Module] = [nn.Linear(in_dim, w), _str_to_act(network_config.activation)]
    for _ in range(network_config.n_hidden_layers - 1):
        layers += [nn.Linear(w, w), _str_to_act(network_config.activation)]
    layers.append(nn.Linear(w, 1))
    if network_config.output_activation != "None":
        layers.append(_str_to_act(network_config.output_activation))
    return nn.Sequential(*layers)


class SexCubesKPlanes(nn.Module):
    """SexCubes encoder + K-Planes product decoder.

    Feeds both the 6 raw plane features and their 3 complementary Hadamard
    products into a plain PyTorch MLP (float32). The products give the MLP
    explicit 4D selectivity so it does not need extreme depth to couple coords.

    Input to MLP: cat([f_zy, f_zx, f_yx, f_zt, f_yt, f_xt,
                        f_zy*f_xt, f_zx*f_yt, f_yx*f_zt])  →  9 × D features.

    Config:
        encoding_config : HashEncoderConfig  (same 2D grid as SexCubes)
        network_config  : MLPNetConfig
            n_neurons        – width of hidden layers
            n_hidden_layers  – depth of MLP
            activation       – hidden activation (ReLU, LeakyReLU, SiLU, …)
            output_activation – final activation (typically None)
    """

    def __init__(self, encoding_config: "HashEncoderConfig", network_config: "MLPNetConfig"):
        super().__init__()
        self.encoder = _SexEncoder(encoding_config)
        D = self.encoder.n_output_dims
        in_dim = 9 * D  # 6 raw + 3 products
        self.mlp = _build_mlp(in_dim, network_config)

    def forward(self, zyx, t):
        f_zy, f_zx, f_yx, f_zt, f_yt, f_xt = self.encoder(zyx, t)

        p0 = f_zy * f_xt   # z, y, x, t
        p1 = f_zx * f_yt   # z, x, y, t
        p2 = f_yx * f_zt   # y, x, z, t

        h = torch.cat([f_zy, f_zx, f_yx, f_zt, f_yt, f_xt, p0, p1, p2], dim=-1)
        return self.mlp(h).float()


# ─────────────────────────────────────────────────────────────────────────────
# SexCubes variants with richer decoders.
#
# SexCubes decomposes (z,y,x,t) into all 6 pairwise 2D projections.  Each
# 2D encoder captures less information than a 3D encoder (QuadCubes), so the
# features are more "disconnected" and a plain MLP struggles unless it is made
# very large.  A richer decoder (Transformer or UNet) that explicitly models
# interactions between the 6 features can compensate without bloating the MLP.
#
# _SexEncoder wraps the 6 TCNN 2D encoders and exposes pure PyTorch tensors,
# exactly like _QuadEncoder does for QuadCubes.
# ─────────────────────────────────────────────────────────────────────────────


class _SexEncoder(nn.Module):
    """Six independent TCNN 2D hash-encoders for all pairwise projections of (z,y,x,t).

    Pairs (in zyx input convention):
        enc_zy : (z, y)
        enc_zx : (z, x)
        enc_yx : (y, x)
        enc_zt : (z, t)
        enc_yt : (y, t)
        enc_xt : (x, t)

    Returns six float32 feature tensors so the downstream PyTorch decoder
    can operate in full precision (TCNN encoding outputs are float16).
    """

    def __init__(self, encoding_config: HashEncoderConfig):
        super().__init__()
        enc_cfg = encoding_config.get_encoder_config_2D()
        self.enc_zy = tcnn.Encoding(2, enc_cfg)
        self.enc_zx = tcnn.Encoding(2, enc_cfg)
        self.enc_yx = tcnn.Encoding(2, enc_cfg)
        self.enc_zt = tcnn.Encoding(2, enc_cfg)
        self.enc_yt = tcnn.Encoding(2, enc_cfg)
        self.enc_xt = tcnn.Encoding(2, enc_cfg)
        self.n_output_dims = self.enc_zy.n_output_dims  # same for all six

    def forward(self, zyx, t):
        """
        Args:
            zyx: [B, 3] coordinate tensor, columns are (z, y, x) in [0, 1].
            t:   scalar float timestep in [0, 1].

        Returns:
            Tuple of six [B, D] float32 feature tensors.
        """
        tcol = torch.full((zyx.size(0), 1), float(t), device=zyx.device, dtype=zyx.dtype)
        zy = zyx[:, [0, 1]]
        zx = zyx[:, [0, 2]]
        yx = zyx[:, [1, 2]]
        zt = torch.cat([zyx[:, [0]], tcol], dim=1)
        yt = torch.cat([zyx[:, [1]], tcol], dim=1)
        xt = torch.cat([zyx[:, [2]], tcol], dim=1)
        return (
            self.enc_zy(zy).float(),
            self.enc_zx(zx).float(),
            self.enc_yx(yx).float(),
            self.enc_zt(zt).float(),
            self.enc_yt(yt).float(),
            self.enc_xt(xt).float(),
        )


class SexCubesTransformer(nn.Module):
    """SexCubes encoder + Transformer decoder.

    The six 2D encoder feature vectors are treated as a sequence of 6 tokens.
    Self-attention lets each token attend to all others (capturing cross-pair
    interactions that a plain MLP must learn implicitly), then tokens are
    mean-pooled and projected to a scalar.

    Inherits all the cuBLASLt workarounds from QuadCubesTransformer:
      - feat_dim padded to multiple of 8 (K alignment)
      - batch dimension padded to multiple of 8 before token_proj (N alignment)
      - manual matmul in _SelfAttnBlock to avoid flash-attn grid-size limits

    Config:
        encoding_config : HashEncoderConfig  (same as SexCubes)
        decoder_config  : TransformerDecoderConfig
            d_model  – token projection dimension
            n_heads  – number of attention heads (must divide d_model)
            n_layers – number of _SelfAttnBlock layers
            dropout  – dropout probability
    """

    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        decoder_config: TransformerDecoderConfig,
    ):
        super().__init__()
        self.encoder = _SexEncoder(encoding_config)

        feat_dim_raw = self.encoder.n_output_dims
        d_model = decoder_config.d_model
        n_heads = decoder_config.n_heads
        n_layers = decoder_config.n_layers
        dropout = decoder_config.dropout

        self._feat_dim_raw = feat_dim_raw
        self._feat_dim = math.ceil(feat_dim_raw / 8) * 8
        self.token_proj = nn.Linear(self._feat_dim, d_model)

        # 6 learnable positional embeddings, one per pair
        self.pos_embed = nn.Parameter(torch.zeros(6, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [_SelfAttnBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        self.out_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, zyx, t):
        B = zyx.size(0)
        feats = self.encoder(zyx, t)  # tuple of 6 × [B, D]

        # [B, 6, feat_dim_raw] — pad K to multiple of 8 if needed
        tokens = torch.stack(list(feats), dim=1)
        if self._feat_dim > self._feat_dim_raw:
            tokens = F.pad(tokens, (0, self._feat_dim - self._feat_dim_raw))

        # Reshape to 2D; pad batch dim to multiple of 8 for cuBLASLt FP16
        tokens_2d = tokens.contiguous().reshape(B * 6, -1)
        N = tokens_2d.size(0)
        pad = (-N) % 8
        if pad:
            tokens_2d = F.pad(tokens_2d, (0, 0, 0, pad))
        tokens_2d = self.token_proj(tokens_2d)
        if pad:
            tokens_2d = tokens_2d[:N]
        tokens = tokens_2d.reshape(B, 6, -1)
        tokens = tokens + self.pos_embed.unsqueeze(0)  # [B, 6, d_model]

        for block in self.blocks:
            tokens = block(tokens)

        # Mean pool → [B, d_model] → [B, 1]
        return self.out_head(tokens.mean(dim=1))


class SexCubesUNet(nn.Module):
    """SexCubes encoder + U-Net style decoder.

    Each of the 6 pairwise 2D encoders produces multi-resolution features.
    The UNet splits each encoder's output into coarse/medium/fine scale groups
    and processes them with skip connections, so coarse-scale structure from
    all 6 pairs is merged first, then medium, then fine detail.

    This directly addresses the disconnection problem: instead of expecting
    a single MLP to bridge all 6 feature streams, the down-path gradually
    fuses them scale by scale.

    Config:
        encoding_config : HashEncoderConfig  (same as SexCubes)
        decoder_config  : UNetDecoderConfig
            hidden_dims   – list of 3 integers [d1, d2, d3]
            levels_coarse – how many hash levels go into the coarse group
            levels_medium – how many hash levels go into the medium group
                            remaining levels form the fine group
    """

    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        decoder_config: UNetDecoderConfig,
    ):
        super().__init__()
        self.encoder = _SexEncoder(encoding_config)

        nfpl = encoding_config.n_features_per_level
        n_levels = encoding_config.n_levels
        lc = decoder_config.levels_coarse
        lm = decoder_config.levels_medium
        lf = n_levels - lc - lm
        if lf <= 0:
            raise ValueError(
                f"levels_coarse ({lc}) + levels_medium ({lm}) must be < n_levels ({n_levels})"
            )
        self._lc = lc
        self._lm = lm
        self._nfpl = nfpl

        # 6 encoders × levels × features per level
        coarse_dim = 6 * lc * nfpl
        medium_dim = 6 * lm * nfpl
        fine_dim   = 6 * lf * nfpl

        d1, d2, d3 = decoder_config.hidden_dims

        # ── Down-path ────────────────────────────────────────────────────────
        self.down1 = nn.Sequential(nn.Linear(coarse_dim, d1), nn.ReLU())
        self.down2 = nn.Sequential(nn.Linear(medium_dim + d1, d2), nn.ReLU())
        self.bottleneck = nn.Sequential(nn.Linear(fine_dim + d2, d3), nn.ReLU())

        # ── Up-path (with skip connections from down-path) ────────────────────
        self.up2 = nn.Sequential(nn.Linear(d3 + d2, d2), nn.ReLU())
        self.up1 = nn.Sequential(nn.Linear(d2 + d1, d1), nn.ReLU())

        self.out_head = nn.Linear(d1, 1)

    def _split_levels(self, feat):
        """Split a single encoder's output into (coarse, medium, fine) groups."""
        c = self._lc * self._nfpl
        m = self._lm * self._nfpl
        return feat[:, :c], feat[:, c : c + m], feat[:, c + m :]

    def forward(self, zyx, t):
        f_zy, f_zx, f_yx, f_zt, f_yt, f_xt = self.encoder(zyx, t)

        # Split each encoder's features by resolution scale
        c0, m0, f0 = self._split_levels(f_zy)
        c1, m1, f1 = self._split_levels(f_zx)
        c2, m2, f2 = self._split_levels(f_yx)
        c3, m3, f3 = self._split_levels(f_zt)
        c4, m4, f4 = self._split_levels(f_yt)
        c5, m5, f5 = self._split_levels(f_xt)

        # Concatenate across all 6 encoders at each scale
        coarse = torch.cat([c0, c1, c2, c3, c4, c5], dim=-1)
        medium = torch.cat([m0, m1, m2, m3, m4, m5], dim=-1)
        fine   = torch.cat([f0, f1, f2, f3, f4, f5], dim=-1)

        # Down-path
        e1 = self.down1(coarse)
        e2 = self.down2(torch.cat([medium, e1], dim=-1))
        e3 = self.bottleneck(torch.cat([fine, e2], dim=-1))

        # Up-path with skip connections
        d2 = self.up2(torch.cat([e3, e2], dim=-1))
        d1 = self.up1(torch.cat([d2, e1], dim=-1))

        return self.out_head(d1)

# ─────────────────────────────────────────────────────────────────────────────
# CombinedCubesKPlanes / MixedCubesKPlanes
#
# CombinedCubes factorisation: 3 temporal 2D planes (zt,yt,xt) + 1 spatial 3D
# encoder (zyx).  For 4D selectivity we form three pairwise products:
#   f_zyx * f_xt  →  z, y, x, t  (x couples spatial and temporal)
#   f_zyx * f_yt  →  z, y, x, t  (y couples)
#   f_zyx * f_zt  →  z, y, x, t  (z couples)
# Each product is non-zero only when both the spatial and the specific
# spatial-temporal plane agree, giving the MLP explicit 4D selectivity.
#
# CombinedCubesKPlanes uses the same HashEncoderConfig for all four encoders
# so D_2d == D_3d and the element-wise products are legal.
#
# MixedCubesKPlanes uses DenseGridEncoderConfig for the 2D temporal planes so
# D_2d may differ from D_3d.  Instead we use the triple temporal product
# f_xt * f_yt * f_zt which lives entirely in D_2d-dimensional space.
# ─────────────────────────────────────────────────────────────────────────────


class _CombinedEncoder(nn.Module):
    """Three 2D hash-encoders (zt, yt, xt) + one 3D hash-encoder (zyx).

    All four encoders share the same HashEncoderConfig so their output
    dimension D is identical, enabling element-wise products downstream.
    """

    def __init__(self, encoding_config: "HashEncoderConfig"):
        super().__init__()
        enc2d = encoding_config.get_encoder_config_2D()
        enc3d = encoding_config.get_encoder_config()
        self.enc_zt  = tcnn.Encoding(2, enc2d)
        self.enc_yt  = tcnn.Encoding(2, enc2d)
        self.enc_xt  = tcnn.Encoding(2, enc2d)
        self.enc_zyx = tcnn.Encoding(3, enc3d)
        self.n_output_dims_2d = self.enc_zt.n_output_dims
        self.n_output_dims_3d = self.enc_zyx.n_output_dims

    def forward(self, zyx, t):
        tcol = torch.full((zyx.size(0), 1), float(t), device=zyx.device, dtype=zyx.dtype)
        zt = torch.cat([zyx[:, [0]], tcol], dim=1)
        yt = torch.cat([zyx[:, [1]], tcol], dim=1)
        xt = torch.cat([zyx[:, [2]], tcol], dim=1)
        return (
            self.enc_zt(zt).float(),
            self.enc_yt(yt).float(),
            self.enc_xt(xt).float(),
            self.enc_zyx(zyx).float(),
        )


class CombinedCubesKPlanes(nn.Module):
    """CombinedCubes encoder + K-Planes product decoder.

    Uses the same four encoders as CombinedCubes (3x 2D hash + 1x 3D hash),
    all sharing one HashEncoderConfig so D_2d == D_3d == D.

    MLP input: cat([f_zt, f_yt, f_xt, f_zyx,
                    f_zyx*f_xt, f_zyx*f_yt, f_zyx*f_zt])  ->  7 x D features.

    Each pairwise product couples the full spatial feature with one
    spatial-temporal plane, giving the MLP explicit 4D selectivity.
    """

    def __init__(self, encoding_config: "HashEncoderConfig", network_config: "MLPNetConfig"):
        super().__init__()
        self.encoder = _CombinedEncoder(encoding_config)
        D = self.encoder.n_output_dims_2d  # == n_output_dims_3d (same config)
        in_dim = 7 * D  # 4 raw + 3 pairwise products
        self.mlp = _build_mlp(in_dim, network_config)

    def forward(self, zyx, t):
        f_zt, f_yt, f_xt, f_zyx = self.encoder(zyx, t)

        p_x = f_zyx * f_xt   # spatial x (x, t)
        p_y = f_zyx * f_yt   # spatial x (y, t)
        p_z = f_zyx * f_zt   # spatial x (z, t)

        h = torch.cat([f_zt, f_yt, f_xt, f_zyx, p_x, p_y, p_z], dim=-1)
        return self.mlp(h).float()


class _MixedEncoder(nn.Module):
    """Three 2D dense-grid encoders (zt, yt, xt) + one 3D hash-encoder (zyx).

    Mirrors MixedCubes: collision-free dense grids for temporal planes,
    hash grid for spatial.  D_2d and D_3d may differ.
    """

    def __init__(self, encoding_config: "HashEncoderConfig", encoding_config_2d: "DenseGridEncoderConfig"):
        super().__init__()
        enc2d = encoding_config_2d.get_encoder_config()
        enc3d = encoding_config.get_encoder_config()
        self.enc_zt  = tcnn.Encoding(2, enc2d)
        self.enc_yt  = tcnn.Encoding(2, enc2d)
        self.enc_xt  = tcnn.Encoding(2, enc2d)
        self.enc_zyx = tcnn.Encoding(3, enc3d)
        self.n_output_dims_2d = self.enc_zt.n_output_dims
        self.n_output_dims_3d = self.enc_zyx.n_output_dims

    def forward(self, zyx, t):
        tcol = torch.full((zyx.size(0), 1), float(t), device=zyx.device, dtype=zyx.dtype)
        zt = torch.cat([zyx[:, [0]], tcol], dim=1)
        yt = torch.cat([zyx[:, [1]], tcol], dim=1)
        xt = torch.cat([zyx[:, [2]], tcol], dim=1)
        return (
            self.enc_zt(zt).float(),
            self.enc_yt(yt).float(),
            self.enc_xt(xt).float(),
            self.enc_zyx(zyx).float(),
        )


class MixedCubesKPlanes(nn.Module):
    """MixedCubes encoder + K-Planes product decoder.

    Uses the same encoders as MixedCubes (3x dense 2D + 1x hash 3D).
    Because D_2d (dense) may differ from D_3d (hash), pairwise cross-type
    products are not used.  Instead the triple temporal product captures
    full 4D selectivity within the dense-grid feature space.

    MLP input: cat([f_zt, f_yt, f_xt, f_zyx,
                    f_xt*f_yt*f_zt])  ->  4*D_2d + D_3d features.

    The triple product is non-zero only when all three temporal planes agree,
    making it strongly selective in (z,y,x,t) space.
    """

    def __init__(
        self,
        encoding_config: "HashEncoderConfig",
        encoding_config_2d: "DenseGridEncoderConfig",
        network_config: "MLPNetConfig",
    ):
        super().__init__()
        self.encoder = _MixedEncoder(encoding_config, encoding_config_2d)
        D2 = self.encoder.n_output_dims_2d
        D3 = self.encoder.n_output_dims_3d
        in_dim = 4 * D2 + D3  # 3 raw temporal + triple product + spatial
        self.mlp = _build_mlp(in_dim, network_config)

    def forward(self, zyx, t):
        f_zt, f_yt, f_xt, f_zyx = self.encoder(zyx, t)

        temporal_product = f_xt * f_yt * f_zt   # (x,t) x (y,t) x (z,t) -> full 4D

        h = torch.cat([f_zt, f_yt, f_xt, f_zyx, temporal_product], dim=-1)
        return self.mlp(h).float()
