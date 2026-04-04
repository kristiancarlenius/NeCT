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
    ):
        super().__init__()
        encoding = {
            "otype": "Composite",
            "nested": [
                {
                    "n_dims_to_encode": 2,
                    **encoding_config.get_encoder_config_2D() #x, t
                },
                {
                    "n_dims_to_encode": 2,
                    **encoding_config.get_encoder_config_2D() #y, t
                },
                {
                    "n_dims_to_encode": 2,
                    **encoding_config.get_encoder_config_2D() #z, t
                },
                {
                    "n_dims_to_encode": 3,
                    **encoding_config.get_encoder_config() #x, y, z
                }
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

        # ── Self-attention (pre-LN) ──────────────────────────────────────────
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, S, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)          # each [B, H, S, Dh]

        # [B, H, S, S]  — S=4, so this is always a tiny 4×4 matrix
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)      # [B, H, S, Dh]
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        x = x + self.out_proj(out)

        # ── Feed-forward (pre-LN) ────────────────────────────────────────────
        x = x + self.ff(self.norm2(x))
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
        f_zyx, f_yxt, f_xzt, f_zyt = self.encoder(zyx, t)

        # [B, 4, feat_dim_raw] — pad K to multiple of 8 if needed
        tokens = torch.stack([f_zyx, f_yxt, f_xzt, f_zyt], dim=1)
        if self._feat_dim > self._feat_dim_raw:
            tokens = F.pad(tokens, (0, self._feat_dim - self._feat_dim_raw))

        # → [B, 4, d_model]
        tokens = self.token_proj(tokens) + self.pos_embed.unsqueeze(0)

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