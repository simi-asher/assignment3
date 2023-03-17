import torch
import torch.nn.functional as F

from ray_utils import RayBundle
from typing import Tuple


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.reshape(-1, 3)

        return torch.linalg.norm(
            sample_points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.reshape(-1, 3)
        diff = torch.abs(sample_points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)


sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.reshape(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", frequencies * omega0, persistent=False)
        self.append_input = include_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = (x[..., None] * self._frequencies).reshape(*x.shape[:-1], -1)
        embed = torch.cat(
            (embed.sin(), embed.cos(), x)
            if self.append_input
            else (embed.sin(), embed.cos()),
            dim=-1,
        )
        return embed


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)

class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips: Tuple[int, ...] = (),
    ):
        super().__init__()
        layers = []
        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim
            linear = torch.nn.Linear(dimin, dimout)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x
        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)
        return y

def _xavier_init(linear):
    torch.nn.init.xavier_uniform_(linear.weight.data)

# TODO (3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.harmonic_embedding_xyz = HarmonicEmbedding(cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(cfg.n_harmonic_functions_dir)
        embedding_dim_xyz = cfg.n_harmonic_functions_xyz * 2 * 3 + 3
        embedding_dim_dir = cfg.n_harmonic_functions_dir * 2 * 3 + 3

        self.mlp_xyz = MLPWithInputSkips(
            cfg.n_layers_xyz,
            embedding_dim_xyz,
            cfg.n_hidden_neurons_xyz,
            embedding_dim_xyz,
            cfg.n_hidden_neurons_xyz,
            input_skips=cfg.append_xyz,
        )

        self.intermediate_linear = torch.nn.Linear(
            cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz
        )
        _xavier_init(self.intermediate_linear)

        self.density_layer = torch.nn.Linear(cfg.n_hidden_neurons_xyz, 1)
        _xavier_init(self.density_layer)
        self.density_layer.bias.data[:] = 0.0

        self.color_layer = torch.nn.Sequential(
            LinearWithRepeat(
                cfg.n_hidden_neurons_xyz + embedding_dim_dir, cfg.n_hidden_neurons_dir
            ),
            torch.nn.ReLU(True),
            torch.nn.Linear(cfg.n_hidden_neurons_dir, 3),
            torch.nn.Sigmoid(),
        )
        # NOTES:
        # Use ray bundles dir values into embedding and 
        # same embedding from mlp, concat embedding to o/p and pass to rgb layer
        # num rays * num samples/rays * hidden_layer_dim
        # RGB layer last 3
        # density layer to 1

    def get_densities(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        raw_densities = self.density_layer(features)
        densities = torch.relu(raw_densities)
        return densities

    def get_colors(
        self, features: torch.Tensor, rays_directions: torch.Tensor
    ) -> torch.Tensor:
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1)
        rays_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        return self.color_layer((self.intermediate_linear(features), rays_embedding))

    def get_densities_and_colors(
        self, features: torch.Tensor, ray_bundle: RayBundle
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rays_densities = self.get_densities(
            features
        )
        rays_colors = self.get_colors(features, ray_bundle.directions)
        return rays_densities, rays_colors

    def forward(
        self,
        ray_bundle: RayBundle,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds_xyz = self.harmonic_embedding_xyz(ray_bundle.sample_points)
        features = self.mlp_xyz(embeds_xyz, embeds_xyz)
        rays_densities, rays_colors = self.get_densities_and_colors(
            features, ray_bundle
        )
        out = {
            'density': rays_densities,
            'feature': rays_colors,
        }
        return out

        

volume_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
}