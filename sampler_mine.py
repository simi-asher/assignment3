import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray, device=ray_bundle.origins.device)

        print('Z vals',z_vals.shape) # [n_pts_per_ray, 1]
        print('Ray origins',ray_bundle.origins.shape) # [pixels, 3]
        print('Ray directions',ray_bundle.directions.shape) # [pixels, 3]
        # [pixels , n_pts_per_ray, 3]
        # origins = ray_bundle.origins.expand(self.n_pts_per_ray, -1, -1)
        # print('Origins',origins.shape)
        # dirs = z_vals * ray_bundle.directions.expand(self.n_pts_per_ray, -1, -1)
        # print('Dirs',dirs.shape)
        # print(ray_bundle.directions.expand(-1, -1, self.n_pts_per_ray, -1))
        # TODO (1.4): Sample points from z values
        sample_points = ray_bundle.origins.unsqueeze(2) + z_vals * ray_bundle.directions.unsqueeze(2)
        sample_points = sample_points.permute(0, 2, 1)
        print('Sample points',sample_points.shape)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}