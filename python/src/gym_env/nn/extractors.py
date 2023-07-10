import gymnasium as gym
import torch
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MusiciansCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=features_dim)

        extractors = {}

        num_musicians = 0
        num_attendee = 0
        for key, subspace in observation_space.spaces.items():
            if key == 'musicians_placed':
                num_musicians = subspace.n
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(subspace.n, 8, bias=False)
                )
            # elif key == 'musician_instruments':
            #     num_musicians = subspace.shape[0]
            #     extractors[key] = nn.Sequential(
            #         nn.BatchNorm1d(num_musicians),
            #         nn.Linear(num_musicians, 32, bias=False)
            #     )
            elif key == 'attendee_happiness':
                num_attendee = subspace.shape[0]
                extractors[key] = nn.Sequential(
                    nn.BatchNorm1d(num_attendee),
                    nn.Linear(num_attendee, num_attendee, bias=True)
                )
            else:
                extractors[key] = nn.Identity()

        self.extractors = nn.ModuleDict(extractors)
        self.musicians_line = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(num_musicians * 2),
            nn.Dropout(p=0.2),
            nn.Linear(num_musicians * 2, num_musicians, bias=False),
            nn.BatchNorm1d(num_musicians),
            nn.PReLU(num_musicians)
        )
        # self.attendee_line = nn.Sequential(
        #     nn.Flatten(),
        #     nn.BatchNorm1d(num_attendee * 2),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(num_attendee * 2, num_attendee, bias=False),
        #     nn.BatchNorm1d(num_attendee),
        #     nn.PReLU(num_attendee)
        # )
        # self.cross_line_musicians = nn.Sequential(
        #     nn.BatchNorm1d(num_musicians),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(num_musicians, 128, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.PReLU(128),
        #     nn.Linear(128, 32),
        #     nn.PReLU(32)
        # )
        # self.cross_line_attendee = nn.Sequential(
        #     nn.BatchNorm1d(num_attendee),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(num_attendee, 128, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.PReLU(128),
        #     nn.Linear(128, 32),
        #     nn.PReLU(32)
        # )
        num_to_merge = 8 + num_attendee * 3
        self.merging = nn.Sequential(
            nn.BatchNorm1d(num_to_merge),
            nn.Dropout(p=0.2),
            nn.Linear(num_to_merge, num_to_merge, bias=False),
            nn.BatchNorm1d(num_to_merge),
            nn.PReLU(num_to_merge),
            nn.Dropout(p=0.2),
            nn.Linear(num_to_merge, num_to_merge, bias=False),
            nn.BatchNorm1d(num_to_merge),
            nn.PReLU(num_to_merge),
            nn.Linear(num_to_merge, features_dim),
            nn.PReLU(features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        # self.extractors contain nn.Modules that do all the processing.
        placed_enc = None
        placed = None
        # instr_enc = None
        mus_pos = None
        att_pos = None
        att_hap = None
        for key, extractor in self.extractors.items():
            if key == 'musicians_placed':
                placed = observations[key]
                placed_enc = extractor(observations[key])
            # elif key == 'musician_instruments':
            #     instr_enc = extractor(observations[key])
            elif key == 'attendee_happiness':
                att_hap = extractor(observations[key])
            elif key == 'musician_placements':
                mus_pos = observations[key] / 1e5
            elif key == 'attendee_placements':
                att_pos = observations[key] / 1e5
        mus_enc = self.musicians_line(mus_pos)
        # att_enc = self.attendee_line(att_pos)
        dist = mus_pos @ att_pos.transpose(-2, -1) \
               + (mus_pos ** 2).sum(-1, keepdims=True) \
               + (att_pos ** 2).sum(-1, keepdims=True).transpose(-2, -1)
        # sm = torch.softmax(dist, dim=-1)
        if len(placed.shape) > 2:
            placed = placed.squeeze(1)
        pos_dist = dist[placed == 1]
        # mus_att = torch.matmul(sm, att_enc.unsqueeze(2)).squeeze(2)
        # mus_att_hap = torch.matmul(sm, att_hap.unsqueeze(2)).squeeze(2)
        att_att = torch.matmul(torch.softmax(dist.transpose(-2, -1), dim=-1), mus_enc.unsqueeze(2)).squeeze(2)
        # cross_mus = self.cross_line_musicians(mus_att)
        # cross_att = self.cross_line_attendee(att_att)
        encodings = torch.cat([
            placed_enc,
            # instr_enc,
            # mus_enc,
            # att_enc,
            att_hap,
            # mus_att,
            # mus_att_hap,
            att_att,
            # cross_mus,
            # cross_att,
            pos_dist,
        ], dim=1)
        merged = self.merging(encodings)
        return merged
