r"""
MIL model from `"Obtaining Spatially Resolved Tumor Purity Maps Using Deep Multiple Instance Learning In A Pan-cancer
Study" <https://www.biorxiv.org/content/10.1101/2021.07.08.451443v1.full.pdf>`
[Source Code]: <https://github.com/onermustafaumit/SRTPMs.git>
"""

import math

import torch
import torch.nn as nn

from resnet import resnet18


class DistributionPoolingFilter(nn.Module):
    __constants__ = ['num_bins', 'sigma']

    def __init__(self, num_bins=21, sigma=0.05):
        super(DistributionPoolingFilter, self).__init__()

        self.num_bins = num_bins
        self.sigma = sigma
        self.alfa = 1 / math.sqrt(2 * math.pi * (sigma ** 2))
        self.beta = -1 / (2 * (sigma ** 2))

        sample_points = torch.linspace(0, 1, steps=num_bins, dtype=torch.float32, requires_grad=False)
        self.register_buffer('sample_points', sample_points)

    def extra_repr(self):
        return 'num_bins={}, sigma={}'.format(
            self.num_bins, self.sigma
        )

    def forward(self, data):
        batch_size, num_instances, num_features = data.size()

        sample_points = self.sample_points.repeat(batch_size, num_instances, num_features, 1)
        # (batch_size,num_instances,num_features,num_bins)

        data = torch.reshape(data, (batch_size, num_instances, num_features, 1))
        # (batch_size,num_instances,num_features,1)

        diff = sample_points - data.repeat(1, 1, 1, self.num_bins)
        diff_2 = diff ** 2
        # (batch_size,num_instances,num_features,num_bins)

        result = self.alfa * torch.exp(self.beta * diff_2)
        # (batch_size,num_instances,num_features,num_bins)

        out_unnormalized = torch.sum(result, dim=1)
        # (batch_size,num_features,num_bins)

        norm_coeff = torch.sum(out_unnormalized, dim=2, keepdim=True)
        # (batch_size,num_features,num_bins)

        out = out_unnormalized / norm_coeff
        # (batch_size,num_features,num_bins)

        return out


class FeatureExtractor(nn.Module):

    def __init__(self, num_features=32):
        super(FeatureExtractor, self).__init__()

        self._model_conv = resnet18()

        num_ftrs = self._model_conv.fc.in_features
        self._model_conv.fc = nn.Linear(num_ftrs, num_features)
        # print("resnet18: {}".format(self._model_conv))

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self._model_conv(x)
        out = self.relu(out)

        return out


class RepresentationTransformation(nn.Module):
    def __init__(self, num_features=32, num_bins=11, num_classes=10):
        super(RepresentationTransformation, self).__init__()

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features * num_bins, 384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        out = self.fc(x)

        return out


class Model(nn.Module):

    def __init__(self, num_classes=1, num_instances=100, num_features=32,
                 num_bins=11, sigma=0.05):
        super(Model, self).__init__()
        self._num_classes = num_classes
        self._num_instances = num_instances
        self._num_features = num_features
        self._num_bins = num_bins
        self._sigma = sigma

        self._feature_extractor = FeatureExtractor(num_features=num_features)

        self._mil_pooling_filter = DistributionPoolingFilter(num_bins=num_bins, sigma=sigma)

        self._representation_transformation = RepresentationTransformation(num_features=num_features,
                                                                           num_bins=num_bins,
                                                                           num_classes=num_classes)

    def forward(self, x):

        extracted_features = self._feature_extractor(x)
        extracted_features = torch.reshape(extracted_features, (-1, self._num_instances, self._num_features))

        out = self._mil_pooling_filter(extracted_features)
        out = torch.flatten(out, 1)

        out = self._representation_transformation(out)

        return out
