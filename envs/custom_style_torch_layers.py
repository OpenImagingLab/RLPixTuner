from typing import Dict, List, Tuple, Type, Union

import gymnasium as gym
import torch as th
from gymnasium import spaces
import torch
from torch import nn
import torch.nn.functional as F

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
from config import cfg
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp, MlpExtractor, FlattenExtractor, get_actor_critic_arch
import VGG
from tools.hist_loss import batch_histogram, batch_histograms1

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class Biliteral_Grid(nn.Module):
    def __init__(self):
        super(Biliteral_Grid, self).__init__()
        self.SB1 = SplattingBlock(64,8,128) # 32 is not real
        self.SB2 = SplattingBlock(8, 16,256)
        self.SB3 = SplattingBlock(16, 32,512)
        self.conv1 = ConvLayer(32, 64,3, 2)
        self.conv2 = ConvLayer(64, 64, 3, 1)

        # local feature
        self.L1 = ConvLayer(64, 64, 3, 1)
        self.L2 = ConvLayer(64, 64, 3, 1)

        # global feature
        self.G1 = ConvLayer(64, 64, 3, 2)
        self.G2 = ConvLayer(64, 64, 3, 2)
        self.G3 = nn.Linear(1024,256)
        self.G4 = nn.Linear(256,128)
        self.G5 = nn.Linear(128,64)
        self.G6 = nn.Linear(64,64)
        self.F = ConvLayer(128, 64, 1, 1)
        self.T = ConvLayer(64, 96, 3, 1)
        return
    def forward(self,c,s,feat):
        c,s = self.SB1(c,s,feat[0])
        c, s = self.SB2(c, s, feat[1])
        c, s = self.SB3(c, s, feat[2])

        c = F.relu(self.conv1(c))
        c = F.relu(self.conv2(c))

        # local feature
        L = F.relu(self.L1(c))
        L = F.relu(self.L2(L))

        # global feature
        G = F.relu(self.G1(c))
        G = F.relu(self.G2(G))
        G = G.reshape((G.shape[0],-1))
        G = F.relu(self.G3(G))
        G = F.relu(self.G4(G))
        G = F.relu(self.G5(G))
        G = F.relu(self.G6(G))

        G = G.reshape(G.shape+(1,1)).expand(G.shape+(16,16))
        f = torch.cat((L,G),dim=1)
        f = F.relu(self.F(f))
        f = self.T(f)
        # this is grid
        return f
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Biliteral_Grid = Biliteral_Grid()

    def forward(self,cont,cont_feat,style_feat):
        feat = []
        for i in range(1,len(cont_feat)):
            feat.append(adaptive_instance_normalization(cont_feat[i],style_feat[i]))

        coeffs_out = self.Biliteral_Grid(cont_feat[0],style_feat[0],feat)
        coeffs = coeffs_out.reshape(coeffs_out.shape[0],12,-1,coeffs_out.shape[-2],coeffs_out.shape[-1])
        return coeffs_out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2  # same dimension after padding
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)  # remember this dimension

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class SplattingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut_channel):
        super(SplattingBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels,out_channels,kernel_size=3,stride=2)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.conv_short = nn.Conv2d(shortcut_channel, out_channels, 1, 1)
        return

    def forward(self, c, s, shortcut):
        c = F.relu(self.conv1(c))
        s = F.relu(self.conv1(s))
        c = adaptive_instance_normalization(c, s)
        shortcut = self.conv_short(shortcut)
        c += shortcut
        c = F.relu(self.conv2(c))
        return c, s


class SplattingBlock_1(nn.Module):
    def __init__(self, in_channels, out_channels, top_path_channel):
        super(SplattingBlock_1, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=2)
        self.conv2 = ConvLayer(out_channels + top_path_channel, out_channels, kernel_size=3, stride=1)

    def forward(self, c, s, top_path):
        c = F.relu(self.conv1(c))
        s = F.relu(self.conv1(s))
        c = adaptive_instance_normalization(c, s)
        c = torch.cat([c, top_path], dim=1)
        c = F.relu(self.conv2(c))
        return c, s


class SplattingBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut_channel):
        super(SplattingBlock_2, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=2)
        self.conv_short = nn.Conv2d(shortcut_channel, out_channels, 1, 1)
        return

    def forward(self, c, s, shortcut):
        c = F.relu(self.conv1(c))
        s = F.relu(self.conv1(s))
        c = adaptive_instance_normalization(c, s)
        shortcut = self.conv_short(shortcut)
        c += shortcut
        return c, s


class StyleSplatEncoder(nn.Module):
    def __init__(self, features_dim: int = 256, splatting_option: int = 0):
        super(StyleSplatEncoder, self).__init__()
        splatting_options = [SplattingBlock, SplattingBlock_1, SplattingBlock_2]
        Splatting = splatting_options[splatting_option]
        self.SB1 = Splatting(64, 8, 128)
        self.SB2 = Splatting(8, 16, 256)
        self.SB3 = Splatting(16, 32, 512)  # 8x8x32
        self.conv1 = ConvLayer(32, 64, 3, 2)  # 4x4x64
        self.conv2 = ConvLayer(64, 64, 3, 1)  # 4x4x64
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(1024, features_dim), nn.ReLU())

    def forward(self, cont_feat, style_feat):
        c = cont_feat[0]
        s = style_feat[0]
        feat = []
        for i in range(1, len(cont_feat)):
            feat.append(adaptive_instance_normalization(cont_feat[i], style_feat[i]))
        c, s = self.SB1(c, s, feat[0])
        c, s = self.SB2(c, s, feat[1])
        c, s = self.SB3(c, s, feat[2])
        c = F.relu(self.conv1(c))
        c = F.relu(self.conv2(c))
        return self.linear(c)


class StyleEncoder(nn.Module):
    def __init__(self, features_dim: int = 256, vgg_option: int = 3):
        super(StyleEncoder, self).__init__()
        self.vgg_option = vgg_option
        if vgg_option == 3:
            self.conv1 = ConvLayer(256, 256, 3, 2)
            self.conv2 = ConvLayer(256, 256, 3, 2)
            self.conv3 = ConvLayer(256, 64, 3, 1)
        elif vgg_option == 4:
            self.conv1 = ConvLayer(512, 256, 3, 2)
            self.conv2 = ConvLayer(256, 64, 3, 1)
        else:
            raise NotImplementedError("vgg layer not supported")

        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(1024, features_dim), nn.ReLU())

    def forward(self, cont_feat, style_feat):
        """
        vgg feats - img-res=64:
        conv1_1 64x64 64
        conv2_1 32x32 128
        conv3_1 16x16 256: 8x8x256 4x4x256 4x4x64 256
        conv4_1 8x8 512: 4x4x256 4x4x64 256
        -> self.vgg option 1 2 3 4
        -> we can have 3_1 and 4_1
        """
        c = cont_feat[self.vgg_option - 1]
        s = style_feat[self.vgg_option - 1]
        c = adaptive_instance_normalization(c, s)
        if self.vgg_option == 3:
            c = F.relu(self.conv1(c))
            c = F.relu(self.conv2(c))
            c = F.relu(self.conv3(c))
        elif self.vgg_option == 4:
            c = F.relu(self.conv1(c))
            c = F.relu(self.conv2(c))
        return self.linear(c)


class NatureCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.cnt = 0

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class ExposureCNN(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 512,
            normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(32), 32 32 32
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(64), 64 16 16
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(256), 256 8 8
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(256), 256 4 4
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.cnt = 0

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # a = self.cnn(observations)
        # print(a.shape)
        # exit()
        return self.linear(self.cnn(observations))


class StyleExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        style_feat_dim: int = 256,
        splatting_option: int = -1,
        vgg_option: int = 3,
        concat_hist_type: str = 'None',
        concat_exp_cnn: bool = False,
        normalized_image: bool = False,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)
        '''
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=1, shape=(obs_img_dim, obs_img_shape[1], obs_img_shape[2]), dtype=np.float32),
            'vector': spaces.Box(low=-1, high=1, shape=(obs_vec_dim,), dtype=np.float32),
        })
        '''
        vgg = VGG.vgg
        vgg.load_state_dict(torch.load('envs/checkpoints/vgg_normalised.pth'))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        self.vgg_net = VGG.Net(vgg).to('cuda')
        total_concat_size = 0
        if splatting_option == -1:
            assert vgg_option in [3, 4]
            self.style_encoder = StyleEncoder(features_dim=style_feat_dim, vgg_option=vgg_option)
        else:
            self.style_encoder = StyleSplatEncoder(features_dim=style_feat_dim, splatting_option=splatting_option)
        self.style_encoder = self.style_encoder.to('cuda')

        self.concat_exp_cnn = concat_exp_cnn
        if concat_exp_cnn:
            self.image_encoder = ExposureCNN(observation_space.spaces['image'], features_dim=cnn_output_dim, normalized_image=normalized_image)
            total_concat_size += cnn_output_dim

        self.vector_encoder = nn.Flatten()

        self.concat_hist = concat_hist_type in ['yuv', 'rgb']
        if self.concat_hist:
            self.hist_bins = 64
            self.hist_dim = 128
            self.hist_encoder = nn.Linear(self.hist_bins * 6, self.hist_dim)
            total_concat_size += self.hist_dim
            self.hist_type = concat_hist_type

        total_concat_size += style_feat_dim
        total_concat_size += get_flattened_obs_dim(observation_space.spaces['vector'])
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        """
        for style feature encoder options:
        - one vgg feat + adaIN
        - multi-scale vgg + splatting + adaIN
        - old twin img encoder
        """
        encoded_tensor_list = []
        encoded_tensor_list.append(self.vector_encoder(observations["vector"]))
        if self.concat_exp_cnn:
            encoded_tensor_list.append(self.image_encoder(observations["image"]))

        img_input = observations["image"][:, 0:3, :, :]
        img_target = observations["image"][:, 3:6, :, :]
        cont_feat = self.vgg_net.encode_with_intermediate(img_input)
        style_feat = self.vgg_net.encode_with_intermediate(img_target)
        encoded_tensor_list.append(self.style_encoder(cont_feat, style_feat))

        if self.concat_hist:
            yuv = (self.hist_type == 'yuv')
            hists_input = batch_histograms1(img_input, self.hist_bins, yuv)
            hists_target = batch_histograms1(img_target, self.hist_bins, yuv)
            encoded_tensor_list.append(self.hist_encoder(torch.cat([hists_input, hists_target], dim=1)))

        return th.cat(encoded_tensor_list, dim=1)


class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        concat_hist: bool = False,
        hist_type: str = "rgb",
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                if cfg.use_exposure_cnn:
                    extractors[key] = ExposureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                else:
                    extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)
        self.concat_hist = concat_hist
        if concat_hist:
            self.hist_bins = 64
            self.hist_dim = 128
            self.hist_encoder = nn.Linear(self.hist_bins * 6, self.hist_dim)
            total_concat_size += self.hist_dim
            self.hist_type = hist_type

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        if self.concat_hist:
            yuv = (self.hist_type == 'yuv')
            imgs_input = observations["image"][:, 0:3, :, :]
            imgs_target = observations["image"][:, 3:6, :, :]
            hists_input = batch_histograms1(imgs_input, self.hist_bins, yuv)
            hists_target = batch_histograms1(imgs_target, self.hist_bins, yuv)
            # print('img', imgs_input.shape)
            # print('hist', hists_input.shape)
            encoded_tensor_list.append(self.hist_encoder(torch.cat([hists_input, hists_target], dim=1)))

        return th.cat(encoded_tensor_list, dim=1)


class StyleExtractor_test(nn.Module):
    def __init__(
        self,
        cnn_output_dim: int = 256,
        style_feat_dim: int = 256,
        splatting_option: int = 2,
        normalized_image: bool = False,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__()
        vgg = VGG.vgg
        vgg.load_state_dict(torch.load('envs/checkpoints/vgg_normalised.pth'))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        self.vgg_net = VGG.Net(vgg).to('cuda')
        self.StyleEncoder = StyleSplatEncoder(features_dim=style_feat_dim, splatting_option=splatting_option)

        #self.image_encoder = ExposureCNN(observation_space.spaces['image'], features_dim=cnn_output_dim, normalized_image=normalized_image)
        self.vector_encoder = nn.Flatten()
        total_concat_size = cnn_output_dim + style_feat_dim # + get_flattened_obs_dim(observation_space.spaces['vector'])

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, image) -> th.Tensor:
        '''
        for style feature encoder options:
        - one vgg feat + adaIN
        - multi-scale vgg + splatting + adaIN
        - old twin img encoder
        '''
        encoded_tensor_list = []
        img_input = image[:, 0:3, :, :]
        img_target = image[:, 3:6, :, :]
        cont_feat = self.vgg_net.encode_with_intermediate(img_input)
        style_feat = self.vgg_net.encode_with_intermediate(img_target)
        style_embedding = self.StyleEncoder(cont_feat, style_feat)

        return style_embedding


class HdrnetCNN(nn.Module):
    def __init__(
            self,
            features_dim: int = 512,
            normalized_image: bool = False,
    ) -> None:
        super().__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = 6
        self.cnn_splat = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(32), 32 32 32
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(64), 64 16 16
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(64), 64 8 8
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.cnn_local = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256), 64 8 8
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256), 64 8 8
        )
        self.cnn_global = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(256), 64 4 4
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        self.relu = nn.ReLU()

        self.linear = nn.Sequential(nn.Linear(4096, features_dim), nn.ReLU())
        self.cnt = 0

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # a = self.cnn(observations)
        # print(a.shape)
        # exit()
        B = observations.shape[0]
        feat = self.cnn_splat(observations)
        print(feat.shape)
        feat_local = self.cnn_local(feat)
        print(feat_local.shape)
        feat_global = self.cnn_global(feat)
        print(feat_global.shape)
        fuse = self.relu(feat_local + feat_global[:, :, None, None])
        x = self.linear(fuse.view(B, 4096))
        return x


if __name__ == '__main__':
    net = HdrnetCNN()
    img = torch.randn(1, 6, 64, 64).to('cuda')
    net.to('cuda')
    feat = net(img)
    print(feat.shape)
    a =  torch.randn(2, 3, 64, 64).to('cuda')
    b = torch.randn(2, 3, 64, 64).to('cuda')
    print(net.vgg_net.style_loss_adain(a, b))
    print(net.vgg_net.content_loss(a, b))