import torch
import torch.nn as nn
from model.beats.BEATs import BEATsConfig, BEATs
from model.classifiers import SingleLinearClassifier, ConvClassifier
from model.shared import ConvBnRelu, ResNorm, AdaResNorm, BroadcastBlock, TimeFreqSepConvolutions

class _BaseBackbone(nn.Module):
    """ Base Module for backbones. """

class DCASEBaselineCnn3(_BaseBackbone):
    """
    Previous baseline system of the task 1 of DCASE Challenge.
    A simple CNN consists of 3 conv layers and 2 linear layers.

    Note: Kernel size of the Max-pooling layers need to be changed for adapting to different size of inputs.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        base_channels (int): Number of base channels.
        kernel_size (int): Kernel size of convolution layers.
        dropout (float): Dropout rate.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_channels: int = 16, kernel_size: int = 7,
                 dropout: float = 0.3):
        super(DCASEBaselineCnn3, self).__init__()
        self.conv1 = ConvBnRelu(in_channels, base_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=True)
        self.conv2 = ConvBnRelu(base_channels, base_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=True)
        self.conv3 = ConvBnRelu(base_channels, base_channels * 2, kernel_size, padding=(kernel_size - 1) // 2,
                                bias=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        # Adjust the kernel of max_pooling util the output shape is (F=2, T=1)
        self.max_pooling1 = nn.MaxPool2d((5, 5))
        self.max_pooling2 = nn.MaxPool2d((4, 10))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(base_channels * 4, 100)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(100, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        # print('INPUT SHAPE:', x.shape)
        x = self.conv1(x)

        # print('CONV2 INPUT SHAPE:', x.shape)
        x = self.conv2(x)
        x = self.max_pooling1(x)
        x = self.dropout1(x)

        # print('CONV3 INPUT SHAPE:', x.shape)
        x = self.conv3(x)
        x = self.max_pooling2(x)
        x = self.dropout2(x)

        # print('FLATTEN INPUT SHAPE:', x.shape)
        x = self.flatten(x)

        # print('LINEAR INPUT SHAPE:', x.shape)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        # print('OUTPUT SHAPE:', x.shape)
        return x


class BCResNet(_BaseBackbone):
    """
    Implementation of BC-ResNet, based on Broadcasted Residual Learning.
    Check more details at: https://arxiv.org/abs/2106.04140

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        base_channels (int): Number of base channels that controls the complexity of model.
        depth (int): Network depth with single option: 15.
        kernel_size (int): Kernel size of each convolutional layer in BC blocks.
        dropout (float): Dropout rate.
        sub_bands (int): Number of sub-bands for SubSpectralNorm (SSN). ``1`` indicates SSN is not applied.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_channels: int = 40, depth: int = 15,
                 kernel_size: int = 3, dropout: float = 0.1, sub_bands: int = 1):
        super(BCResNet, self).__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.sub_bands = sub_bands

        cfg = {
            15: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 2.5, 2.5, 2.5, 'N'],
        }

        self.conv_layers = nn.Conv2d(in_channels, 2 * base_channels, 5, stride=2, bias=False, padding=2)
        # Compute the number of channels for each layer.
        layer_config = [int(i * base_channels) if not isinstance(i, str) else i for i in cfg[depth]]
        self.middle_layers = self._make_layers(base_channels, layer_config)
        # Get the index of channel number for the cla_layer.
        last_num_index = -1 if not isinstance(layer_config[-1], str) else -2
        # 1x1 convolution layer as the cla_layer.
        self.classifier = ConvClassifier(layer_config[last_num_index], num_classes)

    def _make_layers(self, width: int, layer_config: list):
        layers = []
        vt = 2 * width
        for v in layer_config:
            if v == 'N':
                layers += [ResNorm(channels=vt)]
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v != vt:
                layers += [BroadcastBlock(vt, v, self.kernel_size, self.dropout, self.sub_bands)]
                vt = v
            else:
                layers += [BroadcastBlock(vt, vt, self.kernel_size, self.dropout, self.sub_bands)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.middle_layers(x)
        x = self.classifier(x)
        return x


class TFSepNet(_BaseBackbone):
    """
    Implementation of TF-SepNet-64, based on Time-Frequency Separate Convolutions. Check more details at:
    https://ieeexplore.ieee.org/abstract/document/10447999 and
    https://dcase.community/documents/challenge2024/technical_reports/DCASE2024_Cai_61_t1.pdf

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        base_channels (int): Number of base channels that controls the complexity of model.
        depth (int): Network depth with two options: 16 or 17. When depth = 17, an additional Max-pooling layer is inserted before the last TF-SepConvs black.
        kernel_size (int): Kernel size of each convolutional layer in TF-SepConvs blocks.
        dropout (float): Dropout rate.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_channels: int = 64, depth: int = 17,
                 kernel_size: int = 3, dropout: float = 0.1):
        super(TFSepNet, self).__init__()
        assert base_channels % 2 == 0, "Base_channels should be divisible by 2."
        self.dropout = dropout
        self.kernel_size = kernel_size

        # Two settings of the depth. ``17`` have an additional Max-pooling layer before the final block of TF-SepConvs.
        cfg = {
            16: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 2.5, 2.5, 2.5, 'N'],
            17: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 'M', 2.5, 2.5, 2.5, 'N'],
        }

        self.conv_layers = nn.Sequential(ConvBnRelu(in_channels, base_channels // 2, 3, stride=2, padding=1),
                                         ConvBnRelu(base_channels // 2, 2 * base_channels, 3, stride=2, padding=1,
                                                    groups=base_channels // 2))
        # Compute the number of channels for each layer.
        layer_config = [int(i * base_channels) if not isinstance(i, str) else i for i in cfg[depth]]
        self.middle_layers = self._make_layers(base_channels, layer_config)
        # Get the index of channel number for the cla_layer.
        last_num_index = -1 if not isinstance(layer_config[-1], str) else -2
        # 1x1 convolution layer as the cla_layer.
        self.classifier = ConvClassifier(layer_config[last_num_index], num_classes)

    def _make_layers(self, width: int, layer_config: list):
        layers = []
        vt = width * 2
        for v in layer_config:
            if v == 'N':
                layers += [ResNorm(channels=vt)]
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v != vt:
                layers += [TimeFreqSepConvolutions(vt, v, self.kernel_size, self.dropout)]
                vt = v
            else:
                layers += [TimeFreqSepConvolutions(vt, vt, self.kernel_size, self.dropout)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.middle_layers(x)
        x = self.classifier(x)
        return x


class PretrainedBEATs(_BaseBackbone):
    """
    Module wrapping a BEATs encoder with pretrained weights and a new linear classifier.
    Check more details at: https://arxiv.org/abs/2212.09058

    Args:
        pretrained (str): path to the pretrained checkpoint. Leave ``None`` when no need pretrained.
    """

    def __init__(self, pretrained=None, num_classes=10, **kwargs):
        super(PretrainedBEATs, self).__init__()
        # Load model config and weights from checkpoints when use pretrained, otherwise use default settings
        ckpt = torch.load(pretrained) if pretrained else None
        hyperparams = ckpt['cfg'] if pretrained else kwargs
        cfg = BEATsConfig(hyperparams)
        self.encoder = BEATs(cfg)
        if pretrained:
            self.encoder.load_state_dict(ckpt['model'], strict=False)
        # Create a new linear classifier
        self.classifier = SingleLinearClassifier(in_features=cfg.encoder_embed_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.encoder.extract_features(x)[0]
        return self.classifier(x)

class PretrainedMN40(_BaseBackbone):
    def __init__(self, pretrained=None, num_classes=10, **kwargs):
        super().__init__()
        import sys
        sys.path.append('/home/work/LDH/EfficientAT')
        from models.mn.model import get_model

        encoder = get_model(
            num_classes=527,
            width_mult=4.0,
            head_type="fully_convolutional",
            input_dim_f=128,
            input_dim_t=1000
        )
        self.encoder = encoder.features  # ✅ 먼저 정의

        if pretrained:
            state_dict = torch.load(pretrained, map_location='cpu')
            # 'features.' prefix 제거
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('features.'):
                    new_state_dict[k[len('features.'):]] = v
            print("=== Pretrained weight 로딩 직전 ===")
            missing = self.encoder.load_state_dict(new_state_dict, strict=False)
            print("=== Pretrained weight 로딩 결과 ===")
            print(missing)

        # 마지막 Conv2d 계층을 찾아서 out_channels를 가져오기
        last_conv = None
        for m in reversed(list(self.encoder.modules())):
            if isinstance(m, nn.Conv2d):
                last_conv = m
                break
        last_channel = last_conv.out_channels

        self.classifier = nn.Sequential(
            nn.Conv2d(last_channel, num_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.requires_4d_input = True  # ← 이 줄 추가!

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.classifier(x)
        return x.squeeze(-1).squeeze(-1)

#FPASST
class PretrainedFPASST(_BaseBackbone):
    def __init__(
        self,
        pretrained=None,
        num_classes=10,
        input_fdim=128,
        input_tdim=1000,
        arch="passt_s_kd_p16_128_ap486",
        frame_patchout=0,
        pos_embed_length=1000,
        **kwargs
    ):
        super().__init__()
        import sys
        sys.path.append('/home/work/LDH/PretrainedSED')
        from models.frame_passt.fpasst import get_model

        if pos_embed_length is None:
            pos_embed_length = input_tdim

        self.encoder = get_model(
            arch=arch,
            pretrained=False,
            n_classes=num_classes,
            in_channels=1,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            frame_patchout=frame_patchout,
            pos_embed_length=pos_embed_length,
        )

        if pretrained and isinstance(pretrained, str) and pretrained.endswith('.pt'):
            state_dict = torch.load(pretrained, map_location='cpu')
            if 'pos_embed' in state_dict and hasattr(self.encoder, 'pos_embed'):
                pe_pre = state_dict['pos_embed']
                pe_cur = self.encoder.pos_embed
                if pe_pre.shape != pe_cur.shape:
                    import torch.nn.functional as F
                    print(f"[INFO] Interpolating pos_embed from {pe_pre.shape} to {pe_cur.shape}")
                    pe_pre = F.interpolate(
                        pe_pre.permute(0, 3, 1, 2),
                        size=pe_cur.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                    state_dict['pos_embed'] = pe_pre
            self.encoder.load_state_dict(state_dict, strict=False)
            print("=== PaSST Pretrained weight 로딩 완료 ===")

        self.requires_4d_input = True
        # PaSST 임베딩 차원은 768로 고정 (모델 구조에 따라 다를 수 있음)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        # (B, 1, 1, F, T) → (B, 1, F, T)
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        # (B, F, T) → (B, 1, F, T)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        # patch_embed 입력 shape이 1채널이 아니면 오류
        if x.shape[1] != 1:
            raise RuntimeError(f"patch_embed 입력 채널이 1이어야 합니다. 현재 shape: {x.shape}")
        out = self.encoder(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        # out: [B, T, 768] or [B, 250, 768] 등
        if out.ndim == 3:
            out = out.mean(dim=1)  # time 평균
        out = self.head(out)  # [B, num_classes]
        return out