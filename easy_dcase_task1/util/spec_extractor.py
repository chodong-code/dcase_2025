import numpy as np
import torch
import torchaudio
from torch import nn
import torchaudio.compliance.kaldi as ta_kaldi
import dcase_util

class _SpecExtractor(nn.Module):
    """ Base Module for spectrogram extractors. """

class Cnn3Mel(_SpecExtractor):
    """ Mel extractor for previous CNN3 baseline system. """
    def __init__(self,
                 spectrogram_type="magnitude",
                 hop_length_seconds=0.02,
                 win_length_seconds=0.04,
                 window_type="hamming_asymmetric",
                 n_mels=40,
                 n_fft=2048,
                 fmin=0,
                 fmax=22050,
                 htk=False,
                 normalize_mel_bands=False,
                 **kwargs):
        super().__init__()
        self.extractor = dcase_util.features.MelExtractor(spectrogram_type=spectrogram_type,
                                                          hop_length_seconds=hop_length_seconds,
                                                          win_length_seconds=win_length_seconds,
                                                          window_type=window_type,
                                                          n_mels=n_mels,
                                                          n_fft=n_fft,
                                                          fmin=fmin,
                                                          fmax=fmax,
                                                          htk=htk,
                                                          normalize_mel_bands=normalize_mel_bands,
                                                          **kwargs)

    def forward(self, x):
        mel = []
        for wav in x:
            wav = wav.cpu().numpy()
            mel.append(self.extractor.extract(wav))
        mel = np.stack(mel)
        mel = torch.from_numpy(mel).to(x.device)
        return mel


class CpMel(_SpecExtractor):
    """
    Mel extractor for CP-JKU systems. Adapted from: https://github.com/fschmid56/cpjku_dcase23
    """
    def __init__(self, n_mels=256, sr=32000, win_length=3072, hop_size=500, n_fft=4096, fmin=0.0, fmax=None):
        super().__init__()
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        self.fmax = sr // 2 if fmax is None else fmax
        self.hop_size = hop_size
        self.register_buffer('window', torch.hann_window(win_length, periodic=False),
                             persistent=False)
        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)

    def forward(self, x):
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient.to(x.device)).squeeze(1)
        x = torch.stft(x, self.n_fft, hop_length=self.hop_size, win_length=self.win_length,
                       center=True, normalized=False, window=self.window.to(x.device), return_complex=True)
        x = torch.view_as_real(x)
        x = (x ** 2).sum(dim=-1)  # power mag
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels, self.n_fft, self.sr,
                                                                 self.fmin, self.fmax, vtln_low=100.0, vtln_high=-500.,
                                                                 vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)
        # Log mel spectrogram
        melspec = (melspec + 0.00001).log()
        # Fast normalization
        melspec = (melspec + 4.5) / 5.
        return melspec


class BEATsMel(_SpecExtractor):
    """ Mel extractor for BEATs model. """
    def __init__(self, dataset_mean: float = 15.41663, dataset_std: float = 6.55582):
        super(BEATsMel, self).__init__()
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

    def forward(self, x):
        fbanks = []
        for waveform in x:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - self.dataset_mean) / (2 * self.dataset_std)
        return fbank

class AugmentMelSTFT(_SpecExtractor):
    def __init__(self, n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192,
                 fmin=0.0, fmax=None, fmin_aug_range=10, fmax_aug_range=2000):
        super().__init__()

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = float(fmax)
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(win_length, periodic=False),
                             persistent=False)

        assert fmin_aug_range >= 1, f"fmin_aug_range={fmin_aug_range} should be >=1"
        assert fmax_aug_range >= 1, f"fmax_aug_range={fmax_aug_range} should be >=1"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)
        self.freqm = torch.nn.Identity() if freqm == 0 else torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        self.timem = torch.nn.Identity() if timem == 0 else torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x):
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)

        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=True)
        x = x.abs() ** 2  # power magnitude

        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels, self.n_fft, self.sr,
            fmin, fmax, vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0
        )
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 1e-5).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        # ✅ 시간축을 1000으로 맞추기
        target_T = 1000
        current_T = melspec.shape[-1]
        if current_T < target_T:
            pad_len = target_T - current_T
            melspec = nn.functional.pad(melspec, (0, pad_len), value=0)
        elif current_T > target_T:
            melspec = melspec[..., :target_T]

        melspec = (melspec + 4.5) / 5.0  # fast normalization

        return melspec


class pAugmentMelSTFT(_SpecExtractor):
    def __init__(
            self,
            n_mels=128,
            sr=32000,
            win_length=None,
            hopsize=320,
            n_fft=1024,
            freqm=0,
            timem=0,
            htk=False,
            fmin=0.0,
            fmax=None,
            norm=1,
            fmin_aug_range=1,
            fmax_aug_range=1,
            fast_norm=False,
            preamp=True,
            padding="center",
            periodic_window=True,
    ):
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        # Similar config to the spectrograms used in AST: https://github.com/YuanGongND/ast

        if win_length is None:
            win_length = n_fft

        if isinstance(win_length, list) or isinstance(win_length, tuple):
            assert isinstance(n_fft, list) or isinstance(n_fft, tuple)
            assert len(win_length) == len(n_fft)
        else:
            win_length = [win_length]
            n_fft = [n_fft]

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.htk = htk
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize
        self.preamp = preamp
        for win_l in self.win_length:
            self.register_buffer(
                f"window_{win_l}",
                torch.hann_window(win_l, periodic=periodic_window),
                persistent=False,
            )
        assert (
                fmin_aug_range >= 1
        ), f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert (
                fmin_aug_range >= 1
        ), f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer(
            "preemphasis_coefficient", torch.as_tensor([[[-0.97, 1]]]), persistent=False
        )
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=False)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=False)
        self.fast_norm = fast_norm
        self.padding = padding
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.iden = nn.Identity()

    def forward(self, x):
        if self.preamp:
            x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient)
        x = x.squeeze(1)

        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()

        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mels = []
        for n_fft, win_length in zip(self.n_fft, self.win_length):
            x_temp = x
            if self.padding == "same":
                pad = win_length - self.hopsize
                self.iden(x_temp)  # printing
                x_temp = torch.nn.functional.pad(x_temp, (pad // 2, pad // 2), mode="reflect")
                self.iden(x_temp)  # printing

            x_temp = torch.stft(
                x_temp,
                n_fft,
                hop_length=self.hopsize,
                win_length=win_length,
                center=self.padding == "center",
                normalized=False,
                window=getattr(self, f"window_{win_length}"),
                return_complex=True
            )
            x_temp = torch.view_as_real(x_temp)
            x_temp = (x_temp ** 2).sum(dim=-1)  # power mag

            mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels, n_fft, self.sr,
                                                                     fmin, fmax, vtln_low=100.0, vtln_high=-500.,
                                                                     vtln_warp_factor=1.0)
            mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                        device=x.device)

            with torch.cuda.amp.autocast(enabled=False):
                x_temp = torch.matmul(mel_basis, x_temp)

            x_temp = torch.log(torch.clip(x_temp, min=1e-7))

            mels.append(x_temp)

        mels = torch.stack(mels, dim=1)

        if self.training:
            mels = self.freqm(mels)
            mels = self.timem(mels)
        if self.fast_norm:
            mels = (mels + 4.5) / 5.0  # fast normalization

        return mels

    def extra_repr(self):
        return "winsize={}, hopsize={}".format(self.win_length, self.hopsize)
