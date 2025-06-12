# 임시 스크립트 예시
import torch
from util.spec_extractor import PAugmentMelSTFT

# 원하는 time dimension
T = 993
n_fft = 1024
hopsize = 320
pad = n_fft // 2

# T에 맞는 wave 길이 자동 계산
wave_len = (T - 1) * hopsize - 2 * pad + n_fft

print(f"wave_len for T={T}: {wave_len}")

x = torch.randn(1, wave_len)

spec = PAugmentMelSTFT(
    n_mels=128, sr=32000, win_length=800, hopsize=hopsize, n_fft=n_fft,
    freqm=64, timem=256, htk=False, fmin=0, fmax=None, norm=1,
    fmin_aug_range=10, fmax_aug_range=2000, fast_norm=False,
    preamp=True, padding='center', periodic_window=True
)
mel = spec(x)
print(mel.shape)  # [B, 16, 128, 992]가 나오면 성공!