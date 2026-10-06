"""Resample osztály csere a torchaudio_compat.py-ben (helyes racionális struktúra)."""
import io
import re
import sys

PATH = r"C:\Users\istva\Dev\portfolio\Projects\audiobook narrator\auris\reader\core\torchaudio_compat.py"

NEW = '''class Resample:
    """``torchaudio.transforms.Resample`` — racionális (polyfázisú) átméretezés.

    A torchaudio ``sinc_interp_hann`` módszerének lényege: a
    ``rolloff * min(f_in, f_out) / 2`` Hz levágású, Hann-ablakos szinc
    anti-alias szűrő, ``gcd``-redukció után. A megvalósítás a
    ``scipy.signal.resample_poly`` struktúráját követi:

    1. nullabeillesztés ``up = q`` arányban,
    2. szűrés a felfelé mintázott rácson,
    3. kivétel ``down = p`` lépésenként.
    """

    def __init__(
        self,
        orig_freq: int = 16000,
        new_freq: int = 16000,
        resampling_method: str = "sinc_interp_hann",
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        beta=None,
        align_end: bool = False,
    ):
        self.orig_freq = int(orig_freq)
        self.new_freq = int(new_freq)
        self.rolloff = float(rolloff)
        self.width = int(lowpass_filter_width)

    def _prototype(self, dtype, device):
        """Normált, ``up``-szel felszorzott anti-alias szűrő (1 csatorna)."""
        torch = _torch()
        g = math.gcd(self.orig_freq, self.new_freq)
        p = self.orig_freq // g
        q = self.new_freq // g
        if p == q:
            return None, p, q

        up = q
        rate = float(self.orig_freq) * up
        cutoff_hz = self.rolloff * min(float(self.orig_freq), float(self.new_freq)) / 2.0
        fc = cutoff_hz / rate  # ciklus/minta a felfelé mintázott rácson
        half = max(1, int(math.ceil(self.width * up)))
        n = torch.arange(-half, half + 1, dtype=torch.float64)
        x = 2.0 * fc * n
        sinc = torch.where(
            x == 0, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x)
        )
        window = 0.5 * (1.0 + torch.cos(math.pi * n / half))
        kernel = sinc * window
        kernel = kernel / kernel.abs().sum() * up
        return kernel.to(dtype=dtype).to(device), p, q

    def forward(self, waveform):
        torch = _torch()
        import torch.nn.functional as F

        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
        channels, length = waveform.shape

        kernel, p, q = self._prototype(waveform.dtype, waveform.device)
        if kernel is None:
            out = waveform
        else:
            up, down = q, p
            flat = waveform.reshape(1, -1)
            if up > 1:
                flat = F.fold(
                    flat.unsqueeze(-1).transpose(1, 2).reshape(1, -1),
                    output_size=(1, (length - 1) * up + 1),
                    kernel_size=(1, up),
                    stride=(1, up),
                )
            pad = kernel.numel() // 2
            filtered = F.conv1d(flat, kernel.flip(-1).view(1, 1, -1), padding=pad)
            if down > 1:
                filtered = filtered[..., ::down]
            out = filtered.reshape(channels, -1)

        target = new_len(length, self.orig_freq, self.new_freq)
        out = out[..., :target]
        return out.squeeze(0) if squeeze else out

    __call__ = forward


'''

with io.open(PATH, "r", encoding="utf-8") as fh:
    src = fh.read()

start = src.index("class Resample:")
end = src.index("def new_len(")
src = src[:start] + NEW + src[end:]

with io.open(PATH, "w", encoding="utf-8") as fh:
    fh.write(src)
print("Resample osztaly lecserelve")