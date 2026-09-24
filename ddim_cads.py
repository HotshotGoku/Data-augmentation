"""CADS (Condition-Annealed Diffusion Sampling) for the ControlNet augmenter.

Adapted from Sadat et al., "CADS: Unleashing the Diversity of Diffusion Models
through Condition-Annealed Sampling" (ICLR 2024). Since our text prompt is empty,
the meaningful conditioning is the SPATIAL CONTROL (the source image in c_concat).
We anneal it over the sampling trajectory: early denoising steps (noisy) see a
perturbed source -> more layout diversity; late steps (clean) see the exact source
-> faithful detail. This is a diversity knob that needs NO training.

Drop-in subclass of cldm.ddim_hacked.DDIMSampler. To use, replace the sampler
instance (see gen_cads.py). Setting tau2 <= tau1 makes gamma == 1 at every step, i.e.
CADS is OFF and behavior is byte-identical to the stock sampler.

All knobs are constructor args (set from env by gen_cads.py):
  tau1, tau2 : annealing schedule over normalized diffusion progress p in [0,1]
               (p = 1 at the noisy start, p = 0 near the clean end).
               p >= tau2 -> gamma = 0 (fully noised condition)
               p <= tau1 -> gamma = 1 (clean condition); linear in between.
  s          : noise scale added to the condition.
  psi        : rescale mixing factor (CADS eq. 4); 1.0 = fully rescaled.
  rescale    : preserve the condition's per-tensor mean/std after noising.
  clamp01    : clamp the annealed condition back to [0,1] (correct for image control).
  keys       : which conditioning entries to anneal (default: c_concat only).
  cads_uncond: also anneal the classifier-free-guidance uncond branch's control.
"""
import torch
import numpy as np

from cldm.ddim_hacked import DDIMSampler


class CADSSampler(DDIMSampler):
    def __init__(self, model, schedule="linear",
                 tau1=0.6, tau2=0.9, s=0.1, psi=1.0, rescale=True, clamp01=True,
                 keys=("c_concat",), cads_uncond=True, **kwargs):
        super().__init__(model, schedule=schedule, **kwargs)
        self.tau1 = float(tau1)
        self.tau2 = float(tau2)
        self.s = float(s)
        self.psi = float(psi)
        self.rescale = bool(rescale)
        self.clamp01 = bool(clamp01)
        self.keys = tuple(keys)
        self.cads_uncond = bool(cads_uncond)

    def _gamma(self, index, total):
        # index counts DOWN from total-1 (noisy) to 0 (clean) -> normalized progress p.
        p = index / max(total - 1, 1)
        if self.tau2 <= self.tau1:
            return 1.0
        g = (self.tau2 - p) / (self.tau2 - self.tau1)
        return float(min(max(g, 0.0), 1.0))

    def _anneal_tensor(self, y, gamma):
        n = torch.randn_like(y)
        y_hat = (gamma ** 0.5) * y + self.s * ((1.0 - gamma) ** 0.5) * n
        if self.rescale:
            m0, s0 = y.mean(), y.std()
            mh, sh = y_hat.mean(), y_hat.std()
            y_res = (y_hat - mh) / (sh + 1e-8) * s0 + m0
            y_hat = self.psi * y_res + (1.0 - self.psi) * y_hat
        if self.clamp01:
            y_hat = y_hat.clamp(0.0, 1.0)
        return y_hat

    def _anneal_cond(self, cond, gamma):
        if cond is None or gamma >= 1.0:
            return cond
        out = dict(cond)
        for k in self.keys:
            lst = cond.get(k)
            if not lst:
                continue
            out[k] = [None if t is None else self._anneal_tensor(t, gamma) for t in lst]
        return out

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, unconditional_conditioning=None, **kwargs):
        total = self.ddim_timesteps.shape[0]
        gamma = self._gamma(index, total)
        c_a = self._anneal_cond(c, gamma)
        uc_a = (self._anneal_cond(unconditional_conditioning, gamma)
                if self.cads_uncond else unconditional_conditioning)
        return super().p_sample_ddim(x, c_a, t, index,
                                     unconditional_conditioning=uc_a, **kwargs)
