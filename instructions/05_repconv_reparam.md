# P5 — Structural Reparameterization (RepConv-style) for HFF Projections

## Objective
Secondary **efficiency** novelty (not an accuracy lever — don't expect this
to move the 97.5%/90% targets). Honest framing for the paper: *"we adapt
structural reparameterization (RepVGG, Ding et al. CVPR 2021; applied at
scale by GCNet, Peng et al. CVPR 2025) to HFF's stage-projection layers."*
We are not claiming to have invented reparameterization — only this
specific, correctly-derived application to HFF's non-overlapping strided
projection convs.

## Priority / Cost / Risk
Low cost, **but not a drop-in swap** — see the checkpoint-compatibility
warning below. Scope this as its own ablation row, not a silent default
change.

---

## Important technical detail (get this right, it differs from textbook RepVGG)
`HFFBlock`'s three projection convs (`src/models/components.py`, lines
99/101/103) use **stride == kernel_size** (8/8, 4/4, 2/2) — i.e.
non-overlapping "patchify" convolutions, not standard overlapping convs.
Textbook RepVGG fusion pads a 1×1 kernel into the *center* of a k×k kernel,
which assumes an *odd* kernel size with a well-defined center pixel. Here
all three kernels are **even**, so there is no center tap. The correct
fusion for this specific stride-equals-kernel-size case: with `stride=k,
padding=0`, a 1×1 conv's single receptive-field position aligns with the
**(0,0) corner** of each k×k input patch (not a geometric center) — so the
exact (not approximate) fusion adds the 1×1 branch's weights into kernel
position **(0,0)** of the k×k kernel, leaving all other taps unchanged.
This is mathematically exact for this stride configuration — worth stating
explicitly in the paper as it shows the fusion was actually derived for
this architecture, not copy-pasted from a different conv configuration.

---

## New module: `src/models/components.py` — add `RepProjConv`

Add this class before `HFFBlock` (before line 74):
```python
class RepProjConv(nn.Module):
    """
    Structural reparameterization (RepVGG/GCNet-style) applied to HFF's
    stage-projection convolutions. During training, a k×k strided conv-BN
    branch runs in parallel with a 1×1 strided conv-BN branch, summed.
    At inference, fuse() folds both branches (+ BN affine params) into a
    single k×k conv — no accuracy cost, fewer inference-time params/FLOPs.

    NOTE: stride == kernel_size here (non-overlapping patchify conv), so
    the 1×1 branch aligns with each patch's (0,0) corner, not a geometric
    center — fusion adds the 1×1 weights into kernel tap (0,0) exactly
    (see 05_repconv_reparam.md for the derivation).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.fused = False

        self.kxk_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False)
        self.kxk_bn = nn.BatchNorm2d(out_channels)
        self.oxo_conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.oxo_bn = nn.BatchNorm2d(out_channels)
        self.fused_conv = None

    def forward(self, x):
        if self.fused:
            return self.fused_conv(x)
        return self.kxk_bn(self.kxk_conv(x)) + self.oxo_bn(self.oxo_conv(x))

    @staticmethod
    def _fuse_bn(conv, bn):
        kernel = conv.weight
        gamma, beta = bn.weight, bn.bias
        mean, var, eps = bn.running_mean, bn.running_var, bn.eps
        std = torch.sqrt(var + eps)
        fused_kernel = kernel * (gamma / std).reshape(-1, 1, 1, 1)
        fused_bias = beta - mean * gamma / std
        return fused_kernel, fused_bias

    @torch.no_grad()
    def fuse(self):
        """Call once after training (model should be in eval() mode so BN
        running stats are what you expect), before inference/export."""
        k_kernel, k_bias = self._fuse_bn(self.kxk_conv, self.kxk_bn)
        o_kernel, o_bias = self._fuse_bn(self.oxo_conv, self.oxo_bn)

        fused_kernel = k_kernel.clone()
        fused_kernel[:, :, 0, 0] += o_kernel[:, :, 0, 0]
        fused_bias = k_bias + o_bias

        self.fused_conv = nn.Conv2d(
            self.kxk_conv.in_channels, self.kxk_conv.out_channels,
            self.kernel_size, stride=self.stride, bias=True
        ).to(fused_kernel.device)
        self.fused_conv.weight.copy_(fused_kernel)
        self.fused_conv.bias.copy_(fused_bias)
        self.fused = True
```

## `HFFBlock.__init__` (lines 93-110) — replace lines 99/101/103
```python
        use_rep_proj = kwargs.get('use_rep_proj', False)   # see wiring note below
        ConvCls = RepProjConv if use_rep_proj else nn.Conv2d
        if use_rep_proj:
            self.proj_stage1 = RepProjConv(stage_channels[0], target_channels, kernel_size=8, stride=8)
            self.proj_stage2 = RepProjConv(stage_channels[1], target_channels, kernel_size=4, stride=4)
            self.proj_stage3 = RepProjConv(stage_channels[2], target_channels, kernel_size=2, stride=2)
        else:
            self.proj_stage1 = nn.Conv2d(stage_channels[0], target_channels, kernel_size=8, stride=8)
            self.proj_stage2 = nn.Conv2d(stage_channels[1], target_channels, kernel_size=4, stride=4)
            self.proj_stage3 = nn.Conv2d(stage_channels[2], target_channels, kernel_size=2, stride=2)
```
`HFFBlock.__init__`'s signature needs `use_rep_proj: bool = False` added
explicitly (don't rely on `**kwargs` — write it as a real named parameter).
**`_project_and_norm` (line 123) and `forward` (line 131) need no changes**
— both call `proj_conv(x)` generically, which works identically whether
`proj_stageX` is a plain `nn.Conv2d` or a `RepProjConv`.

## ⚠️ Checkpoint compatibility warning
Enabling `use_rep_proj=True` changes `HFFBlock`'s submodule structure
(`proj_stageX.weight` → `proj_stageX.kxk_conv.weight` /
`proj_stageX.kxk_bn.*` / `proj_stageX.oxo_conv.weight` /
`proj_stageX.oxo_bn.*`). **Existing checkpoints trained with
`use_rep_proj=False` cannot be loaded into a `use_rep_proj=True` model and
vice versa.** This requires a fresh training run, not a drop-in swap —
treat it as its own ablation row (`+RepHFF`), never silently default it on.

## `src/models/dr_model.py` — thread the flag through
In `RetiNA_Net.__init__`, the `HFFBlock(...)` construction (inside
`if self.use_hff:`) needs `use_rep_proj=use_rep_proj` added, with
`use_rep_proj: bool = False` added to `RetiNA_Net.__init__`'s signature
too. Also add a convenience method for inference-time fusion:
```python
    def fuse_reparam_blocks(self):
        """Call after loading a trained use_rep_proj=True checkpoint, in
        eval() mode, before running inference/TTA/ensemble — folds each
        RepProjConv's two branches into one conv for faster inference."""
        if self.use_hff and getattr(self.hff.proj_stage1, 'fuse', None):
            self.hff.proj_stage1.fuse()
            self.hff.proj_stage2.fuse()
            self.hff.proj_stage3.fuse()
```

## `main.py` — wiring
- `create_model`: read `use_rep_proj = config.get("use_rep_proj", False)`,
  pass through to `RetiNA_Net(...)`.
- `run_test` / `run_external_validation`: after loading each fold's
  checkpoint and before running TTA/ensemble inference, call
  `model.fuse_reparam_blocks()` on each loaded model (only has an effect
  when `use_rep_proj=True`; harmless no-op otherwise).

## `configs/config.yaml`
```yaml
use_rep_proj: False   # structural reparameterization on HFF projections (P5)
```

---

## Validation / ablation design
Row: `+RepHFF` — same config as `arch_full`/`proposed` but with
`use_rep_proj: True`, fresh training run. Report: accuracy/QWK (should
match `arch_full` within noise — this is an efficiency change, not an
accuracy change), plus params/FLOPs/inference-time-per-image before vs.
after calling `fuse_reparam_blocks()`.

## Expected outcome
Modest (few-%) reduction in HFF's own param/FLOP count at inference, no
accuracy regression. Report as a dedicated efficiency ablation, not mixed
into the main architecture-ablation table.
