Ekhane rakho: **project root-e `checkpoints/vae_pretrained_cache/`** folder banaye tar bhitore.

```bash
mkdir -p checkpoints/vae_pretrained_cache
```

Browser theke download korle, HuggingFace-er "Files and versions" tab-e **duita file** lagbe (shudhu `.safetensors` ta alone kaj korbe na) — same folder-e dao:
```
checkpoints/vae_pretrained_cache/
├── config.json
└── diffusion_pytorch_model.safetensors
```

Tarpor `configs/config_diffusion.yaml`-e path ta update koro:
```yaml
vae_pretrained_path: "checkpoints/vae_pretrained_cache"
```
(ei path-i `06b_diffusion_pipeline_fix.md`-e `AutoencoderKL.from_pretrained(pretrained_path)`-e direct use hoy, `checkpoints/` er bhitore rakha existing `checkpoints/best_model_*.pth` convention-er sathe consistent.)

**Quick verify (offline mode-e thik load hocche kina check):**
```bash
python -c "
import os
os.environ['HF_HUB_OFFLINE'] = '1'
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained('checkpoints/vae_pretrained_cache')
print('Loaded OK:', sum(p.numel() for p in vae.parameters()), 'params')
"
```
Eita run kore error na dile, path thik ache.