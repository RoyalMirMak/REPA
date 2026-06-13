import os
from torchvision.datasets.utils import download_url
import torch
import torchvision.models as torchvision_models
import timm
from models import mocov3_vit
import math
import warnings


# code from SiT repository
pretrained_models = {'last.pt'}

def download_model(model_name):
    """
    Downloads a pre-trained SiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0'
        download_url(web_path, 'pretrained_models', filename=model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model

def fix_mocov3_state_dict(state_dict):
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder."):]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            if 'head' not in new_k and new_k.split('.')[0] != 'fc':
                state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if 'pos_embed' in state_dict.keys():
        state_dict['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
            state_dict['pos_embed'], [16, 16],
        )
    return state_dict


def get_dinov2_model_path(model_name: str) -> str:
    """
    Get the path to a DINOv2 model weight file.
    
    Checks the following locations in order:
    1. DINOV2_WEIGHTS_DIR environment variable (if set)
    2. ./REPA/ckpts/dinov2/ (relative to project root)
    3. ./ckpts/dinov2/ (relative to current directory)
    
    Args:
        model_name: Model name (e.g., 'dinov2_vitb14_pretrain.pth')
    
    Returns:
        Full path to the weight file
    
    Raises:
        FileNotFoundError: If the weight file is not found
    """
    # Check environment variable first
    weights_dir = os.environ.get('DINOV2_WEIGHTS_DIR')
    if weights_dir:
        path = os.path.join(weights_dir, model_name)
        if os.path.isfile(path):
            return path
    
    # Check relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Try ./REPA/ckpts/dinov2/
    path = os.path.join(project_root, 'REPA', 'ckpts', 'dinov2', model_name)
    if os.path.isfile(path):
        return path
    
    # Try ./ckpts/dinov2/
    path = os.path.join(script_dir, 'ckpts', 'dinov2', model_name)
    if os.path.isfile(path):
        return path
    
    raise FileNotFoundError(
        f"DINOv2 weights not found at any of the expected locations.\n"
        f"Please download weights using: bash scripts/download_dinov2_weights.sh\n"
        f"Or set DINOV2_WEIGHTS_DIR environment variable.\n"
        f"Searched:\n  - {weights_dir or '(DINOV2_WEIGHTS_DIR not set)'}\n"
        f"  - {path}"
    )


def _load_dinov2_from_weights(model_name: str, img_size: int = 224):
    """
    Load DINOv2 model from local weight file using a timm backbone.
    """
    import timm

    model_configs = {
        'dinov2_vitb14':     ('dinov2_vitb14_pretrain.pth',       'vit_base_patch14_dinov2'),
        'dinov2_vitl14':     ('dinov2_vitl14_pretrain.pth',       'vit_large_patch14_dinov2'),
        'dinov2_vitg14':     ('dinov2_vitg14_pretrain.pth',       'vit_giant_patch14_dinov2'),
        'dinov2_vitb14_reg': ('dinov2_vitb14_reg4_pretrain.pth',  'vit_base_patch14_reg4_dinov2'),
        'dinov2_vitl14_reg': ('dinov2_vitl14_reg4_pretrain.pth',  'vit_large_patch14_reg4_dinov2'),
        'dinov2_vitg14_reg': ('dinov2_vitg14_reg4_pretrain.pth',  'vit_giant_patch14_reg4_dinov2'),
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown DINOv2 model: {model_name}")

    weight_file, model_type = model_configs[model_name]
    weight_path = get_dinov2_model_path(weight_file)

    # Build timm backbone at the resolution REPA actually feeds in (16*14 = 224).
    encoder = timm.create_model(model_type, pretrained=False, img_size=img_size)

    print(f"Loading DINOv2 weights from {weight_path}")
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)

    if isinstance(state_dict, dict) and 'teacher' in state_dict:
        state_dict = state_dict['teacher']
    elif isinstance(state_dict, dict) and 'student' in state_dict:
        state_dict = state_dict['student']

    # Map FAIR keys -> timm keys
    remapped = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        if nk.startswith('backbone.'):
            nk = nk[len('backbone.'):]
        nk = nk.replace('.mlp.w12.', '.mlp.fc1.')
        nk = nk.replace('.mlp.w3.',  '.mlp.fc2.')
        remapped[nk] = v

    # Drop keys that timm's backbone doesn't have
    remapped = {k: v for k, v in remapped.items() if k not in {'mask_token'}}

    # Resample pos_embed (518/14=37 grid in FAIR weights -> img_size/14 grid in timm)
    if 'pos_embed' in remapped:
        num_prefix_tokens = getattr(encoder, 'num_prefix_tokens', 1)
        new_grid = img_size // 14
        remapped['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
            remapped['pos_embed'],
            new_size=[new_grid, new_grid],
            num_prefix_tokens=num_prefix_tokens,
        )

    missing, unexpected = encoder.load_state_dict(remapped, strict=False)
    missing    = [k for k in missing    if not k.startswith('head')]
    unexpected = [k for k in unexpected if not k.startswith('head')]
    if missing or unexpected:
        print(f"[DINOv2] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        print(f"[DINOv2] unexpected  : {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        raise RuntimeError(
            f"State dict mismatch when loading {model_name}: "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )

    # Patch forward_features to return a FAIR-style dict expected by REPA's train.py
    num_prefix_tokens = getattr(encoder, 'num_prefix_tokens', 1)  # 1 for non-reg, 5 for reg4
    timm_forward_features = encoder.forward_features

    def forward_features_dinov2_style(x):
        feats = timm_forward_features(x)            # [B, P + N, C]
        cls   = feats[:, 0]                          # [B, C]
        patch = feats[:, num_prefix_tokens:]         # [B, N, C]
        return {
            'x_norm_clstoken':    cls,
            'x_norm_patchtokens': patch,
            'x_prenorm':          feats,             # not pre-norm, but kept for API compat
            'masks':              None,
        }

    encoder.forward_features = forward_features_dinov2_style

    print(f"✓ Loaded DINOv2 {model_name} from {weight_path} (img_size={img_size})")
    return encoder

@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    assert (resolution == 256) or (resolution == 512)
    
    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        # Currently, we only support 512x512 experiments with DINOv2 encoders.
        if resolution == 512:
            if encoder_type != 'dinov2':
                raise NotImplementedError(
                    "Currently, we only support 512x512 experiments with DINOv2 encoders."
                    )

        architectures.append(architecture)
        encoder_types.append(encoder_type)
        if encoder_type == 'mocov3':
            if architecture == 'vit':
                if model_config == 's':
                    encoder = mocov3_vit.vit_small()
                elif model_config == 'b':
                    encoder = mocov3_vit.vit_base()
                elif model_config == 'l':
                    encoder = mocov3_vit.vit_large()
                ckpt = torch.load(f'./ckpts/mocov3_vit{model_config}.pth')
                state_dict = fix_mocov3_state_dict(ckpt['state_dict'])
                del encoder.head
                encoder.load_state_dict(state_dict, strict=True)
                encoder.head = torch.nn.Identity()
            elif architecture == 'resnet':
                raise NotImplementedError()
 
            encoder = encoder.to(device)
            encoder.eval()

        elif 'dinov2' in encoder_type:
            import timm
            # Load DINOv2 from local weights (no torch.hub to avoid race conditions)
            if 'reg' in encoder_type:
                model_name = 'dinov2_vitb14_reg' if model_config == 'b' else \
                             'dinov2_vitl14_reg' if model_config == 'l' else \
                             'dinov2_vitg14_reg'
            else:
                model_name = 'dinov2_vitb14' if model_config == 'b' else \
                             'dinov2_vitl14' if model_config == 'l' else \
                             'dinov2_vitg14'
            
            # Load from local weights file
            # REPA feeds DINOv2 at (resolution // 256) * 224 (i.e. 224 for res=256, 448 for res=512)
            dinov2_input = 224 if resolution == 256 else 448
            encoder = _load_dinov2_from_weights(model_name, img_size=dinov2_input)
            del encoder.head
            # patch_resolution = 16 * (resolution // 256)
            # encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            #     encoder.pos_embed.data, [patch_resolution, patch_resolution],
            # )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()
        
        elif 'dinov1' == encoder_type:
            import timm
            from models import dinov1
            encoder = dinov1.vit_base()
            ckpt =  torch.load(f'./ckpts/dinov1_vit{model_config}.pth') 
            if 'pos_embed' in ckpt.keys():
                ckpt['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    ckpt['pos_embed'], [16, 16],
                )
            del encoder.head
            encoder.head = torch.nn.Identity()
            encoder.load_state_dict(ckpt, strict=True)
            encoder = encoder.to(device)
            encoder.forward_features = encoder.forward
            encoder.eval()

        elif encoder_type == 'clip':
            import clip
            from models.clip_vit import UpdatedVisionTransformer
            encoder_ = clip.load(f"ViT-{model_config}/14", device='cpu')[0].visual
            encoder = UpdatedVisionTransformer(encoder_).to(device)
             #.to(device)
            encoder.embed_dim = encoder.model.transformer.width
            encoder.forward_features = encoder.forward
            encoder.eval()
        
        elif encoder_type == 'mae':
            from models.mae_vit import vit_large_patch16
            import timm
            kwargs = dict(img_size=256)
            encoder = vit_large_patch16(**kwargs).to(device)
            with open(f"ckpts/mae_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f)
            if 'pos_embed' in state_dict["model"].keys():
                state_dict["model"]['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    state_dict["model"]['pos_embed'], [16, 16],
                )
            encoder.load_state_dict(state_dict["model"])

            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [16, 16],
            )

        elif encoder_type == 'jepa':
            from models.jepa import vit_huge
            kwargs = dict(img_size=[224, 224], patch_size=14)
            encoder = vit_huge(**kwargs).to(device)
            with open(f"ckpts/ijepa_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f, map_location=device)
            new_state_dict = dict()
            for key, value in state_dict['encoder'].items():
                new_state_dict[key[7:]] = value
            encoder.load_state_dict(new_state_dict)
            encoder.forward_features = encoder.forward

        encoders.append(encoder)
    
    return encoders, encoder_types, architectures


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def load_legacy_checkpoints(state_dict, encoder_depth):
    new_state_dict = dict()
    for key, value in state_dict.items():
        if 'decoder_blocks' in key:
            parts =key.split('.')
            new_idx = int(parts[1]) + encoder_depth
            parts[0] = 'blocks'
            parts[1] = str(new_idx)
            new_key = '.'.join(parts)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict