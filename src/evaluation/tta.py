import torch
import torch.nn.functional as F


@torch.no_grad()
def predict_with_tta(model, inputs, use_tta: bool = True,
                     use_flips: bool = True, use_rotations: bool = True):
    """
    Multi-crop Test-Time Augmentation.

    Averages softmax probabilities over multiple augmented views:
      - Original image
      - Horizontal flip
      - Vertical flip
      - 4 rotations (0°, 90°, 180°, 270°)

    For fundus images, all these transforms are valid since there is no
    canonical orientation. Averaging over multiple views reduces prediction
    variance and typically yields 1-3% accuracy improvement.

    Args:
        model: Trained model (returns dict with 'logits' and 'ordinal_logits')
        inputs: (B, C, H, W) input tensor
        use_tta: Whether to use TTA at all
        use_flips: Include horizontal/vertical flips
        use_rotations: Include 90°/180°/270° rotations

    Returns:
        (probs, preds) — averaged softmax probabilities and argmax predictions
    """
    model.eval()

    # Get base prediction
    outputs = model(inputs)
    probs = _get_probs(outputs)

    if not use_tta:
        preds = torch.argmax(probs, dim=1)
        return probs, preds

    n_views = 1  # original view

    # Horizontal flip
    if use_flips:
        flipped_h = torch.flip(inputs, dims=[3])
        out_h = model(flipped_h)
        probs += _get_probs(out_h)
        n_views += 1

    # Vertical flip
    if use_flips:
        flipped_v = torch.flip(inputs, dims=[2])
        out_v = model(flipped_v)
        probs += _get_probs(out_v)
        n_views += 1

    # Rotations (90°, 180°, 270°)
    if use_rotations:
        for k in [1, 2, 3]:
            rotated = torch.rot90(inputs, k, dims=[2, 3])
            out_r = model(rotated)
            probs += _get_probs(out_r)
            n_views += 1

    # Average over all views
    probs = probs / n_views

    preds = torch.argmax(probs, dim=1)
    return probs, preds


def _get_probs(outputs):
    """
    Extract probabilities from model output.
    Combines classification and ordinal head probabilities.
    """
    if isinstance(outputs, dict):
        logits = outputs['logits']
        ordinal_logits = outputs.get('ordinal_logits', None)

        cls_probs = F.softmax(logits, dim=1)

        if ordinal_logits is not None:
            from ..models.components import OrdinalRegressionHead
            ord_probs = OrdinalRegressionHead.ordinal_logits_to_class_probs(ordinal_logits)
            return 0.5 * cls_probs + 0.5 * ord_probs
        else:
            return cls_probs
    else:
        return F.softmax(outputs, dim=1)


@torch.no_grad()
def predict_multiscale_tta(model, inputs, scales=(0.9, 1.0, 1.1),
                           use_flips=True, use_rotations=True):
    """
    Multi-scale TTA: runs TTA at multiple scales and averages.

    For each scale, the input is resized to scale * img_size, then back to
    img_size, creating slightly different views.

    Args:
        model: Trained model
        inputs: (B, C, H, W) input tensor
        scales: List of scale factors
        use_flips: Include flips in TTA
        use_rotations: Include rotations in TTA

    Returns:
        (probs, preds) — averaged softmax probabilities and argmax predictions
    """
    model.eval()
    _, _, H, W = inputs.shape
    all_probs = []

    for scale in scales:
        if scale != 1.0:
            scaled_h, scaled_w = int(H * scale), int(W * scale)
            scaled_inputs = F.interpolate(inputs, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
            scaled_inputs = F.interpolate(scaled_inputs, size=(H, W), mode='bilinear', align_corners=False)
        else:
            scaled_inputs = inputs

        probs, _ = predict_with_tta(
            model, scaled_inputs, use_tta=True,
            use_flips=use_flips, use_rotations=use_rotations
        )
        all_probs.append(probs)

    # Average across scales
    final_probs = torch.stack(all_probs).mean(dim=0)
    preds = torch.argmax(final_probs, dim=1)
    return final_probs, preds
