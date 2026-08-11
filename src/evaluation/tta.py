import torch


@torch.no_grad()
def predict_with_tta(model, inputs, use_tta: bool = True):
    """
    Test-time augmentation via horizontal-flip averaging.

    Fundus images have no canonical left/right orientation (unlike the
    rotate/flip augmentations used at train time, a horizontal flip at test
    time is a "free" second view of the same lesion patterns), so averaging
    softmax probabilities over the original and flipped image is a standard,
    cheap way to reduce prediction variance without any retraining. Typically
    worth ~0.5-2 QWK points on this kind of task.

    Returns: (probs, preds) — probs is the averaged softmax, preds is argmax.
    """
    model.eval()
    outputs = model(inputs)
    probs = torch.softmax(outputs, dim=1)

    if use_tta:
        flipped = torch.flip(inputs, dims=[3])  # horizontal flip (W axis)
        outputs_flipped = model(flipped)
        probs_flipped = torch.softmax(outputs_flipped, dim=1)
        probs = (probs + probs_flipped) / 2.0

    preds = torch.argmax(probs, dim=1)
    return probs, preds
