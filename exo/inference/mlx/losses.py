import mlx.core as mx
import mlx.nn as nn
def masked_ce_from_logits(logits, targets, lengths):
  # Mask padding tokens
  length_mask = mx.arange(logits.shape[1])[None, :] < lengths[:, None]

  # Calculate the loss
  ce = nn.losses.cross_entropy(logits, targets) * length_mask
  ntoks = length_mask.sum()
  return ce.sum() / ntoks, ntoks

def length_masked_ce_loss(model, inputs, targets, lengths):
    # Run model on inputs
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    # Mask padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the loss
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks

