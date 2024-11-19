import mlx.core as mx
import mlx.nn as nn
def length_masked_ce_loss(model, inputs, targets, mask):
  # Run model on inputs
  logits = model(inputs).astype(mx.float32)

  # Calculate the loss
  ce = nn.losses.cross_entropy(logits, targets) * mask
  loss = ce.sum() / mask.sum()
  return loss

#Naive intermediate layer loss, where we replace the targets with gradients and just multiply the output by the gradients to derive the loss. This is naive and may warrant some further iteration, but will do the job for now
def back_gradient_loss(model, inputs, gradients, mask):
  out = model(inputs)
  logits = (out.sum(axis=-1) * mask)
  loss = (logits * gradients).sum() / mask.sum()
  return loss

loss_fns = {
  "back_gradient": back_gradient_loss,
  "length_masked_ce": length_masked_ce_loss,
}
