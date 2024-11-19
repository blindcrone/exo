from tinygrad import Tensor, dtypes
import numpy as np
def length_masked_ce_loss(model, inputs, targets, mask):
  # Run model on inputs
  logits = model(inputs).cast(dtypes.float32).contiguous()

  # Calculate the loss
  ce = logits.sparse_categorical_crossentropy(Tensor(targets, requires_grad=False)).mul(mask)
  loss = ce.sum() / mask.sum()
  return loss

def back_gradient_loss(model, inputs, gradients, mask):
  out = model(inputs).cast(dtypes.float32).contiguous()
  logits = (out.sum(axis=-1) * mask)
  loss = (logits * gradients).sum() / mask.sum()
  return loss

loss_fns = {
  "back_gradient": back_gradient_loss,
  "length_masked_ce": length_masked_ce_loss,
}
