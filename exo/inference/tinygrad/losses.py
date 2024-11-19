from tinygrad import Tensor, dtypes
import numpy as np
def length_masked_ce_loss(model, inputs, targets, lengths):
  # Run model on inputs
  logits = model(inputs).cast(dtypes.float32).contiguous()

  #Calculate length mask
  length_mask = Tensor.arange(inputs.shape[1])[None, :] < lengths[:, None]
  
  # Calculate the loss
  ce = logits.sparse_categorical_crossentropy(Tensor(targets, requires_grad=False)).mul(length_mask)
  loss = ce.sum() / length_mask.sum()
  return loss

def back_gradient_loss(model, inputs, gradients, lengths):
  out = model(inputs).cast(dtypes.float32).contiguous()
  
  #Calculate length mask
  length_mask = Tensor.arange(inputs.shape[1])[None, :] < lengths[:, None]

  approximation = (out.sum(axis=-1) * length_mask * gradients)
  loss = approximation.sum() / length_mask.sum()
  return loss

loss_fns = {
  "back_gradient": back_gradient_loss,
  "length_masked_ce": length_masked_ce_loss,
}
