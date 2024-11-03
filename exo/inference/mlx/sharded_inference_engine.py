import numpy as np
import mlx.core as mx
from ..inference_engine import InferenceEngine
from .sharded_model import StatefulShardedModel
from .sharded_utils import load_shard, get_image_from_str
from ..shard import Shard
from typing import Optional
from exo.download.shard_download import ShardDownloader
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
def sample_from_logits(logits: mx.array, temp, top_p, logit_bias: Optional[Dict[int, float]] = None) -> Tuple[mx.array, float]:
  if logit_bias:
    indices = mx.array(list(logit_bias.keys()))
    values = mx.array(list(logit_bias.values()))
    logits[:, indices] += values

  if temp == 0:
    token = mx.argmax(logits, axis=-1)
  else:
    if top_p > 0 and top_p < 1.0:
      token = top_p_sampling(logits, top_p, temp)
    else:
      token = mx.random.categorical(logits*(1/temp))

  return token

def masked_ce_from_logits(logits, targets, lengths):
  # Mask padding tokens
  length_mask = mx.arange(logits.shape[1])[None, :] < lengths[:, None]

  # Calculate the loss
  ce = nn.losses.cross_entropy(logits, targets) * length_mask
  ntoks = length_mask.sum()
  return ce.sum() / ntoks, ntoks

def ce_loss(model, inputs, targets, lengths):
  # Run model on inputs
  logits, _ = model(inputs)
  logits = logits.astype(mx.float32)
  return masked_ce_from_logits(logits, targets, lengths):


class MLXDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.executor = ThreadPoolExecutor(max_workers=1)

  async def sample(self, tensor, temp: float = 0.0, top_p: float = 1.0, logit_bias: Optional[Dict[int, float]] = None)
    loop = asyncio.get_running_loop()
    logits = tensor[:, -1, :]
    return sample_from_logits(logits, temp, top_p, logit_bias=logit_bias)

  async def encode_prompt(self, prompt: str):
    loop = asyncio.get_running_loop()
    tokens = await loop.run_in_executor(self.executor, self.tokenizer.encode, prompt)
    return tokens

  async def evaluate_batch(self, batch, metric=masked_ce_from_logits):
    loop = asyncio.get_running_loop()
    return metric(*batch)

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    loop = asyncio.get_running_loop()
    if image_str:
      image = await get_image_from_str(image_str)
      tokenize = partial(self.tokenizer, prompt, image, return_tensors="np")
      inputs = await loop.run_in_executor(self.executor, tokenize)
      pixel_values = mx.array(inputs["pixel_values"])
      input_ids = mx.array(inputs["input_ids"])
      output_data: np.ndarray = np.array(await loop.run_in_executor(self.executor, self.stateful_sharded_model.step, request_id, input_ids, pixel_values))
    else:
      input_ids = mx.array(await loop.run_in_executor(self.executor, self.tokenizer.encode, prompt))
      output_data: np.ndarray = np.array(await loop.run_in_executor(self.executor, self.stateful_sharded_model.step, request_id, input_ids))
    return output_data

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    output_data: np.ndarray = np.array(await asyncio.get_running_loop().run_in_executor(self.executor, self.stateful_sharded_model.step, request_id, mx.array(input_data)))
    return output_data

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return

    model_path = await self.shard_downloader.ensure_shard(shard)

    if self.shard != shard:
      loop = asyncio.get_running_loop()

      def load_shard_wrapper():
        return asyncio.run(load_shard(model_path, shard))

      model_shard, self.tokenizer = await loop.run_in_executor(self.executor, load_shard_wrapper)
      self.stateful_sharded_model = await loop.run_in_executor(self.executor, StatefulShardedModel, shard, model_shard)
      self.shard = shard
