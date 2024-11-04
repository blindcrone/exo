import numpy as np
import mlx.core as mx
import mlx.nn as nn
from ..inference_engine import InferenceEngine
from .sharded_model import StatefulShardedModel, sample_logits
from .sharded_utils import load_shard, get_image_from_str
from ..shard import Shard
from typing import Optional
from exo.download.shard_download import ShardDownloader
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
def masked_ce_from_logits(logits, targets, lengths):
  # Mask padding tokens
  length_mask = mx.arange(logits.shape[1])[None, :] < lengths[:, None]

  # Calculate the loss
  ce = nn.losses.cross_entropy(logits, targets) * length_mask
  ntoks = length_mask.sum()
  return ce.sum() / ntoks, ntoks

class MLXDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.executor = ThreadPoolExecutor(max_workers=1)

  def eval_metric(self, outputs, targets, lengths):
    x = mx.array(outputs[:lengths-1])
    y = x
    y = mx.array(targets)
    l = mx.array([lengths])
    loss, toks = masked_ce_from_logits(x, y, l)
    return np.array(loss), toks

  async def sample(self, x):
    y = mx.array(x)
    logits = y[:, -1, :]
    y = np.array(sample_logits(logits))
    return y

  async def encode(self, shard: Shard, prompt: str):
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.encode, prompt)
    return tokens
    
  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[str] = None) -> (np.ndarray, bool):
    output_data = await self.infer_tensor(request_id, shard, await self.encode(shard, prompt), inference_state)
    return output_data 

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> (np.ndarray, bool):
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
