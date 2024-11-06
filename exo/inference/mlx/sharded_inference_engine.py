import numpy as np
import mlx.core as mx
import mlx.nn as nn
from ..inference_engine import InferenceEngine
from .sharded_model import StatefulModel
from .sharded_utils import load_shard, get_image_from_str
from .losses import masked_ce_from_logits
from ..shard import Shard
from typing import Dict, Optional, Tuple
from exo.download.shard_download import ShardDownloader
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
def sample_logits(
  logits: mx.array,
  temp: float = 0.0,
  top_p: float = 1.0,
  logit_bias: Optional[Dict[int, float]] = None
) -> Tuple[mx.array, float]:
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

class MLXDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.executor = ThreadPoolExecutor(max_workers=1)

  def eval_metric(self, outputs, targets, lengths, metric=masked_ce_from_logits):
    x = mx.array(outputs)
    y = mx.array(targets)
    l = mx.array([lengths])
    loss, toks = metric(x, y, l)
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

  async def decode(self, shard: Shard, tokens):
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.decode, tokens)
    return tokens
    
  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[str] = None) -> (np.ndarray, bool):
    output_data = await self.infer_tensor(request_id, shard, await self.encode(shard, prompt), inference_state)
    return output_data 

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> (np.ndarray, bool):
    await self.ensure_shard(shard)
    output_data: np.ndarray = np.array(await asyncio.get_running_loop().run_in_executor(self.executor, self.model, mx.array(input_data), request_id))
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
      self.shard = shard
      self.model = await loop.run_in_executor(self.executor, StatefulModel, model_shard) 
