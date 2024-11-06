from typing import Dict, Generator, Optional, Tuple
from collections import OrderedDict

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import top_p_sampling

from ..shard import Shard

class StatefulShardedModel:
  def __init__(self, shard: Shard, model: nn.Module, max_kv_size: int = 1024, max_caches: int = 2):
    self.shard = shard
    self.model = model
    self.max_kv_size = max_kv_size
    self.max_caches = max_caches
    self.caches = OrderedDict()

  def __call__(
    self,
    x,
    request_id: str,
  ) -> Generator[Tuple[mx.array, mx.array], None, None]:
    if request_id not in self.caches:
      self.init_cache(request_id)
    else:
      self.caches.move_to_end(request_id)

    cache = self.caches[request_id]

    output = self.model(x, cache=cache)
    return output

  def init_cache(self, request_id: str):
    kv_heads = ([self.model.n_kv_heads]*len(self.model.layers) if isinstance(self.model.n_kv_heads, int) else self.model.n_kv_heads)
    # if self.max_kv_size is not None:
      # cache = [RotatingKVCache(self.model.head_dim, n, max_size=self.max_kv_size, keep=4) for n in kv_heads]
      # cache = [KVCache(self.model.head_dim, n) for n in kv_heads]
    # else:
      # cache = [KVCache(self.model.head_dim, n) for n in kv_heads]
    cache = make_prompt_cache(self.model)

    if len(self.caches) >= self.max_caches:
      self.caches.popitem(last=False)

    self.caches[request_id] = cache
