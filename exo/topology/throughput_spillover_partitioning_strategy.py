from typing import List
from .partitioning_strategy import PartitioningStrategy, map_partitions_to_shards
from .topology import Topology
from .partitioning_strategy import Partition
from exo.inference.shard import Shard

class ThroughputSpilloverPartitioningStrategy(PartitioningStrategy):
  def __init__(self):
    self.order = lambda x: (x[1].cores, x[1].memory, x[0])

  def repartition(self, topology: Topology, num_layers: int) -> List[Shard]:
    estimated_layer_size: float = .5 * (1024 ** 3) #Likely overestimated for llama
    occupancy_limit: float = .9 #To be safe
    estimated_model_size = estimated_layer_size * num_layers
    nodes = sorted(list(topology.all_nodes()), key=self.order, reverse=True)
    repartitioned = []
    start = 0
    for node in nodes:
      if start < 1:
        end = min(start + (node[1].memory * occupancy_limit) / estimated_model_size, 1)
        repartitioned.append(Partition(node[0], start, end))
        start = end
    return repartitioned

