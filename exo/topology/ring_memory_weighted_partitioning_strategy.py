from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition
from exo.inference.shard import Shard
from .partitioning_strategy import PartitioningStrategy, map_partitions_to_shards

class RingMemoryWeightedPartitioningStrategy(PartitioningStrategy):
  def __init__(self):
    self.order = lambda x: (x[1].memory, x[0])

  def repartition(self, topology: Topology, num_layers: int) -> List[Shard]:
    return self.partition(topology)

