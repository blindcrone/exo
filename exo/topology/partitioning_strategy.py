from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
from .topology import Topology
from exo.inference.shard import Shard


# Partitions shard-space into pieces of contiguous shards, represented by floating point range [start, end) between 0 and 1
@dataclass
class Partition:
  node_id: str
  start: float
  end: float

def map_partitions_to_shards(partitions: List[Partition], num_layers: int, model_id: str) -> List[Shard]:
  shards = []
  for i, partition in enumerate(partitions):
    start_layer = int(partition.start*num_layers)
    end_layer = int(partition.end*num_layers) - 1

    # Ensure the last partition covers up to num_layers - 1
    if i == len(partitions) - 1:
      end_layer = num_layers - 1

    # Ensure no empty shards
    if start_layer <= end_layer:
      shards.append(Shard(model_id, start_layer, end_layer, num_layers))

  # Ensure full coverage
  if shards and shards[-1].end_layer < num_layers - 1:
    shards[-1] = Shard(model_id, shards[-1].start_layer, num_layers - 1, num_layers)

  return shards

class PartitioningStrategy(ABC):
  def __init__(self):
    self.order = lambda x: x[0]

  def partition(self, topology: Topology) -> List[Partition]:
    nodes = sorted(list(topology.all_nodes()), key=self.order, reverse=True)
    total_memory = sum(node[1].memory for node in nodes)
    partitions = []
    start = 0
    for node in nodes:
      end = round(start + (node[1].memory/total_memory), 5)
      partitions.append(Partition(node[0], start, end))
      start = end
    return partitions

  @abstractmethod
  def repartition(self, topology: Topology, num_layers: int) -> List[Shard]:
    pass

  def allocate(self, topology: Topology, num_layers: int, model_id: str) -> List[Shard]:
    return map_partitions_to_shards(self.repartition(topology, num_layers), num_layers, model_id)

