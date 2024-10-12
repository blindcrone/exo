import unittest
from exo.topology.core_memory_weighted_partitioning_strategy import CoreMemoryWeightedPartitioningStrategy
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.partitioning_strategy import Partition


class TestThroughputSpilloverPartitioningStrategy(unittest.TestCase):
  def test_partition(self):
    # triangle
    # node1 -> node2 -> node3 -> node1
    gig = 1024 ** 3
    n_layers = 12
    estimated_size = .5 * gig * n_layers
    topology = Topology()
    topology.update_node(
      "node1",
      DeviceCapabilities(model="test1", chip="test1", memory=6 * gig, flops=DeviceFlops(fp32=0, fp16=0, int8=0), cores=4096),
    )
    topology.update_node(
      "node2",
      DeviceCapabilities(model="test2", chip="test2", memory=3 * gig, flops=DeviceFlops(fp32=0, fp16=0, int8=0), cores=128),
    )
    topology.update_node(
      "node3",
      DeviceCapabilities(model="test3", chip="test3", memory=3 * gig, flops=DeviceFlops(fp32=0, fp16=0, int8=0), cores=8),
    )
    topology.update_node(
      "node4",
      DeviceCapabilities(model="test4", chip="test4", memory=3 * gig, flops=DeviceFlops(fp32=0, fp16=0, int8=0), cores=2),
    )

    topology.add_edge("node1", "node2")
    topology.add_edge("node2", "node3")
    topology.add_edge("node3", "node4")
    topology.add_edge("node4", "node1")
    topology.add_edge("node1", "node4")

    strategy = ThroughputSpilloverPartitioningStrategy()
    partitions = strategy.partition(topology)
    repartition = strategy.repartition(topology, n_layers

    self.assertEqual(len(partitions), 4)
    self.assertEqual(
      partitions,
      [
        Partition("node1", 0.0, 0.4),
        Partition("node2", 0.4, 0.6),
        Partition("node3", 0.6, 0.8),
        Partition("node3", 0.8, 1.0),
      ],
    )

    self.assertEqual(len(repartition), 1)
    self.assertEqual(
      partitions,
      [
        Partition("node1", 0.0, 0.9),
        Partition("node2", 0.9, 1),
      ],
    )

if __name__ == "__main__":
  unittest.main()
