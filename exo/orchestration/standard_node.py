import numpy as np
import json
import asyncio
import uuid
import time
import traceback
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Set
from exo.networking import Discovery, PeerHandle, Server
from exo.inference.inference_engine import InferenceEngine, Shard
from .node import Node
from exo.topology.topology import Topology
from exo.topology.device_capabilities import device_capabilities
from exo.topology.partitioning_strategy import Partition, PartitioningStrategy, map_partitions_to_shards
from exo import DEBUG
from exo.helpers import AsyncCallbackSystem
from exo.viz.topology_viz import TopologyViz
from exo.download.hf.hf_helpers import RepoProgressEvent
from exo.inference.inference_engine import get_inference_engine, InferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader


class StandardNode(Node):
  def __init__(
    self,
    _id: str,
    server: Server,
    inference_engine: InferenceEngine,
    discovery: Discovery,
    partitioning_strategy: PartitioningStrategy = None,
    max_generate_tokens: int = 1024,
    topology_viz: Optional[TopologyViz] = None,
    shard_downloader: Optional[HFShardDownloader] = None,
  ):
    self.id = _id
    self.inference_engine = inference_engine
    self.server = server
    self.discovery = discovery
    self.partitioning_strategy = partitioning_strategy
    self.peers: List[PeerHandle] = {}
    self.topology: Topology = Topology()
    self.device_capabilities = device_capabilities()
    self.buffered_token_output: Dict[str, Tuple[List[int], bool]] = {}
    self.request_raw_output: Dict[str, Tuple[List[np.ndarray], bool]] = {}
    self.request_target: Dict[str, Tuple[np.ndarray, int]] = {}
    
    self.max_generate_tokens = max_generate_tokens
    self.topology_viz = topology_viz
    self._on_token = AsyncCallbackSystem[str, Tuple[str, List[int], bool]]()
    self._on_opaque_status = AsyncCallbackSystem[str, Tuple[str, str]]()
    self._on_opaque_status.register("node_status").on_next(self.on_node_status)
    self.node_download_progress: Dict[str, RepoProgressEvent] = {}
    self.topology_inference_engines_pool: List[List[str]] = []
    self.shard_downloader = shard_downloader

  async def start(self, wait_for_peers: int = 0) -> None:
    await self.server.start()
    await self.discovery.start()
    await self.update_peers(wait_for_peers)
    await self.collect_topology()
    if DEBUG >= 2: print(f"Collected topology: {self.topology}")
    asyncio.create_task(self.periodic_topology_collection(1.0))

  async def stop(self) -> None:
    await self.discovery.stop()
    await self.server.stop()

  def on_node_status(self, request_id, opaque_status):
    try:
      status_data = json.loads(opaque_status)
      if status_data.get("type", "") == "supported_inference_engines":
        node_id = status_data.get("node_id")
        engines = status_data.get("engines", [])
        self.topology_inference_engines_pool.append(engines)
      if status_data.get("type", "") == "node_status":
        if status_data.get("status", "").startswith("start_"):
          self.current_topology.active_node_id = status_data.get("node_id")
        elif status_data.get("status", "").startswith("end_"):
          if status_data.get("node_id") == self.current_topology.active_node_id:
            self.current_topology.active_node_id = None
      download_progress = None
      if status_data.get("type", "") == "download_progress":
        if DEBUG >= 8: print(f"Download progress from {status_data.get('node_id')}: {status_data.get('progress')}")
        download_progress = RepoProgressEvent.from_dict(status_data.get('progress'))
        self.node_download_progress[status_data.get('node_id')] = download_progress
      if self.topology_viz:
        self.topology_viz.update_visualization(self.current_topology, self.partitioning_strategy.partition(self.current_topology), self.id, self.node_download_progress)
    except Exception as e:
      if DEBUG >= 1: print(f"Error updating visualization: {e}")
      if DEBUG >= 1: traceback.print_exc()

  def get_supported_inference_engines(self):
    supported_engine_names = []
    if self.inference_engine.__class__.__name__ == 'MLXDynamicShardInferenceEngine':
      supported_engine_names.append('mlx')
      supported_engine_names.append('tinygrad')
    else:
      supported_engine_names.append('tinygrad')
    return supported_engine_names

  async def broadcast_supported_engines(self, supported_engines_names: List[str]):
    status_message = json.dumps({"type": "supported_inference_engines", "node_id": self.id, "engines": supported_engines_names})
    await self.broadcast_opaque_status("", status_message)

  def get_topology_inference_engines(self) -> List[List[str]]:
    return self.topology_inference_engines_pool

  async def process_prompt(self, base_shard: Shard, prompt: str, image_str: Optional[str] = None, request_id: Optional[str] = None, inference_state: Optional[str] = None) -> Optional[np.ndarray]:
    shard = self.get_current_shard(base_shard)
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "start_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "prompt": prompt,
          "image_str": image_str,
          "inference_state": inference_state,
          "request_id": request_id,
        }),
      )
    )
    start_time = time.perf_counter_ns()
    resp = await self._process_prompt(base_shard, prompt, image_str, request_id, inference_state)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "end_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "prompt": prompt,
          "image_str": image_str,
          "inference_state": inference_state,
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
          "result_size": resp.size if resp is not None else 0,
        }),
      )
    )
    return resp

  async def evaluate_prompt(self, base_shard: Shard, prompt: str, request_id: str):
    while True:
      if request_id not in self.request_target:
        tokens = await self.inference_engine.encode(prompt)
        length = len(tokens)
        self.request_target[request_id] = (tokens[1:], length)
      if request_id in self.request_raw_output and self.request_raw_output[1] == True:
        batch = ( np.array([np.array(self.request_raw_output[request_id][0][:-1])])
                , np.array([self.request_target[request_id][0]])
                , [self.request_target[request_id][1]])
        score = await node.inference_engine.evaluate_batch(batch)
        return score
      else:
        await self.process_prompt(self, base_shard, request_id=request_id)
  
  async def _process_prompt(self, base_shard: Shard, prompt: str, image_str: Optional[str] = None, request_id: Optional[str] = None, inference_state: Optional[str] = None, target: Optional[str] = None) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    if request_id not in self.buffered_token_output:
      self.buffered_token_output[request_id] = ([], False)
    if request_id not in self.request_raw_output:
      self.request_raw_output[request_id] = ([], False)
    shard = self.get_current_shard(base_shard)

    if DEBUG >= 2: print(f"[{request_id}] process prompt: {base_shard=} {shard=} {prompt=} {image_str=}")
    if not shard.is_first_layer():
      if DEBUG >= 2: print(f"[{request_id}] forwarding to next shard: {base_shard=} {shard=} {prompt=} {image_str=}")
      await self.forward_to_next_shard(shard, prompt, request_id, image_str=image_str, inference_state=inference_state)
      return

    result = await self.inference_engine.infer_prompt(request_id, shard, prompt, image_str, inference_state=inference_state)
    self.request_raw_output[request_id][0].append(result)
    
    is_finished = False
    if shard.is_last_layer():  # we got a new token out
      token = await self.inference_engine.sample(result).item()
      self.buffered_token_output[request_id][0].append(token)
      is_finished = token == self.inference_engine.tokenizer.eos_token_id or len(self.buffered_token_output[request_id][0]) >= self.max_generate_tokens
      if is_finished:
        self.buffered_token_output[request_id] = (self.buffered_token_output[request_id][0], True)
        self.request_raw_output[request_id] = (self.buffered_token_output[request_id][0], True)
      self.trigger_on_token_callbacks(request_id, self.buffered_token_output[request_id][0], is_finished)

    asyncio.create_task(self.broadcast_result(request_id, self.buffered_token_output[request_id][0], is_finished))  # TODO: this is n^2 communication complexity


    if DEBUG >= 2: print(f"[{request_id}] result size: {result.size}, is finished: {is_finished}, buffered tokens: {len(self.buffered_token_output[request_id][0])}")

    if not is_finished:
      asyncio.create_task(self.forward_to_next_shard(shard, result, request_id, image_str=image_str, inference_state=inference_state))

    return np.array(self.buffered_token_output[request_id][0]) if len(self.buffered_token_output[request_id][0]) > 0 else None

  async def process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    shard = self.get_current_shard(base_shard)
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "start_process_tensor",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "tensor_size": tensor.size,
          "tensor_shape": tensor.shape,
          "request_id": request_id,
          "inference_state": inference_state,
        }),
      )
    )
    start_time = time.perf_counter_ns()
    resp = await self._process_tensor(shard, tensor, request_id, inference_state)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "end_process_tensor",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
          "result_size": resp.size if resp is not None else 0,
        }),
      )
    )
    return resp

  async def _process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    if request_id not in self.buffered_token_output:
      self.buffered_token_output[request_id] = ([], False)
    if request_id not in self.request_raw_output:
      self.request_raw_output[request_id] = ([], False)
    shard = self.get_current_shard(base_shard)

    try:
      if DEBUG >= 1: print(f"[{request_id}] process_tensor: {tensor.size=} {tensor.shape=}")
      result = await self.inference_engine.infer_prompt(request_id, shard, prompt, image_str, inference_state=inference_state)
      self.request_raw_output[request_id][0].append(result)
      
      is_finished = False
      if shard.is_last_layer():  # we got a new token out
        token = await self.inference_engine.sample(result).item()
        self.buffered_token_output[request_id][0].append(token)
        is_finished = token == self.inference_engine.tokenizer.eos_token_id or len(self.buffered_token_output[request_id][0]) >= self.max_generate_tokens
        if is_finished:
          self.buffered_token_output[request_id] = (self.buffered_token_output[request_id][0], True)
          self.request_raw_output[request_id] = (self.buffered_token_output[request_id][0], True)
          
        self.trigger_on_token_callbacks(request_id, self.buffered_token_output[request_id][0], is_finished)
      if DEBUG >= 2: print(f"[{request_id}] result size: {result.size}, is finished: {is_finished}, buffered tokens: {len(self.buffered_token_output[request_id][0])}")

      if not is_finished:
        asyncio.create_task(self.forward_to_next_shard(shard, result, request_id, inference_state=inference_state))

      return np.array(self.buffered_token_output[request_id][0]) if len(self.buffered_token_output[request_id][0]) > 0 else None
    except Exception as e:
      print(f"Error processing tensor for shard {shard}: {e}")
      traceback.print_exc()
      return None

  async def forward_to_next_shard(
    self,
    base_shard: Shard,
    tensor_or_prompt: Union[np.ndarray, str],
    request_id: str,
    image_str: Optional[str] = None,
    inference_state: Optional[str] = None,
  ) -> None:
    if not self.partitioning_strategy:
      if DEBUG >= 1: print("No partitioning strategy found. Skipping forward.")
      return
    shard = self.get_current_shard(base_shard)

    partitions = self.partitioning_strategy.partition(self.topology)
    shards = map_partitions_to_shards(self.partitioning_strategy.partition(self.topology), base_shard.n_layers, base_shard.model_id)
    current_partition_index = next((i for i, p in enumerate(partitions) if p.node_id == self.id), None)
    if DEBUG >= 1: print(f"Current partition index: {current_partition_index}")
    if current_partition_index is not None:
      next_partition_index = (current_partition_index+1) % len(partitions)
      next_partition: Partition = partitions[next_partition_index]
      next_shard = shards[next_partition_index]
      if DEBUG >= 2: print(f"Computed next from: {shard}, {self.topology}. Next partition: {next_partition}")

      if next_partition.node_id == self.id:
        if isinstance(tensor_or_prompt, np.ndarray):
          await self.process_tensor(shard, tensor_or_prompt, request_id, inference_state=inference_state)
        else:
          await self.process_prompt(shard, tensor_or_prompt, image_str, request_id, inference_state=inference_state)
        return

      target_peer = next((p for p in self.peers if p.id() == next_partition.node_id), None)
      if not target_peer:
        raise ValueError(f"Peer for {next_partition} not found")

      if DEBUG >= 1: print(f"Sending tensor_or_prompt to {target_peer.id()}: {tensor_or_prompt}")

      if isinstance(tensor_or_prompt, np.ndarray):
        await target_peer.send_tensor(next_shard, tensor_or_prompt, request_id=request_id, inference_state=inference_state)
      else:
        await target_peer.send_prompt(next_shard, tensor_or_prompt, image_str=image_str, request_id=request_id, inference_state=inference_state)

  def get_current_shard(self, base_shard: Shard) -> Shard:
    partitions = self.partitioning_strategy.partition(self.topology)
    shards = map_partitions_to_shards(partitions, base_shard.n_layers, base_shard.model_id)
    current_partition_index = next((i for i, p in enumerate(partitions) if p.node_id == self.id), None)
    if current_partition_index is None:
      raise ValueError(f"No current partition found for node: {self.id}")
    return shards[current_partition_index]

  async def update_peers(self, wait_for_peers: int = 0) -> bool:
    next_peers = await self.discovery.discover_peers(wait_for_peers)
    current_peer_ids = {peer.id() for peer in self.peers}
    next_peer_ids = {peer.id() for peer in next_peers}
    peers_added = [peer for peer in next_peers if peer.id() not in current_peer_ids]
    peers_removed = [peer for peer in self.peers if peer.id() not in next_peer_ids]
    peers_updated = [peer for peer in next_peers if peer.id() in current_peer_ids and any(p.addr() != peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_unchanged = [peer for peer in next_peers if peer.id() in current_peer_ids and all(p.addr() == peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_to_disconnect = [peer for peer in peers_removed if await peer.is_connected()]
    peers_to_connect = [peer for peer in peers_added + peers_updated + peers_unchanged if not await peer.is_connected()]

    def _pretty(peers: List[PeerHandle]) -> List[str]:
      return [f"{peer.id()}@{peer.addr()}" for peer in peers]

    if DEBUG >= 2:
      print(f"update_peers: added={peers_added} removed={peers_removed} updated={peers_updated} unchanged={peers_unchanged} to_disconnect={peers_to_disconnect} to_connect={peers_to_connect}")

    async def disconnect_with_timeout(peer, timeout=5):
      try:
        await asyncio.wait_for(peer.disconnect(), timeout)
        return True
      except Exception as e:
        print(f"Error disconnecting peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    async def connect_with_timeout(peer, timeout=5):
      try:
        await asyncio.wait_for(peer.connect(), timeout)
        return True
      except Exception as e:
        print(f"Error connecting peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    disconnect_results = await asyncio.gather(*(disconnect_with_timeout(peer) for peer in peers_to_disconnect), return_exceptions=True)
    connect_results = await asyncio.gather(*(connect_with_timeout(peer) for peer in peers_to_connect), return_exceptions=True)

    successful_disconnects = [peer for peer, result in zip(peers_to_disconnect, disconnect_results) if result is True]
    failed_disconnects = [peer for peer, result in zip(peers_to_disconnect, disconnect_results) if result is False]
    successful_connects = [peer for peer, result in zip(peers_to_connect, connect_results) if result is True]
    failed_connects = [peer for peer, result in zip(peers_to_connect, connect_results) if result is False]
    if DEBUG >= 1:
      if successful_disconnects: print(f"Successfully disconnected peers: {_pretty(successful_disconnects)}")
      if failed_disconnects: print(f"Failed to disconnect peers: {_pretty(failed_disconnects)}")
      if successful_connects: print(f"Successfully connected peers: {_pretty(successful_connects)}")
      if failed_connects: print(f"Failed to connect peers: {_pretty(failed_connects)}")

    self.peers = next_peers
    return len(peers_added) > 0 or len(peers_removed) > 0 or len(peers_updated) > 0

  async def select_best_inference_engine(self):
    supported_engines = self.get_supported_inference_engines()
    await self.broadcast_supported_engines(supported_engines)
    if len(self.get_topology_inference_engines()):
      if any(len(engines) == 1 and "tinygrad" in engines for engines in self.get_topology_inference_engines()):
        if DEBUG >= 1: print("Found node with only tinygrad, using tinygrad on all nodes")
        self.inference_engine = get_inference_engine("tinygrad", self.shard_downloader)
      else:
        if DEBUG >= 1: print("All nodes can use mlx, using mlx for inference")
        self.inference_engine = get_inference_engine("mlx", self.shard_downloader)

  async def periodic_topology_collection(self, interval: int):
    while True:
      await asyncio.sleep(interval)
      try:
        did_peers_change = await self.update_peers()
        if DEBUG >= 2: print(f"{did_peers_change=}")
        if did_peers_change:
          await self.collect_topology()
          await self.select_best_inference_engine()
      except Exception as e:
        print(f"Error collecting topology: {e}")
        traceback.print_exc()
  
  async def get_inference_result(self, request_id: str) -> Tuple[Optional[np.ndarray], bool]:
    if request_id not in self.buffered_token_output:
      return None, False
    return np.array(self.buffered_token_output[request_id][0]), self.buffered_token_output[request_id][1]

  async def collect_topology(self, visited: set[str] = set(), max_depth: int = 4) -> Topology:
    next_topology = Topology()
    next_topology.update_node(self.id, self.device_capabilities)

    if DEBUG >= 2: print(f"Collecting topology {max_depth=} {visited=}")

    prev_visited = visited.copy()
    visited.add(self.id)
    visited.update(p.id() for p in self.peers)

    for peer in self.peers:
      next_topology.update_node(peer.id(), peer.device_capabilities())
      next_topology.add_edge(self.id, peer.id())

      if peer.id() in prev_visited:
        continue

      if max_depth <= 0:
        if DEBUG >= 2: print("Max depth reached. Skipping...")
        continue

      try:
        other_topology = await asyncio.wait_for(peer.collect_topology(visited, max_depth=max_depth - 1), timeout=5.0)
        if DEBUG >= 2: print(f"Collected topology from: {peer.id()}: {other_topology}")
        self.topology.merge(other_topology)
      except Exception as e:
        print(f"Error collecting topology from {peer.id()}: {e}")
        traceback.print_exc()

    next_topology.active_node_id = self.topology.active_node_id  # this is not so clean.
    self.topology = next_topology
    if self.topology_viz:
      self.topology_viz.update_visualization(self.current_topology, self.partitioning_strategy.partition(self.current_topology), self.id)
    return next_topology

  @property
  def on_token(self) -> AsyncCallbackSystem[str, Tuple[str, List[int], bool]]:
    return self._on_token

  @property
  def on_opaque_status(self) -> AsyncCallbackSystem[str, Tuple[str, str]]:
    return self._on_opaque_status

  def trigger_on_token_callbacks(self, request_id: str, tokens: List[int], is_finished: bool) -> None:
    if DEBUG >= 2: print(f"Triggering all on_token callbacks with {request_id=} num_tokens={len(tokens)} {is_finished=}")
    self.on_token.trigger_all(request_id, tokens, is_finished)

  async def broadcast_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
    async def send_result_to_peer(peer):
      try:
        await asyncio.wait_for(peer.send_result(request_id, result, is_finished), timeout=15.0)
      except asyncio.TimeoutError:
        print(f"Timeout broadcasting result to {peer.id()}")
      except Exception as e:
        print(f"Error broadcasting result to {peer.id()}: {e}")
        traceback.print_exc()

    await asyncio.gather(*[send_result_to_peer(peer) for peer in self.peers], return_exceptions=True)

  async def broadcast_opaque_status(self, request_id: str, status: str) -> None:
    if DEBUG >= 8: print(f"Broadcasting opaque status: {request_id=} {status=}")

    async def send_status_to_peer(peer):
      try:
        await asyncio.wait_for(peer.send_opaque_status(request_id, status), timeout=15.0)
      except asyncio.TimeoutError:
        print(f"Timeout sending opaque status to {peer.id()}")
      except Exception as e:
        print(f"Error sending opaque status to {peer.id()}: {e}")
        traceback.print_exc()

    await asyncio.gather(*[send_status_to_peer(peer) for peer in self.peers], return_exceptions=True)
    # in the case of opaque status, we also want to receive our own opaque statuses
    self.on_opaque_status.trigger_all(request_id, status)

  @property
  def current_topology(self) -> Topology:
    return self.topology
