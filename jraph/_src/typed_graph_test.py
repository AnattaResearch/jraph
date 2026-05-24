# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for jraph._src.typed_graph."""

from absl.testing import absltest
import jax.numpy as jnp
from jraph._src import typed_graph


def _make_edge_set(num_edges):
  return typed_graph.EdgeSet(
      n_edge=jnp.asarray([num_edges]),
      indices=typed_graph.EdgesIndices(
          senders=jnp.arange(num_edges), receivers=jnp.arange(num_edges)),
      features=jnp.ones((num_edges, 2)))


class TypedGraphTest(absltest.TestCase):

  def test_edge_key_by_name_returns_matching_edge_key(self):
    mesh_key = typed_graph.EdgeSetKey("mesh", ("mesh_nodes", "mesh_nodes"))
    grid_key = typed_graph.EdgeSetKey("grid2mesh",
                                      ("grid_nodes", "mesh_nodes"))
    graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=jnp.asarray([1]), features=()),
        nodes={
            "mesh_nodes": typed_graph.NodeSet(
                n_node=jnp.asarray([3]), features=jnp.ones((3, 1))),
            "grid_nodes": typed_graph.NodeSet(
                n_node=jnp.asarray([2]), features=jnp.ones((2, 1))),
        },
        edges={
            mesh_key: _make_edge_set(3),
            grid_key: _make_edge_set(2),
        })

    self.assertEqual(graph.edge_key_by_name("grid2mesh"), grid_key)

  def test_edge_by_name_returns_matching_edge_set(self):
    edge_key = typed_graph.EdgeSetKey("mesh", ("mesh_nodes", "mesh_nodes"))
    edge_set = _make_edge_set(3)
    graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=jnp.asarray([1]), features=()),
        nodes={
            "mesh_nodes": typed_graph.NodeSet(
                n_node=jnp.asarray([3]), features=jnp.ones((3, 1))),
        },
        edges={edge_key: edge_set})

    self.assertIs(graph.edge_by_name("mesh"), edge_set)

  def test_edge_key_by_name_raises_for_missing_name(self):
    edge_key = typed_graph.EdgeSetKey("mesh", ("mesh_nodes", "mesh_nodes"))
    graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=jnp.asarray([1]), features=()),
        nodes={},
        edges={edge_key: _make_edge_set(1)})

    with self.assertRaisesRegex(KeyError, "invalid edge key 'missing'"):
      graph.edge_key_by_name("missing")

  def test_edge_key_by_name_raises_for_ambiguous_name(self):
    mesh_key = typed_graph.EdgeSetKey("shared_name",
                                      ("mesh_nodes", "mesh_nodes"))
    grid_key = typed_graph.EdgeSetKey("shared_name",
                                      ("grid_nodes", "mesh_nodes"))
    graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=jnp.asarray([1]), features=()),
        nodes={},
        edges={
            mesh_key: _make_edge_set(1),
            grid_key: _make_edge_set(1),
        })

    with self.assertRaisesRegex(KeyError, "invalid edge key 'shared_name'"):
      graph.edge_key_by_name("shared_name")


if __name__ == "__main__":
  absltest.main()
