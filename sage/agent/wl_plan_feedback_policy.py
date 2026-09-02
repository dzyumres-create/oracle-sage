"""
.. module:: wl_plan_feedback_policy
   :synopsis: WLPlanFeedbackPolicy - Taxi's meta-controller with a
   Weisfeiler-Leman colour-embedding encoder in place of message-passing,
   AND (per a later supervisor-directed redesign) a discriminator
   (PathValueNet) that reads WL's global histogram output directly,
   instead of running its own independent GNN pass.

   self.gnn_extractor2 (a genuine, independent GNNExtractor - see the
   isinstance-guard note below) and self.a2 are still constructed (the
   shared base class always builds them), but are now UNUSED/inert for
   this policy: _encode_current_state/_encode_projected_state no longer
   call gnn_extractor2 at all. Left in place deliberately rather than
   removed - nothing assumes it's absent, only code that assumed it was
   present-and-GNN-shaped, none of which runs anymore here.

      Note: GNNFeedbackPolicy.__init__ aliases self.gnn_extractor2 =
   self.gnn_extractor whenever shared_gnn=True (graph_feedback_policy.py:
   244-249). This previously had no type check, meaning gnn_extractor2
   would silently become the SAME WLEmbeddingExtractor object as this
   policy's meta-controller encoder under shared_gnn=True - not the
   independent GNN discriminator the design requires. This has since been
   fixed there (an isinstance(self.gnn_extractor, GNNExtractor) check) -
   gnn_extractor2 is now always a genuine, independent GNNExtractor,
   regardless of shared_gnn's value, whenever the meta-controller's
   encoder isn't itself a GNNExtractor. Now moot for the discriminator's
   own data path (see above), but left as-is since other things (self.a2)
   still exist unconditionally in the shared base class regardless.
"""
import json

import torch as th
from torch import nn
from torch_geometric.data import Batch

from sage.agent.graph_policy import EMB_SIZE
from sage.agent.graph_feedback_policy import PathValueNet
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.domains.gym_taxi.utils.wl_vocab_cache import WL_VOCAB_PATH, _decode_signature


def _load_vocab(path) -> dict:
    """
    Loads a frozen WL-colour vocab from an arbitrary `path`, in the same
    signature -> colour id format wl_colours() expects.

    Reuses wl_vocab_cache._decode_signature (the actual JSON encoding
    scheme) rather than re-deriving it. wl_vocab_cache.get_wl_vocab()
    itself isn't reusable here as-is: it's hardcoded to one fixed path
    (WL_VOCAB_PATH) with no path parameter, whereas WLPlanFeedbackPolicy
    needs an arbitrary, CLI-configurable path (--wl-vocab-path). Only this
    thin, path-parameterised loop is new; the encoding scheme is shared.
    """
    with open(path) as f:
        payload = json.load(f)
    vocab = {}
    for entry in payload["entries"]:
        vocab[_decode_signature(entry["signature"])] = entry["id"]
    return vocab


class WLEmbeddingExtractor(nn.Module):
    """
    Meta-controller encoder: a trainable embedding lookup over each node's
    frozen WL colour id, replacing GNN message-passing. Produces
    EMB_SIZE-dim per-node embeddings, feeding action_net exactly as the
    GNN encoder's node output did.
    """

    def __init__(self, vocab_size: int, emb_size: int = EMB_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)

    def forward(self, wl_colours: th.Tensor) -> th.Tensor:
        return self.embedding(wl_colours)


class WLPlanFeedbackPolicy(GNNPlanFeedbackPolicy):
    """
    GNNPlanFeedbackPolicy with the meta-controller's encoder replaced by a
    WL-colour embedding lookup (WLEmbeddingExtractor) over batch.wl_colours,
    and value_net resized to read the frozen vocab's per-graph colour
    histogram (batch.wl_histogram) directly, instead of a learned global
    embedding. See the module docstring for the one known, deferred gap
    (gnn_extractor2 aliasing under shared_gnn=True).
    """

    def __init__(self, *args, wl_vocab_path: str = str(WL_VOCAB_PATH), **kwargs):
        # Must be set before super().__init__(): _build_gnn_extractor(),
        # called from within that chain, needs self.wl_vocab_path already
        # present to load the vocab and size the embedding table.
        self.wl_vocab_path = wl_vocab_path
        # GNNFeedbackPolicy.__init__ doesn't retain layer_norm as an
        # attribute (it's a bare local, used once to construct the base
        # class's path_value_net) - capture it here (mirroring its own
        # default of False) so _build_path_value_net can reconstruct
        # PathValueNet with it after super().__init__() runs.
        self._layer_norm = kwargs.get("layer_norm", False)
        super().__init__(*args, **kwargs)
        # path_value_net is built by GNNFeedbackPolicy.__init__ (inside
        # the super().__init__() call above) sized for EMB_SIZE - replace
        # it here with one sized for wl_vocab_size + 1. Same relative
        # timing as the base class's own construction (after the
        # optimizer is already built - a pre-existing condition, not
        # introduced or fixed here), so this doesn't change whether
        # path_value_net's parameters are optimized, only its shape.
        self._build_path_value_net()

    def _build_gnn_extractor(self) -> None:
        self._wl_vocab = _load_vocab(self.wl_vocab_path)
        self.wl_vocab_size = len(self._wl_vocab)
        self.gnn_extractor = WLEmbeddingExtractor(self.wl_vocab_size, EMB_SIZE)

    def _build_value_net(self) -> None:
        # +1 for time_left, concatenated onto the histogram in _get_latent.
        self.value_net = nn.Linear(self.wl_vocab_size + 1, 1)

    def _build_path_value_net(self) -> None:
        # +1 for time_left, concatenated onto the histogram in
        # _encode_current_state/_encode_projected_state below - matches
        # value_net's own +1 convention.
        self.path_value_net = PathValueNet(layer_norm=self._layer_norm, input_dim=self.wl_vocab_size + 1)

    def _encode_current_state(self, symbolic_batch: Batch) -> th.Tensor:
        """
        Overrides GNNPlanFeedbackPolicy._encode_current_state: no GNN pass.
        symbolic_batch (the raw, untouched current-state batch) already
        carries wl_histogram (from env_to_graph's wiring) and the original
        global_features (index 0 = time_left, untouched by the
        meta-controller's encoder) - concatenate them directly, mirroring
        _get_latent's own construction of latent_global.

        :param symbolic_batch: the raw current-state batch
        :return: [num_graphs, wl_vocab_size + 1] discriminator input
        """
        time_left = symbolic_batch.global_features[:, 0:1]
        return th.cat([symbolic_batch.wl_histogram, time_left], dim=1)

    def _encode_projected_state(self, projected_batch: Batch) -> th.Tensor:
        """
        Overrides GNNPlanFeedbackPolicy._encode_projected_state: same
        pattern as _encode_current_state, for a projected (post-planner)
        state. Depends on Planner.plan() attaching wl_colours/wl_histogram
        to projections (sage/domains/gym_taxi/simulator/planner.py).

        :param projected_batch: a projected-state batch from project_actions
        :return: [num_graphs, wl_vocab_size + 1] discriminator input
        """
        time_left = projected_batch.global_features[:, 0:1]
        return th.cat([projected_batch.wl_histogram, time_left], dim=1)

    def _get_latent(self, obs: th.Tensor):
        """
        Same structure as GNNPlanFeedbackPolicy._get_latent - only the
        encoder call differs: a WL-colour embedding lookup over
        batch.wl_colours instead of message-passing over
        batch.x/edge_attr/edge_index, and the (untrainable, precomputed)
        WL colour histogram, concatenated with time_left, is used directly
        as the global embedding - no transform applied to either, per
        design.

        :param obs: Observation
        :return: (batch, symbolic_batch), batch.x/global_features now
            holding the WL-based latent_nodes/latent_global
        """
        batch, symbolic_batch = self.extract_features(obs, self.device)

        # batch.global_features here is still the ORIGINAL raw global_feats
        # from env_to_graph (NodeExtractor/features_extractor only ever
        # mutates .x, never .global_features) - index 0 is time_left,
        # (env.timeout-env.time)/env.timeout. Must be read before the
        # overwrite below replaces it with the WL-based latent_global.
        time_left = batch.global_features[:, 0:1]  # [num_graphs, 1]

        latent_nodes = self.gnn_extractor(batch.wl_colours)  # [total_nodes, EMB_SIZE]
        latent_global = th.cat([batch.wl_histogram, time_left], dim=1)  # [num_graphs, wl_vocab_size + 1]

        batch.x = latent_nodes
        batch.global_features = latent_global

        return batch, symbolic_batch
