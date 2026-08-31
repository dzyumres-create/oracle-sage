"""
.. module:: wl_plan_feedback_policy
   :synopsis: WLPlanFeedbackPolicy - Taxi's meta-controller with a
   Weisfeiler-Leman colour-embedding encoder in place of message-passing.

   Per the confirmed scope: ONLY the meta-controller's encoder changes.
   The discriminator (self.gnn_extractor2, PathValueNet,
   self._encode_current_state, self.a2) stays fully GNN-based, for both
   the current state and every projected state, completely untouched here.

      Note: GNNFeedbackPolicy.__init__ aliases self.gnn_extractor2 =
   self.gnn_extractor whenever shared_gnn=True (graph_feedback_policy.py:
   244-249). This previously had no type check, meaning gnn_extractor2
   would silently become the SAME WLEmbeddingExtractor object as this
   policy's meta-controller encoder under shared_gnn=True - not the
   independent GNN discriminator the design requires. This has since been
   fixed there (an isinstance(self.gnn_extractor, GNNExtractor) check) -
   gnn_extractor2 is now always a genuine, independent GNNExtractor,
   regardless of shared_gnn's value, whenever the meta-controller's
   encoder isn't itself a GNNExtractor.
"""
import json

import torch as th
from torch import nn

from sage.agent.graph_policy import EMB_SIZE
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
        super().__init__(*args, **kwargs)

    def _build_gnn_extractor(self) -> None:
        self._wl_vocab = _load_vocab(self.wl_vocab_path)
        self.wl_vocab_size = len(self._wl_vocab)
        self.gnn_extractor = WLEmbeddingExtractor(self.wl_vocab_size, EMB_SIZE)

    def _build_value_net(self) -> None:
        # +1 for time_left, concatenated onto the histogram in _get_latent.
        self.value_net = nn.Linear(self.wl_vocab_size + 1, 1)

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
