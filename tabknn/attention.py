"""Read test-to-training attention without changing prediction computation.

Adapters target installed TabPFN v3.5 and TabICLv2. Scores are averaged over
heads and ensemble members, separately for each ICL layer. Feature/column
attention is excluded. Temporary instrumentation is process-global: do not
call this concurrently from multiple threads.
"""

from contextlib import ExitStack
import importlib
from unittest.mock import patch

import numpy as np
import torch


def _weights(q, k, mask=None):
    """SDPA probabilities from scaled/rotated Q,K: (...,heads,rows,dim)."""
    # Compute in float32 even under mixed precision, so the softmax and the
    # top-k ranking are not affected by fp16/bf16 rounding.
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k = q.float(), k.float()
        # Grouped-query attention: repeat each key head for its query heads.
        if q.shape[-3] != k.shape[-3]:
            if q.shape[-3] % k.shape[-3]:
                raise ValueError("Incompatible query/key head counts")
            k = k.repeat_interleave(q.shape[-3] // k.shape[-3], dim=-3)
        scores = (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -torch.inf)
            else:
                scores = scores + mask
        return scores.softmax(dim=-1)


def extract_attention(model, X_train, y_train, X_test, *, verify_predictions=False):
    """Fit and return ({layer: (n_test,n_train)}, probabilities, metadata).

    Unsupported architectures, row subsampling, unexpected sequence layouts,
    and missing captures fail rather than silently misassign training rows.
    """
    model.fit(X_train, y_train)
    n_train, n_test = len(X_train), len(X_test)
    if not n_test:
        raise ValueError("At least one test row is required")
    is_pfn = type(model).__module__.startswith("tabpfn.")
    if is_pfn:
        nets = list({id(net): net for net in model.models_}.values())
        if any(type(net).__name__ != "TabPFNV3p5" for net in nets):
            raise NotImplementedError("Attention adapter requires TabPFNV3p5")
        if model.ensemble_preprocessor_.subsample_row_indices is not None:
            raise NotImplementedError("Row-subsampled ensembles need an index mapping")
        modules = [(i, block.icl_attention) for net in nets
                   for i, block in enumerate(net.icl_blocks)]
        kernel_module = importlib.import_module(type(nets[0]).__module__)
        kernel_name = "scaled_dot_product_attention"
    else:
        net = model.model_
        if type(net).__module__ != "tabicl._model.tabicl":
            raise NotImplementedError("Attention adapter requires TabICLv2")
        modules = [(i, block.attn) for i, block in
                   enumerate(net.icl_predictor.tf_icl.blocks)]
        kernel_module = importlib.import_module("tabicl._model.attention")
        kernel_name = "sdpa_with_flattened_batch"
    original = getattr(kernel_module, kernel_name)
    totals, counts = {}, {}
    active = []

    def enter(layer):
        def hook(module, args, kwargs):
            active.append(layer)
        return hook

    def leave(module, args, kwargs, output):
        active.pop()

    def capture(q, k, v, *args, **kwargs):
        if active:
            layer = active[-1]
            if is_pfn:
                queries, keys = q.transpose(-3, -2), k.transpose(-3, -2)
                mask = None
            else:
                queries, keys = q, k
                mask = args[0] if args else kwargs.get("attn_mask")
                scaler = kwargs.get("ssmax_layer")
                if scaler is not None:
                    shape = queries.shape
                    queries = scaler(queries.reshape(-1, *shape[-3:]), keys.shape[-2]).view(shape)
            if keys.shape[-2] != n_train:
                raise ValueError(f"Expected {n_train} training keys, got {keys.shape[-2]}")
            rows = queries.shape[-2]
            if rows == n_train + n_test:
                queries = queries[..., n_train:, :]
                if mask is not None and mask.shape[-2] != 1:
                    mask = mask[..., n_train:, :]
            elif rows == n_train:
                return original(q, k, v, *args, **kwargs)
            elif rows != n_test:
                raise ValueError(f"Unexpected query row count: {rows}")
            pieces = []
            for start in range(0, n_test, 32):
                chunk_mask = mask
                if mask is not None and mask.shape[-2] != 1:
                    chunk_mask = mask[..., start:start + 32, :]
                weights = _weights(queries[..., start:start + 32, :], keys, chunk_mask)
                weights = weights.reshape(-1, *weights.shape[-3:]).mean(dim=1)
                pieces.append(weights.sum(dim=0).detach().cpu().numpy())
            summed = np.concatenate(pieces, axis=0)
            batch_count = int(np.prod(queries.shape[:-3]))
            totals[layer] = totals.get(layer, 0) + summed
            counts[layer] = counts.get(layer, 0) + batch_count
        return original(q, k, v, *args, **kwargs)

    with ExitStack() as stack:
        for layer, module in modules:
            pre = module.register_forward_pre_hook(enter(layer), with_kwargs=True)
            post = module.register_forward_hook(leave, with_kwargs=True, always_call=True)
            stack.callback(pre.remove)
            stack.callback(post.remove)
        stack.enter_context(patch.object(kernel_module, kernel_name, capture))
        with torch.inference_mode():
            probabilities = model.predict_proba(X_test)
    expected = {i for i, _ in modules}
    if set(totals) != expected:
        raise RuntimeError(f"Missing attention layers: {expected - set(totals)}")
    scores = {i: totals[i] / counts[i] for i in sorted(totals)}
    for layer, weights in scores.items():
        if weights.shape != (n_test, n_train) or not np.isfinite(weights).all():
            raise RuntimeError(f"Invalid attention in layer {layer}")
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=2e-5)
    prediction_delta = None
    if verify_predictions:
        reference = model.predict_proba(X_test)
        prediction_delta = float(np.max(np.abs(probabilities - reference)))
        np.testing.assert_allclose(probabilities, reference, atol=1e-6, rtol=1e-5)
    metadata = dict(architecture=type(nets[0] if is_pfn else net).__name__,
                    ensemble_members_per_layer=counts,
                    prediction_max_abs_delta=prediction_delta)
    return scores, probabilities, metadata


def tabpfn_attention(model, X_train, y_train, X_test, layer=-1):
    scores, _, _ = extract_attention(model, X_train, y_train, X_test)
    return scores[sorted(scores)[layer]]


def tabicl_attention(model, X_train, y_train, X_test, layer=-1):
    scores, _, _ = extract_attention(model, X_train, y_train, X_test)
    return scores[sorted(scores)[layer]]
