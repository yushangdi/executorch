# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F

from .sdpa_with_kv_cache import custom_ops_lib  # noqa


def _sdpa_with_kv_cache_ref(q, k, v, k_cache, v_cache, attn_mask, start_pos, seq_len):
    q = q.transpose(1, 2)
    k_cache[:, start_pos : start_pos + seq_len, :, :] = k
    v_cache[:, start_pos : start_pos + seq_len, :, :] = v
    sliced_k_cache = k_cache[:, : start_pos + seq_len, :, :]
    sliced_v_cache = v_cache[:, : start_pos + seq_len, :, :]
    sliced_k_cache = sliced_k_cache.transpose(1, 2)
    sliced_v_cache = sliced_v_cache.transpose(1, 2)

    num_heads_q = q.size(1)
    num_heads_kv = sliced_k_cache.size(1)
    if num_heads_q != num_heads_kv:
        assert (
            num_heads_q % num_heads_kv == 0
        ), f"{num_heads_q} not divisible by {num_heads_kv}"
    n_reps = num_heads_q // num_heads_kv
    if n_reps > 1:
        sliced_k_cache = sliced_k_cache.repeat_interleave(n_reps, dim=1)
        sliced_v_cache = sliced_v_cache.repeat_interleave(n_reps, dim=1)
    out = F.scaled_dot_product_attention(
        q, sliced_k_cache, sliced_v_cache, attn_mask=attn_mask
    )
    out = out.transpose(1, 2)
    return out


class SDPATest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.k_cache = torch.zeros((1, 10, 8, 4))
        self.v_cache = torch.zeros((1, 10, 8, 4))
        self.mask = torch.full(
            (10, 10),
            float("-inf"),
        )
        self.mask = torch.triu(self.mask, diagonal=1)
        self.use_mask_with_custom_op = False
        self.is_causal = False

    def test_sdpa_with_cache_no_mqa_1(self):
        q = torch.rand((1, 1, 8, 4))
        k = torch.rand((1, 1, 8, 4))
        v = torch.rand((1, 1, 8, 4))
        start_pos = 0
        seq_len = q.size(1)
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]
        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )
        if self.use_mask_with_custom_op:
            attn_mask = attn_mask.contiguous()
            op_output = torch.ops.llama.sdpa_with_kv_cache(
                q,
                k,
                v,
                self.k_cache,
                self.v_cache,
                start_pos,
                seq_len,
                attn_mask,
                0,
                False,
            )
        else:
            op_output = torch.ops.llama.sdpa_with_kv_cache(
                q,
                k,
                v,
                self.k_cache,
                self.v_cache,
                start_pos,
                seq_len,
                None,
                0,
                self.is_causal,
            )
        self.assertTrue(torch.allclose(ref_output, op_output))

    def test_sdpa_with_cache_no_mqa_2(self):
        q = torch.rand((1, 1, 8, 4))
        k = torch.rand((1, 1, 8, 4))
        v = torch.rand((1, 1, 8, 4))
        start_pos = 1
        seq_len = q.size(1)
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]

        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )
        if self.use_mask_with_custom_op:
            attn_mask = attn_mask.contiguous()
            op_output = torch.ops.llama.sdpa_with_kv_cache(
                q,
                k,
                v,
                self.k_cache,
                self.v_cache,
                start_pos,
                seq_len,
                attn_mask,
                0,
                False,
            )
        else:
            op_output = torch.ops.llama.sdpa_with_kv_cache(
                q,
                k,
                v,
                self.k_cache,
                self.v_cache,
                start_pos,
                seq_len,
                None,
                0,
                self.is_causal,
            )

        self.assertTrue(torch.allclose(ref_output, op_output))

    def test_sdpa_with_cache_no_mqa_3(self):
        q = torch.rand((1, 1, 8, 4))
        k = torch.rand((1, 1, 8, 4))
        v = torch.rand((1, 1, 8, 4))
        start_pos = 2
        seq_len = q.size(1)
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]

        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )
        if self.use_mask_with_custom_op:
            attn_mask = attn_mask.contiguous()
            op_output = torch.ops.llama.sdpa_with_kv_cache(
                q,
                k,
                v,
                self.k_cache,
                self.v_cache,
                start_pos,
                seq_len,
                attn_mask,
                0,
                False,
            )
        else:
            op_output = torch.ops.llama.sdpa_with_kv_cache(
                q,
                k,
                v,
                self.k_cache,
                self.v_cache,
                start_pos,
                seq_len,
                None,
                0,
                self.is_causal,
            )
        self.assertTrue(torch.allclose(ref_output, op_output))

    def test_sdpa_with_cache_no_mqa_4(self):
        q = torch.rand((1, 1, 8, 4))
        k = torch.rand((1, 1, 8, 4))
        v = torch.rand((1, 1, 8, 4))
        start_pos = 3
        seq_len = q.size(1)
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]

        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )
        if self.use_mask_with_custom_op:
            attn_mask = attn_mask.contiguous()
            op_output = torch.ops.llama.sdpa_with_kv_cache(
                q,
                k,
                v,
                self.k_cache,
                self.v_cache,
                start_pos,
                seq_len,
                attn_mask,
                0,
                False,
            )
        else:
            op_output = torch.ops.llama.sdpa_with_kv_cache(
                q,
                k,
                v,
                self.k_cache,
                self.v_cache,
                start_pos,
                seq_len,
                None,
                0,
                self.is_causal,
            )
        self.assertTrue(torch.allclose(ref_output, op_output))


class SDPAWithAttentionMaskTest(SDPATest):

    def setUp(self):
        SDPATest.setUp(self)
        self.mask = torch.full(
            (10, 10),
            100.642,
        )
        self.use_mask_with_custom_op = True


class SDPAWithCausalTest(SDPATest):

    def setUp(self):
        SDPATest.setUp(self)
        self.is_causal = True


class SDPAWithDynamicShape(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.k_cache = torch.zeros((1, 10, 8, 4))
        self.v_cache = torch.zeros((1, 10, 8, 4))
        self.mask = torch.full(
            (10, 10),
            float("-inf"),
        )
        self.mask = torch.triu(self.mask, diagonal=1)
        self.use_mask_with_custom_op = False
        self.is_causal = False

    def test_sdpa_with_cache_dynamic_shape_0(self):
        q = torch.rand((1, 4, 8, 4))
        k = torch.rand((1, 4, 8, 4))
        v = torch.rand((1, 4, 8, 4))
        seq_len = q.size(1)
        start_pos = 0
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]

        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )

        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, start_pos, seq_len, None, 0, True
        )
        self.assertTrue(torch.allclose(ref_output, op_output))

    def test_sdpa_with_cache_dynamic_shape_2(self):
        q = torch.rand((1, 3, 8, 4))
        k = torch.rand((1, 3, 8, 4))
        v = torch.rand((1, 3, 8, 4))
        seq_len = q.size(1)
        start_pos = 2
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]

        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )

        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, start_pos, seq_len, None, 0, True
        )
        self.assertTrue(torch.allclose(ref_output, op_output))

    @unittest.skip("This test will expect failure but runtime is not bubbling it up.")
    def test_sdpa_with_cache_dynamic_shape_4(self):
        q = torch.rand((1, 11, 8, 4))
        k = torch.rand((1, 11, 8, 4))
        v = torch.rand((1, 11, 8, 4))
        seq_len = q.size(1)
        start_pos = 4

        torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, start_pos, seq_len, None, 0, True
        )


class SDPATestWithMQA(unittest.TestCase):

    def setup_caches(self):
        self.k_cache = torch.zeros((1, 5, self.n_heads_kv, 4))
        self.v_cache = torch.zeros((1, 5, self.n_heads_kv, 4))

    def setUp(self):
        torch.manual_seed(42)
        self.n_heads_kv = 4
        self.n_heads_q = 8
        self.setup_caches()
        self.mask = torch.full(
            (5, 5),
            float("-inf"),
        )
        self.mask = torch.triu(self.mask, diagonal=1)

    def test_sdpa_with_cache_mqa_1(self):
        q = torch.rand((1, 1, self.n_heads_q, 4))
        k = torch.rand((1, 1, self.n_heads_kv, 4))
        v = torch.rand((1, 1, self.n_heads_kv, 4))
        start_pos = 0
        seq_len = q.size(1)
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]
        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, 0, 1, None, 0, False
        )
        self.assertTrue(torch.allclose(ref_output, op_output))

    def test_sdpa_with_cache_mqa_2(self):
        q = torch.rand((1, 1, self.n_heads_q, 4))
        k = torch.rand((1, 1, self.n_heads_kv, 4))
        v = torch.rand((1, 1, self.n_heads_kv, 4))
        start_pos = 1
        seq_len = q.size(1)
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]
        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, 1, 1, None, 0, False
        )
        self.assertTrue(torch.allclose(ref_output, op_output))

    def test_sdpa_with_cache_mqa_3(self):
        self.n_heads_q = 14
        self.n_heads_kv = 7
        self.setup_caches()
        q = torch.rand((1, 1, self.n_heads_q, 4))
        k = torch.rand((1, 1, self.n_heads_kv, 4))
        v = torch.rand((1, 1, self.n_heads_kv, 4))
        start_pos = 1
        seq_len = q.size(1)
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]
        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, 1, 1, None, 0, False
        )
        self.assertTrue(torch.allclose(ref_output, op_output))


class SDPATestForLargeSeqLength(unittest.TestCase):

    def setup_caches(self):
        self.k_cache = torch.zeros(
            (1, self.max_seq_len, self.n_heads_kv, self.head_dim)
        )
        self.v_cache = torch.zeros(
            (1, self.max_seq_len, self.n_heads_kv, self.head_dim)
        )
        self.mask = torch.full(
            (self.max_seq_len, self.max_seq_len),
            float("-inf"),
        )
        self.mask = torch.triu(self.mask, diagonal=1)

    def setUp(self):
        torch.manual_seed(42)
        self.n_heads_kv = 32
        self.n_heads_q = 32
        self.head_dim = 128
        self.max_seq_len = 2048
        self.setup_caches()

    def test_sdpa_with_cache_seq_len_130(self):
        self.n_heads_kv = 32
        self.n_heads_q = 32
        self.head_dim = 128
        self.max_seq_len = 2048
        self.setup_caches()
        seq_len = 130
        q = torch.rand((1, seq_len, self.n_heads_kv, self.head_dim))
        k = torch.rand((1, seq_len, self.n_heads_kv, self.head_dim))
        v = torch.rand((1, seq_len, self.n_heads_kv, self.head_dim))
        start_pos = 0
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]
        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, start_pos, seq_len, None, 0, True
        )
        self.assertTrue(torch.allclose(ref_output, op_output))

        q = torch.rand((1, 1, self.n_heads_kv, self.head_dim))
        k = torch.rand((1, 1, self.n_heads_kv, self.head_dim))
        v = torch.rand((1, 1, self.n_heads_kv, self.head_dim))
        start_pos = seq_len
        seq_len = q.size(1)
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]
        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, start_pos, seq_len, None, 0, True
        )
        self.assertTrue(torch.allclose(ref_output, op_output))

    def test_sdpa_with_cache_seq_len_small(self):
        self.n_heads_kv = 4
        self.n_heads_q = 4
        self.head_dim = 4
        self.max_seq_len = 8
        self.setup_caches()
        q = torch.rand((1, 4, self.n_heads_q, 4))
        k = torch.rand((1, 4, self.n_heads_q, 4))
        v = torch.rand((1, 4, self.n_heads_q, 4))
        start_pos = 0
        seq_len = q.size(1)
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]
        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, start_pos, seq_len, None, 0, True
        )
        self.assertTrue(torch.allclose(ref_output, op_output))

        q = torch.rand((1, 1, self.n_heads_q, 4))
        k = torch.rand((1, 1, self.n_heads_q, 4))
        v = torch.rand((1, 1, self.n_heads_q, 4))
        start_pos = 4
        seq_len = q.size(1)
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]
        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, start_pos, seq_len, None, 0, True
        )
        self.assertTrue(torch.allclose(ref_output, op_output))

    def test_sdpa_with_cache_seq_len_llava_example(self):
        self.n_heads_kv = 32
        self.n_heads_q = 32
        self.head_dim = 128
        self.max_seq_len = 2048
        self.setup_caches()
        seq_len = 634
        q = torch.rand((1, seq_len, self.n_heads_kv, self.head_dim))
        k = torch.rand((1, seq_len, self.n_heads_kv, self.head_dim))
        v = torch.rand((1, seq_len, self.n_heads_kv, self.head_dim))
        start_pos = 0
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]
        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, start_pos, seq_len, None, 0, True
        )
        self.assertTrue(torch.allclose(ref_output, op_output))

        q = torch.rand((1, 1, self.n_heads_kv, self.head_dim))
        k = torch.rand((1, 1, self.n_heads_kv, self.head_dim))
        v = torch.rand((1, 1, self.n_heads_kv, self.head_dim))
        start_pos = seq_len
        seq_len = q.size(1)
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]
        ref_output = _sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, attn_mask, start_pos, seq_len
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, start_pos, seq_len, None, 0, True
        )
        self.assertTrue(torch.allclose(ref_output, op_output))
