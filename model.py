"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(
    sentences: list[str], specials: tuple[str] = ('<pad>', '<bos>', '<eos>', '<unk>')
):
    '''build a token-to-id dict with specials first, then corpus tokens in first-seen order.'''

    vocab = {x: i for i, x in enumerate(specials)}
    i = len(specials)
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if word not in vocab:
                vocab[word] = i
                i += 1
    return vocab

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id: dict[str, int]):
    '''build the inverse id-to-token dictionary from token_to_id'''

    return {i: token for token, i in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence: str, token_to_id: dict[str, int], unk_token: str = '<unk>'):
    '''convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV'''

    return [token_to_id.get(token, token_to_id[unk_token]) for token in sentence.split()]

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids: list[int], id_to_token: dict[int, str]):
    '''map each id in ids to its token string via id_to_token and return the list'''

    return [id_to_token[i] for i in ids]

# Step 5 - pad_id_sequence
def pad_id_sequence(ids: list[int], max_len: int, pad_id: int):
    '''return a list of length exactly max_len, padding with pad_id or truncating.'''

    return ids[:max_len] + [pad_id] * (max_len - len(ids))

# Step 6 - stack_padded_sequences_to_batch
import torch


def stack_padded_sequences_to_batch(padded_sequences: list[list[int]]):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""

    return torch.LongTensor(padded_sequences)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math

import torch
from torch import Tensor


def scale_embeddings_by_sqrt_d_model(embeddings: Tensor, d_model: int):
    """Scale a token embedding tensor by sqrt(d_model)."""

    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch


def compute_positional_div_term(d_model: int):
    '''return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors'''

    return 10000 ** (-torch.arange(0, d_model, 2) / d_model)

# Step 9 - build_position_index_column
import torch


def build_position_index_column(max_len: int):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""

    return torch.arange(max_len, dtype=torch.float32)[:, None]

# Step 10 - fill_even_indices_with_sin
import torch
from torch import Tensor


def fill_even_indices_with_sin(pe: Tensor, position: Tensor, div_term: Tensor):
    """Fill even feature indices of pe with sin(position * div_term)."""

    pe[:, ::2] = torch.sin(position * div_term)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch
from torch import Tensor


def fill_odd_indices_with_cos(pe: Tensor, position: Tensor, div_term: Tensor):
    '''fill the odd-indexed columns of pe with cos(position * div_term)'''

    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch


def build_sinusoidal_positional_encoding(max_len: int, d_model: int):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""

    pe = torch.zeros(max_len, d_model)
    position = build_position_index_column(max_len)
    div_term = compute_positional_div_term(d_model)
    fill_even_indices_with_sin(pe, position, div_term)
    fill_odd_indices_with_cos(pe, position, div_term)
    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch
from torch import Tensor


def add_positional_encoding_to_embeddings(embedded_batch: Tensor, positional_encoding: Tensor):
    '''add the first L rows of positional_encoding to embedded_batch and return the sum.'''

    return embedded_batch + positional_encoding[None, : embedded_batch.shape[1]]

# Step 14 - build_padding_mask
import torch
from torch import Tensor


def build_padding_mask(token_ids: Tensor, pad_id: int):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""

    return (token_ids != pad_id)[:, None, None, :]

# Step 15 - build_causal_mask
import torch


def build_causal_mask(seq_len: int):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""

    return torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool).tril()

# Step 16 - combine_padding_and_causal_masks
import torch
from torch import Tensor


def combine_padding_and_causal_masks(padding_mask: Tensor, causal_mask: Tensor):
    '''combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).'''

    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch
from torch import Tensor


def compute_raw_attention_scores(query: Tensor, key: Tensor):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""

    return torch.einsum('...ld,...id->...li', query, key)

# Step 18 - scale_attention_scores
import math

import torch
from torch import Tensor


def scale_attention_scores(scores: Tensor, d_k: int):
    '''divide raw attention scores by sqrt(d_k) to stabilize softmax inputs'''

    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch
from torch import Tensor


def mask_attention_scores_with_neg_inf(scores: Tensor, mask: Tensor):
    """Set entries of scores where mask is False to -inf."""

    return torch.where(mask, scores, -torch.inf)

# Step 20 - softmax_attention_weights
import torch
from torch import Tensor


def softmax_attention_weights(masked_scores: Tensor):
    '''softmax over the last axis, zeroing rows that are entirely -inf'''

    # Find rows whose entries are all -inf.
    all_masked = torch.isneginf(masked_scores).all(dim=-1, keepdim=True)

    # Avoid softmax([-inf, ..., -inf]) -> NaN.
    safe_scores = masked_scores.masked_fill(all_masked, 0.0)

    weights = torch.softmax(safe_scores, dim=-1)

    # Fully masked rows should have zero attention weights.
    return weights.masked_fill(all_masked, 0.0)

# Step 21 - apply_attention_weights_to_values
import torch
from torch import Tensor


def apply_attention_weights_to_values(attention_weights: Tensor, value: Tensor):
    """Multiply attention weights by the value matrix to produce context vectors."""

    return torch.einsum('...li,...id->...ld', attention_weights, value)

# Step 22 - scaled_dot_product_attention
import torch
from torch import Tensor


def scaled_dot_product_attention(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
):
    """Run scaled dot-product attention; return (context, attention_weights)."""

    raw_attention_scores = compute_raw_attention_scores(query, key)
    attention_scores = scale_attention_scores(raw_attention_scores, query.shape[-1])
    masked_scores = (
        attention_scores
        if mask is None
        else mask_attention_scores_with_neg_inf(attention_scores, mask)
    )
    attention_weights = softmax_attention_weights(masked_scores)
    context = apply_attention_weights_to_values(attention_weights, value)
    return context, attention_weights

# Step 23 - split_last_dim_into_heads
import torch
from torch import Tensor


def split_last_dim_into_heads(tensor: Tensor, num_heads: int):
    '''reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)'''

    return tensor.reshape(*tensor.shape[:-1], num_heads, -1)

# Step 24 - transpose_heads_before_sequence
import torch
from torch import Tensor


def transpose_heads_before_sequence(split_tensor: Tensor):
    '''rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).'''

    return split_tensor.transpose(-2, -3)

# Step 25 - merge_heads_back_to_model_dim
import torch
from torch import Tensor


def merge_heads_back_to_model_dim(multi_head_tensor: Tensor):
    '''merge the head axis back into the feature axis to reconstruct d_model'''

    x = multi_head_tensor.transpose(-2, -3)
    return x.reshape(*x.shape[:-2], -1)

# Step 26 - apply_linear_projection
import torch
from torch import Tensor


def apply_linear_projection(x: Tensor, weight: Tensor, bias: Tensor | None = None):
    '''return x @ weight^T + bias (bias may be None) with shape (..., out_features)'''

    y = torch.einsum('...i,oi->...o', x, weight)
    if bias is not None:
        y += bias
    return y

# Step 27 - project_to_query_key_value
from torch import Tensor


def project_to_query_key_value(
    x: Tensor, w_q: Tensor, b_q: Tensor, w_k: Tensor, b_k: Tensor, w_v: Tensor, b_v: Tensor
):
    '''project x into separate query, key, and value tensors via three linear layers'''

    return tuple(
        map(
            lambda wb: apply_linear_projection(x, wb[0], wb[1]),
            ((w_q, b_q), (w_k, b_k), (w_v, b_v)),
        )
    )

# Step 28 - split_qkv_into_heads
import torch
from torch import Tensor


def split_qkv_into_heads(q: Tensor, k: Tensor, v: Tensor, num_heads: int):
    '''split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple'''

    return tuple(
        map(
            lambda x: transpose_heads_before_sequence(split_last_dim_into_heads(x, num_heads)),
            (q, k, v),
        )
    )

# Step 29 - multi_head_scaled_dot_product_attention
import torch
from torch import Tensor


def multi_head_scaled_dot_product_attention(
    q_h: Tensor, k_h: Tensor, v_h: Tensor, mask: Tensor | None = None
):
    '''run scaled dot-product attention over per-head Q, K, V and return (context, weights)'''

    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
import torch
from torch import Tensor


def merge_heads_and_project_output(context: Tensor, w_o: Tensor, b_o: Tensor | None = None):
    '''merge the head axis back into d_model and apply the output linear projection.'''

    return apply_linear_projection(merge_heads_back_to_model_dim(context), w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
from torch import Tensor


def assemble_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    num_heads: int,
    mask: Tensor | None = None,
):
    '''project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.'''

    q, k, v = map(
        lambda xw: apply_linear_projection(xw[0], xw[1]), ((query, w_q), (key, w_k), (value, w_v))
    )
    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads)
    context, _ = multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask)
    return merge_heads_and_project_output(context, w_o)

# Step 32 - apply_ffn_first_linear_and_relu
import torch
from torch import Tensor


def relu(x: Tensor):
    return x.maximum(torch.tensor(0))


def apply_ffn_first_linear_and_relu(x: Tensor, w1: Tensor, b1: Tensor | None = None):
    '''project x by w1, add b1, then apply a ReLU activation.'''

    y = torch.einsum('...m,mf->...f', x, w1)
    if b1 is not None:
        y += b1
    return relu(y)

# Step 33 - apply_ffn_second_linear
import torch
from torch import Tensor


def apply_ffn_second_linear(hidden: Tensor, w2: Tensor, b2: Tensor | None = None):
    '''project hidden (..., d_ff) back to (..., d_model) via w2 and b2.'''

    y = torch.einsum('...f,fm->...m', hidden, w2)
    if b2 is not None:
        y += b2
    return y

# Step 34 - position_wise_feed_forward_network
from torch import Tensor


def position_wise_feed_forward_network(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor):
    '''compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).'''

    hidden = apply_ffn_first_linear_and_relu(x, w1, b1)
    return apply_ffn_second_linear(hidden, w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch
from torch import Tensor


def compute_layer_norm_mean_and_variance(x: Tensor):
    '''return (mean, variance) reduced over the last dim with shape (..., 1)'''

    return torch.mean(x, dim=-1, keepdim=True), torch.var(x, dim=-1, keepdim=True, correction=0)

# Step 36 - normalize_and_scale_with_gamma_beta
import torch
from torch import Tensor


def normalize_and_scale_with_gamma_beta(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5):
    '''standardize x along the last axis then apply gamma and beta affine transform'''

    mean, var = compute_layer_norm_mean_and_variance(x)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta

# Step 37 - apply_residual_add_and_norm
import torch
from torch import Tensor


def apply_residual_add_and_norm(
    residual_input: Tensor, sublayer_output: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5
):
    '''combine the residual with the sublayer output and layer-normalize the result.'''

    return normalize_and_scale_with_gamma_beta(residual_input + sublayer_output, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
from torch import Tensor


def apply_dropout_with_keep_mask(x: Tensor, keep_mask: Tensor, keep_prob: float):
    '''multiply x by the boolean keep_mask and rescale by 1/keep_prob.'''

    return x * keep_mask / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
from torch import Tensor


def encoder_layer_self_attention_sublayer(
    x: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    gamma: Tensor,
    beta: Tensor,
    num_heads: int,
    src_mask: Tensor | None = None,
):
    '''run multi-head self-attention on x and wrap with residual add-and-norm.'''

    return apply_residual_add_and_norm(
        x,
        assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask),
        gamma,
        beta,
    )

# Step 40 - encoder_layer_feed_forward_sublayer
from torch import Tensor


def encoder_layer_feed_forward_sublayer(
    x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor, gamma: Tensor, beta: Tensor
):
    '''run the position-wise FFN on x and wrap it with residual add-and-norm.'''

    return apply_residual_add_and_norm(
        x, position_wise_feed_forward_network(x, w1, b1, w2, b2), gamma, beta
    )

# Step 41 - assemble_encoder_layer
from torch import Tensor


def assemble_encoder_layer(
    x: Tensor, layer_params: dict[str, Tensor], num_heads: int, src_mask: Tensor | None = None
):
    '''chain the self-attention sublayer and the feed-forward sublayer using layer_params.'''

    h = encoder_layer_self_attention_sublayer(
        x,
        layer_params['w_q'],
        layer_params['w_k'],
        layer_params['w_v'],
        layer_params['w_o'],
        layer_params['attn_gamma'],
        layer_params['attn_beta'],
        num_heads,
        src_mask,
    )
    return encoder_layer_feed_forward_sublayer(
        h,
        layer_params['w1'],
        layer_params['b1'],
        layer_params['w2'],
        layer_params['b2'],
        layer_params['ffn_gamma'],
        layer_params['ffn_beta'],
    )

# Step 42 - stack_encoder_layers
from torch import Tensor


def stack_encoder_layers(
    x: Tensor,
    encoder_layer_params_list: list[dict[str, Tensor]],
    num_heads: int,
    src_mask: Tensor | None = None,
):
    '''sequentially apply each encoder layer to the running hidden state and return the final tensor.'''

    for layer_params in encoder_layer_params_list:
        x = assemble_encoder_layer(x, layer_params, num_heads, src_mask)
    return x

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch
from torch import Tensor


def decoder_layer_masked_self_attention_sublayer(
    y: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    gamma: Tensor,
    beta: Tensor,
    num_heads: int,
    tgt_mask: Tensor | None = None,
):
    '''run masked multi-head self-attention on y and wrap with residual add-and-norm.'''

    return apply_residual_add_and_norm(
        y,
        assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask),
        gamma,
        beta,
    )

# Step 44 - decoder_layer_cross_attention_sublayer
import torch
from torch import Tensor


def decoder_layer_cross_attention_sublayer(
    y: Tensor,
    encoder_output: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    gamma: Tensor,
    beta: Tensor,
    num_heads: int,
    src_mask: Tensor | None = None,
):
    '''run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm'''

    if src_mask is not None:
        src_mask = src_mask[:, None, None, :]
    return apply_residual_add_and_norm(
        y,
        assemble_multi_head_attention_forward(
            y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask
        ),
        gamma,
        beta,
    )

# Step 45 - decoder_layer_feed_forward_sublayer
import torch
from torch import Tensor


def decoder_layer_feed_forward_sublayer(
    y: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor, gamma: Tensor, beta: Tensor
):
    '''run the position-wise FFN on y and wrap it with residual add-and-norm'''

    return apply_residual_add_and_norm(
        y, position_wise_feed_forward_network(y, w1, b1, w2, b2), gamma, beta
    )

# Step 46 - assemble_decoder_layer
from torch import Tensor


def assemble_decoder_layer(
    y: Tensor,
    encoder_output: Tensor,
    layer_params: dict[str, Tensor],
    num_heads: int,
    src_mask: Tensor | None = None,
    tgt_mask: Tensor | None = None,
):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""

    h = decoder_layer_masked_self_attention_sublayer(
        y,
        layer_params['w_q_self'],
        layer_params['w_k_self'],
        layer_params['w_v_self'],
        layer_params['w_o_self'],
        layer_params['self_gamma'],
        layer_params['self_beta'],
        num_heads,
        tgt_mask,
    )
    h = decoder_layer_cross_attention_sublayer(
        h,
        encoder_output,
        layer_params['w_q_cross'],
        layer_params['w_k_cross'],
        layer_params['w_v_cross'],
        layer_params['w_o_cross'],
        layer_params['cross_gamma'],
        layer_params['cross_beta'],
        num_heads,
        src_mask,
    )
    return decoder_layer_feed_forward_sublayer(
        h,
        layer_params['w1'],
        layer_params['b1'],
        layer_params['w2'],
        layer_params['b2'],
        layer_params['ffn_gamma'],
        layer_params['ffn_beta'],
    )

# Step 47 - stack_decoder_layers
from torch import Tensor


def stack_decoder_layers(
    y: Tensor,
    encoder_output: Tensor,
    decoder_layer_params_list: list[dict[str, Tensor]],
    num_heads: int,
    src_mask: Tensor | None = None,
    tgt_mask: Tensor | None = None,
):
    '''sequentially apply each decoder layer to the running target hidden state.'''

    for layer_params in decoder_layer_params_list:
        y = assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask)
    return y

# Step 48 - apply_final_output_projection
from torch import Tensor


def apply_final_output_projection(
    decoder_output: Tensor,
    output_projection_weight: Tensor,
    output_projection_bias: Tensor | None = None,
):
    '''project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).'''

    return apply_linear_projection(
        decoder_output, output_projection_weight, output_projection_bias
    )

# Step 49 - tie_output_projection_to_token_embeddings
import torch
from torch import Tensor


def tie_output_projection_to_token_embeddings(token_embedding_weight: Tensor):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """

    return token_embedding_weight.transpose(0, 1)

# Step 50 - apply_log_softmax_over_vocab
import torch
from torch import Tensor


def logsoftmax(x: Tensor):
    return x - torch.log(torch.sum(torch.exp(x), dim=-1, keepdim=True))


def apply_log_softmax_over_vocab(logits: Tensor):
    '''Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.'''

    return logsoftmax(logits)

# Step 51 - run_transformer_forward
from torch import Tensor


def run_transformer_forward(
    src_ids: Tensor,
    tgt_ids: Tensor,
    model_params: dict[str, Tensor | list[dict[str, Tensor]]],
    num_heads: int,
    pad_id: int,
):
    '''embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.'''

    max_len = max(src_ids.shape[1], tgt_ids.shape[1])

    token_embedding: Tensor = model_params['token_embedding']
    _, d_model = token_embedding.shape
    x = token_embedding[src_ids]
    y = token_embedding[tgt_ids]

    x, y = map(lambda x: scale_embeddings_by_sqrt_d_model(x, d_model), (x, y))

    pe = build_sinusoidal_positional_encoding(max_len, d_model)
    x, y = map(lambda x: add_positional_encoding_to_embeddings(x, pe), (x, y))

    src_mask = build_padding_mask(src_ids, pad_id)
    tgt_mask = build_padding_mask(tgt_ids, pad_id)
    causal_mask = build_causal_mask(tgt_ids.shape[1])
    tgt_mask = combine_padding_and_causal_masks(tgt_mask, causal_mask)

    encoder_output = stack_encoder_layers(x, model_params['encoder_layers'], num_heads, src_mask)
    decoder_output = stack_decoder_layers(
        y, encoder_output, model_params['decoder_layers'], num_heads, src_mask, tgt_mask
    )

    logits = apply_final_output_projection(decoder_output, model_params['output_projection'])
    log_probs = apply_log_softmax_over_vocab(logits)

    return log_probs

# Step 52 - init_encoder_layer_parameters
import math

import torch
from torch.nn.init import kaiming_uniform_, xavier_uniform_


def init_encoder_layer_parameters(d_model: int, num_heads: int, d_ff: int):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""

    return {
        'w_q': xavier_uniform_(torch.empty(d_model, d_model, requires_grad=True)),
        'w_k': xavier_uniform_(torch.empty(d_model, d_model, requires_grad=True)),
        'w_v': xavier_uniform_(torch.empty(d_model, d_model, requires_grad=True)),
        'w_o': kaiming_uniform_(torch.empty(d_model, d_model, requires_grad=True), a=math.sqrt(5)),
        'w1': kaiming_uniform_(torch.empty(d_model, d_ff, requires_grad=True), a=math.sqrt(5)),
        'b1': torch.zeros(d_ff, requires_grad=True),
        'w2': kaiming_uniform_(torch.empty(d_ff, d_model, requires_grad=True), a=math.sqrt(5)),
        'b2': torch.zeros(d_model, requires_grad=True),
        'attn_gamma': torch.ones(d_model, requires_grad=True),
        'attn_beta': torch.zeros(d_model, requires_grad=True),
        'ffn_gamma': torch.ones(d_model, requires_grad=True),
        'ffn_beta': torch.zeros(d_model, requires_grad=True),
    }

# Step 53 - init_decoder_layer_parameters
import math

import torch
from torch.nn.init import kaiming_uniform_, xavier_uniform_


def init_decoder_layer_parameters(d_model: int, num_heads: int, d_ff: int):
    '''return a dict of requires_grad tensors for one decoder layer'''

    return {
        'w_q_self': xavier_uniform_(torch.empty(d_model, d_model, requires_grad=True)),
        'w_k_self': xavier_uniform_(torch.empty(d_model, d_model, requires_grad=True)),
        'w_v_self': xavier_uniform_(torch.empty(d_model, d_model, requires_grad=True)),
        'w_o_self': kaiming_uniform_(
            torch.empty(d_model, d_model, requires_grad=True), a=math.sqrt(5)
        ),
        'w_q_cross': xavier_uniform_(torch.empty(d_model, d_model, requires_grad=True)),
        'w_k_cross': xavier_uniform_(torch.empty(d_model, d_model, requires_grad=True)),
        'w_v_cross': xavier_uniform_(torch.empty(d_model, d_model, requires_grad=True)),
        'w_o_cross': kaiming_uniform_(
            torch.empty(d_model, d_model, requires_grad=True), a=math.sqrt(5)
        ),
        'w1': kaiming_uniform_(torch.empty(d_model, d_ff, requires_grad=True), a=math.sqrt(5)),
        'b1': torch.zeros(d_ff, requires_grad=True),
        'w2': kaiming_uniform_(torch.empty(d_ff, d_model, requires_grad=True), a=math.sqrt(5)),
        'b2': torch.zeros(d_model, requires_grad=True),
        'self_gamma': torch.ones(d_model, requires_grad=True),
        'self_beta': torch.zeros(d_model, requires_grad=True),
        'cross_gamma': torch.ones(d_model, requires_grad=True),
        'cross_beta': torch.zeros(d_model, requires_grad=True),
        'ffn_gamma': torch.ones(d_model, requires_grad=True),
        'ffn_beta': torch.zeros(d_model, requires_grad=True),
    }

# Step 54 - init_embedding_and_projection_parameters (not yet solved)
# TODO: implement

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

