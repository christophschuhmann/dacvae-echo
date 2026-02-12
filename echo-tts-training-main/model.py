from typing import List, Tuple, Sequence

import numpy as np

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh, NamedSharding
from jax.experimental import shard_map

from einops import rearrange

from enum import Enum

class ModuleType(Enum):
    DECODER = 'decoder'
    ENCODER = 'encoder' 
    TEXT = 'text'
    SPEAKER = 'speaker'


@struct.dataclass
class BlockDiTConfig:
    model_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    norm_eps: float

    encoder_patch_size: int
    encoder_model_size: int
    encoder_intermediate_size: int
    encoder_num_layers: int
    encoder_num_heads: int

    text_vocab_size: int
    text_model_size: int
    text_intermediate_size: int
    text_num_heads: int
    text_num_layers: int

    timestep_embed_size: int

    dtype: jnp.dtype = jnp.float32

    remat: bool = False
    adaln_rank: int | None = None


KERNEL_INIT = nn.initializers.truncated_normal(0.02)
EMBED_INIT = nn.initializers.truncated_normal(0.02)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = np.arange(end)
    freqs = np.outer(t, freqs)
    freqs_cis = np.complex64(np.cos(freqs) + 1j * np.sin(freqs))
    return jnp.array(freqs_cis) # (seq, head_dim // 2)

def apply_rotary_emb_single(
    x: jax.Array,
    freqs_cis: jax.Array,
) -> jax.Array:

    x_ = jax.lax.complex(*rearrange(x.astype(jnp.float32), '... (s a) -> a ... s', a=2))
    x_ = x_ * freqs_cis[..., None, :]
    x_out = rearrange(jnp.concatenate([jax.lax.real(x_), jax.lax.imag(x_)], axis=-1), '... (a s) -> ... (s a)', a=2) 
    return x_out.astype(x.dtype)

def apply_rotary_emb(
    xq: jax.Array,
    xk: jax.Array,
    freqs_cis: jax.Array,
) -> Tuple[jax.Array, jax.Array]:

    return apply_rotary_emb_single(xq, freqs_cis), apply_rotary_emb_single(xk, freqs_cis)


class DenseGeneralIn(nn.Module):
    shape: Tuple[int, ...]
    use_bias: bool = True
    kernel_init: nn.initializers.Initializer = KERNEL_INIT
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # for muon so 2d shape
        assert len(self.shape) == 2
        y = nn.Dense(self.shape[0] * self.shape[1], use_bias=self.use_bias, kernel_init=self.kernel_init, dtype=self.dtype)(x)
        y = rearrange(y, '... (h d) -> ... h d', h=self.shape[0], d=self.shape[1])
        return y

class DenseGeneralOut(nn.Module):
    shape: int
    use_bias: bool = True
    kernel_init: nn.initializers.Initializer = KERNEL_INIT
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        y = rearrange(x, '... h d -> ... (h d)')
        y = nn.Dense(self.shape, use_bias=self.use_bias, kernel_init=self.kernel_init, dtype=self.dtype)(y)
        return y

class TimestepEmbedding(nn.Module):
    config: BlockDiTConfig

    @nn.compact
    def __call__(
        self,
        timestep: jnp.ndarray
    ):
        assert self.config.timestep_embed_size % 2 == 0
        half = self.config.timestep_embed_size // 2
        freqs = 1000 * jnp.exp(-jnp.log(10000.0) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = timestep[..., None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding
    

class AdaLN(nn.Module):
    config: BlockDiTConfig

    @nn.compact
    def __call__(self, x: jax.Array, cond_embed: jax.Array) -> Tuple[jax.Array, jax.Array]:
        initializer = nn.with_logical_partitioning(jax.nn.initializers.zeros, ('norm_in', 'norm_out'))
        initializer_random = nn.with_logical_partitioning(jax.nn.initializers.truncated_normal(stddev=0.02), ('norm_out', 'norm_in'))

        cond_embed_global_shift, cond_embed_global_scale, cond_embed_global_gate = jnp.split(cond_embed, 3, axis=-1)
        
        if self.config.adaln_rank is not None:
            cond_embed_shift = nn.Dense(self.config.adaln_rank, use_bias=False, name='adaln_rank_shift', kernel_init=initializer_random, dtype=self.config.dtype)(nn.silu(cond_embed_global_shift))
            cond_embed_scale = nn.Dense(self.config.adaln_rank, use_bias=False, name='adaln_rank_scale', kernel_init=initializer_random, dtype=self.config.dtype)(nn.silu(cond_embed_global_scale))
            cond_embed_gate = nn.Dense(self.config.adaln_rank, use_bias=False, name='adaln_rank_gate', kernel_init=initializer_random, dtype=self.config.dtype)(nn.silu(cond_embed_global_gate))
        else:
            cond_embed_shift, cond_embed_scale, cond_embed_gate = nn.silu(cond_embed_global_shift), nn.silu(cond_embed_global_scale), nn.silu(cond_embed_global_gate)


        shift = nn.Dense(x.shape[-1], use_bias=True, name='shift', kernel_init=initializer, dtype=self.config.dtype)(cond_embed_shift)
        scale = nn.Dense(x.shape[-1], use_bias=True, name='scale', kernel_init=initializer, dtype=self.config.dtype)(cond_embed_scale)
        gate = nn.Dense(x.shape[-1], use_bias=True, name='gate', kernel_init=initializer, dtype=self.config.dtype)(cond_embed_gate)

        shift = shift + cond_embed_global_shift
        scale = scale + cond_embed_global_scale
        gate = gate + cond_embed_global_gate

        x = x.astype(jnp.float32)
        normed = (x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.config.norm_eps))

        out = (normed * (scale + 1) + shift)

        gate = jnp.tanh(gate)


        return out.astype(self.config.dtype), gate



class RMSNorm(nn.Module):
    config: BlockDiTConfig
    is_qk: bool = False
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if not self.is_qk:
            weight = self.param('weight', nn.with_logical_partitioning(jax.nn.initializers.ones, ('norm_weight', )), (self.config.model_size, ))
        else:
            weight = self.param('weight', nn.with_logical_partitioning(jax.nn.initializers.ones, ('qknorm_weight_head', 'qknorm_weight_model')), (x.shape[-2], x.shape[-1]))
        x = x.astype(jnp.float32)
        normed = x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.config.norm_eps)
        out = normed * weight
        return out.astype(self.config.dtype)


class SelfAttention(nn.Module):
    config: BlockDiTConfig
    module_type: ModuleType

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:

        cfg = self.config
        head_size = cfg.model_size // cfg.num_heads

        q_kernel_init = nn.with_logical_partitioning(KERNEL_INIT, ('w_model', 'w_head'))
        kv_kernel_init = nn.with_logical_partitioning(KERNEL_INIT, ('w_model', 'w_head'))

        xq = DenseGeneralIn((cfg.num_heads, head_size), use_bias=False, name='wq', kernel_init=q_kernel_init, dtype=cfg.dtype)(x)
        xk = DenseGeneralIn((cfg.num_heads, head_size), use_bias=False, name='wk', kernel_init=kv_kernel_init, dtype=cfg.dtype)(x)
        xv = DenseGeneralIn((cfg.num_heads, head_size), use_bias=False, name='wv', kernel_init=kv_kernel_init, dtype=cfg.dtype)(x)

        gate = DenseGeneralIn((cfg.num_heads, head_size), use_bias=False, name='wgate', kernel_init=q_kernel_init, dtype=cfg.dtype)(x)

        xq = RMSNorm(cfg, is_qk=True, name='q_norm')(xq)
        xk = RMSNorm(cfg, is_qk=True, name='k_norm')(xk)

        xq = nn.with_logical_constraint(xq, ('a_data', 'a_sequence', 'a_head', 'a_head_model'))
        xk = nn.with_logical_constraint(xk, ('a_data', 'a_sequence', 'a_kv_head', 'a_head_model'))
        xv = nn.with_logical_constraint(xv, ('a_data', 'a_sequence', 'a_kv_head', 'a_head_model'))

        freq_cis = precompute_freqs_cis(xq.shape[-1], xq.shape[-3])
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freq_cis)

        if mask is None:
            mask = jnp.ones(x.shape[:2])
        mask = mask[:, None, None, :]

        output = jax.nn.dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            mask=mask.astype(jnp.bool_),
            is_causal=self.module_type == ModuleType.ENCODER,
        )

        output = output * jax.nn.sigmoid(gate)

        o_kernel_init = nn.with_logical_partitioning(KERNEL_INIT, ('w_head', 'w_model'))
        output = DenseGeneralOut(cfg.model_size, use_bias=False, name='wo', kernel_init=o_kernel_init, dtype=cfg.dtype)(output)
        output = nn.with_logical_constraint(output, ('a_data', 'a_sequence', 'a_model'))

        return output


class JointSelfCrossAttention(nn.Module):
    config: BlockDiTConfig

    @nn.compact
    def __call__(self, 
        x: jax.Array,
        text_state: jax.Array,
        text_mask: jax.Array,
        speaker_state: jax.Array,
        speaker_mask: jax.Array,
    ) -> jax.Array:
        
        cfg = self.config
        
        num_kv_heads = cfg.num_heads
        head_size = cfg.model_size // cfg.num_heads

        q_kernel_init = nn.with_logical_partitioning(KERNEL_INIT, ('w_model', 'w_head'))
        kv_kernel_init = nn.with_logical_partitioning(KERNEL_INIT, ('w_model', 'w_head'))

        xq = DenseGeneralIn((cfg.num_heads, head_size), use_bias=False, name='wq', kernel_init=q_kernel_init, dtype=cfg.dtype)(x)
        xq = RMSNorm(cfg, is_qk=True, name='q_norm')(xq)


        gate = DenseGeneralIn((cfg.num_heads, head_size), use_bias=False, name='wgate', kernel_init=q_kernel_init, dtype=cfg.dtype)(x)

        xk_self = DenseGeneralIn((num_kv_heads, head_size), use_bias=False, name='wk_self', kernel_init=kv_kernel_init, dtype=cfg.dtype)(x)
        xv_self = DenseGeneralIn((num_kv_heads, head_size), use_bias=False, name='wv_self', kernel_init=kv_kernel_init, dtype=cfg.dtype)(x)
        
        k_norm = RMSNorm(cfg, is_qk=True, name='k_norm')
        xk_self = k_norm(xk_self)

        xk_text = DenseGeneralIn((num_kv_heads, head_size), use_bias=False, name='wk_text', kernel_init=kv_kernel_init, dtype=cfg.dtype)(text_state)
        xv_text = DenseGeneralIn((num_kv_heads, head_size), use_bias=False, name='wv_text', kernel_init=kv_kernel_init, dtype=cfg.dtype)(text_state)

        xk_speaker = DenseGeneralIn((num_kv_heads, head_size), use_bias=False, name='wk_speaker', kernel_init=kv_kernel_init, dtype=cfg.dtype)(speaker_state)
        xv_speaker = DenseGeneralIn((num_kv_heads, head_size), use_bias=False, name='wv_speaker', kernel_init=kv_kernel_init, dtype=cfg.dtype)(speaker_state)

        xk_text = k_norm(xk_text)
        xk_speaker = k_norm(xk_speaker)

        freq_cis_q = precompute_freqs_cis(xq.shape[-1], xq.shape[-3])

        def apply_rotary_half(y, freqs_cis):
            assert y.shape[-2] % 2 == 0
            y1, y2 = y[..., :y.shape[-2]//2, :], y[..., y.shape[-2]//2:, :]
            y1 = apply_rotary_emb_single(y1, freqs_cis)
            return jnp.concatenate([y1, y2], axis=-2)
        
        xq, xk_self = map(apply_rotary_half, (xq, xk_self), (freq_cis_q, freq_cis_q))

        xk = jnp.concatenate([xk_self, xk_text, xk_speaker], axis=-3)
        xv = jnp.concatenate([xv_self, xv_text, xv_speaker], axis=-3)

        xq = nn.with_logical_constraint(xq, ('a_data', 'a_sequence', 'a_head', 'a_head_model'))
        xk = nn.with_logical_constraint(xk, ('a_data', 'a_sequence', 'a_kv_head', 'a_head_model'))
        xv = nn.with_logical_constraint(xv, ('a_data', 'a_sequence', 'a_kv_head', 'a_head_model'))

        self_mask = jnp.ones((x.shape[0], x.shape[1], x.shape[1]), dtype=jnp.bool_)

        text_mask = jnp.broadcast_to(text_mask[:, None, :], (text_mask.shape[0], x.shape[1], text_mask.shape[-1])) # (1, sd, st)
        speaker_mask = jnp.broadcast_to(speaker_mask[:, None, :], (speaker_mask.shape[0], x.shape[1], speaker_mask.shape[-1])) # (1, sd, ss)

        mask = jnp.concatenate([self_mask, text_mask, speaker_mask], axis=-1)[:, None, :, :] # (b, h, sd, sc)

        output = jax.nn.dot_product_attention( # will scale down by default
            query=xq,
            key=xk,
            value=xv,
            mask=mask.astype(jnp.bool_),
            is_causal=False,
        )

        output = output * jax.nn.sigmoid(gate)

        o_kernel_init = nn.with_logical_partitioning(KERNEL_INIT, ('w_head', 'w_model'))
        output = DenseGeneralOut(cfg.model_size, use_bias=False, name='wo', kernel_init=o_kernel_init, dtype=cfg.dtype)(output)
        output = nn.with_logical_constraint(output, ('a_data', 'a_sequence', 'a_model'))

        return output

class MLP(nn.Module):
    config: BlockDiTConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        cfg = self.config

        u_kernel_init = nn.with_logical_partitioning(KERNEL_INIT, ('w_model', 'w_intermediate'))
        d_kernel_init = nn.with_logical_partitioning(KERNEL_INIT, ('w_intermediate', 'w_model'))

        y = nn.silu(nn.Dense(cfg.intermediate_size, use_bias=False, name='w1', kernel_init=u_kernel_init, dtype=cfg.dtype)(x))
        y = nn.with_logical_constraint(y, ('a_data', 'a_sequence', 'a_intermediate'))
        z = nn.Dense(cfg.intermediate_size, use_bias=False, name='w3', kernel_init=u_kernel_init, dtype=cfg.dtype)(x)
        z = nn.with_logical_constraint(z, ('a_data', 'a_sequence', 'a_intermediate'))
        out = nn.Dense(cfg.model_size, use_bias=False, name='w2', kernel_init=d_kernel_init, dtype=cfg.dtype)(y * z)
        out = nn.with_logical_constraint(out, ('a_data', 'a_sequence', 'a_model'))

        return out
    
class EncoderTransformerBlock(nn.Module):
    config: BlockDiTConfig
    module_type: ModuleType

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:

        assert self.module_type != ModuleType.DECODER
        
        if self.module_type == ModuleType.ENCODER:
            cfg = self.config.replace(
                model_size=self.config.encoder_model_size,
                intermediate_size=self.config.encoder_intermediate_size,
                num_heads=self.config.encoder_num_heads,
            )
        elif self.module_type == ModuleType.TEXT:
            cfg = self.config.replace(
                model_size=self.config.text_model_size,
                intermediate_size=self.config.text_intermediate_size,
                num_heads=self.config.text_num_heads,
            )
        else:
            raise ValueError(f'Invalid module type: {self.module_type}')

        x = x + SelfAttention(cfg, self.module_type, name='attention')(RMSNorm(cfg, name='attention_norm')(x), mask)
        x = x + MLP(cfg, name='mlp')(RMSNorm(cfg, name='mlp_norm')(x))

        return x




class TransformerBlock(nn.Module):
    config: BlockDiTConfig

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array,
        cond_embed: jax.Array,
        text_state: jax.Array,
        text_mask: jax.Array,
        speaker_state: jax.Array,
        speaker_mask: jax.Array,
    ) -> jax.Array:
        
        cfg = self.config

        assert cond_embed.shape[1] == 1 and cond_embed.ndim == 3

        norm_out, attn_gate = AdaLN(cfg, name='attn_adaln')(x, cond_embed)
        x = x + attn_gate * JointSelfCrossAttention(cfg, name='attention')(norm_out, text_state, text_mask, speaker_state, speaker_mask)

        norm_out, mlp_gate = AdaLN(cfg, name='mlp_adaln')(x, cond_embed)
        x = x + mlp_gate * MLP(cfg, name='mlp')(norm_out)

        return x
    
class LatentEncoder(nn.Module):
    config: BlockDiTConfig
    
    @nn.compact
    def __call__(self, latent: jax.Array, mask: jax.Array | None = None) -> Tuple[jax.Array, jax.Array]:
        cfg = self.config
        x = rearrange(latent, 'b (s p) d -> b s (p d)', p=cfg.encoder_patch_size)
        x = nn.Dense(cfg.encoder_model_size, use_bias=True, name='in_proj', kernel_init=KERNEL_INIT, dtype=cfg.dtype)(x)
        x = nn.with_logical_constraint(x, ('a_data', 'a_sequence', 'a_model'))

        for i in range(cfg.encoder_num_layers):
            Block = EncoderTransformerBlock if not cfg.remat else nn.remat(EncoderTransformerBlock, prevent_cse=True)
            x = Block(cfg, ModuleType.ENCODER, name=f'EncoderTransformerBlock_{i}')(x, None)
        
        if mask is not None:
            mask = mask[..., ::cfg.encoder_patch_size]

        return x, mask
    
class TextEncoder(nn.Module):
    config: BlockDiTConfig
    
    @nn.compact
    def __call__(self, text_input_ids: jax.Array, text_mask: jax.Array) -> jax.Array:
        cfg = self.config
        text_state = nn.Embed(cfg.text_vocab_size, cfg.text_model_size, name='text_embedding', embedding_init=EMBED_INIT, dtype=cfg.dtype)(text_input_ids)

        for i in range(cfg.text_num_layers):
            Block = EncoderTransformerBlock if not cfg.remat else nn.remat(EncoderTransformerBlock, prevent_cse=True)
            text_state = Block(cfg, ModuleType.TEXT, name=f'TextTransformerBlock_{i}')(text_state, text_mask)

        return text_state


class BlockDiT(nn.Module):
    config: BlockDiTConfig

    @nn.compact
    def __call__(
        self, 
        x: jax.Array,
        t: jax.Array,
        text_input_ids: jax.Array,
        text_mask: jax.Array,
        speaker_latent: jax.Array,
        speaker_mask: jax.Array,
    ) -> jax.Array:
        
        cfg = self.config
        latent_size = x.shape[-1]

        mask = None

        # ENCODERS
        text_state = TextEncoder(cfg, name='text_encoder')(text_input_ids, text_mask)
        speaker_state, speaker_mask = LatentEncoder(cfg, name='speaker_encoder')(speaker_latent, speaker_mask)

        text_state = RMSNorm(cfg.replace(model_size=cfg.text_model_size), name='text_norm')(text_state)
        text_state = nn.with_logical_constraint(text_state, ('a_data', 'a_sequence', 'a_model'))

        speaker_state = RMSNorm(cfg.replace(model_size=cfg.encoder_model_size), name='speaker_norm')(speaker_state)
        speaker_state = nn.with_logical_constraint(speaker_state, ('a_data', 'a_sequence', 'a_model'))

        # TIMESTEP
        cond_dense_0_init = nn.with_logical_partitioning(KERNEL_INIT, ('w_timestep_in', 'w_timestep_model'))
        cond_dense_1_init = nn.with_logical_partitioning(KERNEL_INIT, ('w_timestep_model', 'w_timestep_out'))
        timestep_embed = TimestepEmbedding(cfg, name='timestep_embedding')(t)

        cond_embed = nn.Dense(
            cfg.model_size, 
            use_bias=False, 
            name='cond_dense_0', 
            kernel_init=cond_dense_0_init,
            dtype=cfg.dtype
        )(timestep_embed)
        cond_embed = nn.silu(cond_embed)
        cond_embed = nn.Dense(
            cfg.model_size, 
            use_bias=False, 
            name='cond_dense_1', 
            kernel_init=cond_dense_1_init,
            dtype=cfg.dtype
        )(cond_embed)

        cond_embed_global_shift = nn.Dense(cfg.model_size, use_bias=False, name='cond_dense_global_shift', kernel_init=KERNEL_INIT, dtype=cfg.dtype)(nn.silu(cond_embed))
        cond_embed_global_scale = nn.Dense(cfg.model_size, use_bias=False, name='cond_dense_global_scale', kernel_init=KERNEL_INIT, dtype=cfg.dtype)(nn.silu(cond_embed))
        cond_embed_global_gate = nn.Dense(cfg.model_size, use_bias=False, name='cond_dense_global_gate', kernel_init=KERNEL_INIT, dtype=cfg.dtype)(nn.silu(cond_embed))

        cond_embed = jnp.concatenate([cond_embed_global_shift, cond_embed_global_scale, cond_embed_global_gate], axis=-1)


        if cond_embed.ndim == 2:
            cond_embed = cond_embed[:, None, :]
        assert cond_embed.ndim == 3

        # DECODER
        x = nn.Dense(cfg.model_size, use_bias=True, name='in_proj', kernel_init=KERNEL_INIT, dtype=cfg.dtype)(x)
        x = nn.with_logical_constraint(x, ('a_data', 'a_sequence', 'a_model'))

        for i in range(cfg.num_layers):
            Block = TransformerBlock if not cfg.remat else nn.remat(TransformerBlock, prevent_cse=True)
            x = Block(cfg, name=f'TransformerBlock_{i}')(x, mask, cond_embed, text_state, text_mask, speaker_state, speaker_mask)

        x = RMSNorm(cfg, name='norm')(x)

        out_kernel_init = nn.with_logical_partitioning(jax.nn.initializers.zeros, ('w_model', 'w_latent'))
        x = nn.Dense(latent_size, use_bias=True, name='out_proj', kernel_init=out_kernel_init, dtype=cfg.dtype)(x)

        return x.astype(jnp.float32)


def get_logical_rules(
    mesh: Mesh,
) -> List[Tuple[str | Tuple, ...]]:
    rules = [
        ('w_model', 'fsdp'),
        ('w_head', 'tp'),
        ('w_kv_head', 'tp'),
        ('w_intermediate', 'tp'),
        ('w_vocab', 'fsdp'),
        ('a_data', ('dp', 'fsdp')),
        ('a_head', 'tp'),
        ('a_kv_head', 'tp'),
        ('a_intermediate', 'tp'),
        ('norm_in', 'fsdp'),
    ]
    return rules
    

    
