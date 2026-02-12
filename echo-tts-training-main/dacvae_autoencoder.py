# jax_model.py
# Minimal Flax port of facebookresearch/dacvae's DACVAE architecture (watermarker excluded).

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from transpose_conv import gradient_based_conv_transpose

_EPS = 1e-8


class Snake1d(nn.Module):
    """Snake activation (per-channel alpha), channel-last.

    PyTorch checkpoint stores alpha as (1, C, 1). In this Flax port we store alpha as (1, 1, C).
    """
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        c = x.shape[-1]
        alpha = self.param("alpha", nn.initializers.ones, (1, 1, c))
        return x + (1.0 / (alpha + _EPS)) * (jnp.sin(alpha * x) ** 2)


class WNConv1d(nn.Module):
    """Weight-normalized 1D conv (PyTorch-style weight_v/weight_g parameterization), channel-last."""
    features: int
    kernel: int
    stride: int = 1
    padding: int = 0
    dilation: int = 1

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        in_ch = x.shape[-1]

        # Store params in torch-friendly shapes: (out, in, k)
        weight_v = self.param("weight_v", nn.initializers.lecun_normal(), (self.features, in_ch, self.kernel))
        weight_g = self.param("weight_g", nn.initializers.ones, (self.features, 1, 1))
        bias = self.param("bias", nn.initializers.zeros, (self.features,))

        # WeightNorm: w = g * v / ||v||
        v_norm = jnp.linalg.norm(weight_v, axis=(1, 2), keepdims=True)
        weight = weight_v * (weight_g / (v_norm + _EPS))

        # Flax Conv kernel layout: (k, in, out)
        kernel = weight.transpose(2, 1, 0)

        conv = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel,),
            strides=(self.stride,),
            padding=[(self.padding, self.padding)],
            kernel_dilation=(self.dilation,),
            use_bias=True,
        )
        return conv.apply({"params": {"kernel": kernel, "bias": bias}}, x)


class WNConvTranspose1d(nn.Module):
    """Weight-normalized 1D transposed conv (PyTorch-style weight_v/weight_g), channel-last."""
    features: int
    kernel: int
    stride: int = 1
    padding: int = 0

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        in_ch = x.shape[-1]

        # Store params in torch-friendly shapes: (in, out, k) for ConvTranspose1d
        weight_v = self.param("weight_v", nn.initializers.lecun_normal(), (in_ch, self.features, self.kernel))
        weight_g = self.param("weight_g", nn.initializers.ones, (in_ch, 1, 1))
        bias = self.param("bias", nn.initializers.zeros, (self.features,))

        v_norm = jnp.linalg.norm(weight_v, axis=(1, 2), keepdims=True)
        weight = weight_v * (weight_g / (v_norm + _EPS))

        # Convert to (k, out, in) for gradient_based_conv_transpose with transpose_kernel=True
        # PyTorch weight is (in_ch, out_ch, k), need (k, out_ch, in_ch) for the "forward conv" format
        kernel = weight.transpose(2, 1, 0)

        y = gradient_based_conv_transpose(
            x, kernel,
            strides=(self.stride,),
            padding=[(self.padding, self.padding)],
            transpose_kernel=True,
            dimension_numbers=('NHC', 'HIO', 'NHC'),
        )
        return y + bias


class ResidualUnit(nn.Module):
    dim: int
    dilation: int

    def setup(self) -> None:
        pad = ((7 - 1) * self.dilation) // 2
        self.snake_0 = Snake1d()
        self.conv_0 = WNConv1d(self.dim, kernel=7, stride=1, padding=pad, dilation=self.dilation)
        self.snake_1 = Snake1d()
        self.conv_1 = WNConv1d(self.dim, kernel=1, stride=1, padding=0, dilation=1)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.snake_0(x)
        y = self.conv_0(y)
        y = self.snake_1(y)
        y = self.conv_1(y)
        return x + y


class EncoderBlock(nn.Module):
    out_dim: int
    stride: int

    def setup(self) -> None:
        in_dim = self.out_dim // 2
        self.res_0 = ResidualUnit(in_dim, dilation=1)
        self.res_1 = ResidualUnit(in_dim, dilation=3)
        self.res_2 = ResidualUnit(in_dim, dilation=9)
        self.snake = Snake1d()
        # In the released model, downsample conv uses kernel=2*stride and padding=stride//2
        self.down = WNConv1d(self.out_dim, kernel=2 * self.stride, stride=self.stride, padding=self.stride // 2)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.res_0(x)
        y = self.res_1(y)
        y = self.res_2(y)
        y = self.snake(y)
        y = self.down(y)
        return y


class Encoder(nn.Module):
    def setup(self) -> None:
        # Matches the dumped architecture:
        # 1 -> 64 -> (stride2)->128 -> (stride8)->256 -> (stride10)->512 -> (stride12)->1024
        self.block_0 = WNConv1d(64, kernel=7, stride=1, padding=3)
        self.block_1 = EncoderBlock(out_dim=128, stride=2)
        self.block_2 = EncoderBlock(out_dim=256, stride=8)
        self.block_3 = EncoderBlock(out_dim=512, stride=10)
        self.block_4 = EncoderBlock(out_dim=1024, stride=12)
        self.block_5 = Snake1d()
        self.block_6 = WNConv1d(1024, kernel=3, stride=1, padding=1)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.block_0(x)
        y = self.block_1(y)
        y = self.block_2(y)
        y = self.block_3(y)
        y = self.block_4(y)
        y = self.block_5(y)
        y = self.block_6(y)
        return y


class VAEBottleneck(nn.Module):
    """Continuous VAE bottleneck.

    in_proj: 1024 -> 256 (interpreted as [mu(128), logvar(128)])
    out_proj: 128 -> 1024
    """
    def setup(self) -> None:
        self.in_proj = WNConv1d(256, kernel=1, stride=1, padding=0)
        self.out_proj = WNConv1d(1024, kernel=1, stride=1, padding=0)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
        rng: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        stats = self.in_proj(x)
        mu, logvar = jnp.split(stats, 2, axis=-1)

        if deterministic:
            z = mu
        else:
            if rng is None:
                raise ValueError("rng must be provided when deterministic=False")
            eps = jax.random.normal(rng, mu.shape, dtype=mu.dtype)
            z = mu + eps * jnp.exp(0.5 * logvar)

        return z, mu, logvar

    def decode(self, z: jax.Array) -> jax.Array:
        return self.out_proj(z)


class DecoderBlock(nn.Module):
    out_dim: int
    stride: int

    def setup(self) -> None:
        self.snake = Snake1d()
        self.up = WNConvTranspose1d(self.out_dim, kernel=2 * self.stride, stride=self.stride, padding=self.stride // 2)
        self.res_0 = ResidualUnit(self.out_dim, dilation=1)
        self.res_1 = ResidualUnit(self.out_dim, dilation=3)
        self.res_2 = ResidualUnit(self.out_dim, dilation=9)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.snake(x)
        y = self.up(y)
        y = self.res_0(y)
        y = self.res_1(y)
        y = self.res_2(y)
        return y


class Decoder(nn.Module):
    def setup(self) -> None:
        # Matches dumped architecture (main path only):
        # 1024 -> 1536 -> (x12)->768 -> (x10)->384 -> (x8)->192 -> (x2)->96 -> audio head
        self.conv_in = WNConv1d(1536, kernel=7, stride=1, padding=3)
        self.block_1 = DecoderBlock(out_dim=768, stride=12)
        self.block_2 = DecoderBlock(out_dim=384, stride=10)
        self.block_3 = DecoderBlock(out_dim=192, stride=8)
        self.block_4 = DecoderBlock(out_dim=96, stride=2)

        # Audio head reuses the weights that live in the watermark encoder "pre" for the released checkpoint.
        self.out_snake = Snake1d()
        self.out_conv = WNConv1d(1, kernel=7, stride=1, padding=3)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.conv_in(x)
        y = self.block_1(y)
        y = self.block_2(y)
        y = self.block_3(y)
        y = self.block_4(y)
        y = self.out_snake(y)
        y = self.out_conv(y)
        y = jnp.tanh(y)
        return y


class DACVAE(nn.Module):
    """Full DACVAE (encoder -> VAE bottleneck -> decoder), watermarker excluded."""
    def setup(self) -> None:
        self.encoder = Encoder()
        self.quantizer = VAEBottleneck()
        self.decoder = Decoder()

    def encode(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
        rng: Optional[jax.Array] = None,
        return_stats: bool = False,
    ):
        h = self.encoder(x)
        z, mu, logvar = self.quantizer(h, deterministic=deterministic, rng=rng)
        return (z, mu, logvar) if return_stats else z

    def decode(self, z: jax.Array) -> jax.Array:
        h = self.quantizer.decode(z)
        return self.decoder(h)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
        rng: Optional[jax.Array] = None,
        return_stats: bool = False,
    ):
        z, mu, logvar = self.encode(x, deterministic=deterministic, rng=rng, return_stats=True)
        y = self.decode(z)
        return (y, (z, mu, logvar)) if return_stats else y

