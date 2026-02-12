from functools import partial
import os
import pickle
import subprocess
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils, multihost_utils

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


import flax
import flax.linen as nn
from flax import struct
from flax.training import train_state

import optax

from datasets.dataset import Dataset, DataBatch, maybe_download_and_open, tokenizer_decode
from model import BlockDiT, BlockDiTConfig, get_logical_rules

import orbax.checkpoint as ocp
import tensorflow as tf

from einops import rearrange

from dacvae_autoencoder import DACVAE

from tensorboardX import SummaryWriter


@struct.dataclass
class TrainConfig:
    log_dir: str
    run_name: str
    # 
    batch_size: int
    max_latent_length: int
    max_text_length: int
    speaker_max_latent_length: int
    learning_rate: float
    weight_decay: float | None
    clip_by_global_norm: float | None
    #
    num_steps: int
    adam_b1: float
    adam_b2: float
    #
    text_cfg_rate: float
    speaker_cfg_rate: float
    #
    save_every_n: int
    keep_every_n: int 
    write_train_logs_every_n: int
    save_fp16_every_n: int | None
    # eval
    eval_every_n: int
    eval_batch_size: int
    eval_num_batches: int
    # sample
    sample_every_n: int | None
    sample_batch_size: int | None
    sample_num_save: int | None
    sample_num_steps: int | None
    sample_cfg_scale: float | None


PyTreeDef = type(jax.tree_util.tree_structure(None))

def l2_norm(x: PyTreeDef) -> jax.Array:
    return jnp.sqrt(jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y ** 2), x, initializer=0.))

def count_params(params: PyTreeDef) -> int:
    return jax.tree_util.tree_reduce(lambda x, y: x + y.size, params, initializer=0)

def train(
    mesh: Mesh,
    train_config: TrainConfig,
    model_config: BlockDiTConfig,
    train_dataset: Dataset,
    rng: jax.Array,
    eval_dataset: Dataset | None,
    ae_param_path: str,
):
    
    LATENT_SIZE = 128

    LATENT_SCALE = 1.

    @partial(jax.jit)
    def ae_decode(ae_params: PyTreeDef, latents: jax.Array) -> jax.Array:
        with jax.default_matmul_precision('float32'):
            latents = latents / LATENT_SCALE
            decoder_out = DACVAE().apply({'params': ae_params}, latents, method='decode')

        return decoder_out

    ae_params = maybe_download_and_open(ae_param_path)
    del ae_params['encoder']

    ae_params = jax.tree_util.tree_map(
        lambda x: multihost_utils.host_local_array_to_global_array(x, mesh, P()), ae_params
    )


    model = BlockDiT(model_config)

    dataset_state = train_dataset.get_initial_dataset_state()

    dummy_bs = train_config.batch_size * jax.process_count()
    dummy_seq_len = 768
    dummy_seq_len_text = 768
    dummy_seq_len_speaker = 512
    dummy_batch = {
        'x': jnp.ones((dummy_bs, dummy_seq_len, LATENT_SIZE), dtype=jnp.float32),
        't': jnp.ones((dummy_bs, ), dtype=jnp.float32),
        'text_input_ids': jnp.ones((dummy_bs, dummy_seq_len_text), dtype=jnp.int32),
        'text_mask': jnp.ones((dummy_bs, dummy_seq_len_text), dtype=jnp.int32),
        'speaker_latent': jnp.ones((dummy_bs, dummy_seq_len_speaker, LATENT_SIZE), dtype=jnp.float32),
        'speaker_mask': jnp.ones((dummy_bs, dummy_seq_len_speaker), dtype=jnp.int32),
    }
    init_rng, rng = jax.random.split(rng)

    def get_optimizer_and_lr() -> Tuple[optax.GradientTransformation, Callable | float]:

        lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
            init_value=0.,
            peak_value=train_config.learning_rate,
            warmup_steps=int(train_config.num_steps * 0.05),
            decay_steps=train_config.num_steps,
        )

        tx_chain = []

        if train_config.clip_by_global_norm:
            tx_chain.append(optax.clip_by_global_norm(train_config.clip_by_global_norm))

        if (train_config.weight_decay or 0) > 0:
            def wd_mask(p: PyTreeDef) -> PyTreeDef:
                flat_p = flax.traverse_util.flatten_dict(p)
                nodecay_names = ['bias', 'scale', 'weight', 'gate', 'shift', 'freqs', 'phases', 'out_proj', 'adaln_rank_shift', 'adaln_rank_scale', 'adaln_rank_gate']
                has_decay = lambda name: not any(ndn in name for ndn in nodecay_names)
                mask = {name: has_decay(name) for name in flat_p}
                return flax.traverse_util.unflatten_dict(mask)
            mask = wd_mask(jax.eval_shape(lambda: model.init(init_rng, **dummy_batch)['params']))
            tx_chain.append(optax.adamw(lr_schedule, weight_decay=train_config.weight_decay, mask=mask, b1=train_config.adam_b1, b2=train_config.adam_b2))
        else:
            tx_chain.append(optax.adam(lr_schedule, b1=train_config.adam_b1, b2=train_config.adam_b2))

        return optax.chain(*tx_chain), lr_schedule

    
    tx, lr_schedule = get_optimizer_and_lr()

    def unbox_logicallypartioned_pytree(boxed_pytree: PyTreeDef) -> PyTreeDef:
        return jax.tree_util.tree_map(
            lambda x: x.unbox() if isinstance(x, flax.linen.spmd.LogicallyPartitioned) else x, 
            boxed_pytree, 
            is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned)
        )

    

    def create_train_state() -> train_state.TrainState:
        params = model.init(init_rng, **dummy_batch)['params']
        state = train_state.TrainState.create(
            apply_fn=None,
            params=params,
            tx=tx,
        )
        return state


    abstract_state = jax.eval_shape(lambda: create_train_state())
    logical_state_sharding = nn.get_partition_spec(abstract_state)
    state_sharding = nn.logical_to_mesh_sharding(logical_state_sharding, mesh, get_logical_rules(model_config))
    unboxed_abstract_state = unbox_logicallypartioned_pytree(abstract_state)

    ckpt_dir = os.path.join(train_config.log_dir, train_config.run_name, 'ckpts')

    checkpoint_manager = ocp.CheckpointManager(
        ckpt_dir,
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=train_config.save_every_n,
            max_to_keep=1, 
            keep_period=train_config.keep_every_n
        )
    )
    ckpt_step = checkpoint_manager.latest_step()

    if ckpt_step is not None:
        state, dataset_state = checkpoint_manager.restore(
            ckpt_step,
            args=ocp.args.StandardRestore((unboxed_abstract_state, dataset_state))
        )
    else:
        state = jax.jit(
            create_train_state,
            out_shardings=state_sharding,
        )()
        state = unbox_logicallypartioned_pytree(state)


    
    def train_step(
        state: train_state.TrainState,
        batch: PyTreeDef,
        rng: jax.Array,
        is_eval: bool = False,
    ) -> Tuple[train_state.TrainState, Dict]:
        
        rng_sigma, rng_sigma_mod, rng_x, rng_speaker_cfg, rng_text_cfg = jax.random.split(rng, 5)

        def loss_fn(
            params: PyTreeDef,
        ) -> Tuple[jax.Array, Dict]:
            
            batch_size = batch['latent'].shape[0]
            offset = jax.random.uniform(rng_sigma, ())
            mod_offset = jax.random.randint(rng_sigma_mod, (), 0, batch_size)
            quantiles = ((jnp.arange(batch_size, dtype=jnp.int32) + mod_offset) % batch_size + offset) / batch_size
            t = jax.lax.logistic(jax.scipy.stats.norm.ppf(quantiles))
            
            speaker_cfg_mask = is_eval | jax.random.bernoulli(rng_speaker_cfg, 1 - train_config.speaker_cfg_rate, batch_size)
            text_cfg_mask = is_eval | jax.random.bernoulli(rng_text_cfg, 1 - train_config.text_cfg_rate, batch_size)

            noise = jax.random.normal(rng_x, batch['latent'].shape)
            x_noised = batch['latent'] * (1 - t[..., None, None]) + noise * t[..., None, None]

            model_out = model.apply(
                {'params': params}, 
                x=x_noised,
                t=t, 
                text_input_ids=batch['text_input_ids'], 
                text_mask=batch['text_mask'] * text_cfg_mask[:, None],
                speaker_latent=batch['speaker_latent'] * speaker_cfg_mask[..., None, None],
                speaker_mask=batch['speaker_latent_mask'] * speaker_cfg_mask[..., None],
            )

            loss_mask = jnp.ones_like(batch['latent_mask'])
            loss_mask = loss_mask * (batch['latent_mask'].sum(axis=-1) > 0)[..., None]
            
            diffusion_loss = (model_out - (noise - batch['latent'])) ** 2 * loss_mask[..., None]

            diffusion_loss = jnp.mean(diffusion_loss) / (jnp.mean(loss_mask * batch['latent_mask']) + 1e-6)
            
            loss = diffusion_loss

            metrics = {'train/loss': loss}

            metrics['train/average_latent_mask'] = jnp.mean(batch['latent_mask'])
            metrics['train/average_speaker_latent_mask'] = jnp.mean(batch['speaker_latent_mask'])

            return loss, metrics

        if is_eval:
            return loss_fn(state.params)
    
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(state.params)

        metrics['train_other/raw_grad_norm'] = l2_norm(grads)
        state = state.apply_gradients(grads=grads)

        metrics['train_other/param_norm'] = l2_norm(state.params)
        metrics['train_other/lr'] = lr_schedule(state.step) if callable(lr_schedule) else lr_schedule

        return state, metrics


    data_sharding = NamedSharding(mesh, P(('dp', 'fsdp')))
    
    jit_train_step = jax.jit(
        train_step,
        in_shardings=(state_sharding, data_sharding, None),
        out_shardings=(state_sharding, None),
        donate_argnums=(0,),
        static_argnums=(3,)
    )


    def eval_step(
        state: train_state.TrainState,
        batch: PyTreeDef,
        rng: jax.Array,
    ) -> Dict:
        
        _, metrics = train_step(state, batch, rng, is_eval=True)
        metrics = {
            k.replace('train', 'eval'): v
            for k, v in metrics.items()
        }
        return metrics
    
    jit_eval_step = jax.jit(
        eval_step,
        in_shardings=(state_sharding, data_sharding, None),
        out_shardings=None,
    )


    def sample_fn(
        state: PyTreeDef,
        batch: PyTreeDef,
        rng: jax.Array,
        cfg_scale: float,
    ) -> jax.Array:

        num_steps = train_config.sample_num_steps

        def model_fn(
            x: jax.Array,
            t: jax.Array,
        ) -> jax.Array:
            
            t = t.reshape(1,)

            model_out = model.apply(
                {'params': state.params},
                x=x,
                t=t,
                text_input_ids=batch['text_input_ids'], 
                text_mask=batch['text_mask'],
                speaker_latent=batch['speaker_latent'],
                speaker_mask=batch['speaker_latent_mask'],
            )
            model_out_uncond = model.apply(
                {'params': state.params},
                x,
                t=t,
                text_input_ids=batch['uncond_text_input_ids'], 
                text_mask=batch['uncond_text_mask'],
                speaker_latent=batch['speaker_latent'] * 0.,
                speaker_mask=batch['speaker_latent_mask'] * 0.,
            )
            model_out = model_out + cfg_scale * (model_out - model_out_uncond)

            return model_out

        t_schedule = jnp.linspace(0, 1, num_steps + 1)[::-1]

        def body_fn(
            i: int,
            x_t: jax.Array,
        ) -> jax.Array:            
            model_pred = model_fn(x_t, t_schedule[i])
            dt = t_schedule[i] - t_schedule[i + 1]
            x_t = x_t - model_pred * dt
            return x_t

        x_t = jax.random.normal(rng, batch['latent'].shape)
        sample = jax.lax.fori_loop(
            0, num_steps, body_fn, x_t
        )
            
        return sample


    jit_sample_fn = jax.jit(
        sample_fn,
        in_shardings=(state_sharding, data_sharding, None, None),
        out_shardings=data_sharding,
    )

    # training

    def get_train_input(batch: DataBatch) -> PyTreeDef:

        batch = dict(
            latent=batch.audio_latent.astype(jnp.float32),
            latent_mask=batch.audio_mask.astype(jnp.int32),
            text_input_ids=batch.text_tokens.astype(jnp.int32),
            text_mask=batch.text_mask.astype(jnp.int32),
            uncond_text_input_ids=np.zeros_like(batch.text_tokens).astype(jnp.int32),
            uncond_text_mask=np.zeros_like(batch.text_mask).astype(jnp.int32),
            speaker_latent=batch.audio_speaker_latent.astype(jnp.float32),
            speaker_latent_mask=batch.audio_speaker_mask.astype(jnp.int32),
        )

        batch = jax.tree_util.tree_map(lambda x: jax.make_array_from_process_local_data(NamedSharding(mesh, P(('dp', 'fsdp'))), x).astype(x.dtype), batch)

        batch['latent'] = batch['latent'] * batch['latent_mask'][..., None] * LATENT_SCALE
        batch['speaker_latent'] = batch['speaker_latent'] * batch['speaker_latent_mask'][..., None] * LATENT_SCALE

        return batch

    train_dataset_rng = 0
    train_dataset_generator = train_dataset.get_dataset_generator(
        train_dataset_rng,
        batch_size=train_config.batch_size,
        max_latent_length=train_config.max_latent_length,
        max_speaker_latent_length=train_config.speaker_max_latent_length,
        max_text_length=train_config.max_text_length,
        resume_state=dataset_state,
    )


    num_params = count_params(jax.eval_shape(lambda: create_train_state().params))

    tb_run_dir = os.path.join(train_config.log_dir, train_config.run_name, 'runs')

    if jax.process_index() == 0:
        tf.io.gfile.makedirs(tb_run_dir)
        writer = SummaryWriter(tb_run_dir)

    def add_scalar(key: str, value: float, step: int):
        if jax.process_index() == 0:
            writer.add_scalar(key, value, step)

    def flush_writer():
        if jax.process_index() == 0:
            writer.flush()

    add_scalar('train_other/num_params', num_params, 0)


    compiled_flag = False

    t1 = time.time()

    train_rng, eval_rng, sample_rng = jax.random.split(rng, 3)

    audio_written_flag = False    
    text_written_flag = False


    next_batch, next_dataset_state = next(train_dataset_generator)
    next_batch = get_train_input(next_batch)

    step = int(np.array(state.step))

    while step < train_config.num_steps:

        batch = next_batch
        dataset_state = next_dataset_state

        train_step_rng = jax.random.fold_in(train_rng, step)
        
        state, metrics = jit_train_step(state, batch, train_step_rng)

        next_batch, next_dataset_state = next(train_dataset_generator)
        next_batch = get_train_input(next_batch)

        step = int(np.array(state.step))

        t2 = time.time()

        step_time = t2 - t1
        metrics['train_other/step_time'] = step_time
        t1 = t2

        metrics['train_other/epoch'] = dataset_state.epoch

        if not compiled_flag:
            metrics['train_other/compile_time'] = step_time
            compiled_flag = True

        for k, v in metrics.items():
            v = np.array(v)
            add_scalar(k, v, step)

        if step % train_config.write_train_logs_every_n == 0:
            flush_writer()
                    
        if eval_dataset is not None and (step % train_config.eval_every_n == 0 or step == 1):

            eval_dataset_rng = 0

            eval_dataset_generator = eval_dataset.get_dataset_generator(
                eval_dataset_rng,
                batch_size=train_config.eval_batch_size,
                max_latent_length=train_config.max_latent_length,
                max_speaker_latent_length=train_config.speaker_max_latent_length,
                max_text_length=train_config.max_text_length,
                resume_state=None,
                shuffle=True,
            )

            all_eval_metrics = []

            for j in range(train_config.eval_num_batches):

                eval_batch, _ = next(eval_dataset_generator)
                eval_batch = get_train_input(eval_batch)

                eval_metrics = jit_eval_step(state, eval_batch, jax.random.fold_in(eval_rng, j))

                all_eval_metrics.append(eval_metrics)

            eval_metrics = {
                k: np.mean([np.array(em[k]) for em in all_eval_metrics])
                for k in all_eval_metrics[0].keys()
            }

            for k, v in eval_metrics.items():
                add_scalar(k, v, step)

            if step % train_config.write_train_logs_every_n == 0:
                flush_writer()

            del eval_dataset_generator
            del all_eval_metrics

        if eval_dataset is not None and (step % (train_config.sample_every_n or 1e9) == 0 or step == 50):

            eval_dataset_rng = 0

            eval_dataset_generator = eval_dataset.get_dataset_generator(
                eval_dataset_rng,
                train_config.sample_batch_size,
                max_latent_length=train_config.max_latent_length,
                max_speaker_latent_length=train_config.speaker_max_latent_length,
                max_text_length=train_config.max_text_length,
                resume_state=None,
                shuffle=True,
            )

            eval_batch, _ = next(eval_dataset_generator)
            eval_batch = get_train_input(eval_batch)

            latent_out = jit_sample_fn(state, eval_batch, sample_rng, train_config.sample_cfg_scale)
            sample_out = ae_decode(ae_params, latent_out)

            all_samples = multihost_utils.process_allgather(sample_out, tiled=True)[:train_config.sample_num_save]

            all_text_input_ids = multihost_utils.process_allgather(eval_batch['text_input_ids'], tiled=True)[:train_config.sample_num_save]
            all_text_mask = multihost_utils.process_allgather(eval_batch['text_mask'], tiled=True)[:train_config.sample_num_save]

            for i in range(len(all_samples)):
                if jax.process_index() == 0:
                    writer.add_audio(f'sample/audio/{i}_pred', all_samples[i], step, sample_rate=48_000)

            if not audio_written_flag:
                gt_audio = ae_decode(ae_params, eval_batch['latent'])
                gt_audio = multihost_utils.process_allgather(gt_audio, tiled=True)[:train_config.sample_num_save]
                for i in range(min(len(gt_audio), train_config.sample_num_save)):
                    text = tokenizer_decode(all_text_input_ids[i].tolist()[1:all_text_mask[i].sum()])

                    if jax.process_index() == 0:
                        writer.add_audio(f'sample/audio/{i}_gt', gt_audio[i], step, sample_rate=48_000)
                        writer.add_text(f'sample/text/{i}_gt', text, step)

                        if not text_written_flag:
                            writer.add_text(f'sample/text/{i}_pred', text, step)

                speaker_audio = ae_decode(ae_params, eval_batch['speaker_latent'])
                speaker_audio = multihost_utils.process_allgather(speaker_audio, tiled=True)[:train_config.sample_num_save]
                for i in range(min(len(speaker_audio), train_config.sample_num_save)):
                    if jax.process_index() == 0:
                        writer.add_audio(f'sample/audio/{i}_speaker', speaker_audio[i], step, sample_rate=48_000)

            audio_written_flag = True

            del eval_dataset_generator

        

        if step % train_config.save_every_n == 0:

            checkpoint_manager.save(
                step, args=ocp.args.StandardSave((state, dataset_state)) # could split into multiple items
            )
            if checkpoint_manager.reached_preemption(step):
                checkpoint_manager.wait_until_finished()
                exit()


        if train_config.save_fp16_every_n is not None and step % train_config.save_fp16_every_n == 0:
            np_params = jax.tree_util.tree_map(lambda x: np.array(x).astype(np.float16), state.params)
            if jax.process_index() == 0:
                if not train_config.log_dir.startswith('gs://'):
                    os.makedirs(os.path.join(train_config.log_dir, train_config.run_name, 'weights_fp16'), exist_ok=True)
                with tf.io.gfile.GFile(os.path.join(train_config.log_dir, train_config.run_name, 'weights_fp16', f'{step}.pkl'), 'wb') as f:
                    pickle.dump(np_params, f)
            del np_params

    # save fp32

    np_params = jax.tree_util.tree_map(lambda x: np.array(x).astype(np.float32), state.params)
    if jax.process_index() == 0:
        if not train_config.log_dir.startswith('gs://'):
            os.makedirs(os.path.join(train_config.log_dir, train_config.run_name, 'weights_fp32'), exist_ok=True)
        with tf.io.gfile.GFile(os.path.join(train_config.log_dir, train_config.run_name, 'weights_fp32', f'{step}.pkl'), 'wb') as f:
            pickle.dump(np_params, f)
    del np_params




def assert_mp_identity(mesh: Mesh):

    arr_np = np.arange(16) + jax.process_index() * 16
    arr_jax = jax.make_array_from_process_local_data(NamedSharding(mesh, P('dp')), arr_np)
    arr_local = np.array(multihost_utils.global_array_to_host_local_array(arr_jax, mesh, P(('dp'))))

    assert np.array_equal(arr_np, arr_local)


def initialize_and_get_mesh(
    mesh_shape: Tuple[int, int, int],
    is_local: bool = False,
    initialize: bool = True,
) -> Mesh:
    
    if initialize:
        if is_local:
            jax.distributed.initialize("localhost:8889", num_processes=1, process_id=0)
        else:
            jax.distributed.initialize()

    mesh = jax.make_mesh(mesh_shape, ('dp', 'fsdp', 'tp'))

    # def reshape(flat_list: List, shape: Tuple[int]) -> List:
    #     import math
    #     def _reshape_recursive(flat_list, shape, index):
    #         if len(shape) == 1:
    #             return flat_list[index:index+shape[0]]
    #         result = []
    #         stride = int(math.prod(shape[1:]))
    #         for i in range(shape[0]):
    #             result.append(_reshape_recursive(flat_list, shape[1:], index + i * stride))
    #         return result
    #     return _reshape_recursive(flat_list, shape, 0)

    # mesh = Mesh(reshape(jax.devices(), mesh_shape), axis_names=('dp', 'fsdp', 'tp'))

    # assert_mp_identity(mesh)

    return mesh




if __name__ == '__main__':

    mesh = initialize_and_get_mesh((2, 2, 1))

    train_config = TrainConfig(
        log_dir='gs_oroptionallylocalifonehost_log_dir_folder_path',
        run_name='some_name',
        batch_size=16,
        max_latent_length=768,
        max_text_length=768,
        speaker_max_latent_length=512,
        #
        learning_rate=1e-4,
        weight_decay=1e-2,
        clip_by_global_norm=1.,
        num_steps=300_000,
        adam_b1=0.9,
        adam_b2=0.99,
        #
        text_cfg_rate=0.1,
        speaker_cfg_rate=0.1,
        #
        save_every_n=2_000,
        keep_every_n=1_000_000,
        write_train_logs_every_n=200,
        save_fp16_every_n=50_000,
        #
        #
        eval_every_n=2000,
        eval_batch_size=32,
        eval_num_batches=50,
        #
        sample_every_n=20_000,
        sample_batch_size=8,
        sample_num_save=8,
        sample_num_steps=40,
        sample_cfg_scale=3.,
        
    )


    from datasets.dataset import DummyLoader
    # from datasets.spotify_dataset import SpotifyLoader
    # from datasets.podcast_dataset import PodcastLoader

    train_dataset = Dataset([DummyLoader(10000, 768, 128)])
    eval_dataset = Dataset([DummyLoader(1000, 768, 128)])

    model_config = BlockDiTConfig(
        model_size=1024,
        intermediate_size=2816,
        num_layers=12,
        num_heads=8,
        norm_eps=1e-5,
        #
        encoder_patch_size=4,
        encoder_model_size=768,
        encoder_intermediate_size=2048,
        encoder_num_layers=8,
        encoder_num_heads=6, # dummy
        #
        text_vocab_size=256,
        text_model_size=768,
        text_intermediate_size=2048,
        text_num_heads=6, # dummy
        text_num_layers=8,
        #
        timestep_embed_size=256,
        dtype=jnp.bfloat16,
        remat=True,
        adaln_rank=256,
    )

    rng = jax.random.PRNGKey(0) # was 0

    AE_PATH = 'gs_or_local_path_to_ae_params_converted_to_numpy_flaxmodule'

    train(
        mesh=mesh,
        train_config=train_config,
        model_config=model_config,
        train_dataset=train_dataset,
        rng=rng,
        eval_dataset=eval_dataset,
        ae_param_path=AE_PATH,
    )


    time.sleep(180)



