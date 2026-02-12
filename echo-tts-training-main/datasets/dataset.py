
import os
import glob
import pickle

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils, multihost_utils

from typing import Any, Generator, Iterable, List, Optional, Tuple, Union

from flax import struct

import tensorflow as tf

@struct.dataclass
class DataSample:
    text: str
    audio_latent: np.ndarray
    text_tokens: np.ndarray | None = None
    audio_speaker_latent: np.ndarray | None = None


@struct.dataclass
class DataBatch:
    audio_latent: np.ndarray
    audio_mask: np.ndarray
    text_tokens: np.ndarray
    text_mask: np.ndarray
    text: List[str] | None = None
    audio_speaker_latent: np.ndarray | None = None
    audio_speaker_mask: np.ndarray | None = None


class DatasetLoader():
    def length(self) -> int:
        raise NotImplementedError

    def get_sample(self, ind: int, rng: List[int]) -> DataSample:
        raise NotImplementedError

from google.cloud import storage

USER = os.getenv('USER')

LOCAL_SAVE_PATH_PREFIX = f'/home/{USER}/data'


def maybe_download_and_open(data_path: str, force_download: bool = False, cache: bool = True):

    if not cache:
        assert data_path.endswith('.pkl')
        import pickle
        with tf.io.gfile.GFile(data_path, 'rb') as f:
            return pickle.load(f)


    if data_path.startswith('gs://'):
        gs_path = data_path[5:]
        bucket_name, blob_name = gs_path.split('/', 1)
        local_path = os.path.join(LOCAL_SAVE_PATH_PREFIX, blob_name)
        if not os.path.exists(local_path) or force_download:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
    else:
        local_path = data_path
    
    if local_path.endswith('.pkl'):
        import pickle
        with open(local_path, 'rb') as f:
            return pickle.load(f)
    elif local_path.endswith('.json'):
        import json
        with open(local_path, 'r') as f:
            return json.load(f)
    elif local_path.endswith('.jsonl'):
        import json
        with open(local_path, 'r') as f:
            return [json.loads(line) for line in f]
    else:
        raise ValueError(f'Unknown file type: {local_path}')



def tokenizer_encode(text: str) -> np.ndarray:
    return np.frombuffer(text.encode('utf-8'), dtype=np.uint8)

def tokenizer_decode(tokens: np.ndarray) -> str:
    if isinstance(tokens, List):
        return bytes(tokens).decode('utf-8')
    return bytes(tokens.astype(np.uint8)).decode('utf-8')

def batch_tokenizer_encode(strings: list[str], max_length: int) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = len(strings)
    tokens = np.zeros((batch_size, max_length), dtype=np.uint8)
    mask = np.zeros((batch_size, max_length), dtype=np.uint8)
    
    for i, text in enumerate(strings):
        encoded = tokenizer_encode(text)
        length = min(len(encoded), max_length)
        tokens[i, :length] = encoded[:length]
        mask[i, :length] = 1
        
    return tokens, mask


class DummyLoader(DatasetLoader):
    def __init__(self, data_length: int, sequence_length: int, latent_size: int):
        self._length = data_length
        self._sequence_length = sequence_length
        self._latent_size = latent_size

    def length(self) -> int:
        return self._length
    
    def get_sample(self, ind: int, rng: List[int]) -> DataSample:

        return DataSample(
            text=f'test_{ind}',
            audio_latent=np.zeros((self._sequence_length, self._latent_size), dtype=np.float32),
            audio_speaker_latent=np.zeros((self._sequence_length, self._latent_size), dtype=np.float32),    
        )



@struct.dataclass
class DatasetState:
    sample_ind: int
    epoch: int

class Dataset():

    def __init__(
        self,
        dataset_loaders: Iterable[Union[Tuple[DatasetLoader, int], DatasetLoader]],
        prepend_bos_token: bool = True,
    ):
        
        all_inds_arr = []
        just_loaders = []
        for i, dataset_loader_t in enumerate(dataset_loaders):
            if isinstance(dataset_loader_t, tuple):
                dataset_loader, num_repeats = dataset_loader_t
            else:
                dataset_loader, num_repeats = dataset_loader_t, 1

            just_loaders.append(dataset_loader)
            repeat_len = dataset_loader.length() * num_repeats
            all_inds_arr.append((np.arange(repeat_len) % dataset_loader.length(), np.full(repeat_len, i)))

        all_inds_arr = np.concatenate(all_inds_arr, axis=-1).T
        self.all_inds_arr = all_inds_arr

        self.dataset_loaders = just_loaders

        self.prepend_bos_token = prepend_bos_token

        min_epoch_len = self.length()
        
        if jax.process_count() > 1:
            all_lens = multihost_utils.process_allgather(min_epoch_len)
            min_epoch_len = int(min(all_lens))

        self.min_epoch_len = min_epoch_len

    def length(self):
        return len(self.all_inds_arr)
    
    def get_raw_sample(self, ind: int, rng: List[int]) -> DataSample:
        dataset_loader = self.dataset_loaders[self.all_inds_arr[ind, 1]]
        loader_ind = self.all_inds_arr[ind, 0]
        return dataset_loader.get_sample(loader_ind, rng)
    
    def get_tokenized_sample(self, ind: int, rng: List[int]) -> DataSample:
        sample = self.get_raw_sample(ind, rng)
        return sample.replace(text_tokens=tokenizer_encode(sample.text))
    

    def get_initial_dataset_state(self) -> DatasetState:
        return DatasetState(
            sample_ind=0,
            epoch=0
        )

    def get_dataset_generator(
        self, 
        rng_seed: int,
        batch_size: int, 
        max_latent_length: int, 
        max_text_length: int,
        max_speaker_latent_length: int,
        resume_state: Optional[DatasetState] = None,
        shuffle: bool = True,
    ) -> Generator[Tuple[DataBatch, DatasetState], None, None]:

        dataset_state = resume_state
        if dataset_state is None:
            dataset_state = self.get_initial_dataset_state()

        rng = [rng_seed, rng_seed]
        rng = tf.random.fold_in(rng, jax.process_index())
        shuffle_rng, sample_rng = tf.random.split(rng, 2)

        epoch_len = self.min_epoch_len // batch_size * batch_size

        def _get_epoch_inds(epoch: int) -> np.ndarray:

            if shuffle:
                s_rng = tf.random.fold_in(shuffle_rng, epoch)
                inds = np.array(tf.random.experimental.stateless_shuffle(np.arange(self.length()), s_rng))[:epoch_len]
            else:
                inds = np.arange(epoch_len)    

            return inds
        

        def _get_next_dataset_state(current_state: DatasetState, ind: int) -> DatasetState:
            if ind + batch_size >= epoch_len:
                return DatasetState(
                    sample_ind=0,
                    epoch=current_state.epoch + 1
                )
            return DatasetState(
                sample_ind=ind + batch_size,
                epoch=current_state.epoch
            )

        while True:

            epoch_inds = _get_epoch_inds(dataset_state.epoch)

            epoch_sample_rng = tf.random.fold_in(sample_rng, dataset_state.epoch)

            start_ind = dataset_state.sample_ind

            for i in range(start_ind, len(epoch_inds), batch_size):
                batch_inds = epoch_inds[i:i+batch_size]
                samples = [self.get_raw_sample(ind, tf.random.fold_in(epoch_sample_rng, ind)) for ind in batch_inds]

                # truncate, pad and stack
                audio_lens = np.array([sample.audio_latent.shape[0] for sample in samples])
                audio_latent = np.stack([np.pad(sample.audio_latent[:max_latent_length], ((0, max(max_latent_length - sample.audio_latent.shape[0], 0)), (0, 0))) for sample in samples])
                audio_mask = np.arange(max_latent_length)[None] < audio_lens[:, None]

                text_arr = [sample.text for sample in samples]

                text_tokens, text_mask = batch_tokenizer_encode(text_arr, max_text_length)

                if self.prepend_bos_token:
                    text_tokens = np.hstack([np.zeros((text_tokens.shape[0], 1), dtype=np.int32), text_tokens])
                    text_mask = np.hstack([np.ones((text_mask.shape[0], 1), dtype=np.int32), text_mask])
                
                text_tokens = text_tokens[:, :max_text_length]
                text_mask = text_mask[:, :max_text_length]

                dataset_state = _get_next_dataset_state(dataset_state, i)
                
                speaker_latent_mask = np.arange(max_speaker_latent_length)[None] < np.array([sample.audio_speaker_latent.shape[0] for sample in samples])[:, None]
                speaker_latent = np.stack([
                    np.pad(sample.audio_speaker_latent, ((0, max(0, max_speaker_latent_length - sample.audio_speaker_latent.shape[0])), (0, 0)))[:max_speaker_latent_length] for sample in samples])
                

                yield DataBatch(
                    audio_latent=audio_latent,
                    audio_mask=audio_mask,
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                    text=text_arr,
                    audio_speaker_latent=speaker_latent,
                    audio_speaker_mask=speaker_latent_mask,
                ), dataset_state

