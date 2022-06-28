"""This module implements the demo for Dalle Mini
Checkout the original implementation [here]((https://github.com/borisdayma/dalle-mini/)
The app integration is done at `research_app/components/model_demo.py`.
"""
import logging
import random
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from dalle_mini import DalleBart, DalleBartProcessor
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from PIL import Image
from rich import print
from rich.logging import RichHandler
from tqdm.notebook import trange
from vqgan_jax.modeling_flax_vqgan import VQModel

# check how many devices are available
jax.local_device_count()
FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger(__name__)


class DalleMini:
    # DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
    DALLE_COMMIT_ID = None

    # if the notebook crashes too often you can use dalle-mini instead by uncommenting below line
    DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"

    # VQGAN model
    VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
    VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

    def __init__(self):
        self.vqgan = None
        self.model = None
        self.vqgan_params = None
        self.params = None
        self.build_model()
        self.processor = DalleBartProcessor.from_pretrained(self.DALLE_MODEL, revision=self.DALLE_COMMIT_ID)

    def build_model(self):

        # Load dalle-mini
        self.model, self.params = DalleBart.from_pretrained(
            self.DALLE_MODEL, revision=self.DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
        )

        # Load VQGAN
        self.vqgan, self.vqgan_params = VQModel.from_pretrained(
            self.VQGAN_REPO, revision=self.VQGAN_COMMIT_ID, _do_init=False
        )

        params = replicate(self.params)
        vqgan_params = replicate(self.vqgan_params)
        return dict(vqgan=self.vqgan, model=self.model, vqgan_params=vqgan_params, params=params)

    # decode image
    @partial(jax.pmap, axis_name="batch")
    def p_decode(self, indices, params):
        return self.vqgan.decode_code(indices, params=params)

    # model inference
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(self, tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
        return self.model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    def predict(self, prompt: str) -> List[Image.Image]:
        prompts = [prompt]
        # create a random key
        seed = random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)

        tokenized_prompts = self.processor(prompts)

        tokenized_prompt = replicate(tokenized_prompts)

        # number of predictions per prompt
        n_predictions = 8

        # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
        gen_top_k = None
        gen_top_p = None
        temperature = None
        cond_scale = 10.0
        print(f"Prompts: {prompts}\n")
        # generate images
        images = []
        for i in trange(max(n_predictions // jax.device_count(), 1)):
            # get a new key
            key, subkey = jax.random.split(key)
            # generate images
            encoded_images = self.p_generate(
                tokenized_prompt,
                shard_prng_key(subkey),
                self.params,
                gen_top_k,
                gen_top_p,
                temperature,
                cond_scale,
            )
            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]
            # decode images
            decoded_images = self.p_decode(encoded_images, self.vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for decoded_img in decoded_images:
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)

            return images
