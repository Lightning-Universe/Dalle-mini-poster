#!/usr/bin/env python
# coding: utf-8
import random
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

# Load models & tokenizer
from dalle_mini import DalleBart
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from loguru import logger
from PIL import Image
from tqdm.notebook import trange
from vqgan_jax.modeling_flax_vqgan import VQModel

model, params, vqgan, vqgan_params = [None] * 4
DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"


# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)


# Keys are passed to the model on each device to generate unique inference per device.
# create a random key
SEED = random.randint(0, 2**32 - 1)
KEY = jax.random.PRNGKey(SEED)

# ## 🖍 Text Prompt
# Our model requires processing prompts.
from dalle_mini import DalleBartProcessor

processor: DalleBartProcessor = None


def get_concat(im1, im2):
    # credit: https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new("RGB", (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def concat_images(images: List[Image.Image]) -> Image.Image:
    n = len(images)
    image = images[0]
    for i in range(1, n):
        image = get_concat(image, images[i])
    return image


class DalleMini:
    # number of predictions per prompt
    # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
    n_predictions = 1
    gen_top_k = None
    gen_top_p = None
    temperature = None
    cond_scale = 10.0

    def __init__(self):
        # check how many devices are available
        jax.local_device_count()

        global model, params, vqgan, vqgan_params, processor

        if model is None:
            processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

            # Load dalle-mini
            model, params = DalleBart.from_pretrained(
                DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
            )

            # Load VQGAN
            vqgan, vqgan_params = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False)

            # Model parameters are replicated on each device for faster inference.

            params = replicate(params)
            vqgan_params = replicate(vqgan_params)

        logger.info("created model")

    def predict(self, prompt: str) -> Image.Image:
        # Note: we could use the same prompt multiple times for faster inference.
        prompts = [prompt]
        logger.info(f"Prompts: {prompts}\n")

        tokenized_prompts = processor(prompts)

        # Finally we replicate the prompts onto each device.
        tokenized_prompt = replicate(tokenized_prompts)

        # ## 🎨 Generate images
        # We generate images using dalle-mini model and decode them with the VQGAN.

        # generate images
        images = []
        for i in trange(max(self.n_predictions // jax.device_count(), 1)):
            # get a new key
            KEY = jax.random.PRNGKey(SEED)
            key, subkey = jax.random.split(KEY)
            # generate images
            encoded_images = p_generate(
                tokenized_prompt,
                shard_prng_key(subkey),
                params,
                self.gen_top_k,
                self.gen_top_p,
                self.temperature,
                self.cond_scale,
            )
            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]
            # decode images
            decoded_images = p_decode(encoded_images, vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for decoded_img in decoded_images:
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)
                print("created image: ", i)

        return concat_images(images)
