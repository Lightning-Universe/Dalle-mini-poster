#!/usr/bin/env python
# coding: utf-8

# # DALL·E mini - Inference pipeline
#
# *Generate images from a text prompt*
#
# <img src="https://github.com/borisdayma/dalle-mini/blob/main/img/logo.png?raw=true" width="200">
#
# This notebook illustrates [DALL·E mini](https://github.com/borisdayma/dalle-mini) inference pipeline.
#
# Just want to play? Use directly [the app](https://www.craiyon.com/).
#
# For more understanding of the model, refer to [the report](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini--Vmlldzo4NjIxODA).


# We load required models:
# * DALL·E mini for text to encoded images
# * VQGAN for decoding images
# * CLIP for scoring predictions

# In[1]:


# Model references

# dalle-mega
# DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
DALLE_COMMIT_ID = None

# if the notebook crashes too often you can use dalle-mini instead by uncommenting below line
DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

# In[2]:


import jax
import jax.numpy as jnp

# check how many devices are available
jax.local_device_count()

# In[3]:


# Load models & tokenizer
from dalle_mini import DalleBart
from vqgan_jax.modeling_flax_vqgan import VQModel

# Load dalle-mini
model, params = DalleBart.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False)

# Load VQGAN
vqgan, vqgan_params = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False)

# Model parameters are replicated on each device for faster inference.

# In[4]:


from flax.jax_utils import replicate

params = replicate(params)
vqgan_params = replicate(vqgan_params)

# Model functions are compiled and parallelized to take advantage of multiple devices.

# In[5]:


from functools import partial


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

# In[6]:


import random

# create a random key
SEED = random.randint(0, 2 ** 32 - 1)
KEY = jax.random.PRNGKey(SEED)

# ## 🖍 Text Prompt

# Our model requires processing prompts.

# In[7]:


from dalle_mini import DalleBartProcessor

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

# Let's define some text prompts.

# In[8]:


# In[ ]:


from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange


def predict(prompt: str):
    # Note: we could use the same prompt multiple times for faster inference.
    prompts = [prompt]

    tokenized_prompts = processor(prompts)

    # Finally we replicate the prompts onto each device.

    # In[10]:


    tokenized_prompt = replicate(tokenized_prompts)

    # ## 🎨 Generate images
    #
    # We generate images using dalle-mini model and decode them with the VQGAN.

    # In[11]:


    # number of predictions per prompt
    n_predictions = 1

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
        key, subkey = jax.random.split(KEY)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
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

        return images[0]
