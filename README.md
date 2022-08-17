# 🥑 Dalle-Mini Poster App ⚡️

This app is a research poster demo of [min-dalle](https://github.com/kuprel/min-dalle) (PyTorch port of Dalle-Mini).
It showcases a notebook, a blog, and a model demo where you generate images from text prompts.

## Getting started

### Installation

#### With Lightning CLI (This method is not activated yet)

`lightning install app lightning/dalle_mini_poster`

#### From GitHub

You can clone the app repo and follow the steps below to install the app.

```
git clone https://github.com/lightning-AI/LAI-dalle-mini-poster-App.git
cd LAI-dalle-mini-poster-App
pip install -r requirements.txt
pip install -e .
```

Once you have installed the app, you can goto the `LAI-dalle-mini-poster-App` folder and
run `lightning run app app.py --cloud` from terminal.
This will launch the Dalle app in your default browser with tabs containing blog, Training
logs, and Model Demo.

You can control the number of generated images using `OUTPUT_IMAGES` environment variable. To generate 4 images you can
do `lightning run app app.py --env OUTPUT_IMAGES=4 --cloud`


### Example

```python
# update app.py at the root of the repo
import lightning as L

poster_dir = "resources"
blog = "https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-Mini-Explained-with-Demo--Vmlldzo4NjIxODA"
github = "https://github.com/borisdayma/dalle-mini"
wandb = "https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-Mega-Training-Journal--VmlldzoxODMxMDI2"
tabs = ["Poster", "Blog", "Notebook Viewer", "Training Logs", "Demo: Generate images from a text prompt"]

app = L.LightningApp(
    ResearchApp(
        poster_dir=poster_dir,
        blog=blog,
        training_log_url=wandb,
        # notebook_path="resources/DALL·E_mini_Inference_pipeline.ipynb",
        launch_gradio=True,
        tab_order=tabs,
        launch_jupyter_lab=False,  # don't launch for public app, can expose to security vulnerability
    )
)

```

## FAQs

1. How to pull from the latest template
   code? [Answer](https://stackoverflow.com/questions/56577184/github-pull-changes-from-a-template-repository)

## Acknowledgement

Credits to [Boris Dayma](https://twitter.com/borisdayma) for this awesome
work [Dalle Mini](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini-Generate-images-from-any-text-prompt--VmlldzoyMDE4NDAy)
and [Brett Kuprel](https://github.com/kuprel) for [min-dalle](https://github.com/kuprel/min-dalle).
