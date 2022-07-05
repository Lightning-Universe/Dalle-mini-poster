# 🥑 Dalle-Mini Poster App ⚡️

Use this app to share your research paper results. This app lets you connect a blogpost, arxiv paper, and a jupyter
notebook and even have an interactive demo for people to play with the model. This app also allows industry
practitioners to reproduce your work.

## Getting started

To create a Research Poster you can install this app via the [Lightning CLI](https://lightning.ai/lightning-docs/) or
[use the template](https://docs.github.com/en/articles/creating-a-repository-from-a-template) from GitHub and
manually install the app as mentioned below.

### Installation

#### With Lightning CLI

`lightning install app lightning/dalle_mini`

#### Use GitHub template

Click on the "Use this template" button at the top, name your app repo, and GitHub will create a fork of this app to
your account.

> ![use-template.png](./assets/use-template.png)

You can clone the forked app repo and follow the steps below to install the app.

```
git clone https://github.com/YOUR-USERNAME/LAI-dalle-mini-poster-App.git
cd LAI-dalle-mini-poster-App
pip install -r requirements.txt
pip install -e .
```

Once you have installed the app, you can goto the `LAI-dalle-mini-poster-App` folder and
run `lightning run app app.py --cloud` from terminal.
This will launch the template app in your default browser with tabs containing research paper, blog, Training
logs, and Model Demo.

You should see something like this in your browser:

> ![image](./assets/demo.png)

You can modify the content of this app and customize it to your research.
At the root of this template, you will find [app.py](./app.py) that contains the `ResearchApp` class. This class
provides arguments like a link to a paper, a blog, and whether to launch a Gradio demo. You can read more about what
each of the arguments does in the docstrings.

### Highlights

- Provide the link for paper, blog, or training logger like WandB as an argument, and `ResearchApp` will create a tab
  for each.
- Make a poster for your research by editing the markdown file in the [resources](./resources/poster.md) folder.
- Add interactive model demo with Gradio app, update the gradio component present in the \[research_app (
  ./research_app/components/model_demo.py) folder.
- View a Jupyter Notebook or launch a fully-fledged notebook instance (Sharing a Jupyter Notebook instance can expose
  the cloud instance to security vulnerability.)
- Reorder the tab layout using the `tab_order` argument.

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
