from PIL import Image

from research_app.dalle_mini import concat_images


def test_concat_images():
    im = [Image.new(mode="RGB", size=(200, 200))] * 4
    assert concat_images(im).size == (200, 800)
