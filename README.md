# dalle-mini generations sharing

To compare sampling parameters (top_k, top_p, temperature, condition_scale) I generated >70,000 images with the *mega-1:v16* [dalle-mini](https://github.com/borisdayma/dalle-mini) model.

[JPEG compressed versions can be browsed at http://muh.freedu.ms/dm-sampling](http://muh.freedu.ms/dm-sampling). To regenerate the lossless originals, see the instructions below.

---

I've tested all combinations of the following parameters:

* **top_k:** *50*, 128, 256, 0 (no limit)
* **top_p:** 0.9, 0.95, 0.99, *1.0*
* **temperature:** 0.4, 0.7, 0.9, *1.0*, 3.0
* **condition_scale:** 1.0 (no super conditioning), 3.0, *10.0*, 30.0

(*italics* means default as of July 12, 2022 - these may change in the future)

On the following prompts from the [original test collection](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini-Generate-Images-from-Any-Text-Prompt--VmlldzoyMDE4NDAy#results-from-latest-model):

1. painting of a forest full of elves and fairies
2. a facebook-branded dinosaur
3. a rocket in the shape of the Eiffel tower taking off
4. times square underwater, times square at the bottom of the ocean
5. a picture of a castle from minecraft
6. happy, happiness
7. a cute avocado armchair singing karaoke on stage in front of a crowd of strawberry shaped lamps

### Usage

#### Decoding

You need a Python 3 environment with [JAX](https://github.com/google/jax#installation), packages in [`requirements.txt`](requirements.txt) plus Pillow or Pillow-SIMD. Decoding all 70,000 images takes 25 minutes on an RTX 3090, 75 minutes on Colab's T4. CPU decoding works but is orders of magnitude slower.

**Warning: This produces a folder structure with >70,000 files totalling over 7 GB.**

Run `python3 dm_generations_decode.py --db dm_generations_sampling.db [options]`. See the script for details. Be patient, Options you may find useful:

* `--batch-size`: number of images to decode at once (default 32 - reduce this if you run out of VRAM)
* `--num-workers`: worker threads for PNG encoding (default is [concurrent.futures.ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor)'s - probably too high below Python 3.8)
* `--mode`: `images` (default) produces one file per generated image (needed for the sample browser), `gallery` creates grids showing all images generated with the same settings, sorted by CLIP score (good for browsing in an image viewer)
* `--prompt-ids`: space-separated list of prompts you're interested in (default is all, numbering follows list above)
* `--cpu-only`: enjoy the wait...

Alternatively, [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdaxxy/dm_generations_decode/blob/master/dm_generations_decode.ipynb).

#### Viewing

After generating the images, open *samplebrowser.html*.

### Raw data

`dm_generations_sampling.db` is a SQLite database holding the results. Notes on the database layout:

* If `batch` is DalleBart's output, `quantvec = jax.device_get(batch[i]).astype('<u2').tobytes(); quantvec_md5 = hashlib.md5(quantvec).hexdigest()`
* `clip_score_input` is the image-to-prompt accuracy score calculated with [https://huggingface.co/openai/clip-vit-base-patch32](openai/clip-vit-base-patch32) as in the [dalle-mini example notebook](https://github.com/borisdayma/dalle-mini/blob/ec07a902e440077efc84dde401ac6f65af5c0a09/tools/inference/inference_pipeline.ipynb)
* Due to text normalization differences, what CLIP reads isn't always *exactly* what DalleBart reads. `clip_score_tokenized` is the score for `processor.tokenizer.batch_decode(processor(prompt), skip_special_tokens=True)` instead of just `prompt`. All rankings in this experiment use `clip_score_input`, though.
* Times are UTC Unix timestamps in seconds
* I've made [several](https://github.com/borisdayma/dalle-mini/issues/247) [performance](https://github.com/borisdayma/dalle-mini/issues/247#issuecomment-1158484392) [tweaks](https://github.com/borisdayma/dalle-mini/pull/269) in the code that generated these results; an unfortunate side effect is that using the RNG seeds (key0, key1) and input arguments with unmodified dalle-mini code will not reproduce them.