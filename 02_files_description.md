# Data processing

- `datasets.py`. Contains a single class `CaptionDataset`. This is our custom dataset that we use in a data loader.
- `utils.py`. Contains `create_input_files()` to create files that we use in `CaptionDataset`.

# Models

- `models.py`. Contains classes `Encoder`, `Attention` and `DecoderWithAttention`. We use for image captioning Encoder/Decoder model with attention and these are classes that we need for it. Class `Attention` is used inside `DecoderWithAttention` in `forward()` method.

# Captioning

- `caption.py`. Contain 2 functions:
	- `caption_image_beam_search()` - generates captions using pre-trained Encoder/Decoder model. This function uses BEAM search and that's the reason it's a bit involved.
	- `visualize_att()` - this function creates those nice iluustrations that you can find in the paper or in this tutorial. The trick is to use attention weights `alpha` and plot them on the image. 

# Train and misc files

We don't use the following files in this tutorial:

- `train.py`. 
- `eval.py`. 
- `caption.py`
- `utils.py`
- `create_input_files`. 