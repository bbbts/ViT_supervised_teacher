# Segmenter: Vision Transformer-Based Fully Supervised Semantic Segmentation

This repository contains a Vision Transformer (ViT)-based fully supervised semantic segmentation model trained on the Meta wildfire imagery dataset. The implementation is based on the [Segmenter](https://github.com/rstrudel/segmenter) framework and provides the fully supervised teacher model used as the teacher initialization for the semi-supervised experiments.

## Installation

Define environment variables pointing to your checkpoint and dataset directories, for example:

```bash
export DATASET=/path/to/dataset/dir
```

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

You can also install the repository as a package from the root directory:

```bash
pip install .
```

## Model Zoo

We provide a fully supervised semantic segmentation model trained on the Meta dataset using a Vision Transformer Tiny backbone (`vit_tiny_patch16_384`) with a Mask Transformer decoder.

### Meta Dataset

The Meta dataset uses a unified four-class taxonomy consisting of background, fire, burned area, and water.

| Model | Training Data | Backbone | Decoder | Download |
| --- | --- | --- | --- | --- |
| MODEL_FILE_Meta | 100% labeled | ViT-Tiny-Patch16-384 | Mask Transformer | [model files](https://drive.google.com/drive/u/1/folders/1-lCf_FMXILvCJuo2uVbo8DZrl7qHwdxJ) |

The model folder contains the trained checkpoint, evaluation metrics, training configuration, and training-loss visualization.

### Meta Dataset Download

The Meta dataset used for the experiments is available here:

[Download the Meta dataset](https://drive.google.com/drive/u/1/folders/1cDXOGkIqONxV0w9yTdpo6sUfOpoPRJjZ)

## Inference

Download the trained checkpoint from the Model Zoo and place it in an appropriate folder.

For Meta dataset inference, define the dataset directory:

```bash
export DATASET=/path/to/Datasets/Meta
```

For example, to run inference using the fully supervised Meta teacher model:

```bash
python3 inference.py \
  --model-path /path/to/segmenter_supervised_META/segm/MODEL_FILE_Meta_OLD1/checkpoint.pth \
  --input-dir $DATASET/images/test/ \
  --output-dir /path/to/segmenter_supervised_META/segm/PREDICTION_new/meta/ \
  --dataset meta \
  --gt-dir $DATASET/masks/test/
```

The `--gt-dir` argument can be provided when ground-truth masks are available. The inference script generates segmentation predictions and evaluates them against the corresponding ground-truth masks.

## Train

### Meta Dataset Training

The fully supervised teacher model was trained using the Meta dataset with all available training samples labeled.

The training configuration uses a Vision Transformer Tiny backbone (`vit_tiny_patch16_384`) and a Mask Transformer decoder:

```bash
python3 train.py \
  --log-dir /path/to/segmenter_supervised_META/segm/MODEL_FILE_Meta_OLD1/ \
  --dataset meta \
  --backbone vit_tiny_patch16_384 \
  --decoder mask_transformer \
  --batch-size 4 \
  --epochs 100 \
  --learning-rate 0.001
```

The supervised teacher model was trained using 100% labeled Meta training data. The resulting checkpoint was subsequently used as the teacher initialization for the semi-supervised experiments.

## Logs

The trained model folder contains the following experimental artifacts:

```text
checkpoint.pth
evaluation_metrics.csv
training_losses.png
variant.yml
```

The `checkpoint.pth` file contains the trained model weights, while `variant.yml` stores the model and experiment configuration. The `evaluation_metrics.csv` file contains the evaluation results, and `training_losses.png` provides a visualization of the training loss.

## Attention Maps

To visualize attention maps from the Vision Transformer, you can use the attention-map visualization utilities provided by Segmenter.

For example:

```bash
python -m segm.scripts.show_attn_map checkpoint.pth \
  images/im0.jpg output_dir/ \
  --layer-id 0 \
  --x-patch 0 \
  --y-patch 21 \
  --enc
```

Different options are provided to select the generated attention maps:

* `--enc` or `--dec`: Select encoder or decoder attention maps respectively.
* `--patch` or `--cls`: `--patch` generates attention maps for the patch with coordinates `(x_patch, y_patch)`. `--cls` combined with `--enc` generates attention maps for the CLS token of the encoder. `--cls` combined with `--dec` generates maps for each class embedding of the decoder.
* `--x-patch` and `--y-patch`: Coordinates of the patch to draw attention maps from. This flag is ignored when `--cls` is used.
* `--layer-id`: Select the layer for which the attention map is drawn.

For example, to generate attention maps for the decoder class embeddings:

```bash
python -m segm.scripts.show_attn_map checkpoint.pth \
  images/im0.jpg output_dir/ \
  --layer-id 0 \
  --dec \
  --cls
```

## Video Segmentation

The original Segmenter framework also provides support for zero-shot video segmentation using trained segmentation models.

## BibTex

If you use the original Segmenter framework, please cite:

```bibtex
@article{strudel2021,
  title={Segmenter: Transformer for Semantic Segmentation},
  author={Strudel, Robin and Garcia, Ricardo and Laptev, Ivan and Schmid, Cordelia},
  journal={arXiv preprint arXiv:2105.05633},
  year={2021}
}
```

## Acknowledgements

The Vision Transformer code is based on the [timm](https://github.com/huggingface/pytorch-image-models) library and the semantic segmentation training and evaluation pipeline is based on the [Segmenter](https://github.com/rstrudel/segmenter) framework.

This repository contains the fully supervised Meta-dataset model used as the teacher initialization for the corresponding semi-supervised semantic segmentation experiments.
