# AI Mask Prototype

Prototype based on Meta's [Segment Anything Model](https://github.com/facebookresearch/segment-anything/) implemented in pure C/C++ using [GGML](https://ggml.ai/) tensor library for machine learning.

## Description

The prototype currently supports only the [ViT-B SAM model checkpoint](https://huggingface.co/facebook/sam-vit-base).

## Build

Clone repository

```bash
git clone git@github.com:andriiryzhkov/ai_mask_prototype.git
cd ggml
git submodule update --init --recursive
```

Install python dependencies in a virtual environment

```bash
poetry install
```

Download PTH model

```bash
curl --create-dirs --output-dir weights -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Convert PTH model to ggml

```bash
poetry run python scripts/convert-pth-to-ggml.py weights/sam_vit_b_01ec64.pth checkpoints/ 1
```

Build

```bash
mkdir build && cd build
cmake -G Ninja ..
cmake --build . --config Release -j 8
```

## Run

Run command line inference

```bash
./bin/sam_cli -t 12 -i ../example1.jpg -p "414, 162" -m ../weights/ggml-model-f16.bin 
./bin/sam_cli -t 12 -i ../example2.jpg -p "3860, 2600" -m ../weights/ggml-model-f16.bin 
```

## License

GPL3

## References

- [ggml](https://github.com/ggerganov/ggml)
- [SAM](https://segment-anything.com/)
- [SAM demo](https://segment-anything.com/demo)
