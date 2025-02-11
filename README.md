# AI Mask Prototype

Prototype based on Meta's [Segment Anything Model](https://github.com/facebookresearch/segment-anything/) implemented in pure C/C++ using [GGML](https://ggml.ai/) tensor library for machine learning.

## Description

The prototype currently supports only the [ViT-B SAM model checkpoint](https://huggingface.co/facebook/sam-vit-base).

## Build

Clone repository

```bash
git clone git@github.com:andriiryzhkov/ai_mask_prototype.git
cd ai_mask_prototype
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
./build.sh
```
or

```bash
mkdir build && cd build
cmake -G Ninja ..
cmake --build . --config Release -j 8
```

## Run

Run command line inference

```bash
./build/bin/sam_cli -t 12 -i ./images/in/example1.jpg -o ./images/out/example1 -p "2070, 1170, 1" -m ./weights/sam_vit_b-ggml-model-f16.bin 
./build/bin/sam_cli -t 12 -i ./images/in/example2.jpg -o ./images/out/example2 -p "650, 700, 1" -m ./weights/sam_vit_b-ggml-model-f16.bin 
```

or 

```bash
./build/bin/sam_cli.exe -i ./images/in/example1.jpg -o ./images/out/example1 -p "2070, 1170, 1"
./build/bin/sam_cli.exe -i ./images/in/example2.jpg -o ./images/out/example2 -p "650, 700, 1" 
```

GTK3 application

```bash
./build/bin/sam_gui
```

or

```bash
./build/bin/sam_gui.exe
```

## Sample images

1. Photo by [Aaron Doucett](https://unsplash.com/@adoucett?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) on [Unsplash](https://unsplash.com/photos/black-and-brown-turtle-on-brown-wood-iz2C8o4zyP4?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash)
2. Photo by [Anoir Chafik](https://unsplash.com/@anoirchafik?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) on [Unsplash](https://unsplash.com/photos/selective-focus-photography-of-three-brown-puppies-2_3c4dIFYFU?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash)

## License

GPL3

## References

- [ggml](https://github.com/ggerganov/ggml)
- [SAM](https://segment-anything.com/)
- [SAM demo](https://segment-anything.com/demo)
- [sam.cpp](https://github.com/YavorGIvanov/sam.cpp)
