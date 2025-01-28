#include "sam.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <thread>
#include <cinttypes>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

bool sam_image_load_from_file(const std::string & fname, sam_image_u8 & img) {
    int nx, ny, nc;
    auto data = stbi_load(fname.c_str(), &nx, &ny, &nc, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
        return false;
    }

    img.nx = nx;
    img.ny = ny;
    img.data.resize(nx * ny * 3);
    memcpy(img.data.data(), data, nx * ny * 3);

    stbi_image_free(data);

    return true;
}

bool sam_image_write_to_file(const std::string & fname, std::vector<sam_image_u8> masks) {
    for (size_t i = 0; i < masks.size(); i++) {
        if (masks[i].data.empty()) {
            fprintf(stderr, "%s: mask data is empty for mask %zu\n", __func__, i);
            return false;
        }

        const std::string fname_i = fname + std::to_string(i) + ".png";
        if (!stbi_write_png(fname_i.c_str(), masks[i].nx, masks[i].ny, 3, masks[i].data.data(), masks[i].nx * 3)) {
            fprintf(stderr, "%s: failed to write '%s'\n", __func__, fname_i.c_str());
            return false;
        } else {
            fprintf(stderr, "%s: wrote maks to '%s'\n", __func__, fname_i.c_str());
        }
    }

    return true;
}

void sam_print_usage(int argc, char ** argv, const sam_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -i FNAME, --inp FNAME\n");
    fprintf(stderr, "                        input file (default: %s)\n", params.fname_inp.c_str());
    fprintf(stderr, "  -o FNAME, --out FNAME\n");
    fprintf(stderr, "                        mask file name prefix (default: %s)\n", params.fname_out.c_str());
    fprintf(stderr, "SAM hyperparameters:\n");
    fprintf(stderr, "  -mt FLOAT, --mask-threshold\n");
    fprintf(stderr, "                        mask threshold (default: %f)\n", params.mask_threshold);
    fprintf(stderr, "  -it FLOAT, --iou-threshold\n");
    fprintf(stderr, "                        iou threshold (default: %f)\n", params.iou_threshold);
    fprintf(stderr, "  -st FLOAT, --score-threshold\n");
    fprintf(stderr, "                        score threshold (default: %f)\n", params.stability_score_threshold);
    fprintf(stderr, "  -so FLOAT, --score-offset\n");
    fprintf(stderr, "                        score offset (default: %f)\n", params.stability_score_offset);
    fprintf(stderr, "  -e FLOAT, --epsilon\n");
    fprintf(stderr, "                        epsilon (default: %f)\n", params.eps);
    fprintf(stderr, "  -ed FLOAT, --epsilon-decoder-transformer\n");
    fprintf(stderr, "                        epsilon decoder transformer (default: %f)\n", params.eps_decoder_transformer);
    fprintf(stderr, "SAM prompt:\n");
    fprintf(stderr, "  -p TUPLE, --point-prompt\n");
    fprintf(stderr, "                        point to be used as prompt for SAM (default: %f,%f). Must be in a format FLOAT,FLOAT \n", params.pt.x, params.pt.y);
    fprintf(stderr, "\n");
}

bool sam_params_parse(int argc, char ** argv, sam_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-i" || arg == "--inp") {
            params.fname_inp = argv[++i];
        } else if (arg == "-o" || arg == "--out") {
            params.fname_out = argv[++i];
        } else if (arg == "-mt" || arg == "--mask-threshold") {
            params.mask_threshold = std::stof(argv[++i]);
        } else if (arg == "-it" || arg == "--iou-threshold") {
            params.iou_threshold = std::stof(argv[++i]);
        } else if (arg == "-st" || arg == "--score-threshold") {
            params.stability_score_threshold = std::stof(argv[++i]);
        } else if (arg == "-so" || arg == "--score-offset") {
            params.stability_score_offset = std::stof(argv[++i]);
        } else if (arg == "-e" || arg == "--epsilon") {
            params.eps = std::stof(argv[++i]);
        } else if (arg == "-ed" || arg == "--epsilon-decoder-transformer") {
            params.eps_decoder_transformer = std::stof(argv[++i]);
        } else if (arg == "-p" || arg == "--point-prompt") {
            // TODO multiple points per model invocation
            char* point = argv[++i];

            char* coord = strtok(point, ",");
            if (!coord){
                fprintf(stderr, "Error while parsing prompt!\n");
                exit(1);
            }
            params.pt.x = std::stof(coord);
            coord = strtok(NULL, ",");
            if (!coord){
                fprintf(stderr, "Error while parsing prompt!\n");
                exit(1);
            }
            params.pt.y = std::stof(coord);
        } else if (arg == "-h" || arg == "--help") {
            sam_print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            sam_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    const int64_t t_main_start_us = ggml_time_us();

    sam_params params;
    params.model = "ggml-model-f16.bin";

    sam_model model;
    sam_state state;
    int64_t t_load_us = 0;

    sam_image_u8 img0;

    if (sam_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }
    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    // load the image
    if (!sam_image_load_from_file(params.fname_inp, img0)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, params.fname_inp.c_str(), img0.nx, img0.ny);

    // load the model
    if (!sam_model_load(params, model, state)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        return 1;
    }

    // encode image
    if (!sam_compute_embd_img(img0, params.n_threads, model, state)) {
        fprintf(stderr, "%s: failed to encode image\n", __func__);
        return 1;
    }

    // decode prompt
    std::vector<sam_image_u8> masks = sam_compute_masks(img0,
        params.n_threads, params.pt, model, state);

    // write masks to file
    if (!sam_image_write_to_file(params.fname_out, masks)) {
        fprintf(stderr, "%s: failed to write amsks to '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }

    // report timing
    {
        fprintf(stderr, "\n\n");
        fprintf(stderr, "%s:     load time = %i ms\n", __func__, state.t_load_ms);
        fprintf(stderr, "%s:    total time = %i ms\n", __func__, state.t_load_ms + state.t_compute_img_ms + state.t_compute_masks_ms);
    }

    sam_deinit(model, state);

    return 0;
}
