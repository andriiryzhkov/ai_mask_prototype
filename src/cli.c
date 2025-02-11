#include "sam-c.h"
#include "sam-config.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// Load image using stb_image
bool sam_image_load_from_file(const char* fname, sam_image_t* img) {
    int nx, ny, nc;
    uint8_t* data = stbi_load(fname, &nx, &ny, &nc, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname);
        return false;
    }

    img->nx = nx;
    img->ny = ny;
    img->data = (uint8_t*)malloc(nx * ny * 3);
    if (!img->data) {
        stbi_image_free(data);
        return false;
    }
    memcpy(img->data, data, nx * ny * 3);

    stbi_image_free(data);
    return true;
}

// Write masks to files
bool sam_image_write_to_file(const char* fname_prefix, sam_image_t* masks, int n_masks) {
    char fname[1024];
    
    for (int i = 0; i < n_masks; i++) {
        if (!masks[i].data) {
            fprintf(stderr, "%s: mask data is empty for mask %d\n", __func__, i);
            return false;
        }

        if (!fname_prefix) {
            fprintf(stderr, "%s: filename prefix is empty\n", __func__);
            return false;
        }

        snprintf(fname, sizeof(fname), "%s%d.png", fname_prefix, i);

        if (!stbi_write_png(fname, masks[i].nx, masks[i].ny, 1, masks[i].data, masks[i].nx * 1)) {
            fprintf(stderr, "%s: failed to write '%s'\n", __func__, fname);
            return false;
        } else {
            fprintf(stderr, "%s: wrote mask to '%s'\n", __func__, fname);
        }
    }

    return true;
}

void sam_print_usage(const char* program_name, const sam_params_t* params) {
    fprintf(stderr, "usage: %s [options]\n", program_name);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params->n_threads);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params->model);
    fprintf(stderr, "  -i FNAME, --inp FNAME\n");
    fprintf(stderr, "                        input file (default: %s)\n", params->fname_inp);
    fprintf(stderr, "  -o FNAME, --out FNAME\n");
    fprintf(stderr, "                        mask file name prefix (default: %s)\n", params->fname_out);
    fprintf(stderr, "SAM hyperparameters:\n");
    fprintf(stderr, "  -mt FLOAT, --mask-threshold\n");
    fprintf(stderr, "                        mask threshold (default: %f)\n", params->mask_threshold);
    fprintf(stderr, "  -it FLOAT, --iou-threshold\n");
    fprintf(stderr, "                        iou threshold (default: %f)\n", params->iou_threshold);
    fprintf(stderr, "  -st FLOAT, --score-threshold\n");
    fprintf(stderr, "                        score threshold (default: %f)\n", params->stability_score_threshold);
    fprintf(stderr, "  -so FLOAT, --score-offset\n");
    fprintf(stderr, "                        score offset (default: %f)\n", params->stability_score_offset);
    fprintf(stderr, "  -e FLOAT, --epsilon\n");
    fprintf(stderr, "                        epsilon (default: %f)\n", params->eps);
    fprintf(stderr, "  -ed FLOAT, --epsilon-decoder-transformer\n");
    fprintf(stderr, "                        epsilon decoder transformer (default: %f)\n", params->eps_decoder_transformer);
    fprintf(stderr, "SAM prompt:\n");
    fprintf(stderr, "  -p TUPLE, --point-prompt\n");
    fprintf(stderr, "                        point to be used as prompt for SAM (default: %f, %f, %d). Must be in a format FLOAT, FLOAT, INT \n", 
        params->pt.x, params->pt.y, params->pt.label);
    fprintf(stderr, "\n");
}

bool sam_params_parse(int argc, char** argv, sam_params_t* params) {
    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];

        if (strcmp(arg, "-s") == 0 || strcmp(arg, "--seed") == 0) {
            params->seed = atoi(argv[++i]);
        } else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--threads") == 0) {
            params->n_threads = atoi(argv[++i]);
        } else if (strcmp(arg, "-m") == 0 || strcmp(arg, "--model") == 0) {
            params->model = argv[++i];
        } else if (strcmp(arg, "-i") == 0 || strcmp(arg, "--inp") == 0) {
            params->fname_inp = argv[++i];
        } else if (strcmp(arg, "-o") == 0 || strcmp(arg, "--out") == 0) {
            params->fname_out = argv[++i];
        } else if (strcmp(arg, "-mt") == 0 || strcmp(arg, "--mask-threshold") == 0) {
            params->mask_threshold = (float)atof(argv[++i]);
        } else if (strcmp(arg, "-it") == 0 || strcmp(arg, "--iou-threshold") == 0) {
            params->iou_threshold = (float)atof(argv[++i]);
        } else if (strcmp(arg, "-st") == 0 || strcmp(arg, "--score-threshold") == 0) {
            params->stability_score_threshold = (float)atof(argv[++i]);
        } else if (strcmp(arg, "-so") == 0 || strcmp(arg, "--score-offset") == 0) {
            params->stability_score_offset = (float)atof(argv[++i]);
        } else if (strcmp(arg, "-e") == 0 || strcmp(arg, "--epsilon") == 0) {
            params->eps = (float)atof(argv[++i]);
        } else if (strcmp(arg, "-ed") == 0 || strcmp(arg, "--epsilon-decoder-transformer") == 0) {
                params->eps_decoder_transformer = (float)atof(argv[++i]);
        } else if (strcmp(arg, "-p") == 0 || strcmp(arg, "--point-prompt") == 0) {
            char* point = argv[++i];
            char* coord = strtok(point, ",");
            if (!coord) {
                fprintf(stderr, "Error while parsing prompt!\n");
                return false;
            }
            params->pt.x = (float)atof(coord);
            
            coord = strtok(NULL, ",");
            if (!coord) {
                fprintf(stderr, "Error while parsing prompt!\n");
                return false;
            }
            params->pt.y = (float)atof(coord);
            
            coord = strtok(NULL, ",");
            if (!coord) {
                fprintf(stderr, "Error while parsing prompt!\n");
                return false;
            }
            params->pt.label = atoi(coord);
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            sam_print_usage(argv[0], params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg);
            sam_print_usage(argv[0], params);
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv) {
    sam_params_t params;
    sam_params_init(&params);
    get_params_from_config_file(&params);

    sam_image_t img = {0};
    int n_masks = 0;
    sam_image_t* masks = NULL;

    if (!sam_params_parse(argc, argv, &params)) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = (int32_t)time(NULL);
    }
    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    // Load the image
    if (!sam_image_load_from_file(params.fname_inp, &img)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp);
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, params.fname_inp, img.nx, img.ny);

    // Load the model
    sam_context_t* ctx = sam_load_model(&params);
    if (!ctx) {
        fprintf(stderr, "%s: failed to load model\n", __func__);
        free(img.data);
        return 1;
    }

    // Encode image
    if (!sam_compute_image_embeddings(ctx, &img, params.n_threads)) {
        fprintf(stderr, "%s: failed to encode image\n", __func__);
        free(img.data);
        sam_free(ctx);
        return 1;
    }

    // Decode prompt
    masks = sam_compute_masks(ctx, &img, params.n_threads, &params.pt, 1, &n_masks, 255, 0);
    if (!masks || n_masks == 0) {
        fprintf(stderr, "%s: failed to compute masks\n", __func__);
        free(img.data);
        sam_free(ctx);
        return 1;
    }

    // Write masks to file
    if (!sam_image_write_to_file(params.fname_out, masks, n_masks)) {
        fprintf(stderr, "%s: failed to write masks to '%s'\n", __func__, params.fname_out);
        free(img.data);
        sam_free_masks(masks, n_masks);
        sam_free(ctx);
        return 1;
    }

    // Report timing
    int t_load_ms = 0, t_compute_img_ms = 0, t_compute_masks_ms = 0;
    sam_get_timings(ctx, &t_load_ms, &t_compute_img_ms, &t_compute_masks_ms);
    fprintf(stderr, "\n\n");
    fprintf(stderr, "%s:     load time = %d ms\n", __func__, t_load_ms);
    fprintf(stderr, "%s:    total time = %d ms\n", __func__, 
            t_load_ms + t_compute_img_ms + t_compute_masks_ms);

    // Cleanup
    free(img.data);
    sam_free_masks(masks, n_masks);
    sam_free(ctx);

    return 0;
}