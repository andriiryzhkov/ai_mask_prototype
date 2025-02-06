#ifndef SAM_H
#define SAM_H

#pragma once

#define _USE_MATH_DEFINES // for M_PI
#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows

#include <string>
#include <vector>
#include <thread>
#include <cinttypes>

struct sam_point {
    float x;
    float y;
    int label;
};

// RGB uint8 image
struct sam_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> data;
};

struct sam_params {
    int32_t seed      = -1; // RNG seed
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    std::string model     = "sam_vit_b-ggml-model-f16.bin"; // model path
    std::string fname_inp = "img.jpg";
    std::string fname_out = "img";
    float   mask_threshold            = 0.f;
    float   iou_threshold             = 0.88f;
    float   stability_score_threshold = 0.95f;
    float   stability_score_offset    = 1.0f;
    float   eps                       = 1e-6f;
    float   eps_decoder_transformer   = 1e-5f;
    sam_point pt = { 414.375f, 162.796875f, 1 };
}; 

// struct sam_state;
// struct sam_model;
struct sam_ggml_state;
struct sam_ggml_model;
struct sam_state {
    std::unique_ptr<sam_ggml_state> state;
    std::unique_ptr<sam_ggml_model> model;
    int t_load_ms = 0;
    int t_compute_img_ms = 0;
    int t_compute_masks_ms = 0;
};

// load the model's weights from a file
std::shared_ptr<sam_state> sam_load_model(
    const sam_params & params);

bool sam_compute_embd_img(
    sam_image_u8 & img,
    int n_threads,
    sam_state & state);

// returns masks sorted by the sum of the iou_score 
// and stability_score in descending order
std::vector<sam_image_u8> sam_compute_masks(
    sam_image_u8 & img,
    int n_threads,
    std::vector<sam_point> points,
    sam_state & state,
    int mask_on_val = 255,
    int mask_off_val = 0);

void sam_deinit(
    sam_state & state);

#endif // SAM_H