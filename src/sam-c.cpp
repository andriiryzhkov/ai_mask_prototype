#include "sam-c.h"
#include "sam.h"
#include <memory>
#include <cstring>

struct sam_context_t {
    std::shared_ptr<sam_state> state;
};

void sam_params_init(sam_params_t* params) {
    if (!params) return;
    
    sam_params cpp_params;
    params->seed = cpp_params.seed;
    params->n_threads = cpp_params.n_threads;
    params->model = "ggml-model-f16.bin";
    params->fname_inp = "img.jpg";
    params->fname_out = "img";
    params->mask_threshold = cpp_params.mask_threshold;
    params->iou_threshold = cpp_params.iou_threshold;
    params->stability_score_threshold = cpp_params.stability_score_threshold;
    params->stability_score_offset = cpp_params.stability_score_offset;
    params->eps = cpp_params.eps;
    params->eps_decoder_transformer = cpp_params.eps_decoder_transformer;
    params->pt = {cpp_params.pt.x, cpp_params.pt.y};
}

sam_context_t* sam_load_model(const sam_params_t* params) {
    if (!params) return nullptr;

    sam_params cpp_params;
    cpp_params.seed = params->seed;
    cpp_params.n_threads = params->n_threads;
    cpp_params.model = params->model ? params->model : cpp_params.model;
    cpp_params.fname_inp = params->fname_inp ? params->fname_inp : cpp_params.fname_inp;
    cpp_params.fname_out = params->fname_out ? params->fname_out : cpp_params.fname_out;
    cpp_params.mask_threshold = params->mask_threshold;
    cpp_params.iou_threshold = params->iou_threshold;
    cpp_params.stability_score_threshold = params->stability_score_threshold;
    cpp_params.stability_score_offset = params->stability_score_offset;
    cpp_params.eps = params->eps;
    cpp_params.eps_decoder_transformer = params->eps_decoder_transformer;
    cpp_params.pt = {params->pt.x, params->pt.y};

    auto state = sam_load_model(cpp_params);
    if (!state) {
        return nullptr;
    }

    auto* ctx = new sam_context_t;
    ctx->state = state;
    return ctx;
}

bool sam_compute_image_embeddings(sam_context_t* ctx, sam_image_t* img, int n_threads) {
    if (!ctx || !ctx->state || !img || !img->data) return false;

    sam_image_u8 cpp_img;
    cpp_img.nx = img->nx;
    cpp_img.ny = img->ny;
    cpp_img.data.assign(img->data, img->data + (img->nx * img->ny * 3));

    return sam_compute_embd_img(cpp_img, n_threads, *ctx->state);
}

sam_image_t* sam_compute_masks(sam_context_t* ctx, const sam_image_t* img, int n_threads,
                              const sam_point_t* points, int n_points, int* n_masks,
                              int mask_on_val, int mask_off_val) {
   if (!ctx || !ctx->state || !img || !img->data || !points || n_points <= 0 || !n_masks) return nullptr;

    sam_image_u8 cpp_img;
    cpp_img.nx = img->nx;
    cpp_img.ny = img->ny;
    cpp_img.data.assign(img->data, img->data + (img->nx * img->ny * 3));

    std::vector<sam_point> cpp_points;
    cpp_points.reserve(n_points);
    for (int i = 0; i < n_points; i++) {
        cpp_points.push_back({points[i].x, points[i].y, points[i].label});
    }

    auto masks = sam_compute_masks(cpp_img, n_threads, cpp_points, *ctx->state, mask_on_val, mask_off_val);
    if (masks.empty()) {
        *n_masks = 0;
        return nullptr;
    }
    
    *n_masks = masks.size();
    auto* result = new sam_image_t[*n_masks];
    
    for (size_t i = 0; i < masks.size(); i++) {
        result[i].nx = masks[i].nx;
        result[i].ny = masks[i].ny;
        result[i].data = new uint8_t[masks[i].data.size()];
        std::memcpy(result[i].data, masks[i].data.data(), masks[i].data.size());
    }

    return result;
}

void sam_free_masks(sam_image_t* masks, int n_masks) {
    if (!masks) return;
    
    for (int i = 0; i < n_masks; i++) {
        delete[] masks[i].data;
    }
    delete[] masks;
}

void sam_free(sam_context_t* ctx) {
    if (!ctx) return;
    
    if (ctx->state) {
        sam_deinit(*ctx->state);
    }
    delete ctx;
}

void sam_get_timings(sam_context_t* ctx, int* t_load_ms, int* t_compute_img_ms, int* t_compute_masks_ms) {
    if (!ctx || !ctx->state) return;

    if (t_load_ms) *t_load_ms = ctx->state->t_load_ms;
    if (t_compute_img_ms) *t_compute_img_ms = ctx->state->t_compute_img_ms;
    if (t_compute_masks_ms) *t_compute_masks_ms = ctx->state->t_compute_masks_ms;
}