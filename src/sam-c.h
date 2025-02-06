#ifndef SAM_C_H
#define SAM_C_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to internal state
typedef struct sam_context_t sam_context_t;

typedef struct sam_point_t {
    float x;
    float y;
    int label;
} sam_point_t;

typedef struct sam_image_t {
    int nx;
    int ny;
    uint8_t* data;  // RGB format
} sam_image_t;

typedef struct sam_params_t {
    int32_t seed;
    int32_t n_threads;
    const char* model;
    const char* fname_inp;
    const char* fname_out;
    float mask_threshold;
    float iou_threshold;
    float stability_score_threshold;
    float stability_score_offset;
    float eps;
    float eps_decoder_transformer;
    sam_point_t pt;
} sam_params_t;

// Initialize default parameters
void sam_params_init(sam_params_t* params);

// Load the model and return a context
sam_context_t* sam_load_model(const sam_params_t* params);

// Compute image embeddings
bool sam_compute_image_embeddings(sam_context_t* ctx, sam_image_t* img, int n_threads);

// Compute masks for given point
// Returns array of masks and writes number of masks to n_masks
sam_image_t* sam_compute_masks(sam_context_t* ctx, const sam_image_t* img, int n_threads,
                              const sam_point_t* points, int n_points, int* n_masks,
                              int mask_on_val, int mask_off_val);

// Free a mask array returned by sam_compute_masks
void sam_free_masks(sam_image_t* masks, int n_masks);

// Free the context and associated resources
void sam_free(sam_context_t* ctx);

// Get timing information
void sam_get_timings(sam_context_t* ctx, int* t_load_ms, int* t_compute_img_ms, int* t_compute_masks_ms);

#ifdef __cplusplus
}
#endif

#endif // SAM_C_H