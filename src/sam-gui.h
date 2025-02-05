#ifndef SAM_GUI_H
#define SAM_GUI_H

#include <gtk/gtk.h>
#include "sam-c.h"

typedef struct {
    float x;        // x coordinate on original image
    float y;        // y coordinate on original image
    int label;      // 1 for positive, 0 for negative
} click_point;

typedef struct {
    // Image data
    char* image_filename;
    int image_width;
    int image_height;
    GdkPixbuf* image_pixbuf;
    double image_scale;

    // Click points
    GArray* points;  // Array of click_point structures

    // SAM model data
    sam_context_t* sam_ctx;
    sam_params_t sam_params;
    sam_image_t* current_image;
    sam_image_t* mask;

    // GUI elements
    GtkWidget* window;
    GtkWidget* drawing_area;
    GtkWidget* load_button;
    GtkWidget* save_button;
} app_context;

// Function declarations
void app_context_init(app_context* ctx);
void app_context_free(app_context* ctx);
void app_context_clear_points(app_context* ctx);
void app_context_clear_image(app_context* ctx);
gboolean load_sam_model(app_context* ctx);
gboolean compute_image_embedding(app_context* ctx);
gboolean compute_and_save_mask(app_context* ctx, const char* filename);

#endif // SAM_GUI_H