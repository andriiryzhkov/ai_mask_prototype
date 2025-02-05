#include "sam-c.h"

#include <string.h>
#include <gtk/gtk.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

static void app_context_init(app_context* ctx) {
    ctx->image_filename = NULL;
    ctx->image_width = 0;
    ctx->image_height = 0;
    ctx->image_pixbuf = NULL;
    ctx->image_scale = 1.0;
    ctx->points = g_array_new(FALSE, TRUE, sizeof(click_point));
    ctx->sam_ctx = NULL;
    ctx->current_image = NULL;
    ctx->mask = NULL;
    
    // Initialize SAM parameters
    sam_params_init(&ctx->sam_params);
    ctx->sam_params.model = "ggml-model-f16.bin";
}

static void app_context_free(app_context* ctx) {
    g_free(ctx->image_filename);
    if (ctx->image_pixbuf) g_object_unref(ctx->image_pixbuf);
    g_array_free(ctx->points, TRUE);
    if (ctx->current_image) {
        free(ctx->current_image->data);
        free(ctx->current_image);
    }
    if (ctx->mask) {
        free(ctx->mask->data);
        free(ctx->mask);
    }
    if (ctx->sam_ctx) sam_free(ctx->sam_ctx);
}

static void app_context_clear_points(app_context* ctx) {
    g_array_set_size(ctx->points, 0);
    gtk_widget_queue_draw(ctx->drawing_area);
}

static void app_context_clear_image(app_context* ctx) {
    if (ctx->current_image) {
        free(ctx->current_image->data);
        free(ctx->current_image);
        ctx->current_image = NULL;
    }
    if (ctx->image_pixbuf) {
        g_object_unref(ctx->image_pixbuf);
        ctx->image_pixbuf = NULL;
    }
    g_free(ctx->image_filename);
    ctx->image_filename = NULL;
    ctx->image_width = 0;
    ctx->image_height = 0;
    app_context_clear_points(ctx);
}

static gboolean load_sam_model(app_context* ctx) {
    if (ctx->sam_ctx) return TRUE;  // Already loaded
    
    ctx->sam_ctx = sam_load_model(&ctx->sam_params);
    return ctx->sam_ctx != NULL;
}

static gboolean compute_image_embedding(app_context* ctx) {
    if (!ctx->current_image || !ctx->sam_ctx) return FALSE;
    
    return sam_compute_image_embeddings(ctx->sam_ctx, ctx->current_image, ctx->sam_params.n_threads);
}

static gboolean compute_and_save_mask(app_context* ctx, const char* filename) {
    if (!ctx->current_image || !ctx->sam_ctx || ctx->points->len == 0) return FALSE;

    int n_masks = 0;
    sam_image_t* masks = NULL;
    
    // Convert points to SAM format and compute masks
    for (guint i = 0; i < ctx->points->len; i++) {
        click_point* pt = &g_array_index(ctx->points, click_point, i);
        sam_point_t sam_pt = { pt->x, pt->y };
        
        masks = sam_compute_masks(ctx->sam_ctx, ctx->current_image, 
                                ctx->sam_params.n_threads, sam_pt, &n_masks, 255, 0);
        if (masks && n_masks > 0) break;
    }
    
    if (!masks || n_masks == 0) return FALSE;

    // Save the first mask
    gboolean success = (stbi_write_png(filename, masks[0].nx, masks[0].ny, 
                                     1, masks[0].data, masks[0].nx) != 0);
    
    sam_free_masks(masks, n_masks);
    return success;
}

static void on_load_button_clicked(GtkButton* button, app_context* ctx) {
    GtkWidget* dialog = gtk_file_chooser_dialog_new("Open Image",
                                                   GTK_WINDOW(ctx->window),
                                                   GTK_FILE_CHOOSER_ACTION_OPEN,
                                                   "_Cancel", GTK_RESPONSE_CANCEL,
                                                   "_Open", GTK_RESPONSE_ACCEPT,
                                                   NULL);

    GtkFileFilter* filter = gtk_file_filter_new();
    gtk_file_filter_add_pattern(filter, "*.jpg");
    gtk_file_filter_add_pattern(filter, "*.jpeg");
    gtk_file_filter_set_name(filter, "JPEG images");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char* filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        app_context_clear_image(ctx);
        
        // Load the image
        ctx->image_filename = g_strdup(filename);
        ctx->image_pixbuf = gdk_pixbuf_new_from_file(filename, NULL);
        
        if (ctx->image_pixbuf) {
            ctx->image_width = gdk_pixbuf_get_width(ctx->image_pixbuf);
            ctx->image_height = gdk_pixbuf_get_height(ctx->image_pixbuf);
            
            // Create SAM image
            ctx->current_image = malloc(sizeof(sam_image_t));
            ctx->current_image->nx = ctx->image_width;
            ctx->current_image->ny = ctx->image_height;
            ctx->current_image->data = malloc(ctx->image_width * ctx->image_height * 3);
            
            // Copy image data
            const guchar* pixels = gdk_pixbuf_get_pixels(ctx->image_pixbuf);
            int stride = gdk_pixbuf_get_rowstride(ctx->image_pixbuf);
            int n_channels = gdk_pixbuf_get_n_channels(ctx->image_pixbuf);
            
            for (int y = 0; y < ctx->image_height; y++) {
                for (int x = 0; x < ctx->image_width; x++) {
                    const guchar* p = pixels + y * stride + x * n_channels;
                    guchar* q = ctx->current_image->data + (y * ctx->image_width + x) * 3;
                    q[0] = p[0];  // R
                    q[1] = p[1];  // G
                    q[2] = p[2];  // B
                }
            }
            
            // Compute embeddings
            compute_image_embedding(ctx);
            gtk_widget_queue_draw(ctx->drawing_area);
        }
        
        g_free(filename);
    }
    
    gtk_widget_destroy(dialog);
}

static void on_save_button_clicked(GtkButton* button, app_context* ctx) {
    if (!ctx->current_image || ctx->points->len == 0) return;

    GtkWidget* dialog = gtk_file_chooser_dialog_new("Save Mask",
                                                   GTK_WINDOW(ctx->window),
                                                   GTK_FILE_CHOOSER_ACTION_SAVE,
                                                   "_Cancel", GTK_RESPONSE_CANCEL,
                                                   "_Save", GTK_RESPONSE_ACCEPT,
                                                   NULL);
    
    gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(dialog), TRUE);
    
    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char* filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        compute_and_save_mask(ctx, filename);
        g_free(filename);
    }
    
    gtk_widget_destroy(dialog);
}

static gboolean on_draw(GtkWidget* widget, cairo_t* cr, app_context* ctx) {
    GtkAllocation allocation;
    gtk_widget_get_allocation(widget, &allocation);
    
    // Fill background with grey
    cairo_set_source_rgb(cr, 0.8, 0.8, 0.8);
    cairo_paint(cr);
    
    if (!ctx->image_pixbuf) return FALSE;
    
    // Calculate scaling to fit image while maintaining aspect ratio
    double scale_x = (double)allocation.width / ctx->image_width;
    double scale_y = (double)allocation.height / ctx->image_height;
    ctx->image_scale = MIN(scale_x, scale_y);
    
    int scaled_width = ctx->image_width * ctx->image_scale;
    int scaled_height = ctx->image_height * ctx->image_scale;
    
    // Center the image
    int x_offset = (allocation.width - scaled_width) / 2;
    int y_offset = (allocation.height - scaled_height) / 2;
    
    // Draw the image
    cairo_translate(cr, x_offset, y_offset);
    cairo_scale(cr, ctx->image_scale, ctx->image_scale);
    gdk_cairo_set_source_pixbuf(cr, ctx->image_pixbuf, 0, 0);
    cairo_paint(cr);
    
    // Draw points
    cairo_scale(cr, 1.0/ctx->image_scale, 1.0/ctx->image_scale);
    for (guint i = 0; i < ctx->points->len; i++) {
        click_point* pt = &g_array_index(ctx->points, click_point, i);
        
        // Calculate screen coordinates
        double screen_x = pt->x * ctx->image_scale;
        double screen_y = pt->y * ctx->image_scale;
        
        // Set color based on label
        if (pt->label)
            cairo_set_source_rgb(cr, 0, 1, 0);  // Green for positive
        else
            cairo_set_source_rgb(cr, 1, 0, 0);  // Red for negative
            
        // Draw point
        cairo_arc(cr, screen_x, screen_y, 5.0, 0, 2 * G_PI);
        cairo_fill(cr);
    }
    
    return FALSE;
}

static gboolean on_button_press(GtkWidget* widget, GdkEventButton* event, app_context* ctx) {
    if (!ctx->image_pixbuf) return FALSE;
    
    GtkAllocation allocation;
    gtk_widget_get_allocation(widget, &allocation);
    
    int scaled_width = ctx->image_width * ctx->image_scale;
    int scaled_height = ctx->image_height * ctx->image_scale;
    
    int x_offset = (allocation.width - scaled_width) / 2;
    int y_offset = (allocation.height - scaled_height) / 2;
    
    // Convert screen coordinates to image coordinates
    double image_x = (event->x - x_offset) / ctx->image_scale;
    double image_y = (event->y - y_offset) / ctx->image_scale;
    
    // Check if click is within image bounds
    if (image_x >= 0 && image_x < ctx->image_width &&
        image_y >= 0 && image_y < ctx->image_height) {
        
        click_point pt = {
            .x = image_x,
            .y = image_y,
            .label = (event->button == 1) ? 1 : 0  // Left click = 1, Right click = 0
        };
        
        g_array_append_val(ctx->points, pt);
        gtk_widget_queue_draw(widget);
    }
    
    return TRUE;
}

int main(int argc, char* argv[]) {
    gtk_init(&argc, &argv);
    
    // Create and initialize context
    app_context ctx;
    app_context_init(&ctx);
    
    // Create main window
    ctx.window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(ctx.window), "SAM GUI");
    gtk_window_set_default_size(GTK_WINDOW(ctx.window), 800, 600);
    g_signal_connect(ctx.window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    
    // Create vertical box for layout
    GtkWidget* vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_add(GTK_CONTAINER(ctx.window), vbox);
    
    // Create button box
    GtkWidget* button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_box_pack_start(GTK_BOX(vbox), button_box, FALSE, FALSE, 5);
    
    // Create buttons
    ctx.load_button = gtk_button_new_with_label("Load Image");
    ctx.save_button = gtk_button_new_with_label("Save Mask");
    gtk_box_pack_start(GTK_BOX(button_box), ctx.load_button, FALSE, FALSE, 5);
    gtk_box_pack_start(GTK_BOX(button_box), ctx.save_button, FALSE, FALSE, 5);
    
    g_signal_connect(ctx.load_button, "clicked", G_CALLBACK(on_load_button_clicked), &ctx);
    g_signal_connect(ctx.save_button, "clicked", G_CALLBACK(on_save_button_clicked), &ctx);

    // Create drawing area
    ctx.drawing_area = gtk_drawing_area_new();
    gtk_widget_set_size_request(ctx.drawing_area, 400, 300);
    gtk_box_pack_start(GTK_BOX(vbox), ctx.drawing_area, TRUE, TRUE, 0);
    
    g_signal_connect(ctx.drawing_area, "draw", G_CALLBACK(on_draw), &ctx);
    g_signal_connect(ctx.drawing_area, "button-press-event", G_CALLBACK(on_button_press), &ctx);
    gtk_widget_set_events(ctx.drawing_area, gtk_widget_get_events(ctx.drawing_area) | 
                         GDK_BUTTON_PRESS_MASK);

    // Load SAM model
    if (!load_sam_model(&ctx)) {
        GtkWidget* dialog = gtk_message_dialog_new(GTK_WINDOW(ctx.window),
                                                 GTK_DIALOG_DESTROY_WITH_PARENT,
                                                 GTK_MESSAGE_ERROR,
                                                 GTK_BUTTONS_CLOSE,
                                                 "Failed to load SAM model!");
        gtk_dialog_run(GTK_DIALOG(dialog));
        gtk_widget_destroy(dialog);
        return 1;
    }

    // Show all widgets
    gtk_widget_show_all(ctx.window);
    
    // Start main loop
    gtk_main();
    
    // Cleanup
    app_context_free(&ctx);
    
    return 0;
}