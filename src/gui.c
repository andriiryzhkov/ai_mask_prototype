#include "sam-c.h"
#include "sam-config.h"

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

typedef struct app_context {
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
    GtkWidget* clear_button;
    GtkWidget* save_button;
    GtkWidget* spinner;      // Spinner widget for loading indication
    gboolean computing;      // Flag to track if we're computing embeddings
    GdkPixbuf* mask_overlay;  // For displaying the computed mask
    GtkWidget* encoding_overlay;     // Overlay widget for encoding message
    GtkWidget* encoding_label;       // Label for encoding message
    gboolean is_encoding;            // Flag to track encoding state
} app_context;

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
    if (ctx->mask_overlay) g_object_unref(ctx->mask_overlay);
}

// Create new app_context
static app_context* app_context_new(void) {
    app_context* ctx = malloc(sizeof(app_context));
    if (!ctx) {
        g_print("Failed to allocate memory for context\n"); // More specific error message
        return NULL;
    }

    ctx->image_filename = NULL;
    ctx->image_width = 0;
    ctx->image_height = 0;
    ctx->image_pixbuf = NULL;
    ctx->image_scale = 1.0;
    ctx->points = g_array_new(FALSE, TRUE, sizeof(click_point));
    ctx->sam_ctx = NULL;
    ctx->current_image = NULL;
    ctx->mask = NULL;
    ctx->computing = FALSE;
    ctx->mask_overlay = NULL;
    ctx->window = NULL;
    ctx->drawing_area = NULL;
    ctx->load_button = NULL;
    ctx->clear_button = NULL;
    ctx->save_button = NULL;
    ctx->spinner = NULL;
    ctx->encoding_overlay = NULL;
    ctx->encoding_label = NULL;
    ctx->is_encoding = FALSE;
        
    sam_params_init(&ctx->sam_params);
    if (!get_params_from_config_file(&ctx->sam_params)) {
        g_print("Failed to initialize context due to config loading error\n");
        app_context_free(ctx);
        return NULL;
    }

    return ctx;
}

// Clear click points
static void app_context_clear_points(app_context* ctx) {
    g_array_set_size(ctx->points, 0);
    gtk_widget_queue_draw(ctx->drawing_area);
}

// Clear image data
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
    if (ctx->mask_overlay) {
        g_object_unref(ctx->mask_overlay);
        ctx->mask_overlay = NULL;
    }
}

// Clear all prompts and mask
static void clear_prompts(app_context* ctx) {
    // Clear points
    g_array_set_size(ctx->points, 0);
    
    // Clear mask
    if (ctx->mask) {
        free(ctx->mask->data);
        free(ctx->mask);
        ctx->mask = NULL;
    }
    
    // Clear mask overlay
    if (ctx->mask_overlay) {
        g_object_unref(ctx->mask_overlay);
        ctx->mask_overlay = NULL;
    }
    
    // Redraw
    gtk_widget_queue_draw(ctx->drawing_area);
}

// Load SAM model from file
static gboolean load_sam_model(app_context* ctx) {
    if (!ctx || !ctx->sam_params.model || !*ctx->sam_params.model) {
        g_print("Invalid model path\n");
        return FALSE;
    }

    if (ctx->sam_ctx) return TRUE;  // Already loaded

    g_print("Model: %s\n", ctx->sam_params.model);
    
    ctx->sam_ctx = sam_load_model(&ctx->sam_params);
    return ctx->sam_ctx != NULL;
}

// Show/hide encoding overlay
static void set_encoding_state(app_context* ctx, gboolean encoding) {
    ctx->is_encoding = encoding;
    if (encoding) {
        gtk_widget_show(ctx->encoding_overlay);
        gtk_widget_show(ctx->encoding_label);
    } else {
        gtk_widget_hide(ctx->encoding_overlay);
        gtk_widget_hide(ctx->encoding_label);
    }
    
    // Block/unblock mouse events on drawing area
    gtk_widget_set_sensitive(ctx->drawing_area, !encoding);
    
    // Process events to update UI
    while (gtk_events_pending()) gtk_main_iteration();
}

// Modify compute_image_embedding to run in background
static gboolean compute_image_embedding_idle(gpointer user_data) {
    app_context* ctx = (app_context*)user_data;
    
    // Compute embeddings
    gboolean result = sam_compute_image_embeddings(ctx->sam_ctx, ctx->current_image, 
                                                 ctx->sam_params.n_threads);
    
    // Update UI in main thread
    set_encoding_state(ctx, FALSE);
    
    return FALSE; // Don't repeat
}

// Start computation of image embeddings
static void start_compute_image_embedding(app_context* ctx) {
    if (!ctx->current_image || !ctx->sam_ctx) return;
    
    // Show overlay and block interaction
    set_encoding_state(ctx, TRUE);

        
    // Schedule computation in background
    g_idle_add(compute_image_embedding_idle, ctx);
}

// Compute mask using click points
static gboolean compute_mask(app_context* ctx, GArray* points) {
    if (!ctx->current_image || !ctx->sam_ctx) return FALSE;

    // Show spinner while computing
    ctx->computing = TRUE;
    gtk_widget_show(ctx->spinner);
    gtk_spinner_start(GTK_SPINNER(ctx->spinner));
    while (gtk_events_pending()) gtk_main_iteration();

    int n_masks = 0;
    sam_image_t* masks = NULL;

    // Create array of sam_point_t from click_points
    sam_point_t* sam_points = malloc(points->len * sizeof(sam_point_t));
    for (guint i = 0; i < points->len; i++) {
        click_point* pt = &g_array_index(points, click_point, i);
        sam_points[i].x = pt->x;
        sam_points[i].y = pt->y;
        sam_points[i].label = pt->label;
    }
    
    masks = sam_compute_masks(ctx->sam_ctx, ctx->current_image, 
                              ctx->sam_params.n_threads,
                              sam_points, points->len,
                              &n_masks, 255, 0);

    // Hide spinner
    ctx->computing = FALSE;
    gtk_spinner_stop(GTK_SPINNER(ctx->spinner));
    gtk_widget_hide(ctx->spinner);

    if (!masks || n_masks == 0) return FALSE;

    // Store the mask in ctx->mask
    if (ctx->mask) {
        free(ctx->mask->data);
        free(ctx->mask);
    }
    ctx->mask = malloc(sizeof(sam_image_t));
    ctx->mask->nx = masks[0].nx;
    ctx->mask->ny = masks[0].ny;
    ctx->mask->data = malloc(masks[0].nx * masks[0].ny);
    memcpy(ctx->mask->data, masks[0].data, masks[0].nx * masks[0].ny);

    // Clear previous mask overlay
    if (ctx->mask_overlay) {
        g_object_unref(ctx->mask_overlay);
        ctx->mask_overlay = NULL;
    }

    // Create new mask overlay
    ctx->mask_overlay = gdk_pixbuf_new(GDK_COLORSPACE_RGB, TRUE, 8, 
                                      ctx->image_width, ctx->image_height);
    
    // Fill mask overlay with transparent black
    guchar* pixels = gdk_pixbuf_get_pixels(ctx->mask_overlay);
    int stride = gdk_pixbuf_get_rowstride(ctx->mask_overlay);
    
    for (int y = 0; y < ctx->image_height; y++) {
        for (int x = 0; x < ctx->image_width; x++) {
            guchar* p = pixels + y * stride + x * 4;
            if (masks[0].data[y * ctx->image_width + x] > 0) {
                p[0] = 135;    // R
                p[1] = 206;    // G
                p[2] = 235;    // B
                p[3] = 128;  // A (50% opacity)
            } else {
                p[0] = p[1] = p[2] = p[3] = 0;  // Fully transparent
            }
        }
    }

    sam_free_masks(masks, n_masks);
    gtk_widget_queue_draw(ctx->drawing_area);
    return TRUE;
}

// Callbacks on "Load image" button click
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
            
            gtk_widget_queue_draw(ctx->drawing_area);

            // Start background computation
            start_compute_image_embedding(ctx);
        }
        
        g_free(filename);
    }
    
    gtk_widget_destroy(dialog);
}

// Callbacks on "Clear prompts" button click
static void on_clear_button_clicked(GtkButton* button, app_context* ctx) {
    clear_prompts(ctx);
}

// Callbacks on "Save mask" button click
static void on_save_button_clicked(GtkButton* button, app_context* ctx) {
    if (!ctx->image_pixbuf || ctx->computing) return;

    if (!ctx->mask) {
        GtkWidget* dialog = gtk_message_dialog_new(GTK_WINDOW(ctx->window),
                                                 GTK_DIALOG_DESTROY_WITH_PARENT,
                                                 GTK_MESSAGE_ERROR,
                                                 GTK_BUTTONS_CLOSE,
                                                 "No mask available to save!");
        gtk_dialog_run(GTK_DIALOG(dialog));
        gtk_widget_destroy(dialog);
        return;
    }

    GtkWidget* dialog = gtk_file_chooser_dialog_new("Save Mask",
                                                   GTK_WINDOW(ctx->window),
                                                   GTK_FILE_CHOOSER_ACTION_SAVE,
                                                   "_Cancel", GTK_RESPONSE_CANCEL,
                                                   "_Save", GTK_RESPONSE_ACCEPT,
                                                   NULL);

    // Add PNG file filter
    GtkFileFilter* filter = gtk_file_filter_new();
    gtk_file_filter_add_pattern(filter, "*.png");
    gtk_file_filter_set_name(filter, "PNG images");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);
    gtk_file_chooser_set_filter(GTK_FILE_CHOOSER(dialog), filter);
    
    gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(dialog), TRUE);
    
    // Set default filename if we have original image name
    if (ctx->image_filename) {
        char* default_name = g_strconcat("mask_", g_path_get_basename(ctx->image_filename), ".png", NULL);
        gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog), default_name);
        g_free(default_name);
    }
    
    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char* filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        
        // Save the mask directly from ctx->mask
        gboolean success = (stbi_write_png(filename, ctx->mask->nx, ctx->mask->ny, 
                                         1, ctx->mask->data, ctx->mask->nx) != 0);
        
        if (!success) {
            GtkWidget* error_dialog = gtk_message_dialog_new(GTK_WINDOW(ctx->window),
                                                           GTK_DIALOG_DESTROY_WITH_PARENT,
                                                           GTK_MESSAGE_ERROR,
                                                           GTK_BUTTONS_CLOSE,
                                                           "Failed to save mask!");
            gtk_dialog_run(GTK_DIALOG(error_dialog));
            gtk_widget_destroy(error_dialog);
        }
        
        g_free(filename);
    }
    
    gtk_widget_destroy(dialog);
}

// Draw callback for drawing area
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

    // Draw mask overlay if available
    if (ctx->mask_overlay) {
        gdk_cairo_set_source_pixbuf(cr, ctx->mask_overlay, 0, 0);
        cairo_paint(cr);
    }

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

// Callback for button press event on drawing area
static gboolean on_button_press(GtkWidget* widget, GdkEventButton* event, app_context* ctx) {
    if (!ctx->image_pixbuf || ctx->is_encoding) return FALSE;
    
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
        fprintf(stderr, "%s: point (%f, %f, %i)\n", __func__, pt.x, pt.y, pt.label);
        
        g_array_append_val(ctx->points, pt);

        // Compute new mask using this point
        compute_mask(ctx, ctx->points);

        gtk_widget_queue_draw(widget);
    }
    
    return TRUE;
}

// Main function
int main(int argc, char* argv[]) {
    gtk_init(&argc, &argv);
    
    // Create and initialize context
    app_context* ctx = app_context_new();
    if (!ctx) {
        g_print("Failed to initialize context\n");
        return 1;
    }
    // Create main window
    ctx->window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(ctx->window), "AI Object Mask Demo");
    gtk_window_set_default_size(GTK_WINDOW(ctx->window), 1200, 800);
    g_signal_connect(ctx->window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    
    // Create vertical box for layout
    GtkWidget* vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_add(GTK_CONTAINER(ctx->window), vbox);
    
    // Create button box
    GtkWidget* button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_box_pack_start(GTK_BOX(vbox), button_box, FALSE, FALSE, 5);
    
    // Create buttons
    ctx->load_button = gtk_button_new_with_label("Load Image");
    ctx->clear_button = gtk_button_new_with_label("Clear Prompts");
    ctx->save_button = gtk_button_new_with_label("Save Mask");
    gtk_box_pack_start(GTK_BOX(button_box), ctx->load_button, FALSE, FALSE, 5);
    gtk_box_pack_start(GTK_BOX(button_box), ctx->clear_button, FALSE, FALSE, 5);
    gtk_box_pack_start(GTK_BOX(button_box), ctx->save_button, FALSE, FALSE, 5);
    
    g_signal_connect(ctx->load_button, "clicked", G_CALLBACK(on_load_button_clicked), ctx);
    g_signal_connect(ctx->clear_button, "clicked", G_CALLBACK(on_clear_button_clicked), ctx);
    g_signal_connect(ctx->save_button, "clicked", G_CALLBACK(on_save_button_clicked), ctx);

    // Create overlay for drawing area and spinner
    GtkWidget* overlay = gtk_overlay_new();
    gtk_box_pack_start(GTK_BOX(vbox), overlay, TRUE, TRUE, 0);

    // Create drawing area
    ctx->drawing_area = gtk_drawing_area_new();
    gtk_widget_set_size_request(ctx->drawing_area, 400, 300);
    gtk_container_add(GTK_CONTAINER(overlay), ctx->drawing_area);
    
    g_signal_connect(ctx->drawing_area, "draw", G_CALLBACK(on_draw), ctx);
    g_signal_connect(ctx->drawing_area, "button-press-event", G_CALLBACK(on_button_press), ctx);
    gtk_widget_set_events(ctx->drawing_area, gtk_widget_get_events(ctx->drawing_area) | 
                         GDK_BUTTON_PRESS_MASK);

                             // Create encoding overlay
    ctx->encoding_overlay = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_widget_set_halign(ctx->encoding_overlay, GTK_ALIGN_FILL);
    gtk_widget_set_valign(ctx->encoding_overlay, GTK_ALIGN_FILL);
    
    // Set semi-transparent grey background
    GtkStyleContext* style_context = gtk_widget_get_style_context(ctx->encoding_overlay);
    GtkCssProvider* provider = gtk_css_provider_new();
    gtk_css_provider_load_from_data(provider,
        "box { background-color: rgba(128, 128, 128, 0.7); }", -1, NULL);
    gtk_style_context_add_provider(style_context,
        GTK_STYLE_PROVIDER(provider), GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_unref(provider);
    
    // Create and add encoding label
    ctx->encoding_label = gtk_label_new("Encoding image...");
    gtk_widget_set_name(ctx->encoding_label, "encoding-label");
    provider = gtk_css_provider_new();
    gtk_css_provider_load_from_data(provider,
        "label { color: white; font-size: 24px; }", -1, NULL);
    style_context = gtk_widget_get_style_context(ctx->encoding_label);
    gtk_style_context_add_provider(style_context,
        GTK_STYLE_PROVIDER(provider), GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_unref(provider);
    
    gtk_box_pack_start(GTK_BOX(ctx->encoding_overlay), ctx->encoding_label, TRUE, TRUE, 0);
    gtk_overlay_add_overlay(GTK_OVERLAY(overlay), ctx->encoding_overlay);
    
    // Create and position spinner
    ctx->spinner = gtk_spinner_new();
    gtk_widget_set_halign(ctx->spinner, GTK_ALIGN_CENTER);
    gtk_widget_set_valign(ctx->spinner, GTK_ALIGN_CENTER);
    gtk_overlay_add_overlay(GTK_OVERLAY(overlay), ctx->spinner);

    // Load SAM model
    if (!load_sam_model(ctx)) {
        GtkWidget* dialog = gtk_message_dialog_new(GTK_WINDOW(ctx->window),
                                                 GTK_DIALOG_DESTROY_WITH_PARENT,
                                                 GTK_MESSAGE_ERROR,
                                                 GTK_BUTTONS_CLOSE,
                                                 "Failed to load SAM model!");
        gtk_dialog_run(GTK_DIALOG(dialog));
        gtk_widget_destroy(dialog);
        return 1;
    }

    // Show all widgets
    gtk_widget_show_all(ctx->window);
    gtk_widget_hide(ctx->encoding_overlay);  // Initially hidden
    gtk_widget_hide(ctx->encoding_label);    // Initially hidden
    gtk_widget_hide(ctx->spinner);  // Initially hidden
    
    // Start main loop
    gtk_main();
    
    // Cleanup
    app_context_free(ctx);
    free(ctx);

    return 0;
}