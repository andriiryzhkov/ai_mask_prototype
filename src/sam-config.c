#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "sam-c.h"

const char* CONFIG_FILE_PATH = "sam-config.ini";

char* trim(char *s) {
  int i = 0, j = 0;
  while (s[i] == " ") i++; 
  while (s[j++] = s[i++]);
  return s;
}

static char* get_current_path() {
    static char exe_path[1024];
    
#ifdef _WIN32
    // Windows version
    DWORD len = GetModuleFileName(NULL, exe_path, sizeof(exe_path)-1);
    if (len == 0 || len == sizeof(exe_path)-1) {
        fprintf(stderr, "Failed to get executable path\n");
        return NULL;
    }
    exe_path[len] = '\0';
#else
    // Linux/Unix version
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path)-1);
    if (len == -1) {
        fprintf(stderr, "Failed to get executable path\n");
        return NULL;
    }
    exe_path[len] = '\0';
#endif

    // Get directory by removing executable name
    char* last_slash = NULL;
#ifdef _WIN32
    // Windows: look for both forward and backward slashes
    char* last_forward = strrchr(exe_path, '/');
    char* last_backward = strrchr(exe_path, '\\');
    if (last_forward && last_backward) {
        last_slash = (last_forward > last_backward) ? last_forward : last_backward;
    } else {
        last_slash = last_forward ? last_forward : last_backward;
    }
#else
    last_slash = strrchr(exe_path, '/');
#endif

    if (last_slash != NULL) {
        *last_slash = '\0';
    }
    
    return exe_path;
}

static void create_default_config(const char* config_path) {
    FILE* f = fopen(config_path, "w");
    if (!f) {
        fprintf(stderr, "Failed to create default config at: %s\n", config_path);
        return;
    }

    fprintf(f, "# SAM Configuration File\n\n");
    fprintf(f, "# Model path (relative to executable or absolute)\n");
    fprintf(f, "model=sam_vit_b-ggml-model-f16.bin\n\n");
    fprintf(f, "# Number of CPU threads to use\n");
    fprintf(f, "n_threads=10\n\n");
    fprintf(f, "# Model parameters\n");
    fprintf(f, "mask_threshold=0.0\n");
    fprintf(f, "iou_threshold=0.88\n");
    fprintf(f, "stability_score_threshold=0.95\n");
    fprintf(f, "stability_score_offset=1.0\n");
    fprintf(f, "eps=1e-6\n");
    fprintf(f, "eps_decoder_transformer=1e-5\n");

    fclose(f);
}

static bool read_config_file(sam_params_t* sam_params) {
    const char* exe_dir = get_current_path();
    if (!exe_dir) return FALSE;
    
    // Create config file path
    char config_path[1024];
    snprintf(config_path, sizeof(config_path),
             "%s\\%s", exe_dir, CONFIG_FILE_PATH);

    // Try to open config file
    FILE* f = fopen(config_path, "r");
    if (!f) {
        fprintf(stderr, "Config file not found at: %s\n", config_path);
        return FALSE;
    }

    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        // Remove newline
        char* newline = strchr(line, '\n');
        if (newline) *newline = '\0';

        // Skip empty lines and comments
        if (line[0] == '\0' || line[0] == '#') continue;

        char key[256], value[768];
        if (sscanf(line, "%255[^=]=%767s", key, value) == 2) {
            // Trim whitespace
            char* k = trim(key);
            char* v = trim(value);

            if (strcmp(k, "model") == 0) {
                char model_path[1024];
                snprintf(model_path, sizeof(model_path), "%s\\%s", exe_dir, v);
                sam_params->model = strdup(model_path);
            } else if (strcmp(k, "mask_threshold") == 0) {
                sam_params->mask_threshold = atof(v);
            } else if (strcmp(k, "iou_threshold") == 0) {
                sam_params->iou_threshold = atof(v);
            } else if (strcmp(k, "stability_score_threshold") == 0) {
                sam_params->stability_score_threshold = atof(v);
            } else if (strcmp(k, "stability_score_offset") == 0) {
                sam_params->stability_score_offset = atof(v);
            } else if (strcmp(k, "eps") == 0) {
                sam_params->eps = atof(v);
            } else if (strcmp(k, "eps_decoder_transformer") == 0) {
                sam_params->eps_decoder_transformer = atof(v);
            } else if (strcmp(k, "n_threads") == 0) {
                sam_params->n_threads = atoi(v);
            }
        }
    }

    fclose(f);
    return TRUE;
}

bool get_params_from_config_file(sam_params_t* sam_params) {   
    const char* exe_dir = get_current_path();
    if (exe_dir) {
        char config_path[1024];
        snprintf(config_path, sizeof(config_path),
                 "%s\\%s", exe_dir, CONFIG_FILE_PATH);

        // Try to read config, create default if it doesn't exist
        if (!read_config_file(sam_params)) {
            create_default_config(config_path);
            // Try reading again
            read_config_file(sam_params);
        }
    } else {
        return FALSE;
    }

    return TRUE;
}