#ifndef SAM_CONFIG_H
#define SAM_CONFIG_H

#include "sam-c.h"

#ifdef __cplusplus
extern "C" {
#endif

bool get_params_from_config_file(sam_params_t* sam_params);

#ifdef __cplusplus
}
#endif

#endif // SAM_CONFIG_H