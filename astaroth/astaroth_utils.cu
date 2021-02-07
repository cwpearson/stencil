#include "astaroth_utils.h"

#include <cstdint> // uint8_t, uint32_t
#include <cstring>

#include "errchk.h"

/**
 \brief Find the index of the keyword in names
 \return Index in range 0...n if the keyword is in names. -1 if the keyword was
 not found.
 */
static int
find_str(const char keyword[], const char* names[], const int& n)
{
    for (int i = 0; i < n; ++i)
        if (!strcmp(keyword, names[i]))
            return i;

    return -1;
}

void parse_config(const char* path, AcMeshInfo* config)
{
    FILE* fp;
    fp = fopen(path, "r");
    // For knowing which .conf file will be used
    fprintf(stderr,"Config file path: %s\n", path);
    ERRCHK_ALWAYS(fp != NULL);

    const size_t BUF_SIZE = 128;
    char keyword[BUF_SIZE];
    char value[BUF_SIZE];
    int items_matched;
    while ((items_matched = fscanf(fp, "%s = %s", keyword, value)) != EOF) {

        if (items_matched < 2)
            continue;

        int idx = -1;
        if ((idx = find_str(keyword, intparam_names, NUM_INT_PARAMS)) >= 0)
            config->int_params[idx] = atoi(value);
        else if ((idx = find_str(keyword, realparam_names, NUM_REAL_PARAMS)) >= 0)
            config->real_params[idx] = (AcReal)(atof(value));
    }

    fclose(fp);
}

AcResult
acHostUpdateBuiltinParams(AcMeshInfo* config)
{
    config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER;
    ///////////// PAD TEST
    // config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER + PAD_SIZE;
    ///////////// PAD TEST
    config->int_params[AC_my] = config->int_params[AC_ny] + STENCIL_ORDER;
    config->int_params[AC_mz] = config->int_params[AC_nz] + STENCIL_ORDER;

    // Bounds for the computational domain, i.e. nx_min <= i < nx_max
    config->int_params[AC_nx_min] = STENCIL_ORDER / 2;
    config->int_params[AC_nx_max] = config->int_params[AC_nx_min] + config->int_params[AC_nx];
    config->int_params[AC_ny_min] = STENCIL_ORDER / 2;
    config->int_params[AC_ny_max] = config->int_params[AC_ny] + STENCIL_ORDER / 2;
    config->int_params[AC_nz_min] = STENCIL_ORDER / 2;
    config->int_params[AC_nz_max] = config->int_params[AC_nz] + STENCIL_ORDER / 2;

// These do not have to be defined by empty projects any more.
// These should be set only if stdderiv.h is included
#ifdef AC_dsx
    config->real_params[AC_inv_dsx] = (AcReal)(1.) / config->real_params[AC_dsx];
#endif
#ifdef AC_dsy
    config->real_params[AC_inv_dsy] = (AcReal)(1.) / config->real_params[AC_dsy];
#endif
#ifdef AC_dsz
    config->real_params[AC_inv_dsz] = (AcReal)(1.) / config->real_params[AC_dsz];
#endif

    /* Additional helper params */
    // Int helpers
    config->int_params[AC_mxy]  = config->int_params[AC_mx] * config->int_params[AC_my];
    config->int_params[AC_nxy]  = config->int_params[AC_nx] * config->int_params[AC_ny];
    config->int_params[AC_nxyz] = config->int_params[AC_nxy] * config->int_params[AC_nz];

    return AC_SUCCESS;
}

/**
\brief Loads data from astaroth.conf into a config struct.
\return AC_SUCCESS on success, AC_FAILURE if there are potentially uninitialized values.
*/
AcResult
acLoadConfig(const char* config_path, AcMeshInfo* config)
{
    AcResult retval = AC_SUCCESS;
    ERRCHK_ALWAYS(config_path);

    // memset reads the second parameter as a byte even though it says int in
    // the function declaration
    memset(config, (uint8_t)0xFF, sizeof(*config));

    parse_config(config_path, config);
    acHostUpdateBuiltinParams(config);
#if AC_VERBOSE
    printf("###############################################################\n");
    printf("Config dimensions loaded:\n");
    acPrintMeshInfo(*config);
    printf("###############################################################\n");
#endif

    // sizeof(config) must be a multiple of 4 bytes for this to work
    ERRCHK_ALWAYS(sizeof(*config) % sizeof(uint32_t) == 0);
    for (size_t i = 0; i < sizeof(*config) / sizeof(uint32_t); ++i) {
        if (((uint32_t*)config)[i] == (uint32_t)0xFFFFFFFF) {
#if AC_VERBOSE
            fprintf(stderr, "Some config values may be uninitialized. "
                            "See that all are defined in astaroth.conf\n");
#endif
            retval = AC_FAILURE;
        }
    }
    return retval;
}