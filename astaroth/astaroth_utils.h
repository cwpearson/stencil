#pragma once
#include "astaroth.h"

#include <cstdbool>

/** Loads data from the config file */
AcResult acLoadConfig(const char* config_path, AcMeshInfo* config);

