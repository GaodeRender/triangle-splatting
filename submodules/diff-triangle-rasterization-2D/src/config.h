#pragma once

#define MAX_CHANNELS 3
#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE / 32)
#define ALIGNMENT 128

// #define DEBUG
