#pragma once

#include <cstdint>
#include <crt/host_defines.h>

namespace fen
{
	// Helper functions
	__global__ void kernel_scale(float* arr, const float scale, const float offset, const unsigned int n);
	__device__ void kernel_sum_reduce(unsigned int* values, unsigned int* out);

	// Init phase
	__global__ void kernel_init_cells(uint32_t* cells, uint32_t* objects, const float* positions, const float* radius, const float cell_dim,
	                                  const float min_pos_x, const uint32_t max_cell_pos_x, const float min_pos_y, const uint32_t max_cell_pos_y, const size_t n, unsigned int* cell_count);

	// Solving phase
	__global__ void kernel_check_cell_cols(uint32_t* cells, uint32_t* objects, unsigned int m, unsigned int cells_per_thread, uint64_t* col_cells, unsigned int* error_flag);
	__global__ void kernel_solve_cols(const uint64_t* col_cells, const uint32_t* objects, const float* positions, const float* radius, float* delta_mov, const uint32_t n, const unsigned int m, const unsigned char cell_type, unsigned int* collision_count);

	// Applying results phase
	__global__ void kernel_move_entities(float* positions, float* delta_mov, const uint32_t n, const float min_pos_x, const float max_pos_x, const float min_pos_y, const float max_pos_y);
};