#include "col_solver_kernel_fun.cuh"

#include "defines.h"
#include "on_collision.cuh"

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <thrust/extrema.h>


namespace fen
{

__global__ void kernel_scale(float* arr, const float scale, const float offset, const unsigned int n)
{
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
	{
		arr[i] = arr[i] * scale + offset;
	}
}

__device__ void kernel_sum_reduce(unsigned int* values, unsigned int* out)
{
	// wait for the whole array to be populated
	__syncthreads();

	// sum by reduction, using half the threads in each subsequent iteration
	unsigned int threads = blockDim.x;
	unsigned int half = threads / 2;

	while (half)
	{
		if (threadIdx.x < half)
		{
			// only keep going if the thread is in the first half threads
			for (int k = threadIdx.x + half; k < threads; k += half)
				values[threadIdx.x] += values[k];

			threads = half;
		}

		half /= 2;

		// make sure all the threads are on the same iteration
		__syncthreads();
	}

	// only let one thread update the current sum
	if (!threadIdx.x)
		atomicAdd(out, values[0]);
}

__global__ void kernel_init_cells(uint32_t* cells, uint32_t* objects, const float* positions, const float* radius, const float cell_dim,
	const float min_pos_x, const uint32_t max_cell_pos_x, const float min_pos_y, const uint32_t max_cell_pos_y, const size_t n, unsigned int* cell_count)
{
	const unsigned BITS = 16; // pos_x is allocated 15 bits because we need space for the home/phantom cell flag

	extern __shared__ unsigned int t[];
	unsigned int count = 0;

	for (unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
	{
		uint32_t hash = 0;
		unsigned int sides = 0;

		const unsigned int h = i * DIM_2;

		float dist;

		float x = positions[i] - min_pos_x;
		float y = positions[i + n] - min_pos_y;

		// Home cell the entity occupies
		const uint32_t cell_pos_x = (uint32_t)(x / cell_dim);
		const uint32_t cell_pos_y = (uint32_t)(y / cell_dim);

		float rad = radius[i] * 1.41421356f; // sqrtf(2.0f)

		hash = cell_pos_x << BITS;
		hash = hash | cell_pos_y;

		// Cell ID [0 - 4 in 2D) the entity occupies
		const uint8_t home_cell_t = ((cell_pos_y & 0b1) << 1) | (cell_pos_x & 0b1);

		cells[h] = hash << 1 | 0b0;

		// Leave one bit space
		unsigned int home_cells_t_sides = 0b1 << home_cell_t;

		// Determine which side the entity overlaps
		dist = y - floor(y / cell_dim) * cell_dim;
		if (dist < rad) // overlap with top cell
		{
			if (cell_pos_y > 0) // Not already in the top position
				sides |= 0x1;
		}
		else if (cell_dim - dist < rad)
		{
			if(cell_pos_y <= max_cell_pos_y) // overlap with bottom cell and not in the bottom position
				sides |= 0x2;
		}
		dist = x - floor(x / cell_dim) * cell_dim;
		sides <<= 2;
		if (dist < rad)
		{
			if (cell_pos_x > 0) // overlap with left cell
				sides |= 0x1;
		}
		else if (cell_dim - dist < rad) {
			if (cell_pos_x <= max_cell_pos_x)// overlap with right cell
				sides |= 0x2;
		}

		// Which specific cell the entity overlaps
		if (((sides >> 2) & 0x1) == 0x1) // check top
		{
			if ((sides & 0x1) == 0x1) // check left
			{
				// overlaps cells: top, top left, left
				cells[h + 1] = ((cell_pos_x << BITS) | (cell_pos_y - 1)) << 1 | 0b1;
				cells[h + 2] = (((cell_pos_x - 1) << BITS) | (cell_pos_y - 1)) << 1 | 0b1;
				cells[h + 3] = (((cell_pos_x - 1) << BITS) | cell_pos_y) << 1 | 0b1;

				home_cells_t_sides = 0b1111;

				count += 4;
			}
			else if ((sides & 0x2) == 0x2) // check right
			{
				// overlaps cells: top, top right, right
				cells[h + 1] = ((cell_pos_x << BITS) | (cell_pos_y - 1)) << 1 | 0b1;
				cells[h + 2] = (((cell_pos_x + 1) << BITS) | (cell_pos_y - 1)) << 1 | 0b1;
				cells[h + 3] = (((cell_pos_x + 1) << BITS) | cell_pos_y) << 1 | 0b1;

				home_cells_t_sides = 0b1111;

				count += 4;
			}
			else
			{
				// overlaps cells: top
				cells[h + 1] = ((cell_pos_x << BITS) | (cell_pos_y - 1)) << 1 | 0b1;

				home_cells_t_sides |= 0b1 << (home_cell_t + 2) % 4;

				count += 2;
			}
		}
		else if (((sides >> 2) & 0x2) == 0x2) // check bottom
		{
			if ((sides & 0x1) == 0x1) // check left
			{
				// overlaps cells: bottom, bottom left, left
				cells[h + 1] = ((cell_pos_x << BITS) | (cell_pos_y + 1)) << 1 | 0b1;
				cells[h + 2] = (((cell_pos_x - 1) << BITS) | (cell_pos_y + 1)) << 1 | 0b1;
				cells[h + 3] = (((cell_pos_x - 1) << BITS) | cell_pos_y) << 1 | 0b1;

				home_cells_t_sides = 0b1111;
				count += 4;
			}
			else if ((sides & 0x2) == 0x2) // check right
			{
				// overlaps cells: bottom, bottom right, right
				cells[h + 1] = ((cell_pos_x << BITS) | (cell_pos_y + 1)) << 1 | 0b1;
				cells[h + 2] = (((cell_pos_x + 1) << BITS) | (cell_pos_y + 1)) << 1 | 0b1;
				cells[h + 3] = (((cell_pos_x + 1) << BITS) | cell_pos_y) << 1 | 0b1;

				home_cells_t_sides = 0b1111;
				count += 4;
			}
			else
			{
				// overlaps cells: bottom
				cells[h + 1] = ((cell_pos_x << BITS) | (cell_pos_y + 1)) << 1 | 0b1;

				home_cells_t_sides |= 0b1 << (home_cell_t + 2) % 4;

				count += 2;
			}
		}
		else // check left and right
		{
			if ((sides & 0x1) == 0x1) // check left
			{
				// overlaps cells: left
				cells[h + 1] = (((cell_pos_x - 1) << BITS) | cell_pos_y) << 1 | 0b1;

				if (home_cell_t & 0b1)
					home_cells_t_sides |= 0b1 << (home_cell_t - 1);
				else
					home_cells_t_sides |= 0b1 << (home_cell_t + 1);

				count += 2;
			}
			else if ((sides & 0x2) == 0x2) // check right
			{
				// overlaps cells: right
				cells[h + 1] = (((cell_pos_x + 1) << BITS) | cell_pos_y) << 1 | 0b1;

				if (home_cell_t & 0b1)
					home_cells_t_sides |= 0b1 << (home_cell_t - 1);
				else
					home_cells_t_sides |= 0b1 << (home_cell_t + 1);

				count += 2;
			}
			else
			{
				// does not overlap with any other cell
				count++;
			}
		}

		objects[h] = (i << 7) | (home_cells_t_sides << 3) | (home_cell_t << 1) | 0b1;

		// Phantom cells
		objects[h + 1] = (i << 7) | (home_cells_t_sides << 3) | (home_cell_t << 1) | 0b0;
		objects[h + 2] = (i << 7) | (home_cells_t_sides << 3) | (home_cell_t << 1) | 0b0;
		objects[h + 3] = (i << 7) | (home_cells_t_sides << 3) | (home_cell_t << 1) | 0b0;
	}

	// perform reduction to count number of cells occupied
	t[threadIdx.x] = count;
	kernel_sum_reduce(t, cell_count);
}

__global__ void kernel_check_cell_cols(uint32_t* cells, uint32_t* objects, unsigned int m, unsigned int cells_per_thread, uint64_t* col_cells, unsigned int* error_flag)
{
	unsigned int thread_start = ((blockDim.x * blockIdx.x) + threadIdx.x) * cells_per_thread;

	if (thread_start >= m)
	{
		return;
	}

	unsigned int thread_end = thread_start + cells_per_thread;
	unsigned int i = thread_start;
	unsigned int cell;
	uint64_t h;
	uint64_t p;
	uint64_t start = thread_start;
	unsigned int num_col_list;

	if (thread_end > m)
	{
		thread_end = m;
	}

	// The first thread does not skip the first occurrence
	if (blockIdx.x == 0 && threadIdx.x == 0 || cells[thread_start - 1] >> 1 != cells[thread_start] >> 1)
		cell = UINT32_MAX;
	else
		cell = cells[thread_start] >> 1;

	// Each thread look [thread_start, thread_end) intervals to check if a new collision cell list starts 
	while (i < m)
	{
		h = 0;
		p = 0;

		while (i < thread_end)
		{
			// Searches until it finds a valid home cell to start with 
			if ((cells[i] >> 1) == cell) //same as before or if it is a phantom cell
			{
				++i;
				continue;
			}

			// Found the first home cell
			cell = cells[i] >> 1;
			start = i;
			break;
		}

		// If i reached the end AND the end is not the start of a new collision list
		if (i >= thread_end)
			break;

		while ((cells[i] >> 1) == cell)
		{
			if (objects[i] & 0x01)
				++h;
			else
				++p;
			++i;
		}

		num_col_list = h + p;

		// A collision cell list
		if (h > 0 && num_col_list > 1) {
			if (start < START_LIMIT && h < HOME_LIMIT && p < PHANTOM_LIMIT)
			{
				col_cells[start] = (start << BITS_OFFSET_START) | (h << BITS_OFFSET_HOME) | p;
			}
			else
				*error_flag = true;
		}
	}
}

__global__ void kernel_solve_cols(const uint64_t* col_cells, const uint32_t* objects, const float* positions, const float* radius, float* delta_mov, const uint32_t n, const unsigned int m, const unsigned char cell_type, unsigned int* collision_count)
{
	extern __shared__ unsigned int t[];

	unsigned int collisions = 0;

	for (unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < m; i += (gridDim.x * blockDim.x))
	{
		const uint64_t& col_cell_data = col_cells[i];

		const unsigned int p = col_cell_data & (PHANTOM_LIMIT - 1u); 
		const unsigned int h = (col_cell_data >> BITS_OFFSET_HOME) & (HOME_LIMIT - 1u); 
		const unsigned int start = (col_cell_data >> BITS_OFFSET_START) & (START_LIMIT - 1u); 

		// If its my turn to solve collisions. Done to prevent two threads to change the same entity
		if(cell_type == ((objects[start] >> 1) & 0b11))
		{
			float d_c1, d_c2;
			uint32_t _c1, _c2;
			uint32_t t_c1, t_c2;
			uint32_t ts_c1, ts_c2;
			float dist, dx;

			const unsigned int num_col_list = h + p;

			for (unsigned int c1 = 0; c1 < h; ++c1)
			{
				unsigned int offset = start + c1;
				_c1 = objects[offset] >> 7;
				t_c1 = objects[offset] >> 1 & 0b11;
				ts_c1 = objects[offset] >> 3 & 0b1111;

				d_c1 = radius[_c1];

				for (unsigned int c2 = c1 + 1; c2 < num_col_list; ++c2)
				{
					offset = start + c2;
					_c2 = objects[offset] >> 7;
					t_c2 = objects[offset] >> 1 & 0b11;
					ts_c2 = objects[offset] >> 3 & 0b1111;

					d_c2 = radius[_c2] + d_c1;

					dist = 0;

					// Dont check the same collision more than once
					if (t_c2 < t_c1 && (0b1 << t_c2 & ts_c1) && (0b1 << t_c1 & ts_c2))
						continue;

					for (int l = 0; l < DIM; ++l)
					{
						dx = positions[_c2 + l * n] - positions[_c1 + l * n];
						dist += dx * dx;
					}

					if (dist < (d_c2 * d_c2))
					{
						collisions++;
						kernel_on_collision(_c1, _c2, n, positions, radius, delta_mov, dist);
					}
				}
			}
		}
	}

	t[threadIdx.x] = collisions;

	__syncthreads();

	if (!threadIdx.x)
	{
		atomicAdd(collision_count, thrust::reduce(thrust::device, t, t + blockDim.x));
	}
}

__global__ void kernel_move_entities(float* positions, float* delta_mov, const uint32_t n, const float min_pos_x, const float max_pos_x, const float min_pos_y, const float max_pos_y)
{
	for (unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += (gridDim.x * blockDim.x))
	{
		positions[i] += delta_mov[i];
		positions[i + n] += delta_mov[i + n];

		// Constrain
		if (positions[i] < min_pos_x) positions[i] = min_pos_x;
		else if (positions[i] > max_pos_x) positions[i] = max_pos_x;

		if (positions[i + n] < min_pos_y) positions[i + n] = min_pos_y;
		else if (positions[i + n] > max_pos_y) positions[i + n] = max_pos_y;
	}
}

} // namespace fen
