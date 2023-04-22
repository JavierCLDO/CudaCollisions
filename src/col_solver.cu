#include "col_solver.cuh"
#include "defines.h"
#include "on_collision.cuh"

#include <device_launch_parameters.h>
#include <curand.h>

#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>

#include <thrust/sort.h>
#include <thrust/remove.h>

INIT_INSTANCE_STATIC(fen::col_solver);

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

		const int h = i * DIM_2;

		float dist;

		float x = positions[i] - min_pos_x;
		float y = positions[i + n] - min_pos_y;

		const uint32_t cell_pos_x = (uint32_t)(x / cell_dim);
		const uint32_t cell_pos_y = (uint32_t)(y / cell_dim);

		float rad = radius[i] * 1.41421356f; // sqrtf(2.0f)

		hash = cell_pos_x << BITS;
		hash = hash | cell_pos_y;

		const uint8_t home_cell_t = ((cell_pos_y & 0b1) << 1) | (cell_pos_x & 0b1);

		cells[h] = hash << 1 | 0b0;

		unsigned int home_cells_t_sides = 0b1 << home_cell_t;

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

__global__ void kernel_count_cols(uint32_t* cells, uint32_t* objects, float* positions, float* radius, unsigned int n, unsigned int m, unsigned int cells_per_thread, unsigned int* collision_count)
{
	extern __shared__ unsigned int t[];

	unsigned int thread_start = ((blockDim.x * blockIdx.x) + threadIdx.x) * cells_per_thread;

	if (thread_start >= m)
	{
		t[threadIdx.x] = 0;
		return;
	}

	unsigned int thread_end = thread_start + cells_per_thread;
	unsigned int i = thread_start;
	unsigned int cell;
	unsigned int collisions = 0;
	unsigned int h;
	unsigned int p;
	unsigned int start;
	unsigned int num_col_list;

	float d_c1, d_c2;
	uint32_t _c1, _c2;
	uint32_t t_c1, t_c2;
	uint32_t ts_c1, ts_c2;
	float dist, dx;

	if (thread_end > m)
	{
		thread_end = m;
	}

	// The first thread does not skip the first occurrence
	if (blockIdx.x == 0 && threadIdx.x == 0 || cells[thread_start - 1] >> 1 != cells[thread_start] >> 1)
		cell = UINT32_MAX;
	else
		cell = cells[thread_start] >> 1;

	while (true)
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

		if (h > 0 && num_col_list > 1) {

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

					d_c2 = radius[_c2];

					dist = 0;

					if (t_c2 < t_c1 && (0b1 << t_c2 & ts_c1) && (0b1 << t_c1 & ts_c2))
						continue;

					for (int l = 0; l < DIM; ++l)
					{
						dx = positions[_c2 + l * n] - positions[_c1 + l * n];
						dist += dx * dx;
					}

					if (dist < ((d_c1 + d_c2) * (d_c2 + d_c1)))
					{
						collisions++;
					}
				}
			}
		}
	}

	t[threadIdx.x] = collisions;

	__syncthreads();

	if(!threadIdx.x)
	{
		atomicAdd(collision_count, thrust::reduce(thrust::device, t, t + blockDim.x));
	}
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

		// A collision cell
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

		if (positions[i] < min_pos_x) positions[i] = min_pos_x;
		else if (positions[i] > max_pos_x) positions[i] = max_pos_x;

		if (positions[i + n] < min_pos_y) positions[i + n] = min_pos_y;
		else if (positions[i + n] > max_pos_y) positions[i + n] = max_pos_y;
	}
}


#define GET_RAW_PTR(v) thrust::raw_pointer_cast(v.data())

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

namespace fen
{

void init_objects(unsigned long long seed, unsigned num_blocks, unsigned num_threads, thrust::device_vector<float>& positions, thrust::device_vector<float>& radius, thrust::device_vector<float>& delta_mov, float min_pos_x, float max_pos_x, float min_pos_y, float max_pos_y, float min_radius, float max_radius)
{
	auto& generator = col_solver::Instance()->get_generator();
	curandSetPseudoRandomGeneratorSeed(generator, seed);
	
	curandGenerateUniform(generator, GET_RAW_PTR(positions), positions.size());
	CUDA_CALL_2(kernel_scale, num_blocks, num_threads)(GET_RAW_PTR(positions), max_pos_x - min_pos_x, min_pos_x, positions.size() / 2);
	CUDA_CALL_2(kernel_scale, num_blocks, num_threads)(GET_RAW_PTR(positions) + positions.size() / 2, max_pos_y - min_pos_y, min_pos_y, positions.size() / 2);

	curandGenerateUniform(generator, GET_RAW_PTR(radius), radius.size());
	CUDA_CALL_2(kernel_scale, num_blocks, num_threads)(GET_RAW_PTR(radius), max_radius - min_radius, min_radius, radius.size());

	thrust::fill(thrust::device, delta_mov.begin(), delta_mov.end(), 0.0f);
}


unsigned int col_solver::solve_cols_1(unsigned num_blocks, unsigned num_threads, thrust::device_vector<float>& positions, thrust::device_vector<float>& radius, thrust::device_vector<float>& delta_mov, const size_t num_entities)
{

	profiler.start_timing<Cells_Init>();
	unsigned int num_cells = init_cells(num_blocks, num_threads, positions, radius, num_entities);
	cudaDeviceSynchronize();
	profiler.finish_timing<Cells_Init>();

	profiler.start_timing<Sort>();
	sort_cells();
	cudaDeviceSynchronize();
	profiler.finish_timing<Sort>();

	unsigned int collisions = count_cols_1(num_blocks, num_threads, positions, radius, delta_mov, num_entities, num_cells);

	profiler.start_timing<Move>();
	CUDA_CALL_2(kernel_move_entities, num_blocks, num_threads)(GET_RAW_PTR(positions), GET_RAW_PTR(delta_mov), num_entities, min_pos_x, max_pos_x, min_pos_y, max_pos_y);
	thrust::fill(thrust::device, delta_mov.begin(), delta_mov.end(), 0.0f);
	cudaDeviceSynchronize();
	profiler.finish_timing<Move>();

	return collisions;
}

col_solver::col_solver() : Singleton()
{
	cudaMalloc((void**)&temp, sizeof(unsigned int));

	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
}

col_solver::~col_solver()
{
	cudaFree(temp);
}

void col_solver::init_solver(const size_t num_entities, const float max_rad_, const float min_pos_x_, const float min_pos_y_, const float max_pos_x_, const float max_pos_y_)
{
	//printf("Init solver\n");
	cells = thrust::device_vector<uint32_t>(num_entities * DIM_2);
	objects = thrust::device_vector<uint32_t>(num_entities * DIM_2);
	col_cells = thrust::device_vector<uint64_t>();

	max_rad = max_rad_;

	cell_size = max_rad * 4.0f;

	min_pos_x = min_pos_x_;
	min_pos_y = min_pos_y_;
	max_pos_x = max_pos_x_;
	max_pos_y = max_pos_y_;

	width = max_pos_x_ - min_pos_x_;
	height = max_pos_y_ - min_pos_y_;

	profiler.reset();
}

unsigned int col_solver::init_cells(unsigned num_blocks, unsigned num_threads, thrust::device_vector<float>& positions, thrust::device_vector<float>& radius, const size_t num_entities)
{
	//printf("Init ");

	// reset
	cudaMemset(GET_RAW_PTR(cells), 0xff, num_entities * DIM_2 * sizeof(decltype(cells)::value_type));
	cudaMemset(temp, 0, sizeof(unsigned int));

	// If max rad isn't specified, choose from largest radius
	if (max_rad < 0)
		cell_size = *thrust::max_element(thrust::device, radius.cbegin(), radius.cend());

	CUDA_CALL_3(kernel_init_cells, num_blocks, num_threads, num_threads * sizeof(unsigned int))(GET_RAW_PTR(cells), GET_RAW_PTR(objects), GET_RAW_PTR(positions), GET_RAW_PTR(radius),
		cell_size, min_pos_x, (uint32_t)(width / cell_size), min_pos_y, (uint32_t)(height / cell_size), num_entities, temp);

	gpuErrchk(cudaPeekAtLastError());

	unsigned int num_cells = 0;
	cudaMemcpy(&num_cells, temp, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	//printf(" n_cells:%u ", num_cells);
	return num_cells;
}

void col_solver::sort_cells()
{
	//printf("Sort ");

	thrust::stable_sort_by_key(thrust::device, cells.begin(), cells.end(), objects.begin(), thrust::less<uint32_t>());
}

unsigned int col_solver::count_cols_1(unsigned num_blocks, unsigned num_threads, thrust::device_vector<float>& positions, thrust::device_vector<float>& radius, thrust::device_vector<float>& delta_mov, const size_t num_entities, const unsigned num_cells)
{
	//printf("Count\n");

	profiler.start_timing<Cols_Init>();

	unsigned int cells_per_thread = ((num_cells - 1) / num_blocks) /
		num_threads +
		1;

	col_cells.resize(num_cells);

	// Reset to 0
	cudaMemset(GET_RAW_PTR(col_cells), 0x00, col_cells.size() * sizeof(decltype(col_cells)::value_type));
	cudaMemset(temp, 0, sizeof(unsigned int));

	CUDA_CALL_2(kernel_check_cell_cols, num_blocks, num_threads) (
		GET_RAW_PTR(cells), GET_RAW_PTR(objects),
		num_cells,
		cells_per_thread,
		GET_RAW_PTR(col_cells),
		temp
	);

	gpuErrchk(cudaPeekAtLastError());

	unsigned int error_flag = 0;
	cudaMemcpy(&error_flag, temp, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	if(error_flag)
	{
		printf("Error: Too many entities in a single cell\n");
		return 0;
	}

	const unsigned int dist = thrust::remove(col_cells.begin(), col_cells.end(), 0u) - col_cells.begin();

	num_blocks = std::min<unsigned int>(num_blocks, (dist / num_threads) + 1u);


	profiler.finish_timing<Cols_Init>();

	profiler.start_timing<Cols_Resolve>();

	for (int i = 0; i < 4; ++i)
	{
		CUDA_CALL_3(kernel_solve_cols, num_blocks, num_threads, num_threads * sizeof(unsigned int)) (
			GET_RAW_PTR(col_cells), GET_RAW_PTR(objects),
			GET_RAW_PTR(positions), GET_RAW_PTR(radius), GET_RAW_PTR(delta_mov),
			num_entities,
			dist,
			i,
			temp
		);

		gpuErrchk(cudaPeekAtLastError());
	}

	cudaDeviceSynchronize();
	gpuErrchk(cudaPeekAtLastError());

	unsigned int collisions = 0;
	cudaMemcpy(&collisions, temp, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	profiler.finish_timing<Cols_Resolve>();

	return collisions;
}

}
