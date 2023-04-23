#include "col_solver.cuh"

#include "defines.h"
#include "col_solver_kernel_fun.cuh"

#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/remove.h>

// Singleton requirements
INIT_INSTANCE_STATIC(fen::col_solver);

// Gets raw pointer from device_vector
#define GET_RAW_PTR(v) thrust::raw_pointer_cast((v).data())

// Gpu assert function
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
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
	CUDA_CALL_2(kernel_scale, num_blocks, num_threads)(GET_RAW_PTR(positions), max_pos_x - min_pos_x, min_pos_x, positions.size() / 2); // X dim 
	CUDA_CALL_2(kernel_scale, num_blocks, num_threads)(GET_RAW_PTR(positions) + positions.size() / 2, max_pos_y - min_pos_y, min_pos_y, positions.size() / 2); // Y dim 

	curandGenerateUniform(generator, GET_RAW_PTR(radius), radius.size());
	CUDA_CALL_2(kernel_scale, num_blocks, num_threads)(GET_RAW_PTR(radius), max_radius - min_radius, min_radius, radius.size()); // Radius

	thrust::fill(thrust::device, delta_mov.begin(), delta_mov.end(), 0.0f); // Move delta
}


unsigned int col_solver::solve_cols(unsigned num_blocks, unsigned num_threads, thrust::device_vector<float>& positions, thrust::device_vector<float>& radius, thrust::device_vector<float>& delta_mov, const size_t num_entities)
{
	// Initializes each entity cell which they occupy
	profiler.start_timing<Cells_Init>();
	const unsigned int num_cells = init_cells(num_blocks, num_threads, positions, radius, num_entities);
	cudaDeviceSynchronize();
	profiler.finish_timing<Cells_Init>();

	// Sorting the cells
	profiler.start_timing<Sort>();
	sort_cells();
	cudaDeviceSynchronize();
	profiler.finish_timing<Sort>();

	// Count and process the collisions
	const unsigned int collisions = count_cols(num_blocks, num_threads, positions, radius, delta_mov, num_entities, num_cells);

	// Solve the collision by moving the entities apart
	profiler.start_timing<Move>();
	CUDA_CALL_2(kernel_move_entities, num_blocks, num_threads)(GET_RAW_PTR(positions), GET_RAW_PTR(delta_mov), num_entities, min_pos_x, max_pos_x, min_pos_y, max_pos_y);
	thrust::fill(thrust::device, delta_mov.begin(), delta_mov.end(), 0.0f);
	cudaDeviceSynchronize();
	profiler.finish_timing<Move>();

	return collisions;
}

col_solver::col_solver() : Singleton(), max_rad(), min_pos_x(), min_pos_y(), max_pos_x(), max_pos_y(), width(), height(),
                           cell_size()
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

	// Copy result to host
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

unsigned int col_solver::count_cols(unsigned num_blocks, unsigned num_threads, thrust::device_vector<float>& positions, thrust::device_vector<float>& radius, thrust::device_vector<float>& delta_mov, const size_t num_entities, const unsigned num_cells)
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

	// Create the collision lists
	CUDA_CALL_2(kernel_check_cell_cols, num_blocks, num_threads) (
		GET_RAW_PTR(cells), GET_RAW_PTR(objects),
		num_cells,
		cells_per_thread,
		GET_RAW_PTR(col_cells),
		temp
	);

	gpuErrchk(cudaPeekAtLastError());

	// Copy result to host
	unsigned int error_flag = 0;
	cudaMemcpy(&error_flag, temp, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	if(error_flag)
	{
		printf("Error: Too many entities in a single cell\n");
		return 0;
	}

	// Number of collision lists
	const unsigned int dist = thrust::remove(col_cells.begin(), col_cells.end(), 0u) - col_cells.begin();

	num_blocks = std::min<unsigned int>(num_blocks, (dist / num_threads) + 1u);

	profiler.finish_timing<Cols_Init>();
	profiler.start_timing<Cols_Resolve>();

	// 4 steps to avoid two threads to check the same collision
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

	// Copy result to host
	unsigned int collisions = 0;
	cudaMemcpy(&collisions, temp, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	profiler.finish_timing<Cols_Resolve>();

	return collisions;
}

}
