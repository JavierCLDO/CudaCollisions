#include "col_solver.cuh"
#include "defines.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <chrono>
#include <future>

#include "simple_profiler.h"

unsigned int test_cpu_cols(thrust::host_vector<float>& positions, thrust::host_vector<float>& radius, size_t num_objects)
{
	unsigned int collisions = 0;
	float dh;
	float dp;
	float dx;
	float d;

	for (unsigned int j = 0; j < num_objects; j++) {
		dh = radius[j];

		for (unsigned int k = j + 1; k < num_objects; k++) {

			// assume dims are radii of balls
			dp = radius[k];

			d = 0;

			for (unsigned int l = 0; l < DIM; l++) {
				dx = positions[j + l * num_objects] - positions[k + l * num_objects];
				d += dx * dx;
			}

			auto d_s = (dh + dp) * (dp + dh);

			// if collision
			if (d < d_s) {
				collisions++;
			}
		}
	}

	return collisions;
}

void print_entities(thrust::host_vector<float>& positions, thrust::host_vector<float>& radius, size_t num_entities, unsigned int n)
{
	for (int i = 1020; i < 1020 + n; i++) {

		printf("\n%.4i: ", i);

		for (int j = 0; j < DIM; j++) {
			printf(" %10.5f ", positions[i + j * num_entities]);
		}

		printf("\t%.5f", radius[i]);
	}

	printf("\n");
}

float rand_float(const float& low, const float& high)
{
	return low + static_cast<float>(rand()) * (high - low) / RAND_MAX;
}

int main()
{
	fen::col_solver::CreateInstance();

	constexpr size_t NUM_OBJECTS = 1 << 18;

	constexpr unsigned int NUM_BLOCKS = 100;
	constexpr unsigned int NUM_THREADS = 512;

	constexpr float MIN_RAD = 0.5f;
	constexpr float MAX_RAD = 1.0f;

	constexpr float MIN_X = -512.0f;
	constexpr float MIN_Y = -512.0f;
	constexpr float MAX_X = +512.0f;
	constexpr float MAX_Y = +512.0f;

	auto seed = time(nullptr);
	srand(seed);

	printf("Seed: %llu\n", seed);


	printf("MAX_X: %.2f; MAX_Y: %.2f\t\n", MAX_X, MAX_Y);

	constexpr bool CALCULATE_RAD = false;

	auto positions = thrust::device_vector<float>(NUM_OBJECTS * DIM);
	auto delta_mov = thrust::device_vector<float>(NUM_OBJECTS * DIM);
	auto radius = thrust::device_vector<float>(NUM_OBJECTS);

	auto col_solver = fen::col_solver::Instance();

	col_solver->init_objects(NUM_BLOCKS, NUM_THREADS, positions, radius, delta_mov, MIN_X, MAX_X, MIN_Y, MAX_Y, MIN_RAD, MAX_RAD);

	thrust::host_vector<float> h_positions = positions;
	thrust::host_vector<float> h_radius = radius;

	//print_entities(h_positions, h_radius, NUM_OBJECTS, 16);

	col_solver->reset(NUM_OBJECTS, CALCULATE_RAD ? -1.0f : MAX_RAD, MIN_X, MIN_Y, MAX_X, MAX_Y);

	for (int i = 0; i < 30; ++i) {

		const unsigned int collisions_1 = col_solver->solve_cols_1(NUM_BLOCKS, NUM_THREADS, positions, radius, delta_mov, NUM_OBJECTS);
		for (int s = 0; s < 8 - 1; ++s)
		{
			col_solver->solve_cols_1(NUM_BLOCKS, NUM_THREADS, positions, radius, delta_mov, NUM_OBJECTS);
		}

		col_solver->get_profiler().next_step();

		printf("col:%u\n", collisions_1);
	}

	col_solver->get_profiler().print_avg_times(std::cout);
	col_solver->get_profiler().print_times(std::cout);

    return 0;
}
