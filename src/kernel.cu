#include "col_solver.cuh"
#include "defines.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <chrono>
#include <future>
#include <thread>

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

		printf("\t%.5f\n", radius[i]);
	}

	printf("\n");
}

float rand_float(const float& low, const float& high)
{
	return low + static_cast<float>(rand()) * (high - low) / RAND_MAX;
}

void test_col(float pos1_x, float pos1_y, float pos2_x, float pos2_y, float radius1, float radius2, float& delta_mov1_x, float& delta_mov1_y, float& delta_mov2_x, float& delta_mov2_y)
{
	delta_mov1_x = delta_mov1_y = delta_mov2_x = delta_mov2_y = 0.0f;

	const float col_axis_x = pos2_x - pos1_x;
	const float col_axis_y = pos2_y - pos1_y;

	const float dist_sq = (col_axis_x * col_axis_x) + (col_axis_y * col_axis_y);

	const float dist = sqrtf(dist_sq);

	const float min_dist = radius1 + radius2;

	if (dist >= min_dist)
		return;

	const float mass_ratio_1 = radius1 / min_dist;
	const float mass_ratio_2 = radius2 / min_dist;

	const float delta = (dist - min_dist);

	const float n_x = col_axis_x / dist;
	const float n_y = col_axis_y / dist;

	printf("c_dist: %.3f; ", dist);

	delta_mov1_x += n_x * mass_ratio_1 * delta;
	delta_mov1_y += n_y * mass_ratio_1 * delta;

	delta_mov2_x -= n_x * mass_ratio_2 * delta;
	delta_mov2_y -= n_y * mass_ratio_2 * delta;



}

int main()
{
	printf("\n CPU: \n");
	float pos1_x = 10.0f, pos1_y = 10.0f, pos2_x = 11.0f, pos2_y = 11.0f;
	float delta_mov1_x = 0.0f, delta_mov1_y = 0.0f, delta_mov2_x = 0.0f, delta_mov2_y = 0.0f;

	test_col(pos1_x, pos1_y, pos2_x, pos2_y, 
		0.9f, 0.9f, delta_mov1_x, delta_mov1_y, delta_mov2_x, delta_mov2_y);

	test_col(pos1_x + delta_mov1_x, pos1_y + delta_mov1_y, pos2_x + delta_mov2_x, pos2_y + delta_mov2_x, 
		0.9f, 0.9f, delta_mov1_x, delta_mov1_y, delta_mov2_x, delta_mov2_y);

	printf("\n GPU: \n");
	constexpr size_t NUM_OBJECTS = 1 << 18;

	constexpr unsigned int NUM_BLOCKS = 100;
	constexpr unsigned int NUM_THREADS = 512;

	constexpr float   = 0.5f;
	constexpr float MAX_RAD = 1.0f;

	constexpr float MIN_X = 4.f;
	constexpr float MIN_Y = 4.f;

	fen::col_solver::Create();

	auto seed = time(nullptr);
	srand(seed);

	printf("Seed: %llu\n", seed);

	float MAX_X = 1024.0f;
	float MAX_Y = 1024.0f;

	printf("MAX_X: %.2f; MAX_Y: %.2f\t\n", MAX_X, MAX_Y);

	constexpr bool CALCULATE_RAD = false;

	auto positions = thrust::device_vector<float>(NUM_OBJECTS * DIM);
	auto delta_mov = thrust::device_vector<float>(NUM_OBJECTS * DIM);
	auto radius = thrust::device_vector<float>(NUM_OBJECTS);

	//fen::hack_init(NUM_BLOCKS, 1024, positions, radius, MIN_RAD, MAX_RAD);
	fen::col_solver::Instance()->init_objects(NUM_BLOCKS, NUM_THREADS, positions, radius, delta_mov, MIN_X, MAX_X - MIN_X, MIN_Y, MAX_Y - MIN_Y, MIN_RAD, MAX_RAD);

	thrust::host_vector<float> h_positions = positions;
	thrust::host_vector<float> h_radius = radius;

	//print_entities(h_positions, h_radius, NUM_OBJECTS, 16);

	fen::col_solver::Instance()->reset(NUM_OBJECTS, CALCULATE_RAD ? -1.0f : MAX_RAD, MIN_X, MIN_Y, MAX_X, MAX_Y);

	for (int i = 0; i < 60; ++i) {
		const auto start = std::chrono::high_resolution_clock::now();

		unsigned int collisions_1 = fen::col_solver::Instance()->solve_cols_1(NUM_BLOCKS, NUM_THREADS, positions, radius, delta_mov, NUM_OBJECTS);
		for (int s = 0; s < 8 - 1; ++s)
		{
			fen::col_solver::Instance()->solve_cols_1(NUM_BLOCKS, NUM_THREADS, positions, radius, delta_mov, NUM_OBJECTS);
		}

		printf("col:%u", collisions_1);

		const auto end = std::chrono::high_resolution_clock::now();
		const std::chrono::duration<double, std::milli> ms_double = end - start;
		printf("\tElapsed time: %.3f ms\n", ms_double);

	}

    return 0;
}
