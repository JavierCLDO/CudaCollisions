#include "col_solver.cuh"
#include "defines.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <future>
#include <cstdlib>
#include <sstream>

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
	for (unsigned i = 1020u; i < 1020u + n; i++) {

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

void print_usage(const char* name)
{
	printf("defaults args are: 30 18 100 512 8 0.5 1.0 -2048.0 -2048.0 2048.0 2048.0\n");

	printf("usage: \t%s \n"
		"\t%s help\n"
		"\t%s ITERATIONS\n"
		"\t%s ITERATIONS NUM_OBJECTS(2^n)\n"
		"\t%s ITERATIONS NUM_OBJECTS(2^n) NUM_BLOCKS\n"
		"\t%s ITERATIONS NUM_OBJECTS(2^n) NUM_BLOCKS NUM_THREADS\n"
		"\t%s ITERATIONS NUM_OBJECTS(2^n) NUM_BLOCKS NUM_THREADS SUB_STEPS\n"
		"\t%s ITERATIONS NUM_OBJECTS(2^n) NUM_BLOCKS NUM_THREADS SUB_STEPS RAD\n"
		"\t%s ITERATIONS NUM_OBJECTS(2^n) NUM_BLOCKS NUM_THREADS SUB_STEPS MIN_RAD MAX_RAD\n"
		"\t%s ITERATIONS NUM_OBJECTS(2^n) NUM_BLOCKS NUM_THREADS SUB_STEPS MIN_RAD MAX_RAD MIN_X MIN_Y MAX_X MAX_Y \n"
		, name, name, name, name, name, name, name, name, name, name
	);
}

template<typename T>
T get_var(const char* arg)
{

	std::istringstream iss(arg);

	T var;
	if (iss >> var) 
		return var;

	throw std::invalid_argument("wrong argument");
}

int main(int argc, char** argv)
{
	fen::col_solver::CreateInstance();

	size_t NUM_OBJECTS = 1 << 18;

	unsigned int NUM_BLOCKS = 100;
	unsigned int NUM_THREADS = 512;
	unsigned int SUB_STEPS = 8;
	unsigned int ITERATIONS = 30;
	float MIN_RAD = 0.5f;
	float MAX_RAD = 1.0f;

	float MIN_X = -2048.0f;
	float MIN_Y = -2048.0f;
	float MAX_X = +2048.0f;
	float MAX_Y = +2048.0f;

	int it = 1;

	try
	{
		if (argc > it) {

			if(!strcmp("help", argv[it]))
			{
				print_usage(argv[0]);
				return 0;
			}

			ITERATIONS = get_var<unsigned>(argv[it++]);
		}
		if (argc > it) NUM_OBJECTS = 1 << get_var<unsigned>(argv[it++]);
		if (argc > it) NUM_BLOCKS = get_var<unsigned>(argv[it++]);
		if (argc > it) NUM_THREADS = get_var<unsigned>(argv[it++]);
		if (argc > it) SUB_STEPS = get_var<unsigned>(argv[it++]);
		if (argc > it) MIN_RAD = MAX_RAD = get_var<float>(argv[it++]);
		if (argc > it) MAX_RAD = get_var<float>(argv[it++]);
		if (argc > it + 3) {
			MIN_X = get_var<float>(argv[it++]);
			MIN_Y = get_var<float>(argv[it++]);
			MAX_X = get_var<float>(argv[it++]);
			MAX_Y = get_var<float>(argv[it++]);
		}
		if (MAX_RAD < MIN_RAD ||
			MIN_X + MAX_RAD >= MAX_X ||
			MIN_Y + MAX_RAD >= MAX_Y ||
			MAX_X - MAX_RAD <= MIN_X ||
			MAX_Y - MAX_RAD <= MIN_Y)
		{
			print_usage(argv[0]);
			return 0;
		}
	} catch (...)
	{
		print_usage(argv[0]);
		return 0;
	}


	auto seed = time(nullptr);
	srand(seed);

	printf("Seed: %llu\n", seed);

	printf("Args used: ITERATIONS: %u; NUM_OBJECTS: %u; NUM_BLOCKS: %u; NUM_THREADS: %u; SUB_STEPS: %u;\n\tMIN_RAD: %.2f; MAX_RAD: %.2f; MIN_X: %.2f; MIN_Y: %.2f; MAX_X: %.2f; MAX_Y: %.2f\n", 
		ITERATIONS, NUM_OBJECTS, NUM_BLOCKS, NUM_THREADS, SUB_STEPS, MIN_RAD, MAX_RAD, MIN_X, MIN_Y, MAX_X, MAX_Y);

	constexpr bool CALCULATE_RAD = false;

	auto positions = thrust::device_vector<float>(NUM_OBJECTS * DIM);
	auto delta_mov = thrust::device_vector<float>(NUM_OBJECTS * DIM);
	auto radius = thrust::device_vector<float>(NUM_OBJECTS);

	auto col_solver = fen::col_solver::Instance();

	fen::init_objects(seed, NUM_BLOCKS, NUM_THREADS, positions, radius, delta_mov, MIN_X, MAX_X, MIN_Y, MAX_Y, MIN_RAD, MAX_RAD);

	thrust::host_vector<float> h_positions = positions;
	thrust::host_vector<float> h_radius = radius;

	//print_entities(h_positions, h_radius, NUM_OBJECTS, 16);

	col_solver->reset(NUM_OBJECTS, CALCULATE_RAD ? -1.0f : MAX_RAD, MIN_X, MIN_Y, MAX_X, MAX_Y);

	for (int i = 0; i < ITERATIONS; ++i) {

		const unsigned int collisions_1 = col_solver->solve_cols_1(NUM_BLOCKS, NUM_THREADS, positions, radius, delta_mov, NUM_OBJECTS);
		for (unsigned s = 0; s < SUB_STEPS - 1u; ++s)
		{
			col_solver->solve_cols_1(NUM_BLOCKS, NUM_THREADS, positions, radius, delta_mov, NUM_OBJECTS);
		}

		col_solver->get_profiler().next_step();
		printf("col:%u\n", collisions_1);
	}
	std::cout << '\n';
	col_solver->get_profiler().print_avg_times<Solver_Execution_Steps>(std::cout);
	std::cout << '\n';
	col_solver->get_profiler().print_times(std::cout);

    return 0;
}
