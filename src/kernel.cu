#include "col_solver.cuh"
#include "defines.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <future>
#include <cstdlib>
#include <sstream>

#include "simple_profiler.h"

/**
 * \brief Used to test whether the results from col_solver were correct or not
 * \param positions host vector
 * \param radius host vector
 * \param num_objects number of entities
 * \return collisions
 */
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


/**
 * \brief Prints n random entities
 * \param positions host vector
 * \param radius host vector
 * \param num_entities number of entities
 * \param n number of prints to do
 */
void print_entities(thrust::host_vector<float>& positions, thrust::host_vector<float>& radius, size_t num_entities, unsigned int n)
{
	for (unsigned i = 0u; i < n; i++) {

		int e = rand() % n;

		printf("\n%.4i: ", i);

		for (int j = 0; j < DIM; j++) {
			printf(" %10.5f ", positions[i + j * num_entities]);
		}

		printf("\t%.5f", radius[i]);
	}

	printf("\n");
}


/**
 * \brief Prints usage
 * \param name executable file name
 */
void print_usage(const char* name)
{
	printf("defaults args are: 30 18 100 512 8 0.5 1.0 0 -2048.0 -2048.0 2048.0 2048.0\n");

	printf("usage: \t%s \n", name);
	printf("\t%s help\n", name);
	printf("\t%s ITERATIONS\n", name);
	printf("\t%s ITERATIONS NUM_OBJECTS(2^n, n<%llu)\n", name, BITS_START - 2);
	printf("\t%s ITERATIONS NUM_OBJECTS(2^n, n<%llu) NUM_BLOCKS\n", name, BITS_START - 2);
	printf("\t%s ITERATIONS NUM_OBJECTS(2^n, n<%llu) NUM_BLOCKS NUM_THREADS(<=1024)\n", name, BITS_START - 2);
	printf("\t%s ITERATIONS NUM_OBJECTS(2^n, n<%llu) NUM_BLOCKS NUM_THREADS(<=1024) SUB_STEPS\n", name, BITS_START - 2);
	printf("\t%s ITERATIONS NUM_OBJECTS(2^n, n<%llu) NUM_BLOCKS NUM_THREADS(<=1024) SUB_STEPS RAD\n", name, BITS_START - 2);
	printf("\t%s ITERATIONS NUM_OBJECTS(2^n, n<%llu) NUM_BLOCKS NUM_THREADS(<=1024) SUB_STEPS MIN_RAD MAX_RAD\n", name, BITS_START - 2);
	printf("\t%s ITERATIONS NUM_OBJECTS(2^n, n<%llu) NUM_BLOCKS NUM_THREADS(<=1024) SUB_STEPS MIN_RAD MAX_RAD CALCULATE_RAD\n", name, BITS_START - 2);
	printf("\t%s ITERATIONS NUM_OBJECTS(2^n, n<%llu) NUM_BLOCKS NUM_THREADS(<=1024) SUB_STEPS MIN_RAD MAX_RAD CALCULATE_RAD MIN_X MIN_Y MAX_X MAX_Y\n", name, BITS_START - 2);
}


/**
 * \brief Gets a var value from c string
 * \tparam T Type of value to read
 * \param arg c string
 * \return the value
 * \throws invalid_argument: when the parsing fails (due to mismatch of types)
 */
template<typename T>
T get_var(const char* arg)
{

	std::istringstream iss(arg);

	T var;
	if (iss >> var) 
		return var;

	std::ostringstream ss;
	ss << "wrong argument error -> " << typeid(T).name() << ": " << arg;
	throw std::invalid_argument(ss.str());
}


/**
 * \brief Reads and applies command line arguments
 * \return False when a wrong argument was inputted
 */
bool read_args(int argc, char** argv, size_t& NUM_OBJECTS, unsigned int& NUM_BLOCKS, unsigned int& NUM_THREADS, unsigned int& SUB_STEPS, unsigned int& ITERATIONS, float& MIN_RAD, float& MAX_RAD, bool& CALCULATE_RAD, float& MIN_X, float& MIN_Y, float& MAX_X, float& MAX_Y)
{
	int it = 1;
	unsigned int n_objects_raise = 0;

	try
	{
		if (argc > it) {

			if (!strcmp("help", argv[it]))
			{
				return false;
			}

			ITERATIONS = get_var<unsigned>(argv[it++]);
		}
		if (argc > it) n_objects_raise = get_var<unsigned>(argv[it++]);
		if (argc > it) NUM_BLOCKS = get_var<unsigned>(argv[it++]);
		if (argc > it) NUM_THREADS = get_var<unsigned>(argv[it++]);
		if (argc > it) SUB_STEPS = get_var<unsigned>(argv[it++]);
		if (argc > it) MIN_RAD = MAX_RAD = get_var<float>(argv[it++]);
		if (argc > it) MAX_RAD = get_var<float>(argv[it++]);
		if (argc > it) CALCULATE_RAD = get_var<bool>(argv[it++]);
		if (argc > it + 3) {
			MIN_X = get_var<float>(argv[it++]);
			MIN_Y = get_var<float>(argv[it++]);
			MAX_X = get_var<float>(argv[it++]);
			MAX_Y = get_var<float>(argv[it++]);
		}

		// Ranges check
		if (n_objects_raise > (BITS_START - 2) ||
			NUM_THREADS > 1024 || 
			MAX_RAD < MIN_RAD ||
			MIN_X + MAX_RAD >= MAX_X ||
			MIN_Y + MAX_RAD >= MAX_Y ||
			MAX_X - MAX_RAD <= MIN_X ||
			MAX_Y - MAX_RAD <= MIN_Y)
		{
			return false;
		}
	}
	catch (const std::invalid_argument& e)
	{
		fprintf(stderr, "%s\n", e.what());
		return false;
	}

	// convert to 2 ^ n
	if(n_objects_raise)
		NUM_OBJECTS = 1 << n_objects_raise;

	printf("Args used: ITERATIONS: %u; NUM_OBJECTS: %llu; NUM_BLOCKS: %u; NUM_THREADS: %u; SUB_STEPS: %u;\n\tMIN_RAD: %.2f; MAX_RAD: %.2f; CALCULATE_RAD: %s; MIN_X: %.2f; MIN_Y: %.2f; MAX_X: %.2f; MAX_Y: %.2f\n",
		ITERATIONS, NUM_OBJECTS, NUM_BLOCKS, NUM_THREADS, SUB_STEPS, MIN_RAD, MAX_RAD, CALCULATE_RAD ? "true" : "false", MIN_X, MIN_Y, MAX_X, MAX_Y);  // NOLINT(clang-diagnostic-double-promotion)

	return true;
}

int main(int argc, char** argv)
{
	// Create a solver instance
	fen::col_solver::CreateInstance();

	// Default values for parameters
	size_t NUM_OBJECTS = 1 << 18;

	unsigned int NUM_BLOCKS = 100;
	unsigned int NUM_THREADS = 512;
	unsigned int SUB_STEPS = 8;
	unsigned int ITERATIONS = 30;
	float MIN_RAD = 0.5f;
	float MAX_RAD = 1.0f;
	bool CALCULATE_RAD = false;

	float MIN_X = -2048.0f;
	float MIN_Y = -2048.0f;
	float MAX_X = +2048.0f;
	float MAX_Y = +2048.0f;

	// Read user args
	if(!read_args(argc, argv, NUM_OBJECTS, NUM_BLOCKS, NUM_THREADS, SUB_STEPS, ITERATIONS, MIN_RAD, MAX_RAD, CALCULATE_RAD, MIN_X, MIN_Y, MAX_X, MAX_Y))
	{
		print_usage(argv[0]);
		return 0;
	}

	// Create host vectors
	auto positions = thrust::device_vector<float>(NUM_OBJECTS * DIM);
	auto delta_mov = thrust::device_vector<float>(NUM_OBJECTS * DIM);
	auto radius = thrust::device_vector<float>(NUM_OBJECTS);

	// Get random seed
	auto seed = time(nullptr);
	srand(seed);
	printf("Seed used: %llu\n", seed);

	// Initialize objects using the GPU
	fen::init_objects(seed, NUM_BLOCKS, NUM_THREADS, positions, radius, delta_mov, MIN_X, MAX_X, MIN_Y, MAX_Y, MIN_RAD, MAX_RAD);

	// Print a couple entities 
	//thrust::host_vector<float> h_positions = positions;
	//thrust::host_vector<float> h_radius = radius;
	//print_entities(h_positions, h_radius, NUM_OBJECTS, 16);

	// Get a solver instance
	auto col_solver = fen::col_solver::Instance();

	// Reset the instance
	col_solver->reset(NUM_OBJECTS, CALCULATE_RAD ? -1.0f : MAX_RAD, MIN_X, MIN_Y, MAX_X, MAX_Y);

	// Run the solver ITERATIONS times 
	for (unsigned i = 0; i < ITERATIONS; ++i) {

		// Only get the collisions value for the first sub-step
		const unsigned int collisions_1 = col_solver->solve_cols(NUM_BLOCKS, NUM_THREADS, positions, radius, delta_mov, NUM_OBJECTS);

		// Run the remaining sub-steps
		for (unsigned s = 0; s < SUB_STEPS - 1u; ++s)
		{
			col_solver->solve_cols(NUM_BLOCKS, NUM_THREADS, positions, radius, delta_mov, NUM_OBJECTS);
		}

		// Add a step to the profiler (to compute the avg times accordingly)
		col_solver->get_profiler().next_step();

		printf("col:%u\n", collisions_1);
	}

	// Print the profiling results
	std::cout << '\n';
	col_solver->get_profiler().print_avg_times<Solver_Execution_Steps>(std::cout);
	std::cout << '\n';
	col_solver->get_profiler().print_times(std::cout);

    return 0;
}
