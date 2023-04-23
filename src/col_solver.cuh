#pragma once

#include "singleton.h"

#include <cstdint>
#include <curand.h>
#include <thrust/device_vector.h>

#include "profiler_steps_enum.h"
#include "simple_profiler.h"

namespace fen
{

/**
 * \brief Initializes the device vector with random data
 */
void init_objects(unsigned long long seed, unsigned num_blocks, unsigned num_threads, thrust::device_vector<float>& positions, thrust::device_vector<float>& radius, thrust::device_vector<float>& delta_mov, 
	float min_pos_x, float max_pos_x, float min_pos_y, float max_pos_y, float min_radius, float max_radius);


/**
 * \brief Singleton class that computes the collision of 2D particles using spatial subdivision
 */
class col_solver : public Singleton<col_solver>
{
	friend Singleton;
public:

	// Singleton requirements
	col_solver(const col_solver& other) = delete;
	col_solver& operator=(const col_solver& other) = delete;
	col_solver(col_solver&& other) = delete;
	col_solver& operator=(col_solver&& other) = delete;

	/**
	 * \brief Resets the solver
	 */
	void reset(const size_t num_entities, const float max_rad, const float min_pos_x, const float min_pos_y, const float max_pos_x, const float max_pos_y)
	{
		init_solver(num_entities, max_rad, min_pos_x, min_pos_y, max_pos_x, max_pos_y);
	}

	/**
	 * \brief counts the number of collisions of entities (circles) with a given position and radius
	 * \param num_blocks number of blocks cuda will use
	 * \param num_threads number of threads per block cuda will use
	 * \param positions array of x and y positions of entities
	 * \param radius array of entities radius
	 * \param num_entities number of entities
	 * \return number of collisions
	 */
	unsigned int solve_cols(unsigned int num_blocks, unsigned int num_threads, thrust::device_vector<float>& positions, thrust::device_vector<float>& radius, thrust::device_vector<float>& delta_mov, const size_t num_entities);

private:

	// Singleton requirements
	col_solver();
	virtual ~col_solver() override;

protected:

	
	/**
	 * \brief Initializes the solver
	 */
	void init_solver(const size_t num_entities, const float max_rad, const float min_pos_x, const float min_pos_y, const float max_pos_x, const float max_pos_y);


	unsigned int init_cells(unsigned num_blocks, unsigned num_threads, thrust::device_vector<float>& positions, thrust::device_vector<float>& radius, const size_t num_entities);
	void sort_cells();
	unsigned int count_cols(unsigned num_blocks, unsigned num_threads, thrust::device_vector<float>& positions, thrust::device_vector<float>& radius, thrust::device_vector<float>& delta_mov, const size_t num_entities, const unsigned int num_cells);

	// <device> temp unsigned int
	unsigned int* temp{};

	// <device> entities vector
	thrust::device_vector<uint32_t> cells;
	thrust::device_vector<uint32_t> objects;
	thrust::device_vector<uint64_t> col_cells;

	// world parameters
	float max_rad, min_pos_x, min_pos_y, max_pos_x, max_pos_y;
	float width, height;

	// cell size used for spatial subdivision
	float cell_size;

	// random generator
	curandGenerator_t generator{};

	// profiler, double precision and in milliseconds
	SimpleProfiler<Solver_Execution_Steps::ALL_, double, std::milli> profiler;

public:
	// getters

	auto& get_profiler() { return profiler; }
	auto& get_generator() { return generator; }
};


};