#pragma once

#include "singleton.h"

#include <cstdint>
#include <curand.h>
#include <thrust/device_vector.h>

#include "simple_profiler.h"

namespace fen
{

class col_solver : public Singleton<col_solver>
{
	friend Singleton;
public:


	void reset(const size_t num_entities, const float max_rad, const float min_pos_x, const float min_pos_y, const float max_pos_x, const float max_pos_y)
	{
		init_solver(num_entities, max_rad, min_pos_x, min_pos_y, max_pos_x, max_pos_y);
	}

	void init_objects(unsigned num_blocks, unsigned num_threads, thrust::device_vector<float>& positions, thrust::device_vector<float>& radius, thrust::device_vector<float>& delta_mov, float min_pos_x, float max_pos_x, float min_pos_y, float max_pos_y, float min_radius, float max_radius);

	/**
	 * \brief counts the number of collisions of entities (circles) with a given position and radius
	 * \param num_blocks number of blocks cuda will use
	 * \param num_threads number of threads per block cuda will use
	 * \param positions array of x and y positions of entities
	 * \param radius array of entities radius
	 * \param num_entities number of entities
	 * \return number of collisions
	 */
	unsigned int solve_cols_1(unsigned int num_blocks, unsigned int num_threads, thrust::device_vector<float>& positions, thrust::device_vector<float>& radius, thrust::device_vector<float>& delta_mov, const size_t num_entities);

private:

	col_solver();

	virtual ~col_solver();


protected:


	void init_solver(const size_t num_entities, const float max_rad, const float min_pos_x, const float min_pos_y, const float max_pos_x, const float max_pos_y);

	unsigned int init_cells(unsigned num_blocks, unsigned num_threads, thrust::device_vector<float>& positions, thrust::device_vector<float>& radius, const size_t num_entities);
	void sort_cells();
	unsigned int count_cols_1(unsigned num_blocks, unsigned num_threads, thrust::device_vector<float>& positions, thrust::device_vector<float>& radius, thrust::device_vector<float>& delta_mov, const size_t num_entities, const unsigned int num_cells);

	unsigned int* temp;

	thrust::device_vector<uint32_t> cells;
	thrust::device_vector<uint32_t> objects;
	thrust::device_vector<uint64_t> col_cells;

	float max_rad, min_pos_x, min_pos_y, max_pos_x, max_pos_y;

	float width, height;

	float cell_size;

	curandGenerator_t generator;

	SimpleProfiler<5, double, std::milli> profiler;

public:

	auto& get_profiler() { return profiler; }
};


};