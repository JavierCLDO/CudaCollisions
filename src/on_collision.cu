#include "on_collision.cuh"

#include <corecrt_math.h>

namespace fen
{
	__device__ void kernel_on_collision(const size_t o1, const size_t o2, const size_t& n, float const* positions, float const* radius, float* delta_mov, const float& dist_sq)
	{
		const float dist = sqrtf(dist_sq);
		const float min_dist = radius[o1] + radius[o2];

		const float col_axis_x = positions[o2] - positions[o1];
		const float col_axis_y = positions[o2 + n] - positions[o1 + n];

		const float mass_ratio_1 = radius[o1] / min_dist;
		const float mass_ratio_2 = radius[o2] / min_dist;

		const float delta = (dist - min_dist);

		const float n_x = col_axis_x / dist;
		const float n_y = col_axis_y / dist;

		delta_mov[o1] += n_x * mass_ratio_1 * delta;
		delta_mov[o1 + n] += n_y * mass_ratio_1 * delta;

		delta_mov[o2] -= n_x * mass_ratio_2 * delta;
		delta_mov[o2 + n] -= n_y * mass_ratio_2 * delta;
	}
}

