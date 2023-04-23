#pragma once
#include <crt/host_defines.h>

namespace fen
{
	/**
	 * \brief Function that gets called in the GPU when a collision happens between o1 and o2
	 */
	__device__ void kernel_on_collision(const size_t o1, const size_t o2, const size_t& n, float const* positions, float const* radius, float* delta_mov, const float& dist_sq);
}
