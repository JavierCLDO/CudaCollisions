#pragma once

constexpr auto DIM = 2;
constexpr auto DIM_2 = 4;
constexpr auto DIM_3 = 9;

// Bits assigned to the collision list data
constexpr uint64_t BITS_START = 28ui64; // 2^BITS_START possible cells 
constexpr uint64_t BITS_HOME = 16ui64; // 2^BITS_HOME Number of home cells possible in a collision cell
constexpr uint64_t BITS_PHANTOM = (sizeof(uint64_t) * 8ui64) - BITS_START - BITS_HOME; // 2^BITS_PHANTOM Number of phantom cells possible in a collision cell

// Bits offset calculation for bit shift operations
constexpr uint64_t BITS_OFFSET_HOME = BITS_PHANTOM;
constexpr uint64_t BITS_OFFSET_START = BITS_PHANTOM + BITS_HOME;

// Range limit
constexpr uint64_t START_LIMIT = 1ui64 << BITS_START;
constexpr uint64_t HOME_LIMIT = 1ui64 << BITS_HOME;
constexpr uint64_t PHANTOM_LIMIT = 1ui64 << BITS_PHANTOM;

// Defines to avoid using the <<< >>> ugly operator
#define CUDA_CALL_2(fun, n_blocks, n_threads) fun <<< (n_blocks), (n_threads) >>>
#define CUDA_CALL_3(fun, n_blocks, n_threads, mem) fun <<< (n_blocks), (n_threads), (mem) >>>