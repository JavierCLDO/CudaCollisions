#pragma once

constexpr auto DIM = 2;
constexpr auto DIM_2 = 4;
constexpr auto DIM_3 = 9;

constexpr uint64_t BITS_START = 24ui64;
constexpr uint64_t BITS_HOME = 16ui64;
constexpr uint64_t BITS_PHANTOM = (sizeof(uint64_t) * 8ui64) - BITS_START - BITS_HOME;

constexpr uint64_t BITS_OFFSET_HOME = BITS_PHANTOM;
constexpr uint64_t BITS_OFFSET_START = BITS_PHANTOM + BITS_HOME;

constexpr uint64_t START_LIMIT = 1ui64 << BITS_START;
constexpr uint64_t HOME_LIMIT = 1ui64 << BITS_HOME;
constexpr uint64_t PHANTOM_LIMIT = 1ui64 << BITS_PHANTOM;

#define CUDA_CALL_2(fun, n_blocks, n_threads) fun <<< n_blocks, n_threads >>>
#define CUDA_CALL_3(fun, n_blocks, n_threads, mem) fun <<< n_blocks, n_threads, mem >>>