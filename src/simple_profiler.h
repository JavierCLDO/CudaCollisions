#pragma once

#include "singleton.h"

#include <array>
#include <chrono>
#include <numeric>
#include <string_view>

template<unsigned N, typename Precision, typename TimeRatio = std::ratio<1, 1>,
	typename = std::enable_if_t < std::is_floating_point_v<Precision> && std::_Is_ratio_v<TimeRatio> >>
	class SimpleProfiler
{
private:
	template <typename T> [[nodiscard]] constexpr const char* _get_unit()			const noexcept { return ""; }
	template <> [[nodiscard]] constexpr const char* _get_unit<std::milli>()			const noexcept { return "ms"; }
	template <> [[nodiscard]] constexpr const char* _get_unit<std::micro>()			const noexcept { return "us"; }
	template <> [[nodiscard]] constexpr const char* _get_unit<std::nano>()			const noexcept { return "ns"; }
	template <> [[nodiscard]] constexpr const char* _get_unit<std::ratio<1, 1>>()	const noexcept { return "s"; }

	using timing_array = std::array<std::chrono::time_point<std::chrono::high_resolution_clock>, N>;

public:

	SimpleProfiler()
	{
		const auto now = std::chrono::high_resolution_clock::now();
		std::fill(std::begin(timings), std::end(timings), now);
	}

	using profiler_array = std::array<Precision, N>;

	[[nodiscard]] constexpr const char* unit() const
	{
		return _get_unit<TimeRatio>();
	}

	/**
	 * \brief Adds time to M timer
	 * \tparam M the timer
	 * \param t time
	 */
	template<unsigned M, typename = std::enable_if_t<M < N>>
	void add_time(Precision t)
	{
		timers[M] += t;
	}

	/**
	 * \brief Starts timing M 
	 * \tparam M the timer
	 */
	template<unsigned M, typename = std::enable_if_t<M < N>>
	void start_timing(void)
	{
		timings[M] = std::chrono::high_resolution_clock::now();
	}

	/**
	 * \brief Ends timing M
	 * \tparam M the timer
	 */
	template<unsigned M, typename = std::enable_if_t<M < N>>
	void finish_timing(void)
	{
		add_time<M>(std::chrono::duration<Precision, TimeRatio>(std::chrono::high_resolution_clock::now() - timings[M]).count());
	}

	/**
	 * \brief Same as templated version, intended for use in loops
	 * \param m the timer
	 * \param t time
	 */
	void add_time(unsigned m, Precision t)
	{
		assert(m < N && "m needs to be less than template N");
		timers[m] += t;
	}

	/**
	 * \brief Same as templated version, intended for use in loops
	 * \param m the timer
	 */
	void start_timing(unsigned m)
	{
		assert(m < N && "m needs to be less than template N");
		timings[m] = std::chrono::high_resolution_clock::now();
	}

	/**
	 * \brief Same as templated version, intended for use in loops
	 * \param m the timer
	 */
	void finish_timing(unsigned m)
	{
		assert(m < N && "m needs to be less than template N");
		add_time(m, std::chrono::duration<Precision>(std::chrono::high_resolution_clock::now() - timings[m]).count());
	}


	/**
	 * \brief Finishes an step and computes the avg time
	 */
	void next_step(void)
	{
		++steps;

		// update avg timers
		for (unsigned i{0}; i < N; ++i) 
			avg_timers[i] = timers[i] / steps;
	}

	/**
	 * \brief Resets the profiling
	 */
	void reset(void)
	{
		steps = 0;
		std::fill(std::begin(timers), std::end(timers), 0);
		std::fill(std::begin(avg_timers), std::end(avg_timers), 0);

		const auto now = std::chrono::high_resolution_clock::now();
		std::fill(std::begin(timings), std::end(timings), now);
	}

	/**
	 * \brief Measures the time that it takes func to run
	 * \param func functor
	 * \param args arguments to forward the functor
	 * \return If the functor returns a void, returns the time it took to run.\n
	 * If the functor returns something, returns a tuple where the first element is the time and the second is whatever was returned by functor
	 */
	template <typename Func, typename ...Args>
	auto measure_time(Func func, Args&&... args)
	{
		using FuncReturnType = decltype(func(std::forward<Args>(args)...));

		const auto start = std::chrono::high_resolution_clock::now();
		if constexpr (std::is_same_v<void, FuncReturnType>) { // If returns a void, no need to return a tuple
			func(std::forward<Args>(args)...);
			const auto end = std::chrono::high_resolution_clock::now();
			return std::chrono::duration<Precision, TimeRatio>(end - start).count();
		}
		else { // If returns something, return a tuple
			FuncReturnType ret = func(std::forward<Args>(args)...); 
			const auto end = std::chrono::high_resolution_clock::now();
			return std::tuple<Precision, FuncReturnType> {
				std::chrono::duration<Precision, TimeRatio>(end - start).count(), ret
			};
		}
	}

	/**
	 * \brief Measures and adds the time to timer M
	 * \param func functor
	 * \param args arguments to forward the functor
	 * \return If the functor returns something, it returns that. Else it returns void
	 */
	template <unsigned M, typename Func, typename ...Args, typename = std::enable_if_t<M < N>>
	auto add_time(Func func, Args&&... args)
	{
		using FuncReturnType = decltype(func(std::forward<Args>(args)...));

		const auto start = std::chrono::high_resolution_clock::now();
		if constexpr (std::is_same_v<void, FuncReturnType>) { // If returns a void, no need to return a tuple
			func(std::forward<Args>(args)...);
			const auto end = std::chrono::high_resolution_clock::now();
			add_time<M>(std::chrono::duration<Precision, TimeRatio>(end - start).count());
			return;
		}
		else { // If returns something, return a tuple
			FuncReturnType ret = func(std::forward<Args>(args)...);
			const auto end = std::chrono::high_resolution_clock::now();
			add_time<M>(std::chrono::duration<Precision, TimeRatio>(end - start).count());
			return ret;
		}
	}

	void print_times(std::ostream& os) const
	{
		Precision total = 0.0;
		for(unsigned i{0}; i < N; ++i)
		{
			os << "time for " << i << ": " << timers[i] << ' ' << unit() << '\n';
			total += timers[i];
		}
		os << "total time: " << total << ' ' << unit() << '\n';
	}

	void print_avg_times(std::ostream& os) const
	{
		assert(steps > 0 && "you need to finish a step before");
		Precision total = 0.0;
		for (unsigned i{ 0 }; i < N; ++i)
		{
			os << "avg time for " << i << ": " << avg_timers[i] << ' ' << unit() << '\n';
			total += avg_timers[i];
		}
		os << "total avg time: " << total << ' ' << unit() << '\n';
	}

	const Precision& total_time()
	{
		return std::accumulate(std::begin(timers), std::end(timers), 0.0);
	}

	const Precision& total_avg_time()
	{
		return std::accumulate(std::begin(avg_timers), std::end(avg_timers), 0.0);
	}

private:

	profiler_array timers{};
	profiler_array avg_timers{};
	timing_array timings{};

	unsigned int steps{};

public:

	[[nodiscard]] const unsigned int& get_steps(void) const noexcept { return steps; }
	[[nodiscard]] const profiler_array& get_times(void) const noexcept { return timers; }
	[[nodiscard]] const profiler_array& get_avg_times(void) const noexcept { assert(steps > 0 && "you need to finish a step before"); return avg_timers; }

	template<unsigned M, typename = std::enable_if_t<M < N>>
	[[nodiscard]] const Precision& get_time(void) const noexcept { return timers[M]; }
	[[nodiscard]] const Precision& get_time(unsigned m) const noexcept { assert(m < N && "m needs to be less than template N"); return timers[m]; }
};