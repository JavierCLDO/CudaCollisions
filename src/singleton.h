#pragma once
#include <cassert>
#include <memory>

#define COMMA ,

#define INIT_INSTANCE_STATIC(T) std::unique_ptr<Singleton<T>> Singleton<T>::instance_ = { nullptr }

template <typename T>
class Singleton
{
public:
	virtual ~Singleton() = default;

	Singleton(const Singleton& other) = delete;
	Singleton& operator=(const Singleton& other) = delete;

	Singleton(Singleton&& other) = delete;
	Singleton& operator=(Singleton&& other) = delete;

	template <typename ...Ts>
	static void CreateInstance(Ts&& ...args)
	{
		assert(instance_.get() == nullptr);

		instance_.reset(new T(std::forward<Ts>(args)...));
	}

	static void DeleteInstance()
	{
		assert(instance_.get() != nullptr);

		instance_.reset(nullptr);
	}

	[[nodiscard]] static T* Instance() noexcept {
		return static_cast<T*>(instance_.get());
	}

protected:

	Singleton() = default;

	static std::unique_ptr<Singleton> instance_;
};
