#pragma once
#include <cassert>
#include <memory>

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

	/**
	 * \brief Creates an instance
	 * \param args Arguments forwarded to T
	 */
	template <typename ...Ts>
	static void CreateInstance(Ts&& ...args)
	{
		assert(instance_.get() == nullptr);

		instance_.reset(new T(std::forward<Ts>(args)...));
	}

	/**
	 * \brief Deletes the instance
	 */
	static void DeleteInstance()
	{
		assert(instance_.get() != nullptr);
		instance_.reset(nullptr);
	}


	/**
	 * \brief Retrieves the static instance. It does not create an instance!!! \n
	 * To create an instance use CreateInstance. It's done this way so you can be sure this function will do exactly as the name suggests.
	 * \return The instance casted to T
	 */
	[[nodiscard]] static T* Instance()
	{
		assert(instance_.get() != nullptr);
		return static_cast<T*>(instance_.get());
	}

protected:

	Singleton() = default;

	static std::unique_ptr<Singleton> instance_;
};
