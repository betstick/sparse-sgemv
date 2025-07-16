// Copyright © 2025 betstick
// All rights reserved.
// This code is proprietary and confidential. It is NOT licensed for training AI models.

#pragma once
#include "driver_types.h"
#ifdef __CUDACC__
#	define CPU_GPU __host__ __device__
#else
#	define CPU_GPU
#endif
#include <stdint.h>
#include <string.h>
#include <string>
#include <chrono>

using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

#define PCG32_DEFAULT_STATE 0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT 0x5851f42d4c957f2dULL

CPU_GPU inline uint64_t mix_bits(uint64_t v)
{
	v ^= (v >> 31);
	v *= 0x7fb5d329728ea185;
	v ^= (v >> 27);
	v *= 0x81dadef4bc2dd44d;
	v ^= (v >> 33);
	return v;
};

struct alignas(16) pcg32_t
{
	u64 state;
	u64 inc;

	inline pcg32_t() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {};

	inline pcg32_t(u64 seq_index, u64 offset) {set_sequence(seq_index,offset);};

	inline pcg32_t(u64 seq_index) {set_sequence(seq_index);};

	inline void set_sequence(u64 seq_index) {set_sequence(seq_index,mix_bits(seq_index));};

	inline void set_sequence(u64 seq_index, u64 seed)
	{
		state = 0u;
		inc = (seq_index << 1u) | 1u;
		uniform_u32_1();
		state += seed;
		uniform_u32_1();
	};

	inline void advance(i64 i_delta)
	{
		u64 cur_mult = PCG32_MULT;
		u64 cur_plus = inc;
		u64 acc_mult = 1u;
		u64 acc_plus = 0u;
		u64 delta = (u64)i_delta;

		while(delta > 0)
		{
			if(delta & 1)
			{
				acc_mult *= cur_mult;
				acc_plus = acc_plus * cur_mult + cur_plus;
			}

			cur_plus = (cur_mult + 1) * cur_plus;
			cur_mult *= cur_mult;
			delta /= 2;
		}

		state = acc_mult * state + acc_plus;
	};

	inline float u() {return uniform_u32_1() * 0x1p-32f;};

	inline u32 uniform_u32_1()
	{
		u64 old_state = state;
		state = old_state * PCG32_MULT + inc;
		u32 xorshifted = (u32)(((old_state >> 18u) ^ old_state) >> 27u);
		u32 rot = (u32)(old_state >> 59u);
		return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
	};
};

struct rng_t
{
	pcg32_t m_pcg;

	rng_t() {};

	void seed(u64 s0, u64 s1) {m_pcg.set_sequence(0,s0);};

	inline float u() {return m_pcg.u();};
};

namespace nanobench
{
	template <typename T>
	void doNotOptimizeAway(T const& val)
	{
		// NOLINTNEXTLINE(hicpp-no-assembler)
		asm volatile("" : : "r,m"(val) : "memory");
	};

	template <typename T>
	void doNotOptimizeAway(T& val)
	{
	#   if defined(__clang__)
		// NOLINTNEXTLINE(hicpp-no-assembler)
		asm volatile("" : "+r,m"(val) : : "memory");
	#   else
		// NOLINTNEXTLINE(hicpp-no-assembler)
		//changed this for the NC++ compiler
		asm volatile("" ::: "memory");
	#   endif
	};
};

namespace mem
{
	void __alloc(void** ptr, i64 bytes);

	void __free(void* ptr);

	void __alloc_gpu(void** ptr, i64 bytes);

	void __free_gpu(void* ptr);

	template <typename T>
	inline T* alloc(i64 c)
	{
		T* ptr = nullptr;
		__alloc((void**)&ptr,sizeof(T)*c);
		return ptr;
	};

	template <typename T>
	inline void free(T* ptr) {__free((void*)ptr);};

	template <typename T>
	inline T* alloc_gpu(i64 c)
	{
		T* ptr = nullptr;
		__alloc_gpu((void**)&ptr,sizeof(T)*c);
		return ptr;
	};

	template <typename T>
	inline void free_gpu(T* ptr)
	{
		if(ptr == nullptr)
			return;
		__free_gpu((void*)ptr);
	};

	void __copy_to_host(void* dst, void* src, i64 bytes);

	void __copy_to_dev(void* dst, void* src, i64 bytes);

	template <typename T>
	inline void copy_to_dev(T* src, T* dst, i32 c) {__copy_to_dev(dst,src,sizeof(T)*c);};

	template <typename T>
	inline void copy_to_host(T* src, T* dst, i32 c) {__copy_to_host(dst,src,sizeof(T)*c);};
};

namespace util
{
	//simple utility class for testing and clocking things
	struct timer_t
	{
		std::chrono::steady_clock::time_point start_;
		std::chrono::steady_clock::time_point stop_;
		std::chrono::microseconds duration;

		public:
		inline timer_t() {};

		inline void start() {this->start_ = std::chrono::steady_clock::now();};

		inline void stop()
		{
			this->stop_ = std::chrono::steady_clock::now();
			duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_);
		};

		inline double seconds() {return double((duration / 1000000.00f).count());};
	};
};

inline void print_arr(std::string label, float* f, i32 c)
{
	printf("%s: ",label.c_str());
	for(i32 i = 0; i < c; i++)
		printf("%f ",f[i]);
	printf("\n");
};

#include <cuda.h>
#include <cublas_v2.h>

struct cuda_state_t
{
	cublasHandle_t handle;
	cudaGraph_t graph;
	cudaStream_t stream;
	cudaGraphExec_t inst;

	void init_cublas();

	void free_cublas();

	void start_record();

	void stop_record();

	void run_graph(i32 count = 1);

	void free_graph();
};

struct data_t
{
	float* v = nullptr; //size x
	float* w = nullptr; //size x * y
	float* c = nullptr; //size y
	i32* mask = nullptr; //size x
	i32* scan = nullptr; //size x
	float* temp = nullptr; //size x * y

	i32* idx = nullptr; //size s
	i32* sz = nullptr; //size 1
	i32 x = 0;
	i32 y = 0;
	i32 s = 0;
	size_t bytes = 0;

	bool on_gpu = false;
	bool on_cpu = false;

	void __alloc_cpu()
	{
		v = new float[x];
		w = new float[x*y];
		c = new float[y];
		idx = new i32[x];
		mask = new i32[x];
		scan = new i32[x];
		temp = new float[x*y*10];
		sz = new i32();
		on_gpu = false;
		on_cpu = true;
	};

	void __alloc_uni()
	{
		v = mem::alloc<float>(x);
		w = mem::alloc<float>(x*y);
		c = mem::alloc<float>(y);
		mask = mem::alloc<i32>(x);
		scan = mem::alloc<i32>(x);
		temp = mem::alloc<float>(x*y*10);
		idx = mem::alloc<i32>(x);
		sz = mem::alloc<i32>(1);
		on_gpu = true;
		on_cpu = true;
	};
	
	void __alloc_gpu()
	{
		v = mem::alloc_gpu<float>(x);
		w = mem::alloc_gpu<float>(x*y);
		c = mem::alloc_gpu<float>(y);
		mask = mem::alloc_gpu<i32>(x);
		scan = mem::alloc_gpu<i32>(x);
		temp = mem::alloc_gpu<float>(x*y*10);
		idx = mem::alloc_gpu<i32>(x);
		sz = mem::alloc_gpu<i32>(1);
		on_gpu = true;
		on_cpu = false;
	};

	void set_data(rng_t &rng, float sparsity = 0.1f)
	{
		if(!on_cpu)
			return;
		s = 0;
		for(i32 i = 0; i < x; i++)
		{
			bool high = rng.u() > (1.0f - sparsity);
			v[i] = high ? 1.0f : 0.0f;
			idx[s] = 0;
			mask[i] = 0;
			if(high)
			{
				mask[i] = 1;
				idx[s] = i;
				s++;
			}
			scan[i] = 0;
		}
		for(i32 i = 0; i < x * y; i++)
			w[i] = rng.u();
		for(i32 i = 0; i < y; i++)
			c[i] = 0.0f;
		*sz = s;
	};

	void init_cpu(i32 x_, i32 y_, rng_t &rng, float sparsity = 0.1f)
	{
		x = x_;
		y = y_;
		__alloc_cpu();
		set_data(rng,sparsity);
	};

	void init_uni(i32 x_, i32 y_, rng_t &rng, float sparsity = 0.1f)
	{
		x = x_;
		y = y_;
		__alloc_uni();
		set_data(rng,sparsity);
	};

	void init_gpu(i32 x_, i32 y_, rng_t &rng, float sparsity = 0.1f)
	{
		x = x_;
		y = y_;
		__alloc_gpu();
		data_t t; t.init_cpu(x,y,rng,sparsity);
		mem::copy_to_dev(t.v,v,x);
		mem::copy_to_dev(t.w,w,x*y);
		mem::copy_to_dev(t.c,c,y);
		mem::copy_to_dev(t.idx,idx,x);
		mem::copy_to_dev(t.mask,mask,x);
		mem::copy_to_dev(t.scan,scan,x);
		mem::copy_to_dev(t.temp,temp,x*y*10);
		mem::copy_to_dev(t.sz,sz,1);
		s = t.s;
		t.free();
	};

	void free()
	{
		if(on_gpu)
		{
			mem::free_gpu(v);
			mem::free_gpu(w);
			mem::free_gpu(c);
			mem::free_gpu(idx);
			mem::free_gpu(mask);
			mem::free_gpu(scan);
			mem::free_gpu(temp);
			mem::free_gpu(sz);
		}
		else
		{
			delete[] v;
			delete[] w;
			delete[] c;
			delete[] idx;
			delete[] mask;
			delete[] scan;
			delete[] temp;
			delete sz;
		}

		x = 0;
		y = 0;
		s = 0;
	};

	void init_temp_data(cuda_state_t &state);
};

void cpu_naive_xyxy(data_t &data);

void cpu_naive_xyyx(data_t &data);

void cpu_naive_yxxy(data_t &data);

void cpu_naive_yxyx(data_t &data);

void cpu_index_xysy(data_t &data);

void cpu_index_xyys(data_t &data);

void cpu_index_yxsy(data_t &data);

void cpu_index_yxys(data_t &data);

void cpu_sgemv_n(data_t &data);

void cpu_sgemv_t(data_t &data);

//

void gpu_spgemv_xy2(cuda_state_t &state, data_t &data);

void gpu_acc_atomic(cuda_state_t &state, data_t &data);

void gpu_sgemv_n(cuda_state_t &state, data_t &data);

void gpu_sgemv_t(cuda_state_t &state, data_t &data);