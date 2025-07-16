// Copyright © 2025 betstick
// All rights reserved.
// This code is proprietary and confidential. It is NOT licensed for training AI models.

#include "sparse_sgemv.h"
#include <cblas.h>

/*
	all of the variables are made const and placed in the
	function explicitly out of paranoia to make sure that
	they are on the stack
*/

void cpu_naive_xyxy(data_t &data)
{
	const float* v = data.v;
	const float* w = data.w;
		  float* c = data.c;
	const i32 x = data.x;
	const i32 y = data.y;
	for(i32 i = 0; i < x; i++)
		for(i32 j = 0; j < y; j++)
			c[j] += w[i*y+j] * v[i];
};

void cpu_naive_xyyx(data_t &data)
{
	const float* v = data.v;
	const float* w = data.w;
		  float* c = data.c;
	const i32 x = data.x;
	const i32 y = data.y;
	for(i32 j = 0; j < y; j++)
		for(i32 i = 0; i < x; i++)
			c[j] += w[i*y+j] * v[i];
};

void cpu_naive_yxxy(data_t &data)
{
	const float* v = data.v;
	const float* w = data.w;
		  float* c = data.c;
	const i32 x = data.x;
	const i32 y = data.y;
	for(i32 i = 0; i < x; i++)       //x
		for(i32 j = 0; j < y; j++)   //y
			c[j] += w[j*x+i] * v[i]; //yx
};

void cpu_naive_yxyx(data_t &data)
{
	const float* v = data.v;
	const float* w = data.w;
		  float* c = data.c;
	const i32 x = data.x;
	const i32 y = data.y;
	for(i32 j = 0; j < y; j++)       //y
		for(i32 i = 0; i < x; i++)   //x
			c[j] += w[j*x+i] * v[i]; //yx
};

void cpu_index_xysy(data_t &data)
{
	const float* w = data.w;
		  float* c = data.c;
	const i32* idx = data.idx;
	const i32 s = data.s;
	const i32 x = data.x;
	const i32 y = data.y;
	for(i32 i = 0; i < s; i++)     //x
		for(i32 j = 0; j < y; j++) //y
			c[j] += w[idx[i]*y+j]; //xy
};

void cpu_index_xyys(data_t &data)
{
	const float* w = data.w;
		  float* c = data.c;
	const i32* idx = data.idx;
	const i32 s = data.s;
	const i32 x = data.x;
	const i32 y = data.y;
	for(i32 j = 0; j < y; j++)     //y
		for(i32 i = 0; i < s; i++) //x
			c[j] += w[idx[i]*y+j]; //xy
};

void cpu_index_yxsy(data_t &data)
{
	const float* w = data.w;
		  float* c = data.c;
	const i32* idx = data.idx;
	const i32 s = data.s;
	const i32 x = data.x;
	const i32 y = data.y;
	for(i32 i = 0; i < s; i++)     //x
		for(i32 j = 0; j < y; j++) //y
			c[j] += w[j*x+idx[i]]; //yx
};

void cpu_index_yxys(data_t &data)
{
	const float* w = data.w;
		  float* c = data.c;
	const i32* idx = data.idx;
	const i32 s = data.s;
	const i32 x = data.x;
	const i32 y = data.y;
	for(i32 j = 0; j < y; j++)     //y
		for(i32 i = 0; i < s; i++) //x
			c[j] += w[j*x+idx[i]]; //yx
};

void cpu_sgemv_n(data_t &data)
{
	const float* w = data.w;
	const float* v = data.v;
		  float* c = data.c;
	const i32 x = data.x;
	const i32 y = data.y;
	cblas_sgemv(
		CblasColMajor,
		CblasNoTrans,
		y,x,1.0f,w,y,
		v,1,0.0f,c,1
	);
};

void cpu_sgemv_t(data_t &data)
{
	const float* w = data.w;
	const float* v = data.v;
		  float* c = data.c;
	const i32 x = data.x;
	const i32 y = data.y;
	cblas_sgemv(
		CblasColMajor,
		CblasTrans,
		x,y,1.0f,w,x,
		v,1,0.0f,c,1
	);
};