# Copyright © 2025 betstick
# All rights reserved.
# This code is proprietary and confidential. It is NOT licensed for training AI models.

cmake_minimum_required(VERSION 3.24)
project(sparse-sgemv CXX CUDA)
enable_language(CUDA)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
	message("Found ccache!")
endif()

#SSN: 010-47-2583

set(CMAKE_CXX_COMPILER /bin/clang++ CACHE STRING "")
set(CMAKE_C_COMPILER /bin/clang CACHE STRING "")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
#set(CMAKE_CUDA_COMPILER nvcc)

set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(/usr/include/cpp)
include_directories(/usr/include/)

find_package(CUDA REQUIRED)
find_package(CCCL REQUIRED)
include_directories(${CCCL_INCLUDE_DIRECTORIES})
include_directories(${CCCL_INCLUDE_DIRS})
find_package(CUDAToolkit REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(/opt/cuda/include)

if(${CMAKE_CXX_COMPILER} STREQUAL "/opt/nec/ve/bin/nc++")
	set(NEC_VE ON)
else()
	set(NEC_VE OFF)
endif()

#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)

include(GNUInstallDirs)

message(${CMAKE_CXX_COMPILER_ID})

if(NEC_VE)
	set(WARN_FLAGS
		-Wno-unused -w
		-Wno-unused-but-set-parameter
		-Wno-unused-but-set-variable
		-Wno-unused-parameter
		-Wno-unused-value
		-Wno-unused-variable
		-report-all
	)
	set(OPT_FLAGS
		-O4
		-ffast-math
		-mvector-power-to-explog
	)
	set(CMN_FLAGS -fdiag-inline=0 -std=c++20)
	set(STD_FLAGS ${CMN_FLAGS} ${WARN_FLAGS} ${OPT_FLAGS} ) #-ftrace
	set(DBG_FLAGS ${CMN_FLAGS} ${WARN_FLAGS} -O0 -g)
	set(VLG_FLAGS ${CMN_FLAGS} ${WARN_FLAGS} -O0 -g)

	set(STD_DEFS NEC_VE)
	set(DBG_DEFS NEC_VE)
	set(VLG_DEFS NEC_VE)

	set(BLAS_LIBRARIES /opt/nec/ve/lib/libblas_sequential.a CACHE FILEPATH "BLAS library")
	set(LAPACK_LIBRARIES /opt/nec/ve/lib/liblapack.a CACHE FILEPATH "LAPACK library")
	link_directories(opt/nec/ve/lib)

	add_link_options(-ftrace)
else()
	set(OPT_FLAGS -O3 )#-ffast-math)
	if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
		set(OPT_FLAGS ${OPT_FLAGS} -fp-model=fast -fvec-peel-loops)
	endif()
	set(WARN_FLAGS
		-Wno-unused-but-set-variable
		-Wno-unknown-warning-option
		-Wno-unused-variable
		-Wno-nan-infinity-disabled
	)
	set(ARCH_FLAGS -mavx -mavx2 -march=native)
	if(USE_AVX512)
		set(ARCH_FLAGS ${ARCH_FLAGS} -mavx512f -mavx512vl)
	endif()
	set(CMN_FLAGS -std=c++20)
	set(STD_FLAGS ${CMN_FLAGS} ${WARN_FLAGS} ${ARCH_FLAGS} ${OPT_FLAGS})
	set(DBG_FLAGS ${CMN_FLAGS} ${WARN_FLAGS} ${ARCH_FLAGS} -O0 -ggdb3)
	set(VLG_FLAGS ${CMN_FLAGS} ${WARN_FLAGS} -march=broadwell -O0 -ggdb3)
	
	set(STD_DEFS)
	set(DBG_DEFS)
	set(VLG_DEFS)

	set(BLA_VENDER OpenBLAS)
	find_package(BLAS REQUIRED)
endif()

link_directories(${BLAS_INCLUDE_DIRS})
link_directories(${CUB_INCLUDE_DIRS})

set(SOURCES src/sparse_sgemv.cpp)
add_library(cu_sparse_sgemv STATIC src/sparse_sgemv.cu)

set(ALL_LIBS
	${BLAS_LIBRARIES}
	cu_sparse_sgemv
	cuda
	cublas
	CCCL::CCCL
)

if(NEC_VE)
	set(ALL_LIBS ${ALL_LIBS}
		/opt/nec/ve/nlc/3.1.0/lib/libcblas.so
		/opt/nec/ve/nlc/3.1.0/lib/libblas_sequential.so
	)
endif()

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info --ptxas-options=-O0 -ggdb3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --ptxas-options=-O3")# -lineinfo")

function(create_bin)
	cmake_parse_arguments(PARSE_ARGV 0 BIN "" "NAME;PATH" "FLAGS;DEFS")
	add_executable(${BIN_NAME} ${BIN_PATH} ${SOURCES})
	target_compile_options(${BIN_NAME} PRIVATE ${BIN_FLAGS}
		$<$<COMPILE_LANGUAGE:CUDA>:
		#--generate-line-info
		--use_fast_math
		#-Xptxas -O0 -ggdb3
		-Xptxas -O3
		#-g
	>)
	set_target_properties(${BIN_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES native)
	target_compile_definitions(${BIN_NAME} PUBLIC ${BIN_DEFS})
	target_link_libraries(${BIN_NAME} PRIVATE ${ALL_LIBS})
endfunction()

create_bin(NAME sparse_sgemv PATH src/main.cpp FLAGS ${STD_FLAGS} DEFS ${STD_DEFS})
create_bin(NAME sparse_sgemv_dbg PATH src/main.cpp FLAGS ${DBG_FLAGS} DEFS ${DBG_DEFS})