/////////////////////////////////////////////////////////////////////////////////
// File : CUDAPython.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA Module for Python, PyBind11 Bindings
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef CUDAPYTHON_CPP
#define CUDAPYTHON_CPP

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../Scarab-Engine-64/ThirdParty/CUDA/CUDAMappings.h"
#include "../Scarab-Engine-64/ThirdParty/CUDA/CUDAContext.h"

#include "Python.h"
#include "pybind11/embed.h"
namespace py = pybind11;

/////////////////////////////////////////////////////////////////////////////////
// CUDAPython module start
PYBIND11_MODULE( CUDAPython, hModule ) {
	hModule.doc() = "CUDA API for Python, powered by Scarab-Engine-64.";

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
//py::enum_<CUDALimit>( hModule, "CUDALimit" )
//	.value( "CUDA_LIMIT_THREAD_STACK_SIZE", 			CUDA_LIMIT_THREAD_STACK_SIZE )
//	.value( "CUDA_LIMIT_PRINTF_FIFO_SIZE", 				CUDA_LIMIT_PRINTF_FIFO_SIZE )
//	.value( "CUDA_LIMIT_HEAP_SIZE", 					CUDA_LIMIT_HEAP_SIZE )
//	.value( "CUDA_LIMIT_RUNTIME_SYNC_DEPTH", 			CUDA_LIMIT_RUNTIME_SYNC_DEPTH )
//	.value( "CUDA_LIMIT_RUNTIME_PENDING_CALLS", 		CUDA_LIMIT_RUNTIME_PENDING_CALLS )
//	.value( "CUDA_LIMIT_L2_MAX_FETCH_GRANULARITY", 		CUDA_LIMIT_L2_MAX_FETCH_GRANULARITY )
//	.value( "CUDA_LIMIT_L2_PERSISTENT_CACHE_LINE_SIZE", CUDA_LIMIT_L2_PERSISTENT_CACHE_LINE_SIZE )
//	.value( "CUDA_LIMIT_COUNT", 						CUDA_LIMIT_COUNT )
//;
//
//py::enum_<CUDADeviceAttribute>( hModule, "CUDADeviceAttribute" )
//	.value( "CUDA_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",				 CUDA_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK )
//	.value( "CUDA_DEVICE_ATTRIBUTE_BLOCK_MAX_DIM_X",                     CUDA_DEVICE_ATTRIBUTE_BLOCK_MAX_DIM_X )
//	.value( "CUDA_DEVICE_ATTRIBUTE_BLOCK_MAX_DIM_Y",                     CUDA_DEVICE_ATTRIBUTE_BLOCK_MAX_DIM_Y )
//	.value( "CUDA_DEVICE_ATTRIBUTE_BLOCK_MAX_DIM_Z",                     CUDA_DEVICE_ATTRIBUTE_BLOCK_MAX_DIM_Z )
//	.value( "CUDA_DEVICE_ATTRIBUTE_GRID_MAX_DIM_X",                      CUDA_DEVICE_ATTRIBUTE_GRID_MAX_DIM_X )
//	.value( "CUDA_DEVICE_ATTRIBUTE_GRID_MAX_DIM_Y",                      CUDA_DEVICE_ATTRIBUTE_GRID_MAX_DIM_Y )
//	.value( "CUDA_DEVICE_ATTRIBUTE_GRID_MAX_DIM_Z",                      CUDA_DEVICE_ATTRIBUTE_GRID_MAX_DIM_Z )
//	.value( "CUDA_DEVICE_ATTRIBUTE_MAX_SHAREDMEM_PER_BLOCK",             CUDA_DEVICE_ATTRIBUTE_MAX_SHAREDMEM_PER_BLOCK )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY",               CUDA_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY )
//	.value( "CUDA_DEVICE_ATTRIBUTE_WARP_THREAD_COUNT",                   CUDA_DEVICE_ATTRIBUTE_WARP_THREAD_COUNT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_MAX_PITCH",                           CUDA_DEVICE_ATTRIBUTE_MAX_PITCH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK",             CUDA_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK )
//	.value( "CUDA_DEVICE_ATTRIBUTE_PEAK_CLOCK_FREQUENCY",                CUDA_DEVICE_ATTRIBUTE_PEAK_CLOCK_FREQUENCY )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT",                   CUDA_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_COPY_AND_EXECUTE",                    CUDA_DEVICE_ATTRIBUTE_COPY_AND_EXECUTE )
//	.value( "CUDA_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT",                CUDA_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_KERNEL_TIMEOUT",                      CUDA_DEVICE_ATTRIBUTE_KERNEL_TIMEOUT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_INTEGRATED",                          CUDA_DEVICE_ATTRIBUTE_INTEGRATED )
//	.value( "CUDA_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY",                 CUDA_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY )
//	.value( "CUDA_DEVICE_ATTRIBUTE_COMPUTE_MODE",                        CUDA_DEVICE_ATTRIBUTE_COMPUTE_MODE )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_1D_MAX_WIDTH",                CUDA_DEVICE_ATTRIBUTE_TEXTURE_1D_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_2D_MAX_WIDTH",                CUDA_DEVICE_ATTRIBUTE_TEXTURE_2D_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_2D_MAX_HEIGHT",               CUDA_DEVICE_ATTRIBUTE_TEXTURE_2D_MAX_HEIGHT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_3D_MAX_WIDTH",                CUDA_DEVICE_ATTRIBUTE_TEXTURE_3D_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_3D_MAX_HEIGHT",               CUDA_DEVICE_ATTRIBUTE_TEXTURE_3D_MAX_HEIGHT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_3D_MAX_DEPTH",                CUDA_DEVICE_ATTRIBUTE_TEXTURE_3D_MAX_DEPTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLAYERED_MAX_WIDTH",         CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLAYERED_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLAYERED_MAX_HEIGHT",        CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLAYERED_MAX_HEIGHT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLAYERED_MAX_LAYERS",        CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLAYERED_MAX_LAYERS )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT",                   CUDA_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS",                  CUDA_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS )
//	.value( "CUDA_DEVICE_ATTRIBUTE_ECC_ENABLED",                         CUDA_DEVICE_ATTRIBUTE_ECC_ENABLED )
//	.value( "CUDA_DEVICE_ATTRIBUTE_PCI_BUS_ID",                          CUDA_DEVICE_ATTRIBUTE_PCI_BUS_ID )
//	.value( "CUDA_DEVICE_ATTRIBUTE_PCI_DEVICE_ID",                       CUDA_DEVICE_ATTRIBUTE_PCI_DEVICE_ID )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TCC_DRIVER",                          CUDA_DEVICE_ATTRIBUTE_TCC_DRIVER )
//	.value( "CUDA_DEVICE_ATTRIBUTE_PEAK_MEMORY_FREQUENCY",               CUDA_DEVICE_ATTRIBUTE_PEAK_MEMORY_FREQUENCY )
//	.value( "CUDA_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH",             CUDA_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_L2_CACHE_SIZE",                       CUDA_DEVICE_ATTRIBUTE_L2_CACHE_SIZE )
//	.value( "CUDA_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR",      CUDA_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR )
//	.value( "CUDA_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT",                  CUDA_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING",                  CUDA_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_1DLAYERED_MAX_WIDTH",         CUDA_DEVICE_ATTRIBUTE_TEXTURE_1DLAYERED_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_1DLAYERED_MAX_LAYERS",        CUDA_DEVICE_ATTRIBUTE_TEXTURE_1DLAYERED_MAX_LAYERS )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DGATHER_MAX_WIDTH",          CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DGATHER_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DGATHER_MAX_HEIGHT",         CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DGATHER_MAX_HEIGHT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_3DALT_MAX_WIDTH",             CUDA_DEVICE_ATTRIBUTE_TEXTURE_3DALT_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_3DALT_MAX_HEIGHT",            CUDA_DEVICE_ATTRIBUTE_TEXTURE_3DALT_MAX_HEIGHT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_3DALT_MAX_DEPTH",             CUDA_DEVICE_ATTRIBUTE_TEXTURE_3DALT_MAX_DEPTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID",                       CUDA_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT",             CUDA_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_CUBE_MAX_WIDTH",              CUDA_DEVICE_ATTRIBUTE_TEXTURE_CUBE_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_CUBELAYERED_MAX_WIDTH",       CUDA_DEVICE_ATTRIBUTE_TEXTURE_CUBELAYERED_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_CUBELAYERED_MAX_LAYERS",      CUDA_DEVICE_ATTRIBUTE_TEXTURE_CUBELAYERED_MAX_LAYERS )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_1D_MAX_WIDTH",                CUDA_DEVICE_ATTRIBUTE_SURFACE_1D_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_2D_MAX_WIDTH",                CUDA_DEVICE_ATTRIBUTE_SURFACE_2D_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_2D_MAX_HEIGHT",               CUDA_DEVICE_ATTRIBUTE_SURFACE_2D_MAX_HEIGHT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_3D_MAX_WIDTH",                CUDA_DEVICE_ATTRIBUTE_SURFACE_3D_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_3D_MAX_HEIGHT",               CUDA_DEVICE_ATTRIBUTE_SURFACE_3D_MAX_HEIGHT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_3D_MAX_DEPTH",                CUDA_DEVICE_ATTRIBUTE_SURFACE_3D_MAX_DEPTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_1DLAYERED_MAX_WIDTH",         CUDA_DEVICE_ATTRIBUTE_SURFACE_1DLAYERED_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_1DLAYERED_MAX_LAYERS",        CUDA_DEVICE_ATTRIBUTE_SURFACE_1DLAYERED_MAX_LAYERS )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_2DLAYERED_MAX_WIDTH",         CUDA_DEVICE_ATTRIBUTE_SURFACE_2DLAYERED_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_2DLAYERED_MAX_HEIGHT",        CUDA_DEVICE_ATTRIBUTE_SURFACE_2DLAYERED_MAX_HEIGHT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_2DLAYERED_MAX_LAYERS",        CUDA_DEVICE_ATTRIBUTE_SURFACE_2DLAYERED_MAX_LAYERS )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_CUBE_MAX_WIDTH",              CUDA_DEVICE_ATTRIBUTE_SURFACE_CUBE_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_CUBELAYERED_MAX_WIDTH",       CUDA_DEVICE_ATTRIBUTE_SURFACE_CUBELAYERED_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SURFACE_CUBELAYERED_MAX_LAYERS",      CUDA_DEVICE_ATTRIBUTE_SURFACE_CUBELAYERED_MAX_LAYERS )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_1DLINEAR_MAX_WIDTH",          CUDA_DEVICE_ATTRIBUTE_TEXTURE_1DLINEAR_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLINEAR_MAX_WIDTH",          CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLINEAR_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLINEAR_MAX_HEIGHT",         CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLINEAR_MAX_HEIGHT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLINEAR_MAX_PITCH",          CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLINEAR_MAX_PITCH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DMIPMAP_MAX_WIDTH",          CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DMIPMAP_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DMIPMAP_MAX_HEIGHT",         CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DMIPMAP_MAX_HEIGHT )
//	.value( "CUDA_DEVICE_ATTRIBUTE_COMPUTE_VERSION_MAJOR",               CUDA_DEVICE_ATTRIBUTE_COMPUTE_VERSION_MAJOR )
//	.value( "CUDA_DEVICE_ATTRIBUTE_COMPUTE_VERSION_MINOR",               CUDA_DEVICE_ATTRIBUTE_COMPUTE_VERSION_MINOR )
//	.value( "CUDA_DEVICE_ATTRIBUTE_TEXTURE_1DMIPMAP_MAX_WIDTH",          CUDA_DEVICE_ATTRIBUTE_TEXTURE_1DMIPMAP_MAX_WIDTH )
//	.value( "CUDA_DEVICE_ATTRIBUTE_STREAM_PRIORITIES",                   CUDA_DEVICE_ATTRIBUTE_STREAM_PRIORITIES )
//	.value( "CUDA_DEVICE_ATTRIBUTE_L1_CACHE_GLOBALS",                    CUDA_DEVICE_ATTRIBUTE_L1_CACHE_GLOBALS )
//	.value( "CUDA_DEVICE_ATTRIBUTE_L1_CACHE_LOCALS",                     CUDA_DEVICE_ATTRIBUTE_L1_CACHE_LOCALS )
//	.value( "CUDA_DEVICE_ATTRIBUTE_MAX_SHAREDMEM_PER_MULTIPROCESSOR",    CUDA_DEVICE_ATTRIBUTE_MAX_SHAREDMEM_PER_MULTIPROCESSOR )
//	.value( "CUDA_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR",    CUDA_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR )
//	.value( "CUDA_DEVICE_ATTRIBUTE_MANAGED_MEMORY",                      CUDA_DEVICE_ATTRIBUTE_MANAGED_MEMORY )
//	.value( "CUDA_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD",                     CUDA_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD )
//	.value( "CUDA_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID",            CUDA_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID )
//	.value( "CUDA_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC",                  CUDA_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SINGLE_DOUBLE_PERF_RATIO",            CUDA_DEVICE_ATTRIBUTE_SINGLE_DOUBLE_PERF_RATIO )
//	.value( "CUDA_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS",              CUDA_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS )
//	.value( "CUDA_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_MEMORY_ACCESS",    CUDA_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_MEMORY_ACCESS )
//	.value( "CUDA_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION",                  CUDA_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION )
//	.value( "CUDA_DEVICE_ATTRIBUTE_HOST_REGISTERED_MEMORY",              CUDA_DEVICE_ATTRIBUTE_HOST_REGISTERED_MEMORY )
//	.value( "CUDA_DEVICE_ATTRIBUTE_RESERVED_92",                         CUDA_DEVICE_ATTRIBUTE_RESERVED_92 )
//	.value( "CUDA_DEVICE_ATTRIBUTE_RESERVED_93",                         CUDA_DEVICE_ATTRIBUTE_RESERVED_93 )
//	.value( "CUDA_DEVICE_ATTRIBUTE_RESERVED_94",                         CUDA_DEVICE_ATTRIBUTE_RESERVED_94 )
//	.value( "CUDA_DEVICE_ATTRIBUTE_COOPERATIVE_KERNELS",                 CUDA_DEVICE_ATTRIBUTE_COOPERATIVE_KERNELS )
//	.value( "CUDA_DEVICE_ATTRIBUTE_COOPERATIVE_KERNELS_MULTIDEVICES",    CUDA_DEVICE_ATTRIBUTE_COOPERATIVE_KERNELS_MULTIDEVICES )
//	.value( "CUDA_DEVICE_ATTRIBUTE_MAX_OPTIN_SHAREDMEM_PER_BLOCK",       CUDA_DEVICE_ATTRIBUTE_MAX_OPTIN_SHAREDMEM_PER_BLOCK )
//	.value( "CUDA_DEVICE_ATTRIBUTE_FLUSH_REMOTE_WRITES",                 CUDA_DEVICE_ATTRIBUTE_FLUSH_REMOTE_WRITES )
//	.value( "CUDA_DEVICE_ATTRIBUTE_HOST_MEMORY_REGISTRATION",            CUDA_DEVICE_ATTRIBUTE_HOST_MEMORY_REGISTRATION )
//	.value( "CUDA_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_HOST_PAGE_TABLES",    CUDA_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_HOST_PAGE_TABLES )
//	.value( "CUDA_DEVICE_ATTRIBUTE_MANAGED_MEMORY_HOST_DIRECT_ACCESS",   CUDA_DEVICE_ATTRIBUTE_MANAGED_MEMORY_HOST_DIRECT_ACCESS )
//	.value( "CUDA_DEVICE_ATTRIBUTE_RESERVED_102",                        CUDA_DEVICE_ATTRIBUTE_RESERVED_102 )
//	.value( "CUDA_DEVICE_ATTRIBUTE_RESERVED_103",                        CUDA_DEVICE_ATTRIBUTE_RESERVED_103 )
//	.value( "CUDA_DEVICE_ATTRIBUTE_RESERVED_104",                        CUDA_DEVICE_ATTRIBUTE_RESERVED_104 )
//	.value( "CUDA_DEVICE_ATTRIBUTE_RESERVED_105",                        CUDA_DEVICE_ATTRIBUTE_RESERVED_105 )
//	.value( "CUDA_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR",       CUDA_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR )
//	.value( "CUDA_DEVICE_ATTRIBUTE_RESERVED_107",                        CUDA_DEVICE_ATTRIBUTE_RESERVED_107 )
//	.value( "CUDA_DEVICE_ATTRIBUTE_RESERVED_108",                        CUDA_DEVICE_ATTRIBUTE_RESERVED_108 )
//	.value( "CUDA_DEVICE_ATTRIBUTE_RESERVED_109",                        CUDA_DEVICE_ATTRIBUTE_RESERVED_109 )
//	.value( "CUDA_DEVICE_ATTRIBUTE_RESERVED_110",                        CUDA_DEVICE_ATTRIBUTE_RESERVED_110 )
//	.value( "CUDA_DEVICE_ATTRIBUTE_RESERVED_SHAREDMEM_PER_BLOCK",        CUDA_DEVICE_ATTRIBUTE_RESERVED_SHAREDMEM_PER_BLOCK )
//	.value( "CUDA_DEVICE_ATTRIBUTE_SPARSE_ARRAYS",                       CUDA_DEVICE_ATTRIBUTE_SPARSE_ARRAYS )
//	.value( "CUDA_DEVICE_ATTRIBUTE_HOST_MEMORY_REGISTRATION_READONLY",   CUDA_DEVICE_ATTRIBUTE_HOST_MEMORY_REGISTRATION_READONLY )
//	.value( "CUDA_DEVICE_ATTRIBUTE_EXTERNAL_TIMELINE_SEMAPHORE_INTEROP", CUDA_DEVICE_ATTRIBUTE_EXTERNAL_TIMELINE_SEMAPHORE_INTEROP )
//	.value( "CUDA_DEVICE_ATTRIBUTE_ASYNC_AND_POOLED_MEMORY",             CUDA_DEVICE_ATTRIBUTE_ASYNC_AND_POOLED_MEMORY )
//    .value( "CUDA_DEVICE_ATTRIBUTE_COUNT",                               CUDA_DEVICE_ATTRIBUTE_COUNT )
//;
//
//py::enum_<CUDADeviceComputeMode>( hModule, "CUDADeviceComputeMode" )
//	.value( "CUDA_DEVICE_COMPUTE_MODE_DEFAULT",		   	  CUDA_DEVICE_COMPUTE_MODE_DEFAULT )
//	.value( "CUDA_DEVICE_COMPUTE_MODE_EXCLUSIVE",         CUDA_DEVICE_COMPUTE_MODE_EXCLUSIVE )
//	.value( "CUDA_DEVICE_COMPUTE_MODE_PROHIBITED",        CUDA_DEVICE_COMPUTE_MODE_PROHIBITED )
//	.value( "CUDA_DEVICE_COMPUTE_MODE_EXCLUSIVE_PROCESS", CUDA_DEVICE_COMPUTE_MODE_EXCLUSIVE_PROCESS )
//	.value( "CUDA_DEVICE_COMPUTE_MODE_COUNT",             CUDA_DEVICE_COMPUTE_MODE_COUNT )
//;
//
//py::enum_<CUDADeviceInitFlags>( hModule, "CUDADeviceInitFlags" )
//	.value( "CUDA_DEVICE_INITFLAG_SCHEDULE_AUTO",          CUDA_DEVICE_INITFLAG_SCHEDULE_AUTO )
//	.value( "CUDA_DEVICE_INITFLAG_SCHEDULE_SPIN_DEFAULT",  CUDA_DEVICE_INITFLAG_SCHEDULE_SPIN_DEFAULT )
//	.value( "CUDA_DEVICE_INITFLAG_SCHEDULE_YIELD_DEFAULT", CUDA_DEVICE_INITFLAG_SCHEDULE_YIELD_DEFAULT )
//	.value( "CUDA_DEVICE_INITFLAG_SCHEDULE_BLOCKING_SYNC", CUDA_DEVICE_INITFLAG_SCHEDULE_BLOCKING_SYNC )
//	.value( "CUDA_DEVICE_INITFLAG_MAPPED_PINNED_ALLOC",    CUDA_DEVICE_INITFLAG_MAPPED_PINNED_ALLOC )
//	.value( "CUDA_DEVICE_INITFLAG_KEEP_LOCAL_MEMORY",      CUDA_DEVICE_INITFLAG_KEEP_LOCAL_MEMORY )
//;
//
//py::enum_<CUDADeviceCacheConfig>( hModule, "CUDADeviceCacheConfig" )
//	.value( "CUDA_DEVICE_CACHE_CONFIG_PREFER_NONE",   CUDA_DEVICE_CACHE_CONFIG_PREFER_NONE )
//	.value( "CUDA_DEVICE_CACHE_CONFIG_PREFER_SHARED", CUDA_DEVICE_CACHE_CONFIG_PREFER_SHARED )
//	.value( "CUDA_DEVICE_CACHE_CONFIG_PREFER_L1",     CUDA_DEVICE_CACHE_CONFIG_PREFER_L1 )
//	.value( "CUDA_DEVICE_CACHE_CONFIG_PREFER_EQUAL",  CUDA_DEVICE_CACHE_CONFIG_PREFER_EQUAL )
//	.value( "CUDA_DEVICE_CACHE_CONFIG_COUNT",         CUDA_DEVICE_CACHE_CONFIG_COUNT )
//;
//
//py::enum_<CUDADeviceSharedMemoryConfig>( hModule, "CUDADeviceSharedMemoryConfig" )
//	.value( "CUDA_DEVICE_SHAREDMEM_CONFIG_DEFAULT", CUDA_DEVICE_SHAREDMEM_CONFIG_DEFAULT )
//	.value( "CUDA_DEVICE_SHAREDMEM_CONFIG_32BITS",  CUDA_DEVICE_SHAREDMEM_CONFIG_32BITS )
//	.value( "CUDA_DEVICE_SHAREDMEM_CONFIG_64BITS",  CUDA_DEVICE_SHAREDMEM_CONFIG_64BITS )
//	.value( "CUDA_DEVICE_SHAREDMEM_CONFIG_COUNT",   CUDA_DEVICE_SHAREDMEM_CONFIG_COUNT )
//;
//
//py::enum_<CUDAHostMemoryAllocFlags>( hModule, "CUDAHostMemoryAllocFlags" )
//	.value( "CUDA_HOSTMEMORY_ALLOC_FLAG_DEFAULT",        CUDA_HOSTMEMORY_ALLOC_FLAG_DEFAULT )
//	.value( "CUDA_HOSTMEMORY_ALLOC_FLAG_PINNED",         CUDA_HOSTMEMORY_ALLOC_FLAG_PINNED )
//	.value( "CUDA_HOSTMEMORY_ALLOC_FLAG_MAPPED",         CUDA_HOSTMEMORY_ALLOC_FLAG_MAPPED )
//	.value( "CUDA_HOSTMEMORY_ALLOC_FLAG_WRITE_COMBINED", CUDA_HOSTMEMORY_ALLOC_FLAG_WRITE_COMBINED )
//;
//
//py::enum_<CUDAHostMemoryWrapFlags>( hModule, "CUDAHostMemoryWrapFlags" )
//	.value( "CUDA_HOSTMEMORY_WRAP_FLAG_DEFAULT",  CUDA_HOSTMEMORY_WRAP_FLAG_DEFAULT )
//	.value( "CUDA_HOSTMEMORY_WRAP_FLAG_PINNED",   CUDA_HOSTMEMORY_WRAP_FLAG_PINNED )
//	.value( "CUDA_HOSTMEMORY_WRAP_FLAG_MAPPED",   CUDA_HOSTMEMORY_WRAP_FLAG_MAPPED )
//	.value( "CUDA_HOSTMEMORY_WRAP_FLAG_IO",       CUDA_HOSTMEMORY_WRAP_FLAG_IO )
//	.value( "CUDA_HOSTMEMORY_WRAP_FLAG_READONLY", CUDA_HOSTMEMORY_WRAP_FLAG_READONLY )
//;
//
//py::enum_<CUDAEventFlags>( hModule, "CUDAEventFlags" )
//	.value( "CUDA_EVENT_FLAG_DEFAULT",      CUDA_EVENT_FLAG_DEFAULT )
//	.value( "CUDA_EVENT_FLAG_BLOCKING",     CUDA_EVENT_FLAG_BLOCKING )
//	.value( "CUDA_EVENT_FLAG_NOTIMING",     CUDA_EVENT_FLAG_NOTIMING )
//	.value( "CUDA_EVENT_FLAG_INTERPROCESS", CUDA_EVENT_FLAG_INTERPROCESS )
//;
//
//py::enum_<CUDAStreamAttachMemoryFlags>( hModule, "CUDAStreamAttachMemoryFlags" )
//	.value( "CUDA_STREAM_ATTACH_MEMORY_FLAG_GLOBAL", CUDA_STREAM_ATTACH_MEMORY_FLAG_GLOBAL )
//	.value( "CUDA_STREAM_ATTACH_MEMORY_FLAG_HOST",   CUDA_STREAM_ATTACH_MEMORY_FLAG_HOST )
//	.value( "CUDA_STREAM_ATTACH_MEMORY_FLAG_SINGLE", CUDA_STREAM_ATTACH_MEMORY_FLAG_SINGLE )
//;
//
//py::enum_<CUDAStreamCaptureMode>( hModule, "CUDAStreamCaptureMode" )
//	.value( "CUDA_STREAM_CAPTURE_MODE_GLOBAL",  CUDA_STREAM_CAPTURE_MODE_GLOBAL )
//	.value( "CUDA_STREAM_CAPTURE_MODE_THREAD",  CUDA_STREAM_CAPTURE_MODE_THREAD )
//	.value( "CUDA_STREAM_CAPTURE_MODE_RELAXED", CUDA_STREAM_CAPTURE_MODE_RELAXED )
//	.value( "CUDA_STREAM_CAPTURE_MODE_COUNT",   CUDA_STREAM_CAPTURE_MODE_COUNT )
//;
//
//py::enum_<CUDAStreamCaptureStatus>( hModule, "CUDAStreamCaptureStatus" )
//	.value( "CUDA_STREAM_CAPTURE_STATUS_NONE",   CUDA_STREAM_CAPTURE_STATUS_NONE )
//	.value( "CUDA_STREAM_CAPTURE_STATUS_ACTIVE", CUDA_STREAM_CAPTURE_STATUS_ACTIVE )
//	.value( "CUDA_STREAM_CAPTURE_STATUS_ZOMBIE", CUDA_STREAM_CAPTURE_STATUS_ZOMBIE )
//	.value( "CUDA_STREAM_CAPTURE_STATUS_COUNT",  CUDA_STREAM_CAPTURE_STATUS_COUNT )
//;
//
//py::class_<CUDAReal32>( hModule, "CUDAReal32" );
//py::class_<CUDAReal64>( hModule, "CUDAReal64" );
//py::class_<CUDAComplex32>( hModule, "CUDAComplex32" )
//	.def( py::init<>() )
//	.def( py::init<CUDAReal32,CUDAReal32>() )
//	.def( "assign", &CUDAComplex32::operator=, py::is_operator() )
//;
//py::class_<CUDAComplex64>( hModule, "CUDAComplex64" )
//	.def( py::init<>() )
//	.def( py::init<CUDAReal64,CUDAReal64>() )
//	.def( "assign", &CUDAComplex64::operator=, py::is_operator() )
//;
//
//py::enum_<CUBLASContextPointerMode>( hModule, "CUBLASContextPointerMode" )
//	.value( "CUBLAS_CONTEXT_POINTER_MODE_HOST",   CUBLAS_CONTEXT_POINTER_MODE_HOST )
//	.value( "CUBLAS_CONTEXT_POINTER_MODE_DEVICE", CUBLAS_CONTEXT_POINTER_MODE_DEVICE )
//	.value( "CUBLAS_CONTEXT_POINTER_MODE_COUNT",  CUBLAS_CONTEXT_POINTER_MODE_COUNT )
//;
//
//py::enum_<CUBLASContextPrecisionMode>( hModule, "CUBLASContextPrecisionMode" )
//	.value( "CUBLAS_CONTEXT_PRECISION_MODE_DEFAULT", CUBLAS_CONTEXT_PRECISION_MODE_DEFAULT )
//	.value( "CUBLAS_CONTEXT_PRECISION_MODE_PRECISE", CUBLAS_CONTEXT_PRECISION_MODE_PRECISE )
//	.value( "CUBLAS_CONTEXT_PRECISION_MODE_TF32",    CUBLAS_CONTEXT_PRECISION_MODE_TF32 )
//;
//
//py::enum_<CUBLASContextLoggingMode>( hModule, "CUBLASContextLoggingMode" )
//	.value( "CUBLAS_CONTEXT_LOGGING_MODE_DISABLED", CUBLAS_CONTEXT_LOGGING_MODE_DISABLED )
//	.value( "CUBLAS_CONTEXT_LOGGING_MODE_STDOUT",   CUBLAS_CONTEXT_LOGGING_MODE_STDOUT )
//	.value( "CUBLAS_CONTEXT_LOGGING_MODE_STDERR",   CUBLAS_CONTEXT_LOGGING_MODE_STDERR )
//	.value( "CUBLAS_CONTEXT_LOGGING_MODE_BOTH",     CUBLAS_CONTEXT_LOGGING_MODE_BOTH )
//;
//
//py::enum_<CUBLASContextTransposeOp>( hModule, "CUBLASContextTransposeOp" )
//	.value( "CUBLAS_CONTEXT_TRANSOP_NONE",      CUBLAS_CONTEXT_TRANSOP_NONE )
//	.value( "CUBLAS_CONTEXT_TRANSOP_TRANSPOSE", CUBLAS_CONTEXT_TRANSOP_TRANSPOSE )
//	.value( "CUBLAS_CONTEXT_TRANSOP_DAGGER",    CUBLAS_CONTEXT_TRANSOP_DAGGER )
//	.value( "CUBLAS_CONTEXT_TRANSOP_COUNT",     CUBLAS_CONTEXT_TRANSOP_COUNT )
//;
//
//py::enum_<CUBLASContextFillMode>( hModule, "CUBLASContextFillMode" )
//	.value( "CUBLAS_CONTEXT_FILLMODE_LOWER", CUBLAS_CONTEXT_FILLMODE_LOWER )
//	.value( "CUBLAS_CONTEXT_FILLMODE_UPPER", CUBLAS_CONTEXT_FILLMODE_UPPER )
//	.value( "CUBLAS_CONTEXT_FILLMODE_FULL",  CUBLAS_CONTEXT_FILLMODE_FULL )
//	.value( "CUBLAS_CONTEXT_FILLMODE_COUNT", CUBLAS_CONTEXT_FILLMODE_COUNT )
//;
//
//py::enum_<CUBLASContextSideMode>( hModule, "CUBLASContextSideMode" )
//	.value( "CUBLAS_CONTEXT_SIDEMODE_LEFT",  CUBLAS_CONTEXT_SIDEMODE_LEFT )
//	.value( "CUBLAS_CONTEXT_SIDEMODE_RIGHT", CUBLAS_CONTEXT_SIDEMODE_RIGHT )
//	.value( "CUBLAS_CONTEXT_SIDEMODE_COUNT", CUBLAS_CONTEXT_SIDEMODE_COUNT )
//;

/////////////////////////////////////////////////////////////////////////////////
// The CUDAContext class
py::class_<CUDAContext, std::unique_ptr<CUDAContext, py::nodelete>>( hModule, "CUDAContext" )
	.def( py::init(&CUDAContext::GetInstance) )
	
	.def("GetDriverVersion", &CUDAContext::GetDriverVersion)
	.def("GetRuntimeVersion", &CUDAContext::GetRuntimeVersion)
	
	//.def("GetLimit", &CUDAContext::GetLimit)
	//.def("SetLimit", &CUDAContext::SetLimit)
;

/////////////////////////////////////////////////////////////////////////////////
// CUDAPython module end
}

/////////////////////////////////////////////////////////////////////////////////
// Entry Point
Int main()
{
	pybind11::scoped_interpreter guard{};

	auto sys = pybind11::module::import("sys");
	pybind11::print(sys.attr("path"));

	return 0;
}

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // CUDAPYTHON_CPP
