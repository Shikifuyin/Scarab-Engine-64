/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDAMappings.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : API-dependant mappings for CUDA
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_CUDA_CUDAMAPPINGS_H
#define SCARAB_THIRDPARTY_CUDA_CUDAMAPPINGS_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../System/Platform.h"

/////////////////////////////////////////////////////////////////////////////////
// General Declarations
DWord _CUDAConvertFlags32( Byte * arrConvert, DWord iFlags );

/////////////////////////////////////////////////////////////////////////////////
// CUDA Runtime Declarations
enum CUDALimit {
	CUDA_LIMIT_THREAD_STACK_SIZE = 0,
	CUDA_LIMIT_PRINTF_FIFO_SIZE,
	CUDA_LIMIT_HEAP_SIZE,
	CUDA_LIMIT_RUNTIME_SYNC_DEPTH,
	CUDA_LIMIT_RUNTIME_PENDING_CALLS,
	CUDA_LIMIT_L2_MAX_FETCH_GRANULARITY,
	CUDA_LIMIT_L2_PERSISTENT_CACHE_LINE_SIZE,
	CUDA_LIMIT_COUNT
};
extern CUDALimit CUDALimitFromCUDA[CUDA_LIMIT_COUNT];
extern DWord CUDALimitToCUDA[CUDA_LIMIT_COUNT];

/////////////////////////////////////////////////////////////////////////////////
// CUDA Device Declarations
enum CUDADeviceAttribute {
	CUDA_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
	CUDA_DEVICE_ATTRIBUTE_BLOCK_MAX_DIM_X,
	CUDA_DEVICE_ATTRIBUTE_BLOCK_MAX_DIM_Y,
	CUDA_DEVICE_ATTRIBUTE_BLOCK_MAX_DIM_Z,
	CUDA_DEVICE_ATTRIBUTE_GRID_MAX_DIM_X,
	CUDA_DEVICE_ATTRIBUTE_GRID_MAX_DIM_Y,
	CUDA_DEVICE_ATTRIBUTE_GRID_MAX_DIM_Z,
	CUDA_DEVICE_ATTRIBUTE_MAX_SHAREDMEM_PER_BLOCK,
	CUDA_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
	CUDA_DEVICE_ATTRIBUTE_WARP_THREAD_COUNT,
	CUDA_DEVICE_ATTRIBUTE_MAX_PITCH,
	CUDA_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
	CUDA_DEVICE_ATTRIBUTE_PEAK_CLOCK_FREQUENCY, // in kHz
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
	CUDA_DEVICE_ATTRIBUTE_COPY_AND_EXECUTE,
	CUDA_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
	CUDA_DEVICE_ATTRIBUTE_KERNEL_TIMEOUT,
	CUDA_DEVICE_ATTRIBUTE_INTEGRATED,
	CUDA_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
	CUDA_DEVICE_ATTRIBUTE_COMPUTE_MODE, // returns a CUDADeviceComputeMode value
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_1D_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_2D_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_2D_MAX_HEIGHT,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_3D_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_3D_MAX_HEIGHT,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_3D_MAX_DEPTH,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLAYERED_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLAYERED_MAX_HEIGHT,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLAYERED_MAX_LAYERS,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
	CUDA_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
	CUDA_DEVICE_ATTRIBUTE_ECC_ENABLED,
	CUDA_DEVICE_ATTRIBUTE_PCI_BUS_ID,
	CUDA_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
	CUDA_DEVICE_ATTRIBUTE_TCC_DRIVER,
	CUDA_DEVICE_ATTRIBUTE_PEAK_MEMORY_FREQUENCY, // in kHz
	CUDA_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, // in bits
	CUDA_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, // in bytes
	CUDA_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
	CUDA_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
	CUDA_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_1DLAYERED_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_1DLAYERED_MAX_LAYERS,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DGATHER_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DGATHER_MAX_HEIGHT,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_3DALT_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_3DALT_MAX_HEIGHT,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_3DALT_MAX_DEPTH,
	CUDA_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_CUBE_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_CUBELAYERED_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_CUBELAYERED_MAX_LAYERS,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_1D_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_2D_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_2D_MAX_HEIGHT,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_3D_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_3D_MAX_HEIGHT,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_3D_MAX_DEPTH,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_1DLAYERED_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_1DLAYERED_MAX_LAYERS,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_2DLAYERED_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_2DLAYERED_MAX_HEIGHT,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_2DLAYERED_MAX_LAYERS,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_CUBE_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_CUBELAYERED_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_SURFACE_CUBELAYERED_MAX_LAYERS,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_1DLINEAR_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLINEAR_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLINEAR_MAX_HEIGHT,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DLINEAR_MAX_PITCH,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DMIPMAP_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_2DMIPMAP_MAX_HEIGHT,
	CUDA_DEVICE_ATTRIBUTE_COMPUTE_VERSION_MAJOR,
	CUDA_DEVICE_ATTRIBUTE_COMPUTE_VERSION_MINOR,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_1DMIPMAP_MAX_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_STREAM_PRIORITIES,
	CUDA_DEVICE_ATTRIBUTE_L1_CACHE_GLOBALS,
	CUDA_DEVICE_ATTRIBUTE_L1_CACHE_LOCALS,
	CUDA_DEVICE_ATTRIBUTE_MAX_SHAREDMEM_PER_MULTIPROCESSOR,
	CUDA_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
	CUDA_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
	CUDA_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,
	CUDA_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
	CUDA_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC,
	CUDA_DEVICE_ATTRIBUTE_SINGLE_DOUBLE_PERF_RATIO, // in FLOP/s
	CUDA_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
	CUDA_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_MEMORY_ACCESS,
	CUDA_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION,
	CUDA_DEVICE_ATTRIBUTE_HOST_REGISTERED_MEMORY,
	CUDA_DEVICE_ATTRIBUTE_RESERVED_92,
	CUDA_DEVICE_ATTRIBUTE_RESERVED_93,
	CUDA_DEVICE_ATTRIBUTE_RESERVED_94,
	CUDA_DEVICE_ATTRIBUTE_COOPERATIVE_KERNELS,
	CUDA_DEVICE_ATTRIBUTE_COOPERATIVE_KERNELS_MULTIDEVICES,
	CUDA_DEVICE_ATTRIBUTE_MAX_OPTIN_SHAREDMEM_PER_BLOCK,
	CUDA_DEVICE_ATTRIBUTE_FLUSH_REMOTE_WRITES,
	CUDA_DEVICE_ATTRIBUTE_HOST_MEMORY_REGISTRATION,
	CUDA_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_HOST_PAGE_TABLES,
	CUDA_DEVICE_ATTRIBUTE_MANAGED_MEMORY_HOST_DIRECT_ACCESS,
	CUDA_DEVICE_ATTRIBUTE_RESERVED_102,
	CUDA_DEVICE_ATTRIBUTE_RESERVED_103,
	CUDA_DEVICE_ATTRIBUTE_RESERVED_104,
	CUDA_DEVICE_ATTRIBUTE_RESERVED_105,
	CUDA_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
	CUDA_DEVICE_ATTRIBUTE_RESERVED_107,
	CUDA_DEVICE_ATTRIBUTE_RESERVED_108,
	CUDA_DEVICE_ATTRIBUTE_RESERVED_109,
	CUDA_DEVICE_ATTRIBUTE_RESERVED_110,
	CUDA_DEVICE_ATTRIBUTE_RESERVED_SHAREDMEM_PER_BLOCK,
	CUDA_DEVICE_ATTRIBUTE_SPARSE_ARRAYS,
	CUDA_DEVICE_ATTRIBUTE_HOST_MEMORY_REGISTRATION_READONLY,
	CUDA_DEVICE_ATTRIBUTE_EXTERNAL_TIMELINE_SEMAPHORE_INTEROP,
	CUDA_DEVICE_ATTRIBUTE_ASYNC_AND_POOLED_MEMORY,
    CUDA_DEVICE_ATTRIBUTE_COUNT
};
extern CUDADeviceAttribute CUDADeviceAttributeFromCUDA[CUDA_DEVICE_ATTRIBUTE_COUNT];
extern DWord CUDADeviceAttributeToCUDA[CUDA_DEVICE_ATTRIBUTE_COUNT];

enum CUDADeviceComputeMode {
	CUDA_DEVICE_COMPUTE_MODE_DEFAULT = 0,
	CUDA_DEVICE_COMPUTE_MODE_EXCLUSIVE,
	CUDA_DEVICE_COMPUTE_MODE_PROHIBITED,
	CUDA_DEVICE_COMPUTE_MODE_EXCLUSIVE_PROCESS,
	CUDA_DEVICE_COMPUTE_MODE_COUNT
};
extern CUDADeviceComputeMode CUDADeviceComputeModeFromCUDA[CUDA_DEVICE_COMPUTE_MODE_COUNT];
extern DWord CUDADeviceComputeModeToCUDA[CUDA_DEVICE_COMPUTE_MODE_COUNT];

enum CUDADeviceInitFlags {
	CUDA_DEVICE_INITFLAG_SCHEDULE_AUTO          = 0x00, // cudaDeviceScheduleAuto
	CUDA_DEVICE_INITFLAG_SCHEDULE_SPIN_DEFAULT  = 0x01, // cudaDeviceScheduleSpin
	CUDA_DEVICE_INITFLAG_SCHEDULE_YIELD_DEFAULT = 0x02, // cudaDeviceScheduleYield
	CUDA_DEVICE_INITFLAG_SCHEDULE_BLOCKING_SYNC = 0x04, // cudaDeviceScheduleBlockingSync
	CUDA_DEVICE_INITFLAG_MAPPED_PINNED_ALLOC    = 0x08, // cudaDeviceMapHost
	CUDA_DEVICE_INITFLAG_KEEP_LOCAL_MEMORY      = 0x10  // cudaDeviceLmemResizeToMax
};

enum CUDADeviceCacheConfig {
	CUDA_DEVICE_CACHE_CONFIG_PREFER_NONE = 0,
	CUDA_DEVICE_CACHE_CONFIG_PREFER_SHARED,
	CUDA_DEVICE_CACHE_CONFIG_PREFER_L1,
	CUDA_DEVICE_CACHE_CONFIG_PREFER_EQUAL,
	CUDA_DEVICE_CACHE_CONFIG_COUNT
};
extern CUDADeviceCacheConfig CUDADeviceCacheConfigFromCUDA[CUDA_DEVICE_CACHE_CONFIG_COUNT];
extern DWord CUDADeviceCacheConfigToCUDA[CUDA_DEVICE_CACHE_CONFIG_COUNT];

enum CUDADeviceSharedMemoryConfig {
	CUDA_DEVICE_SHAREDMEM_CONFIG_DEFAULT = 0,
	CUDA_DEVICE_SHAREDMEM_CONFIG_32BITS,
	CUDA_DEVICE_SHAREDMEM_CONFIG_64BITS,
	CUDA_DEVICE_SHAREDMEM_CONFIG_COUNT
};
extern CUDADeviceSharedMemoryConfig CUDADeviceSharedMemoryConfigFromCUDA[CUDA_DEVICE_SHAREDMEM_CONFIG_COUNT];
extern DWord CUDADeviceSharedMemoryConfigToCUDA[CUDA_DEVICE_SHAREDMEM_CONFIG_COUNT];

/////////////////////////////////////////////////////////////////////////////////
// CUDA Memory Declarations
enum CUDAHostMemoryAllocFlags {
	CUDA_HOSTMEMORY_ALLOC_FLAG_DEFAULT		  = 0x00, // cudaHostAllocDefault
	CUDA_HOSTMEMORY_ALLOC_FLAG_PINNED		  = 0x01, // cudaHostAllocPortable
	CUDA_HOSTMEMORY_ALLOC_FLAG_MAPPED		  = 0x02, // cudaHostAllocMapped
	CUDA_HOSTMEMORY_ALLOC_FLAG_WRITE_COMBINED = 0x04  // cudaHostAllocWriteCombined
};
enum CUDAHostMemoryWrapFlags {
	CUDA_HOSTMEMORY_WRAP_FLAG_DEFAULT  = 0x00, // cudaHostRegisterDefault
	CUDA_HOSTMEMORY_WRAP_FLAG_PINNED   = 0x01, // cudaHostRegisterPortable
	CUDA_HOSTMEMORY_WRAP_FLAG_MAPPED   = 0x02, // cudaHostRegisterMapped
	CUDA_HOSTMEMORY_WRAP_FLAG_IO	   = 0x04, // cudaHostRegisterIoMemory
	CUDA_HOSTMEMORY_WRAP_FLAG_READONLY = 0x08  // cudaHostRegisterReadOnly
};

/////////////////////////////////////////////////////////////////////////////////
// CUDA Asynchronous Declarations
enum CUDAEventFlags {
	CUDA_EVENT_FLAG_DEFAULT 	 = 0x00, // cudaEventDefault
	CUDA_EVENT_FLAG_BLOCKING 	 = 0x01, // cudaEventBlockingSync
	CUDA_EVENT_FLAG_NOTIMING	 = 0x02, // cudaEventDisableTiming
	CUDA_EVENT_FLAG_INTERPROCESS = 0x04  // cudaEventInterprocess
};

// enum CUDAStreamSyncPolicy {
	// CUDA_STREAM_SYNC_POLICY_AUTO = 1,
	// CUDA_STREAM_SYNC_POLICY_SPIN,
	// CUDA_STREAM_SYNC_POLICY_YIELD,
	// CUDA_STREAM_SYNC_POLICY_BLOCKING,
	// CUDA_STREAM_SYNC_POLICY_COUNT
// };
// extern CUDAStreamSyncPolicy CUDAStreamSyncPolicyFromCUDA[CUDA_STREAM_SYNC_POLICY_COUNT];
// extern DWord CUDAStreamSyncPolicyToCUDA[CUDA_STREAM_SYNC_POLICY_COUNT];

enum CUDAStreamAttachMemoryFlags {
	CUDA_STREAM_ATTACH_MEMORY_FLAG_GLOBAL = 0x01, // cudaMemAttachGlobal
	CUDA_STREAM_ATTACH_MEMORY_FLAG_HOST	 = 0x02, // cudaMemAttachHost
	CUDA_STREAM_ATTACH_MEMORY_FLAG_SINGLE = 0x04  // cudaMemAttachSingle
};

enum CUDAStreamCaptureMode {
	CUDA_STREAM_CAPTURE_MODE_GLOBAL = 0,
	CUDA_STREAM_CAPTURE_MODE_THREAD,
	CUDA_STREAM_CAPTURE_MODE_RELAXED,
	CUDA_STREAM_CAPTURE_MODE_COUNT
};
extern CUDAStreamCaptureMode CUDAStreamCaptureModeFromCUDA[CUDA_STREAM_CAPTURE_MODE_COUNT];
extern DWord CUDAStreamCaptureModeToCUDA[CUDA_STREAM_CAPTURE_MODE_COUNT];

enum CUDAStreamCaptureStatus {
	CUDA_STREAM_CAPTURE_STATUS_NONE = 0,
	CUDA_STREAM_CAPTURE_STATUS_ACTIVE,
	CUDA_STREAM_CAPTURE_STATUS_ZOMBIE, // Invalidated but not terminated
	CUDA_STREAM_CAPTURE_STATUS_COUNT
};
extern CUDAStreamCaptureStatus CUDAStreamCaptureStatusFromCUDA[CUDA_STREAM_CAPTURE_STATUS_COUNT];
extern DWord CUDAStreamCaptureStatusToCUDA[CUDA_STREAM_CAPTURE_STATUS_COUNT];

// enum CUDAChannelFormatType {
	// CUDA_CHANNEL_FORMAT_TYPE_SIGNED = 0,
	// CUDA_CHANNEL_FORMAT_TYPE_UNSIGNED,
	// CUDA_CHANNEL_FORMAT_TYPE_FLOAT,
	// CUDA_CHANNEL_FORMAT_TYPE_NONE,
	// CUDA_CHANNEL_FORMAT_TYPE_NV12, // 8-bits integer, planar 4:2:0 YUV
	// CUDA_CHANNEL_FORMAT_TYPE_COUNT
// };
// extern CUDAChannelFormatType CUDAChannelFormatTypeFromCUDA[CUDA_CHANNEL_FORMAT_TYPE_COUNT];
// extern DWord CUDAChannelFormatTypeToCUDA[CUDA_CHANNEL_FORMAT_TYPE_COUNT];

// typedef struct _cuda_channel_format {
	// Void ConvertFrom( const Void * pCUDADesc );
    // Void ConvertTo( Void * outCUDADesc ) const;
	
	// CUDAChannelFormatType iType;
	// Int iW;
	// Int iX;
	// Int iY;
	// Int iZ;
// } CUDAChannelFormat;

// enum CUDAArrayFlags {
	// CUDA_ARRAY_FLAG_DEFAULT		   = 0x00, // cudaArrayDefault
	// CUDA_ARRAY_FLAG_LAYERED 	   = 0x01, // cudaArrayLayered
	// CUDA_ARRAY_FLAG_SURFACE_BIND   = 0x02, // cudaArraySurfaceLoadStore
	// CUDA_ARRAY_FLAG_CUBE_MAP	   = 0x04, // cudaArrayCubemap
	// CUDA_ARRAY_FLAG_TEXTURE_GATHER = 0x08, // cudaArrayTextureGather
	// CUDA_ARRAY_FLAG_SPARSE		   = 0x40  // cudaArraySparse 
// };

// typedef struct _cuda_array_sparse_properties {
	// Void ConvertFrom( const Void * pCUDADesc );
    // Void ConvertTo( Void * outCUDADesc ) const;
	
	// UInt iWidth;
	// UInt iHeight;
	// UInt iDepth;
	// Bool bSingleMipTail;
	// UInt iMipTailFirstLevel;
	// SizeT iMipTailSize;
// } CUDAArraySparseProperties;

/////////////////////////////////////////////////////////////////////////////////
// CUBLAS Context Declarations
typedef Float CUDAReal32;
typedef Double CUDAReal64;
struct __declspec(align(8)) CUDAComplex32 {
	CUDAComplex32()
	{
		fX = 0.0f;
		fY = 0.0f;
	}
	CUDAComplex32( CUDAReal32 X, CUDAReal32 Y )
	{
		fX = X;
		fY = Y;
	}
	~CUDAComplex32() {}

	inline CUDAComplex32 & operator=( const CUDAComplex32 & rhs ) {
		fX = rhs.fX;
		fY = rhs.fY;
		return (*this);
	}

	Void _ConvertFromCUDA( const Void * pCUDAComplex );
	Void _ConvertToCUDA( Void * pCUDAComplex ) const;

	CUDAReal32 fX;
	CUDAReal32 fY;
};
struct __declspec(align(16)) CUDAComplex64 {
	CUDAComplex64()
	{
		fX = 0.0;
		fY = 0.0;
	}
	CUDAComplex64( CUDAReal64 X, CUDAReal64 Y )
	{
		fX = X;
		fY = Y;
	}
	~CUDAComplex64() {}

	inline CUDAComplex64 & operator=( const CUDAComplex64 & rhs ) {
		fX = rhs.fX;
		fY = rhs.fY;
		return (*this);
	}

	Void _ConvertFromCUDA( const Void * pCUDAComplex );
	Void _ConvertToCUDA( Void * pCUDAComplex ) const;

	CUDAReal64 fX;
	CUDAReal64 fY;
};

enum CUBLASContextPointerMode {
	CUBLAS_CONTEXT_POINTER_MODE_HOST = 0,
	CUBLAS_CONTEXT_POINTER_MODE_DEVICE,
	CUBLAS_CONTEXT_POINTER_MODE_COUNT
};
extern CUBLASContextPointerMode CUBLASContextPointerModeFromCUDA[CUBLAS_CONTEXT_POINTER_MODE_COUNT];
extern DWord CUBLASContextPointerModeToCUDA[CUBLAS_CONTEXT_POINTER_MODE_COUNT];

enum CUBLASContextPrecisionMode {
	CUBLAS_CONTEXT_PRECISION_MODE_DEFAULT = 0,
	CUBLAS_CONTEXT_PRECISION_MODE_PRECISE,
	CUBLAS_CONTEXT_PRECISION_MODE_TF32
};

enum CUBLASContextLoggingMode {
	CUBLAS_CONTEXT_LOGGING_MODE_DISABLED = 0,
	CUBLAS_CONTEXT_LOGGING_MODE_STDOUT,
	CUBLAS_CONTEXT_LOGGING_MODE_STDERR,
	CUBLAS_CONTEXT_LOGGING_MODE_BOTH
};

enum CUBLASContextTransposeOp {
	CUBLAS_CONTEXT_TRANSOP_NONE = 0,
	CUBLAS_CONTEXT_TRANSOP_TRANSPOSE,
	CUBLAS_CONTEXT_TRANSOP_DAGGER,
	CUBLAS_CONTEXT_TRANSOP_COUNT
};
extern CUBLASContextTransposeOp CUBLASContextTransposeOpFromCUDA[CUBLAS_CONTEXT_TRANSOP_COUNT];
extern DWord CUBLASContextTransposeOpToCUDA[CUBLAS_CONTEXT_TRANSOP_COUNT];

enum CUBLASContextFillMode {
	CUBLAS_CONTEXT_FILLMODE_LOWER = 0,
	CUBLAS_CONTEXT_FILLMODE_UPPER,
	CUBLAS_CONTEXT_FILLMODE_FULL,
	CUBLAS_CONTEXT_FILLMODE_COUNT
};
extern CUBLASContextFillMode CUBLASContextFillModeFromCUDA[CUBLAS_CONTEXT_FILLMODE_COUNT];
extern DWord CUBLASContextFillModeToCUDA[CUBLAS_CONTEXT_FILLMODE_COUNT];

enum CUBLASContextSideMode {
	CUBLAS_CONTEXT_SIDEMODE_LEFT = 0,
	CUBLAS_CONTEXT_SIDEMODE_RIGHT,
	CUBLAS_CONTEXT_SIDEMODE_COUNT
};
extern CUBLASContextSideMode CUBLASContextSideModeFromCUDA[CUBLAS_CONTEXT_SIDEMODE_COUNT];
extern DWord CUBLASContextSideModeToCUDA[CUBLAS_CONTEXT_SIDEMODE_COUNT];

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUDAMappings.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_CUDAMAPPINGS_H

