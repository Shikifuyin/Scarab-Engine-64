/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDAMappings.cpp
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
// Third-Party Includes
#include <cuda_runtime.h>
#include <cublas_v2.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUDAMappings.h"

/////////////////////////////////////////////////////////////////////////////////
// General Definitions
DWord _CUDAConvertFlags32( Byte * arrConvert, DWord iFlags )
{
    DWord iRes = 0, iLog2 = 0;
    while( iFlags != 0 ) {
        if ( iFlags & 1 )
            iRes |= ( 1 << arrConvert[iLog2] );
        iFlags >>= 1;
        ++iLog2;
    }
    return iRes;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDA Runtime Definitions
CUDALimit CUDALimitFromCUDA[CUDA_LIMIT_COUNT] = {
	CUDA_LIMIT_THREAD_STACK_SIZE,
	CUDA_LIMIT_PRINTF_FIFO_SIZE,
	CUDA_LIMIT_HEAP_SIZE,
	CUDA_LIMIT_RUNTIME_SYNC_DEPTH,
	CUDA_LIMIT_RUNTIME_PENDING_CALLS,
	CUDA_LIMIT_L2_MAX_FETCH_GRANULARITY,
	CUDA_LIMIT_L2_PERSISTENT_CACHE_LINE_SIZE
};
DWord CUDALimitToCUDA[CUDA_LIMIT_COUNT] = {
	cudaLimitStackSize,
	cudaLimitPrintfFifoSize,
	cudaLimitMallocHeapSize,
	cudaLimitDevRuntimeSyncDepth,
	cudaLimitDevRuntimePendingLaunchCount,
	cudaLimitMaxL2FetchGranularity,
	cudaLimitPersistingL2CacheSize
};

/////////////////////////////////////////////////////////////////////////////////
// CUDA Device Definitions
CUDADeviceAttribute CUDADeviceAttributeFromCUDA[CUDA_DEVICE_ATTRIBUTE_COUNT] = {
	(CUDADeviceAttribute)0, // INVALID
	CUDA_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
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
	CUDA_DEVICE_ATTRIBUTE_PEAK_CLOCK_FREQUENCY,
	CUDA_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
	CUDA_DEVICE_ATTRIBUTE_COPY_AND_EXECUTE,
	CUDA_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
	CUDA_DEVICE_ATTRIBUTE_KERNEL_TIMEOUT,
	CUDA_DEVICE_ATTRIBUTE_INTEGRATED,
	CUDA_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
	CUDA_DEVICE_ATTRIBUTE_COMPUTE_MODE,
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
	CUDA_DEVICE_ATTRIBUTE_PEAK_MEMORY_FREQUENCY,
	CUDA_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
	CUDA_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
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
	CUDA_DEVICE_ATTRIBUTE_SINGLE_DOUBLE_PERF_RATIO,
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
	CUDA_DEVICE_ATTRIBUTE_ASYNC_AND_POOLED_MEMORY
};
DWord CUDADeviceAttributeToCUDA[CUDA_DEVICE_ATTRIBUTE_COUNT] = {
	0, // INVALID
	cudaDevAttrMaxThreadsPerBlock,
	cudaDevAttrMaxBlockDimX,
	cudaDevAttrMaxBlockDimY,
	cudaDevAttrMaxBlockDimZ,
	cudaDevAttrMaxGridDimX,
	cudaDevAttrMaxGridDimY,
	cudaDevAttrMaxGridDimZ,
	cudaDevAttrMaxSharedMemoryPerBlock,
	cudaDevAttrTotalConstantMemory,
	cudaDevAttrWarpSize,
	cudaDevAttrMaxPitch,
	cudaDevAttrMaxRegistersPerBlock,
	cudaDevAttrClockRate,
	cudaDevAttrTextureAlignment,
	cudaDevAttrGpuOverlap,
	cudaDevAttrMultiProcessorCount,
	cudaDevAttrKernelExecTimeout,
	cudaDevAttrIntegrated,
	cudaDevAttrCanMapHostMemory,
	cudaDevAttrComputeMode,
	cudaDevAttrMaxTexture1DWidth,
	cudaDevAttrMaxTexture2DWidth,
	cudaDevAttrMaxTexture2DHeight,
	cudaDevAttrMaxTexture3DWidth,
	cudaDevAttrMaxTexture3DHeight,
	cudaDevAttrMaxTexture3DDepth,
	cudaDevAttrMaxTexture2DLayeredWidth,
	cudaDevAttrMaxTexture2DLayeredHeight,
	cudaDevAttrMaxTexture2DLayeredLayers,
	cudaDevAttrSurfaceAlignment,
	cudaDevAttrConcurrentKernels,
	cudaDevAttrEccEnabled,
	cudaDevAttrPciBusId,
	cudaDevAttrPciDeviceId,
	cudaDevAttrTccDriver,
	cudaDevAttrMemoryClockRate,
	cudaDevAttrGlobalMemoryBusWidth,
	cudaDevAttrL2CacheSize,
	cudaDevAttrMaxThreadsPerMultiProcessor,
	cudaDevAttrAsyncEngineCount,
	cudaDevAttrUnifiedAddressing,
	cudaDevAttrMaxTexture1DLayeredWidth,
	cudaDevAttrMaxTexture1DLayeredLayers,
	cudaDevAttrMaxTexture2DGatherWidth,
	cudaDevAttrMaxTexture2DGatherHeight,
	cudaDevAttrMaxTexture3DWidthAlt,
	cudaDevAttrMaxTexture3DHeightAlt,
	cudaDevAttrMaxTexture3DDepthAlt,
	cudaDevAttrPciDomainId,
	cudaDevAttrTexturePitchAlignment,
	cudaDevAttrMaxTextureCubemapWidth,
	cudaDevAttrMaxTextureCubemapLayeredWidth,
	cudaDevAttrMaxTextureCubemapLayeredLayers,
	cudaDevAttrMaxSurface1DWidth,
	cudaDevAttrMaxSurface2DWidth,
	cudaDevAttrMaxSurface2DHeight,
	cudaDevAttrMaxSurface3DWidth,
	cudaDevAttrMaxSurface3DHeight,
	cudaDevAttrMaxSurface3DDepth,
	cudaDevAttrMaxSurface1DLayeredWidth,
	cudaDevAttrMaxSurface1DLayeredLayers,
	cudaDevAttrMaxSurface2DLayeredWidth,
	cudaDevAttrMaxSurface2DLayeredHeight,
	cudaDevAttrMaxSurface2DLayeredLayers,
	cudaDevAttrMaxSurfaceCubemapWidth,
	cudaDevAttrMaxSurfaceCubemapLayeredWidth,
	cudaDevAttrMaxSurfaceCubemapLayeredLayers,
	cudaDevAttrMaxTexture1DLinearWidth,
	cudaDevAttrMaxTexture2DLinearWidth,
	cudaDevAttrMaxTexture2DLinearHeight,
	cudaDevAttrMaxTexture2DLinearPitch,
	cudaDevAttrMaxTexture2DMipmappedWidth,
	cudaDevAttrMaxTexture2DMipmappedHeight,
	cudaDevAttrComputeCapabilityMajor,
	cudaDevAttrComputeCapabilityMinor,
	cudaDevAttrMaxTexture1DMipmappedWidth,
	cudaDevAttrStreamPrioritiesSupported,
	cudaDevAttrGlobalL1CacheSupported,
	cudaDevAttrLocalL1CacheSupported,
	cudaDevAttrMaxSharedMemoryPerMultiprocessor,
	cudaDevAttrMaxRegistersPerMultiprocessor,
	cudaDevAttrManagedMemory,
	cudaDevAttrIsMultiGpuBoard,
	cudaDevAttrMultiGpuBoardGroupID,
	cudaDevAttrHostNativeAtomicSupported,
	cudaDevAttrSingleToDoublePrecisionPerfRatio,
	cudaDevAttrPageableMemoryAccess,
	cudaDevAttrConcurrentManagedAccess,
	cudaDevAttrComputePreemptionSupported,
	cudaDevAttrCanUseHostPointerForRegisteredMem,
	cudaDevAttrReserved92,
	cudaDevAttrReserved93,
	cudaDevAttrReserved94,
	cudaDevAttrCooperativeLaunch,
	cudaDevAttrCooperativeMultiDeviceLaunch,
	cudaDevAttrMaxSharedMemoryPerBlockOptin,
	cudaDevAttrCanFlushRemoteWrites,
	cudaDevAttrHostRegisterSupported,
	cudaDevAttrPageableMemoryAccessUsesHostPageTables,
	cudaDevAttrDirectManagedMemAccessFromHost,
	102,
	103,
	104,
	105,
	cudaDevAttrMaxBlocksPerMultiprocessor,
	107,
	108,
	109,
	110,
	cudaDevAttrReservedSharedMemoryPerBlock,
	cudaDevAttrSparseCudaArraySupported,
	cudaDevAttrHostRegisterReadOnlySupported,
	cudaDevAttrMaxTimelineSemaphoreInteropSupported,
	cudaDevAttrMemoryPoolsSupported
};

CUDADeviceComputeMode CUDADeviceComputeModeFromCUDA[CUDA_DEVICE_COMPUTE_MODE_COUNT] = {
	CUDA_DEVICE_COMPUTE_MODE_DEFAULT,
	CUDA_DEVICE_COMPUTE_MODE_EXCLUSIVE,
	CUDA_DEVICE_COMPUTE_MODE_PROHIBITED,
	CUDA_DEVICE_COMPUTE_MODE_EXCLUSIVE_PROCESS
};
DWord CUDADeviceComputeModeToCUDA[CUDA_DEVICE_COMPUTE_MODE_COUNT] = {
	cudaComputeModeDefault,
	cudaComputeModeExclusive,
	cudaComputeModeProhibited,
	cudaComputeModeExclusiveProcess
};

CUDADeviceCacheConfig CUDADeviceCacheConfigFromCUDA[CUDA_DEVICE_CACHE_CONFIG_COUNT] = {
	CUDA_DEVICE_CACHE_CONFIG_PREFER_NONE,
	CUDA_DEVICE_CACHE_CONFIG_PREFER_SHARED,
	CUDA_DEVICE_CACHE_CONFIG_PREFER_L1,
	CUDA_DEVICE_CACHE_CONFIG_PREFER_EQUAL
};
DWord CUDADeviceCacheConfigToCUDA[CUDA_DEVICE_CACHE_CONFIG_COUNT] = {
	cudaFuncCachePreferNone,
	cudaFuncCachePreferShared,
	cudaFuncCachePreferL1,
	cudaFuncCachePreferEqual
};

CUDADeviceSharedMemoryConfig CUDADeviceSharedMemoryConfigFromCUDA[CUDA_DEVICE_SHAREDMEM_CONFIG_COUNT] = {
	CUDA_DEVICE_SHAREDMEM_CONFIG_DEFAULT,
	CUDA_DEVICE_SHAREDMEM_CONFIG_32BITS,
	CUDA_DEVICE_SHAREDMEM_CONFIG_64BITS
};
DWord CUDADeviceSharedMemoryConfigToCUDA[CUDA_DEVICE_SHAREDMEM_CONFIG_COUNT] = {
	cudaSharedMemBankSizeDefault,
	cudaSharedMemBankSizeFourByte,
	cudaSharedMemBankSizeEightByte
};

/////////////////////////////////////////////////////////////////////////////////
// CUDA Memory Definitions

/////////////////////////////////////////////////////////////////////////////////
// CUDA Asynchronous Definitions
// CUDAStreamSyncPolicy CUDAStreamSyncPolicyFromCUDA[CUDA_STREAM_SYNC_POLICY_COUNT] = {
	// 0, // INVALID
	// CUDA_STREAM_SYNC_POLICY_AUTO,
	// CUDA_STREAM_SYNC_POLICY_SPIN,
	// CUDA_STREAM_SYNC_POLICY_YIELD,
	// CUDA_STREAM_SYNC_POLICY_BLOCKING
// };
// DWord CUDAStreamSyncPolicyToCUDA[CUDA_STREAM_SYNC_POLICY_COUNT] = {
	// 0, // INVALID
	// cudaSyncPolicyAuto,
    // cudaSyncPolicySpin,
    // cudaSyncPolicyYield,
    // cudaSyncPolicyBlockingSync
// };

CUDAStreamCaptureMode CUDAStreamCaptureModeFromCUDA[CUDA_STREAM_CAPTURE_MODE_COUNT] = {
	CUDA_STREAM_CAPTURE_MODE_GLOBAL,
	CUDA_STREAM_CAPTURE_MODE_THREAD,
	CUDA_STREAM_CAPTURE_MODE_RELAXED
};
DWord CUDAStreamCaptureModeToCUDA[CUDA_STREAM_CAPTURE_MODE_COUNT] = {
	cudaStreamCaptureModeGlobal,
	cudaStreamCaptureModeThreadLocal,
	cudaStreamCaptureModeRelaxed
};

CUDAStreamCaptureStatus CUDAStreamCaptureStatusFromCUDA[CUDA_STREAM_CAPTURE_STATUS_COUNT] = {
	CUDA_STREAM_CAPTURE_STATUS_NONE,
	CUDA_STREAM_CAPTURE_STATUS_ACTIVE,
	CUDA_STREAM_CAPTURE_STATUS_ZOMBIE
};
DWord CUDAStreamCaptureStatusToCUDA[CUDA_STREAM_CAPTURE_STATUS_COUNT] = {
	cudaStreamCaptureStatusNone,
	cudaStreamCaptureStatusActive,
	cudaStreamCaptureStatusInvalidated
};

// CUDAChannelFormatType CUDAChannelFormatTypeFromCUDA[CUDA_CHANNEL_FORMAT_TYPE_COUNT] = {
	// CUDA_CHANNEL_FORMAT_TYPE_SIGNED,
	// CUDA_CHANNEL_FORMAT_TYPE_UNSIGNED,
	// CUDA_CHANNEL_FORMAT_TYPE_FLOAT,
	// CUDA_CHANNEL_FORMAT_TYPE_NONE,
	// CUDA_CHANNEL_FORMAT_TYPE_NV12
// };
// DWord CUDAChannelFormatTypeToCUDA[CUDA_CHANNEL_FORMAT_TYPE_COUNT] = {
	// cudaChannelFormatKindSigned,
	// cudaChannelFormatKindUnsigned,
	// cudaChannelFormatKindFloat,
	// cudaChannelFormatKindNone,
	// cudaChannelFormatKindNV12
// };

// Void CUDAChannelFormat::ConvertFrom( const Void * pCUDADesc )
// {
	// const cudaChannelFormatDesc * pDesc = (const cudaChannelFormatDesc *)pCUDADesc;
	
	// iType = CUDAChannelFormatTypeFromCUDA[pDesc->f];
	// iW = pDesc->w;
	// iX = pDesc->x;
	// iY = pDesc->y;
	// iZ = pDesc->z;
// }
// Void CUDAChannelFormat::ConvertTo( Void * outCUDADesc ) const
// {
	// cudaChannelFormatDesc * outDesc = (cudaChannelFormatDesc*)outCUDADesc;
	
	// outDesc->f = (cudaChannelFormatKind)( CUDAChannelFormatTypeToCUDA[iType] );
	// outDesc->w = iW;
	// outDesc->x = iX;
	// outDesc->y = iY;
	// outDesc->z = iZ;
// }

// Void CUDAArraySparseProperties::ConvertFrom( const Void * pCUDADesc )
// {
	// const cudaArraySparseProperties * pDesc = (const cudaArraySparseProperties *)pCUDADesc;
	
	// iWidth = pDesc->width;
	// iHeight = pDesc->height;
	// iDepth = pDesc->depth;
	
	// bSingleMipTail = ( (pDesc->flags & cudaArraySparsePropertiesSingleMipTail) != 0 );
	// iMipTailFirstLevel = pDesc->miptailFirstLevel;
	// iMipTailSize = pDesc->miptailSize;
// }
// Void CUDAArraySparseProperties::ConvertTo( Void * outCUDADesc ) const
// {
	// cudaArraySparseProperties * outDesc = (cudaArraySparseProperties*)outCUDADesc;
	
	// outDesc->width = iWidth;
	// outDesc->height = iHeight;
	// outDesc->depth = iDepth;
	
	// outDesc->flags = 0;
	// if ( bSingleMipTail )
		// outDesc->flags |= cudaArraySparsePropertiesSingleMipTail;
	// outDesc->miptailFirstLevel = iMipTailFirstLevel;
	// outDesc->miptailSize = iMipTailSize;
// }

/////////////////////////////////////////////////////////////////////////////////
// CUBLAS Context Definitions
Void CUDAComplex32::_ConvertFromCUDA( const Void * pCUDAComplex )
{
	const cuComplex * pIn = (const cuComplex *)pCUDAComplex;
	fX = pIn->x;
	fY = pIn->y;
}
Void CUDAComplex32::_ConvertToCUDA( Void * pCUDAComplex ) const
{
	cuComplex * pOut = (cuComplex*)pCUDAComplex;
	pOut->x = fX;
	pOut->y = fY;
}

Void CUDAComplex64::_ConvertFromCUDA( const Void * pCUDAComplex )
{
	const cuDoubleComplex * pIn = (const cuDoubleComplex *)pCUDAComplex;
	fX = pIn->x;
	fY = pIn->y;
}
Void CUDAComplex64::_ConvertToCUDA( Void * pCUDAComplex ) const
{
	cuDoubleComplex * pOut = (cuDoubleComplex*)pCUDAComplex;
	pOut->x = fX;
	pOut->y = fY;
}

CUBLASContextPointerMode CUBLASContextPointerModeFromCUDA[CUBLAS_CONTEXT_POINTER_MODE_COUNT] = {
	CUBLAS_CONTEXT_POINTER_MODE_HOST,
	CUBLAS_CONTEXT_POINTER_MODE_DEVICE
};
DWord CUBLASContextPointerModeToCUDA[CUBLAS_CONTEXT_POINTER_MODE_COUNT] = {
	CUBLAS_POINTER_MODE_HOST,
	CUBLAS_POINTER_MODE_DEVICE
};

CUBLASContextTransposeOp CUBLASContextTransposeOpFromCUDA[CUBLAS_CONTEXT_TRANSOP_COUNT] = {
	CUBLAS_CONTEXT_TRANSOP_NONE,
	CUBLAS_CONTEXT_TRANSOP_TRANSPOSE,
	CUBLAS_CONTEXT_TRANSOP_DAGGER,
};
DWord CUBLASContextTransposeOpToCUDA[CUBLAS_CONTEXT_TRANSOP_COUNT] = {
	CUBLAS_OP_N,
	CUBLAS_OP_T,
	CUBLAS_OP_C
};

CUBLASContextFillMode CUBLASContextFillModeFromCUDA[CUBLAS_CONTEXT_FILLMODE_COUNT] = {
	CUBLAS_CONTEXT_FILLMODE_LOWER,
	CUBLAS_CONTEXT_FILLMODE_UPPER,
	CUBLAS_CONTEXT_FILLMODE_FULL
};
DWord CUBLASContextFillModeToCUDA[CUBLAS_CONTEXT_FILLMODE_COUNT] = {
	CUBLAS_FILL_MODE_LOWER,
	CUBLAS_FILL_MODE_UPPER,
	CUBLAS_FILL_MODE_FULL
};

CUBLASContextSideMode CUBLASContextSideModeFromCUDA[CUBLAS_CONTEXT_SIDEMODE_COUNT] = {
	CUBLAS_CONTEXT_SIDEMODE_LEFT,
	CUBLAS_CONTEXT_SIDEMODE_RIGHT
};
DWord CUBLASContextSideModeToCUDA[CUBLAS_CONTEXT_SIDEMODE_COUNT] = {
	CUBLAS_SIDE_LEFT,
	CUBLAS_SIDE_RIGHT
};

