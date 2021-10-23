/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDAContext.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA Context management
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <cuda_runtime.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUDAContext.h"

/////////////////////////////////////////////////////////////////////////////////
// CUDAContext implementation
CUDAContext::CUDAContext()
{
	// nothing to do
}
CUDAContext::~CUDAContext()
{
	// nothing to do
}

// CUDA Runtime /////////////////////////////////////////////////////////////////
Int CUDAContext::GetDriverVersion() const
{
	Int iDriverVersion = 0;
	
	cudaError_t iError = cudaDriverGetVersion( &iDriverVersion );
	DebugAssert( iError == cudaSuccess );
	
	return iDriverVersion;
}
Int CUDAContext::GetRuntimeVersion() const
{
	Int iRuntimeVersion = 0;
	
	cudaError_t iError = cudaRuntimeGetVersion( &iRuntimeVersion );
	DebugAssert( iError == cudaSuccess );
	
	return iRuntimeVersion;
}

SizeT CUDAContext::GetLimit( CUDALimit iLimit ) const
{
	cudaLimit iCUDALimit = (cudaLimit)( CUDALimitToCUDA[iLimit] );
	SizeT iLimitValue = 0;
	
	cudaError_t iError = cudaDeviceGetLimit( &iLimitValue, iCUDALimit );
	DebugAssert( iError == cudaSuccess );
	
	return iLimitValue;
}
Void CUDAContext::SetLimit( CUDALimit iLimit, SizeT iLimitValue ) const
{
	cudaLimit iCUDALimit = (cudaLimit)( CUDALimitToCUDA[iLimit] );
	
	cudaError_t iError = cudaDeviceSetLimit( iCUDALimit, iLimitValue );
	DebugAssert( iError == cudaSuccess );
}

// Error-Handling ///////////////////////////////////////////////////////////////
DWord CUDAContext::GetLastError() const
{
	return (DWord)( cudaGetLastError() );
}
DWord CUDAContext::PeekLastError() const
{
	return (DWord)( cudaPeekAtLastError() );
}

const Char * CUDAContext::GetErrorName( DWord dwErrorCode ) const
{
	return (const Char *)( cudaGetErrorName((cudaError_t)dwErrorCode) );
}
const Char * CUDAContext::GetErrorString( DWord dwErrorCode ) const
{
	return (const Char *)( cudaGetErrorString((cudaError_t)dwErrorCode) );
}

// Device-Management ////////////////////////////////////////////////////////////
Int CUDAContext::GetDeviceAttribute( CUDADeviceID iDeviceID, CUDADeviceAttribute iAttribute ) const
{
	DebugAssert( iAttribute < CUDA_DEVICE_ATTRIBUTE_COUNT );
	
	cudaDeviceAttr iCUDADeviceAttribute = (cudaDeviceAttr)( CUDADeviceAttributeToCUDA[iAttribute] );
	Int iAttributeValue = 0;
	
	cudaError_t iError = cudaDeviceGetAttribute( &iAttributeValue, iCUDADeviceAttribute, iDeviceID );
	DebugAssert( iError == cudaSuccess );
	
	return iAttributeValue;
}

const Char * CUDAContext::GetDevicePCIBusID( CUDADeviceID iDeviceID ) const
{
	static Char s_strPCIBusID[16];
	
	cudaError_t iError = cudaDeviceGetPCIBusId( (char*)s_strPCIBusID, 15, iDeviceID );
	DebugAssert( iError == cudaSuccess );
	
	return s_strPCIBusID;
}
CUDADeviceID CUDAContext::GetDeviceByPCIBusID( const Char * strPCIBusID ) const
{
	CUDADeviceID iDeviceID = 0;
	
	cudaError_t iError = cudaDeviceGetByPCIBusId( &iDeviceID, (const char *)strPCIBusID );
	DebugAssert( iError == cudaSuccess );
	
	return iDeviceID;
}

UInt CUDAContext::GetDeviceInitFlags() const
{
	UInt iInitFlags = 0;
	
	cudaError_t iError = cudaGetDeviceFlags( &iInitFlags );
	DebugAssert( iError == cudaSuccess );
	
	return iInitFlags;
}
Void CUDAContext::SetDeviceInitFlags( UInt iInitFlags ) const
{
	cudaError_t iError = cudaSetDeviceFlags( iInitFlags );
	DebugAssert( iError == cudaSuccess );
}

UInt CUDAContext::GetDeviceCount() const
{
	Int iDeviceCount = 0;
	
	cudaError_t iError = cudaGetDeviceCount( &iDeviceCount );
	DebugAssert( iError == cudaSuccess );
	
	return (UInt)iDeviceCount;
}
Void CUDAContext::SetValidDevices( CUDADeviceID * arrDeviceIDs, UInt iDeviceCount ) const
{
	cudaError_t iError = cudaSetValidDevices( arrDeviceIDs, iDeviceCount );
	DebugAssert( iError == cudaSuccess );
}

Void CUDAContext::SetCurrentDevice( CUDADeviceID iDeviceID ) const
{
	cudaError_t iError = cudaSetDevice( iDeviceID );
	DebugAssert( iError == cudaSuccess );
}
CUDADeviceID CUDAContext::GetCurrentDevice() const
{
	CUDADeviceID iDeviceID = 0;
	
	cudaError_t iError = cudaGetDevice( &iDeviceID );
	DebugAssert( iError == cudaSuccess );
	
	return iDeviceID;
}

CUDADeviceCacheConfig CUDAContext::GetCurrentDeviceCacheConfig() const
{
	cudaFuncCache iCUDAFunctionCache = cudaFuncCachePreferNone;
	
	cudaError_t iError = cudaDeviceGetCacheConfig( &iCUDAFunctionCache );
	DebugAssert( iError == cudaSuccess );
	
	return CUDADeviceCacheConfigFromCUDA[iCUDAFunctionCache];
}
Void CUDAContext::SetCurrentDeviceCacheConfig( CUDADeviceCacheConfig iCacheConfig ) const
{
	cudaFuncCache iCUDAFunctionCache = (cudaFuncCache)( CUDADeviceComputeModeToCUDA[iCacheConfig] );
	
	cudaError_t iError = cudaDeviceSetCacheConfig( iCUDAFunctionCache );
	DebugAssert( iError == cudaSuccess );
}

CUDADeviceSharedMemoryConfig CUDAContext::GetCurrentDeviceSharedMemoryConfig() const
{
	cudaSharedMemConfig iCUDASharedMemConfig = cudaSharedMemBankSizeDefault;
	
	cudaError_t iError = cudaDeviceGetSharedMemConfig( &iCUDASharedMemConfig );
	DebugAssert( iError == cudaSuccess );
	
	return CUDADeviceSharedMemoryConfigFromCUDA[iCUDASharedMemConfig];
}
Void CUDAContext::SetCurrentDeviceSharedMemoryConfig( CUDADeviceSharedMemoryConfig iSharedMemoryConfig ) const
{
	cudaSharedMemConfig iCUDASharedMemConfig = (cudaSharedMemConfig)( CUDADeviceSharedMemoryConfigToCUDA[iSharedMemoryConfig] );
	
	cudaError_t iError = cudaDeviceSetSharedMemConfig( iCUDASharedMemConfig );
	DebugAssert( iError == cudaSuccess );
}

Void CUDAContext::GetCurrentDeviceStreamPriorityRange( Int * outLowestPriority, Int * outHighestPriority ) const
{
	cudaError_t iError = cudaDeviceGetStreamPriorityRange( outLowestPriority, outHighestPriority );
	DebugAssert( iError == cudaSuccess );
}

Void CUDAContext::ResetCurrentDevice() const
{
	cudaError_t iError = cudaDeviceReset();
	DebugAssert( iError == cudaSuccess );
}
Void CUDAContext::WaitForCurrentDevice() const
{
	cudaError_t iError = cudaDeviceSynchronize();
	DebugAssert( iError == cudaSuccess );
}

Void CUDAContext::ResetPersistentL2Cache() const
{
	cudaError_t iError = cudaCtxResetPersistingL2Cache();
	DebugAssert( iError == cudaSuccess );
}













// CUDAMemoryPool CUDAContext::GetDeviceDefaultMemoryPool( CUDADeviceID iDeviceID ) const
// {
	// cudaMemPool_t hCUDAMemoryPool = NULL;
	
	// cudaError_t iError = cudaDeviceGetDefaultMemPool( &hCUDAMemoryPool, iDeviceID );
	// DebugAssert( iError == cudaSuccess );
	
	// return (CUDAMemoryPool)hCUDAMemoryPool;
// }
// CUDAMemoryPool CUDAContext::GetDeviceMemoryPool( CUDADeviceID iDeviceID ) const
// {
	// cudaMemPool_t hCUDAMemoryPool = NULL;
	
	// cudaError_t iError = cudaDeviceGetMemPool( &hCUDAMemoryPool, iDeviceID );
	// DebugAssert( iError == cudaSuccess );
	
	// return (CUDAMemoryPool)hCUDAMemoryPool;
// }
// Void CUDAContext::SetDeviceMemoryPool( CUDADeviceID iDeviceID, CUDAMemoryPool hMemoryPool ) const
// {
	// cudaMemPool_t hCUDAMemoryPool = (cudaMemPool_t)hMemoryPool;
	
	// cudaError_t iError = cudaDeviceSetMemPool( iDeviceID, hCUDAMemoryPool );
	// DebugAssert( iError == cudaSuccess );
// }

// Void CUDAContext::DeviceMemFree( CUDAArray hArray ) const
// {
	// cudaArray_t hCUDAArray = (cudaArray_t)hArray;
	
	// cudaError_t iError = cudaFreeArray( hCUDAArray );
	// DebugAssert( iError == cudaSuccess );
// }
// Void CUDAContext::DeviceMemFree( CUDAMipMapArray hMipMapArray ) const
// {
	// cudaMipmappedArray_t hCUDAMipMapArray = (cudaMipmappedArray_t)hMipMapArray;
	
	// cudaError_t iError = cudaFreeMipmappedArray( hCUDAMipMapArray );
	// DebugAssert( iError == cudaSuccess );
// }

// Void CUDAContext::GetArrayFormat( CUDAChannelFormat * outChannelFormat, CUDAArray hArray ) const
// {
	// cudaArray_t hCUDAArray = (cudaArray_t)hArray;
	// cudaChannelFormatDesc hCUDADesc;
	
	// cudaError_t iError = cudaArrayGetInfo( &hCUDADesc, NULL, NULL, hCUDAArray );
	// DebugAssert( iError == cudaSuccess );
	
	// outChannelFormat->ConvertFrom( &hCUDADesc );
// }
// Void CUDAContext::GetArrayExtent( CUDAExtent * outExtent, CUDAArray hArray ) const
// {
	// cudaArray_t hCUDAArray = (cudaArray_t)hArray;
	// cudaExtent hCUDADesc;
	
	// cudaError_t iError = cudaArrayGetInfo( NULL, &hCUDADesc, NULL, hCUDAArray );
	// DebugAssert( iError == cudaSuccess );
	
	// outExtent->ConvertFrom( &hCUDADesc );
// }
// Void CUDAContext::GetArrayFlags( UInt * outArrayFlags, CUDAArray hArray ) const
// {
	// cudaArray_t hCUDAArray = (cudaArray_t)hArray;

	// cudaError_t iError = cudaArrayGetInfo( NULL, NULL, outArrayFlags, hCUDAArray );
	// DebugAssert( iError == cudaSuccess );
// }

// CUDAArray CUDAContext::GetArrayPlane( CUDAArray hArray, UInt iPlane ) const
// {
	// cudaArray_t hCUDAArray = (cudaArray_t)hArray;
	// cudaArray_t hCUDAArrayPlane = NULL;
	
	// cudaError_t iError = cudaArrayGetPlane( &hCUDAArrayPlane, hCUDAArray, iPlane );
	// DebugAssert( iError == cudaSuccess );
	
	// return (CUDAArray)hCUDAArrayPlane;
// }
// Void CUDAContext::GetArraySparseProperties( CUDAArraySparseProperties * outSparseProperties, CUDAArray hArray ) const
// {
	// cudaArray_t hCUDAArray = (cudaArray_t)hArray;
	// cudaArraySparseProperties hCUDAArraySparseProperties;
	
	// cudaError_t iError = cudaArrayGetSparseProperties( &hCUDAArraySparseProperties, hCUDAArray );
	// DebugAssert( iError == cudaSuccess );
	
	// outSparseProperties->ConvertFrom( &hCUDAArraySparseProperties );
// }

// CUDAArray CUDAContext::GetMipMapArrayLevel( CUDAMipMapArray hMipMapArray, UInt iLevel ) const
// {
	// cudaMipmappedArray_t hCUDAMipMapArray = (cudaMipmappedArray_t)hMipMapArray;
	// cudaArray_t hCUDAMipMapArrayLevel = NULL;
	
	// cudaError_t iError = cudaGetMipmappedArrayLevel( &hCUDAMipMapArrayLevel, hCUDAMipMapArray, iLevel );
	// DebugAssert( iError == cudaSuccess );
	
	// return (CUDAArray)hCUDAMipMapArrayLevel;
// }






