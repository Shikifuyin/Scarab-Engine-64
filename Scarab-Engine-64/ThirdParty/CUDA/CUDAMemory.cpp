/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDAMemory.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA Memory Containers
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
#include "CUDAMemory.h"

/////////////////////////////////////////////////////////////////////////////////
// Compiler directives
#pragma warning(disable:4244) // Conversion to smaller type

/////////////////////////////////////////////////////////////////////////////////
// CUDAMemory implementation
CUDAMemory::CUDAMemory()
{
	m_bHasOwnerShip = false;
	m_pMemory = NULL;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;

	m_iStride = 0;
	m_iPitch = 0;
	m_iSlice = 0;
	m_iSize = 0;
}
CUDAMemory::~CUDAMemory()
{
	// nothing to do
}

Void * CUDAMemory::GetSymbolAddress( const Void * pSymbol )
{
	Void * pSymbolAddress = NULL;
	
	cudaError_t iError = cudaGetSymbolAddress( &pSymbolAddress, pSymbol );
	DebugAssert( iError == cudaSuccess );
	
	return pSymbolAddress;
}
SizeT CUDAMemory::GetSymbolSize( const Void * pSymbol )
{
	SizeT iSymbolSize = 0;
	
	cudaError_t iError = cudaGetSymbolSize( &iSymbolSize, pSymbol );
	DebugAssert( iError == cudaSuccess );
	
	return iSymbolSize;
}

CUDADeviceID CUDAMemory::GetDeviceID() const
{
	DebugAssert( IsAllocated() );
	
	cudaPointerAttributes hCUDAPtrAttr;
	
	cudaError_t iError = cudaPointerGetAttributes( &hCUDAPtrAttr, m_pMemory );
	DebugAssert( iError == cudaSuccess );
	
	return hCUDAPtrAttr.device;
}

Void CUDAMemory::Set( SizeT iSize, Int iValue )
{
	// Check State is valid
	DebugAssert( IsAllocated() );
	DebugAssert( iSize <= m_iSize );
	
	// Perform Set
	cudaError_t iError = cudaMemset( m_pMemory, iValue, iSize );
	DebugAssert( iError == cudaSuccess );
}
Void CUDAMemory::Set( const CUDAMemoryPosition & hDestPos, const CUDAMemoryRegion & hSetRegion, Int iValue )
{
	// Check State is valid
	DebugAssert( IsAllocated() );
	DebugAssert( IsValidRegion(hDestPos, hSetRegion) );
	
	// Get Set Width
	SizeT iSetWidth = m_iStride * hSetRegion.iWidth;
	
	// Shapeless/1D cases
	if ( m_iShape <= CUDA_MEMORY_SHAPE_1D ) {
		Void * pDestMemory = GetPointer( hDestPos );
		
		// Perform Set
		cudaError_t iError = cudaMemset( pDestMemory, iValue, iSetWidth );
		DebugAssert( iError == cudaSuccess );
		
		return;
	}
	
	// 2D case
	if ( m_iShape == CUDA_MEMORY_SHAPE_2D ) {
		Void * pDestMemory = GetPointer( hDestPos );
		
		// Perform Set
		cudaError_t iError = cudaMemset2D( pDestMemory, m_iPitch, iValue, iSetWidth, hSetRegion.iHeight );
		DebugAssert( iError == cudaSuccess );
		
		return;
	}
	
	// 3D case
	cudaExtent hExtent;
	hExtent.width = iSetWidth;
	hExtent.height = hSetRegion.iHeight;
	hExtent.depth = hSetRegion.iDepth;
	
	cudaPitchedPtr hDestPtr;
	hDestPtr.ptr = GetPointer( hDestPos );
	hDestPtr.pitch = m_iPitch;
	hDestPtr.xsize = m_iStride * m_iWidth;
	hDestPtr.ysize = m_iHeight;
	
	// Perform Set
	cudaError_t iError = cudaMemset3D( hDestPtr, iValue, hExtent );
	DebugAssert( iError == cudaSuccess );
}

Void CUDAMemory::Copy( const CUDAMemory * pSrc, SizeT iSize )
{
	// Check States are valid
	DebugAssert( IsAllocated() && pSrc->IsAllocated() );
	DebugAssert( m_iShape == CUDA_MEMORY_SHAPE_NONE && pSrc->m_iShape == CUDA_MEMORY_SHAPE_NONE );
	DebugAssert( iSize <= m_iSize && iSize <= pSrc->m_iSize );
	
	// Get Transfer Kind
	cudaMemcpyKind iKind = (cudaMemcpyKind)( _GetMemCopyKind(pSrc) );
	
	// Perform Copy
	cudaError_t iError = cudaMemcpy( m_pMemory, pSrc->m_pMemory, iSize, iKind );
	DebugAssert( iError == cudaSuccess );
}
Void CUDAMemory::Copy( const CUDAMemoryPosition & hDestPos,
					   const CUDAMemory * pSrc, const CUDAMemoryPosition & hSrcPos,
					   const CUDAMemoryRegion & hCopyRegion )
{
	// Check States are valid
	DebugAssert( IsAllocated() && pSrc->IsAllocated() );
	DebugAssert( IsValidRegion(hDestPos, hCopyRegion) );
	DebugAssert( pSrc->IsValidRegion(hSrcPos, hCopyRegion) );
	DebugAssert( m_iStride == pSrc->m_iStride );
	
	// Get Transfer Kind
	cudaMemcpyKind iKind = (cudaMemcpyKind)( _GetMemCopyKind(pSrc) );
	
	// Get Copy Width
	SizeT iCopyWidth = m_iStride * hCopyRegion.iWidth;
	
	// Shapeless/1D cases
	if ( m_iShape <= CUDA_MEMORY_SHAPE_1D && pSrc->m_iShape <= CUDA_MEMORY_SHAPE_1D ) {
		Void * pDestMemory = GetPointer( hDestPos );
		const Void * pSrcMemory = pSrc->GetPointer( hSrcPos );
		
		// Perform Copy
		cudaError_t iError = cudaMemcpy( pDestMemory, pSrcMemory, iCopyWidth, iKind );
		DebugAssert( iError == cudaSuccess );
		
		return;
	}
	
	// 2D cases
	if ( m_iShape <= CUDA_MEMORY_SHAPE_2D && pSrc->m_iShape <= CUDA_MEMORY_SHAPE_2D ) {
		Void * pDestMemory = GetPointer( hDestPos );
		const Void * pSrcMemory = pSrc->GetPointer( hSrcPos );
		
		// Perform Copy
		cudaError_t iError = cudaMemcpy2D( pDestMemory, m_iPitch, pSrcMemory, pSrc->m_iPitch, iCopyWidth, hCopyRegion.iHeight, iKind );
		DebugAssert( iError == cudaSuccess );
		
		return;
	}
	
	// 3D cases
	cudaMemcpy3DParms hParams;
	hParams.kind = iKind;
	hParams.dstArray = NULL;
	hParams.srcArray = NULL;

	hParams.extent.width = iCopyWidth;
	hParams.extent.height = hCopyRegion.iHeight;
	hParams.extent.depth = hCopyRegion.iDepth;

	hParams.dstPtr.ptr = m_pMemory;
	hParams.dstPtr.pitch = m_iPitch;
	hParams.dstPtr.xsize = m_iStride * m_iWidth;
	hParams.dstPtr.ysize = m_iHeight;

	hParams.dstPos.x = hDestPos.iX * m_iStride;
	hParams.dstPos.y = hDestPos.iY;
	hParams.dstPos.z = hDestPos.iZ;

	hParams.srcPtr.ptr = pSrc->m_pMemory;
	hParams.srcPtr.pitch = pSrc->m_iPitch;
	hParams.srcPtr.xsize = pSrc->m_iStride * pSrc->m_iWidth;
	hParams.srcPtr.ysize = pSrc->m_iHeight;

	hParams.srcPos.x = hSrcPos.iX * pSrc->m_iStride;
	hParams.srcPos.y = hSrcPos.iY;
	hParams.srcPos.z = hSrcPos.iZ;
	
	// Perform Copy
	cudaError_t iError = cudaMemcpy3D( &hParams );
	DebugAssert( iError == cudaSuccess );
}

/////////////////////////////////////////////////////////////////////////////////

UInt CUDAMemory::_GetMemCopyKind( const CUDAMemory * pSrc ) const
{
	cudaMemcpyKind iKind;
	
	if ( IsHostMemory() ) {
		if ( pSrc->IsHostMemory() )
			iKind = cudaMemcpyHostToHost;
		else if ( pSrc->IsDeviceMemory() )
			iKind = cudaMemcpyDeviceToHost;
		else
			iKind = cudaMemcpyDefault;
	} else if ( IsDeviceMemory() ) {
		if ( pSrc->IsHostMemory() )
			iKind = cudaMemcpyHostToDevice;
		else if ( pSrc->IsDeviceMemory() )
			iKind = cudaMemcpyDeviceToDevice;
		else
			iKind = cudaMemcpyDefault;
	} else
		iKind = cudaMemcpyDefault;
	
	return (UInt)iKind;
}
							 
/////////////////////////////////////////////////////////////////////////////////
// CUDAHostMemory implementation
CUDAHostMemory::CUDAHostMemory():
	CUDAMemory()
{
	m_iHostMemoryAllocFlags = 0;
	m_bIsWrapped = false;
	m_iHostMemoryWrapFlags = 0;
}
CUDAHostMemory::~CUDAHostMemory()
{
	if ( m_bHasOwnerShip )
		Free();
	else if ( m_bIsWrapped )
		UnWrap();
}

Void CUDAHostMemory::Allocate( SizeT iSize, UInt iHostMemoryAllocFlags )
{
	DebugAssert( !IsAllocated() );
	
	Void * pHostMemory = NULL;
	
	cudaError_t iError = cudaHostAlloc( &pHostMemory, iSize, iHostMemoryAllocFlags );
	DebugAssert( iError == cudaSuccess && pHostMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pHostMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iWidth = iSize;
	m_iHeight = 1;
	m_iDepth = 1;

	m_iStride = 1;
	m_iPitch = m_iStride * m_iWidth;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
	
	m_iHostMemoryAllocFlags = iHostMemoryAllocFlags;
	m_bIsWrapped = false;
	m_iHostMemoryWrapFlags = 0;
}
Void CUDAHostMemory::Allocate1D( SizeT iElementSize, SizeT iWidth, UInt iHostMemoryAllocFlags )
{
	DebugAssert( !IsAllocated() );
	
	Void * pHostMemory = NULL;
	UInt iSize = iElementSize * iWidth;
	
	cudaError_t iError = cudaHostAlloc( &pHostMemory, iSize, iHostMemoryAllocFlags );
	DebugAssert( iError == cudaSuccess && pHostMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pHostMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_1D;
	m_iWidth = iWidth;
	m_iHeight = 1;
	m_iDepth = 1;

	m_iStride = iElementSize;
	m_iPitch = m_iStride * m_iWidth;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
	
	m_iHostMemoryAllocFlags = iHostMemoryAllocFlags;
	m_bIsWrapped = false;
	m_iHostMemoryWrapFlags = 0;
}
Void CUDAHostMemory::Allocate2D( SizeT iElementSize, SizeT iWidth, SizeT iHeight, UInt iHostMemoryAllocFlags )
{
	DebugAssert( !IsAllocated() );
	
	Void * pHostMemory = NULL;
	UInt iSize = iElementSize * iWidth * iHeight;
	
	cudaError_t iError = cudaHostAlloc( &pHostMemory, iSize, iHostMemoryAllocFlags );
	DebugAssert( iError == cudaSuccess && pHostMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pHostMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_2D;
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = 1;

	m_iStride = iElementSize;
	m_iPitch = m_iStride * m_iWidth;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
	
	m_iHostMemoryAllocFlags = iHostMemoryAllocFlags;
	m_bIsWrapped = false;
	m_iHostMemoryWrapFlags = 0;
}
Void CUDAHostMemory::Allocate3D( SizeT iElementSize, SizeT iWidth, SizeT iHeight, SizeT iDepth, UInt iHostMemoryAllocFlags )
{
	DebugAssert( !IsAllocated() );
	
	Void * pHostMemory = NULL;
	UInt iSize = iElementSize * iWidth * iHeight * iDepth;
	
	cudaError_t iError = cudaHostAlloc( &pHostMemory, iSize, iHostMemoryAllocFlags );
	DebugAssert( iError == cudaSuccess && pHostMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pHostMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_2D;
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = iDepth;

	m_iStride = iElementSize;
	m_iPitch = m_iStride * m_iWidth;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
	
	m_iHostMemoryAllocFlags = iHostMemoryAllocFlags;
	m_bIsWrapped = false;
	m_iHostMemoryWrapFlags = 0;
}
Void CUDAHostMemory::Free()
{
	DebugAssert( IsAllocated() );
	DebugAssert( m_bHasOwnerShip && !m_bIsWrapped );
	
	cudaError_t iError = cudaFreeHost( m_pMemory );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = NULL;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;

	m_iStride = 0;
	m_iPitch = 0;
	m_iSlice = 0;
	m_iSize = 0;
	
	m_iHostMemoryAllocFlags = 0;
	m_bIsWrapped = false;
	m_iHostMemoryWrapFlags = 0;
}

Void CUDAHostMemory::Wrap( Void * pSystemMemory, SizeT iSize, UInt iHostMemoryWrapFlags )
{
	DebugAssert( !IsAllocated() );
	
	cudaError_t iError = cudaHostRegister( pSystemMemory, iSize, iHostMemoryWrapFlags );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = pSystemMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iWidth = iSize;
	m_iHeight = 1;
	m_iDepth = 1;

	m_iStride = 1;
	m_iPitch = m_iStride * m_iWidth;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
	
	m_iHostMemoryAllocFlags = 0;
	m_bIsWrapped = true;
	m_iHostMemoryWrapFlags = iHostMemoryWrapFlags;
}
Void CUDAHostMemory::Wrap1D( Void * pSystemMemory, SizeT iElementSize, SizeT iWidth, UInt iHostMemoryWrapFlags )
{
	DebugAssert( !IsAllocated() );
	
	UInt iSize = iElementSize * iWidth;
	
	cudaError_t iError = cudaHostRegister( pSystemMemory, iSize, iHostMemoryWrapFlags );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = pSystemMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_1D;
	m_iWidth = iWidth;
	m_iHeight = 1;
	m_iDepth = 1;

	m_iStride = iElementSize;
	m_iPitch = m_iStride * m_iWidth;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
	
	m_iHostMemoryAllocFlags = 0;
	m_bIsWrapped = true;
	m_iHostMemoryWrapFlags = iHostMemoryWrapFlags;
}
Void CUDAHostMemory::Wrap2D( Void * pSystemMemory, SizeT iElementSize, SizeT iWidth, SizeT iHeight, UInt iHostMemoryWrapFlags )
{
	DebugAssert( !IsAllocated() );
	
	UInt iSize = iElementSize * iWidth * iHeight;
	
	cudaError_t iError = cudaHostRegister( pSystemMemory, iSize, iHostMemoryWrapFlags );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = pSystemMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_2D;
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = 1;

	m_iStride = iElementSize;
	m_iPitch = m_iStride * m_iWidth;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
	
	m_iHostMemoryAllocFlags = 0;
	m_bIsWrapped = true;
	m_iHostMemoryWrapFlags = iHostMemoryWrapFlags;
}
Void CUDAHostMemory::Wrap3D( Void * pSystemMemory, SizeT iElementSize, SizeT iWidth, SizeT iHeight, SizeT iDepth, UInt iHostMemoryWrapFlags )
{
	DebugAssert( !IsAllocated() );
	
	UInt iSize = iElementSize * iWidth * iHeight * iDepth;
	
	cudaError_t iError = cudaHostRegister( pSystemMemory, iSize, iHostMemoryWrapFlags );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = pSystemMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_3D;
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = iDepth;

	m_iStride = iElementSize;
	m_iPitch = m_iStride * m_iWidth;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
	
	m_iHostMemoryAllocFlags = 0;
	m_bIsWrapped = true;
	m_iHostMemoryWrapFlags = iHostMemoryWrapFlags;
}
Void CUDAHostMemory::UnWrap()
{
	DebugAssert( IsAllocated() );
	DebugAssert( !m_bHasOwnerShip && m_bIsWrapped );
	
	cudaError_t iError = cudaHostUnregister( m_pMemory );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = NULL;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;

	m_iStride = 0;
	m_iPitch = 0;
	m_iSlice = 0;
	m_iSize = 0;
	
	m_iHostMemoryAllocFlags = 0;
	m_bIsWrapped = false;
	m_iHostMemoryWrapFlags = 0;
}

// Void CUDAHostMemory::GetMappedDeviceMemory( CUDADeviceMemory * outDeviceMemory ) const
// {
// DebugAssert( m_pMemory != NULL );
// DebugAssert( IsMapped() );
// DebugAssert( !(outDeviceMemory->IsAllocated()) );

// Void * pDeviceMemory = NULL;

// cudaError_t iError = cudaHostGetDevicePointer( &pDeviceMemory, m_pMemory, 0 );
// DebugAssert( iError == cudaSuccess && pDeviceMemory != NULL );

// outDeviceMemory->m_bHasOwnerShip = false;
// outDeviceMemory->m_pMemory = pDeviceMemory;

// outDeviceMemory->m_iShape = m_iShape;
// outDeviceMemory->m_iStride = m_iStride;
// outDeviceMemory->m_iPitch = m_iPitch; // Pitch value on device might be different ! Any way to query this ?
// outDeviceMemory->m_iSlice = m_iSlice; // Slice value depends on Pitch value

// outDeviceMemory->m_iWidth = m_iWidth;
// outDeviceMemory->m_iHeight = m_iHeight;
// outDeviceMemory->m_iDepth = m_iDepth;
// outDeviceMemory->m_iSize = m_iSize;
// }

/////////////////////////////////////////////////////////////////////////////////
// CUDADeviceMemory implementation
CUDADeviceMemory::CUDADeviceMemory():
	CUDAMemory()
{
	// nothing to do
}
CUDADeviceMemory::~CUDADeviceMemory()
{
	if ( m_bHasOwnerShip )
		Free();
}

Void CUDADeviceMemory::GetCapacity( SizeT * outFreeMemory, SizeT * outTotalMemory )
{
	cudaError_t iError = cudaMemGetInfo( outFreeMemory, outTotalMemory );
	DebugAssert( iError == cudaSuccess );
}

Void CUDADeviceMemory::Allocate( SizeT iSize )
{
	DebugAssert( !IsAllocated() );
	
	Void * pDeviceMemory = NULL;
	
	cudaError_t iError = cudaMalloc( &pDeviceMemory, iSize );
	DebugAssert( iError == cudaSuccess && pDeviceMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pDeviceMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iWidth = iSize;
	m_iHeight = 1;
	m_iDepth = 1;

	m_iStride = 1;
	m_iPitch = m_iStride * m_iWidth;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
}
Void CUDADeviceMemory::Allocate1D( SizeT iElementSize, SizeT iWidth )
{
	DebugAssert( !IsAllocated() );
	
	cudaPitchedPtr hPitchedDeviceMemory;
	cudaExtent hExtent;
	hExtent.width = iElementSize * iWidth;
	hExtent.height = 1;
	hExtent.depth = 1;
	
	cudaError_t iError = cudaMalloc3D( &hPitchedDeviceMemory, hExtent );
	DebugAssert( iError == cudaSuccess && hPitchedDeviceMemory.ptr != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = hPitchedDeviceMemory.ptr;
	
	m_iShape = CUDA_MEMORY_SHAPE_1D;
	m_iWidth = iWidth;
	m_iHeight = 1;
	m_iDepth = 1;

	m_iStride = iElementSize;
	m_iPitch = hPitchedDeviceMemory.pitch;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
}
Void CUDADeviceMemory::Allocate2D( SizeT iElementSize, SizeT iWidth, SizeT iHeight )
{
	DebugAssert( !IsAllocated() );
	
	cudaPitchedPtr hPitchedDeviceMemory;
	cudaExtent hExtent;
	hExtent.width = iElementSize * iWidth;
	hExtent.height = iHeight;
	hExtent.depth = 1;
	
	cudaError_t iError = cudaMalloc3D( &hPitchedDeviceMemory, hExtent );
	DebugAssert( iError == cudaSuccess && hPitchedDeviceMemory.ptr != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = hPitchedDeviceMemory.ptr;
	
	m_iShape = CUDA_MEMORY_SHAPE_2D;
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = 1;

	m_iStride = iElementSize;
	m_iPitch = hPitchedDeviceMemory.pitch;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
}
Void CUDADeviceMemory::Allocate3D( SizeT iElementSize, SizeT iWidth, SizeT iHeight, SizeT iDepth )
{
	DebugAssert( !IsAllocated() );
	
	cudaPitchedPtr hPitchedDeviceMemory;
	cudaExtent hExtent;
	hExtent.width = iElementSize * iWidth;
	hExtent.height = iHeight;
	hExtent.depth = iDepth;
	
	cudaError_t iError = cudaMalloc3D( &hPitchedDeviceMemory, hExtent );
	DebugAssert( iError == cudaSuccess && hPitchedDeviceMemory.ptr != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = hPitchedDeviceMemory.ptr;
	
	m_iShape = CUDA_MEMORY_SHAPE_3D;
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = iDepth;

	m_iStride = iElementSize;
	m_iPitch = hPitchedDeviceMemory.pitch;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
}
Void CUDADeviceMemory::Free()
{
	DebugAssert( IsAllocated() );
	DebugAssert( m_bHasOwnerShip );
	
	cudaError_t iError = cudaFree( m_pMemory );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = NULL;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;

	m_iStride = 0;
	m_iPitch = 0;
	m_iSlice = 0;
	m_iSize = 0;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDAManagedMemory implementation
CUDAManagedMemory::CUDAManagedMemory():
	CUDAMemory()
{
	// nothing to do
}
CUDAManagedMemory::~CUDAManagedMemory()
{
	if ( m_bHasOwnerShip )
		Free();
}

Void CUDAManagedMemory::Allocate( SizeT iSize, Bool bAttachHost )
{
	DebugAssert( !IsAllocated() );
	
	Void * pDeviceMemory = NULL;
	
	UInt iFlags = cudaMemAttachGlobal;
	if ( bAttachHost )
		iFlags = cudaMemAttachHost;
	
	cudaError_t iError = cudaMallocManaged( &pDeviceMemory, iSize, iFlags );
	DebugAssert( iError == cudaSuccess && pDeviceMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pDeviceMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iWidth = iSize;
	m_iHeight = 1;
	m_iDepth = 1;

	m_iStride = 1;
	m_iPitch = m_iStride * m_iWidth;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
}
Void CUDAManagedMemory::Allocate1D( SizeT iElementSize, SizeT iWidth, Bool bAttachHost )
{
	DebugAssert( !IsAllocated() );
	
	Void * pDeviceMemory = NULL;
	UInt iSize = iElementSize * iWidth;
	
	UInt iFlags = cudaMemAttachGlobal;
	if ( bAttachHost )
		iFlags = cudaMemAttachHost;
	
	cudaError_t iError = cudaMallocManaged( &pDeviceMemory, iSize, iFlags );
	DebugAssert( iError == cudaSuccess && pDeviceMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pDeviceMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_1D;
	m_iWidth = iWidth;
	m_iHeight = 1;
	m_iDepth = 1;

	m_iStride = iElementSize;
	m_iPitch = m_iStride * m_iWidth;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
}
Void CUDAManagedMemory::Allocate2D( SizeT iElementSize, SizeT iWidth, SizeT iHeight, Bool bAttachHost )
{
	DebugAssert( !IsAllocated() );
	
	Void * pDeviceMemory = NULL;
	UInt iSize = iElementSize * iWidth * iHeight;
	
	UInt iFlags = cudaMemAttachGlobal;
	if ( bAttachHost )
		iFlags = cudaMemAttachHost;
	
	cudaError_t iError = cudaMallocManaged( &pDeviceMemory, iSize, iFlags );
	DebugAssert( iError == cudaSuccess && pDeviceMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pDeviceMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_2D;
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = 1;

	m_iStride = iElementSize;
	m_iPitch = m_iStride * m_iWidth;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
}
Void CUDAManagedMemory::Allocate3D( SizeT iElementSize, SizeT iWidth, SizeT iHeight, SizeT iDepth, Bool bAttachHost )
{
	DebugAssert( !IsAllocated() );
	
	Void * pDeviceMemory = NULL;
	UInt iSize = iElementSize * iWidth * iHeight * iDepth;
	
	UInt iFlags = cudaMemAttachGlobal;
	if ( bAttachHost )
		iFlags = cudaMemAttachHost;
	
	cudaError_t iError = cudaMallocManaged( &pDeviceMemory, iSize, iFlags );
	DebugAssert( iError == cudaSuccess && pDeviceMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pDeviceMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_3D;
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = iDepth;

	m_iStride = iElementSize;
	m_iPitch = m_iStride * m_iWidth;
	m_iSlice = m_iPitch * m_iHeight;
	m_iSize = m_iSlice * m_iDepth;
}
Void CUDAManagedMemory::Free()
{
	DebugAssert( IsAllocated() );
	DebugAssert( m_bHasOwnerShip );
	
	cudaError_t iError = cudaFree( m_pMemory );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = NULL;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;

	m_iStride = 0;
	m_iPitch = 0;
	m_iSlice = 0;
	m_iSize = 0;
}



