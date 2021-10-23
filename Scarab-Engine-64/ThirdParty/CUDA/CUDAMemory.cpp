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
// CUDAMemory implementation
CUDAMemory::CUDAMemory()
{
	m_bHasOwnerShip = false;
	m_pMemory = NULL;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iStride = 0;
	m_iPitch = 0;
	m_iSlice = 0;
	
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;
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
	DebugAssert( m_pMemory != NULL );
	
	cudaPointerAttributes hCUDAPtrAttr;
	
	cudaError_t iError = cudaPointerGetAttributes( &hCUDAPtrAttr, m_pMemory );
	DebugAssert( iError == cudaSuccess );
	
	return hCUDAPtrAttr.device;
}

Void * CUDAMemory::GetPointer( const CUDAMemoryPosition & hPosition )
{
	DebugAssert( m_pMemory != NULL );
	switch( m_iShape ) {
		case CUDA_MEMORY_SHAPE_NONE:
			return GetPointer( hPosition.iX );
		case CUDA_MEMORY_SHAPE_1D:
			DebugAssert( hPosition.iX < m_iWidth );
			return ((Byte*)m_pMemory) + hPosition.iX * m_iStride;
		case CUDA_MEMORY_SHAPE_2D:
			DebugAssert( hPosition.iX < m_iWidth && hPosition.iY < m_iHeight );
			return ((Byte*)m_pMemory) + hPosition.iY * m_iPitch + hPosition.iX * m_iStride;
		case CUDA_MEMORY_SHAPE_3D:
			DebugAssert(  hPosition.iX < m_iWidth &&  hPosition.iY < m_iHeight &&  hPosition.iZ < m_iDepth );
			return ((Byte*)m_pMemory) +  hPosition.iZ * m_iSlice +  hPosition.iY * m_iPitch +  hPosition.iX * m_iStride;
		default: DebugAssert(false); return NULL;
	}
}
const Void * CUDAMemory::GetPointer( const CUDAMemoryPosition & hPosition ) const
{
	DebugAssert( m_pMemory != NULL );
	switch( m_iShape ) {
		case CUDA_MEMORY_SHAPE_NONE:
			return GetPointer( hPosition.iX );
		case CUDA_MEMORY_SHAPE_1D:
			DebugAssert( hPosition.iX < m_iWidth );
			return ((Byte*)m_pMemory) + hPosition.iX * m_iStride;
		case CUDA_MEMORY_SHAPE_2D:
			DebugAssert( hPosition.iX < m_iWidth && hPosition.iY < m_iHeight );
			return ((Byte*)m_pMemory) + hPosition.iY * m_iPitch + hPosition.iX * m_iStride;
		case CUDA_MEMORY_SHAPE_3D:
			DebugAssert(  hPosition.iX < m_iWidth &&  hPosition.iY < m_iHeight &&  hPosition.iZ < m_iDepth );
			return ((Byte*)m_pMemory) +  hPosition.iZ * m_iSlice +  hPosition.iY * m_iPitch +  hPosition.iX * m_iStride;
		default: DebugAssert(false); return NULL;
	}
}

Void CUDAMemory::MemSet( Int iValue, SizeT iSize )
{
	// Check State is valid
	DebugAssert( IsAllocated() );
	
	// Check iSize is valid
	DebugAssert( iSize <= m_iSize );
	
	// Perform Set
	cudaError_t iError = cudaMemset( m_pMemory, iValue, iSize );
	DebugAssert( iError == cudaSuccess );
}
Void CUDAMemory::MemSet( const CUDAMemoryPosition & hDestPos, const CUDAMemoryRegion & hSetRegion, UInt iValue )
{
	// Check State is valid
	DebugAssert( IsAllocated() );
	
	// Get Stride
	SizeT iStride = 1;
	if ( m_iShape != CUDA_MEMORY_SHAPE_NONE )
		iStride = m_iStride;
	
	// Get Set Width
	SizeT iSetWidth = iStride * hSetRegion.iWidth;
	
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
	hDestPtr.xsize = iStride * m_iWidth;
	hDestPtr.ysize = m_iHeight;
	
	// Perform Set
	cudaError_t iError = cudaMemset3D( hDestPtr, iValue, hExtent );
	DebugAssert( iError == cudaSuccess );
}

Void CUDAMemory::MemCopy( const CUDAMemory * pSrc, SizeT iSize )
{
	// Check States are valid
	DebugAssert( IsAllocated() && pSrc->IsAllocated() );
	
	// Check iSize is valid
	DebugAssert( iSize <= m_iSize && iSize <= pSrc->m_iSize );
	
	// Get Transfer Kind
	cudaMemcpyKind iKind = (cudaMemcpyKind)( _GetMemCopyKind(pSrc) );
	
	// Perform Copy
	cudaError_t iError = cudaMemcpy( m_pMemory, pSrc->m_pMemory, iSize, iKind );
	DebugAssert( iError == cudaSuccess );
}
Void CUDAMemory::MemCopy( const CUDAMemoryPosition & hDestPos,
						  const CUDAMemory * pSrc, const CUDAMemoryPosition & hSrcPos,
						  const CUDAMemoryRegion & hCopyRegion )
{
	// Check States are valid
	DebugAssert( IsAllocated() && pSrc->IsAllocated() );
	
	// Get Transfer Kind
	cudaMemcpyKind iKind = (cudaMemcpyKind)( _GetMemCopyKind(pSrc) );
	
	// Get Stride and check match
	SizeT iStride = 0;
	if ( m_iShape != CUDA_MEMORY_SHAPE_NONE )
		iStride = m_iStride;
	if ( pSrc->m_iShape != CUDA_MEMORY_SHAPE_NONE ) {
		DebugAssert( iStride == 0 || iStride == pSrc->m_iStride );
		iStride = pSrc->m_iStride;
	}
	if ( iStride == 0 )
		iStride = 1;
	
	// Get Copy Width
	SizeT iCopyWidth = iStride * hCopyRegion.iWidth;
	
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
		
		UInt iDestPitch = 0;
		if ( m_iShape == CUDA_MEMORY_SHAPE_2D )
			iDestPitch = m_iPitch;
		else
			iDestPitch = iStride * hCopyRegion.iWidth;
		
		UInt iSrcPitch = 0;
		if ( pSrc->m_iShape == CUDA_MEMORY_SHAPE_2D )
			iSrcPitch = m_iPitch;
		else
			iSrcPitch = iStride * hCopyRegion.iWidth;
		
		// Perform Copy
		cudaError_t iError = cudaMemcpy2D( pDestMemory, iDestPitch, pSrcMemory, iSrcPitch, iCopyWidth, hCopyRegion.iHeight, iKind );
		DebugAssert( iError == cudaSuccess );
		
		return;
	}
	
	// 3D cases
	cudaMemcpy3DParms hParams;
	_ConvertCopyParams( &hParams, hDestPos, pSrc, hSrcPos, hCopyRegion );
	
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
Void CUDAMemory::_ConvertCopyParams( Void * outParams,
									 const CUDAMemoryPosition & hDestPos,
									 const CUDAMemory * pSrc, const CUDAMemoryPosition & hSrcPos,
									 const CUDAMemoryRegion & hCopyRegion ) const
{
	cudaMemcpy3DParms * pParams = (cudaMemcpy3DParms*)outParams;
	
	// Check States are valid
	DebugAssert( IsAllocated() && pSrc->IsAllocated() );
	
	// Get Transfer Kind
	cudaMemcpyKind iKind = (cudaMemcpyKind)( _GetMemCopyKind(pSrc) );
	
	// Get Stride and check match
	SizeT iStride = 0;
	if ( m_iShape != CUDA_MEMORY_SHAPE_NONE )
		iStride = m_iStride;
	if ( pSrc->m_iShape != CUDA_MEMORY_SHAPE_NONE ) {
		DebugAssert( iStride == 0 || iStride == pSrc->m_iStride );
		iStride = pSrc->m_iStride;
	}
	if ( iStride == 0 )
		iStride = 1;
	
	// Get Copy Width
	SizeT iCopyWidth = iStride * hCopyRegion.iWidth;
		
	// Most general 3D copy
	pParams->kind = iKind;
	pParams->dstArray = NULL;
	pParams->srcArray = NULL;
	
	pParams->extent.width = iCopyWidth;
	pParams->extent.height = hCopyRegion.iHeight;
	pParams->extent.depth = hCopyRegion.iDepth;
	
	pParams->dstPtr.ptr = m_pMemory;
	pParams->dstPtr.pitch = iCopyWidth;
	pParams->dstPtr.xsize = iCopyWidth;
	pParams->dstPtr.ysize = hCopyRegion.iHeight;
	
	pParams->dstPos.x = iStride * hDestPos.iX;
	pParams->dstPos.y = 0;
	pParams->dstPos.z = 0;
	
	pParams->srcPtr.ptr = pSrc->m_pMemory;
	pParams->srcPtr.pitch = iCopyWidth;
	pParams->srcPtr.xsize = iCopyWidth;
	pParams->srcPtr.ysize = hCopyRegion.iHeight;
	
	pParams->srcPos.x = iStride * hSrcPos.iX;
	pParams->srcPos.y = 0;
	pParams->srcPos.z = 0;
	
	// Destination Shape
	if ( m_iShape >= CUDA_MEMORY_SHAPE_2D ) {
		pParams->dstPtr.pitch = m_iPitch;
		pParams->dstPtr.xsize = iStride * m_iWidth;
		pParams->dstPos.y = hDestPos.iY;
	}
	if ( m_iShape == CUDA_MEMORY_SHAPE_3D ) {
		pParams->dstPtr.ysize = m_iHeight;
		pParams->dstPos.z = hDestPos.iZ;
	}

	// Source Shape
	if ( pSrc->m_iShape >= CUDA_MEMORY_SHAPE_2D ) {
		pParams->srcPtr.pitch = pSrc->m_iPitch;
		pParams->srcPtr.xsize = iStride * pSrc->m_iWidth;
		pParams->srcPos.y = hSrcPos.iY;
	}
	if ( pSrc->m_iShape == CUDA_MEMORY_SHAPE_3D ) {
		pParams->srcPtr.ysize = pSrc->m_iHeight;
		pParams->srcPos.z = hSrcPos.iZ;
	}
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
	DebugAssert( m_pMemory == NULL );
	
	Void * pHostMemory = NULL;
	
	cudaError_t iError = cudaHostAlloc( &pHostMemory, iSize, iHostMemoryAllocFlags );
	DebugAssert( iError == cudaSuccess && pHostMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pHostMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iStride = 0;
	m_iPitch = 0;
	m_iSlice = 0;
	
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;
	m_iSize = iSize;
	
	m_iHostMemoryAllocFlags = iHostMemoryAllocFlags;
	m_bIsWrapped = false;
	m_iHostMemoryWrapFlags = 0;
}
Void CUDAHostMemory::Allocate( SizeT iElementSize, SizeT iWidth, UInt iHostMemoryAllocFlags )
{
	DebugAssert( m_pMemory == NULL );
	
	Void * pHostMemory = NULL;
	UInt iSize = iElementSize * iWidth;
	
	cudaError_t iError = cudaHostAlloc( &pHostMemory, iSize, iHostMemoryAllocFlags );
	DebugAssert( iError == cudaSuccess && pHostMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pHostMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_1D;
	m_iStride = iElementSize;
	m_iPitch = 0;
	m_iSlice = 0;
	
	m_iWidth = iWidth;
	m_iHeight = 0;
	m_iDepth = 0;
	m_iSize = iSize;
	
	m_iHostMemoryAllocFlags = iHostMemoryAllocFlags;
	m_bIsWrapped = false;
	m_iHostMemoryWrapFlags = 0;
}
Void CUDAHostMemory::Allocate( SizeT iElementSize, SizeT iWidth, SizeT iHeight, UInt iHostMemoryAllocFlags )
{
	DebugAssert( m_pMemory == NULL );
	
	Void * pHostMemory = NULL;
	UInt iSize = iElementSize * iWidth * iHeight;
	
	cudaError_t iError = cudaHostAlloc( &pHostMemory, iSize, iHostMemoryAllocFlags );
	DebugAssert( iError == cudaSuccess && pHostMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pHostMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_2D;
	m_iStride = iElementSize;
	m_iPitch = m_iStride * iWidth;
	m_iSlice = 0;
	
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = 0;
	m_iSize = iSize;
	
	m_iHostMemoryAllocFlags = iHostMemoryAllocFlags;
	m_bIsWrapped = false;
	m_iHostMemoryWrapFlags = 0;
}
Void CUDAHostMemory::Allocate( SizeT iElementSize, SizeT iWidth, SizeT iHeight, SizeT iDepth, UInt iHostMemoryAllocFlags )
{
	DebugAssert( m_pMemory == NULL );
	
	Void * pHostMemory = NULL;
	UInt iSize = iElementSize * iWidth * iHeight * iDepth;
	
	cudaError_t iError = cudaHostAlloc( &pHostMemory, iSize, iHostMemoryAllocFlags );
	DebugAssert( iError == cudaSuccess && pHostMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pHostMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_3D;
	m_iStride = iElementSize;
	m_iPitch = m_iStride * iWidth;
	m_iSlice = m_iPitch * iHeight;
	
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = iDepth;
	m_iSize = iSize;
	
	m_iHostMemoryAllocFlags = iHostMemoryAllocFlags;
	m_bIsWrapped = false;
	m_iHostMemoryWrapFlags = 0;
}
Void CUDAHostMemory::Free()
{
	if ( m_pMemory == NULL )
		return;
	DebugAssert( m_bHasOwnerShip && !m_bIsWrapped );
	
	cudaError_t iError = cudaFreeHost( m_pMemory );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = NULL;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iStride = 0;
	m_iPitch = 0;
	m_iSlice = 0;
	
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;
	m_iSize = 0;
	
	m_iHostMemoryAllocFlags = 0;
	m_bIsWrapped = false;
	m_iHostMemoryWrapFlags = 0;
}

Void CUDAHostMemory::Wrap( Void * pSystemMemory, SizeT iSize, UInt iHostMemoryWrapFlags )
{
	DebugAssert( m_pMemory == NULL );
	
	cudaError_t iError = cudaHostRegister( pSystemMemory, iSize, iHostMemoryWrapFlags );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = pSystemMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iStride = 0;
	m_iPitch = 0;
	m_iSlice = 0;
	
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;
	m_iSize = iSize;
	
	m_iHostMemoryAllocFlags = 0;
	m_bIsWrapped = true;
	m_iHostMemoryWrapFlags = iHostMemoryWrapFlags;
}
Void CUDAHostMemory::Wrap( Void * pSystemMemory, SizeT iElementSize, SizeT iWidth, UInt iHostMemoryWrapFlags )
{
	DebugAssert( m_pMemory == NULL );
	
	UInt iSize = iElementSize * iWidth;
	
	cudaError_t iError = cudaHostRegister( pSystemMemory, iSize, iHostMemoryWrapFlags );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = pSystemMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_1D;
	m_iStride = iElementSize;
	m_iPitch = 0;
	m_iSlice = 0;
	
	m_iWidth = iWidth;
	m_iHeight = 0;
	m_iDepth = 0;
	m_iSize = iSize;
	
	m_iHostMemoryAllocFlags = 0;
	m_bIsWrapped = true;
	m_iHostMemoryWrapFlags = iHostMemoryWrapFlags;
}
Void CUDAHostMemory::Wrap( Void * pSystemMemory, SizeT iElementSize, SizeT iWidth, SizeT iHeight, UInt iHostMemoryWrapFlags )
{
	DebugAssert( m_pMemory == NULL );
	
	UInt iSize = iElementSize * iWidth * iHeight;
	
	cudaError_t iError = cudaHostRegister( pSystemMemory, iSize, iHostMemoryWrapFlags );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = pSystemMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_2D;
	m_iStride = iElementSize;
	m_iPitch = m_iStride * iWidth;
	m_iSlice = 0;
	
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = 0;
	m_iSize = iSize;
	
	m_iHostMemoryAllocFlags = 0;
	m_bIsWrapped = true;
	m_iHostMemoryWrapFlags = iHostMemoryWrapFlags;
}
Void CUDAHostMemory::Wrap( Void * pSystemMemory, SizeT iElementSize, SizeT iWidth, SizeT iHeight, SizeT iDepth, UInt iHostMemoryWrapFlags )
{
	DebugAssert( m_pMemory == NULL );
	
	UInt iSize = iElementSize * iWidth * iHeight * iDepth;
	
	cudaError_t iError = cudaHostRegister( pSystemMemory, iSize, iHostMemoryWrapFlags );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = pSystemMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_3D;
	m_iStride = iElementSize;
	m_iPitch = m_iStride * iWidth;
	m_iSlice = m_iPitch * iHeight;
	
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = iDepth;
	m_iSize = iSize;
	
	m_iHostMemoryAllocFlags = 0;
	m_bIsWrapped = true;
	m_iHostMemoryWrapFlags = iHostMemoryWrapFlags;
}
Void CUDAHostMemory::UnWrap()
{
	if ( m_pMemory == NULL )
		return;
	DebugAssert( m_bIsWrapped );
	
	cudaError_t iError = cudaHostUnregister( m_pMemory );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = NULL;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iStride = 0;
	m_iPitch = 0;
	m_iSlice = 0;
	
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;
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
	DebugAssert( m_pMemory == NULL );
	
	Void * pDeviceMemory = NULL;
	
	cudaError_t iError = cudaMalloc( &pDeviceMemory, iSize );
	DebugAssert( iError == cudaSuccess && pDeviceMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pDeviceMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iStride = 0;
	m_iPitch = 0;
	m_iSlice = 0;
	
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;
	m_iSize = iSize;
}
Void CUDADeviceMemory::Allocate( SizeT iElementSize, SizeT iWidth )
{
	DebugAssert( m_pMemory == NULL );
	
	cudaPitchedPtr hPitchedDeviceMemory;
	cudaExtent hExtent;
	hExtent.width = iElementSize * iWidth;
	hExtent.height = 0;
	hExtent.depth = 0;
	
	cudaError_t iError = cudaMalloc3D( &hPitchedDeviceMemory, hExtent );
	DebugAssert( iError == cudaSuccess && hPitchedDeviceMemory.ptr != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = hPitchedDeviceMemory.ptr;
	
	m_iShape = CUDA_MEMORY_SHAPE_1D;
	m_iStride = iElementSize;
	m_iPitch = 0;
	m_iSlice = 0;
	
	m_iWidth = iWidth;
	m_iHeight = 0;
	m_iDepth = 0;
	m_iSize = hPitchedDeviceMemory.pitch;
}
Void CUDADeviceMemory::Allocate( SizeT iElementSize, SizeT iWidth, SizeT iHeight )
{
	DebugAssert( m_pMemory == NULL );
	
	cudaPitchedPtr hPitchedDeviceMemory;
	cudaExtent hExtent;
	hExtent.width = iElementSize * iWidth;
	hExtent.height = iHeight;
	hExtent.depth = 0;
	
	cudaError_t iError = cudaMalloc3D( &hPitchedDeviceMemory, hExtent );
	DebugAssert( iError == cudaSuccess && hPitchedDeviceMemory.ptr != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = hPitchedDeviceMemory.ptr;
	
	m_iShape = CUDA_MEMORY_SHAPE_2D;
	m_iStride = iElementSize;
	m_iPitch = hPitchedDeviceMemory.pitch;
	m_iSlice = 0;
	
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = 0;
	m_iSize = m_iPitch * iHeight;
}
Void CUDADeviceMemory::Allocate( SizeT iElementSize, SizeT iWidth, SizeT iHeight, SizeT iDepth )
{
	DebugAssert( m_pMemory == NULL );
	
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
	m_iStride = iElementSize;
	m_iPitch = hPitchedDeviceMemory.pitch;
	m_iSlice = hPitchedDeviceMemory.pitch * iHeight;
	
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = iDepth;
	m_iSize = m_iSlice * iDepth;
}
Void CUDADeviceMemory::Free()
{
	if ( m_pMemory == NULL )
		return;
	DebugAssert( m_bHasOwnerShip );
	
	cudaError_t iError = cudaFree( m_pMemory );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = NULL;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iStride = 0;
	m_iPitch = 0;
	m_iSlice = 0;
	
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;
	m_iSize = 0;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDAManagedMemory implementation
CUDAManagedMemory::CUDAManagedMemory():
	CUDAMemory()
{
	
}
CUDAManagedMemory::~CUDAManagedMemory()
{
	if ( m_bHasOwnerShip )
		Free();
}

Void CUDAManagedMemory::Allocate( SizeT iSize, Bool bAttachHost )
{
	DebugAssert( m_pMemory == NULL );
	
	Void * pDeviceMemory = NULL;
	
	UInt iFlags = cudaMemAttachGlobal;
	if ( bAttachHost )
		iFlags = cudaMemAttachHost;
	
	cudaError_t iError = cudaMallocManaged( &pDeviceMemory, iSize, iFlags );
	DebugAssert( iError == cudaSuccess && pDeviceMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_pMemory = pDeviceMemory;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iStride = 0;
	m_iPitch = 0;
	m_iSlice = 0;
	
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;
	m_iSize = iSize;
}
Void CUDAManagedMemory::Allocate( SizeT iElementSize, SizeT iWidth, Bool bAttachHost )
{
	DebugAssert( m_pMemory == NULL );
	
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
	m_iStride = iElementSize;
	m_iPitch = 0;
	m_iSlice = 0;
	
	m_iWidth = iWidth;
	m_iHeight = 0;
	m_iDepth = 0;
	m_iSize = iSize;
}
Void CUDAManagedMemory::Allocate( SizeT iElementSize, SizeT iWidth, SizeT iHeight, Bool bAttachHost )
{
	DebugAssert( m_pMemory == NULL );
	
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
	m_iStride = iElementSize;
	m_iPitch = m_iStride * iWidth;
	m_iSlice = 0;
	
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = 0;
	m_iSize = iSize;
}
Void CUDAManagedMemory::Allocate( SizeT iElementSize, SizeT iWidth, SizeT iHeight, SizeT iDepth, Bool bAttachHost )
{
	DebugAssert( m_pMemory == NULL );
	
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
	m_iStride = iElementSize;
	m_iPitch = m_iStride * iWidth;
	m_iSlice = m_iPitch * iHeight;
	
	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_iDepth = iDepth;
	m_iSize = iSize;
}
Void CUDAManagedMemory::Free()
{
	if ( m_pMemory == NULL )
		return;
	DebugAssert( m_bHasOwnerShip );
	
	cudaError_t iError = cudaFree( m_pMemory );
	DebugAssert( iError == cudaSuccess );
	
	m_bHasOwnerShip = false;
	m_pMemory = NULL;
	
	m_iShape = CUDA_MEMORY_SHAPE_NONE;
	m_iStride = 0;
	m_iPitch = 0;
	m_iSlice = 0;
	
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;
	m_iSize = 0;
}



