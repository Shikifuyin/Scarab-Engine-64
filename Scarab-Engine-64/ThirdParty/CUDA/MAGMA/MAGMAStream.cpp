/////////////////////////////////////////////////////////////////////////////////
// File : MAGMAStream.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : MAGMA Streams
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <magma_v2.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "MAGMAStream.h"

/////////////////////////////////////////////////////////////////////////////////
// MAGMAStream implementation
MAGMAStream::MAGMAStream()
{
	m_iDeviceID = 0;
	m_hMAGMAStream = NULL;
}
MAGMAStream::~MAGMAStream()
{
	if ( m_hMAGMAStream != NULL )
		Destroy();
}

Void MAGMAStream::Create( CUDADeviceID iDeviceID )
{
	Assert( m_hMAGMAStream == NULL );
	
	magma_queue_t hMAGMAQueue = NULL;
	magma_queue_create( iDeviceID, &hMAGMAQueue );
	
	m_iDeviceID = iDeviceID;
	m_hMAGMAStream = (Void*)hMAGMAQueue;
}
Void MAGMAStream::Destroy()
{
	if ( m_hMAGMAStream == NULL )
		return;
	
	magma_queue_t hMAGMAQueue = (magma_queue_t)m_hMAGMAStream;
	magma_queue_destroy( hMAGMAQueue );
	
	m_iDeviceID = 0;
	m_hMAGMAStream = NULL;
}

CUDADeviceID MAGMAStream::GetDevice() const
{
	Assert( m_hMAGMAStream != NULL );
	
	magma_queue_t hMAGMAQueue = (magma_queue_t)m_hMAGMAStream;
	magma_int_t iDeviceID = magma_queue_get_device( hMAGMAQueue );
	
	return (CUDADeviceID)iDeviceID;
}
SizeT MAGMAStream::GetAvailableMemory() const
{
	Assert( m_hMAGMAStream != NULL );
	
	magma_queue_t hMAGMAQueue = (magma_queue_t)m_hMAGMAStream;
	SizeT iSize = magma_mem_size( hMAGMAQueue );
	
	return iSize;
}

Void MAGMAStream::Synchronize()
{
	Assert( m_hMAGMAStream != NULL );
	
	magma_queue_t hMAGMAQueue = (magma_queue_t)m_hMAGMAStream;
	magma_queue_sync( hMAGMAQueue );
}

Void MAGMAStream::CopyVector( MAGMAMemory * pDest, UInt iDestStride, UInt iDestOffset,
							  const MAGMAMemory * pSrc, UInt iSrcStride, UInt iSrcOffset,
							  UInt iElementSize, UInt iElementCount,
							  Bool bAsynchronous )
{
	// Ensure all states are OK
	Assert( m_hMAGMAStream != NULL );
	Assert( pDest->IsAllocated() && pSrc->IsAllocated() );
	
	// Ensure iElementSize is valid
	Assert( iElementSize > 0 );
	Assert( iElementSize <= iSrcStride && iElementSize <= iDestStride );
	
	// Ensure Offsets are valid
	Assert( iSrcOffset * iSrcStride < pSrc->GetSize() );
	Assert( iDestOffset * iDestStride < pDest->GetSize() );
	
	// Nothing to do case
	if ( iElementCount == 0 )
		return;
	
	// Ensure iElementCount is valid
	Assert( (iSrcOffset + iElementCount) * iSrcStride <= pSrc->GetSize() );
	Assert( (iDestOffset + iElementCount) * iDestStride <= pDest->GetSize() );
	
	// Perform the copy
	magma_queue_t hMAGMAQueue = (magma_queue_t)m_hMAGMAStream;
	
	Byte * pDestData = (Byte*)( pDest->GetPointer() );
	pDestData += iDestOffset * iDestStride;
	
	const Byte * pSrcData = (const Byte *)( pSrc->GetPointer() );
	pSrcData += iSrcOffset * iSrcStride;
	
	if ( pDest->IsDeviceMemory() ) {
		if ( pSrc->IsDeviceMemory() ) {
			// Device to Device
			if ( bAsynchronous ) {
				magma_copyvector_async(
					iElementCount, iElementSize,
					(const Void *)pSrcData, iSrcStride,
					(Void*)pDestData, iDestStride,
					hMAGMAQueue
				);
			} else {
				magma_copyvector(
					iElementCount, iElementSize,
					(const Void *)pSrcData, iSrcStride,
					(Void*)pDestData, iDestStride,
					hMAGMAQueue
				);
			}
		} else {
			// Host to Device
			if ( bAsynchronous ) {
				magma_setvector_async(
					iElementCount, iElementSize,
					(const Void *)pSrcData, iSrcStride,
					(Void*)pDestData, iDestStride,
					hMAGMAQueue
				);
			} else {
				magma_setvector(
					iElementCount, iElementSize,
					(const Void *)pSrcData, iSrcStride,
					(Void*)pDestData, iDestStride,
					hMAGMAQueue
				);
			}
		}
	} else {
		if ( pSrc->IsDeviceMemory() ) {
			// Device to Host
			if ( bAsynchronous ) {
				magma_getvector_async(
					iElementCount, iElementSize,
					(const Void *)pSrcData, iSrcStride,
					(Void*)pDestData, iDestStride,
					hMAGMAQueue
				);
			} else {
				magma_getvector(
					iElementCount, iElementSize,
					(const Void *)pSrcData, iSrcStride,
					(Void*)pDestData, iDestStride,
					hMAGMAQueue
				);
			}
		} else {
			// Host to Host (regular copy)
			for( UInt i = 0; i < iElementCount; ++i ) {
				for( UInt j = 0; j < iElementSize; ++j )
					pDestData[j] = pSrcData[j];
				pDestData += iDestStride;
				pSrcData += iSrcStride;
			}
		}
	}
}
Void MAGMAStream::CopyMatrix( MAGMAMemory * pDest, UInt iDestRowCount, UInt iDestRow, UInt iDestColumn,
							  const MAGMAMemory * pSrc, UInt iSrcRowCount, UInt iSrcRow, UInt iSrcColumn,
							  UInt iElementSize, UInt iRowCount, UInt iColumnCount,
							  Bool bAsynchronous )
{
	// Ensure all states are OK
	Assert( m_hMAGMAStream != NULL );
	Assert( pDest->IsAllocated() && pSrc->IsAllocated() );
	
	// Ensure iElementSize is valid
	Assert( iElementSize > 0 );
	UInt iSrcStride = iElementSize;
	UInt iDestStride = iElementSize;
	UInt iSrcPitch = iSrcStride * iSrcRowCount;
	UInt iDestPitch = iDestStride * iDestRowCount;
	
	// Ensure Offsets are valid
	Assert( iSrcRow * iSrcStride < iSrcPitch );
	Assert( iDestRow * iDestStride < iDestPitch );
	Assert( iSrcColumn * iSrcPitch < pSrc->GetSize() );
	Assert( iDestColumn * iDestPitch < pDest->GetSize() );
	
	// Nothing to do case
	if ( iRowCount == 0 || iColumnCount == 0 )
		return;
	
	// Ensure iRowCount is valid
	Assert( (iSrcRow + iRowCount) * iSrcStride <= iSrcPitch );
	Assert( (iDestRow + iRowCount) * iDestStride <= iDestPitch );
	
	// Ensure iColumnCount is valid
	Assert( (iSrcColumn + iColumnCount) * iSrcPitch <= pSrc->GetSize() );
	Assert( (iDestColumn + iColumnCount) * iDestPitch <= pDest->GetSize() );
	
	// Perform the copy
	magma_queue_t hMAGMAQueue = (magma_queue_t)m_hMAGMAStream;
	
	Byte * pDestData = (Byte*)( pDest->GetPointer() );
	pDestData += iDestRow * iDestStride + iDestColumn * iDestPitch;
	
	const Byte * pSrcData = (const Byte *)( pSrc->GetPointer() );
	pSrcData += iSrcRow * iSrcStride + iSrcColumn * iSrcPitch;

	if ( pDest->IsDeviceMemory() ) {
		if ( pSrc->IsDeviceMemory() ) {
			// Device to Device
			if ( bAsynchronous ) {
				magma_copymatrix_async(
					iRowCount, iColumnCount, iElementSize,
					(const Void *)pSrcData, iSrcRowCount,
					(Void*)pDestData, iDestRowCount,
					hMAGMAQueue
				);
			} else {
				magma_copymatrix(
					iRowCount, iColumnCount, iElementSize,
					(const Void *)pSrcData, iSrcRowCount,
					(Void*)pDestData, iDestRowCount,
					hMAGMAQueue
				);
			}
		} else {
			// Host to Device
			if ( bAsynchronous ) {
				magma_setmatrix_async(
					iRowCount, iColumnCount, iElementSize,
					(const Void *)pSrcData, iSrcRowCount,
					(Void*)pDestData, iDestRowCount,
					hMAGMAQueue
				);
			} else {
				magma_setmatrix(
					iRowCount, iColumnCount, iElementSize,
					(const Void *)pSrcData, iSrcRowCount,
					(Void*)pDestData, iDestRowCount,
					hMAGMAQueue
				);
			}
		}
	} else {
		if ( pSrc->IsDeviceMemory() ) {
			// Device to Host
			if ( bAsynchronous ) {
				magma_getmatrix_async(
					iRowCount, iColumnCount, iElementSize,
					(const Void *)pSrcData, iSrcRowCount,
					(Void*)pDestData, iDestRowCount,
					hMAGMAQueue
				);
			} else {
				magma_getmatrix(
					iRowCount, iColumnCount, iElementSize,
					(const Void *)pSrcData, iSrcRowCount,
					(Void*)pDestData, iDestRowCount,
					hMAGMAQueue
				);
			}
		} else {
			// Host to Host (regular copy)
			for( UInt iCol = 0; iCol < iColumnCount; ++iCol ) {
				Byte * pDestCol = pDestData;
				const Byte * pSrcCol = pSrcData;
				for( UInt iRow = 0; iRow < iRowCount; ++iRow ) {
					for( UInt j = 0; j < iElementSize; ++j )
						pDestCol[j] = pSrcCol[j];
					pDestCol += iDestStride;
					pSrcCol += iSrcStride;
				}
				pDestData += iDestPitch;
				pSrcData += iSrcPitch;
			}
		}
	}
}


