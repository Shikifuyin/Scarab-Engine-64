/////////////////////////////////////////////////////////////////////////////////
// File : MAGMAStream.h
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
// Header prelude
#ifndef MATRIXSIMULATION_CUDA_MAGMA_MAGMASTREAM_H
#define MATRIXSIMULATION_CUDA_MAGMA_MAGMASTREAM_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "MAGMAMemory.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The MAGMAStream class
class MAGMAStream
{
public:
	MAGMAStream();
	~MAGMAStream();
	
	// Creation / Destruction
	inline Bool IsCreated() const;
	
	Void Create( CUDADeviceID iDeviceID );
	//Void Create( const CUDAStream & hCUDAStream, ... );
	Void Destroy();
	
	// Getters / Setters
	CUDADeviceID GetDevice() const;
	SizeT GetAvailableMemory() const;
	
	// Synchronization
	Void Synchronize();
	
	// Copy Vector
	// Matrices are assumed to be stored Column-Wise ! Vectors are Columns !
	// - pDest = Destination memory to write to
	// - iDestStride = Increment to next column element in bytes
	// - iDestOffset = Starting element index
	// - pSrc = Source memory to read from
	// - iSrcStride = Increment to next column element in bytes
	// - iSrcOffset = Starting element index
	// - iElementSize = Size of a single element in bytes
	// - iElementCount = Number of elements to copy
	// - bAsynchronous = Set to true to perform a non-blocking asynchronous copy
	//                   Default = false, Stream is synchronized after operation.
	// CopyVector can be used to copy row/column vectors stored inside a matrix.
	Void CopyVector( MAGMAMemory * pDest, UInt iDestStride, UInt iDestOffset,
					 const MAGMAMemory * pSrc, UInt iSrcStride, UInt iSrcOffset,
					 UInt iElementSize, UInt iElementCount,
					 Bool bAsynchronous = false );

	// Copy Matrix
	// Matrices are assumed to be stored Column-Wise ! Vectors are Columns !
	// - pDest = Destination memory to write to
	// - iDestRowCount = Leading dimension, Pitch = iDestRowCount * iElementSize
	// - iDestRow = Starting row index
	// - iDestColumn = Starting column index
	// - pSrc = Source memory to read from
	// - iSrcRowCount = Leading dimension, Pitch = iSrcRowCount * iElementSize
	// - iSrcRow = Starting row index
	// - iSrcColumn = Starting column index
	// - iElementSize = Size of a single element in bytes
	// - iRowCount = Number of rows to copy
	// - iColumnCount = Number of columns to copy
	// - bAsynchronous = Set to true to perform a non-blocking asynchronous copy
	//                   Default = false, Stream is synchronized after operation.
	// CopyMatrix allows to copy arbitrary matrix sub-blocks.
	Void CopyMatrix( MAGMAMemory * pDest, UInt iDestRowCount, UInt iDestRow, UInt iDestColumn,
					 const MAGMAMemory * pSrc, UInt iSrcRowCount, UInt iSrcRow, UInt iSrcColumn,
					 UInt iElementSize, UInt iRowCount, UInt iColumnCount,
					 Bool bAsynchronous = false );
	
private:
	Void * m_hMAGMAStream;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "MAGMAStream.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // MATRIXSIMULATION_CUDA_MAGMA_MAGMASTREAM_H

