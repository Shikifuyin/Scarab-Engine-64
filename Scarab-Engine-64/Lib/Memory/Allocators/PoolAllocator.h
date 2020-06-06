/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/Allocators/PoolAllocator.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Very simple, but very efficient, fixed-size allocator.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : Beware !!! Here is a tricky & subtle thing :
//              If we use a pool allocator with a stack-fashion allocation
//              pattern, we are guaranted to keep our blocks contiguous because
//              the free list acts symmetrically with head add/remove ...
//              This is of course the optimal case for page-growing arrays !
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_LIB_MEMORY_ALLOCATORS_POOLALLOCATOR_H
#define SCARAB_LIB_MEMORY_ALLOCATORS_POOLALLOCATOR_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "MemoryAllocator.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Reporting System
#define POOLREPORT_LOGFILE      TEXT("Logs/Memory/PoolReports.log")
#define POOLREPORT_MAX_FREELIST 1024

typedef struct _pool_report : public AllocatorReport
{
    Void * pBaseAddress;
    SizeT iTotalSize;
    SizeT iChunkSize;
    UInt iTotalChunks;
    UInt iAllocatedChunks;
    UInt iFreeChunks;
    UInt * arrFreeChunksList; // size = iFreeChunks
} PoolReport;

///////////////////////////////////////////////////////////////////////////////
// The PoolAllocator class
class PoolAllocator : public MemoryAllocator
{
public:
	PoolAllocator( const MemoryContext * pParentContext, MemoryAllocatorID iAllocatorID, const GChar * strAllocatorName, SizeT iChunkSize, UInt iTotalChunks );
	virtual ~PoolAllocator();

    // Getters
    inline virtual AllocatorType GetType() const;
    inline virtual Bool CheckAddressRange( Void * pMemory ) const;
    inline virtual SizeT GetBlockSize( Void * pMemory ) const;

    inline SizeT ChunkSize() const;
	inline UInt ChunkCount() const;
    inline UInt ChunkTotal() const;

    // Alloc/Free interface
    virtual Void * Allocate( SizeT );
	virtual Void Free( Void * pMemory );

    // Reporting
    virtual Void GenerateReport( AllocatorReport * outReport ) const;
    virtual Void LogReport( const AllocatorReport * pReport ) const;

private:
	Byte * m_pBuffer;
    SizeT m_iChunkSize;
	UInt m_iTotalChunks;
	UInt m_iNextFree;
    UInt m_iChunkCount;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "PoolAllocator.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MEMORY_ALLOCATORS_POOLALLOCATOR_H
