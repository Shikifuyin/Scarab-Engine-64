/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/Allocators/BreakAllocator.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : The old-school dirty-trashy assembly memory block-breaker ...
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : Implementation is disabled ! Seriously, don't use it !
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_LIB_MEMORY_ALLOCATORS_BREAKALLOCATOR_H
#define SCARAB_LIB_MEMORY_ALLOCATORS_BREAKALLOCATOR_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "MemoryAllocator.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
typedef struct _block_break_header
{
    struct _block_break_header * pPrev;
    struct _block_break_header * pNext;
    UIntPtr iBreak;
} BlockBreakHeader;

// Reporting System
#define BREAKREPORT_LOGFILE TEXT("Logs/Memory/BreakReports.log")

typedef struct _break_report : public AllocatorReport
{
    Byte * pBaseAddress;
    SizeT iTotalSize;
    SizeT iBlockSize;
    UInt iAllocatedBlocks;
    UInt iFreeBlocks;
    UInt iCurrentBlock;
    UIntPtr * arrBreakList; // size = iAllocatedBlocks
} BreakReport;

///////////////////////////////////////////////////////////////////////////////
// The BreakAllocator class
class BreakAllocator : public MemoryAllocator
{
public:
	BreakAllocator( const MemoryContext * pParentContext, MemoryAllocatorID iAllocatorID, const GChar * strAllocatorName, SizeT iRangeSize, SizeT iBlockSize );
	virtual ~BreakAllocator();

    // Getters
    inline virtual AllocatorType GetType() const;
    inline virtual Bool CheckAddressRange( Void * pMemory ) const;
    inline virtual SizeT GetBlockSize( Void * pMemory ) const;

    // Alloc/Free interface
    virtual Void * Allocate( SizeT iSize );
    virtual Void Free( Void * pMemory );

        // Chunk-Break allocations
    //Void ChunkRequest( UInt iMinFreeSize, MemoryContextID idContext = INVALID_OFFSET );
    //Void ChunkRelease( MemoryContextID idContext = INVALID_OFFSET );
    //Void ChunkFirst( MemoryContextID idContext = INVALID_OFFSET );
    //Void ChunkPrev( MemoryContextID idContext = INVALID_OFFSET );
    //Void ChunkNext( MemoryContextID idContext = INVALID_OFFSET );
    //Void ChunkSelect( UInt iHistoryIndex, MemoryContextID idContext = INVALID_OFFSET );
    //Byte * ChunkBreak( UInt iSize, MemoryContextID idContext = INVALID_OFFSET );
    //Void ChunkUnbreak( Byte * pMemory, MemoryContextID idContext = INVALID_OFFSET );

    // Reporting System
    virtual Void GenerateReport( AllocatorReport * outReport ) const;
    virtual Void LogReport( const AllocatorReport * pReport ) const;

private:
    Byte * m_pMemoryRange;
    SizeT m_iRangeSize;
    SizeT m_iBlockSize;
    BlockBreakHeader * m_pFirstBlock;
    BlockBreakHeader * m_pCurrentBlock;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "BreakAllocator.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MEMORY_ALLOCATORS_BREAKALLOCATOR_H
