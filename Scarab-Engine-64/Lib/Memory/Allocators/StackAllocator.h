/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/Allocators/StackAllocator.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Very simple, but efficient, LIFO allocator.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_LIB_MEMORY_ALLOCATORS_STACKALLOCATOR_H
#define SCARAB_LIB_MEMORY_ALLOCATORS_STACKALLOCATOR_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "MemoryAllocator.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Reporting System
#define STACKREPORT_LOGFILE    TEXT("Logs/Memory/StackReports.log")
#define STACKREPORT_MAX_FRAMES 1024

typedef struct _stack_report : public AllocatorReport
{
    Void * pBaseAddress;
    SizeT iTotalSize;
    SizeT iAllocatedSize;
    SizeT iFreeSize;
    UInt iFrameLevel;
    UInt iFrameCount;
    Byte ** arrFrameBases; // size = iFrameCount
    SizeT * arrFrameSizes; // size = iFrameCount
} StackReport;

///////////////////////////////////////////////////////////////////////////////
// The StackAllocator class
class StackAllocator : public MemoryAllocator
{
public:
	StackAllocator( const MemoryContext * pParentContext, MemoryAllocatorID iAllocatorID, const GChar * strAllocatorName, SizeT iStackSize );
	virtual ~StackAllocator();

    // Getters
    inline virtual AllocatorType GetType() const;
    inline virtual Bool CheckAddressRange( Void * pMemory ) const;
    inline virtual SizeT GetBlockSize( Void * pMemory ) const;

    inline UInt FrameLevel() const;
    inline UInt FrameSize() const;
    inline SizeT AllocatedSize() const;
    inline SizeT TotalSize() const;

    // Alloc/Free interface
    virtual Void * Allocate( SizeT iSize );
    virtual Void Free( Void * pMemory );

    Void Free( SizeT iSize );
    Void Free();

    // Frame management
    UInt BeginFrame();
    Void EndFrame();
    Void UnrollFrames( UInt iTargetFrame );

    // Reporting
    virtual Void GenerateReport( AllocatorReport * outReport ) const;
    virtual Void LogReport( const AllocatorReport * pReport ) const;

private:
	Byte * m_pBuffer;
	SizeT m_iTotalSize;
    Byte * m_pStackTop;
    Byte * m_pFrameBase;
    UInt m_iFrameLevel;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "StackAllocator.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MEMORY_ALLOCATORS_STACKALLOCATOR_H
