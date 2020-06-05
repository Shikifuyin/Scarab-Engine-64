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
#define STACKREPORT_LOGFILE    TEXT("Logs/Memory/StackReports.log")
#define STACKREPORT_MAX_FRAMES 1024
typedef struct _stack_report : public AllocatorReport
{
    Byte * pBaseAddress;
    UInt iTotalSize;
    UInt iAllocatedSize;
    UInt iFreeSize;
    UInt iFrameLevel;
    UInt iFrameCount;
    Byte ** arrFrameBases; // size = iFrameCount
    UInt * arrFrameSizes;  // size = iFrameCount
} StackReport;

///////////////////////////////////////////////////////////////////////////////
// The StackAllocator class
class StackAllocator : public MemoryAllocator
{
public:
	StackAllocator( UInt iContextID, const GChar * strContextName,
                    UInt iAllocatorID, const GChar * strAllocatorName );
	virtual ~StackAllocator();

    // Deferred initialization
    Void Initialize( UInt iStackSize );
    Void Cleanup();

    // Getters
    inline AllocatorType GetType() const;
    inline UInt GetBlockSize( Byte * pMemory ) const;

    inline UInt FrameLevel() const;
    inline UInt FrameSize() const;
    inline UInt AllocatedSize() const;
    inline UInt TotalSize() const;

    // Alloc/Free interface
    Byte * Allocate( UInt iSize );
    Void Free( UInt iSize );
    Void Free();

    // Frame management
    UInt BeginFrame();
    Void EndFrame();
    Void UnrollFrames( UInt iTargetFrame );

    // Reporting
    Void GenerateReport( AllocatorReport * outReport ) const;
    Void LogReport( const AllocatorReport * pReport ) const;

private:
	Byte * m_pBuffer;
	UInt m_iTotalSize;
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
