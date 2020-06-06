/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/Allocators/MemoryAllocator.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Base interface for allocators to comply with the manager.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_LIB_MEMORY_ALLOCATORS_MEMORYALLOCATOR_H
#define SCARAB_LIB_MEMORY_ALLOCATORS_MEMORYALLOCATOR_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../../Error/ErrorManager.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define MEMORY_MAX_NAMELENGTH 128

#define MEMORY_CONTEXT_MAX_ALLOCATORS 256

// Allocator Types
enum AllocatorType {
    ALLOCATOR_RESIDENT = 0, // Reserved for internal use
    ALLOCATOR_BREAK,        // Assembler-Like Allocator, completely unsafe !
    ALLOCATOR_STACK,        // Fixed Order Allocator, constant time.
    ALLOCATOR_POOL,         // Fixed Size Allocator, constant time.
    ALLOCATOR_HEAP          // Usual AVLTree Implementation, logarithmic in worst case.
    // Add new child-classes an ID here ...
};

class MemoryAllocator;

// Memory Context System
typedef UInt MemoryContextID;
typedef UInt MemoryAllocatorID;

typedef struct _memory_context
{
    MemoryContextID iContextID;
    GChar strName[MEMORY_MAX_NAMELENGTH];

    Void * pResidentMemory;
    SizeT iResidentSize;

    UInt iAllocatorCount;
    UInt iNextFreeAllocator;
    MemoryAllocator * arrAllocators[MEMORY_CONTEXT_MAX_ALLOCATORS];
} MemoryContext;

// Reporting System
typedef struct _allocator_report
{
    MemoryContextID iContextID;
    const GChar * strContextName;
    MemoryAllocatorID iAllocatorID;
    const GChar * strAllocatorName;
    // specific data follows ...
} AllocatorReport;

// Tracing system
#define MEMORYTRACE_LOGFILE     TEXT("Logs/Memory/Traces.log")
#define MEMORYTRACE_MAX_RECORDS 1024

typedef struct _memory_trace_record
{
    MemoryContextID iContextID;
    const GChar * strContextName;
    MemoryAllocatorID iAllocatorID;
    const GChar * strAllocatorName;
    AllocatorType iAllocatorType;
    Void * pAddress;           // Address of memory block
    SizeT iSize;               // Size of memory block
    Bool bIsAlloc;             // Allocate or Free
    Bool bIsArray;             // scalar or vector allocation
    const GChar * strFileName; // File where the trace was initiated
    UInt iFileLine;            // Line number in this file
} MemoryTraceRecord;

///////////////////////////////////////////////////////////////////////////////
// The MemoryAllocator class
class MemoryAllocator
{
protected:
    MemoryAllocator( const MemoryContext * pParentContext, MemoryAllocatorID iAllocatorID, const GChar * strAllocatorName );
public:
	virtual ~MemoryAllocator();

    // In-Place Allocation
    inline Void * operator new( SizeT, Void * pAddress );
    inline Void operator delete( Void * );

    // Getters
    inline const MemoryContext * GetParentContext() const;
    inline MemoryAllocatorID GetAllocatorID() const;
    inline const GChar * GetAllocatorName() const;

    virtual AllocatorType GetType() const = 0;
    virtual Bool CheckAddressRange( Void * pMemory ) const = 0;
    virtual SizeT GetBlockSize( Void * pMemory ) const = 0;

    // Alloc/Free interface
    virtual Void * Allocate( SizeT iSize ) = 0;
    virtual Void Free( Void * pMemory ) = 0;

    // Reporting System
    virtual Void GenerateReport( AllocatorReport * outReport ) const = 0;
    virtual Void LogReport( const AllocatorReport * pReport ) const = 0;

    // Tracing System
    inline Bool IsTracing() const;

    inline Void TraceStart();
    inline Void TraceStop();

    inline UInt TraceCount() const;
    inline const MemoryTraceRecord * TracePick( UInt iIndex ) const;

    inline Void TraceFlush();

protected:
    const MemoryContext * m_pParentContext;

    MemoryAllocatorID m_iAllocatorID;
    GChar m_strAllocatorName[MEMORY_MAX_NAMELENGTH];

    // Tracing System
    friend class MemoryManager;
    Void _Tracing_Record( Void * pAddress, SizeT iSize, Bool bIsAlloc, Bool bIsArray, const GChar * strFileName, UInt iFileLine );
    Void _Tracing_LogAndFlush();

    Bool m_bTracing;
    UInt m_iTraceCount;
    MemoryTraceRecord m_arrTraceRecords[MEMORYTRACE_MAX_RECORDS];
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "MemoryAllocator.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MEMORY_ALLOCATORS_MEMORYALLOCATOR_H
