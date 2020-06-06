/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/Allocators/MemoryAllocator.cpp
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
// Includes
#include "MemoryAllocator.h"

///////////////////////////////////////////////////////////////////////////////
// MemoryAllocator implementation
MemoryAllocator::MemoryAllocator( const MemoryContext * pParentContext, MemoryAllocatorID iAllocatorID, const GChar * strAllocatorName )
{
    m_pParentContext = pParentContext;

    m_iAllocatorID = iAllocatorID;
    StringFn->NCopy( m_strAllocatorName, strAllocatorName, MEMORY_MAX_NAMELENGTH - 1 );

    m_bTracing = false;
    m_iTraceCount = 0;
}
MemoryAllocator::~MemoryAllocator()
{
    // nothing to do
}

///////////////////////////////////////////////////////////////////////////////

Void MemoryAllocator::_Tracing_Record( Void * pAddress, SizeT iSize, Bool bIsAlloc, Bool bIsArray, const GChar * strFileName, UInt iFileLine )
{
    if ( m_iTraceCount >= MEMORYTRACE_MAX_RECORDS )
        _Tracing_LogAndFlush();

    MemoryTraceRecord * pTrace = ( m_arrTraceRecords + m_iTraceCount );

    pTrace->iContextID = m_pParentContext->iContextID;
    pTrace->strContextName = m_pParentContext->strName;
    pTrace->iAllocatorID = m_iAllocatorID;
    pTrace->strAllocatorName = m_strAllocatorName;
    pTrace->iAllocatorType = GetType();
    pTrace->pAddress = pAddress;
    pTrace->iSize = iSize;
    pTrace->bIsAlloc = bIsAlloc;
    pTrace->bIsArray = bIsArray;
    pTrace->strFileName = strFileName;
    pTrace->iFileLine = iFileLine;

    ++m_iTraceCount;
}
Void MemoryAllocator::_Tracing_LogAndFlush()
{
    // Log
    HFile logFile = SystemFn->OpenFile( MEMORYTRACE_LOGFILE, FILE_WRITE );
    Assert( logFile.IsValid() );
    Bool bOk = logFile.Seek( FILE_SEEK_END, 0 );
    Assert( bOk );

    static const GChar * s_arrAllocatorTypeNames[5] = {
        TEXT("Resident"),
        TEXT("Break"),
        TEXT("Stack"),
        TEXT("Pool"),
        TEXT("Heap")
    };
    for( UInt i = 0; i < m_iTraceCount; ++i ) {
        const MemoryTraceRecord * pTrace = ( m_arrTraceRecords + i );

        ErrorFn->Log( logFile, TEXT("Trace Record : %s"), (pTrace->bIsAlloc) ? TEXT("Allocate") : TEXT("Free") );

        ErrorFn->Log( logFile, TEXT("\n => Context ID     : %ud"), pTrace->iContextID );
        ErrorFn->Log( logFile, TEXT("\n => Context Name   : %s"), pTrace->strContextName );
        ErrorFn->Log( logFile, TEXT("\n => Allocator ID   : %ud"), pTrace->iAllocatorID );
        ErrorFn->Log( logFile, TEXT("\n => Allocator Name : %s"), pTrace->strAllocatorName );
        ErrorFn->Log( logFile, TEXT("\n => Allocator Type : %s"), s_arrAllocatorTypeNames[pTrace->iAllocatorType] );
        ErrorFn->Log( logFile, TEXT("\n => Block Address  : %u8x"), (UIntPtr)(pTrace->pAddress) );
        ErrorFn->Log( logFile, TEXT("\n => Block Size     : %ud"), pTrace->iSize );
        ErrorFn->Log( logFile, TEXT("\n => Block Is Array : %s"), (pTrace->bIsArray) ? TEXT("Yes") : TEXT("No") );
        ErrorFn->Log( logFile, TEXT("\n => File Name      : %s"), pTrace->strFileName );
        ErrorFn->Log( logFile, TEXT("\n => Line Number    : %ud"), pTrace->iFileLine );

        ErrorFn->Log( logFile, TEXT("\n\n") );
    }

    logFile.Close();

    // Flush
    m_iTraceCount = 0;
}
