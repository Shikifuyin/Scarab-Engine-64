/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/Allocators/PoolAllocator.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Very simple, but efficient, fixed-size allocator.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "PoolAllocator.h"

/////////////////////////////////////////////////////////////////////////////////
// PoolAllocator implementation
PoolAllocator::PoolAllocator( const MemoryContext * pParentContext, MemoryAllocatorID iAllocatorID, const GChar * strAllocatorName, SizeT iChunkSize, UInt iTotalChunks ):
    MemoryAllocator( pParentContext, iAllocatorID, strAllocatorName )
{
    Assert( iChunkSize >= sizeof(UInt) );
    Assert( (iChunkSize & 0x03) == 0 );

    m_pBuffer = (Byte*)( SystemFn->MemAlloc(iTotalChunks * iChunkSize) );
	m_iChunkSize = iChunkSize;
    m_iTotalChunks = iTotalChunks;
	m_iNextFree = 0;
    m_iChunkCount = 0;

    Byte * pChunk = m_pBuffer;
	for( UInt i = 0; i < (m_iTotalChunks - 1); ++i ) {
		*((UInt*)pChunk) = i + 1;
        pChunk += m_iChunkSize;
	}
	*((UInt*)pChunk) = INVALID_OFFSET;
}
PoolAllocator::~PoolAllocator()
{
    SystemFn->MemFree( m_pBuffer );
}

Void * PoolAllocator::Allocate( SizeT )
{
    if ( m_iNextFree == INVALID_OFFSET )
        return NULL;
    Assert( m_iChunkCount < m_iTotalChunks );

	Byte * pChunk = m_pBuffer + (m_iNextFree * m_iChunkSize);
	m_iNextFree = *((UInt*)pChunk);
    ++m_iChunkCount;
	return pChunk;
}
Void PoolAllocator::Free( Void * pMemory )
{
    Assert( pMemory != NULL );

	Byte * pChunk = (Byte*)pMemory;
	*((UInt*)pChunk) = m_iNextFree;
	m_iNextFree = (UInt)( ((SizeT)(pChunk - m_pBuffer)) / m_iChunkSize );
    --m_iChunkCount;
}

Void PoolAllocator::GenerateReport( AllocatorReport * outReport ) const
{
    static UInt s_ScratchMemory1[POOLREPORT_MAX_FREELIST];
    Assert( (m_iTotalChunks - m_iChunkCount) < POOLREPORT_MAX_FREELIST );

    Assert( outReport != NULL );
    PoolReport * outPoolReport = (PoolReport*)outReport;
    outPoolReport->iContextID = m_pParentContext->iContextID;
    outPoolReport->strContextName = m_pParentContext->strName;
    outPoolReport->iAllocatorID = m_iAllocatorID;
    outPoolReport->strAllocatorName = m_strAllocatorName;
    outPoolReport->pBaseAddress = m_pBuffer;
    outPoolReport->iTotalSize = (m_iTotalChunks * m_iChunkSize);
    outPoolReport->iChunkSize = m_iChunkSize;
    outPoolReport->iTotalChunks = m_iTotalChunks;
    outPoolReport->iAllocatedChunks = m_iChunkCount;
    outPoolReport->iFreeChunks = (m_iTotalChunks - m_iChunkCount);
    outPoolReport->arrFreeChunksList = s_ScratchMemory1;

    // Extract FreeChunks List
    UInt iFreeChunk = m_iNextFree;
    UInt i = 0;
    while( iFreeChunk != INVALID_OFFSET ) {
        outPoolReport->arrFreeChunksList[i] = iFreeChunk;
        iFreeChunk = *( (UInt*)( m_pBuffer + (iFreeChunk * m_iChunkSize) ) );
        ++i;
    }
    outPoolReport->arrFreeChunksList[i] = INVALID_OFFSET;
}
Void PoolAllocator::LogReport( const AllocatorReport * pReport ) const
{
    Assert( pReport != NULL );
    const PoolReport * pPoolReport = (const PoolReport*)pReport;

    HFile logFile = SystemFn->OpenFile( POOLREPORT_LOGFILE, FILE_WRITE );
    Assert( logFile.IsValid() );
    Bool bOk = logFile.Seek( FILE_SEEK_END, 0 );
    Assert( bOk );

    ErrorFn->Log( logFile, TEXT("Pool Report :") );

    ErrorFn->Log( logFile, TEXT("\n => Memory Context ID               : %ud"),  pPoolReport->iContextID );
    ErrorFn->Log( logFile, TEXT("\n => Memory Context Name             : %s"),   pPoolReport->strContextName );
    ErrorFn->Log( logFile, TEXT("\n => Memory Allocator ID             : %ud"),  pPoolReport->iAllocatorID );
    ErrorFn->Log( logFile, TEXT("\n => Memory Allocator Name           : %s"),   pPoolReport->strAllocatorName );
    ErrorFn->Log( logFile, TEXT("\n => Base Address                    : %u8x"), (UIntPtr)(pPoolReport->pBaseAddress) );
    ErrorFn->Log( logFile, TEXT("\n => Total size                      : %ud"),  pPoolReport->iTotalSize );
    ErrorFn->Log( logFile, TEXT("\n => Chunk size                      : %ud"),  pPoolReport->iChunkSize );
    ErrorFn->Log( logFile, TEXT("\n => Total chunks                    : %ud"),  pPoolReport->iTotalChunks );
    ErrorFn->Log( logFile, TEXT("\n => Allocated chunks                : %ud"),  pPoolReport->iAllocatedChunks );
    ErrorFn->Log( logFile, TEXT("\n => Free chunks                     : %ud"),  pPoolReport->iFreeChunks );

    ErrorFn->Log( logFile, TEXT("\n => FreeChunks List (Index,Address) :") );
    for( UInt i = 0; i < pPoolReport->iFreeChunks; ++i ) {
        UInt iIndex = pPoolReport->arrFreeChunksList[i];
        Byte * pAddress = ((Byte*)(pPoolReport->pBaseAddress)) + (iIndex * pPoolReport->iChunkSize);
        ErrorFn->Log( logFile, TEXT("\n\t -> (%ud,%u8x)"), iIndex, (UIntPtr)pAddress );
    }

    ErrorFn->Log( logFile, TEXT("\n\n") );

    logFile.Close();
}
