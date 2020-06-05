/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/Allocators/StackAllocator.cpp
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
// Includes
#include "StackAllocator.h"

///////////////////////////////////////////////////////////////////////////////
// StackAllocator implementation
StackAllocator::StackAllocator( UInt iContextID, const GChar * strContextName,
                                UInt iAllocatorID, const GChar * strAllocatorName ):
    MemoryAllocator( iContextID, strContextName, iAllocatorID, strAllocatorName )
{
    m_pBuffer = NULL;
    m_iTotalSize = 0;
    m_pStackTop = NULL;
    m_pFrameBase = NULL;
    m_iFrameLevel = 0;
}
StackAllocator::~StackAllocator()
{
    if ( m_pBuffer != NULL )
        SystemFn->MemFree( m_pBuffer );
}

Void StackAllocator::Initialize( UInt iStackSize )
{
    Assert( m_pBuffer == NULL );

    m_iTotalSize = iStackSize;
    m_pBuffer = (Byte*)( SystemFn->MemAlloc(m_iTotalSize) );

    m_pStackTop = m_pBuffer;
    m_pFrameBase = m_pBuffer;
    m_iFrameLevel = 0;
}
Void StackAllocator::Cleanup()
{
    Assert( m_pBuffer != NULL );
    
    SystemFn->MemFree( m_pBuffer );

    m_pBuffer = NULL;
    m_iTotalSize = 0;
    m_pStackTop = NULL;
    m_pFrameBase = NULL;
    m_iFrameLevel = 0;
}

Byte * StackAllocator::Allocate( UInt iSize )
{
    Assert( m_pBuffer != NULL );
    if ( ((UInt)(m_pStackTop - m_pBuffer)) + iSize > m_iTotalSize )
        return NULL;

    Byte * pPtr = m_pStackTop;
    m_pStackTop += iSize;
    return pPtr;
}
Void StackAllocator::Free( UInt iSize )
{
    Assert( m_pBuffer != NULL );

    if ( iSize > (UInt)(m_pStackTop - m_pFrameBase) )
        m_pStackTop = m_pFrameBase;
    else
        m_pStackTop -= iSize;
}
Void StackAllocator::Free()
{
    Assert( m_pBuffer != NULL );

    m_pStackTop = m_pFrameBase;
}

UInt StackAllocator::BeginFrame()
{
    Assert( m_pBuffer != NULL );
    Assert( (m_pStackTop + sizeof(UIntPtr) - m_pBuffer) < m_iTotalSize );

    *((UIntPtr*)m_pStackTop) = (UIntPtr)m_pFrameBase;
    m_pStackTop += sizeof(UIntPtr);
    m_pFrameBase = m_pStackTop;
    ++m_iFrameLevel;

    return m_iFrameLevel;
}
Void StackAllocator::EndFrame()
{
    Assert( m_pBuffer != NULL );
    Assert( m_iFrameLevel > 0 );

    m_pStackTop = m_pFrameBase;
    m_pStackTop -= sizeof(UIntPtr);
    m_pFrameBase = (Byte*)( *((UIntPtr*)m_pStackTop) );
    --m_iFrameLevel;
}
Void StackAllocator::UnrollFrames( UInt iTargetFrame )
{
    Assert( m_pBuffer != NULL );
    Assert( m_iFrameLevel > iTargetFrame );

    while( m_iFrameLevel > iTargetFrame )
        EndFrame();
}

Void StackAllocator::GenerateReport( AllocatorReport * outReport ) const
{
    static Byte * s_ScratchMemory1[STACKREPORT_MAX_FRAMES];
    static UInt s_ScratchMemory2[STACKREPORT_MAX_FRAMES];
    Assert( m_iFrameLevel < STACKREPORT_MAX_FRAMES );

    Assert( outReport != NULL );
    StackReport * outStackReport = (StackReport*)outReport;
    outStackReport->idMemoryContext = m_idContext;
    outStackReport->strContextName = m_strContextName;
    outStackReport->idMemoryAllocator = m_idAllocator;
    outStackReport->strAllocatorName = m_strAllocatorName;
    outStackReport->pBaseAddress = m_pBuffer;
    outStackReport->iTotalSize = m_iTotalSize;
    outStackReport->iAllocatedSize = (UInt)( m_pStackTop - m_pBuffer );
    outStackReport->iFreeSize = ( m_iTotalSize - outStackReport->iAllocatedSize );
    outStackReport->iFrameLevel = m_iFrameLevel;
    outStackReport->iFrameCount = m_iFrameLevel + 1;
    outStackReport->arrFrameBases = s_ScratchMemory1;
    outStackReport->arrFrameSizes = s_ScratchMemory2;

    // Extract frame layout
    Int iCurFrame = m_iFrameLevel;
    Byte * pCurTop = m_pStackTop;
    Byte * pCurBase = m_pFrameBase;
    while( true ) {
        outStackReport->arrFrameBases[iCurFrame] = pCurBase;
        outStackReport->arrFrameSizes[iCurFrame] = (UInt)( pCurTop - pCurBase );
        if ( iCurFrame == 0 )
            break;
        pCurTop = pCurBase - sizeof(UIntPtr);
        pCurBase = (Byte*)( *((UIntPtr*)pCurTop) );
        --iCurFrame;
    }
}
Void StackAllocator::LogReport( const AllocatorReport * pReport ) const
{
    Assert( pReport != NULL );
    const StackReport * pStackReport = (const StackReport*)pReport;

    HFile logFile = SystemFn->OpenFile( STACKREPORT_LOGFILE, FILE_WRITE );
    Assert( logFile.IsValid() );
    Bool bOk = logFile.Seek( FILE_SEEK_END, 0 );
    Assert( bOk );

    ErrorFn->Log( logFile, TEXT("Stack Report :") ); // Add Timestamps ? maybe usefull later ...

    ErrorFn->Log( logFile, TEXT("\n => Memory Context ID       : %ud"),  pStackReport->idMemoryContext );
    ErrorFn->Log( logFile, TEXT("\n => Memory Context Name     : %s"),   pStackReport->strContextName );
    ErrorFn->Log( logFile, TEXT("\n => Memory Allocator ID     : %ud"),  pStackReport->idMemoryAllocator );
    ErrorFn->Log( logFile, TEXT("\n => Memory Allocator Name   : %s"),   pStackReport->strAllocatorName );
    ErrorFn->Log( logFile, TEXT("\n => Base Address            : %u8x"), pStackReport->pBaseAddress );
    ErrorFn->Log( logFile, TEXT("\n => Total size              : %ud"),  pStackReport->iTotalSize );
    ErrorFn->Log( logFile, TEXT("\n => Allocated size          : %ud"),  pStackReport->iAllocatedSize );
    ErrorFn->Log( logFile, TEXT("\n => Free size               : %ud"),  pStackReport->iFreeSize );
    ErrorFn->Log( logFile, TEXT("\n => Frame Level             : %ud"),  pStackReport->iFrameLevel );
    ErrorFn->Log( logFile, TEXT("\n => Frame Count             : %ud"),  pStackReport->iFrameCount );

    ErrorFn->Log( logFile, TEXT("\n => FrameLayout (Address,Size) :") );
    for( UInt i = 0; i < pStackReport->iFrameCount; ++i )
        ErrorFn->Log( logFile, TEXT(" (%u8x,%ud)"), (UIntPtr)(pStackReport->arrFrameBases[i]), pStackReport->arrFrameSizes[i] );

    ErrorFn->Log( logFile, TEXT("\n\n") );

    logFile.Close();
}
