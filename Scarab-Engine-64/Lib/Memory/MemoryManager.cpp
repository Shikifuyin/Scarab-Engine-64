/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/MemoryManager.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : The VERY important and fine-tuned memory manager ...
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
#include "MemoryManager.h"

/////////////////////////////////////////////////////////////////////////////////
// We do some shady things in here ... Those are all intentional !
#pragma warning(disable:4291) // no matching delete operator for MemoryAllocator
#pragma warning(disable:4302) // type cast truncation (pointer to UInt)
#pragma warning(disable:4311) // type cast pointer truncation (pointer to UInt)
#pragma warning(disable:4312) // type cast to greater size (UInt to pointer)

/////////////////////////////////////////////////////////////////////////////////
// MemoryManager implementation
MemoryManager * MemoryManager::sm_pInstance = NULL;

MemoryManager::MemoryManager():
    m_hBreakFactory( NULL, INVALID_OFFSET, TEXT("_ManagerFactory_Break_"), sizeof(BreakAllocator), MEMORY_MAX_ALLOCATORS ),
    m_hStackFactory( NULL, INVALID_OFFSET, TEXT("_ManagerFactory_Stack_"), sizeof(StackAllocator), MEMORY_MAX_ALLOCATORS ),
    m_hPoolFactory( NULL, INVALID_OFFSET, TEXT("_ManagerFactory_Pool_"), sizeof(PoolAllocator), MEMORY_MAX_ALLOCATORS ),
    m_hHeapFactory( NULL, INVALID_OFFSET, TEXT("_ManagerFactory_Heap_"), sizeof(HeapAllocator), MEMORY_MAX_ALLOCATORS )
{
    // Init shared context
    _MemoryContext_Initialize( &m_hSharedContext, MEMORY_CONTEXT_SHARED, TEXT("Shared"), 0 );
    m_iSharedScratchID = _MemoryAllocator_Create( MEMORY_CONTEXT_SHARED, ALLOCATOR_HEAP, TEXT("Scratch"), MEMORY_CONTEXT_SHARED_SCRATCH_SIZE, 0 );

    // Setup Context Array
    m_iContextCount = 0;
    m_iNextFreeContext = 0;
    for( UInt i = 0; i < MEMORY_MAX_CONTEXTS; ++i ) {
        m_arrContexts[i].iContextID = i + 1; // Free ID list links
        m_arrContexts[i].strName[0] = NULLBYTE;
        m_arrContexts[i].iResidentSize = 0;
        m_arrContexts[i].pResidentMemory = NULL;
        m_arrContexts[i].iAllocatorCount = 0;
        for ( UInt j = 0; j < MEMORY_CONTEXT_MAX_ALLOCATORS; ++j )
            m_arrContexts[i].arrAllocators[j] = NULL;
    }
    m_arrContexts[MEMORY_MAX_CONTEXTS - 1].iContextID = INVALID_OFFSET; // End of free ID list
}
MemoryManager::~MemoryManager()
{
    // All contexts should have been cleared at this point !
    Assert( m_iContextCount == 0 );

    // All shared allocators should have been cleared at this point !
    Assert( m_hSharedContext.iAllocatorCount == 1 );

    // Cleanup shared context
    _MemoryAllocator_Destroy( MEMORY_CONTEXT_SHARED, m_iSharedScratchID );
    _MemoryContext_Cleanup( &m_hSharedContext );
}

Void * MemoryManager::Allocate( SizeT iSize, Bool bIsArray, const GChar * strFile, UInt iLine, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID )
{
    MemoryAllocator * pAllocator = _MemoryAllocator_Get( iContextID, iAllocatorID );

    Void * pMemory = pAllocator->Allocate( iSize );
    Assert( pMemory != NULL );

    if ( pAllocator->IsTracing() )
        pAllocator->_Tracing_Record( pMemory, iSize, true, bIsArray, strFile, iLine );

    return pMemory;
}
Void MemoryManager::Free( Void * pMemory, Bool bIsArray, const GChar * strFile, UInt iLine, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID )
{
    MemoryAllocator * pAllocator = _MemoryAllocator_Get( iContextID, iAllocatorID );
    Assert( pAllocator->CheckAddressRange(pMemory) );

    if ( pAllocator->IsTracing() ) {
        SizeT iSize = pAllocator->GetBlockSize( pMemory );
        pAllocator->_Tracing_Record( pMemory, iSize, false, bIsArray, strFile, iLine );
    }

    pAllocator->Free( pMemory );
}

/////////////////////////////////////////////////////////////////////////////////

MemoryContext * MemoryManager::_MemoryContext_Get( MemoryContextID iContextID )
{
    if ( iContextID == MEMORY_CONTEXT_SHARED )
        return &m_hSharedContext;
    else {
        Assert( iContextID < MEMORY_MAX_CONTEXTS );
        Assert( m_arrContexts[iContextID].iContextID == iContextID );
        return ( m_arrContexts + iContextID );
    }
}
MemoryContextID MemoryManager::_MemoryContext_Create( const GChar * strName, SizeT iResidentSize )
{
    Assert( m_iContextCount < MEMORY_MAX_CONTEXTS );
    
    // Allocate ID
    MemoryContextID iContextID = m_iNextFreeContext;
    m_iNextFreeContext = m_arrContexts[iContextID].iContextID;

    // Initialize Context
    _MemoryContext_Initialize( m_arrContexts + iContextID, iContextID, strName, iResidentSize );

    // Done
    ++m_iContextCount;
    return iContextID;
}
Void MemoryManager::_MemoryContext_Destroy( MemoryContextID iContextID )
{
    Assert( iContextID < MEMORY_MAX_CONTEXTS ); // Did you try to destroy shared contex ?!?

    // Cleanup Context
    _MemoryContext_Cleanup( m_arrContexts + iContextID );

    // Free ID
    m_arrContexts[iContextID].iContextID = m_iNextFreeContext;
    m_iNextFreeContext = iContextID;

    // Done
    --m_iContextCount;
}

Void MemoryManager::_MemoryContext_Initialize( MemoryContext * pContext, MemoryContextID iContextID, const GChar * strName, SizeT iResidentSize )
{
    // Initialize
    pContext->iContextID = iContextID;
    StringFn->NCopy( pContext->strName, strName, MEMORY_MAX_NAMELENGTH - 1 );

    // Allocate Resident Memory
    pContext->iResidentSize = iResidentSize;
    pContext->pResidentMemory = NULL;
    if ( iResidentSize > 0 )
        pContext->pResidentMemory = (Byte*)( SystemFn->MemAlloc(iResidentSize) );

    // Setup Allocator Array
    pContext->iAllocatorCount = 0;
    pContext->iNextFreeAllocator = 0;
    for ( UInt i = 0; i < MEMORY_CONTEXT_MAX_ALLOCATORS; ++i )
        pContext->arrAllocators[i] = (MemoryAllocator*)( i + 1 ); // Free ID list links
    pContext->arrAllocators[MEMORY_CONTEXT_MAX_ALLOCATORS - 1] = (MemoryAllocator*)( INVALID_OFFSET ); // End of free ID list
}
Void MemoryManager::_MemoryContext_Cleanup( MemoryContext * pContext )
{
    // All allocators should have been destroyed
    Assert( pContext->iAllocatorCount == 0 );

    // Free Resident Memory
    if ( pContext->iResidentSize > 0 )
        SystemFn->MemFree( pContext->pResidentMemory );
    pContext->iResidentSize = 0;
    pContext->pResidentMemory = NULL;

    // Cleanup
    pContext->iContextID = INVALID_OFFSET;
    pContext->strName[0] = NULLBYTE;
}

MemoryAllocator * MemoryManager::_MemoryAllocator_Get( MemoryContextID iContextID, MemoryAllocatorID iAllocatorID )
{
    MemoryContext * pContext = _MemoryContext_Get( iContextID );

    Assert( iAllocatorID < MEMORY_CONTEXT_MAX_ALLOCATORS );
    Assert( pContext->arrAllocators[iAllocatorID]->GetAllocatorID() == iAllocatorID );
    return pContext->arrAllocators[iAllocatorID];
}
MemoryAllocatorID MemoryManager::_MemoryAllocator_Create( MemoryContextID iContextID, AllocatorType iType, const GChar * strName, SizeT iSize, SizeT iBlockSize )
{
    MemoryContext * pContext = _MemoryContext_Get( iContextID );
    Assert( pContext->iAllocatorCount < MEMORY_CONTEXT_MAX_ALLOCATORS );

    // Allocate ID
    MemoryAllocatorID iAllocatorID = pContext->iNextFreeAllocator;
    pContext->iNextFreeAllocator = (UInt)( pContext->arrAllocators[iAllocatorID] );

    // Initialize Allocator
    switch( iType ) {
        case ALLOCATOR_BREAK: {
            Assert( iBlockSize != 0 );
            Void * pMemory = m_hBreakFactory.Allocate(0);
            BreakAllocator * pBreak = new(pMemory) BreakAllocator( pContext, iAllocatorID, strName, iSize, iBlockSize );
            pContext->arrAllocators[iAllocatorID] = pBreak;
        } break;
        case ALLOCATOR_STACK: {
            Assert( iBlockSize == 0 );
            Void * pMemory = m_hStackFactory.Allocate(0);
            StackAllocator * pStack = new(pMemory) StackAllocator( pContext, iAllocatorID, strName, iSize );
            pContext->arrAllocators[iAllocatorID] = pStack;
        } break;
        case ALLOCATOR_POOL: {
            Assert( iBlockSize != 0 );
            Void * pMemory = m_hPoolFactory.Allocate(0);
            PoolAllocator * pPool = new(pMemory) PoolAllocator( pContext, iAllocatorID, strName, iBlockSize, iSize );
            pContext->arrAllocators[iAllocatorID] = pPool;
        } break;
        case ALLOCATOR_HEAP: {
            Assert( iBlockSize == 0 );
            Void * pMemory = m_hHeapFactory.Allocate(0);
            HeapAllocator * pHeap = new(pMemory) HeapAllocator( pContext, iAllocatorID, strName, iSize );
            pContext->arrAllocators[iAllocatorID] = pHeap;
        } break;
        default: Assert( false ); break;
    }

    // Done
    ++(pContext->iAllocatorCount);
    return iAllocatorID;
}
Void MemoryManager::_MemoryAllocator_Destroy( MemoryContextID iContextID, MemoryAllocatorID iAllocatorID )
{
    MemoryContext * pContext = _MemoryContext_Get( iContextID );
    MemoryAllocator * pAllocator = pContext->arrAllocators[iAllocatorID];

    // Cleanup Allocator
    switch( pAllocator->GetType() ) {
        case ALLOCATOR_BREAK:
            delete pAllocator;
            m_hBreakFactory.Free( (Byte*)pAllocator );
            break;
        case ALLOCATOR_STACK:
            delete pAllocator;
            m_hStackFactory.Free( (Byte*)pAllocator );
            break;
        case ALLOCATOR_POOL:
            delete pAllocator;
            m_hPoolFactory.Free( (Byte*)pAllocator );
            break;
        case ALLOCATOR_HEAP:
            delete pAllocator;
            m_hHeapFactory.Free( (Byte*)pAllocator );
            break;
        default: Assert(false); return;
    }

    // Free ID
    pContext->arrAllocators[iAllocatorID] = (MemoryAllocator*)( pContext->iNextFreeAllocator );
    pContext->iNextFreeAllocator = iAllocatorID;

    // Done
    --(pContext->iAllocatorCount);
}



