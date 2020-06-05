/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/Allocators/BreakAllocator.inl
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
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// BreakAllocator implementation
inline AllocatorType BreakAllocator::GetType() const {
    return ALLOCATOR_BREAK;
}
inline UInt BreakAllocator::GetBlockSize( Byte * pMemory ) const {
    Assert( m_pMemoryRange != NULL );
    Assert( pMemory >= m_pMemoryRange );
    Assert( pMemory < (m_pMemoryRange + m_iRangeSize) );

    ////////////////////////////////////
    return 0;
}

//inline UInt BreakAllocator::_ChunkBlockSize( Byte * pMemory, MemoryContextID idContext ) {
//    if (idContext == INVALID_OFFSET)
//        idContext = m_iCurrentContext;
//
//    Assert(idContext != INVALID_OFFSET);
//    MemoryContext * pContext = _GetContext(idContext);
//    Assert(pContext->bHasLocation[MEMORY_CHUNK]);
//
//    Assert( pContext->pCurrentChunk != NULL );
//    Assert( pMemory >= ( (Byte*)(pContext->pCurrentChunk) + sizeof(ChunkBreakHeader) ) );
//    Assert( pMemory < ( (Byte*)(pContext->pCurrentChunk) + pContext->pCurrentChunk->iBreak ) );
//    return ( pContext->pCurrentChunk->iBreak - ( pMemory - (Byte*)(pContext->pCurrentChunk) ) );
//}
