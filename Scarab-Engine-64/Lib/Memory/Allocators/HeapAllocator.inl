/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/Allocators/HeapAllocator.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : The "Full-English-Breakfast" binheap-allocator ...
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// HeapAllocator implementation
inline AllocatorType HeapAllocator::GetType() const {
    return ALLOCATOR_HEAP;
}
inline Bool HeapAllocator::CheckAddressRange( Void * pMemory ) const {
    Byte * pAddress = (Byte*)pMemory;
    if ( pMemory < m_pHeapMemory )
        return false;
    if ( pMemory >= m_pHeapMemory + m_iHeapSize )
        return false;
    return true;
}
inline SizeT HeapAllocator::GetBlockSize( Void * pMemory ) const {
    Assert( CheckAddressRange(pMemory) );
    ChunkListNode * pNode = (ChunkListNode*)pMemory;
	ChunkHead * pHead = _Chunk_GetHead( pNode );
    return ( _Chunk_Size(pHead) - ChunkHeadAUSize ) * AlignUnit;
}

/////////////////////////////////////////////////////////////////////////////////

inline AUSize HeapAllocator::_AU_ConvertSize( SizeT iSize ) const {
    if ( iSize < AlignUnit )
        iSize = AlignUnit;
    else {
        UInt iMod = iSize & (AlignUnit - 1);
        if ( iMod > 0 )
            iSize += (AlignUnit - iMod);
    }
    return ( iSize >> AlignUnitShift );
}
inline Bool HeapAllocator::_AU_IsAllocated( AUSize iAUSize ) const {
    return ( (iAUSize & AUSIZE_FREE_MASK) != 0 );
}
inline Bool HeapAllocator::_AU_IsFree( AUSize iAUSize ) const {
    return ( (iAUSize & AUSIZE_FREE_MASK) == 0 );
}
inline AUSize HeapAllocator::_AU_Size( AUSize iAUSize ) const {
    return ( iAUSize & AUSIZE_SIZE_MASK );
}
inline Byte * HeapAllocator::_AU_Next( Byte * pChunk, AUSize nAAU ) const {
    return ( pChunk + (nAAU * AlignUnit) );
}
inline Byte * HeapAllocator::_AU_Prev( Byte * pChunk, AUSize nAAU ) const {
    return ( pChunk - (nAAU * AlignUnit) );
}

inline Void HeapAllocator::_Chunk_MarkAllocated( ChunkHead * pChunk ) const {
    pChunk->thisAUSize |= AUSIZE_FREE_MASK;
}
inline Void HeapAllocator::_Chunk_MarkFree( ChunkHead * pChunk ) const {
    pChunk->thisAUSize &= AUSIZE_SIZE_MASK;
}
inline Bool HeapAllocator::_Chunk_IsAllocated( const ChunkHead * pChunk ) const {
    return _AU_IsAllocated( pChunk->thisAUSize );
}
inline Bool HeapAllocator::_Chunk_IsFree( const ChunkHead * pChunk ) const {
    return _AU_IsFree( pChunk->thisAUSize );
}
inline AUSize HeapAllocator::_Chunk_PrevSize( const ChunkHead * pChunk ) const {
    return _AU_Size( pChunk->prevAUSize );
}
inline AUSize HeapAllocator::_Chunk_Size( const ChunkHead * pChunk ) const {
    return _AU_Size( pChunk->thisAUSize );
}
inline ChunkHeapNode * HeapAllocator::_Chunk_GetHeapNode( ChunkHead * pChunk ) const {
    return (ChunkHeapNode*)_AU_Next( (Byte*)pChunk, ChunkHeadAUSize );
}
inline ChunkListNode * HeapAllocator::_Chunk_GetListNode( ChunkHead * pChunk ) const {
    return (ChunkListNode*)_AU_Next( (Byte*)pChunk, ChunkHeadAUSize );
}
inline ChunkHead * HeapAllocator::_Chunk_GetHead( ChunkHeapNode * pHeapNode ) const {
    return (ChunkHead*)_AU_Prev( (Byte*)pHeapNode, ChunkHeadAUSize );
}
inline ChunkHead * HeapAllocator::_Chunk_GetHead( ChunkListNode * pListNode ) const {
    return (ChunkHead*)_AU_Prev( (Byte*)pListNode, ChunkHeadAUSize );
}
inline Int HeapAllocator::_Compare( const ChunkHead * pLHS, const ChunkHead * pRHS ) const {
    AUSize lhsSize = _Chunk_Size(pLHS);
    AUSize rhsSize = _Chunk_Size(pRHS);
    if (lhsSize < rhsSize) return +1;
    if (lhsSize > rhsSize) return -1;
    return 0;
}

