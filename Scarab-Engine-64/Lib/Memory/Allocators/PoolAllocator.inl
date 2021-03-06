/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/Allocators/PoolAllocator.inl
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
// PoolAllocator implementation
inline AllocatorType PoolAllocator::GetType() const {
    return ALLOCATOR_POOL;
}
inline Bool PoolAllocator::CheckAddressRange( Void * pMemory ) const {
    Byte * pAddress = (Byte*)pMemory;
    if ( pAddress < m_pBuffer )
        return false;
    if ( pAddress >= m_pBuffer + (m_iTotalChunks * m_iChunkSize) )
        return false;
    return true;
}
inline SizeT PoolAllocator::GetBlockSize( Void * pMemory ) const {
    Assert( CheckAddressRange(pMemory) );
    return m_iChunkSize;
}

inline SizeT PoolAllocator::ChunkSize() const {
    return m_iChunkSize;
}
inline UInt PoolAllocator::ChunkCount() const {
    return m_iChunkCount;
}
inline UInt PoolAllocator::ChunkTotal() const {
    return m_iTotalChunks;
}
