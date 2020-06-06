/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/Allocators/StackAllocator.inl
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

///////////////////////////////////////////////////////////////////////////////
// StackAllocator implementation
inline AllocatorType StackAllocator::GetType() const {
    return ALLOCATOR_STACK;
}
inline Bool StackAllocator::CheckAddressRange( Void * pMemory ) const {
    Byte * pAddress = (Byte*)pMemory;
    if ( pAddress < m_pFrameBase )
        return false;
    if ( pAddress >= m_pStackTop )
        return false;
    return true;
}
inline SizeT StackAllocator::GetBlockSize( Void * pMemory ) const {
    Assert( CheckAddressRange(pMemory) );
    return (SizeT)( m_pStackTop - (Byte*)pMemory );
}

inline UInt StackAllocator::FrameLevel() const {
    return m_iFrameLevel;
}
inline UInt StackAllocator::FrameSize() const {
    return (UInt)( m_pStackTop - m_pFrameBase );
}
inline SizeT StackAllocator::AllocatedSize() const {
    return (SizeT)( m_pStackTop - m_pBuffer );
}
inline SizeT StackAllocator::TotalSize() const {
    return m_iTotalSize;
}
