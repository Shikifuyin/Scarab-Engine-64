/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/Allocators/MemoryAllocator.inl
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

///////////////////////////////////////////////////////////////////////////////
// MemoryAllocator implementation
inline UInt MemoryAllocator::GetContextID() const {
    return m_idContext;
}
inline const GChar * MemoryAllocator::GetContextName() const {
    return m_strContextName;
}
inline UInt MemoryAllocator::GetAllocatorID() const {
    return m_idAllocator;
}
inline const GChar * MemoryAllocator::GetAllocatorName() const {
    return m_strAllocatorName;
}
