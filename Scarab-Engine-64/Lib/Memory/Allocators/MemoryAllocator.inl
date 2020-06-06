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
inline Void * MemoryAllocator::operator new( SizeT, Void * pAddress ) {
    return pAddress;
}
inline Void MemoryAllocator::operator delete( Void * ) {
    // nothing to do
}

inline const MemoryContext * MemoryAllocator::GetParentContext() const {
    return m_pParentContext;
}
inline UInt MemoryAllocator::GetAllocatorID() const {
    return m_iAllocatorID;
}
inline const GChar * MemoryAllocator::GetAllocatorName() const {
    return m_strAllocatorName;
}

inline Bool MemoryAllocator::IsTracing() const {
    return m_bTracing;
}

inline Void MemoryAllocator::TraceStart() {
    m_bTracing = true;
}
inline Void MemoryAllocator::TraceStop() {
    m_bTracing = false;
}

inline UInt MemoryAllocator::TraceCount() const {
    return m_iTraceCount;
}
inline const MemoryTraceRecord * MemoryAllocator::TracePick( UInt iIndex ) const {
    Assert( iIndex < m_iTraceCount );
    return ( m_arrTraceRecords + iIndex );
}

inline Void MemoryAllocator::TraceFlush() {
    _Tracing_LogAndFlush();
}
