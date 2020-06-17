/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/MemoryManager.inl
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
// Wrappers
inline _dat * _dat_get_ptr() {
    static _dat _dat_values = { NULL, 0, INVALID_OFFSET, INVALID_OFFSET };
    return &(_dat_values);
}
inline Void _dat_save( const GChar * strFile, UInt iLine, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID ) {
    _dat * pDAT = _dat_get_ptr();
    pDAT->strFile = strFile;
    pDAT->iLine = iLine;
    pDAT->iAllocatorID = iAllocatorID;
    pDAT->iContextID = iContextID;
}

/////////////////////////////////////////////////////////////////////////////////
// MemoryManager implementation
inline Void MemoryManager::Create() {
    Assert( sm_pInstance == NULL );
    sm_pInstance = new MemoryManager();
}
inline Void MemoryManager::Destroy() {
    Assert( sm_pInstance != NULL );
    delete sm_pInstance;
    sm_pInstance = NULL;
}
inline MemoryManager * MemoryManager::GetInstance() {
    Assert( sm_pInstance != NULL );
    return sm_pInstance;
}

inline Void * MemoryManager::operator new( SizeT iSize ) {
    Void * pMemory = SystemFn->MemAlloc( iSize );
    Assert( pMemory != NULL );
    return pMemory;
}
inline Void MemoryManager::operator delete( Void * pMemory ) {
    Assert( pMemory != NULL );
    SystemFn->MemFree( pMemory );
}

inline MemoryContextID MemoryManager::CreateContext( const GChar * strName, SizeT iResidentSize ) {
    return _MemoryContext_Create( strName, iResidentSize );
}
inline Void MemoryManager::DestroyContext( MemoryContextID iContextID ) {
    _MemoryContext_Destroy( iContextID );
}

inline const GChar * MemoryManager::GetContextName( MemoryContextID iContextID ) {
    MemoryContext * pContext = _MemoryContext_Get( iContextID );
    return pContext->strName;
}
inline Void * MemoryManager::GetContextResidentMemory( MemoryContextID iContextID, SizeT * outSize ) {
    MemoryContext * pContext = _MemoryContext_Get( iContextID );
    if ( outSize != NULL )
        *outSize = pContext->iResidentSize;
    return pContext->pResidentMemory;
}

inline MemoryAllocatorID MemoryManager::CreateStack( const GChar * strName, SizeT iStackSize, MemoryContextID iContextID ) {
    return _MemoryAllocator_Create( iContextID, ALLOCATOR_STACK, strName, iStackSize, 0 );
}
inline MemoryAllocatorID MemoryManager::CreatePool( const GChar * strName, SizeT iChunkSize, SizeT iTotalChunks, MemoryContextID iContextID ) {
    return _MemoryAllocator_Create( iContextID, ALLOCATOR_POOL, strName, iTotalChunks, iChunkSize );
}
inline MemoryAllocatorID MemoryManager::CreateHeap( const GChar * strName, SizeT iHeapSize, MemoryContextID iContextID ) {
    return _MemoryAllocator_Create( iContextID, ALLOCATOR_HEAP, strName, iHeapSize, 0 );
}
inline Void MemoryManager::DestroyAllocator( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID ) {
    _MemoryAllocator_Destroy( iContextID, iAllocatorID );
}

inline const GChar * MemoryManager::GetAllocatorName( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID ) {
    MemoryAllocator * pAllocator = _MemoryAllocator_Get( iContextID, iAllocatorID );
    return pAllocator->GetAllocatorName();
}
inline MemoryAllocator * MemoryManager::GetAllocator( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID ) {
    return _MemoryAllocator_Get( iContextID, iAllocatorID );
}

inline MemoryAllocatorID MemoryManager::GetSharedScratchAllocator() const {
    return m_iSharedScratchID;
}

inline Void MemoryManager::GenerateReport( AllocatorReport * outReport, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID ) {
    MemoryAllocator * pAllocator = _MemoryAllocator_Get( iContextID, iAllocatorID );
    pAllocator->GenerateReport( outReport );
}
inline Void MemoryManager::LogReport( const AllocatorReport * pReport, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID ) {
    MemoryAllocator * pAllocator = _MemoryAllocator_Get( iContextID, iAllocatorID );
    pAllocator->LogReport( pReport );
}

inline Bool MemoryManager::IsTracing( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID ) {
    MemoryAllocator * pAllocator = _MemoryAllocator_Get( iContextID, iAllocatorID );
    return pAllocator->IsTracing();
}

inline Void MemoryManager::TraceStart( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID ) {
    MemoryAllocator * pAllocator = _MemoryAllocator_Get( iContextID, iAllocatorID );
    pAllocator->TraceStart();
}
inline Void MemoryManager::TraceStop( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID ) {
    MemoryAllocator * pAllocator = _MemoryAllocator_Get( iContextID, iAllocatorID );
    pAllocator->TraceStop();
}

inline UInt MemoryManager::TraceCount( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID ) {
    MemoryAllocator * pAllocator = _MemoryAllocator_Get( iContextID, iAllocatorID );
    return pAllocator->TraceCount();
}
inline const MemoryTraceRecord * MemoryManager::TracePick( UInt iIndex, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID ) {
    MemoryAllocator * pAllocator = _MemoryAllocator_Get( iContextID, iAllocatorID );
    return pAllocator->TracePick( iIndex );
}

inline Void MemoryManager::TraceFlush( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID ) {
    MemoryAllocator * pAllocator = _MemoryAllocator_Get( iContextID, iAllocatorID );
    pAllocator->TraceFlush();
}

/////////////////////////////////////////////////////////////////////////////////

