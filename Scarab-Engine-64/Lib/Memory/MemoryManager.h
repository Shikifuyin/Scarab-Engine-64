/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/MemoryManager.h
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
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_LIB_MEMORY_MEMORYMANAGER_H
#define SCARAB_LIB_MEMORY_MEMORYMANAGER_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../Error/ErrorManager.h"

#include "Allocators/StackAllocator.h"
#include "Allocators/PoolAllocator.h"
#include "Allocators/HeapAllocator.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define MemoryFn MemoryManager::GetInstance()

// Memory Manager definitions
#define MEMORY_MAX_CONTEXTS   1024
#define MEMORY_MAX_ALLOCATORS 1024

#define MEMORY_CONTEXT_SHARED              (MEMORY_MAX_CONTEXTS + 1)
#define MEMORY_CONTEXT_SHARED_SCRATCH_SIZE 134217728 // 128 mb

// Macro interface
    // New( ClassName, VariableName, ConstructorCall, iAllocatorID, iContextID )
#define New(...)                                                                 VARIADIC_MACRO( _New, __VA_ARGS__ )
#define _New_2( _class_name, _var_name )                                         _New_3( _class_name, _var_name, _class_name() )
#define _New_3( _class_name, _var_name, _ctor_call )                             _New_4( _class_name, _var_name, _ctor_call, 0 )
#define _New_4( _class_name, _var_name, _ctor_call, _allocator_id )              _New_5( _class_name, _var_name, _ctor_call, _allocator_id, MEMORY_CONTEXT_SHARED )
#define _New_5( _class_name, _var_name, _ctor_call, _allocator_id, _context_id ) \
    _var_name = (_class_name*)( MemoryFn->Allocate(sizeof(_class_name), false, TEXT(__FILE__), __LINE__, _allocator_id, _context_id) ); \
    *_var_name = _ctor_call;

    // NewArray( ClassName, VariableName, ArraySize, iAllocatorID, iContextID )
#define NewArray(...)                                                                  VARIADIC_MACRO( _NewArray, __VA_ARGS__ )
#define _NewArray_3( _class_name, _var_name, _array_size )                             _NewArray_4( _class_name, _var_name, _array_size, 0 )
#define _NewArray_4( _class_name, _var_name, _array_size, _allocator_id )              _NewArray_5( _class_name, _var_name, _array_size, _allocator_id, MEMORY_CONTEXT_SHARED )
#define _NewArray_5( _class_name, _var_name, _array_size, _allocator_id, _context_id ) \
    _var_name = (_class_name*)( MemoryFn->Allocate((_array_size)*sizeof(_class_name), true, TEXT(__FILE__), __LINE__, _allocator_id, _context_id) );

    // Delete( VariableName, iAllocatorID, iContextID )
#define Delete(...)                                        VARIADIC_MACRO( _Delete, __VA_ARGS__ )
#define _Delete_1( _var_name )                             _Delete_2( _var_name, 0 )
#define _Delete_2( _var_name, _allocator_id )              _Delete_3( _var_name, _allocator_id, MEMORY_CONTEXT_SHARED )
#define _Delete_3( _var_name, _allocator_id, _context_id ) \
    MemoryFn->Free( _var_name, false, TEXT(__FILE__), __LINE__, _allocator_id, _context_id );

    // DeleteArray( VariableName, iAllocatorID, iContextID )
#define DeleteArray(...)                                        VARIADIC_MACRO( _DeleteArray, __VA_ARGS__ )
#define _DeleteArray_1( _var_name )                             _DeleteArray_2( _var_name, 0 )
#define _DeleteArray_2( _var_name, _allocator_id )              _DeleteArray_3( _var_name, _allocator_id, MEMORY_CONTEXT_SHARED )
#define _DeleteArray_3( _var_name, _allocator_id, _context_id ) \
    MemoryFn->Free( _var_name, true, TEXT(__FILE__), __LINE__, _allocator_id, _context_id );

// new/new[]/delete/delete[] wrappers
//Void * operator new ( SizeT iSize, const GChar * strFile, UInt iLine, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID );
//Void * operator new[] ( SizeT iSize, const GChar * strFile, UInt iLine, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID );
//Void operator delete ( Void * pMemory, const GChar * strFile, UInt iLine, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID );   // Required if constructor throws
//Void operator delete[] ( Void * pMemory, const GChar * strFile, UInt iLine, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID ); // an exception ...
//Void operator delete ( Void * pMemory /*, const GChar * strFile, UInt iLine, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID*/ );
//Void operator delete[] ( Void * pMemory /*, const GChar * strFile, UInt iLine, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID*/ );

// Work-around trick to pass arguments to delete, which has no placement syntax
//typedef struct _delete_argument_trick {
//    const GChar * strFile;
//    UInt iLine;
//    MemoryAllocatorID iAllocatorID;
//    MemoryContextID iContextID;
//} _dat;
//inline _dat * _dat_get_ptr();
//Void _dat_save( const GChar * strFile, UInt iLine, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID );

// Macro interface
//#define New(...)                             VARIADIC_MACRO( _New, __VA_ARGS__ )
//#define _New_0()                             _New_1( 0 )
//#define _New_1( _allocator_id )              _New_2( _allocator_id, MEMORY_CONTEXT_SHARED )
//#define _New_2( _allocator_id, _context_id ) \
//    new( TEXT(__FILE__), __LINE__, _allocator_id, _context_id )
//
//#define Delete(...)                                       VARIADIC_MACRO( _Delete, __VA_ARGS__ )
//#define _Delete_1( _pointer )                             _Delete_2( _pointer, 0 )
//#define _Delete_2( _pointer, _allocator_id )              _Delete_3( _pointer, _allocator_id, MEMORY_CONTEXT_SHARED )
//#define _Delete_3( _pointer, _allocator_id, _context_id ) { \
//    _dat_save( TEXT(__FILE__), __LINE__, _allocator_id, _context_id ); \
//    delete _pointer; \
//}
//
//#define DeleteA(...)                                       VARIADIC_MACRO( _DeleteA, __VA_ARGS__ )
//#define _DeleteA_1( _pointer )                             _DeleteA_2( _pointer, 0 )
//#define _DeleteA_2( _pointer, _allocator_id )              _DeleteA_3( _pointer, _allocator_id, MEMORY_CONTEXT_SHARED )
//#define _DeleteA_3( _pointer, _allocator_id, _context_id ) { \
//    _dat_save( TEXT(__FILE__), __LINE__, _allocator_id, _context_id ); \
//    delete[] _pointer; \
//}

/////////////////////////////////////////////////////////////////////////////////
// The MemoryManager class
class MemoryManager
{
    // Manual singleton interface since the manager
    // template actually depends on this !
public:
    inline static Void Create();
    inline static Void Destroy();
    inline static MemoryManager * GetInstance();

private:
    MemoryManager();
    ~MemoryManager();

    inline Void * operator new( SizeT iSize );
    inline Void operator delete( Void * pMemory );

    static MemoryManager * sm_pInstance;

public:
    // Contexts Management
    inline MemoryContextID CreateContext( const GChar * strName, SizeT iResidentSize );
    inline Void DestroyContext( MemoryContextID iContextID );

    inline const GChar * GetContextName( MemoryContextID iContextID );
    inline Void * GetContextResidentMemory( MemoryContextID iContextID, SizeT * outSize = NULL );

    // Allocators Management
    inline MemoryAllocatorID CreateStack( const GChar * strName, SizeT iStackSize,                    MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );
    inline MemoryAllocatorID CreatePool( const GChar * strName, SizeT iChunkSize, SizeT iTotalChunks, MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );
    inline MemoryAllocatorID CreateHeap( const GChar * strName, SizeT iHeapSize,                      MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );
    inline Void DestroyAllocator( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );

    inline const GChar * GetAllocatorName( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );
    inline MemoryAllocator * GetAllocator( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );

    inline MemoryAllocatorID GetSharedScratchAllocator() const;

    // Main Allocation Routines
    Void * Allocate( SizeT iSize, Bool bIsArray, const GChar * strFile, UInt iLine, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );
    Void Free( Void * pMemory, Bool bIsArray, const GChar * strFile, UInt iLine, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );

    // For specialized memory management (mostly stack allocators), Use GetAllocator to obtain the allocator interface //

    // Reporting System
    inline Void GenerateReport( AllocatorReport * outReport, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );
    inline Void LogReport( const AllocatorReport * pReport, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );

    // Tracing System
    inline Bool IsTracing( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );

    inline Void TraceStart( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );
    inline Void TraceStop( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );

    inline UInt TraceCount( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );
    inline const MemoryTraceRecord * TracePick( UInt iIndex, MemoryAllocatorID iAllocatorID, MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );

    inline Void TraceFlush( MemoryAllocatorID iAllocatorID, MemoryContextID iContextID = MEMORY_CONTEXT_SHARED );

private:
    // Helpers
    MemoryContext * _MemoryContext_Get( MemoryContextID iContextID );
    MemoryContextID _MemoryContext_Create( const GChar * strName, SizeT iResidentSize );
    Void _MemoryContext_Destroy( MemoryContextID iContextID );

    Void _MemoryContext_Initialize( MemoryContext * pContext, MemoryContextID iContextID, const GChar * strName, SizeT iResidentSize );
    Void _MemoryContext_Cleanup( MemoryContext * pContext );

    MemoryAllocator * _MemoryAllocator_Get( MemoryContextID iContextID, MemoryAllocatorID iAllocatorID );
    MemoryAllocatorID _MemoryAllocator_Create( MemoryContextID iContextID, AllocatorType iType, const GChar * strName, SizeT iSize, SizeT iBlockSize );
    Void _MemoryAllocator_Destroy( MemoryContextID iContextID, MemoryAllocatorID iAllocatorID );

    // Allocator Factories
    PoolAllocator m_hStackFactory;
    PoolAllocator m_hPoolFactory;
    PoolAllocator m_hHeapFactory;

    // Memory Contexts
    MemoryContext m_hSharedContext;
    MemoryAllocatorID m_iSharedScratchID;

    UInt m_iContextCount;
    UInt m_iNextFreeContext;
    MemoryContext m_arrContexts[MEMORY_MAX_CONTEXTS];
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "MemoryManager.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MEMORY_MEMORYMANAGER_H

