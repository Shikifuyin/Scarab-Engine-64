/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Memory/Allocators/HeapAllocator.h
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
// Known Bugs : Improvement, use worst-fit when lowly fragmented, first-fit
//              when uniformly fragmented (stable), and best-fit when heavyly
//              fragmented ... or something like that (!)
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_LIB_MEMORY_ALLOCATORS_HEAPALLOCATOR_H
#define SCARAB_LIB_MEMORY_ALLOCATORS_HEAPALLOCATOR_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "MemoryAllocator.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define HEAPMEMORY_ALIGN_NATIVE sizeof(Void*)

#if defined(SCARAB_ARCHITECTURE_X64)
    #define HEAPMEMORY_ALIGN_SHIFT_NATIVE 3 // (a << 3) = a * 8 = a * sizeof(Void*)
#elif defined(SCARAB_ARCHITECTURE_X86)
    #define HEAPMEMORY_ALIGN_SHIFT_NATIVE 2 // (a << 2) = a * 4 = a * sizeof(Void*)
#endif

// A chunk in memory :
// | prevAUSize | AUSize | User Data ............................................... | (allocated)
// |     "      |    "   | pNext | pLeftChild | pRightChild |   iBalance   | ....... | (free, binheapnode)
// |     "      |    "   | pNext |   pPrev    |   Padding   | FakeBalance  | ....... | (free, binlistnode)
typedef SizeT AUSize;
#if defined(SCARAB_ARCHITECTURE_X64)
    #define AUSIZE_FREE_MASK 0x8000000000000000 // Mask of the free-flag bit in an AU's size-pointers
    #define AUSIZE_SIZE_MASK 0x7fffffffffffffff // Mask of the size bits in an AU's size-pointers
#elif defined(SCARAB_ARCHITECTURE_X86)
    #define AUSIZE_FREE_MASK 0x80000000 // Mask of the free-flag bit in an AU's size-pointers
    #define AUSIZE_SIZE_MASK 0x7fffffff // Mask of the size bits in an AU's size-pointers
#endif

typedef struct _chunk_head
{
	AUSize prevAUSize;
	AUSize thisAUSize;
} ChunkHead;

typedef struct _chunk_heapnode
{
    struct _chunk_listnode * pNext;
	struct _chunk_heapnode * pChild[2];
	Int iBalanceFactor;
} ChunkHeapNode;

typedef struct _chunk_listnode
{
    struct _chunk_listnode * pNext;
	struct _chunk_listnode * pPrev;
    UIntPtr Padding; // Allways set to BINHEAP_FAKEBALANCE so we have
    Int FakeBalance; // invalid balance value to distinguish nodes
} ChunkListNode;

#define BINHEAP_FAKEBALANCE ( (Int)0xdeadbeef )

// Height changes for the AVL-Tree :
enum BinHeapHeightChange
{
    BINHEAP_HEIGHT_NOCHANGE = 0,
    BINHEAP_HEIGHT_CHANGE = 1
};

// Childs indices :
enum BinHeapChild
{
    BINHEAP_CHILD_LEFT = 0,
    BINHEAP_CHILD_RIGHT = 1
};
#define BINHEAP_INVERTDIR(_dir) ( (BinHeapChild)(1-(_dir)) )

// Balance factors for the AVL-Tree :
#define BINHEAP_LEFT_HEAVY  (-1)
#define BINHEAP_BALANCED    0
#define BINHEAP_RIGHT_HEAVY 1
#define BINHEAP_IMBALANCE_LEFT(_bal) ( (_bal) < BINHEAP_LEFT_HEAVY )
#define BINHEAP_IMBALANCE_RIGHT(_bal) ( (_bal) > BINHEAP_RIGHT_HEAVY )

// Reporting System
#define HEAPREPORT_LOGFILE      TEXT("Logs/Memory/HeapReports.log")
#define HEAPREPORT_MAX_TREESPAN 256
#define HEAPREPORT_MAX_LISTSIZE 16
#define HEAPREPORT_MAX_CHUNKS   1024

typedef struct _heap_report : public AllocatorReport
{
    Void * pBaseAddress;
    SizeT iTotalSize;
    SizeT iAllocatedSize;
    SizeT iFreeSize;
    Byte * pLastFreed;

    UInt iBinHeapSize;    
    Byte ** arrHeapNodes;  //
    Int * arrBalances;     // size = iBinHeapSize
    UInt * arrListSizes;   //
    Byte ** arrListNodes;  // <= size[i] = arrFreeListSizes[i]

    UInt iChunkMapSize;
    Byte ** arrChunkMap;   //
    SizeT * arrPrevSizes;  //  size = iChunkMapSize
    SizeT * arrSizes;      //
    Bool * arrIsAllocated; //
} HeapReport;

///////////////////////////////////////////////////////////////////////////////
// The HeapAllocator class
class HeapAllocator : public MemoryAllocator
{
public:
    HeapAllocator( const MemoryContext * pParentContext, MemoryAllocatorID iAllocatorID, const GChar * strAllocatorName, SizeT iHeapSize );
	virtual ~HeapAllocator();

    // Getters
    inline virtual AllocatorType GetType() const;
    inline virtual Bool CheckAddressRange( Void * pMemory ) const;
    inline virtual SizeT GetBlockSize( Void * pMemory ) const;

    // Alloc/Free interface
	virtual Void * Allocate( SizeT iSize );
	virtual Void Free( Void * pMemory );

    // Reporting
    virtual Void GenerateReport( AllocatorReport * outReport ) const;
    virtual Void LogReport( const AllocatorReport * pReport ) const;

private:
    // Internal constants
    const UInt AlignUnit;             // Size of an AU, must be power of 2
    const UInt AlignUnitShift;        // Shift-size of an AU, 2^Shift = Size
    const AUSize ChunkHeadAUSize;     // Size of ChunkHead, in AUs
    const AUSize ChunkHeapNodeAUSize; // Size of ChunkHeapNode, in AUs
    const AUSize ChunkListNodeAUSize; // Size of ChunkListNode, in AUs
	const AUSize DummyChunkAUSize;
	const SizeT DummyChunkByteSize;
	const AUSize MinimalChunkAUSize;

    // AU helpers
    inline AUSize _AU_ConvertSize( SizeT iSize ) const;
    inline Bool _AU_IsAllocated( AUSize iAUSize ) const;
	inline Bool _AU_IsFree( AUSize iAUSize ) const;
	inline AUSize _AU_Size( AUSize iAUSize ) const;
	inline Byte * _AU_Next( Byte * pChunk, AUSize nAAU ) const;
	inline Byte * _AU_Prev( Byte * pChunk, AUSize nAAU ) const;

    // Chunk helpers
    inline Void _Chunk_MarkAllocated( ChunkHead * pChunk ) const;
	inline Void _Chunk_MarkFree( ChunkHead * pChunk ) const;
	inline Bool _Chunk_IsAllocated( const ChunkHead * pChunk ) const;
	inline Bool _Chunk_IsFree( const ChunkHead * pChunk ) const;
	inline AUSize _Chunk_PrevSize( const ChunkHead * pChunk ) const;
	inline AUSize _Chunk_Size( const ChunkHead * pChunk ) const;
    inline ChunkHeapNode * _Chunk_GetHeapNode( ChunkHead * pChunk ) const;
	inline ChunkListNode * _Chunk_GetListNode( ChunkHead * pChunk ) const;
    inline ChunkHead * _Chunk_GetHead( ChunkHeapNode * pHeapNode ) const;
    inline ChunkHead * _Chunk_GetHead( ChunkListNode * pListNode ) const;
    inline Int _Compare( const ChunkHead * pLHS, const ChunkHead * pRHS ) const;

    // BinHeap interface
    ChunkHead * _BinHeap_RequestChunk( AUSize iMinSize );
	Void _BinHeap_ReleaseChunk( ChunkHead * pChunk );

    // Deferred coalescing
	ChunkHead * _PerformCoalescing( ChunkHead * pChunk );
	ChunkHead * m_pLastFreed;

    // AVL-Tree sub-routines & data
	BinHeapHeightChange _RotateOnce( ChunkHeapNode ** ppNode, BinHeapChild rotDir );
	BinHeapHeightChange _RotateTwice( ChunkHeapNode ** ppNode, BinHeapChild rotDir );
	BinHeapHeightChange _ReBalance( ChunkHeapNode ** ppNode );
	ChunkHeapNode * _rec_Insert( ChunkHeapNode ** ppNode, BinHeapHeightChange & heightChange, ChunkHead * pChunk );
	Bool _rec_Remove( ChunkHeapNode ** ppNode, BinHeapHeightChange & heightChange, ChunkHead * pChunk );
    ChunkHeapNode * _Search( AUSize iMinSize ) const;
    ChunkHeapNode * _Match( const ChunkHead * pChunk ) const;
    ChunkHead * _Replace( ChunkHead * pNewChunk );

    ChunkHeapNode * m_pBinHeapRoot;
    BinHeapHeightChange m_iHeightChange;

    // Heap memory
	Byte * m_pHeapMemory;
	SizeT m_iHeapSize;
    SizeT m_iTotalFree;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "HeapAllocator.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MEMORY_ALLOCATORS_HEAPALLOCATOR_H
