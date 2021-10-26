/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDAMemory.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA Memory Containers
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
// TODO : - Add Stream-Ordered Memory Allocator support (new in CUDA v11.2)
//        - Add cudaArray support ? Needed ?
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_CUDA_CUDAMEMORY_H
#define SCARAB_THIRDPARTY_CUDA_CUDAMEMORY_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../System/System.h"

#include "CUDAAsynchronous.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
typedef Int CUDADeviceID;

enum CUDAMemoryShape {
	CUDA_MEMORY_SHAPE_NONE = 0,
	CUDA_MEMORY_SHAPE_1D,
	CUDA_MEMORY_SHAPE_2D,
	CUDA_MEMORY_SHAPE_3D
};

struct CUDAMemoryPosition {
	CUDAMemoryPosition()
	{
		iX = 0;
		iY = 0;
		iZ = 0;
	}
	CUDAMemoryPosition( SizeT X, SizeT Y, SizeT Z )
	{
		iX = X;
		iY = Y;
		iZ = Z;
	}
	~CUDAMemoryPosition() {}

	inline CUDAMemoryPosition & operator=( const CUDAMemoryPosition & rhs ) {
		iX = rhs.iX;
		iY = rhs.iY;
		iZ = rhs.iZ;
		return (*this);
	}
	
	SizeT iX;
	SizeT iY;
	SizeT iZ;
};

struct CUDAMemoryRegion {
	CUDAMemoryRegion()
	{
		iWidth = 0;
		iHeight = 0;
		iDepth = 0;
	}
	CUDAMemoryRegion( SizeT Width, SizeT Height, SizeT Depth )
	{
		iWidth = Width;
		iHeight = Height;
		iDepth = Depth;
	}
	~CUDAMemoryRegion() {}

	inline CUDAMemoryRegion & operator=( const CUDAMemoryRegion & rhs ) {
		iWidth = rhs.iWidth;
		iHeight = rhs.iHeight;
		iDepth = rhs.iDepth;
		return (*this);
	}
	
	SizeT iWidth;
	SizeT iHeight;
	SizeT iDepth;
};

// Prototypes
class CUDAGraph;
class CUDAGraphNodeMemSet;
class CUDAGraphNodeMemCopy;

/////////////////////////////////////////////////////////////////////////////////
// The CUDAMemory class
class CUDAMemory
{
public:
	CUDAMemory();
	virtual ~CUDAMemory();
	
	virtual Bool IsHostMemory() const = 0;
	virtual Bool IsDeviceMemory() const = 0;
	virtual Bool IsManagedMemory() const = 0;
	
	// Symbols
	static Void * GetSymbolAddress( const Void * pSymbol );
	static SizeT GetSymbolSize( const Void * pSymbol );
	
	// CUDAHostMemory, CUDAManagedMemory : Returns the Device which was current when this memory was allocated
	// CUDADeviceMemory : Returns the Device on which this memory was allocated
	CUDADeviceID GetDeviceID() const;
	
	inline Bool IsAllocated() const;
	inline Bool HasOwnerShip() const;
	
	inline CUDAMemoryShape GetShape() const;
	inline SizeT GetStride() const;
	inline SizeT GetPitch() const;
	inline SizeT GetSlice() const;
	
	inline SizeT GetWidth() const;
	inline SizeT GetHeight() const;
	inline SizeT GetDepth() const;
	inline SizeT GetSize() const;

	Bool IsValidPosition( const CUDAMemoryPosition & hPosition ) const;
	Bool IsValidRegion( const CUDAMemoryRegion & hRegion ) const;
	Bool IsValidRegion( const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const;
	
	inline Void * GetPointer( UInt iOffset = 0 );
	inline const Void * GetPointer( UInt iOffset = 0 ) const;
	
	Void * GetPointer( const CUDAMemoryPosition & hPosition );
	const Void * GetPointer( const CUDAMemoryPosition & hPosition ) const;
	
	// YOU are responsible for proper behaviour when reading/writing memory !!!
	// For instance, you must respect CUDA_HOSTMEMORY_ALLOC_FLAG_WRITE_COMBINED !
	
		// Flat Set
	Void Set( SizeT iSize, Int iValue );
		// Shaped Set
	Void Set( const CUDAMemoryPosition & hDestPos, const CUDAMemoryRegion & hSetRegion, UInt iValue );
	
		// Flat Copy
	Void Copy( const CUDAMemory * pSrc, SizeT iSize );
		// Shaped Copy, supports any Shape combinations !
	Void Copy( const CUDAMemoryPosition & hDestPos,
			   const CUDAMemory * pSrc, const CUDAMemoryPosition & hSrcPos,
			   const CUDAMemoryRegion & hCopyRegion );
	
protected:
	friend class CUDAGraph;
	friend class CUDANodeMemSet;
	friend class CUDANodeMemCopy;

	UInt _GetMemCopyKind( const CUDAMemory * pSrc ) const;
	Void _ConvertCopyParams( Void * outParams,
							 const CUDAMemoryPosition & hDestPos,
							 const CUDAMemory * pSrc, const CUDAMemoryPosition & hSrcPos,
							 const CUDAMemoryRegion & hCopyRegion ) const;

	Bool m_bHasOwnerShip;
	Void * m_pMemory;
	
	CUDAMemoryShape m_iShape;
	SizeT m_iStride;
	SizeT m_iPitch;
	SizeT m_iSlice;
	
	SizeT m_iWidth;
	SizeT m_iHeight;
	SizeT m_iDepth;
	SizeT m_iSize;
};

/////////////////////////////////////////////////////////////////////////////////
// The CUDAHostMemory class
class CUDAHostMemory : public CUDAMemory
{
public:
	// Host Memory is Page-Locked
	// - Devices with CUDA_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING are automatically mapped
	//   and can access host memory transparently even without CUDA_HOSTMEMORY_ALLOC_FLAG_MAPPED
	CUDAHostMemory();
	virtual ~CUDAHostMemory();
	
	inline virtual Bool IsHostMemory() const;
	inline virtual Bool IsDeviceMemory() const;
	inline virtual Bool IsManagedMemory() const;
	
	inline Bool IsPinned() const;       // Accessible to all CUDA contexts
	inline Bool IsMapped() const;       // Mapped to device memory
	inline Bool IsWriteCombine() const; // Optimized for Host->Device transfers, no write-back
	
	inline Bool IsWrapped() const;		   // Wrapped from conventional system-allocated memory
	inline Bool IsWrappedIO() const;       // Wrapped from memory-mapped IO device
	inline Bool IsWrappedReadOnly() const; // Wrapped as Read-Only
	
	Void Allocate( SizeT iSize, UInt iHostMemoryAllocFlags = CUDA_HOSTMEMORY_ALLOC_FLAG_DEFAULT );
	Void Allocate( SizeT iElementSize, SizeT iWidth, UInt iHostMemoryAllocFlags = CUDA_HOSTMEMORY_ALLOC_FLAG_DEFAULT );
	Void Allocate( SizeT iElementSize, SizeT iWidth, SizeT iHeight, UInt iHostMemoryAllocFlags = CUDA_HOSTMEMORY_ALLOC_FLAG_DEFAULT );
	Void Allocate( SizeT iElementSize, SizeT iWidth, SizeT iHeight, SizeT iDepth, UInt iHostMemoryAllocFlags = CUDA_HOSTMEMORY_ALLOC_FLAG_DEFAULT );
	Void Free();
	
	Void Wrap( Void * pSystemMemory, SizeT iSize, UInt iHostMemoryWrapFlags = CUDA_HOSTMEMORY_WRAP_FLAG_DEFAULT );
	Void Wrap( Void * pSystemMemory, SizeT iElementSize, SizeT iWidth, UInt iHostMemoryWrapFlags = CUDA_HOSTMEMORY_WRAP_FLAG_DEFAULT );
	Void Wrap( Void * pSystemMemory, SizeT iElementSize, SizeT iWidth, SizeT iHeight, UInt iHostMemoryWrapFlags = CUDA_HOSTMEMORY_WRAP_FLAG_DEFAULT );
	Void Wrap( Void * pSystemMemory, SizeT iElementSize, SizeT iWidth, SizeT iHeight, SizeT iDepth, UInt iHostMemoryWrapFlags = CUDA_HOSTMEMORY_WRAP_FLAG_DEFAULT );
	Void UnWrap();
	
	// Dangerous for now ...
	//Void GetMappedDeviceMemory( CUDADeviceMemory * outDeviceMemory ) const;
	
protected:
	UInt m_iHostMemoryAllocFlags; // CUDAHostMemoryAllocFlags
	Bool m_bIsWrapped;
	UInt m_iHostMemoryWrapFlags; // CUDAHostMemoryWrapFlags
};

/////////////////////////////////////////////////////////////////////////////////
// The CUDADeviceMemory class
class CUDADeviceMemory : public CUDAMemory
{
public:
	// Device Memory has strict alignment policy
	CUDADeviceMemory();
	virtual ~CUDADeviceMemory();
	
	static Void GetCapacity( SizeT * outFreeMemory, SizeT * outTotalMemory );
	
	inline virtual Bool IsHostMemory() const;
	inline virtual Bool IsDeviceMemory() const;
	inline virtual Bool IsManagedMemory() const;
	
	Void Allocate( SizeT iSize );
	Void Allocate( SizeT iElementSize, SizeT iWidth );
	Void Allocate( SizeT iElementSize, SizeT iWidth, SizeT iHeight );
	Void Allocate( SizeT iElementSize, SizeT iWidth, SizeT iHeight, SizeT iDepth );
	Void Free();
	
protected:
};

/////////////////////////////////////////////////////////////////////////////////
// The CUDAManagedMemory class
class CUDAManagedMemory : public CUDAMemory
{
public:
	// Managed Memory is accessible for both host and device
	// - Device requires CUDA_DEVICE_ATTRIBUTE_MANAGED_MEMORY and CUDA_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
	// - Device access to Host-Attached memory requires CUDA_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_MEMORY_ACCESS
	//   and explicit Stream Attachment
	CUDAManagedMemory();
	virtual ~CUDAManagedMemory();
	
	inline virtual Bool IsHostMemory() const;
	inline virtual Bool IsDeviceMemory() const;
	inline virtual Bool IsManagedMemory() const;
	
	Void Allocate( SizeT iSize, Bool bAttachHost = false );
	Void Allocate( SizeT iElementSize, SizeT iWidth, Bool bAttachHost = false );
	Void Allocate( SizeT iElementSize, SizeT iWidth, SizeT iHeight, Bool bAttachHost = false );
	Void Allocate( SizeT iElementSize, SizeT iWidth, SizeT iHeight, SizeT iDepth, Bool bAttachHost = false );
	Void Free();
	
protected:
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUDAMemory.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_CUDAMEMORY_H


