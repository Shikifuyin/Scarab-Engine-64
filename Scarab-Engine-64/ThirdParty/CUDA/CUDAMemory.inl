/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDAMemory.inl
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
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// CUDAMemory implementation
inline Bool CUDAMemory::IsAllocated() const {
	return ( m_pMemory != NULL );
}
inline Bool CUDAMemory::HasOwnerShip() const {
	return m_bHasOwnerShip;
}

inline CUDAMemoryShape CUDAMemory::GetShape() const {
	DebugAssert( m_pMemory != NULL );
	return m_iShape;
}
inline SizeT CUDAMemory::GetStride() const {
	DebugAssert( m_pMemory != NULL && m_iShape >= CUDA_MEMORY_SHAPE_1D );
	return m_iStride;
}
inline SizeT CUDAMemory::GetPitch() const {
	DebugAssert( m_pMemory != NULL && m_iShape >= CUDA_MEMORY_SHAPE_2D );
	return m_iPitch;
}
inline SizeT CUDAMemory::GetSlice() const {
	DebugAssert( m_pMemory != NULL && m_iShape >= CUDA_MEMORY_SHAPE_3D );
	return m_iSlice;
}

inline SizeT CUDAMemory::GetWidth() const {
	DebugAssert( m_pMemory != NULL && m_iShape >= CUDA_MEMORY_SHAPE_1D );
	return m_iWidth;
}
inline SizeT CUDAMemory::GetHeight() const {
	DebugAssert( m_pMemory != NULL && m_iShape >= CUDA_MEMORY_SHAPE_2D );
	return m_iHeight;
}
inline SizeT CUDAMemory::GetDepth() const {
	DebugAssert( m_pMemory != NULL && m_iShape >= CUDA_MEMORY_SHAPE_3D );
	return m_iDepth;
}
inline SizeT CUDAMemory::GetSize() const {
	DebugAssert( m_pMemory != NULL );
	return m_iSize;
}

inline Void * CUDAMemory::GetPointer( UInt iOffset ) {
	DebugAssert( m_pMemory != NULL );
	DebugAssert( iOffset < m_iSize );
	return ((Byte*)m_pMemory) + iOffset;
}
inline const Void * CUDAMemory::GetPointer( UInt iOffset ) const {
	DebugAssert( m_pMemory != NULL );
	DebugAssert( iOffset < m_iSize );
	return ((Byte*)m_pMemory) + iOffset;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDAHostMemory implementation
inline Bool CUDAHostMemory::IsHostMemory() const {
	return true;
}
inline Bool CUDAHostMemory::IsDeviceMemory() const {
	return false;
}
inline Bool CUDAHostMemory::IsManagedMemory() const {
	return false;
}

inline Bool CUDAHostMemory::IsPinned() const {
	if ( m_bIsWrapped )
		return ( (m_iHostMemoryWrapFlags & CUDA_HOSTMEMORY_WRAP_FLAG_PINNED) != 0 );
	return ( (m_iHostMemoryAllocFlags & CUDA_HOSTMEMORY_ALLOC_FLAG_PINNED) != 0 );
}
inline Bool CUDAHostMemory::IsMapped() const {
	if ( m_bIsWrapped )
		return ( (m_iHostMemoryWrapFlags & CUDA_HOSTMEMORY_WRAP_FLAG_MAPPED) != 0 );
	return ( (m_iHostMemoryAllocFlags & CUDA_HOSTMEMORY_ALLOC_FLAG_MAPPED) != 0 );
}
inline Bool CUDAHostMemory::IsWriteCombine() const {
	if ( m_bIsWrapped )
		return false;
	return ( (m_iHostMemoryAllocFlags & CUDA_HOSTMEMORY_ALLOC_FLAG_WRITE_COMBINED) != 0 );
}

inline Bool CUDAHostMemory::IsWrapped() const {
	return m_bIsWrapped;
}
inline Bool CUDAHostMemory::IsWrappedIO() const {
	return ( (m_iHostMemoryWrapFlags & CUDA_HOSTMEMORY_WRAP_FLAG_IO) != 0 );
}
inline Bool CUDAHostMemory::IsWrappedReadOnly() const {
	return ( (m_iHostMemoryWrapFlags & CUDA_HOSTMEMORY_WRAP_FLAG_READONLY) != 0 );
}

template<class T>
const T & CUDAHostMemory::Read( const CUDAMemoryPosition & hPosition ) const
{
	DebugAssert( IsAllocated() );
	DebugAssert( IsValidPosition(hPosition) );
	DebugAssert( m_iStride == sizeof(T) );
	T * pValue = (T*)( GetPointer(hPosition) );
	return *pValue;
}
template<class T>
Void CUDAHostMemory::Write( const CUDAMemoryPosition & hPosition, const T & hValue )
{
	DebugAssert( IsAllocated() );
	DebugAssert( IsValidPosition(hPosition) );
	DebugAssert( m_iStride == sizeof(T) );
	T * pValue = (T*)( GetPointer(hPosition) );
	*pValue = hValue;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDADeviceMemory implementation
inline Bool CUDADeviceMemory::IsHostMemory() const {
	return false;
}
inline Bool CUDADeviceMemory::IsDeviceMemory() const {
	return true;
}
inline Bool CUDADeviceMemory::IsManagedMemory() const {
	return false;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDAManagedMemory implementation
inline Bool CUDAManagedMemory::IsHostMemory() const {
	return false;
}
inline Bool CUDAManagedMemory::IsDeviceMemory() const {
	return false;
}
inline Bool CUDAManagedMemory::IsManagedMemory() const {
	return true;
}





