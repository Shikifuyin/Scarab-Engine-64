/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/Types/CUDAVector.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA-Optimized Large Vectors
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// TCUDAVector implementation
template<typename Number>
TCUDAVector<Number>::TCUDAVector( UInt iSize, Number * arrData ):
	m_hDeviceVector()
{
	// Wrap as Page-Locked Host Memory
	CUDAHostMemory hWrapped;
	hWrapped.Wrap( arrData, iSize, CUDA_HOSTMEMORY_WRAP_FLAG_PINNED );
	
	// Allocate Device Memory as 1D-shaped
	m_hDeviceVector.Allocate( sizeof(Number), iSize );
	
	// Copy to Device Memory
	CUDAMemoryPosition hDestPos;
	hDestPos.iX = 0;
	hDestPos.iY = 0;
	hDestPos.iZ = 0;
	
	CUDAMemoryPosition hSrcPos;
	hSrcPos.iX = 0;
	hSrcPos.iY = 0;
	hSrcPos.iZ = 0;
	
	CUDAMemoryRegion hCopyRegion;
	hCopyRegion.iWidth = iSize;
	hCopyRegion.iHeight = 1;
	hCopyRegion.iDepth = 1;
	
	m_hDeviceVector.MemCopy( hDestPos, &hWrapped, hSrcPos, hCopyRegion );
	
	// UnWrap Page-Locked Host Memory
	hWrapped.UnWrap();
}
template<typename Number>
TCUDAVector<Number>::TCUDAVector( const CUDAHostMemory * pHostMemory )
{
	// Allocate Device Memory as 1D-shaped
	m_hDeviceVector.Allocate( pHostMemory->GetStride(), pHostMemory->GetWidth() );
	
	// Copy to Device Memory
	CUDAMemoryPosition hDestPos;
	hDestPos.iX = 0;
	hDestPos.iY = 0;
	hDestPos.iZ = 0;
	
	CUDAMemoryPosition hSrcPos;
	hSrcPos.iX = 0;
	hSrcPos.iY = 0;
	hSrcPos.iZ = 0;
	
	CUDAMemoryRegion hCopyRegion;
	hCopyRegion.iWidth = pHostMemory->GetWidth();
	hCopyRegion.iHeight = 1;
	hCopyRegion.iDepth = 1;
	
	m_hDeviceVector.MemCopy( hDestPos, pHostMemory, hSrcPos, hCopyRegion );
}
template<typename Number>
TCUDAVector<Number>::TCUDAVector( const TCUDAVector<Number> & rhs )
{
	// Allocate Device Memory as 1D-shaped
	m_hDeviceVector.Allocate( rhs.m_hDeviceVector.GetStride(), rhs.m_hDeviceVector.GetWidth() );
	
	// Copy to Device Memory
	CUDAMemoryPosition hDestPos;
	hDestPos.iX = 0;
	hDestPos.iY = 0;
	hDestPos.iZ = 0;
	
	CUDAMemoryPosition hSrcPos;
	hSrcPos.iX = 0;
	hSrcPos.iY = 0;
	hSrcPos.iZ = 0;
	
	CUDAMemoryRegion hCopyRegion;
	hCopyRegion.iWidth = rhs.m_hDeviceVector.GetWidth();
	hCopyRegion.iHeight = 1;
	hCopyRegion.iDepth = 1;
	
	m_hDeviceVector.MemCopy( hDestPos, &(rhs.m_hDeviceVector), hSrcPos, hCopyRegion );
}
template<typename Number>
TCUDAVector<Number>::~TCUDAVector()
{
	// Free Device Memory
	m_hDeviceVector.Free();
}

template<typename Number>
TCUDAVector<Number> & TCUDAVector<Number>::operator=( const TCUDAVector<Number> & rhs )
{
	Assert( m_hDeviceVector.GetWidth() == rhs.m_hDeviceVector.GetWidth() );
	
	// Copy to Device Memory
	CUDAMemoryPosition hDestPos;
	hDestPos.iX = 0;
	hDestPos.iY = 0;
	hDestPos.iZ = 0;
	
	CUDAMemoryPosition hSrcPos;
	hSrcPos.iX = 0;
	hSrcPos.iY = 0;
	hSrcPos.iZ = 0;
	
	CUDAMemoryRegion hCopyRegion;
	hCopyRegion.iWidth = rhs.m_hDeviceVector.GetWidth();
	hCopyRegion.iHeight = 1;
	hCopyRegion.iDepth = 1;
	
	m_hDeviceVector.MemCopy( hDestPos, &(rhs.m_hDeviceVector), hSrcPos, hCopyRegion );
	
	return (*this);
}

template<typename Number>
Number & TCUDAVector<Number>::operator[]( Int i )
{
	Assert( i < m_hDeviceVector.GetWidth() );
	
	CUDAMemoryPosition hPosition;
	hPosition.iX = (SizeT)i;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	
	return *( (Number*)(m_hDeviceVector.GetPointer(hPosition)) );
}
template<typename Number>
const Number & TCUDAVector<Number>::operator[]( Int i ) const
{
	Assert( i < m_hDeviceVector.GetWidth() );
	
	CUDAMemoryPosition hPosition;
	hPosition.iX = (SizeT)i;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	
	return *( (const Number*)(m_hDeviceVector.GetPointer(hPosition)) );
}
template<typename Number>
Number & TCUDAVector<Number>::operator[]( UInt i )
{
	Assert( i < m_hDeviceVector.GetWidth() );
	
	CUDAMemoryPosition hPosition;
	hPosition.iX = (SizeT)i;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	
	return *( (Number*)(m_hDeviceVector.GetPointer(hPosition)) );
}
template<typename Number>
const Number & TCUDAVector<Number>::operator[]( UInt i ) const
{
	Assert( i < m_hDeviceVector.GetWidth() );
	
	CUDAMemoryPosition hPosition;
	hPosition.iX = (SizeT)i;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	
	return *( (const Number*)(m_hDeviceVector.GetPointer(hPosition)) );
}


	
	
	
	