/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/BLAS/CUBLASContext.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA BLAS Context management
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// CUBLASContext implementation
inline Bool CUBLASContext::IsCreated() const {
	return ( m_hContext != NULL );
}

template<class T>
inline Void CUBLASContext::Copy( CUDADeviceMemory * outDeviceVector, const CUDADeviceMemory * pDeviceVector ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iY = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = outDeviceVector->GetWidth();
	hRegion.iHeight = 0;
	hRegion.iDepth = 0;
	Copy<T>( outDeviceVector, hPosition, pDeviceVector, hPosition, hRegion );
}

template<class T>
inline Void CUBLASContext::Swap( CUDADeviceMemory * pDeviceVectorA, CUDADeviceMemory * pDeviceVectorB ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iY = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pDeviceVectorA->GetWidth();
	hRegion.iHeight = 0;
	hRegion.iDepth = 0;
	Swap<T>( pDeviceVectorA, hPosition, pDeviceVectorB, hPosition, hRegion );
}

template<class T>
inline SizeT CUBLASContext::AbsMin( const CUDADeviceMemory * pVector ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iY = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pVector->GetWidth();
	hRegion.iHeight = 0;
	hRegion.iDepth = 0;
	return AbsMin<T>( pVector, hPosition, hRegion );
}

template<class T>
inline SizeT CUBLASContext::AbsMax( const CUDADeviceMemory * pVector ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iY = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pVector->GetWidth();
	hRegion.iHeight = 0;
	hRegion.iDepth = 0;
	return AbsMax<T>( pVector, hPosition, hRegion );
}

template<class T>
inline T CUBLASContext::AbsSum( const CUDADeviceMemory * pVector ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iY = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pVector->GetWidth();
	hRegion.iHeight = 0;
	hRegion.iDepth = 0;
	return AbsSum<T>( pVector, hPosition, hRegion );
}

template<class T>
inline Void CUBLASContext::MulAdd( CUDADeviceMemory * outVectorY, const CUDADeviceMemory * pVectorX, T fAlpha ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iY = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = outVectorY->GetWidth();
	hRegion.iHeight = 0;
	hRegion.iDepth = 0;
	MulAdd<T>( outVectorY, hPosition, pVectorX, hPosition, fAlpha, hRegion );
}

template<class T>
inline T CUBLASContext::Dot( const CUDADeviceMemory * pVectorA, const CUDADeviceMemory * pVectorB, Bool bConjugateB ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iY = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pVectorA->GetWidth();
	hRegion.iHeight = 0;
	hRegion.iDepth = 0;
	return Dot<T>( pVectorA, hPosition, pVectorB, hPosition, hRegion, bConjugateB );
}

template<class T>
inline T CUBLASContext::Norm( const CUDADeviceMemory * pVector ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iY = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pVector->GetWidth();
	hRegion.iHeight = 0;
	hRegion.iDepth = 0;
	return Norm<T>( pVector, hPosition, hRegion );
}

template<class T>
inline Void CUBLASContext::Scale( CUDADeviceMemory * pVector, T fAlpha ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iY = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pVector->GetWidth();
	hRegion.iHeight = 0;
	hRegion.iDepth = 0;
	Scale<T>( pVector, hPosition, hRegion, fAlpha );
}

