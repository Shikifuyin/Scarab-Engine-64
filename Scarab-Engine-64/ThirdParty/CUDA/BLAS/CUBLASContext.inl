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
	hPosition.iZ = 0;
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
	hPosition.iZ = 0;
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
	hPosition.iZ = 0;
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
	hPosition.iZ = 0;
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
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pVector->GetWidth();
	hRegion.iHeight = 0;
	hRegion.iDepth = 0;
	return AbsSum<T>( pVector, hPosition, hRegion );
}

template<class T>
inline T CUBLASContext::Dot( const CUDADeviceMemory * pVectorA, const CUDADeviceMemory * pVectorB, Bool bConjugateB ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
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
	hPosition.iZ = 0;
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
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pVector->GetWidth();
	hRegion.iHeight = 0;
	hRegion.iDepth = 0;
	Scale<T>( pVector, hPosition, hRegion, fAlpha );
}

template<class T>
inline Void CUBLASContext::MulAdd( CUDADeviceMemory * outVectorY, const CUDADeviceMemory * pVectorX, T fAlpha ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = outVectorY->GetWidth();
	hRegion.iHeight = 0;
	hRegion.iDepth = 0;
	MulAdd<T>( outVectorY, hPosition, pVectorX, hPosition, fAlpha, hRegion );
}

template<class T>
inline Void CUBLASContext::Add( CUDADeviceMemory * outVectorY, const CUDADeviceMemory * pVectorX ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = outVectorY->GetWidth();
	hRegion.iHeight = 0;
	hRegion.iDepth = 0;
	Add<T>( outVectorY, hPosition, pVectorX, hPosition, (T)1, hRegion );
}

template<class T>
inline Void CUBLASContext::MulTriangular( CUDADeviceMemory * outVectorX, const CUDADeviceMemory * pTriangularMatrixA,
										  CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pTriangularMatrixA->GetWidth();
	hRegion.iHeight = pTriangularMatrixA->GetHeight();
	hRegion.iDepth = 0;
	MulTriangular<T>( outVectorX, hPosition, pTriangularMatrixA, hPosition, hRegion, iFillMode, iTransOp, bMainDiagIsUnity );
}

template<class T>
inline Void CUBLASContext::MulTriangularBanded( CUDADeviceMemory * outVectorX, const CUDADeviceMemory * pTriangularBandedMatrixA,
												SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pTriangularBandedMatrixA->GetWidth();
	hRegion.iHeight = pTriangularBandedMatrixA->GetHeight();
	hRegion.iDepth = 0;
	MulTriangularBanded<T>( outVectorX, hPosition, pTriangularBandedMatrixA, hPosition, hRegion, iExpandedSizeA, iSubDiagsCount, iFillMode, iTransOp, bMainDiagIsUnity );
}

template<class T>
inline Void CUBLASContext::MulAdd( CUDADeviceMemory * outVectorY, T fBeta, const CUDADeviceMemory * pVectorX, T fAlpha,
								   const CUDADeviceMemory * pMatrixA, CUBLASContextTransposeOp iTransOp ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pMatrixA->GetWidth();
	hRegion.iHeight = pMatrixA->GetHeight();
	hRegion.iDepth = 0;
	MulAdd<T>( outVectorY, hPosition, fBeta, pVectorX, hPosition, fAlpha, pMatrixA, hPosition, hRegion, iTransOp );
}

template<class T>
inline Void CUBLASContext::MulAddSymmetric( CUDADeviceMemory * outVectorY, T fBeta, const CUDADeviceMemory * pVectorX, T fAlpha,
											const CUDADeviceMemory * pSymmetricMatrixA, CUBLASContextFillMode iFillMode ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pSymmetricMatrixA->GetWidth();
	hRegion.iHeight = pSymmetricMatrixA->GetHeight();
	hRegion.iDepth = 0;
	MulAddSymmetric<T>( outVectorY, hPosition, fBeta, pVectorX, hPosition, fAlpha, pSymmetricMatrixA, hPosition, hRegion, iFillMode );
}

template<class T>
inline Void CUBLASContext::MulAddHermitian( CUDADeviceMemory * outVectorY, T fBeta, const CUDADeviceMemory * pVectorX, T fAlpha,
											const CUDADeviceMemory * pHermitianMatrixA, CUBLASContextFillMode iFillMode ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pHermitianMatrixA->GetWidth();
	hRegion.iHeight = pHermitianMatrixA->GetHeight();
	hRegion.iDepth = 0;
	MulAddHermitian<T>( outVectorY, hPosition, fBeta, pVectorX, hPosition, fAlpha, pHermitianMatrixA, hPosition, hRegion, iFillMode );
}

template<class T>
inline Void CUBLASContext::MulAddBanded( CUDADeviceMemory * outVectorY, T fBeta, const CUDADeviceMemory * pVectorX, T fAlpha,
										 const CUDADeviceMemory * pBandedMatrixA, SizeT iExpandedSizeA, SizeT iLowerDiagsCount, SizeT iUpperDiagsCount, CUBLASContextTransposeOp iTransOp ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pBandedMatrixA->GetWidth();
	hRegion.iHeight = pBandedMatrixA->GetHeight();
	hRegion.iDepth = 0;
	MulAddBanded<T>( outVectorY, hPosition, fBeta, pVectorX, hPosition, fAlpha, pBandedMatrixA, hPosition, hRegion, iExpandedSizeA, iLowerDiagsCount, iUpperDiagsCount, iTransOp );
}

template<class T>
inline Void CUBLASContext::MulAddSymmetricBanded( CUDADeviceMemory * outVectorY, T fBeta, const CUDADeviceMemory * pVectorX, T fAlpha,
												  const CUDADeviceMemory * pSymmetricBandedMatrixA, SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pSymmetricBandedMatrixA->GetWidth();
	hRegion.iHeight = pSymmetricBandedMatrixA->GetHeight();
	hRegion.iDepth = 0;
	MulAddSymmetricBanded<T>( outVectorY, hPosition, fBeta, pVectorX, hPosition, fAlpha, pSymmetricBandedMatrixA, hPosition, hRegion, iExpandedSizeA, iSubDiagsCount, iFillMode );
}

template<class T>
inline Void CUBLASContext::MulAddHermitianBanded( CUDADeviceMemory * outVectorY, T fBeta, const CUDADeviceMemory * pVectorX, T fAlpha,
												  const CUDADeviceMemory * pHermitianBandedMatrixA, SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pHermitianBandedMatrixA->GetWidth();
	hRegion.iHeight = pHermitianBandedMatrixA->GetHeight();
	hRegion.iDepth = 0;
	MulAddHermitianBanded<T>( outVectorY, hPosition, fBeta, pVectorX, hPosition, fAlpha, pHermitianBandedMatrixA, hPosition, hRegion, iExpandedSizeA, iSubDiagsCount, iFillMode );
}

template<class T>
inline Void CUBLASContext::SolveTriangular( CUDADeviceMemory * outVectorX, const CUDADeviceMemory * pTriangularMatrixA,
											CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pTriangularMatrixA->GetWidth();
	hRegion.iHeight = pTriangularMatrixA->GetHeight();
	hRegion.iDepth = 0;
	SolveTriangular<T>( outVectorX, hPosition, pTriangularMatrixA, hPosition, hRegion, iFillMode, iTransOp, bMainDiagIsUnity );
}

template<class T>
inline Void CUBLASContext::SolveTriangularBanded( CUDADeviceMemory * outVectorX, const CUDADeviceMemory * pTriangularBandedMatrixA,
												  SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegion;
	hRegion.iWidth = pTriangularBandedMatrixA->GetWidth();
	hRegion.iHeight = pTriangularBandedMatrixA->GetHeight();
	hRegion.iDepth = 0;
	SolveTriangularBanded<T>( outVectorX, hPosition, pTriangularBandedMatrixA, hPosition, hRegion, iExpandedSizeA, iSubDiagsCount, iFillMode, iTransOp, bMainDiagIsUnity );
}

template<class T>
inline Void CUBLASContext::MulAdd( CUDADeviceMemory * outMatrixC, T fBeta, const CUDADeviceMemory * pMatrixA, T fAlpha, const CUDADeviceMemory * pMatrixB,
								   CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB, Bool bUseComplexGaussReduction ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegionA;
	hRegionA.iWidth = pMatrixA->GetWidth();
	hRegionA.iHeight = pMatrixA->GetHeight();
	hRegionA.iDepth = 0;
	CUDAMemoryRegion hRegionB;
	hRegionA.iWidth = pMatrixB->GetWidth();
	hRegionA.iHeight = pMatrixB->GetHeight();
	hRegionA.iDepth = 0;
	CUDAMemoryRegion hRegionC;
	hRegionA.iWidth = outMatrixC->GetWidth();
	hRegionA.iHeight = outMatrixC->GetHeight();
	hRegionA.iDepth = 0;
	MulAdd<T>( outMatrixC, hPosition, hRegionC, fBeta, pMatrixA, hPosition, hRegionA, fAlpha, pMatrixB, hPosition, hRegionB, iTransOpA, iTransOpB, bUseComplexGaussReduction );
}
