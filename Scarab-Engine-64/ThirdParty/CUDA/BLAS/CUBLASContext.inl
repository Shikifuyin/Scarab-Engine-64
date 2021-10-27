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
inline Void CUBLASContext::MulTriangular( CUDADeviceMemory * outMatrixC, const CUDADeviceMemory * pMatrixA, T fAlpha, const CUDADeviceMemory * pMatrixB,
										  CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegionA;
	hRegionA.iWidth = pMatrixA->GetWidth();
	hRegionA.iHeight = pMatrixA->GetHeight();
	hRegionA.iDepth = 0;
	CUDAMemoryRegion hRegionB;
	hRegionB.iWidth = pMatrixB->GetWidth();
	hRegionB.iHeight = pMatrixB->GetHeight();
	hRegionB.iDepth = 0;
	CUDAMemoryRegion hRegionC;
	hRegionC.iWidth = outMatrixC->GetWidth();
	hRegionC.iHeight = outMatrixC->GetHeight();
	hRegionC.iDepth = 0;
	MulTriangular<T>( outMatrixC, hPosition, hRegionC, pMatrixA, hPosition, hRegionA, fAlpha, pMatrixB, hPosition, hRegionB, iSideMode, iFillMode, iTransOpA, bMainDiagIsUnityA );
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
	hRegionB.iWidth = pMatrixB->GetWidth();
	hRegionB.iHeight = pMatrixB->GetHeight();
	hRegionB.iDepth = 0;
	CUDAMemoryRegion hRegionC;
	hRegionC.iWidth = outMatrixC->GetWidth();
	hRegionC.iHeight = outMatrixC->GetHeight();
	hRegionC.iDepth = 0;
	MulAdd<T>( outMatrixC, hPosition, hRegionC, fBeta, pMatrixA, hPosition, hRegionA, fAlpha, pMatrixB, hPosition, hRegionB, iTransOpA, iTransOpB, bUseComplexGaussReduction );
}

template<class T>
inline Void CUBLASContext::MulAddSymmetric( CUDADeviceMemory * outMatrixC, T fBeta, const CUDADeviceMemory * pMatrixA, T fAlpha, const CUDADeviceMemory * pMatrixB,
											CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegionA;
	hRegionA.iWidth = pMatrixA->GetWidth();
	hRegionA.iHeight = pMatrixA->GetHeight();
	hRegionA.iDepth = 0;
	CUDAMemoryRegion hRegionB;
	hRegionB.iWidth = pMatrixB->GetWidth();
	hRegionB.iHeight = pMatrixB->GetHeight();
	hRegionB.iDepth = 0;
	CUDAMemoryRegion hRegionC;
	hRegionC.iWidth = outMatrixC->GetWidth();
	hRegionC.iHeight = outMatrixC->GetHeight();
	hRegionC.iDepth = 0;
	MulAddSymmetric<T>( outMatrixC, hPosition, hRegionC, fBeta, pMatrixA, hPosition, hRegionA, fAlpha, pMatrixB, hPosition, hRegionB, iSideMode, iFillMode );
}

template<class T>
inline Void CUBLASContext::MulAddHermitian( CUDADeviceMemory * outMatrixC, T fBeta, const CUDADeviceMemory * pMatrixA, T fAlpha, const CUDADeviceMemory * pMatrixB,
											CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegionA;
	hRegionA.iWidth = pMatrixA->GetWidth();
	hRegionA.iHeight = pMatrixA->GetHeight();
	hRegionA.iDepth = 0;
	CUDAMemoryRegion hRegionB;
	hRegionB.iWidth = pMatrixB->GetWidth();
	hRegionB.iHeight = pMatrixB->GetHeight();
	hRegionB.iDepth = 0;
	CUDAMemoryRegion hRegionC;
	hRegionC.iWidth = outMatrixC->GetWidth();
	hRegionC.iHeight = outMatrixC->GetHeight();
	hRegionC.iDepth = 0;
	MulAddHermitian<T>( outMatrixC, hPosition, hRegionC, fBeta, pMatrixA, hPosition, hRegionA, fAlpha, pMatrixB, hPosition, hRegionB, iSideMode, iFillMode );
}

template<class T>
inline Void CUBLASContext::SolveTriangular( CUDADeviceMemory * outMatrixX, const CUDADeviceMemory * pMatrixA, T fAlpha,
											CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA ) const {
	CUDAMemoryPosition hPosition;
	hPosition.iX = 0;
	hPosition.iY = 0;
	hPosition.iZ = 0;
	CUDAMemoryRegion hRegionX;
	hRegionX.iWidth = outMatrixX->GetWidth();
	hRegionX.iHeight = outMatrixX->GetHeight();
	hRegionX.iDepth = 0;
	CUDAMemoryRegion hRegionA;
	hRegionA.iWidth = pMatrixA->GetWidth();
	hRegionA.iHeight = pMatrixA->GetHeight();
	hRegionA.iDepth = 0;
	SolveTriangular<T>( outMatrixX, hPosition, hRegionX, pMatrixA, hPosition, hRegionA, fAlpha, iSideMode, iFillMode, iTransOpA, bMainDiagIsUnityA );
}
