/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/BLAS/CUBLASMatrixMatrixOp.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA BLAS : Matrix-Matrix Operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// CUBLASMatrixMatrixOp implementation
inline Void CUBLASMatrixMatrixOp::SetMatrixA( const CUDADeviceMemory * pMatrix ) {
	DebugAssert( pMatrix != NULL );
	m_pMatrixA = pMatrix;
	SetMatrixPositionA();
	SetMatrixRegionA();
}
inline Void CUBLASMatrixMatrixOp::SetMatrixPositionA( const CUDAMemoryPosition * pPosition ) {
	DebugAssert( m_pMatrixA != NULL );
	if ( pPosition != NULL )
		m_hMatrixPositionA = *pPosition;
	else {
		m_hMatrixPositionA.iX = 0;
		m_hMatrixPositionA.iY = 0;
		m_hMatrixPositionA.iZ = 0;
	}
	DebugAssert( m_pMatrixA->IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA ) );
}
inline Void CUBLASMatrixMatrixOp::SetMatrixRegionA( const CUDAMemoryRegion * pRegion ) {
	DebugAssert( m_pMatrixA != NULL );
	if ( pRegion != NULL )
		m_hMatrixRegionA = *pRegion;
	else {
		m_hMatrixRegionA.iWidth = m_pMatrixA->GetWidth();
		m_hMatrixRegionA.iHeight = m_pMatrixA->GetHeight();
		m_hMatrixRegionA.iDepth = 0;
	}
	DebugAssert( m_pMatrixA->IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA ) );
}
inline Void CUBLASMatrixMatrixOp::SetMatrixA( const CUDADeviceMemory * pMatrix, const CUDAMemoryPosition * pPosition, const CUDAMemoryRegion * pRegion ) {
	DebugAssert( pMatrix != NULL );
	m_pMatrixA = pMatrix;
	SetMatrixPositionA( pPosition );
	SetMatrixRegionA( pRegion );
}

inline Void CUBLASMatrixMatrixOp::SetMatrixB( const CUDADeviceMemory * pMatrix ) {
	DebugAssert( pMatrix != NULL );
	m_pMatrixB = pMatrix;
	SetMatrixPositionB();
	SetMatrixRegionB();
}
inline Void CUBLASMatrixMatrixOp::SetMatrixPositionB( const CUDAMemoryPosition * pPosition ) {
	DebugAssert( m_pMatrixB != NULL );
	if ( pPosition != NULL )
		m_hMatrixPositionB = *pPosition;
	else {
		m_hMatrixPositionB.iX = 0;
		m_hMatrixPositionB.iY = 0;
		m_hMatrixPositionB.iZ = 0;
	}
	DebugAssert( m_pMatrixB->IsValidRegion( m_hMatrixPositionB, m_hMatrixRegionB ) );
}
inline Void CUBLASMatrixMatrixOp::SetMatrixRegionB( const CUDAMemoryRegion * pRegion ) {
	DebugAssert( m_pMatrixB != NULL );
	if ( pRegion != NULL )
		m_hMatrixRegionB = *pRegion;
	else {
		m_hMatrixRegionB.iWidth = m_pMatrixB->GetWidth();
		m_hMatrixRegionB.iHeight = m_pMatrixB->GetHeight();
		m_hMatrixRegionB.iDepth = 0;
	}
	DebugAssert( m_pMatrixB->IsValidRegion( m_hMatrixPositionB, m_hMatrixRegionB ) );
}
inline Void CUBLASMatrixMatrixOp::SetMatrixB( const CUDADeviceMemory * pMatrix, const CUDAMemoryPosition * pPosition, const CUDAMemoryRegion * pRegion ) {
	DebugAssert( pMatrix != NULL );
	m_pMatrixB = pMatrix;
	SetMatrixPositionB( pPosition );
	SetMatrixRegionB( pRegion );
}

inline Void CUBLASMatrixMatrixOp::SetMatrixC( CUDADeviceMemory * pMatrix ) {
	DebugAssert( pMatrix != NULL );
	m_pMatrixC = pMatrix;
	SetMatrixPositionC();
	SetMatrixRegionC();
}
inline Void CUBLASMatrixMatrixOp::SetMatrixPositionC( const CUDAMemoryPosition * pPosition ) {
	DebugAssert( m_pMatrixC != NULL );
	if ( pPosition != NULL )
		m_hMatrixPositionC = *pPosition;
	else {
		m_hMatrixPositionC.iX = 0;
		m_hMatrixPositionC.iY = 0;
		m_hMatrixPositionC.iZ = 0;
	}
	DebugAssert( m_pMatrixC->IsValidRegion( m_hMatrixPositionC, m_hMatrixRegionC ) );
}
inline Void CUBLASMatrixMatrixOp::SetMatrixRegionC( const CUDAMemoryRegion * pRegion ) {
	DebugAssert( m_pMatrixC != NULL );
	if ( pRegion != NULL )
		m_hMatrixRegionC = *pRegion;
	else {
		m_hMatrixRegionC.iWidth = m_pMatrixC->GetWidth();
		m_hMatrixRegionC.iHeight = m_pMatrixC->GetHeight();
		m_hMatrixRegionC.iDepth = 0;
	}
	DebugAssert( m_pMatrixC->IsValidRegion( m_hMatrixPositionC, m_hMatrixRegionC ) );
}
inline Void CUBLASMatrixMatrixOp::SetMatrixC( CUDADeviceMemory * pMatrix, const CUDAMemoryPosition * pPosition, const CUDAMemoryRegion * pRegion ) {
	DebugAssert( pMatrix != NULL );
	m_pMatrixC = pMatrix;
	SetMatrixPositionC( pPosition );
	SetMatrixRegionC( pRegion );
}

inline CUDADeviceMemory * CUBLASMatrixMatrixOp::GetMatrixC( CUDAMemoryPosition * outPosition, CUDAMemoryRegion * outRegion ) const {
	if ( outPosition != NULL )
		*outPosition = m_hMatrixPositionC;
	if ( outRegion != NULL )
		*outRegion = m_hMatrixRegionC;
	return m_pMatrixC;
}

template<class T>
inline Bool CUBLASMatrixMatrixOp::ValidateInputA() const {
	return (
		m_pMatrixA != NULL
		&& m_pMatrixA->IsAllocated()
		&& m_pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D
		&& m_pMatrixA->GetStride() == sizeof(T)
		&& m_pMatrixA->IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA )
		&& m_pMatrixC != NULL
		&& m_pMatrixC->IsAllocated()
		&& m_pMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D
		&& m_pMatrixC->GetStride() == sizeof(T)
		&& m_pMatrixC->IsValidRegion( m_hMatrixPositionC, m_hMatrixRegionC )
	);
}
template<class T>
inline Bool CUBLASMatrixMatrixOp::ValidateInputAB() const {
	return (
		m_pMatrixA != NULL
		&& m_pMatrixA->IsAllocated()
		&& m_pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D
		&& m_pMatrixA->GetStride() == sizeof(T)
		&& m_pMatrixA->IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA )
		&& m_pMatrixB != NULL
		&& m_pMatrixB->IsAllocated()
		&& m_pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D
		&& m_pMatrixB->GetStride() == sizeof(T)
		&& m_pMatrixB->IsValidRegion( m_hMatrixPositionB, m_hMatrixRegionB )
		&& m_pMatrixC != NULL
		&& m_pMatrixC->IsAllocated()
		&& m_pMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D
		&& m_pMatrixC->GetStride() == sizeof(T)
		&& m_pMatrixC->IsValidRegion( m_hMatrixPositionC, m_hMatrixRegionC )
	);
}

template<class T>
inline Bool CUBLASMatrixMatrixOp::ValidateInputBatchedA( SizeT iBatchIndex, Bool bStripMode ) const {
	if ( bStripMode ) {
		return (
			m_pMatrixA != NULL
			&& m_pMatrixA->IsAllocated()
			&& m_pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_3D
			&& m_pMatrixA->GetStride() == sizeof(T)
			&& m_pMatrixA->IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA )
			&& m_pMatrixC != NULL
			&& m_pMatrixC->IsAllocated()
			&& m_pMatrixC->GetShape() == CUDA_MEMORY_SHAPE_3D
			&& m_pMatrixC->GetStride() == sizeof(T)
			&& m_pMatrixC->IsValidRegion( m_hMatrixPositionC, m_hMatrixRegionC )
		);
	} else {
		return (
			m_pMatrixA != NULL
			&& m_pMatrixA[iBatchIndex].IsAllocated()
			&& m_pMatrixA[iBatchIndex].GetShape() == CUDA_MEMORY_SHAPE_2D
			&& m_pMatrixA[iBatchIndex].GetStride() == sizeof(T)
			&& m_pMatrixA[iBatchIndex].IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA )
			&& m_pMatrixC != NULL
			&& m_pMatrixC[iBatchIndex].IsAllocated()
			&& m_pMatrixC[iBatchIndex].GetShape() == CUDA_MEMORY_SHAPE_2D
			&& m_pMatrixC[iBatchIndex].GetStride() == sizeof(T)
			&& m_pMatrixC[iBatchIndex].IsValidRegion( m_hMatrixPositionC, m_hMatrixRegionC )
		);
	}
}
template<class T>
inline Bool CUBLASMatrixMatrixOp::ValidateInputBatchedAB( SizeT iBatchIndex, Bool bStripMode ) const {
	if ( bStripMode ) {
		return (
			m_pMatrixA != NULL
			&& m_pMatrixA->IsAllocated()
			&& m_pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_3D
			&& m_pMatrixA->GetStride() == sizeof(T)
			&& m_pMatrixA->IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA )
			&& m_pMatrixB != NULL
			&& m_pMatrixB->IsAllocated()
			&& m_pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_3D
			&& m_pMatrixB->GetStride() == sizeof(T)
			&& m_pMatrixB->IsValidRegion( m_hMatrixPositionB, m_hMatrixRegionB )
			&& m_pMatrixC != NULL
			&& m_pMatrixC->IsAllocated()
			&& m_pMatrixC->GetShape() == CUDA_MEMORY_SHAPE_3D
			&& m_pMatrixC->GetStride() == sizeof(T)
			&& m_pMatrixC->IsValidRegion( m_hMatrixPositionC, m_hMatrixRegionC )
		);
	} else {
		return (
			m_pMatrixA != NULL
			&& m_pMatrixA[iBatchIndex].IsAllocated()
			&& m_pMatrixA[iBatchIndex].GetShape() == CUDA_MEMORY_SHAPE_2D
			&& m_pMatrixA[iBatchIndex].GetStride() == sizeof(T)
			&& m_pMatrixA[iBatchIndex].IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA )
			&& m_pMatrixB != NULL
			&& m_pMatrixB[iBatchIndex].IsAllocated()
			&& m_pMatrixB[iBatchIndex].GetShape() == CUDA_MEMORY_SHAPE_2D
			&& m_pMatrixB[iBatchIndex].GetStride() == sizeof(T)
			&& m_pMatrixB[iBatchIndex].IsValidRegion( m_hMatrixPositionB, m_hMatrixRegionB )
			&& m_pMatrixC != NULL
			&& m_pMatrixC[iBatchIndex].IsAllocated()
			&& m_pMatrixC[iBatchIndex].GetShape() == CUDA_MEMORY_SHAPE_2D
			&& m_pMatrixC[iBatchIndex].GetStride() == sizeof(T)
			&& m_pMatrixC[iBatchIndex].IsValidRegion( m_hMatrixPositionC, m_hMatrixRegionC )
		);
	}
}

template<class T>
Void CUBLASMatrixMatrixOp::MulAdd( T fScaleA, T fScaleC, CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB, Bool bUseComplexGaussReduction )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputAB<T>() );

	// Specific Input Validation
	SizeT iM, iN, iK;
	if ( iTransOpA == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionC.iWidth );
		iM = m_hMatrixRegionA.iWidth;
		iK = m_hMatrixRegionA.iHeight;
	} else {
		DebugAssert( m_hMatrixRegionA.iHeight == m_hMatrixRegionC.iWidth );
		iM = m_hMatrixRegionA.iHeight;
		iK = m_hMatrixRegionA.iWidth;
	}
	if ( iTransOpB == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( m_hMatrixRegionB.iHeight == m_hMatrixRegionC.iHeight );
		iN = m_hMatrixRegionB.iHeight;
		DebugAssert( iK == m_hMatrixRegionB.iWidth );
	} else {
		DebugAssert( m_hMatrixRegionB.iWidth == m_hMatrixRegionC.iHeight );
		iN = m_hMatrixRegionB.iWidth;
		DebugAssert( iK == m_hMatrixRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );

	cublasStatus_t iError;
	switch( typeid(T) ) {
		case typeid(CUDAReal32):
			iError = cublasSgemm( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
								  &fScaleA, (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (const Float *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
								  &fScaleC, (Float*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		case typeid(CUDAReal64):
			iError = cublasDgemm( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
								  &fScaleA, (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (const Double *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
								  &fScaleC, (Double*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		case typeid(CUDAComplex32):
			if ( bUseComplexGaussReduction ) {
				iError = cublasCgemm3m( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
										&fScaleA, (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
										(const cuComplex *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
										&fScaleC, (cuComplex *)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			} else {
				iError = cublasCgemm( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
									  &fScaleA, (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
									  (const cuComplex *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
									  &fScaleC, (cuComplex *)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			}
			break;
		case typeid(CUDAComplex64):
			if ( bUseComplexGaussReduction ) {
				iError = cublasZgemm3m( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
										&fScaleA, (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
										(const cuDoubleComplex *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
										&fScaleC, (cuDoubleComplex*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			} else {
				iError = cublasZgemm( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
									  &fScaleA, (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
									  (const cuDoubleComplex *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
									  &fScaleC, (cuDoubleComplex*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			}
			break;
		default: DebugAssert(false); break;
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<class T>
Void CUBLASMatrixMatrixOp::MulAddSymmetric( T fScaleA, T fScaleC, CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputAB<T>() );

	// Specific Input Validation
	DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionA.iHeight );
	DebugAssert( m_hMatrixRegionB.iWidth == m_hMatrixRegionC.iWidth );
	DebugAssert( m_hMatrixRegionB.iHeight == m_hMatrixRegionC.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( m_hMatrixRegionA.iHeight == m_hMatrixRegionB.iWidth );
	} else {
		DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );

	cublasStatus_t iError;
	switch( typeid(T) ) {
		case typeid(CUDAReal32):
			iError = cublasSsymm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
								  &fScaleA, (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (const Float *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
								  &fScaleC, (Float*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		case typeid(CUDAReal64):
			iError = cublasDsymm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
								  &fScaleA, (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (const Double *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
								  &fScaleC, (Double*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		case typeid(CUDAComplex32):
			iError = cublasCsymm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
								  &fScaleA, (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (const cuComplex *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
								  &fScaleC, (cuComplex *)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		case typeid(CUDAComplex64):
			iError = cublasZsymm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
								  &fScaleA, (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (const cuDoubleComplex *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
								  &fScaleC, (cuDoubleComplex*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		default: DebugAssert(false); break;
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<class T>
Void CUBLASMatrixMatrixOp::MulAddHermitian( T fScaleA, T fScaleC, CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputAB<T>() );

	// Specific Input Validation
	DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionA.iHeight );
	DebugAssert( m_hMatrixRegionB.iWidth == m_hMatrixRegionC.iWidth );
	DebugAssert( m_hMatrixRegionB.iHeight == m_hMatrixRegionC.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( m_hMatrixRegionA.iHeight == m_hMatrixRegionB.iWidth );
	} else {
		DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );

	cublasStatus_t iError;
	switch( typeid(T) ) {
		case typeid(CUDAComplex32):
			iError = cublasChemm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
								  &fScaleA, (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (const cuComplex *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
								  &fScaleC, (cuComplex *)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		case typeid(CUDAComplex64):
			iError = cublasZhemm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
								  &fScaleA, (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (const cuDoubleComplex *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
								  &fScaleC, (cuDoubleComplex*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		default: DebugAssert(false); break;
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<class T>
Void CUBLASMatrixMatrixOp::MulTriangular( T fScaleA, CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputAB<T>() );

	// Specific Input Validation
	DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionA.iHeight );
	DebugAssert( m_hMatrixRegionB.iWidth == m_hMatrixRegionC.iWidth );
	DebugAssert( m_hMatrixRegionB.iHeight == m_hMatrixRegionC.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( m_hMatrixRegionA.iHeight == m_hMatrixRegionB.iWidth );
	} else {
		DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );

	cublasStatus_t iError;
	switch( typeid(T) ) {
		case typeid(CUDAReal32):
			iError = cublasStrmm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
								  m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
								  &fScaleA, (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (const Float *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
								  (Float*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		case typeid(CUDAReal64):
			iError = cublasDtrmm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
								  m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
								  &fScaleA, (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (const Double *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
								  (Double*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		case typeid(CUDAComplex32):
			iError = cublasCtrmm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
								  m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
								  &fScaleA, (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (const cuComplex *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
								  (cuComplex *)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		case typeid(CUDAComplex64):
			iError = cublasZtrmm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
								  m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
								  &fScaleA, (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (const cuDoubleComplex *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(),
								  (cuDoubleComplex*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		default: DebugAssert(false); break;
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<class T>
Void CUBLASMatrixMatrixOp::SolveTriangular( T fScaleA, CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputA<T>() );

	// Specific Input Validation
	DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionA.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( m_hMatrixRegionA.iHeight == m_hMatrixRegionC.iWidth );
	} else {
		DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionC.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );

	cublasStatus_t iError;
	switch( typeid(T) ) {
		case typeid(CUDAReal32):
			iError = cublasStrsm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
								  m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
								  &fScaleA, (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (Float*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		case typeid(CUDAReal64):
			iError = cublasDtrsm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
								  m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
								  &fScaleA, (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (Double*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		case typeid(CUDAComplex32):
			iError = cublasCtrsm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
								  m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
								  &fScaleA, (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (cuComplex *)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		case typeid(CUDAComplex64):
			iError = cublasZtrsm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
								  m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
								  &fScaleA, (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (cuDoubleComplex*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth() );
			break;
		default: DebugAssert(false); break;
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<class T>
Void CUBLASMatrixMatrixOp::MulAddBatched( T fScaleA, T fScaleC, CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB, SizeT iBatchCount, Bool bStripMode )
{
	DebugAssert( m_pCUBLASContext != NULL );

	// Empty Call
	if ( iBatchCount == 0 )
		return;

	// Specific Input Validation
	SizeT iM, iN, iK;
	if ( iTransOpA == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionC.iWidth );
		iM = m_hMatrixRegionA.iWidth;
		iK = m_hMatrixRegionA.iHeight;
	} else {
		DebugAssert( m_hMatrixRegionA.iHeight == m_hMatrixRegionC.iWidth );
		iM = m_hMatrixRegionA.iHeight;
		iK = m_hMatrixRegionA.iWidth;
	}
	if ( iTransOpB == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( m_hMatrixRegionB.iHeight == m_hMatrixRegionC.iHeight );
		iN = m_hMatrixRegionB.iHeight;
		DebugAssert( iK == m_hMatrixRegionB.iWidth );
	} else {
		DebugAssert( m_hMatrixRegionB.iWidth == m_hMatrixRegionC.iHeight );
		iN = m_hMatrixRegionB.iWidth;
		DebugAssert( iK == m_hMatrixRegionB.iHeight );
	}

	if ( bStripMode ) {
		DebugAssert( ValidateInputBatchedAB<T>(0, bStripMode) );
		DebugAssert( iBatchCount <= m_pMatrixA->GetDepth() );
		DebugAssert( iBatchCount <= m_pMatrixB->GetDepth() );
		DebugAssert( iBatchCount <= m_pMatrixC->GetDepth() );

		cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
		cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
		cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );

		cublasStatus_t iError;
		switch( typeid(T) ) {
			case typeid(CUDAReal32):
				iError = cublasSgemmStridedBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
													&fScaleA, (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(), m_pMatrixA->GetWidth() * m_hMatrixRegionA.iHeight,
													(const Float *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(), m_pMatrixB->GetWidth() * m_hMatrixRegionB.iHeight,
													&fScaleC, (Float*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth(), m_pMatrixC->GetWidth() * m_hMatrixRegionC.iHeight, iBatchCount );
				break;
			case typeid(CUDAReal64):
				iError = cublasDgemmStridedBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
													&fScaleA, (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(), m_pMatrixA->GetWidth() * m_hMatrixRegionA.iHeight,
													(const Double *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(), m_pMatrixB->GetWidth() * m_hMatrixRegionB.iHeight,
													&fScaleC, (Double*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth(), m_pMatrixC->GetWidth() * m_hMatrixRegionC.iHeight, iBatchCount );
				break;
			case typeid(CUDAComplex32):
				iError = cublasCgemmStridedBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
													&fScaleA, (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(), m_pMatrixA->GetWidth() * m_hMatrixRegionA.iHeight,
													(const cuComplex *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(), m_pMatrixB->GetWidth() * m_hMatrixRegionB.iHeight,
													&fScaleC, (cuComplex*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth(), m_pMatrixC->GetWidth() * m_hMatrixRegionC.iHeight, iBatchCount );
				break;
			case typeid(CUDAComplex64):
				iError = cublasZgemmStridedBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
													&fScaleA, (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(), m_pMatrixA->GetWidth() * m_hMatrixRegionA.iHeight,
													(const cuDoubleComplex *)( m_pMatrixB->GetPointer(m_hMatrixPositionB) ), m_pMatrixB->GetWidth(), m_pMatrixB->GetWidth() * m_hMatrixRegionB.iHeight,
													&fScaleC, (cuDoubleComplex*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) ), m_pMatrixC->GetWidth(), m_pMatrixC->GetWidth() * m_hMatrixRegionC.iHeight, iBatchCount );
				break;
			default: DebugAssert(false); break;
		}
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	} else {
		DebugAssert( iBatchCount <= CUBLAS_BATCH_MAX_COUNT );

		// Prepare Batch Data
		const T * arrBatchMatricesA[CUBLAS_BATCH_MAX_COUNT];
		const T * arrBatchMatricesB[CUBLAS_BATCH_MAX_COUNT];
		T * arrBatchMatricesC[CUBLAS_BATCH_MAX_COUNT];

		SizeT iReferenceWidthA = m_pMatrixA[0].GetWidth();
		SizeT iReferenceWidthB = m_pMatrixB[0].GetWidth();
		SizeT iReferenceWidthC = m_pMatrixC[0].GetWidth();

		for( SizeT i = 0; i < iBatchCount; ++i ) {
			DebugAssert( ValidateInputBatchedAB<T>(i, bStripMode) );

			DebugAssert( m_pMatrixA[i].GetWidth() == iReferenceWidthA );
			DebugAssert( m_pMatrixB[i].GetWidth() == iReferenceWidthB );
			DebugAssert( m_pMatrixC[i].GetWidth() == iReferenceWidthC );

			arrBatchMatricesA[i] = (const T *)( m_pMatrixA[i].GetPointer(m_hMatrixPositionA) );
			arrBatchMatricesB[i] = (const T *)( m_pMatrixB[i].GetPointer(m_hMatrixPositionB) );
			arrBatchMatricesC[i] = (T*)( m_pMatrixC[i].GetPointer(m_hMatrixPositionC) );
		}

		cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
		cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
		cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );

		cublasStatus_t iError;
		switch( typeid(T) ) {
			case typeid(CUDAReal32):
				iError = cublasSgemmBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
											 &fScaleA, arrBatchMatricesA, iReferenceWidthA,
											 arrBatchMatricesB, iReferenceWidthB,
											 &fScaleC, arrBatchMatricesC, iReferenceWidthC, iBatchCount );
				break;
			case typeid(CUDAReal64):
				iError = cublasDgemmBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
											 &fScaleA, arrBatchMatricesA, iReferenceWidthA,
											 arrBatchMatricesB, iReferenceWidthB,
											 &fScaleC, arrBatchMatricesC, iReferenceWidthC, iBatchCount );
				break;
			case typeid(CUDAComplex32):
				iError = cublasCgemmBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
											 &fScaleA, arrBatchMatricesA, iReferenceWidthA,
											 arrBatchMatricesB, iReferenceWidthB,
											 &fScaleC, arrBatchMatricesC, iReferenceWidthC, iBatchCount );
				break;
			case typeid(CUDAComplex64):
				iError = cublasZgemmBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
											 &fScaleA, arrBatchMatricesA, iReferenceWidthA,
											 arrBatchMatricesB, iReferenceWidthB,
											 &fScaleC, arrBatchMatricesC, iReferenceWidthC, iBatchCount );
				break;
			default: DebugAssert(false); break;
		}
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	}
}

template<class T>
Void CUBLASMatrixMatrixOp::SolveTriangularBatched( T fScaleA, CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA, SizeT iBatchCount, Bool bStripMode )
{
	DebugAssert( m_pCUBLASContext != NULL );

	// Empty Call
	if ( iBatchCount == 0 )
		return;

	// Specific Input Validation
	DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionA.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( m_hMatrixRegionA.iHeight == m_hMatrixRegionC.iWidth );
	} else {
		DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionC.iHeight );
	}

	if ( bStripMode ) {
		// UNAVAILABLE AS OF NOW !
		DebugAssert( false );
	} else {
		DebugAssert( iBatchCount <= CUBLAS_BATCH_MAX_COUNT );

		// Prepare Batch Data
		const T * arrBatchMatricesA[CUBLAS_BATCH_MAX_COUNT];
		T * arrBatchMatricesC[CUBLAS_BATCH_MAX_COUNT];

		SizeT iReferenceWidthA = m_pMatrixA[0].GetWidth();
		SizeT iReferenceWidthC = m_pMatrixC[0].GetWidth();

		for( SizeT i = 0; i < iBatchCount; ++i ) {
			DebugAssert( ValidateInputBatchedA<T>(i, bStripMode) );

			DebugAssert( m_pMatrixA[i].GetWidth() == iReferenceWidthA );
			DebugAssert( m_pMatrixC[i].GetWidth() == iReferenceWidthC );

			arrBatchMatricesA[i] = (const T *)( m_pMatrixA[i].GetPointer(m_hMatrixPositionA) );
			arrBatchMatricesC[i] = (T*)( m_pMatrixC[i].GetPointer(m_hMatrixPositionC) );
		}

		cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
		cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
		cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
		cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );

		cublasStatus_t iError;
		switch( typeid(T) ) {
			case typeid(CUDAReal32):
				iError = cublasStrsmBatched( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
											 m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
											 &fScaleA, arrBatchMatricesA, iReferenceWidthA,
											 arrBatchMatricesC, iReferenceWidthC, iBatchCount );
				break;
			case typeid(CUDAReal64):
				iError = cublasDtrsmBatched( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
											 m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
											 &fScaleA, arrBatchMatricesA, iReferenceWidthA,
											 arrBatchMatricesC, iReferenceWidthC, iBatchCount );
				break;
			case typeid(CUDAComplex32):
				iError = cublasCtrsmBatched( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
											 m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
											 &fScaleA, arrBatchMatricesA, iReferenceWidthA,
											 arrBatchMatricesC, iReferenceWidthC, iBatchCount );
				break;
			case typeid(CUDAComplex64):
				iError = cublasZtrsmBatched( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
											 m_hMatrixRegionC.iWidth, m_hMatrixRegionC.iHeight,
											 &fScaleA, arrBatchMatricesA, iReferenceWidthA,
											 arrBatchMatricesC, iReferenceWidthC, iBatchCount );
				break;
			default: DebugAssert(false); break;
		}
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	}
}

