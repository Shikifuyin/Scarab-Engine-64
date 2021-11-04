/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/BLAS/CUBLASMatrixVectorOp.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA BLAS : Matrix-Vector Operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// CUBLASMatrixVectorOp implementation
inline Void CUBLASMatrixVectorOp::SetMatrixA( const CUDADeviceMemory * pMatrix ) {
	DebugAssert( pMatrix != NULL );
	m_pMatrixA = pMatrix;
	SetMatrixPositionA();
	SetMatrixRegionA();
}
inline Void CUBLASMatrixVectorOp::SetMatrixPositionA( const CUDAMemoryPosition * pPosition ) {
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
inline Void CUBLASMatrixVectorOp::SetMatrixRegionA( const CUDAMemoryRegion * pRegion ) {
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
inline Void CUBLASMatrixVectorOp::SetMatrixA( const CUDADeviceMemory * pMatrix, const CUDAMemoryPosition * pPosition, const CUDAMemoryRegion * pRegion ) {
	DebugAssert( pMatrix != NULL );
	m_pMatrixA = pMatrix;
	SetMatrixPositionA( pPosition );
	SetMatrixRegionA( pRegion );
}

inline Void CUBLASMatrixVectorOp::SetVectorX( CUDADeviceMemory * pVector ) {
	DebugAssert( pVector != NULL );
	m_pVectorX = pVector;
	SetVectorPositionX();
}
inline Void CUBLASMatrixVectorOp::SetVectorPositionX( const CUDAMemoryPosition * pPosition ) {
	DebugAssert( m_pVectorX != NULL );
	if ( pPosition != NULL )
		m_hVectorPositionX = *pPosition;
	else {
		m_hVectorPositionX.iX = 0;
		m_hVectorPositionX.iY = 0;
		m_hVectorPositionX.iZ = 0;
	}
	DebugAssert( m_pVectorX->IsValidPosition( m_hVectorPositionX ) );
}
inline Void CUBLASMatrixVectorOp::SetVectorX( CUDADeviceMemory * pVector, const CUDAMemoryPosition * pPosition ) {
	DebugAssert( pVector != NULL );
	m_pVectorX = pVector;
	SetVectorPositionX( pPosition );
}

inline CUDADeviceMemory * CUBLASMatrixVectorOp::GetVectorX( CUDAMemoryPosition * outPosition, CUDAMemoryRegion * outRegion ) const {
	if ( outPosition != NULL )
		*outPosition = m_hVectorPositionX;
	if ( outRegion != NULL )
		*outRegion = m_hMatrixRegionA;
	return m_pVectorX;
}

inline Void CUBLASMatrixVectorOp::SetVectorY( CUDADeviceMemory * pVector ) {
	DebugAssert( pVector != NULL );
	m_pVectorY = pVector;
	SetVectorPositionY();
}
inline Void CUBLASMatrixVectorOp::SetVectorPositionY( const CUDAMemoryPosition * pPosition ) {
	DebugAssert( m_pVectorY != NULL );
	if ( pPosition != NULL )
		m_hVectorPositionY = *pPosition;
	else {
		m_hVectorPositionY.iX = 0;
		m_hVectorPositionY.iY = 0;
		m_hVectorPositionY.iZ = 0;
	}
	DebugAssert( m_pVectorY->IsValidPosition( m_hVectorPositionY ) );
}
inline Void CUBLASMatrixVectorOp::SetVectorY( CUDADeviceMemory * pVector, const CUDAMemoryPosition * pPosition ) {
	DebugAssert( pVector != NULL );
	m_pVectorY = pVector;
	SetVectorPositionY( pPosition );
}

inline CUDADeviceMemory * CUBLASMatrixVectorOp::GetVectorY( CUDAMemoryPosition * outPosition, CUDAMemoryRegion * outRegion ) const {
	if ( outPosition != NULL )
		*outPosition = m_hVectorPositionY;
	if ( outRegion != NULL )
		*outRegion = m_hMatrixRegionA;
	return m_pVectorY;
}

template<class T>
inline Bool CUBLASMatrixVectorOp::ValidateInputX() const {
	return (
		m_pMatrixA != NULL
		&& m_pMatrixA->IsAllocated()
		&& m_pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D
		&& m_pMatrixA->GetStride() == sizeof(T)
		&& m_pMatrixA->IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA )
		&& m_pVectorX != NULL
		&& m_pVectorX->IsAllocated()
		&& m_pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D
		&& m_pVectorX->GetStride() == sizeof(T)
		&& m_pVectorX->IsValidPosition( m_hVectorPositionX )
	);
}
template<class T>
inline Bool CUBLASMatrixVectorOp::ValidateInputXY() const {
	return (
		m_pMatrixA != NULL
		&& m_pMatrixA->IsAllocated()
		&& m_pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D
		&& m_pMatrixA->GetStride() == sizeof(T)
		&& m_pMatrixA->IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA )
		&& m_pVectorX != NULL
		&& m_pVectorX->IsAllocated()
		&& m_pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D
		&& m_pVectorX->GetStride() == sizeof(T)
		&& m_pVectorX->IsValidPosition( m_hVectorPositionX )
		&& m_pVectorY != NULL
		&& m_pVectorY->IsAllocated()
		&& m_pVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D
		&& m_pVectorY->GetStride() == sizeof(T)
		&& m_pVectorY->IsValidPosition( m_hVectorPositionY )
	);
}

template<class T>
Void CUBLASMatrixVectorOp::MulAdd( T fScaleX, T fScaleY, CUBLASContextTransposeOp iTransOp )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputXY<T>() );
	
	// Specific Input Validation
	CUDAMemoryRegion hRegionX, hRegionY;
	hRegionX.iWidth = 0;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	hRegionY.iWidth = 0;
	hRegionY.iHeight = 0;
	hRegionY.iDepth = 0;
	if ( iTransOp == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		hRegionX.iWidth = m_hMatrixRegionA.iHeight;
		hRegionY.iWidth = m_hMatrixRegionA.iWidth;
	} else {
		hRegionX.iWidth = m_hMatrixRegionA.iWidth;
		hRegionY.iWidth = m_hMatrixRegionA.iHeight;
	}
	DebugAssert( m_pVectorX->IsValidRegion(m_hVectorPositionX, hRegionX) );
	DebugAssert( m_pVectorY->IsValidRegion(m_hVectorPositionY, hRegionY) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasSgemv( hCUBLASContext, iCUBLASTransposeOp, m_hMatrixRegionA.iWidth, m_hMatrixRegionA.iHeight,
							  (const Float *)&fScaleX, (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const Float *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const Float *)&fScaleY, (Float*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDgemv( hCUBLASContext, iCUBLASTransposeOp, m_hMatrixRegionA.iWidth, m_hMatrixRegionA.iHeight,
							  (const Double *)&fScaleX, (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const Double *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const Double *)&fScaleY, (Double*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasCgemv( hCUBLASContext, iCUBLASTransposeOp, m_hMatrixRegionA.iWidth, m_hMatrixRegionA.iHeight,
							  (const cuComplex *)&fScaleX, (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const cuComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const cuComplex *)&fScaleY, (cuComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasZgemv( hCUBLASContext, iCUBLASTransposeOp, m_hMatrixRegionA.iWidth, m_hMatrixRegionA.iHeight,
							  (const cuDoubleComplex *)&fScaleX, (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const cuDoubleComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const cuDoubleComplex *)&fScaleY, (cuDoubleComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<class T>
Void CUBLASMatrixVectorOp::MulAddSymmetric( T fScaleX, T fScaleY, CUBLASContextFillMode iFillMode )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputXY<T>() );
	
	// Specific Input Validation
	DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionA.iHeight );
	DebugAssert( m_pVectorX->IsValidRegion(m_hVectorPositionX, m_hMatrixRegionA) );
	DebugAssert( m_pVectorY->IsValidRegion(m_hVectorPositionY, m_hMatrixRegionA) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasSsymv( hCUBLASContext, iCUBLASFillMode, m_hMatrixRegionA.iWidth,
							  (const Float *)&fScaleX, (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const Float *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const Float *)&fScaleY, (Float*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDsymv( hCUBLASContext, iCUBLASFillMode, m_hMatrixRegionA.iWidth,
							  (const Double *)&fScaleX, (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const Double *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const Double *)&fScaleY, (Double*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasCsymv( hCUBLASContext, iCUBLASFillMode, m_hMatrixRegionA.iWidth,
							  (const cuComplex *)&fScaleX, (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const cuComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const cuComplex *)&fScaleY, (cuComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasZsymv( hCUBLASContext, iCUBLASFillMode, m_hMatrixRegionA.iWidth,
							  (const cuDoubleComplex *)&fScaleX, (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const cuDoubleComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const cuDoubleComplex *)&fScaleY, (cuDoubleComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<class T>
Void CUBLASMatrixVectorOp::MulAddHermitian( T fScaleX, T fScaleY, CUBLASContextFillMode iFillMode )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputXY<T>() );
	
	// Specific Input Validation
	DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionA.iHeight );
	DebugAssert( m_pVectorX->IsValidRegion(m_hVectorPositionX, m_hMatrixRegionA) );
	DebugAssert( m_pVectorY->IsValidRegion(m_hVectorPositionY, m_hMatrixRegionA) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasChemv( hCUBLASContext, iCUBLASFillMode, m_hMatrixRegionA.iWidth,
							  (const cuComplex *)&fScaleX, (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const cuComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const cuComplex *)&fScaleY, (cuComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasZhemv( hCUBLASContext, iCUBLASFillMode, m_hMatrixRegionA.iWidth,
							  (const cuDoubleComplex *)&fScaleX, (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const cuDoubleComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const cuDoubleComplex *)&fScaleY, (cuDoubleComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<class T>
Void CUBLASMatrixVectorOp::MulTriangular( CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputX<T>() );
	
	// Specific Input Validation
	DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionA.iHeight );
	DebugAssert( m_pVectorX->IsValidRegion( m_hVectorPositionX, m_hMatrixRegionA ) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasStrmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iWidth,
							  (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (Float*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDtrmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iWidth,
							  (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (Double*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasCtrmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iWidth,
							  (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (cuComplex*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasZtrmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iWidth,
							  (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (cuDoubleComplex*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<class T>
Void CUBLASMatrixVectorOp::SolveTriangular( CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputX<T>() );
	
	// Specific Input Validation
	DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionA.iHeight );
	DebugAssert( m_pVectorX->IsValidRegion( m_hVectorPositionX, m_hMatrixRegionA ) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasStrsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iWidth,
							  (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (Float*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDtrsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iWidth,
							  (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (Double*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasCtrsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iWidth,
							  (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (cuComplex*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasZtrsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iWidth,
							  (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (cuDoubleComplex*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<class T>
Void CUBLASMatrixVectorOp::MulAddBanded( T fScaleX, T fScaleY, SizeT iExpandedSizeA, SizeT iLowerDiagsCount, SizeT iUpperDiagsCount, CUBLASContextTransposeOp iTransOp )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputXY<T>() );
	
	// Specific Input Validation
	SizeT iRowsA, iColsA;
	CUDAMemoryRegion hRegionX, hRegionY;
	hRegionX.iWidth = 0;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	hRegionY.iWidth = 0;
	hRegionY.iHeight = 0;
	hRegionY.iDepth = 0;
	if ( iTransOp == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		iRowsA = iExpandedSizeA;
		iColsA = m_hMatrixRegionA.iHeight;
		hRegionX.iWidth = iColsA;
		hRegionY.iWidth = iRowsA;
	} else {
		iRowsA = m_hMatrixRegionA.iWidth;
		iColsA = iExpandedSizeA;
		hRegionX.iWidth = iRowsA;
		hRegionY.iWidth = iColsA;
	}
	DebugAssert( iLowerDiagsCount + 1 + iUpperDiagsCount == m_hMatrixRegionA.iWidth );
	DebugAssert( m_pVectorX->IsValidRegion(m_hVectorPositionX, hRegionX) );
	DebugAssert( m_pVectorY->IsValidRegion(m_hVectorPositionY, hRegionY) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasSgbmv( hCUBLASContext, iCUBLASTransposeOp, iRowsA, iColsA, iLowerDiagsCount, iUpperDiagsCount,
							  (const Float *)&fScaleX, (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const Float *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const Float *)&fScaleY, (Float*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDgbmv( hCUBLASContext, iCUBLASTransposeOp, iRowsA, iColsA, iLowerDiagsCount, iUpperDiagsCount,
							  (const Double *)&fScaleX, (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const Double *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const Double *)&fScaleY, (Double*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasCgbmv( hCUBLASContext, iCUBLASTransposeOp, iRowsA, iColsA, iLowerDiagsCount, iUpperDiagsCount,
							  (const cuComplex *)&fScaleX, (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const cuComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const cuComplex *)&fScaleY, (cuComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasZgbmv( hCUBLASContext, iCUBLASTransposeOp, iRowsA, iColsA, iLowerDiagsCount, iUpperDiagsCount,
							  (const cuDoubleComplex *)&fScaleX, (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const cuDoubleComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const cuDoubleComplex *)&fScaleY, (cuDoubleComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<class T>
Void CUBLASMatrixVectorOp::MulAddSymmetricBanded( T fScaleX, T fScaleY, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputXY<T>() );
	
	// Specific Input Validation
	CUDAMemoryRegion hRegionVect;
	hRegionVect.iWidth = m_hMatrixRegionA.iHeight;
	hRegionVect.iHeight = 0;
	hRegionVect.iDepth = 0;
	DebugAssert( (iSubDiagsCount << 1) + 1 == m_hMatrixRegionA.iWidth );
	DebugAssert( m_pVectorX->IsValidRegion(m_hVectorPositionX, hRegionVect) );
	DebugAssert( m_pVectorY->IsValidRegion(m_hVectorPositionY, hRegionVect) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasSsbmv( hCUBLASContext, iCUBLASFillMode, m_hMatrixRegionA.iHeight, iSubDiagsCount,
							  (const Float *)&fScaleX, (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const Float *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const Float *)&fScaleY, (Float*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDsbmv( hCUBLASContext, iCUBLASFillMode, m_hMatrixRegionA.iHeight, iSubDiagsCount,
							  (const Double *)&fScaleX, (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const Double *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const Double *)&fScaleY, (Double*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<class T>
Void CUBLASMatrixVectorOp::MulAddHermitianBanded( T fScaleX, T fScaleY, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputXY<T>() );
	
	// Specific Input Validation
	CUDAMemoryRegion hRegionVect;
	hRegionVect.iWidth = m_hMatrixRegionA.iHeight;
	hRegionVect.iHeight = 0;
	hRegionVect.iDepth = 0;
	DebugAssert( (iSubDiagsCount << 1) + 1 == m_hMatrixRegionA.iWidth );
	DebugAssert( m_pVectorX->IsValidRegion(m_hVectorPositionX, hRegionVect) );
	DebugAssert( m_pVectorY->IsValidRegion(m_hVectorPositionY, hRegionVect) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasChbmv( hCUBLASContext, iCUBLASFillMode, m_hMatrixRegionA.iHeight, iSubDiagsCount,
							  (const cuComplex *)&fScaleX, (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const cuComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const cuComplex *)&fScaleY, (cuComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasZhbmv( hCUBLASContext, iCUBLASFillMode, m_hMatrixRegionA.iHeight, iSubDiagsCount,
							  (const cuDoubleComplex *)&fScaleX, (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (const cuDoubleComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride(),
							  (const cuDoubleComplex *)&fScaleY, (cuDoubleComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), m_pVectorY->GetStride() );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<class T>
Void CUBLASMatrixVectorOp::MulTriangularBanded( SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputX<T>() );
	
	// Specific Input Validation
	CUDAMemoryRegion hRegionVect;
	hRegionVect.iWidth = m_hMatrixRegionA.iHeight;
	hRegionVect.iHeight = 0;
	hRegionVect.iDepth = 0;
	DebugAssert( iSubDiagsCount + 1 == m_hMatrixRegionA.iWidth );
	DebugAssert( m_pVectorX->IsValidRegion(m_hVectorPositionX, hRegionVect) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasStbmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iHeight, iSubDiagsCount,
							  (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (Float*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDtbmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iHeight, iSubDiagsCount,
							  (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (Double*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasCtbmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iHeight, iSubDiagsCount,
							  (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (cuComplex*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasZtbmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iHeight, iSubDiagsCount,
							  (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (cuDoubleComplex*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<class T>
Void CUBLASMatrixVectorOp::SolveTriangularBanded( SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInputX<T>() );
	
	// Specific Input Validation
	CUDAMemoryRegion hRegionVect;
	hRegionVect.iWidth = m_hMatrixRegionA.iHeight;
	hRegionVect.iHeight = 0;
	hRegionVect.iDepth = 0;
	DebugAssert( iSubDiagsCount + 1 == m_hMatrixRegionA.iWidth );
	DebugAssert( m_pVectorX->IsValidRegion(m_hVectorPositionX, hRegionVect) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasStbsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iHeight, iSubDiagsCount,
							  (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (Float*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDtbsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iHeight, iSubDiagsCount,
							  (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (Double*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasCtbsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iHeight, iSubDiagsCount,
							  (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (cuComplex*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasZtbsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
							  m_hMatrixRegionA.iHeight, iSubDiagsCount,
							  (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
							  (cuDoubleComplex*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
