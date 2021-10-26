/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/BLAS/CUBLASMatrixOp.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA BLAS : Matrix Operations
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
inline Bool CUBLASMatrixVectorOp::ValidateInput() const {
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
Void CUBLASMatrixVectorOp::MulTriangular( CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInput() );
	
	// Specific Input Validation
	DebugAssert( m_hMatrixRegionA.iWidth == m_hMatrixRegionA.iHeight );
	DebugAssert( m_pVectorX->IsValidRegion( m_hVectorPositionX, m_hMatrixRegionA ) );
	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError;
	switch( typeid(T) ) {
		case typeid(CUDAReal32):
			iError = cublasStrmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
								  m_hMatrixRegionA.iWidth,
								  (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (Float*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
			break;
		case typeid(CUDAReal64):
			iError = cublasDtrmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
								  m_hMatrixRegionA.iWidth,
								  (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (Double*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
			break;
		case typeid(CUDAComplex32):
			iError = cublasCtrmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
								  m_hMatrixRegionA.iWidth,
								  (const cuComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (cuComplex*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
			break;
		case typeid(CUDAComplex64):
			iError = cublasZtrmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
								  m_hMatrixRegionA.iWidth,
								  (const cuDoubleComplex *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), m_pMatrixA->GetWidth(),
								  (cuDoubleComplex*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), m_pVectorX->GetStride() );
			break;
		default: DebugAssert(false); break;
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

