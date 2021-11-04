/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUSolver/CUSolverDenseEigenValue.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA Solver for Dense systems : Eigen Values
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// CUSolverDenseEigenValue implementation
inline Void CUSolverDenseEigenValue::SetMatrixA( CUDADeviceMemory * pMatrix, const CUDAMemoryPosition * pPosition, const CUDAMemoryRegion * pRegion ) {
	DebugAssert( pMatrix != NULL );
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_RESET );
	m_pMatrixA = pMatrix;
	SetMatrixPositionA( pPosition );
	SetMatrixRegionA( pRegion );
}
inline Void CUSolverDenseEigenValue::SetMatrixPositionA( const CUDAMemoryPosition * pPosition ) {
	DebugAssert( m_pMatrixA != NULL );
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_RESET );
	if ( pPosition != NULL )
		m_hMatrixPositionA = *pPosition;
	else {
		m_hMatrixPositionA.iX = 0;
		m_hMatrixPositionA.iY = 0;
		m_hMatrixPositionA.iZ = 0;
	}
	DebugAssert( m_pMatrixA->IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA ) );
}
inline Void CUSolverDenseEigenValue::SetMatrixRegionA( const CUDAMemoryRegion * pRegion ) {
	DebugAssert( m_pMatrixA != NULL );
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_RESET );
	if ( pRegion != NULL )
		m_hMatrixRegionA = *pRegion;
	else {
		m_hMatrixRegionA.iWidth = m_pMatrixA->GetWidth();
		m_hMatrixRegionA.iHeight = m_pMatrixA->GetHeight();
		m_hMatrixRegionA.iDepth = 0;
	}
	DebugAssert( m_pMatrixA->IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA ) );
}

inline CUDADeviceMemory * CUSolverDenseEigenValue::GetMatrixA( CUDAMemoryPosition * outPosition, CUDAMemoryRegion * outRegion ) const {
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_SUCCESS );
	if ( outPosition != NULL )
		*outPosition = m_hMatrixPositionA;
	if ( outRegion != NULL )
		*outRegion = m_hMatrixRegionA;
	return m_pMatrixA;
}

inline Void CUSolverDenseEigenValue::SetMatrixFillModeA( CUBLASContextFillMode iFillMode ) {
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_RESET );
	m_iFillModeA = iFillMode;
}

inline Void CUSolverDenseEigenValue::SetVectorX( CUDADeviceMemory * pVector, const CUDAMemoryPosition * pPosition ) {
	DebugAssert( pVector != NULL );
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_RESET );
	m_pVectorX = pVector;
	SetVectorPositionX( pPosition );
}
inline Void CUSolverDenseEigenValue::SetVectorPositionX( const CUDAMemoryPosition * pPosition ) {
	DebugAssert( m_pVectorX != NULL );
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_RESET );
	if ( pPosition != NULL )
		m_hVectorPositionX = *pPosition;
	else {
		m_hVectorPositionX.iX = 0;
		m_hVectorPositionX.iY = 0;
		m_hVectorPositionX.iZ = 0;
	}
	DebugAssert( m_pVectorX->IsValidPosition( m_hVectorPositionX ) );
}

inline CUDADeviceMemory * CUSolverDenseEigenValue::GetVectorX( CUDAMemoryPosition * outPosition, CUDAMemoryRegion * outRegion ) const {
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_SUCCESS );
	if ( outPosition != NULL )
		*outPosition = m_hVectorPositionX;
	if ( outRegion != NULL )
		*outRegion = m_hMatrixRegionA;
	return m_pVectorX;
}

template<class T>
Bool CUSolverDenseEigenValue::ValidateInput() const
{
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_RESET );

	if ( typeid(T) == typeid(CUDAReal32) ) {
		return (
			m_pMatrixA != NULL
			&& m_pMatrixA->IsAllocated()
			&& m_pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D
			&& m_pMatrixA->GetStride() == sizeof(Float)
			&& m_hMatrixRegionA.iWidth == m_hMatrixRegionA.iHeight
			&& m_pMatrixA->IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA )
			&& m_pVectorX != NULL
			&& m_pVectorX->IsAllocated()
			&& m_pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D
			&& m_pVectorX->GetStride() == sizeof(Float)
			&& m_hVectorRegionX.iWidth == m_hMatrixRegionA.iWidth
			&& m_pVectorX->IsValidRegion( m_hVectorPositionX, m_hVectorRegionX )
		);
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		return (
			m_pMatrixA != NULL
			&& m_pMatrixA->IsAllocated()
			&& m_pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D
			&& m_pMatrixA->GetStride() == sizeof(Double)
			&& m_hMatrixRegionA.iWidth == m_hMatrixRegionA.iHeight
			&& m_pMatrixA->IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA )
			&& m_pVectorX != NULL
			&& m_pVectorX->IsAllocated()
			&& m_pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D
			&& m_pVectorX->GetStride() == sizeof(Double)
			&& m_hVectorRegionX.iWidth == m_hMatrixRegionA.iWidth
			&& m_pVectorX->IsValidRegion( m_hVectorPositionX, m_hVectorRegionX )
		);
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		return (
			m_pMatrixA != NULL
			&& m_pMatrixA->IsAllocated()
			&& m_pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D
			&& m_pMatrixA->GetStride() == sizeof(cuComplex)
			&& m_hMatrixRegionA.iWidth == m_hMatrixRegionA.iHeight
			&& m_pMatrixA->IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA )
			&& m_pVectorX != NULL
			&& m_pVectorX->IsAllocated()
			&& m_pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D
			&& m_pVectorX->GetStride() == sizeof(Float)
			&& m_hVectorRegionX.iWidth == m_hMatrixRegionA.iWidth
			&& m_pVectorX->IsValidRegion( m_hVectorPositionX, m_hVectorRegionX )
		);
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		return (
			m_pMatrixA != NULL
			&& m_pMatrixA->IsAllocated()
			&& m_pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D
			&& m_pMatrixA->GetStride() == sizeof(cuDoubleComplex)
			&& m_hMatrixRegionA.iWidth == m_hMatrixRegionA.iHeight
			&& m_pMatrixA->IsValidRegion( m_hMatrixPositionA, m_hMatrixRegionA )
			&& m_pVectorX != NULL
			&& m_pVectorX->IsAllocated()
			&& m_pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D
			&& m_pVectorX->GetStride() == sizeof(Double)
			&& m_hVectorRegionX.iWidth == m_hMatrixRegionA.iWidth
			&& m_pVectorX->IsValidRegion( m_hVectorPositionX, m_hVectorRegionX )
		);
	} else {
		DebugAssert( false );
		return false;
	}
}

inline Void CUSolverDenseEigenValue::ComputeEigenVectors( Bool bEnable ) {
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_RESET );
	m_bComputeEigenVectors = bEnable;
}
inline Void CUSolverDenseEigenValue::SetAlgorithm( CUSolverDenseEigenValueAlgorithm iAlgorithm ) {
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_RESET );
	m_iAlgorithm = iAlgorithm;
}
inline Void CUSolverDenseEigenValue::SetJacobiTolerance( Double fTolerance ) {
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_RESET );
	m_fJacobiTolerance = fTolerance;
}
inline Void CUSolverDenseEigenValue::SetJacobiMaxSweeps( UInt iMaxSweeps ) {
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_RESET );
	m_iJacobiMaxSweeps = iMaxSweeps;
}

inline CUSolverDenseEigenValueState CUSolverDenseEigenValue::GetSolverState() const {
	return m_iSolverState;
}

template<class T>
Void CUSolverDenseEigenValue::Prepare()
{
	DebugAssert( m_pCUSolverDenseContext != NULL );
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_RESET );
	DebugAssert( ValidateInput<T>() );

	cusolverDnHandle_t hCUSolverDnContext = (cusolverDnHandle_t)( m_pCUSolverDenseContext->m_hContext );
	cusolverEigMode_t hCUSolverDnEigMode = m_bComputeEigenVectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[m_iFillModeA] );

	Int iWorkspaceSize = 0;
	cusolverStatus_t iError;

	if ( m_iAlgorithm == CUSOLVER_DENSE_EIGENVALUE_ALGORITHM_QR ) {
		if ( typeid(T) == typeid(CUDAReal32) ) {
			iError = cusolverDnSsyevd_bufferSize( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
												  (Int)( m_hMatrixRegionA.iWidth ),
												  (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
												  (const Float *)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
												  &iWorkspaceSize );
		} else if ( typeid(T) == typeid(CUDAReal64) ) {
			iError = cusolverDnDsyevd_bufferSize( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
												  (Int)( m_hMatrixRegionA.iWidth ),
												  (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
												  (const Double *)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
												  &iWorkspaceSize );
		} else if ( typeid(T) == typeid(CUDAComplex32) ) {
			iError = cusolverDnCheevd_bufferSize( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
												  (Int)( m_hMatrixRegionA.iWidth ),
												  (cuComplex*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
												  (Float*)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
												  &iWorkspaceSize );
		} else if ( typeid(T) == typeid(CUDAComplex64) ) {
			iError = cusolverDnZheevd_bufferSize( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
												  (Int)( m_hMatrixRegionA.iWidth ),
												  (cuDoubleComplex*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
												  (Double*)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
												  &iWorkspaceSize );
		} else {
			DebugAssert( false );
		}
		DebugAssert( iError == CUSOLVER_STATUS_SUCCESS && iWorkspaceSize > 0 );
	} else if ( m_iAlgorithm == CUSOLVER_DENSE_EIGENVALUE_ALGORITHM_JACOBI ) {
		DebugAssert( m_hJacobiInfos == NULL );
		
		syevjInfo_t hCUSolverDnSyevjInfo = NULL;
		iError = cusolverDnCreateSyevjInfo( &hCUSolverDnSyevjInfo );
		DebugAssert( iError == CUSOLVER_STATUS_SUCCESS );

		iError = cusolverDnXsyevjSetTolerance( hCUSolverDnSyevjInfo, m_fJacobiTolerance );
		DebugAssert( iError == CUSOLVER_STATUS_SUCCESS );

		iError = cusolverDnXsyevjSetMaxSweeps( hCUSolverDnSyevjInfo, m_iJacobiMaxSweeps );
		DebugAssert( iError == CUSOLVER_STATUS_SUCCESS );

		m_hJacobiInfos = (Void*)hCUSolverDnSyevjInfo;

		if ( typeid(T) == typeid(CUDAReal32) ) {
			iError = cusolverDnSsyevj_bufferSize( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
												  (Int)( m_hMatrixRegionA.iWidth ),
												  (const Float *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
												  (const Float *)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
												  &iWorkspaceSize, hCUSolverDnSyevjInfo );
		} else if ( typeid(T) == typeid(CUDAReal64) ) {
			iError = cusolverDnDsyevj_bufferSize( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
												  (Int)( m_hMatrixRegionA.iWidth ),
												  (const Double *)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
												  (const Double *)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
												  &iWorkspaceSize, hCUSolverDnSyevjInfo );
		} else if ( typeid(T) == typeid(CUDAComplex32) ) {
			iError = cusolverDnCheevj_bufferSize( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
												  (Int)( m_hMatrixRegionA.iWidth ),
												  (cuComplex*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
												  (Float*)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
												  &iWorkspaceSize, hCUSolverDnSyevjInfo );
		} else if ( typeid(T) == typeid(CUDAComplex64) ) {
			iError = cusolverDnZheevj_bufferSize( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
												  (Int)( m_hMatrixRegionA.iWidth ),
												  (cuDoubleComplex*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
												  (Double*)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
												  &iWorkspaceSize, hCUSolverDnSyevjInfo );
		} else {
			DebugAssert( false );
		}
		DebugAssert( iError == CUSOLVER_STATUS_SUCCESS && iWorkspaceSize > 0 );
	} else {
		DebugAssert( false );
		return;
	}

	DebugAssert( !(m_hWorkspace.IsAllocated()) );
	m_hWorkspace.Allocate1D( sizeof(T), iWorkspaceSize );

	m_iSolverState = CUSOLVER_DENSE_EIGENVALUE_STATE_READY;
}
template<class T>
Void CUSolverDenseEigenValue::Solve()
{
	DebugAssert( m_pCUSolverDenseContext != NULL );
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_READY );

	DebugAssert( m_hWorkspace.IsAllocated() );
	DebugAssert( m_hWorkspace.GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( m_hWorkspace.GetStride() == sizeof(T) );

	cusolverDnHandle_t hCUSolverDnContext = (cusolverDnHandle_t)( m_pCUSolverDenseContext->m_hContext );
	cusolverEigMode_t hCUSolverDnEigMode = m_bComputeEigenVectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[m_iFillModeA] );

	cusolverStatus_t iError;

	if ( m_iAlgorithm == CUSOLVER_DENSE_EIGENVALUE_ALGORITHM_QR ) {
		if ( typeid(T) == typeid(CUDAReal32) ) {
			iError = cusolverDnSsyevd( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
									   (Int)( m_hMatrixRegionA.iWidth ),
									   (Float*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
									   (Float*)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
									   (Float*)( m_hWorkspace.GetPointer() ), (Int)( m_hWorkspace.GetWidth() ), &m_iSolverResult );
		} else if ( typeid(T) == typeid(CUDAReal64) ) {
			iError = cusolverDnDsyevd( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
									   (Int)( m_hMatrixRegionA.iWidth ),
									   (Double*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
									   (Double*)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
									   (Double*)( m_hWorkspace.GetPointer() ), (Int)( m_hWorkspace.GetWidth() ), &m_iSolverResult );
		} else if ( typeid(T) == typeid(CUDAComplex32) ) {
			iError = cusolverDnCheevd( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
									   (Int)( m_hMatrixRegionA.iWidth ),
									   (cuComplex*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
									   (Float*)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
									   (cuComplex*)( m_hWorkspace.GetPointer() ), (Int)( m_hWorkspace.GetWidth() ), &m_iSolverResult );
		} else if ( typeid(T) == typeid(CUDAComplex64) ) {
			iError = cusolverDnZheevd( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
									   (Int)( m_hMatrixRegionA.iWidth ),
									   (cuDoubleComplex*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
									   (Double*)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
									   (cuDoubleComplex*)( m_hWorkspace.GetPointer() ), (Int)( m_hWorkspace.GetWidth() ), &m_iSolverResult );
		} else {
			DebugAssert( false );
		}
		DebugAssert( iError == CUSOLVER_STATUS_SUCCESS && m_iSolverResult >= 0 );
	} else if ( m_iAlgorithm == CUSOLVER_DENSE_EIGENVALUE_ALGORITHM_JACOBI ) {
		if ( typeid(T) == typeid(CUDAReal32) ) {
			iError = cusolverDnSsyevj( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
									   (Int)( m_hMatrixRegionA.iWidth ),
									   (Float*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
									   (Float*)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
									   (Float*)( m_hWorkspace.GetPointer() ), (Int)( m_hWorkspace.GetWidth() ), &m_iSolverResult, (syevjInfo_t)m_hJacobiInfos );
		} else if ( typeid(T) == typeid(CUDAReal64) ) {
			iError = cusolverDnDsyevj( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
									   (Int)( m_hMatrixRegionA.iWidth ),
									   (Double*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
									   (Double*)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
									   (Double*)( m_hWorkspace.GetPointer() ), (Int)( m_hWorkspace.GetWidth() ), &m_iSolverResult, (syevjInfo_t)m_hJacobiInfos );
		} else if ( typeid(T) == typeid(CUDAComplex32) ) {
			iError = cusolverDnCheevj( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
									   (Int)( m_hMatrixRegionA.iWidth ),
									   (cuComplex*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
									   (Float*)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
									   (cuComplex*)( m_hWorkspace.GetPointer() ), (Int)( m_hWorkspace.GetWidth() ), &m_iSolverResult, (syevjInfo_t)m_hJacobiInfos );
		} else if ( typeid(T) == typeid(CUDAComplex64) ) {
			iError = cusolverDnZheevj( hCUSolverDnContext, hCUSolverDnEigMode, iCUBLASFillMode,
									   (Int)( m_hMatrixRegionA.iWidth ),
									   (cuDoubleComplex*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) ), (Int)( m_pMatrixA->GetWidth() ),
									   (Double*)( m_pVectorX->GetPointer(m_hVectorPositionX) ),
									   (cuDoubleComplex*)( m_hWorkspace.GetPointer() ), (Int)( m_hWorkspace.GetWidth() ), &m_iSolverResult, (syevjInfo_t)m_hJacobiInfos );
		} else {
			DebugAssert( false );
		}
		DebugAssert( iError == CUSOLVER_STATUS_SUCCESS && m_iSolverResult >= 0 );
	} else {
		DebugAssert( false );
		return;
	}

	m_iSolverState = (m_iSolverResult == 0) ? CUSOLVER_DENSE_EIGENVALUE_STATE_SUCCESS : CUSOLVER_DENSE_EIGENVALUE_STATE_FAILED;
}

inline UInt CUSolverDenseEigenValue::GetFailedToConvergeCount() const {
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_FAILED );
	return (UInt)m_iSolverResult;
}
inline UInt CUSolverDenseEigenValue::GetJacobiExecutedSweeps() const {
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_SUCCESS || m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_FAILED );

	cusolverDnHandle_t hCUSolverDnContext = (cusolverDnHandle_t)( m_pCUSolverDenseContext->m_hContext );
	syevjInfo_t hCUSolverDnSyevjInfo = (syevjInfo_t)m_hJacobiInfos;

	Int iExecutedSweeps = 0;

	cusolverStatus_t iError = cusolverDnXsyevjGetSweeps( hCUSolverDnContext, hCUSolverDnSyevjInfo, &iExecutedSweeps );
	DebugAssert( iError == CUSOLVER_STATUS_SUCCESS );

	return (UInt)iExecutedSweeps;
}
inline Double CUSolverDenseEigenValue::GetJacobiResidual() const {
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_SUCCESS || m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_FAILED );

	cusolverDnHandle_t hCUSolverDnContext = (cusolverDnHandle_t)( m_pCUSolverDenseContext->m_hContext );
	syevjInfo_t hCUSolverDnSyevjInfo = (syevjInfo_t)m_hJacobiInfos;

	Double fResidual = 0.0;

	cusolverStatus_t iError = cusolverDnXsyevjGetResidual( hCUSolverDnContext, hCUSolverDnSyevjInfo, &fResidual );
	DebugAssert( iError == CUSOLVER_STATUS_SUCCESS );

	return fResidual;
}

