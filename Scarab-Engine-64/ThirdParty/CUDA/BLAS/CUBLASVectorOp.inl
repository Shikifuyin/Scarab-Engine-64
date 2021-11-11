/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/BLAS/CUBLASVectorOp.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA BLAS : Vector Operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// CUBLASVectorOp implementation
inline Void CUBLASVectorOp::SetVector( CUDADeviceMemory * pVector, const CUDAMemoryPosition * pPosition, const CUDAMemoryRegion * pRegion ) {
	DebugAssert( pVector != NULL );
	m_pVector = pVector;
	SetVectorPosition( pPosition );
	SetVectorRegion( pRegion );
}
inline Void CUBLASVectorOp::SetVectorPosition( const CUDAMemoryPosition * pPosition ) {
	DebugAssert( m_pVector != NULL );
	if ( pPosition != NULL )
		m_hVectorPosition = *pPosition;
	else {
		m_hVectorPosition.iX = 0;
		m_hVectorPosition.iY = 0;
		m_hVectorPosition.iZ = 0;
	}
	DebugAssert( m_pVector->IsValidRegion( m_hVectorPosition, m_hVectorRegion ) );
}
inline Void CUBLASVectorOp::SetVectorRegion( const CUDAMemoryRegion * pRegion ) {
	DebugAssert( m_pVector != NULL );
	if ( pRegion != NULL )
		m_hVectorRegion = *pRegion;
	else {
		m_hVectorRegion.iWidth = m_pVector->GetWidth();
		m_hVectorRegion.iHeight = 1;
		m_hVectorRegion.iDepth = 1;
	}
	DebugAssert( m_pVector->IsValidRegion( m_hVectorPosition, m_hVectorRegion ) );
}

inline CUDADeviceMemory * CUBLASVectorOp::GetVector( CUDAMemoryPosition * outPosition, CUDAMemoryRegion * outRegion ) const {
	if ( outPosition != NULL )
		*outPosition = m_hVectorPosition;
	if ( outRegion != NULL )
		*outRegion = m_hVectorRegion;
	return m_pVector;
}

template<class T>
inline Bool CUBLASVectorOp::ValidateInput() const {
	return (
		m_pVector != NULL
		&& m_pVector->IsAllocated()
		&& m_pVector->GetShape() >= CUDA_MEMORY_SHAPE_1D
		&& m_pVector->GetStride() == sizeof(T)
		&& m_pVector->IsValidRegion( m_hVectorPosition, m_hVectorRegion )
	);
}

template<class T>
SizeT CUBLASVectorOp::AbsMin() const
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInput<T>() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasIsamin( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							   (const Float *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							   &iResult );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasIdamin( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							   (const Double *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							   &iResult );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasIcamin( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							   (const cuComplex *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							   &iResult );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasIzamin( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							   (const cuDoubleComplex *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							   &iResult );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return (SizeT)iResult;
}

template<class T>
SizeT CUBLASVectorOp::AbsMax() const
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInput<T>() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasIsamax( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							   (const Float *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							   &iResult );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasIdamax( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							   (const Double *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							   &iResult );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasIcamax( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							   (const cuComplex *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							   &iResult );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasIzamax( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							   (const cuDoubleComplex *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							   &iResult );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return (SizeT)iResult;
}

template<class T>
T CUBLASVectorOp::AbsSum() const
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInput<T>() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	
	T fResult;
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasSasum( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							  (const Float *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							  (Float*)&fResult );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDasum( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							   (const Double *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							   (Double*)&fResult );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasScasum( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							   (const cuComplex *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							   (Float*)&( ((CUDAComplex32*)&fResult)->fX ) );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasDzasum( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							   (const cuDoubleComplex *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							   (Double*)&( ((CUDAComplex64*)&fResult)->fX ) );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	
	return fResult;
}

template<class T>
T CUBLASVectorOp::Norm() const
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInput<T>() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	
	T fResult;
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasSnrm2( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							  (const Float *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							  (Float*)&fResult );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDnrm2( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							  (const Double *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							  (Double*)&fResult );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasScnrm2( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							   (const cuComplex *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							   (Float*)&( ((CUDAComplex32*)&fResult)->fX ) );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasDznrm2( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							   (const cuDoubleComplex *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ),
							   (Double*)&( ((CUDAComplex64*)&fResult)->fX ) );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	
	return fResult;
}

template<class T>
Void CUBLASVectorOp::Scale( T fScale )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInput<T>() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasSscal( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ), (const Float *)&fScale,
							  (Float *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ) );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDscal( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ), (const Double *)&fScale,
							  (Double *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ) );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		if ( ((CUDAComplex32*)&fScale)->fY == 0.0f ) {
			iError = cublasCsscal( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ), (const Float *)&( ((CUDAComplex32*)&fScale)->fX ),
								   (cuComplex *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ) );
		} else {
			iError = cublasCscal( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ), (const cuComplex *)( &fScale ),
								  (cuComplex *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ) );
		}
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		if ( ((CUDAComplex64*)&fScale)->fY == 0.0 ) {
			iError = cublasZdscal( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ), (const Double *)&( ((CUDAComplex64*)&fScale)->fX ),
								   (cuDoubleComplex *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ) );
		} else {
			iError = cublasZscal( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ), (const cuDoubleComplex *)( &fScale ),
								  (cuDoubleComplex *)( m_pVector->GetPointer(m_hVectorPosition) ), (Int)( m_pVector->GetStride() ) );
		}
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

/////////////////////////////////////////////////////////////////////////////////
// CUBLASVectorVectorOp implementation
inline Void CUBLASVectorVectorOp::SetVectorX( CUDADeviceMemory * pVector, const CUDAMemoryPosition * pPosition ) {
	DebugAssert( pVector != NULL );
	m_pVectorX = pVector;
	SetVectorPositionX( pPosition );
}
inline Void CUBLASVectorVectorOp::SetVectorPositionX( const CUDAMemoryPosition * pPosition ) {
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

inline CUDADeviceMemory * CUBLASVectorVectorOp::GetVectorX( CUDAMemoryPosition * outPosition, CUDAMemoryRegion * outRegion ) const {
	if ( outPosition != NULL )
		*outPosition = m_hVectorPositionX;
	if ( outRegion != NULL )
		*outRegion = m_hVectorRegion;
	return m_pVectorX;
}

inline Void CUBLASVectorVectorOp::SetVectorY( CUDADeviceMemory * pVector, const CUDAMemoryPosition * pPosition ) {
	DebugAssert( pVector != NULL );
	m_pVectorY = pVector;
	SetVectorPositionY( pPosition );
}
inline Void CUBLASVectorVectorOp::SetVectorPositionY( const CUDAMemoryPosition * pPosition ) {
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

inline CUDADeviceMemory * CUBLASVectorVectorOp::GetVectorY( CUDAMemoryPosition * outPosition, CUDAMemoryRegion * outRegion ) const {
	if ( outPosition != NULL )
		*outPosition = m_hVectorPositionY;
	if ( outRegion != NULL )
		*outRegion = m_hVectorRegion;
	return m_pVectorY;
}

inline Void CUBLASVectorVectorOp::SetVectorRegion( const CUDAMemoryRegion * pRegion ) {
	DebugAssert( m_pVectorX != NULL );
	DebugAssert( m_pVectorY != NULL );
	if ( pRegion != NULL )
		m_hVectorRegion = *pRegion;
	else {
		m_hVectorRegion.iWidth = Min<SizeT>( m_pVectorX->GetWidth(), m_pVectorY->GetWidth() );
		m_hVectorRegion.iHeight = 1;
		m_hVectorRegion.iDepth = 1;
	}
	DebugAssert( m_pVectorX->IsValidRegion( m_hVectorPositionX, m_hVectorRegion ) );
	DebugAssert( m_pVectorY->IsValidRegion( m_hVectorPositionY, m_hVectorRegion ) );
}

template<class T>
inline Bool CUBLASVectorVectorOp::ValidateInput() const {
	return (
		m_pVectorX != NULL
		&& m_pVectorX->IsAllocated()
		&& m_pVectorX->GetShape() >= CUDA_MEMORY_SHAPE_1D
		&& m_pVectorX->GetStride() == sizeof(T)
		&& m_pVectorX->IsValidRegion( m_hVectorPositionX, m_hVectorRegion )
		&& m_pVectorY != NULL
		&& m_pVectorY->IsAllocated()
		&& m_pVectorY->GetShape() >= CUDA_MEMORY_SHAPE_1D
		&& m_pVectorY->GetStride() == sizeof(T)
		&& m_pVectorY->IsValidRegion( m_hVectorPositionY, m_hVectorRegion )
	);
}

template<class T>
Void CUBLASVectorVectorOp::Copy()
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInput<T>() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasScopy( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							  (const Float *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
							  (Float*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ) );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDcopy( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							  (const Double *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
							  (Double*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ) );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasCcopy( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							  (const cuComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
							  (cuComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ) );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasZcopy( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							  (const cuDoubleComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
							  (cuDoubleComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ) );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<class T>
Void CUBLASVectorVectorOp::Swap()
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInput<T>() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasSswap( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							  (Float*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
							  (Float*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ) );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDswap( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							  (Double*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
							  (Double*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ) );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasCswap( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							  (cuComplex*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
							  (cuComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ) );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasZswap( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							  (cuDoubleComplex*)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
							  (cuDoubleComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ) );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<class T>
Void CUBLASVectorVectorOp::MulAdd( T fScaleX )
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInput<T>() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	
	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasSaxpy( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ), (const Float *)( &fScaleX ),
							  (const Float *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
							  (Float*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ) );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDaxpy( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ), (const Double *)( &fScaleX ),
							  (const Double *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
							  (Double*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ) );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		iError = cublasCaxpy( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ), (const cuComplex *)( &fScaleX ),
							  (const cuComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
							  (cuComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ) );
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		iError = cublasZaxpy( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ), (const cuDoubleComplex *)( &fScaleX ),
							  (const cuDoubleComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
							  (cuDoubleComplex*)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ) );
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<class T>
T CUBLASVectorVectorOp::Dot( Bool bConjugateY ) const
{
	DebugAssert( m_pCUBLASContext != NULL );
	DebugAssert( ValidateInput<T>() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)( m_pCUBLASContext->m_hContext );
	
	T fResult;

	cublasStatus_t iError;
	if ( typeid(T) == typeid(CUDAReal32) ) {
		iError = cublasSdot( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							 (const Float *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
							 (const Float *)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ),
							 (Float*)&fResult );
	} else if ( typeid(T) == typeid(CUDAReal64) ) {
		iError = cublasDdot( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
							 (const Double *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
							 (const Double *)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ),
							 (Double*)&fResult );
	} else if ( typeid(T) == typeid(CUDAComplex32) ) {
		if ( bConjugateY ) {
			iError = cublasCdotc( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
								  (const cuComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
								  (const cuComplex *)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ),
								  (cuComplex*)&fResult );
		} else {
			iError = cublasCdotu( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
								  (const cuComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
								  (const cuComplex *)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ),
								  (cuComplex*)&fResult );
		}
	} else if ( typeid(T) == typeid(CUDAComplex64) ) {
		if ( bConjugateY ) {
			iError = cublasZdotc( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
								  (const cuDoubleComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
								  (const cuDoubleComplex *)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ),
								  (cuDoubleComplex*)&fResult );
		} else {
			iError = cublasZdotu( hCUBLASContext, (Int)( m_hVectorRegion.iWidth ),
								  (const cuDoubleComplex *)( m_pVectorX->GetPointer(m_hVectorPositionX) ), (Int)( m_pVectorX->GetStride() ),
								  (const cuDoubleComplex *)( m_pVectorY->GetPointer(m_hVectorPositionY) ), (Int)( m_pVectorY->GetStride() ),
								  (cuDoubleComplex*)&fResult );
		}
	} else {
		DebugAssert( false );
	}
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );

	return fResult;
}

