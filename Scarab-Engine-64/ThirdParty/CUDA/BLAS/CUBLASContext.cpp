/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/BLAS/CUBLASContext.cpp
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
// Third-Party Includes
#include <cublas_v2.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUBLASContext.h"

/////////////////////////////////////////////////////////////////////////////////
// CUBLASContext implementation
CUBLASContext::CUBLASContext()
{
	m_hContext = NULL;
}
CUBLASContext::~CUBLASContext()
{
	if ( IsCreated() )
		Destroy();
}

Void CUBLASContext::Create()
{
	Assert( m_hContext == NULL );
	
	cublasHandle_t hCUBLASContext = NULL;
	
	cublasStatus_t iError = cublasCreate( &hCUBLASContext );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
	
	m_hContext = hCUBLASContext;
}
Void CUBLASContext::Destroy()
{
	Assert( m_hContext != NULL );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasStatus_t iError = cublasDestroy( hCUBLASContext );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
	
	m_hContext = NULL;
}

Int CUBLASContext::GetVersion() const
{
	Assert( m_hContext != NULL );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iVersion = 0;
	
	cublasStatus_t iError = cublasGetVersion( hCUBLASContext, &iVersion );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
	
	return iVersion;
}

Void CUBLASContext::SetPointerMode( CUBLASContextPointerMode iMode ) const
{
	Assert( m_hContext != NULL );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasPointerMode_t iCUBLASPointerMode = (cublasPointerMode_t)( CUBLASContextPointerModeToCUDA[iMode] );
	
	cublasStatus_t iError = cublasSetPointerMode( hCUBLASContext, iCUBLASPointerMode );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
}

Void CUBLASContext::SetPrecisionMode( CUBLASContextPrecisionMode iMode, Bool bDenyLowPrecisionReduction ) const
{
	Assert( m_hContext != NULL );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasMath_t iCUBLASMathMode;
	switch( iMode ) {
		case CUBLAS_CONTEXT_PRECISION_MODE_DEFAULT: iCUBLASMathMode = CUBLAS_DEFAULT_MATH; break;
		case CUBLAS_CONTEXT_PRECISION_MODE_PRECISE: iCUBLASMathMode = CUBLAS_PEDANTIC_MATH; break;
		case CUBLAS_CONTEXT_PRECISION_MODE_TF32:	iCUBLASMathMode = CUBLAS_TF32_TENSOR_OP_MATH; break;
		default: Assert(false); break;
	}
	if ( bDenyLowPrecisionReduction )
		iCUBLASMathMode |= CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
	
	cublasStatus_t iError = cublasSetMathMode( hCUBLASContext, iCUBLASMathMode );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
}

Void CUBLASContext::SetAtomicsMode( Bool bEnable ) const
{
	Assert( m_hContext != NULL );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasStatus_t iError = cublasSetAtomicsMode( hCUBLASContext, bEnable ? CUBLAS_ATOMICS_ALLOWED : CUBLAS_ATOMICS_NOT_ALLOWED );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
}

Void CUBLASContext::SetLoggingMode( CUBLASContextLoggingMode iMode, const Char * strLogFilename )
{
	Int iLogEnable = ( iMode != CUBLAS_CONTEXT_LOGGING_MODE_DISABLED ) ? 1 : 0;
	Int iLogStdOut = ( iMode == CUBLAS_CONTEXT_LOGGING_MODE_STDOUT || iMode == CUBLAS_CONTEXT_LOGGING_MODE_BOTH ) ? 1 : 0;
	Int iLogStdErr = ( iMode == CUBLAS_CONTEXT_LOGGING_MODE_STDERR || iMode == CUBLAS_CONTEXT_LOGGING_MODE_BOTH ) ? 1 : 0;
	
	cublasStatus_t iError = cublasLoggerConfigure( iLogEnable, iLogStdOut, iLogStdErr, strLogFilename );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
}
Void CUBLASContext::SetLogCallback( CUBLASLogCallback pfLogCallback )
{
	cublasStatus_t iError = cublasSetLoggerCallback( pfLogCallback );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
}

Void CUBLASContext::SetStream( CUDAStream * pStream ) const
{
	Assert( m_hContext != NULL );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cudaStream_t hCUDAStream = NULL;
	if ( pStream != NULL ) {
		Assert( pStream->IsCreated() );
		hCUDAStream = (cudaStream_t)( pStream->m_hStream );
	}
	
	cublasStatus_t iError = cublasSetStream( hCUBLASContext, hCUDAStream );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
}

Void CUBLASContext::SetMemory( CUDADeviceMemory * pMemory ) const
{
	Assert( m_hContext != NULL );
	Assert( pMemory->IsAllocated() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasStatus_t iError = cublasSetWorkspace( hCUBLASContext, pMemory->GetPointer(), pMemory->GetSize() );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
}

Int CUBLASContext::MinAbsFR( const CUDADeviceMemory * pVector ) const
{
	Assert( m_hContext != NULL );
	Assert( pVector->IsAllocated() && pVector->GetShape() == CUDA_MEMORY_SHAPE_1D && pVector->GetStride() == sizeof(Float) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIsamin( hCUBLASContext, pVector->GetWidth(), (const Float *)( pVector->GetPointer() ), pVector->GetStride(), &iResult );
	Assert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return iResult;
}
Int CUBLASContext::MinAbsDR( const CUDADeviceMemory * pVector ) const
{
	Assert( m_hContext != NULL );
	Assert( pVector->IsAllocated() && pVector->GetShape() == CUDA_MEMORY_SHAPE_1D && pVector->GetStride() == sizeof(Double) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIdamin( hCUBLASContext, pVector->GetWidth(), (const Double *)( pVector->GetPointer() ), pVector->GetStride(), &iResult );
	Assert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return iResult;
}
Int CUBLASContext::MinAbsFC( const CUDADeviceMemory * pVector ) const
{
	Assert( m_hContext != NULL );
	Assert( pVector->IsAllocated() && pVector->GetShape() == CUDA_MEMORY_SHAPE_1D && pVector->GetStride() == sizeof(cuComplex) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIcamin( hCUBLASContext, pVector->GetWidth(), (const cuComplex *)( pVector->GetPointer() ), pVector->GetStride(), &iResult );
	Assert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return iResult;
}
Int CUBLASContext::MinAbsDC( const CUDADeviceMemory * pVector ) const
{
	Assert( m_hContext != NULL );
	Assert( pVector->IsAllocated() && pVector->GetShape() == CUDA_MEMORY_SHAPE_1D && pVector->GetStride() == sizeof(cuDoubleComplex) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIzamin( hCUBLASContext, pVector->GetWidth(), (const cuDoubleComplex *)( pVector->GetPointer() ), pVector->GetStride(), &iResult );
	Assert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return iResult;
}

Int CUBLASContext::MaxAbsFR( const CUDADeviceMemory * pVector ) const
{
	Assert( m_hContext != NULL );
	Assert( pVector->IsAllocated() && pVector->GetShape() == CUDA_MEMORY_SHAPE_1D && pVector->GetStride() == sizeof(Float) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIsamax( hCUBLASContext, pVector->GetWidth(), (const Float *)( pVector->GetPointer() ), pVector->GetStride(), &iResult );
	Assert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return iResult;
}
Int CUBLASContext::MaxAbsDR( const CUDADeviceMemory * pVector ) const
{
	Assert( m_hContext != NULL );
	Assert( pVector->IsAllocated() && pVector->GetShape() == CUDA_MEMORY_SHAPE_1D && pVector->GetStride() == sizeof(Double) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIdamax( hCUBLASContext, pVector->GetWidth(), (const Double *)( pVector->GetPointer() ), pVector->GetStride(), &iResult );
	Assert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return iResult;
}
Int CUBLASContext::MaxAbsFC( const CUDADeviceMemory * pVector ) const
{
	Assert( m_hContext != NULL );
	Assert( pVector->IsAllocated() && pVector->GetShape() == CUDA_MEMORY_SHAPE_1D && pVector->GetStride() == sizeof(cuComplex) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIcamax( hCUBLASContext, pVector->GetWidth(), (const cuComplex *)( pVector->GetPointer() ), pVector->GetStride(), &iResult );
	Assert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return iResult;
}
Int CUBLASContext::MaxAbsDC( const CUDADeviceMemory * pVector ) const
{
	Assert( m_hContext != NULL );
	Assert( pVector->IsAllocated() && pVector->GetShape() == CUDA_MEMORY_SHAPE_1D && pVector->GetStride() == sizeof(cuDoubleComplex) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIzamax( hCUBLASContext, pVector->GetWidth(), (const cuDoubleComplex *)( pVector->GetPointer() ), pVector->GetStride(), &iResult );
	Assert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return iResult;
}

Float CUBLASContext::SumAbsFR( const CUDADeviceMemory * pVector ) const
{
	Assert( m_hContext != NULL );
	Assert( pVector->IsAllocated() && pVector->GetShape() == CUDA_MEMORY_SHAPE_1D && pVector->GetStride() == sizeof(Float) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Float fResult = 0.0f;
	
	cublasStatus_t iError = cublasSasum( hCUBLASContext, pVector->GetWidth(), (const Float *)( pVector->GetPointer() ), pVector->GetStride(), &fResult );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
	
	return fResult;
}
Double CUBLASContext::SumAbsDR( const CUDADeviceMemory * pVector ) const
{
	Assert( m_hContext != NULL );
	Assert( pVector->IsAllocated() && pVector->GetShape() == CUDA_MEMORY_SHAPE_1D && pVector->GetStride() == sizeof(Double) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Double fResult = 0.0f;
	
	cublasStatus_t iError = cublasDasum( hCUBLASContext, pVector->GetWidth(), (const Double *)( pVector->GetPointer() ), pVector->GetStride(), &fResult );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
	
	return fResult;
}
Float CUBLASContext::SumAbsFC( const CUDADeviceMemory * pVector ) const
{
	Assert( m_hContext != NULL );
	Assert( pVector->IsAllocated() && pVector->GetShape() == CUDA_MEMORY_SHAPE_1D && pVector->GetStride() == sizeof(cuComplex) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Float fResult = 0.0f;
	
	cublasStatus_t iError = cublasScasum( hCUBLASContext, pVector->GetWidth(), (const cuComplex *)( pVector->GetPointer() ), pVector->GetStride(), &fResult );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
	
	return fResult;
}
Double CUBLASContext::SumAbsDC( const CUDADeviceMemory * pVector ) const
{
	Assert( m_hContext != NULL );
	Assert( pVector->IsAllocated() && pVector->GetShape() == CUDA_MEMORY_SHAPE_1D && pVector->GetStride() == sizeof(cuDoubleComplex) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Double fResult = 0.0f;
	
	cublasStatus_t iError = cublasDzasum( hCUBLASContext, pVector->GetWidth(), (const cuDoubleComplex *)( pVector->GetPointer() ), pVector->GetStride(), &fResult );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
	
	return fResult;
}

Void CUBLASContext::MulAddFR( CUDADeviceMemory * outVectorY, const Void * pAlpha, const CUDADeviceMemory * pVectorX ) const
{
	Assert( m_hContext != NULL );
	Assert( outVectorY->IsAllocated() && outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D && outVectorY->GetStride() == sizeof(Float) );
	Assert( pVectorX->IsAllocated() && pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D && pVectorX->GetStride() == sizeof(Float) );
	Assert( outVectorY->GetWidth() == pVectorX->GetWidth() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasStatus_t iError = cublasSaxpy( hCUBLASContext, outVectorY->GetWidth(), (const Float *)pAlpha,
										 (const Float *)( pVectorX->GetPointer() ), pVectorX->GetStride(),
										 (Float *)( outVectorY->GetPointer() ), outVectorY->GetStride() );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
}
Void CUBLASContext::MulAddDR( CUDADeviceMemory * outVectorY, const Void * pAlpha, const CUDADeviceMemory * pVectorX ) const
{
	Assert( m_hContext != NULL );
	Assert( outVectorY->IsAllocated() && outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D && outVectorY->GetStride() == sizeof(Double) );
	Assert( pVectorX->IsAllocated() && pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D && pVectorX->GetStride() == sizeof(Double) );
	Assert( outVectorY->GetWidth() == pVectorX->GetWidth() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasStatus_t iError = cublasDaxpy( hCUBLASContext, outVectorY->GetWidth(), (const Double *)pAlpha,
										 (const Double *)( pVectorX->GetPointer() ), pVectorX->GetStride(),
										 (Double *)( outVectorY->GetPointer() ), outVectorY->GetStride() );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
}
Void CUBLASContext::MulAddFC( CUDADeviceMemory * outVectorY, const Void * pAlpha, const CUDADeviceMemory * pVectorX ) const
{
	Assert( m_hContext != NULL );
	Assert( outVectorY->IsAllocated() && outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D && outVectorY->GetStride() == sizeof(cuComplex) );
	Assert( pVectorX->IsAllocated() && pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D && pVectorX->GetStride() == sizeof(cuComplex) );
	Assert( outVectorY->GetWidth() == pVectorX->GetWidth() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasStatus_t iError = cublasCaxpy( hCUBLASContext, outVectorY->GetWidth(), (const cuComplex *)pAlpha,
										 (const cuComplex *)( pVectorX->GetPointer() ), pVectorX->GetStride(),
										 (cuComplex *)( outVectorY->GetPointer() ), outVectorY->GetStride() );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
}
Void CUBLASContext::MulAddDC( CUDADeviceMemory * outVectorY, const Void * pAlpha, const CUDADeviceMemory * pVectorX ) const
{
	Assert( m_hContext != NULL );
	Assert( outVectorY->IsAllocated() && outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D && outVectorY->GetStride() == sizeof(cuDoubleComplex) );
	Assert( pVectorX->IsAllocated() && pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D && pVectorX->GetStride() == sizeof(cuDoubleComplex) );
	Assert( outVectorY->GetWidth() == pVectorX->GetWidth() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasStatus_t iError = cublasZaxpy( hCUBLASContext, outVectorY->GetWidth(), (const cuDoubleComplex *)pAlpha,
										 (const cuDoubleComplex *)( pVectorX->GetPointer() ), pVectorX->GetStride(),
										 (cuDoubleComplex *)( outVectorY->GetPointer() ), outVectorY->GetStride() );
	Assert( iError == CUBLAS_STATUS_SUCCESS );
}

