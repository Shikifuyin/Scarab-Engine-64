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
	DebugAssert( m_hContext == NULL );
	
	cublasHandle_t hCUBLASContext = NULL;
	
	cublasStatus_t iError = cublasCreate( &hCUBLASContext );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS && hCUBLASContext != NULL );
	
	m_hContext = hCUBLASContext;
}
Void CUBLASContext::Destroy()
{
	DebugAssert( m_hContext != NULL );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasStatus_t iError = cublasDestroy( hCUBLASContext );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	
	m_hContext = NULL;
}

Int CUBLASContext::GetVersion() const
{
	DebugAssert( m_hContext != NULL );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iVersion = 0;
	
	cublasStatus_t iError = cublasGetVersion( hCUBLASContext, &iVersion );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	
	return iVersion;
}

Void CUBLASContext::SetPointerMode( CUBLASContextPointerMode iMode ) const
{
	DebugAssert( m_hContext != NULL );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasPointerMode_t iCUBLASPointerMode = (cublasPointerMode_t)( CUBLASContextPointerModeToCUDA[iMode] );
	
	cublasStatus_t iError = cublasSetPointerMode( hCUBLASContext, iCUBLASPointerMode );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

Void CUBLASContext::SetPrecisionMode( CUBLASContextPrecisionMode iMode, Bool bDenyLowPrecisionReduction ) const
{
	DebugAssert( m_hContext != NULL );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iCUBLASMathMode;
	switch( iMode ) {
		case CUBLAS_CONTEXT_PRECISION_MODE_DEFAULT: iCUBLASMathMode = CUBLAS_DEFAULT_MATH; break;
		case CUBLAS_CONTEXT_PRECISION_MODE_PRECISE: iCUBLASMathMode = CUBLAS_PEDANTIC_MATH; break;
		case CUBLAS_CONTEXT_PRECISION_MODE_TF32:	iCUBLASMathMode = CUBLAS_TF32_TENSOR_OP_MATH; break;
		default: DebugAssert(false); break;
	}
	if ( bDenyLowPrecisionReduction )
		iCUBLASMathMode |= CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
	
	cublasStatus_t iError = cublasSetMathMode( hCUBLASContext, (cublasMath_t)iCUBLASMathMode );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

Void CUBLASContext::SetAtomicsMode( Bool bEnable ) const
{
	DebugAssert( m_hContext != NULL );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasStatus_t iError = cublasSetAtomicsMode( hCUBLASContext, bEnable ? CUBLAS_ATOMICS_ALLOWED : CUBLAS_ATOMICS_NOT_ALLOWED );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

Void CUBLASContext::SetLoggingMode( CUBLASContextLoggingMode iMode, const Char * strLogFilename )
{
	Int iLogEnable = ( iMode != CUBLAS_CONTEXT_LOGGING_MODE_DISABLED ) ? 1 : 0;
	Int iLogStdOut = ( iMode == CUBLAS_CONTEXT_LOGGING_MODE_STDOUT || iMode == CUBLAS_CONTEXT_LOGGING_MODE_BOTH ) ? 1 : 0;
	Int iLogStdErr = ( iMode == CUBLAS_CONTEXT_LOGGING_MODE_STDERR || iMode == CUBLAS_CONTEXT_LOGGING_MODE_BOTH ) ? 1 : 0;
	
	cublasStatus_t iError = cublasLoggerConfigure( iLogEnable, iLogStdOut, iLogStdErr, (const char *)strLogFilename );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
Void CUBLASContext::SetLogCallback( CUBLASLogCallback pfLogCallback )
{
	cublasStatus_t iError = cublasSetLoggerCallback( (cublasLogCallback)pfLogCallback );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

Void CUBLASContext::SetStream( CUDAStream * pStream ) const
{
	DebugAssert( m_hContext != NULL );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cudaStream_t hCUDAStream = NULL;
	if ( pStream != NULL ) {
		DebugAssert( pStream->IsCreated() );
		hCUDAStream = (cudaStream_t)( pStream->m_hStream );
	}
	
	cublasStatus_t iError = cublasSetStream( hCUBLASContext, hCUDAStream );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

Void CUBLASContext::SetMemory( CUDADeviceMemory * pMemory ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( pMemory->IsAllocated() );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasStatus_t iError = cublasSetWorkspace( hCUBLASContext, pMemory->GetPointer(), pMemory->GetSize() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

Void CUBLASContext::GetVector( CUDAHostMemory * outHostVector, const CUDAMemoryPosition & outHostPosition, UInt iHostIncrement,
							   const CUDADeviceMemory * pDeviceVector, const CUDAMemoryPosition & hDevicePosition, UInt iDeviceIncrement,
							   SizeT iElementCount, CUDAStream * pStream )
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outHostVector->IsAllocated() );
	DebugAssert( outHostVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( (outHostPosition.iX + iElementCount * (SizeT)iHostIncrement) <= outHostVector->GetWidth() );

	DebugAssert( pDeviceVector->IsAllocated() );
	DebugAssert( pDeviceVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( (hDevicePosition.iX + iElementCount * (SizeT)iDeviceIncrement) <= pDeviceVector->GetWidth() );

	DebugAssert( outHostVector->GetStride() == pDeviceVector->GetStride() );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	if ( pStream != NULL ) {
		DebugAssert( pStream->IsCreated() );
		cudaStream_t hCUDAStream = (cudaStream_t)( pStream->m_hStream );

		cublasStatus_t iError = cublasGetVectorAsync( (Int)iElementCount, (Int)(pDeviceVector->GetStride()),
													  pDeviceVector->GetPointer(hDevicePosition), (Int)iDeviceIncrement,
													  outHostVector->GetPointer(outHostPosition), (Int)iHostIncrement, hCUDAStream );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	} else {
		cublasStatus_t iError = cublasGetVector( (Int)iElementCount, (Int)(pDeviceVector->GetStride()),
												 pDeviceVector->GetPointer(hDevicePosition), (Int)iDeviceIncrement,
												 outHostVector->GetPointer(outHostPosition), (Int)iHostIncrement );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	}
}
Void CUBLASContext::SetVector( CUDADeviceMemory * outDeviceVector, const CUDAMemoryPosition & outDevicePosition, UInt iDeviceIncrement,
							   const CUDAHostMemory * pHostVector, const CUDAMemoryPosition & hHostPosition, UInt iHostIncrement,
							   SizeT iElementCount, CUDAStream * pStream )
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outDeviceVector->IsAllocated() );
	DebugAssert( outDeviceVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( (outDevicePosition.iX + iElementCount * (SizeT)iDeviceIncrement) <= outDeviceVector->GetWidth() );

	DebugAssert( pHostVector->IsAllocated() );
	DebugAssert( pHostVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( (hHostPosition.iX + iElementCount * (SizeT)iHostIncrement) <= pHostVector->GetWidth() );

	DebugAssert( outDeviceVector->GetStride() == pHostVector->GetStride() );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	if ( pStream != NULL ) {
		DebugAssert( pStream->IsCreated() );
		cudaStream_t hCUDAStream = (cudaStream_t)( pStream->m_hStream );

		cublasStatus_t iError = cublasSetVectorAsync( (Int)iElementCount, (Int)(pHostVector->GetStride()),
													  pHostVector->GetPointer(hHostPosition), (Int)iHostIncrement,
													  outDeviceVector->GetPointer(outDevicePosition), (Int)iDeviceIncrement, hCUDAStream );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	} else {
		cublasStatus_t iError = cublasSetVector( (Int)iElementCount, (Int)(pHostVector->GetStride()),
												 pHostVector->GetPointer(hHostPosition), (Int)iHostIncrement,
												 outDeviceVector->GetPointer(outDevicePosition), (Int)iDeviceIncrement );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	}
}

Void CUBLASContext::GetMatrix( CUDAHostMemory * outHostMatrix, const CUDAMemoryPosition & outHostPosition,
							   const CUDADeviceMemory * pDeviceMatrix, const CUDAMemoryPosition & hDevicePosition,
							   const CUDAMemoryRegion & hCopyRegion, CUDAStream * pStream )
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outHostMatrix->IsAllocated() );
	DebugAssert( outHostMatrix->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outHostMatrix->IsValidRegion(outHostPosition, hCopyRegion) );

	DebugAssert( pDeviceMatrix->IsAllocated() );
	DebugAssert( pDeviceMatrix->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pDeviceMatrix->IsValidRegion(hDevicePosition, hCopyRegion) );

	DebugAssert( outHostMatrix->GetStride() == pDeviceMatrix->GetStride() );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	if ( pStream != NULL ) {
		DebugAssert( pStream->IsCreated() );
		cudaStream_t hCUDAStream = (cudaStream_t)( pStream->m_hStream );

		cublasStatus_t iError = cublasGetMatrixAsync( (Int)(hCopyRegion.iWidth), (Int)(hCopyRegion.iHeight), (Int)(pDeviceMatrix->GetStride()),
													  pDeviceMatrix->GetPointer(hDevicePosition), (Int)(pDeviceMatrix->GetWidth()),
													  outHostMatrix->GetPointer(outHostPosition), (Int)(outHostMatrix->GetWidth()), hCUDAStream );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	} else {
		cublasStatus_t iError = cublasGetMatrix( (Int)(hCopyRegion.iWidth), (Int)(hCopyRegion.iHeight), (Int)(pDeviceMatrix->GetStride()),
												 pDeviceMatrix->GetPointer(hDevicePosition), (Int)(pDeviceMatrix->GetWidth()),
												 outHostMatrix->GetPointer(outHostPosition), (Int)(outHostMatrix->GetWidth()) );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	}
}
Void CUBLASContext::SetMatrix( CUDADeviceMemory * outDeviceMatrix, const CUDAMemoryPosition & outDevicePosition,
							   const CUDAHostMemory * pHostMatrix, const CUDAMemoryPosition & hHostPosition,
							   const CUDAMemoryRegion & hCopyRegion, CUDAStream * pStream )
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outDeviceMatrix->IsAllocated() );
	DebugAssert( outDeviceMatrix->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outDeviceMatrix->IsValidRegion(outDevicePosition, hCopyRegion) );

	DebugAssert( pHostMatrix->IsAllocated() );
	DebugAssert( pHostMatrix->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pHostMatrix->IsValidRegion(hHostPosition, hCopyRegion) );

	DebugAssert( outDeviceMatrix->GetStride() == pHostMatrix->GetStride() );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	if ( pStream != NULL ) {
		DebugAssert( pStream->IsCreated() );
		cudaStream_t hCUDAStream = (cudaStream_t)( pStream->m_hStream );

		cublasStatus_t iError = cublasSetMatrixAsync( (Int)(hCopyRegion.iWidth), (Int)(hCopyRegion.iHeight), (Int)(pHostMatrix->GetStride()),
													  pHostMatrix->GetPointer(hHostPosition), (Int)(pHostMatrix->GetWidth()),
													  outDeviceMatrix->GetPointer(outDevicePosition), (Int)(outDeviceMatrix->GetWidth()), hCUDAStream );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	} else {
		cublasStatus_t iError = cublasSetMatrix( (Int)(hCopyRegion.iWidth), (Int)(hCopyRegion.iHeight), (Int)(pHostMatrix->GetStride()),
												 pHostMatrix->GetPointer(hHostPosition), (Int)(pHostMatrix->GetWidth()),
												 outDeviceMatrix->GetPointer(outDevicePosition), (Int)(outDeviceMatrix->GetWidth()) );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	}
}

