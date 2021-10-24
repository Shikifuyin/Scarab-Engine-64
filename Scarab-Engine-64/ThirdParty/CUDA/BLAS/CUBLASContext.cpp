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
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	
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

template<>
Void CUBLASContext::Copy<Float>( CUDADeviceMemory * outDeviceVector, const CUDAMemoryPosition & outDevicePosition,
								 const CUDADeviceMemory * pDeviceVector, const CUDAMemoryPosition & hDevicePosition,
								 const CUDAMemoryRegion & hRegion )
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outDeviceVector->IsAllocated() );
	DebugAssert( outDeviceVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outDeviceVector->GetStride() == sizeof(Float) );
	DebugAssert( outDeviceVector->IsValidRegion(outDevicePosition, hRegion) );

	DebugAssert( pDeviceVector->IsAllocated() );
	DebugAssert( pDeviceVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pDeviceVector->GetStride() == sizeof(Float) );
	DebugAssert( pDeviceVector->IsValidRegion(hDevicePosition, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	cublasStatus_t iError = cublasScopy( hCUBLASContext, hRegion.iWidth,
										 (const Float *)( pDeviceVector->GetPointer(hDevicePosition) ), pDeviceVector->GetStride(),
										 (Float*)( outDeviceVector->GetPointer(outDevicePosition) ), outDeviceVector->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::Copy<Double>( CUDADeviceMemory * outDeviceVector, const CUDAMemoryPosition & outDevicePosition,
								  const CUDADeviceMemory * pDeviceVector, const CUDAMemoryPosition & hDevicePosition,
								  const CUDAMemoryRegion & hRegion )
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outDeviceVector->IsAllocated() );
	DebugAssert( outDeviceVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outDeviceVector->GetStride() == sizeof(Double) );
	DebugAssert( outDeviceVector->IsValidRegion(outDevicePosition, hRegion) );

	DebugAssert( pDeviceVector->IsAllocated() );
	DebugAssert( pDeviceVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pDeviceVector->GetStride() == sizeof(Double) );
	DebugAssert( pDeviceVector->IsValidRegion(hDevicePosition, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	cublasStatus_t iError = cublasDcopy( hCUBLASContext, hRegion.iWidth,
										 (const Double *)( pDeviceVector->GetPointer(hDevicePosition) ), pDeviceVector->GetStride(),
										 (Double*)( outDeviceVector->GetPointer(outDevicePosition) ), outDeviceVector->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::Copy<cuComplex>( CUDADeviceMemory * outDeviceVector, const CUDAMemoryPosition & outDevicePosition,
									 const CUDADeviceMemory * pDeviceVector, const CUDAMemoryPosition & hDevicePosition,
									 const CUDAMemoryRegion & hRegion )
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outDeviceVector->IsAllocated() );
	DebugAssert( outDeviceVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outDeviceVector->GetStride() == sizeof(cuComplex) );
	DebugAssert( outDeviceVector->IsValidRegion(outDevicePosition, hRegion) );

	DebugAssert( pDeviceVector->IsAllocated() );
	DebugAssert( pDeviceVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pDeviceVector->GetStride() == sizeof(cuComplex) );
	DebugAssert( pDeviceVector->IsValidRegion(hDevicePosition, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	cublasStatus_t iError = cublasCcopy( hCUBLASContext, hRegion.iWidth,
										 (const cuComplex *)( pDeviceVector->GetPointer(hDevicePosition) ), pDeviceVector->GetStride(),
										 (cuComplex*)( outDeviceVector->GetPointer(outDevicePosition) ), outDeviceVector->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::Copy<cuDoubleComplex>( CUDADeviceMemory * outDeviceVector, const CUDAMemoryPosition & outDevicePosition,
										   const CUDADeviceMemory * pDeviceVector, const CUDAMemoryPosition & hDevicePosition,
										   const CUDAMemoryRegion & hRegion )
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outDeviceVector->IsAllocated() );
	DebugAssert( outDeviceVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outDeviceVector->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( outDeviceVector->IsValidRegion(outDevicePosition, hRegion) );

	DebugAssert( pDeviceVector->IsAllocated() );
	DebugAssert( pDeviceVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pDeviceVector->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pDeviceVector->IsValidRegion(hDevicePosition, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	cublasStatus_t iError = cublasZcopy( hCUBLASContext, hRegion.iWidth,
										 (const cuDoubleComplex *)( pDeviceVector->GetPointer(hDevicePosition) ), pDeviceVector->GetStride(),
										 (cuDoubleComplex*)( outDeviceVector->GetPointer(outDevicePosition) ), outDeviceVector->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::Swap<Float>( CUDADeviceMemory * pDeviceVectorA, const CUDAMemoryPosition & hDevicePositionA,
								 CUDADeviceMemory * pDeviceVectorB, const CUDAMemoryPosition & hDevicePositionB,
								 const CUDAMemoryRegion & hRegion )
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pDeviceVectorA->IsAllocated() );
	DebugAssert( pDeviceVectorA->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pDeviceVectorA->GetStride() == sizeof(Float) );
	DebugAssert( pDeviceVectorA->IsValidRegion(hDevicePositionA, hRegion) );

	DebugAssert( pDeviceVectorB->IsAllocated() );
	DebugAssert( pDeviceVectorB->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pDeviceVectorB->GetStride() == sizeof(Float) );
	DebugAssert( pDeviceVectorB->IsValidRegion(hDevicePositionB, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	cublasStatus_t iError = cublasSswap( hCUBLASContext, hRegion.iWidth,
										 (Float*)( pDeviceVectorA->GetPointer(hDevicePositionA) ), pDeviceVectorA->GetStride(),
										 (Float*)( pDeviceVectorB->GetPointer(hDevicePositionB) ), pDeviceVectorB->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::Swap<Double>( CUDADeviceMemory * pDeviceVectorA, const CUDAMemoryPosition & hDevicePositionA,
								  CUDADeviceMemory * pDeviceVectorB, const CUDAMemoryPosition & hDevicePositionB,
								  const CUDAMemoryRegion & hRegion )
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pDeviceVectorA->IsAllocated() );
	DebugAssert( pDeviceVectorA->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pDeviceVectorA->GetStride() == sizeof(Double) );
	DebugAssert( pDeviceVectorA->IsValidRegion(hDevicePositionA, hRegion) );

	DebugAssert( pDeviceVectorB->IsAllocated() );
	DebugAssert( pDeviceVectorB->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pDeviceVectorB->GetStride() == sizeof(Double) );
	DebugAssert( pDeviceVectorB->IsValidRegion(hDevicePositionB, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	cublasStatus_t iError = cublasDswap( hCUBLASContext, hRegion.iWidth,
										 (Double*)( pDeviceVectorA->GetPointer(hDevicePositionA) ), pDeviceVectorA->GetStride(),
										 (Double*)( pDeviceVectorB->GetPointer(hDevicePositionB) ), pDeviceVectorB->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::Swap<cuComplex>( CUDADeviceMemory * pDeviceVectorA, const CUDAMemoryPosition & hDevicePositionA,
									 CUDADeviceMemory * pDeviceVectorB, const CUDAMemoryPosition & hDevicePositionB,
									 const CUDAMemoryRegion & hRegion )
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pDeviceVectorA->IsAllocated() );
	DebugAssert( pDeviceVectorA->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pDeviceVectorA->GetStride() == sizeof(cuComplex) );
	DebugAssert( pDeviceVectorA->IsValidRegion(hDevicePositionA, hRegion) );

	DebugAssert( pDeviceVectorB->IsAllocated() );
	DebugAssert( pDeviceVectorB->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pDeviceVectorB->GetStride() == sizeof(cuComplex) );
	DebugAssert( pDeviceVectorB->IsValidRegion(hDevicePositionB, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	cublasStatus_t iError = cublasCswap( hCUBLASContext, hRegion.iWidth,
										 (cuComplex*)( pDeviceVectorA->GetPointer(hDevicePositionA) ), pDeviceVectorA->GetStride(),
										 (cuComplex*)( pDeviceVectorB->GetPointer(hDevicePositionB) ), pDeviceVectorB->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::Swap<cuDoubleComplex>( CUDADeviceMemory * pDeviceVectorA, const CUDAMemoryPosition & hDevicePositionA,
										   CUDADeviceMemory * pDeviceVectorB, const CUDAMemoryPosition & hDevicePositionB,
										   const CUDAMemoryRegion & hRegion )
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pDeviceVectorA->IsAllocated() );
	DebugAssert( pDeviceVectorA->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pDeviceVectorA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pDeviceVectorA->IsValidRegion(hDevicePositionA, hRegion) );

	DebugAssert( pDeviceVectorB->IsAllocated() );
	DebugAssert( pDeviceVectorB->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pDeviceVectorB->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pDeviceVectorB->IsValidRegion(hDevicePositionB, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	cublasStatus_t iError = cublasZswap( hCUBLASContext, hRegion.iWidth,
										 (cuDoubleComplex*)( pDeviceVectorA->GetPointer(hDevicePositionA) ), pDeviceVectorA->GetStride(),
										 (cuDoubleComplex*)( pDeviceVectorB->GetPointer(hDevicePositionB) ), pDeviceVectorB->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
SizeT CUBLASContext::AbsMin<Float>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(Float) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIsamin( hCUBLASContext, hRegion.iWidth,
										  (const Float *)( pVector->GetPointer(hPosition) ),
										  pVector->GetStride(), &iResult );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return (SizeT)iResult;
}
template<>
SizeT CUBLASContext::AbsMin<Double>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(Double) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIdamin( hCUBLASContext, hRegion.iWidth,
										  (const Double *)( pVector->GetPointer(hPosition) ),
										  pVector->GetStride(), &iResult );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return (SizeT)iResult;
}
template<>
SizeT CUBLASContext::AbsMin<cuComplex>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(cuComplex) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIcamin( hCUBLASContext, hRegion.iWidth,
										  (const cuComplex *)( pVector->GetPointer(hPosition) ),
										  pVector->GetStride(), &iResult );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return (SizeT)iResult;
}
template<>
SizeT CUBLASContext::AbsMin<cuDoubleComplex>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIzamin( hCUBLASContext, hRegion.iWidth,
										  (const cuDoubleComplex *)( pVector->GetPointer(hPosition) ),
										  pVector->GetStride(), &iResult );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return (SizeT)iResult;
}

template<>
SizeT CUBLASContext::AbsMax<Float>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(Float) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIsamax( hCUBLASContext, hRegion.iWidth,
										  (const Float *)( pVector->GetPointer(hPosition) ),
										  pVector->GetStride(), &iResult );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return (SizeT)iResult;
}
template<>
SizeT CUBLASContext::AbsMax<Double>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(Double) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIdamax( hCUBLASContext, hRegion.iWidth,
										  (const Double *)( pVector->GetPointer(hPosition) ),
										  pVector->GetStride(), &iResult );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return (SizeT)iResult;
}
template<>
SizeT CUBLASContext::AbsMax<cuComplex>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(cuComplex) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIcamax( hCUBLASContext, hRegion.iWidth,
										  (const cuComplex *)( pVector->GetPointer(hPosition) ),
										  pVector->GetStride(), &iResult );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return (SizeT)iResult;
}
template<>
SizeT CUBLASContext::AbsMax<cuDoubleComplex>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Int iResult = INVALID_OFFSET;
	
	cublasStatus_t iError = cublasIzamax( hCUBLASContext, hRegion.iWidth,
										  (const cuDoubleComplex *)( pVector->GetPointer(hPosition) ),
										  pVector->GetStride(), &iResult );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS && iResult != INVALID_OFFSET );
	
	return (SizeT)iResult;
}

template<>
Float CUBLASContext::AbsSum<Float>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(Float) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Float fResult = 0.0f;
	
	cublasStatus_t iError = cublasSasum( hCUBLASContext, hRegion.iWidth,
										 (const Float *)( pVector->GetPointer(hPosition) ),
										 pVector->GetStride(), &fResult );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	
	return fResult;
}
template<>
Double CUBLASContext::AbsSum<Double>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(Double) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	Double fResult = 0.0;
	
	cublasStatus_t iError = cublasDasum( hCUBLASContext, hRegion.iWidth,
										 (const Double *)( pVector->GetPointer(hPosition) ),
										 pVector->GetStride(), &fResult );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	
	return fResult;
}
template<>
cuComplex CUBLASContext::AbsSum<cuComplex>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(cuComplex) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cuComplex fResult;
	fResult.x = 0.0f;
	fResult.y = 0.0f;
	
	cublasStatus_t iError = cublasScasum( hCUBLASContext, hRegion.iWidth,
										 (const cuComplex *)( pVector->GetPointer(hPosition) ),
										 pVector->GetStride(), &(fResult.x) );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	
	return fResult;
}
template<>
cuDoubleComplex CUBLASContext::AbsSum<cuDoubleComplex>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );
	
	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cuDoubleComplex fResult;
	fResult.x = 0.0;
	fResult.y = 0.0;
	
	cublasStatus_t iError = cublasDzasum( hCUBLASContext, hRegion.iWidth,
										 (const cuDoubleComplex *)( pVector->GetPointer(hPosition) ),
										 pVector->GetStride(), &(fResult.x) );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	
	return fResult;
}

template<>
Float CUBLASContext::Dot<Float>( const CUDADeviceMemory * pVectorA, const CUDAMemoryPosition & hPositionA,
								 const CUDADeviceMemory * pVectorB, const CUDAMemoryPosition & hPositionB,
								 const CUDAMemoryRegion & hRegion, Bool ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pVectorA->IsAllocated() );
	DebugAssert( pVectorA->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorA->GetStride() == sizeof(Float) );
	DebugAssert( pVectorA->IsValidRegion(hPositionA, hRegion) );

	DebugAssert( pVectorB->IsAllocated() );
	DebugAssert( pVectorB->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorB->GetStride() == sizeof(Float) );
	DebugAssert( pVectorB->IsValidRegion(hPositionB, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	Float fResult = 0.0f;
	
	cublasStatus_t iError = cublasSdot( hCUBLASContext, hRegion.iWidth,
										(const Float *)( pVectorA->GetPointer(hPositionA) ), pVectorA->GetStride(),
										(const Float *)( pVectorB->GetPointer(hPositionB) ), pVectorB->GetStride(),
										&fResult );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );

	return fResult;
}
template<>
Double CUBLASContext::Dot<Double>( const CUDADeviceMemory * pVectorA, const CUDAMemoryPosition & hPositionA,
								   const CUDADeviceMemory * pVectorB, const CUDAMemoryPosition & hPositionB,
								   const CUDAMemoryRegion & hRegion, Bool ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pVectorA->IsAllocated() );
	DebugAssert( pVectorA->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorA->GetStride() == sizeof(Double) );
	DebugAssert( pVectorA->IsValidRegion(hPositionA, hRegion) );

	DebugAssert( pVectorB->IsAllocated() );
	DebugAssert( pVectorB->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorB->GetStride() == sizeof(Double) );
	DebugAssert( pVectorB->IsValidRegion(hPositionB, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	Double fResult = 0.0;
	
	cublasStatus_t iError = cublasDdot( hCUBLASContext, hRegion.iWidth,
										(const Double *)( pVectorA->GetPointer(hPositionA) ), pVectorA->GetStride(),
										(const Double *)( pVectorB->GetPointer(hPositionB) ), pVectorB->GetStride(),
										&fResult );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );

	return fResult;
}
template<>
cuComplex CUBLASContext::Dot<cuComplex>( const CUDADeviceMemory * pVectorA, const CUDAMemoryPosition & hPositionA,
										 const CUDADeviceMemory * pVectorB, const CUDAMemoryPosition & hPositionB,
										 const CUDAMemoryRegion & hRegion, Bool bConjugateB ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pVectorA->IsAllocated() );
	DebugAssert( pVectorA->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorA->GetStride() == sizeof(cuComplex) );
	DebugAssert( pVectorA->IsValidRegion(hPositionA, hRegion) );

	DebugAssert( pVectorB->IsAllocated() );
	DebugAssert( pVectorB->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorB->GetStride() == sizeof(cuComplex) );
	DebugAssert( pVectorB->IsValidRegion(hPositionB, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	cuComplex fResult;
	fResult.x = 0.0f;
	fResult.y = 0.0f;
	
	if ( bConjugateB ) {
		cublasStatus_t iError = cublasCdotc( hCUBLASContext, hRegion.iWidth,
											 (const cuComplex *)(pVectorA->GetPointer( hPositionA )), pVectorA->GetStride(),
											 (const cuComplex *)(pVectorB->GetPointer( hPositionB )), pVectorB->GetStride(),
											 &fResult );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	} else {
		cublasStatus_t iError = cublasCdotu( hCUBLASContext, hRegion.iWidth,
											 (const cuComplex *)(pVectorA->GetPointer( hPositionA )), pVectorA->GetStride(),
											 (const cuComplex *)(pVectorB->GetPointer( hPositionB )), pVectorB->GetStride(),
											 &fResult );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	}

	return fResult;
}
template<>
cuDoubleComplex CUBLASContext::Dot<cuDoubleComplex>( const CUDADeviceMemory * pVectorA, const CUDAMemoryPosition & hPositionA,
													 const CUDADeviceMemory * pVectorB, const CUDAMemoryPosition & hPositionB,
													 const CUDAMemoryRegion & hRegion, Bool bConjugateB ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pVectorA->IsAllocated() );
	DebugAssert( pVectorA->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pVectorA->IsValidRegion(hPositionA, hRegion) );

	DebugAssert( pVectorB->IsAllocated() );
	DebugAssert( pVectorB->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorB->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pVectorB->IsValidRegion(hPositionB, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	cuDoubleComplex fResult;
	fResult.x = 0.0;
	fResult.y = 0.0;
	
	if ( bConjugateB ) {
		cublasStatus_t iError = cublasZdotc( hCUBLASContext, hRegion.iWidth,
											 (const cuDoubleComplex *)(pVectorA->GetPointer( hPositionA )), pVectorA->GetStride(),
											 (const cuDoubleComplex *)(pVectorB->GetPointer( hPositionB )), pVectorB->GetStride(),
											 &fResult );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	} else {
		cublasStatus_t iError = cublasZdotu( hCUBLASContext, hRegion.iWidth,
											 (const cuDoubleComplex *)(pVectorA->GetPointer( hPositionA )), pVectorA->GetStride(),
											 (const cuDoubleComplex *)(pVectorB->GetPointer( hPositionB )), pVectorB->GetStride(),
											 &fResult );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	}

	return fResult;
}

template<>
Float CUBLASContext::Norm<Float>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(Float) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	Float fResult = 0.0f;
	
	cublasStatus_t iError = cublasSnrm2( hCUBLASContext, hRegion.iWidth,
										 (const Float *)( pVector->GetPointer(hPosition) ), pVector->GetStride(),
										 &fResult );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );

	return fResult;
}
template<>
Double CUBLASContext::Norm<Double>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(Double) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	Double fResult = 0.0;
	
	cublasStatus_t iError = cublasDnrm2( hCUBLASContext, hRegion.iWidth,
										 (const Double *)( pVector->GetPointer(hPosition) ), pVector->GetStride(),
										 &fResult );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );

	return fResult;
}
template<>
cuComplex CUBLASContext::Norm<cuComplex>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(cuComplex) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	cuComplex fResult;
	fResult.x = 0.0f;
	fResult.y = 0.0f;
	
	cublasStatus_t iError = cublasScnrm2( hCUBLASContext, hRegion.iWidth,
										 (const cuComplex *)( pVector->GetPointer(hPosition) ), pVector->GetStride(),
										 &(fResult.x) );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );

	return fResult;
}
template<>
cuDoubleComplex CUBLASContext::Norm<cuDoubleComplex>( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	cuDoubleComplex fResult;
	fResult.x = 0.0;
	fResult.y = 0.0;
	
	cublasStatus_t iError = cublasDznrm2( hCUBLASContext, hRegion.iWidth,
										 (const cuDoubleComplex *)( pVector->GetPointer(hPosition) ), pVector->GetStride(),
										 &(fResult.x) );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );

	return fResult;
}

template<>
Void CUBLASContext::Scale<Float>( CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion, Float fAlpha ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(Float) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	cublasStatus_t iError = cublasSscal( hCUBLASContext, hRegion.iWidth, &fAlpha,
										 (Float*)( pVector->GetPointer(hPosition) ), pVector->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::Scale<Double>( CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion, Double fAlpha ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(Double) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	cublasStatus_t iError = cublasDscal( hCUBLASContext, hRegion.iWidth, &fAlpha,
										 (Double*)( pVector->GetPointer(hPosition) ), pVector->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::Scale<cuComplex>( CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion, cuComplex fAlpha ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(cuComplex) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	if ( fAlpha.y == 0.0f ) {
		cublasStatus_t iError = cublasCsscal( hCUBLASContext, hRegion.iWidth, &(fAlpha.x),
											 (cuComplex*)(pVector->GetPointer( hPosition )), pVector->GetStride() );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	} else {
		cublasStatus_t iError = cublasCscal( hCUBLASContext, hRegion.iWidth, &fAlpha,
											 (cuComplex*)(pVector->GetPointer( hPosition )), pVector->GetStride() );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	}
}
template<>
Void CUBLASContext::Scale<cuDoubleComplex>( CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion, cuDoubleComplex fAlpha ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( pVector->IsAllocated() );
	DebugAssert( pVector->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVector->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pVector->IsValidRegion(hPosition, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;

	if ( fAlpha.y == 0.0f ) {
		cublasStatus_t iError = cublasZdscal( hCUBLASContext, hRegion.iWidth, &(fAlpha.x),
											 (cuDoubleComplex*)(pVector->GetPointer( hPosition )), pVector->GetStride() );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	} else {
		cublasStatus_t iError = cublasZscal( hCUBLASContext, hRegion.iWidth, &fAlpha,
											 (cuDoubleComplex*)(pVector->GetPointer( hPosition )), pVector->GetStride() );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	}
}

template<>
Void CUBLASContext::MulAdd<Float>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY,
								   const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX,
								   Float fAlpha, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(Float) );
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegion) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(Float) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasStatus_t iError = cublasSaxpy( hCUBLASContext, hRegion.iWidth, &fAlpha,
										 (const Float *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 (Float *)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAdd<Double>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY,
									const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX,
									Double fAlpha, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(Double) );
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegion) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(Double) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasStatus_t iError = cublasDaxpy( hCUBLASContext, hRegion.iWidth, &fAlpha,
										 (const Double *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 (Double *)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAdd<cuComplex>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY,
									   const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX,
									   cuComplex fAlpha, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(cuComplex) );
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegion) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(cuComplex) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasStatus_t iError = cublasCaxpy( hCUBLASContext, hRegion.iWidth, &fAlpha,
										 (const cuComplex *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 (cuComplex *)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAdd<cuDoubleComplex>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY,
											 const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX,
											 cuDoubleComplex fAlpha, const CUDAMemoryRegion & hRegion ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegion) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegion) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	
	cublasStatus_t iError = cublasZaxpy( hCUBLASContext, hRegion.iWidth, &fAlpha,
										 (const cuDoubleComplex *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 (cuDoubleComplex *)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

