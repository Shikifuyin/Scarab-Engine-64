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

template<>
inline Void CUBLASContext::Add<Float>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY,
									   const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX,
									   const CUDAMemoryRegion & hRegion ) const
{
	MulAdd<Float>( outVectorY, outPositionY, pVectorX, hPositionX, 1.0f, hRegion );
}
template<>
inline Void CUBLASContext::Add<Double>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY,
										const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX,
										const CUDAMemoryRegion & hRegion ) const
{
	MulAdd<Double>( outVectorY, outPositionY, pVectorX, hPositionX, 1.0, hRegion );
}
template<>
inline Void CUBLASContext::Add<cuComplex>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY,
										   const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX,
										   const CUDAMemoryRegion & hRegion ) const
{
	cuComplex fOne;
	fOne.x = 1.0f;
	fOne.y = 0.0f;

	MulAdd<cuComplex>( outVectorY, outPositionY, pVectorX, hPositionX, fOne, hRegion );
}
template<>
inline Void CUBLASContext::Add<cuDoubleComplex>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY,
												 const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX,
												 const CUDAMemoryRegion & hRegion ) const
{
	cuDoubleComplex fOne;
	fOne.x = 1.0;
	fOne.y = 0.0;

	MulAdd<cuDoubleComplex>( outVectorY, outPositionY, pVectorX, hPositionX, fOne, hRegion );
}

template<>
Void CUBLASContext::MulTriangular<Float>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
										  const CUDADeviceMemory * pTriangularMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
										  CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(Float) );
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionA) );

	DebugAssert( pTriangularMatrixA->IsAllocated() );
	DebugAssert( pTriangularMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularMatrixA->GetStride() == sizeof(Float) );
	DebugAssert( hRegionA.iWidth == hRegionA.iHeight );
	DebugAssert( pTriangularMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasStrmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, hRegionA.iWidth,
										 (const Float *)( pTriangularMatrixA->GetPointer(hPositionA) ), pTriangularMatrixA->GetWidth(),
										 (Float*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulTriangular<Double>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
										  const CUDADeviceMemory * pTriangularMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
										  CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(Double) );
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionA) );

	DebugAssert( pTriangularMatrixA->IsAllocated() );
	DebugAssert( pTriangularMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularMatrixA->GetStride() == sizeof(Double) );
	DebugAssert( hRegionA.iWidth == hRegionA.iHeight );
	DebugAssert( pTriangularMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasDtrmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, hRegionA.iWidth,
										 (const Double *)( pTriangularMatrixA->GetPointer(hPositionA) ), pTriangularMatrixA->GetWidth(),
										 (Double*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulTriangular<cuComplex>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
										  const CUDADeviceMemory * pTriangularMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
										  CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(cuComplex) );
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionA) );

	DebugAssert( pTriangularMatrixA->IsAllocated() );
	DebugAssert( pTriangularMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularMatrixA->GetStride() == sizeof(cuComplex) );
	DebugAssert( hRegionA.iWidth == hRegionA.iHeight );
	DebugAssert( pTriangularMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasCtrmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, hRegionA.iWidth,
										 (const cuComplex *)( pTriangularMatrixA->GetPointer(hPositionA) ), pTriangularMatrixA->GetWidth(),
										 (cuComplex*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulTriangular<cuDoubleComplex>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
										  const CUDADeviceMemory * pTriangularMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
										  CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionA) );

	DebugAssert( pTriangularMatrixA->IsAllocated() );
	DebugAssert( pTriangularMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularMatrixA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( hRegionA.iWidth == hRegionA.iHeight );
	DebugAssert( pTriangularMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasZtrmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, hRegionA.iWidth,
										 (const cuDoubleComplex *)( pTriangularMatrixA->GetPointer(hPositionA) ), pTriangularMatrixA->GetWidth(),
										 (cuDoubleComplex*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::MulTriangularBanded<Float>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
												const CUDADeviceMemory * pTriangularBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
												SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(Float) );

	DebugAssert( pTriangularBandedMatrixA->IsAllocated() );
	DebugAssert( pTriangularBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularBandedMatrixA->GetStride() == sizeof(Float) );
	DebugAssert( pTriangularBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	CUDAMemoryRegion hRegionX;
	hRegionX.iWidth = iExpandedSizeA;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasStbmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 iExpandedSizeA, iSubDiagsCount,
										 (const Float *)( pTriangularBandedMatrixA->GetPointer(hPositionA) ), pTriangularBandedMatrixA->GetWidth(),
										 (Float*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulTriangularBanded<Double>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
												 const CUDADeviceMemory * pTriangularBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
												 SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(Double) );

	DebugAssert( pTriangularBandedMatrixA->IsAllocated() );
	DebugAssert( pTriangularBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularBandedMatrixA->GetStride() == sizeof(Double) );
	DebugAssert( pTriangularBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	CUDAMemoryRegion hRegionX;
	hRegionX.iWidth = iExpandedSizeA;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasDtbmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 iExpandedSizeA, iSubDiagsCount,
										 (const Double *)( pTriangularBandedMatrixA->GetPointer(hPositionA) ), pTriangularBandedMatrixA->GetWidth(),
										 (Double*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulTriangularBanded<cuComplex>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
													const CUDADeviceMemory * pTriangularBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
													SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(cuComplex) );

	DebugAssert( pTriangularBandedMatrixA->IsAllocated() );
	DebugAssert( pTriangularBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularBandedMatrixA->GetStride() == sizeof(cuComplex) );
	DebugAssert( pTriangularBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	CUDAMemoryRegion hRegionX;
	hRegionX.iWidth = iExpandedSizeA;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasCtbmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 iExpandedSizeA, iSubDiagsCount,
										 (const cuComplex *)( pTriangularBandedMatrixA->GetPointer(hPositionA) ), pTriangularBandedMatrixA->GetWidth(),
										 (cuComplex*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulTriangularBanded<cuDoubleComplex>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
														  const CUDADeviceMemory * pTriangularBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
														  SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(cuDoubleComplex) );

	DebugAssert( pTriangularBandedMatrixA->IsAllocated() );
	DebugAssert( pTriangularBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularBandedMatrixA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pTriangularBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	CUDAMemoryRegion hRegionX;
	hRegionX.iWidth = iExpandedSizeA;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasZtbmv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 iExpandedSizeA, iSubDiagsCount,
										 (const cuDoubleComplex *)( pTriangularBandedMatrixA->GetPointer(hPositionA) ), pTriangularBandedMatrixA->GetWidth(),
										 (cuDoubleComplex*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::MulAdd<Float>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, Float fBeta,
								   const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, Float fAlpha,
								   const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
								   CUBLASContextTransposeOp iTransOp ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(Float) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(Float) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(Float) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	CUDAMemoryRegion hRegionX, hRegionY;
	hRegionX.iWidth = 0;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	hRegionY.iWidth = 0;
	hRegionY.iHeight = 0;
	hRegionY.iDepth = 0;
	if ( iTransOp == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		hRegionX.iWidth = hRegionA.iHeight;
		hRegionY.iWidth = hRegionA.iWidth;
	} else {
		hRegionX.iWidth = hRegionA.iWidth;
		hRegionY.iWidth = hRegionA.iHeight;
	}
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionY) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasSgemv( hCUBLASContext, iCUBLASTransposeOp, hRegionA.iWidth, hRegionA.iHeight,
										 &fAlpha, (const Float *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const Float *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (Float*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAdd<Double>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, Double fBeta,
									const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, Double fAlpha,
									const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
									CUBLASContextTransposeOp iTransOp ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(Double) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(Double) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(Double) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	CUDAMemoryRegion hRegionX, hRegionY;
	hRegionX.iWidth = 0;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	hRegionY.iWidth = 0;
	hRegionY.iHeight = 0;
	hRegionY.iDepth = 0;
	if ( iTransOp == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		hRegionX.iWidth = hRegionA.iHeight;
		hRegionY.iWidth = hRegionA.iWidth;
	} else {
		hRegionX.iWidth = hRegionA.iWidth;
		hRegionY.iWidth = hRegionA.iHeight;
	}
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionY) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasDgemv( hCUBLASContext, iCUBLASTransposeOp, hRegionA.iWidth, hRegionA.iHeight,
										 &fAlpha, (const Double *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const Double *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (Double*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAdd<cuComplex>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, cuComplex fBeta,
									   const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, cuComplex fAlpha,
									   const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
									   CUBLASContextTransposeOp iTransOp ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(cuComplex) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(cuComplex) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(cuComplex) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	CUDAMemoryRegion hRegionX, hRegionY;
	hRegionX.iWidth = 0;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	hRegionY.iWidth = 0;
	hRegionY.iHeight = 0;
	hRegionY.iDepth = 0;
	if ( iTransOp == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		hRegionX.iWidth = hRegionA.iHeight;
		hRegionY.iWidth = hRegionA.iWidth;
	} else {
		hRegionX.iWidth = hRegionA.iWidth;
		hRegionY.iWidth = hRegionA.iHeight;
	}
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionY) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasCgemv( hCUBLASContext, iCUBLASTransposeOp, hRegionA.iWidth, hRegionA.iHeight,
										 &fAlpha, (const cuComplex *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const cuComplex *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (cuComplex*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAdd<cuDoubleComplex>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, cuDoubleComplex fBeta,
											 const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, cuDoubleComplex fAlpha,
											 const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
											 CUBLASContextTransposeOp iTransOp ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(cuDoubleComplex) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(cuDoubleComplex) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	CUDAMemoryRegion hRegionX, hRegionY;
	hRegionX.iWidth = 0;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	hRegionY.iWidth = 0;
	hRegionY.iHeight = 0;
	hRegionY.iDepth = 0;
	if ( iTransOp == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		hRegionX.iWidth = hRegionA.iHeight;
		hRegionY.iWidth = hRegionA.iWidth;
	} else {
		hRegionX.iWidth = hRegionA.iWidth;
		hRegionY.iWidth = hRegionA.iHeight;
	}
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionY) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasZgemv( hCUBLASContext, iCUBLASTransposeOp, hRegionA.iWidth, hRegionA.iHeight,
										 &fAlpha, (const cuDoubleComplex *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const cuDoubleComplex *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (cuDoubleComplex*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::MulAddSymmetric<Float>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, Float fBeta,
											const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, Float fAlpha,
											const CUDADeviceMemory * pSymmetricMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
											CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(Float) );
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionA) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(Float) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionA) );

	DebugAssert( pSymmetricMatrixA->IsAllocated() );
	DebugAssert( pSymmetricMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pSymmetricMatrixA->GetStride() == sizeof(Float) );
	DebugAssert( hRegionA.iWidth == hRegionA.iHeight );
	DebugAssert( pSymmetricMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasSsymv( hCUBLASContext, iCUBLASFillMode, hRegionA.iWidth,
										 &fAlpha, (const Float *)( pSymmetricMatrixA->GetPointer(hPositionA) ), pSymmetricMatrixA->GetWidth(),
										 (const Float *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (Float*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddSymmetric<Double>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, Double fBeta,
											 const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, Double fAlpha,
											 const CUDADeviceMemory * pSymmetricMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
											 CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(Double) );
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionA) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(Double) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionA) );

	DebugAssert( pSymmetricMatrixA->IsAllocated() );
	DebugAssert( pSymmetricMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pSymmetricMatrixA->GetStride() == sizeof(Double) );
	DebugAssert( hRegionA.iWidth == hRegionA.iHeight );
	DebugAssert( pSymmetricMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasDsymv( hCUBLASContext, iCUBLASFillMode, hRegionA.iWidth,
										 &fAlpha, (const Double *)( pSymmetricMatrixA->GetPointer(hPositionA) ), pSymmetricMatrixA->GetWidth(),
										 (const Double *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (Double*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddSymmetric<cuComplex>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, cuComplex fBeta,
												const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, cuComplex fAlpha,
												const CUDADeviceMemory * pSymmetricMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
												CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(cuComplex) );
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionA) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(cuComplex) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionA) );

	DebugAssert( pSymmetricMatrixA->IsAllocated() );
	DebugAssert( pSymmetricMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pSymmetricMatrixA->GetStride() == sizeof(cuComplex) );
	DebugAssert( hRegionA.iWidth == hRegionA.iHeight );
	DebugAssert( pSymmetricMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasCsymv( hCUBLASContext, iCUBLASFillMode, hRegionA.iWidth,
										 &fAlpha, (const cuComplex *)( pSymmetricMatrixA->GetPointer(hPositionA) ), pSymmetricMatrixA->GetWidth(),
										 (const cuComplex *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (cuComplex*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddSymmetric<cuDoubleComplex>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, cuDoubleComplex fBeta,
													  const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, cuDoubleComplex fAlpha,
													  const CUDADeviceMemory * pSymmetricMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
													  CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionA) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionA) );

	DebugAssert( pSymmetricMatrixA->IsAllocated() );
	DebugAssert( pSymmetricMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pSymmetricMatrixA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( hRegionA.iWidth == hRegionA.iHeight );
	DebugAssert( pSymmetricMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasZsymv( hCUBLASContext, iCUBLASFillMode, hRegionA.iWidth,
										 &fAlpha, (const cuDoubleComplex *)( pSymmetricMatrixA->GetPointer(hPositionA) ), pSymmetricMatrixA->GetWidth(),
										 (const cuDoubleComplex *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (cuDoubleComplex*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::MulAddHermitian<cuComplex>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, cuComplex fBeta,
											const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, cuComplex fAlpha,
											const CUDADeviceMemory * pHermitianMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
											CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(cuComplex) );
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionA) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(cuComplex) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionA) );

	DebugAssert( pHermitianMatrixA->IsAllocated() );
	DebugAssert( pHermitianMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pHermitianMatrixA->GetStride() == sizeof(cuComplex) );
	DebugAssert( hRegionA.iWidth == hRegionA.iHeight );
	DebugAssert( pHermitianMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasChemv( hCUBLASContext, iCUBLASFillMode, hRegionA.iWidth,
										 &fAlpha, (const cuComplex *)( pHermitianMatrixA->GetPointer(hPositionA) ), pHermitianMatrixA->GetWidth(),
										 (const cuComplex *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (cuComplex*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddHermitian<cuDoubleComplex>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, cuDoubleComplex fBeta,
													  const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, cuDoubleComplex fAlpha,
													  const CUDADeviceMemory * pHermitianMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
													  CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionA) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionA) );

	DebugAssert( pHermitianMatrixA->IsAllocated() );
	DebugAssert( pHermitianMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pHermitianMatrixA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( hRegionA.iWidth == hRegionA.iHeight );
	DebugAssert( pHermitianMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasZhemv( hCUBLASContext, iCUBLASFillMode, hRegionA.iWidth,
										 &fAlpha, (const cuDoubleComplex *)( pHermitianMatrixA->GetPointer(hPositionA) ), pHermitianMatrixA->GetWidth(),
										 (const cuDoubleComplex *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (cuDoubleComplex*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::MulAddBanded<Float>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, Float fBeta,
										 const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, Float fAlpha,
										 const CUDADeviceMemory * pBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
										 SizeT iExpandedSizeA, SizeT iLowerDiagsCount, SizeT iUpperDiagsCount, CUBLASContextTransposeOp iTransOp ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(Float) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(Float) );

	DebugAssert( pBandedMatrixA->IsAllocated() );
	DebugAssert( pBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pBandedMatrixA->GetStride() == sizeof(Float) );
	DebugAssert( pBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	CUDAMemoryRegion hRegionX, hRegionY;
	hRegionX.iWidth = 0;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	hRegionY.iWidth = 0;
	hRegionY.iHeight = 0;
	hRegionY.iDepth = 0;
	SizeT iRowsA, iColsA;
	if ( iTransOp == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		hRegionX.iWidth = hRegionA.iHeight;
		hRegionY.iWidth = hRegionA.iWidth;
		iRowsA = iExpandedSizeA;
		iColsA = hRegionA.iHeight;
	} else {
		hRegionX.iWidth = hRegionA.iWidth;
		hRegionY.iWidth = hRegionA.iHeight;
		iRowsA = hRegionA.iWidth;
		iColsA = iExpandedSizeA;
	}
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionY) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasSgbmv( hCUBLASContext, iCUBLASTransposeOp, iRowsA, iColsA, iLowerDiagsCount, iUpperDiagsCount,
										 &fAlpha, (const Float *)( pBandedMatrixA->GetPointer(hPositionA) ), pBandedMatrixA->GetWidth(),
										 (const Float *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (Float*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddBanded<Double>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, Double fBeta,
										  const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, Double fAlpha,
										  const CUDADeviceMemory * pBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
										  SizeT iExpandedSizeA, SizeT iLowerDiagsCount, SizeT iUpperDiagsCount, CUBLASContextTransposeOp iTransOp ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(Double) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(Double) );

	DebugAssert( pBandedMatrixA->IsAllocated() );
	DebugAssert( pBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pBandedMatrixA->GetStride() == sizeof(Double) );
	DebugAssert( pBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	CUDAMemoryRegion hRegionX, hRegionY;
	hRegionX.iWidth = 0;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	hRegionY.iWidth = 0;
	hRegionY.iHeight = 0;
	hRegionY.iDepth = 0;
	SizeT iRowsA, iColsA;
	if ( iTransOp == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		hRegionX.iWidth = hRegionA.iHeight;
		hRegionY.iWidth = hRegionA.iWidth;
		iRowsA = iExpandedSizeA;
		iColsA = hRegionA.iHeight;
	} else {
		hRegionX.iWidth = hRegionA.iWidth;
		hRegionY.iWidth = hRegionA.iHeight;
		iRowsA = hRegionA.iWidth;
		iColsA = iExpandedSizeA;
	}
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionY) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasDgbmv( hCUBLASContext, iCUBLASTransposeOp, iRowsA, iColsA, iLowerDiagsCount, iUpperDiagsCount,
										 &fAlpha, (const Double *)( pBandedMatrixA->GetPointer(hPositionA) ), pBandedMatrixA->GetWidth(),
										 (const Double *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (Double*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddBanded<cuComplex>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, cuComplex fBeta,
											 const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, cuComplex fAlpha,
											 const CUDADeviceMemory * pBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
											 SizeT iExpandedSizeA, SizeT iLowerDiagsCount, SizeT iUpperDiagsCount, CUBLASContextTransposeOp iTransOp ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(cuComplex) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(cuComplex) );

	DebugAssert( pBandedMatrixA->IsAllocated() );
	DebugAssert( pBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pBandedMatrixA->GetStride() == sizeof(cuComplex) );
	DebugAssert( pBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	CUDAMemoryRegion hRegionX, hRegionY;
	hRegionX.iWidth = 0;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	hRegionY.iWidth = 0;
	hRegionY.iHeight = 0;
	hRegionY.iDepth = 0;
	SizeT iRowsA, iColsA;
	if ( iTransOp == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		hRegionX.iWidth = hRegionA.iHeight;
		hRegionY.iWidth = hRegionA.iWidth;
		iRowsA = iExpandedSizeA;
		iColsA = hRegionA.iHeight;
	} else {
		hRegionX.iWidth = hRegionA.iWidth;
		hRegionY.iWidth = hRegionA.iHeight;
		iRowsA = hRegionA.iWidth;
		iColsA = iExpandedSizeA;
	}
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionY) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasCgbmv( hCUBLASContext, iCUBLASTransposeOp, iRowsA, iColsA, iLowerDiagsCount, iUpperDiagsCount,
										 &fAlpha, (const cuComplex *)( pBandedMatrixA->GetPointer(hPositionA) ), pBandedMatrixA->GetWidth(),
										 (const cuComplex *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (cuComplex*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddBanded<cuDoubleComplex>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, cuDoubleComplex fBeta,
												   const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, cuDoubleComplex fAlpha,
												   const CUDADeviceMemory * pBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
												   SizeT iExpandedSizeA, SizeT iLowerDiagsCount, SizeT iUpperDiagsCount, CUBLASContextTransposeOp iTransOp ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(cuDoubleComplex) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(cuDoubleComplex) );

	DebugAssert( pBandedMatrixA->IsAllocated() );
	DebugAssert( pBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pBandedMatrixA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	CUDAMemoryRegion hRegionX, hRegionY;
	hRegionX.iWidth = 0;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	hRegionY.iWidth = 0;
	hRegionY.iHeight = 0;
	hRegionY.iDepth = 0;
	SizeT iRowsA, iColsA;
	if ( iTransOp == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		hRegionX.iWidth = hRegionA.iHeight;
		hRegionY.iWidth = hRegionA.iWidth;
		iRowsA = iExpandedSizeA;
		iColsA = hRegionA.iHeight;
	} else {
		hRegionX.iWidth = hRegionA.iWidth;
		hRegionY.iWidth = hRegionA.iHeight;
		iRowsA = hRegionA.iWidth;
		iColsA = iExpandedSizeA;
	}
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionY) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasZgbmv( hCUBLASContext, iCUBLASTransposeOp, iRowsA, iColsA, iLowerDiagsCount, iUpperDiagsCount,
										 &fAlpha, (const cuDoubleComplex *)( pBandedMatrixA->GetPointer(hPositionA) ), pBandedMatrixA->GetWidth(),
										 (const cuDoubleComplex *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (cuDoubleComplex*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::MulAddSymmetricBanded<Float>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, Float fBeta,
												  const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, Float fAlpha,
												  const CUDADeviceMemory * pSymmetricBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
												  SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(Float) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(Float) );

	DebugAssert( pSymmetricBandedMatrixA->IsAllocated() );
	DebugAssert( pSymmetricBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pSymmetricBandedMatrixA->GetStride() == sizeof(Float) );
	DebugAssert( pSymmetricBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	CUDAMemoryRegion hRegionVect;
	hRegionVect.iWidth = iExpandedSizeA;
	hRegionVect.iHeight = 0;
	hRegionVect.iDepth = 0;
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionVect) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionVect) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasSsbmv( hCUBLASContext, iCUBLASFillMode, iExpandedSizeA, iSubDiagsCount,
										 &fAlpha, (const Float *)( pSymmetricBandedMatrixA->GetPointer(hPositionA) ), pSymmetricBandedMatrixA->GetWidth(),
										 (const Float *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (Float*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddSymmetricBanded<Double>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, Double fBeta,
												   const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, Double fAlpha,
												   const CUDADeviceMemory * pSymmetricBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
												   SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(Double) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(Double) );

	DebugAssert( pSymmetricBandedMatrixA->IsAllocated() );
	DebugAssert( pSymmetricBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pSymmetricBandedMatrixA->GetStride() == sizeof(Double) );
	DebugAssert( pSymmetricBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	CUDAMemoryRegion hRegionVect;
	hRegionVect.iWidth = iExpandedSizeA;
	hRegionVect.iHeight = 0;
	hRegionVect.iDepth = 0;
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionVect) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionVect) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasDsbmv( hCUBLASContext, iCUBLASFillMode, iExpandedSizeA, iSubDiagsCount,
										 &fAlpha, (const Double *)( pSymmetricBandedMatrixA->GetPointer(hPositionA) ), pSymmetricBandedMatrixA->GetWidth(),
										 (const Double *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (Double*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::MulAddHermitianBanded<cuComplex>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, cuComplex fBeta,
													  const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, cuComplex fAlpha,
													  const CUDADeviceMemory * pHermitianBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
													  SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(cuComplex) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(cuComplex) );

	DebugAssert( pHermitianBandedMatrixA->IsAllocated() );
	DebugAssert( pHermitianBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pHermitianBandedMatrixA->GetStride() == sizeof(cuComplex) );
	DebugAssert( pHermitianBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	CUDAMemoryRegion hRegionVect;
	hRegionVect.iWidth = iExpandedSizeA;
	hRegionVect.iHeight = 0;
	hRegionVect.iDepth = 0;
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionVect) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionVect) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasChbmv( hCUBLASContext, iCUBLASFillMode, iExpandedSizeA, iSubDiagsCount,
										 &fAlpha, (const cuComplex *)( pHermitianBandedMatrixA->GetPointer(hPositionA) ), pHermitianBandedMatrixA->GetWidth(),
										 (const cuComplex *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (cuComplex*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddHermitianBanded<cuDoubleComplex>( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, cuDoubleComplex fBeta,
															const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, cuDoubleComplex fAlpha,
															const CUDADeviceMemory * pHermitianBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
															SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorY->IsAllocated() );
	DebugAssert( outVectorY->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorY->GetStride() == sizeof(cuDoubleComplex) );

	DebugAssert( pVectorX->IsAllocated() );
	DebugAssert( pVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( pVectorX->GetStride() == sizeof(cuDoubleComplex) );

	DebugAssert( pHermitianBandedMatrixA->IsAllocated() );
	DebugAssert( pHermitianBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pHermitianBandedMatrixA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pHermitianBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	CUDAMemoryRegion hRegionVect;
	hRegionVect.iWidth = iExpandedSizeA;
	hRegionVect.iHeight = 0;
	hRegionVect.iDepth = 0;
	DebugAssert( outVectorY->IsValidRegion(outPositionY, hRegionVect) );
	DebugAssert( pVectorX->IsValidRegion(hPositionX, hRegionVect) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasZhbmv( hCUBLASContext, iCUBLASFillMode, iExpandedSizeA, iSubDiagsCount,
										 &fAlpha, (const cuDoubleComplex *)( pHermitianBandedMatrixA->GetPointer(hPositionA) ), pHermitianBandedMatrixA->GetWidth(),
										 (const cuDoubleComplex *)( pVectorX->GetPointer(hPositionX) ), pVectorX->GetStride(),
										 &fBeta, (cuDoubleComplex*)( outVectorY->GetPointer(outPositionY) ), outVectorY->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::SolveTriangular<Float>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
											const CUDADeviceMemory * pTriangularMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
											CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(Float) );
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionA) );

	DebugAssert( pTriangularMatrixA->IsAllocated() );
	DebugAssert( pTriangularMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularMatrixA->GetStride() == sizeof(Float) );
	DebugAssert( hRegionA.iWidth == hRegionA.iHeight );
	DebugAssert( pTriangularMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasStrsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, hRegionA.iWidth,
										 (const Float *)( pTriangularMatrixA->GetPointer(hPositionA) ), pTriangularMatrixA->GetWidth(),
										 (Float*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::SolveTriangular<Double>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
											 const CUDADeviceMemory * pTriangularMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
											 CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(Double) );
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionA) );

	DebugAssert( pTriangularMatrixA->IsAllocated() );
	DebugAssert( pTriangularMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularMatrixA->GetStride() == sizeof(Double) );
	DebugAssert( hRegionA.iWidth == hRegionA.iHeight );
	DebugAssert( pTriangularMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasDtrsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, hRegionA.iWidth,
										 (const Double *)( pTriangularMatrixA->GetPointer(hPositionA) ), pTriangularMatrixA->GetWidth(),
										 (Double*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::SolveTriangular<cuComplex>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
												const CUDADeviceMemory * pTriangularMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
												CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(cuComplex) );
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionA) );

	DebugAssert( pTriangularMatrixA->IsAllocated() );
	DebugAssert( pTriangularMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularMatrixA->GetStride() == sizeof(cuComplex) );
	DebugAssert( hRegionA.iWidth == hRegionA.iHeight );
	DebugAssert( pTriangularMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasCtrsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, hRegionA.iWidth,
										 (const cuComplex *)( pTriangularMatrixA->GetPointer(hPositionA) ), pTriangularMatrixA->GetWidth(),
										 (cuComplex*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::SolveTriangular<cuDoubleComplex>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
													  const CUDADeviceMemory * pTriangularMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
													  CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionA) );

	DebugAssert( pTriangularMatrixA->IsAllocated() );
	DebugAssert( pTriangularMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularMatrixA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( hRegionA.iWidth == hRegionA.iHeight );
	DebugAssert( pTriangularMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasZtrsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, hRegionA.iWidth,
										 (const cuDoubleComplex *)( pTriangularMatrixA->GetPointer(hPositionA) ), pTriangularMatrixA->GetWidth(),
										 (cuDoubleComplex*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::SolveTriangularBanded<Float>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
												  const CUDADeviceMemory * pTriangularBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
												  SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(Float) );

	DebugAssert( pTriangularBandedMatrixA->IsAllocated() );
	DebugAssert( pTriangularBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularBandedMatrixA->GetStride() == sizeof(Float) );
	DebugAssert( pTriangularBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	CUDAMemoryRegion hRegionX;
	hRegionX.iWidth = iExpandedSizeA;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasStbsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 iExpandedSizeA, iSubDiagsCount,
										 (const Float *)( pTriangularBandedMatrixA->GetPointer(hPositionA) ), pTriangularBandedMatrixA->GetWidth(),
										 (Float*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::SolveTriangularBanded<Double>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
												   const CUDADeviceMemory * pTriangularBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
												   SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(Double) );

	DebugAssert( pTriangularBandedMatrixA->IsAllocated() );
	DebugAssert( pTriangularBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularBandedMatrixA->GetStride() == sizeof(Double) );
	DebugAssert( pTriangularBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	CUDAMemoryRegion hRegionX;
	hRegionX.iWidth = iExpandedSizeA;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasDtbsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 iExpandedSizeA, iSubDiagsCount,
										 (const Double *)( pTriangularBandedMatrixA->GetPointer(hPositionA) ), pTriangularBandedMatrixA->GetWidth(),
										 (Double*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::SolveTriangularBanded<cuComplex>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
													  const CUDADeviceMemory * pTriangularBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
													  SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(cuComplex) );

	DebugAssert( pTriangularBandedMatrixA->IsAllocated() );
	DebugAssert( pTriangularBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularBandedMatrixA->GetStride() == sizeof(cuComplex) );
	DebugAssert( pTriangularBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	CUDAMemoryRegion hRegionX;
	hRegionX.iWidth = iExpandedSizeA;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasCtbsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 iExpandedSizeA, iSubDiagsCount,
										 (const cuComplex *)( pTriangularBandedMatrixA->GetPointer(hPositionA) ), pTriangularBandedMatrixA->GetWidth(),
										 (cuComplex*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::SolveTriangularBanded<cuDoubleComplex>( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
															const CUDADeviceMemory * pTriangularBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
															SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outVectorX->IsAllocated() );
	DebugAssert( outVectorX->GetShape() == CUDA_MEMORY_SHAPE_1D );
	DebugAssert( outVectorX->GetStride() == sizeof(cuDoubleComplex) );

	DebugAssert( pTriangularBandedMatrixA->IsAllocated() );
	DebugAssert( pTriangularBandedMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pTriangularBandedMatrixA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pTriangularBandedMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( iFillMode < CUBLAS_CONTEXT_FILLMODE_FULL );

	CUDAMemoryRegion hRegionX;
	hRegionX.iWidth = iExpandedSizeA;
	hRegionX.iHeight = 0;
	hRegionX.iDepth = 0;
	DebugAssert( outVectorX->IsValidRegion(outPositionX, hRegionX) );

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOp = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOp] );
	
	cublasStatus_t iError = cublasZtbsv( hCUBLASContext, iCUBLASFillMode, iCUBLASTransposeOp, bMainDiagIsUnity ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 iExpandedSizeA, iSubDiagsCount,
										 (const cuDoubleComplex *)( pTriangularBandedMatrixA->GetPointer(hPositionA) ), pTriangularBandedMatrixA->GetWidth(),
										 (cuDoubleComplex*)( outVectorX->GetPointer(outPositionX) ), outVectorX->GetStride() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
