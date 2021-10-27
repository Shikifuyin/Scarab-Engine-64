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



template<>
Void CUBLASContext::MulTriangular<Float>( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC,
										  const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, Float fAlpha,
										  const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
										  CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixC->IsAllocated() );
	DebugAssert( outMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixC->GetStride() == sizeof(Float) );
	DebugAssert( outMatrixC->IsValidRegion(outPositionC, outRegionC) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(Float) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( pMatrixB->IsAllocated() );
	DebugAssert( pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixB->GetStride() == sizeof(Float) );
	DebugAssert( pMatrixB->IsValidRegion(hPositionB, hRegionB) );

	DebugAssert( hRegionB.iWidth == outRegionC.iWidth );
	DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	
	cublasStatus_t iError = cublasStrmm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 hRegionB.iWidth, hRegionB.iHeight,
										 &fAlpha, (const Float *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const Float *)( pMatrixB->GetPointer(hPositionB) ), pMatrixB->GetWidth(),
										 (Float*)( outMatrixC->GetPointer(outPositionC) ), outMatrixC->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulTriangular<Double>( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC,
										   const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, Double fAlpha,
										   const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
										   CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixC->IsAllocated() );
	DebugAssert( outMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixC->GetStride() == sizeof(Double) );
	DebugAssert( outMatrixC->IsValidRegion(outPositionC, outRegionC) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(Double) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( pMatrixB->IsAllocated() );
	DebugAssert( pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixB->GetStride() == sizeof(Double) );
	DebugAssert( pMatrixB->IsValidRegion(hPositionB, hRegionB) );

	DebugAssert( hRegionB.iWidth == outRegionC.iWidth );
	DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	
	cublasStatus_t iError = cublasDtrmm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 hRegionB.iWidth, hRegionB.iHeight,
										 &fAlpha, (const Double *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const Double *)( pMatrixB->GetPointer(hPositionB) ), pMatrixB->GetWidth(),
										 (Double*)( outMatrixC->GetPointer(outPositionC) ), outMatrixC->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulTriangular<cuComplex>( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC,
											  const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, cuComplex fAlpha,
											  const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
											  CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixC->IsAllocated() );
	DebugAssert( outMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixC->GetStride() == sizeof(cuComplex) );
	DebugAssert( outMatrixC->IsValidRegion(outPositionC, outRegionC) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(cuComplex) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( pMatrixB->IsAllocated() );
	DebugAssert( pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixB->GetStride() == sizeof(cuComplex) );
	DebugAssert( pMatrixB->IsValidRegion(hPositionB, hRegionB) );

	DebugAssert( hRegionB.iWidth == outRegionC.iWidth );
	DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	
	cublasStatus_t iError = cublasCtrmm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 hRegionB.iWidth, hRegionB.iHeight,
										 &fAlpha, (const cuComplex *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const cuComplex *)( pMatrixB->GetPointer(hPositionB) ), pMatrixB->GetWidth(),
										 (cuComplex*)( outMatrixC->GetPointer(outPositionC) ), outMatrixC->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulTriangular<cuDoubleComplex>( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC,
													const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, cuDoubleComplex fAlpha,
													const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
													CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixC->IsAllocated() );
	DebugAssert( outMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixC->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( outMatrixC->IsValidRegion(outPositionC, outRegionC) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( pMatrixB->IsAllocated() );
	DebugAssert( pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixB->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pMatrixB->IsValidRegion(hPositionB, hRegionB) );

	DebugAssert( hRegionB.iWidth == outRegionC.iWidth );
	DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	
	cublasStatus_t iError = cublasZtrmm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 hRegionB.iWidth, hRegionB.iHeight,
										 &fAlpha, (const cuDoubleComplex *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const cuDoubleComplex *)( pMatrixB->GetPointer(hPositionB) ), pMatrixB->GetWidth(),
										 (cuDoubleComplex*)( outMatrixC->GetPointer(outPositionC) ), outMatrixC->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::MulAdd<Float>( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC, Float fBeta,
								   const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, Float fAlpha,
								   const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
								   CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB, Bool ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixC->IsAllocated() );
	DebugAssert( outMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixC->GetStride() == sizeof(Float) );
	DebugAssert( outMatrixC->IsValidRegion(outPositionC, outRegionC) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(Float) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( pMatrixB->IsAllocated() );
	DebugAssert( pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixB->GetStride() == sizeof(Float) );
	DebugAssert( pMatrixB->IsValidRegion(hPositionB, hRegionB) );

	SizeT iM, iN, iK;
	if ( iTransOpA == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionA.iWidth == outRegionC.iWidth );
		iM = hRegionA.iWidth;
		iK = hRegionA.iHeight;
	} else {
		DebugAssert( hRegionA.iHeight == outRegionC.iWidth );
		iM = hRegionA.iHeight;
		iK = hRegionA.iWidth;
	}
	if ( iTransOpB == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
		iN = hRegionB.iHeight;
		DebugAssert( iK == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionB.iWidth == outRegionC.iHeight );
		iN = hRegionB.iWidth;
		DebugAssert( iK == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );
	
	cublasStatus_t iError = cublasSgemm( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
										 &fAlpha, (const Float *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const Float *)( pMatrixB->GetPointer(hPositionB) ), pMatrixB->GetWidth(),
										 &fBeta, (Float*)( outMatrixC->GetPointer(outPositionC) ), outMatrixC->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAdd<Double>( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC, Double fBeta,
									const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, Double fAlpha,
									const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
									CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB, Bool ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixC->IsAllocated() );
	DebugAssert( outMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixC->GetStride() == sizeof(Double) );
	DebugAssert( outMatrixC->IsValidRegion(outPositionC, outRegionC) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(Double) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( pMatrixB->IsAllocated() );
	DebugAssert( pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixB->GetStride() == sizeof(Double) );
	DebugAssert( pMatrixB->IsValidRegion(hPositionB, hRegionB) );

	SizeT iM, iN, iK;
	if ( iTransOpA == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionA.iWidth == outRegionC.iWidth );
		iM = hRegionA.iWidth;
		iK = hRegionA.iHeight;
	} else {
		DebugAssert( hRegionA.iHeight == outRegionC.iWidth );
		iM = hRegionA.iHeight;
		iK = hRegionA.iWidth;
	}
	if ( iTransOpB == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
		iN = hRegionB.iHeight;
		DebugAssert( iK == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionB.iWidth == outRegionC.iHeight );
		iN = hRegionB.iWidth;
		DebugAssert( iK == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );
	
	cublasStatus_t iError = cublasDgemm( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
										 &fAlpha, (const Double *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const Double *)( pMatrixB->GetPointer(hPositionB) ), pMatrixB->GetWidth(),
										 &fBeta, (Double*)( outMatrixC->GetPointer(outPositionC) ), outMatrixC->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAdd<cuComplex>( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC, cuComplex fBeta,
									   const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, cuComplex fAlpha,
									   const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
									   CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB, Bool bUseComplexGaussReduction ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixC->IsAllocated() );
	DebugAssert( outMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixC->GetStride() == sizeof(cuComplex) );
	DebugAssert( outMatrixC->IsValidRegion(outPositionC, outRegionC) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(cuComplex) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( pMatrixB->IsAllocated() );
	DebugAssert( pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixB->GetStride() == sizeof(cuComplex) );
	DebugAssert( pMatrixB->IsValidRegion(hPositionB, hRegionB) );

	SizeT iM, iN, iK;
	if ( iTransOpA == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionA.iWidth == outRegionC.iWidth );
		iM = hRegionA.iWidth;
		iK = hRegionA.iHeight;
	} else {
		DebugAssert( hRegionA.iHeight == outRegionC.iWidth );
		iM = hRegionA.iHeight;
		iK = hRegionA.iWidth;
	}
	if ( iTransOpB == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
		iN = hRegionB.iHeight;
		DebugAssert( iK == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionB.iWidth == outRegionC.iHeight );
		iN = hRegionB.iWidth;
		DebugAssert( iK == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );
	
	if ( bUseComplexGaussReduction ) {
		cublasStatus_t iError = cublasCgemm3m( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
											   &fAlpha, (const cuComplex *)(pMatrixA->GetPointer( hPositionA )), pMatrixA->GetWidth(),
											   (const cuComplex *)(pMatrixB->GetPointer( hPositionB )), pMatrixB->GetWidth(),
											   &fBeta, (cuComplex *)(outMatrixC->GetPointer( outPositionC )), outMatrixC->GetWidth() );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	} else {
		cublasStatus_t iError = cublasCgemm( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
											 &fAlpha, (const cuComplex *)(pMatrixA->GetPointer( hPositionA )), pMatrixA->GetWidth(),
											 (const cuComplex *)(pMatrixB->GetPointer( hPositionB )), pMatrixB->GetWidth(),
											 &fBeta, (cuComplex *)(outMatrixC->GetPointer( outPositionC )), outMatrixC->GetWidth() );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	}
}
template<>
Void CUBLASContext::MulAdd<cuDoubleComplex>( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC, cuDoubleComplex fBeta,
											 const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, cuDoubleComplex fAlpha,
											 const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
											 CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB, Bool bUseComplexGaussReduction ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixC->IsAllocated() );
	DebugAssert( outMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixC->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( outMatrixC->IsValidRegion(outPositionC, outRegionC) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( pMatrixB->IsAllocated() );
	DebugAssert( pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixB->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pMatrixB->IsValidRegion(hPositionB, hRegionB) );

	SizeT iM, iN, iK;
	if ( iTransOpA == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionA.iWidth == outRegionC.iWidth );
		iM = hRegionA.iWidth;
		iK = hRegionA.iHeight;
	} else {
		DebugAssert( hRegionA.iHeight == outRegionC.iWidth );
		iM = hRegionA.iHeight;
		iK = hRegionA.iWidth;
	}
	if ( iTransOpB == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
		iN = hRegionB.iHeight;
		DebugAssert( iK == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionB.iWidth == outRegionC.iHeight );
		iN = hRegionB.iWidth;
		DebugAssert( iK == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );
	
	if ( bUseComplexGaussReduction ) {
		cublasStatus_t iError = cublasZgemm3m( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
											   &fAlpha, (const cuDoubleComplex *)(pMatrixA->GetPointer( hPositionA )), pMatrixA->GetWidth(),
											   (const cuDoubleComplex *)(pMatrixB->GetPointer( hPositionB )), pMatrixB->GetWidth(),
											   &fBeta, (cuDoubleComplex *)(outMatrixC->GetPointer( outPositionC )), outMatrixC->GetWidth() );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	} else {
		cublasStatus_t iError = cublasZgemm( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
											 &fAlpha, (const cuDoubleComplex *)(pMatrixA->GetPointer( hPositionA )), pMatrixA->GetWidth(),
											 (const cuDoubleComplex *)(pMatrixB->GetPointer( hPositionB )), pMatrixB->GetWidth(),
											 &fBeta, (cuDoubleComplex *)(outMatrixC->GetPointer( outPositionC )), outMatrixC->GetWidth() );
		DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
	}
}

template<>
Void CUBLASContext::MulAddSymmetric<Float>( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC, Float fBeta,
											const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, Float fAlpha,
											const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
											CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixC->IsAllocated() );
	DebugAssert( outMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixC->GetStride() == sizeof(Float) );
	DebugAssert( outMatrixC->IsValidRegion(outPositionC, outRegionC) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(Float) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( pMatrixB->IsAllocated() );
	DebugAssert( pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixB->GetStride() == sizeof(Float) );
	DebugAssert( pMatrixB->IsValidRegion(hPositionB, hRegionB) );

	DebugAssert( hRegionB.iWidth == outRegionC.iWidth );
	DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasSsymm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, outRegionC.iWidth, outRegionC.iHeight,
										 &fAlpha, (const Float *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const Float *)( pMatrixB->GetPointer(hPositionB) ), pMatrixB->GetWidth(),
										 &fBeta, (Float*)( outMatrixC->GetPointer(outPositionC) ), outMatrixC->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddSymmetric<Double>( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC, Double fBeta,
											 const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, Double fAlpha,
											 const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
											 CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixC->IsAllocated() );
	DebugAssert( outMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixC->GetStride() == sizeof(Double) );
	DebugAssert( outMatrixC->IsValidRegion(outPositionC, outRegionC) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(Double) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( pMatrixB->IsAllocated() );
	DebugAssert( pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixB->GetStride() == sizeof(Double) );
	DebugAssert( pMatrixB->IsValidRegion(hPositionB, hRegionB) );

	DebugAssert( hRegionB.iWidth == outRegionC.iWidth );
	DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasDsymm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, outRegionC.iWidth, outRegionC.iHeight,
										 &fAlpha, (const Double *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const Double *)( pMatrixB->GetPointer(hPositionB) ), pMatrixB->GetWidth(),
										 &fBeta, (Double*)( outMatrixC->GetPointer(outPositionC) ), outMatrixC->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddSymmetric<cuComplex>( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC, cuComplex fBeta,
												const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, cuComplex fAlpha,
												const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
												CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixC->IsAllocated() );
	DebugAssert( outMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixC->GetStride() == sizeof(cuComplex) );
	DebugAssert( outMatrixC->IsValidRegion(outPositionC, outRegionC) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(cuComplex) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( pMatrixB->IsAllocated() );
	DebugAssert( pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixB->GetStride() == sizeof(cuComplex) );
	DebugAssert( pMatrixB->IsValidRegion(hPositionB, hRegionB) );

	DebugAssert( hRegionB.iWidth == outRegionC.iWidth );
	DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasCsymm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, outRegionC.iWidth, outRegionC.iHeight,
										 &fAlpha, (const cuComplex *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const cuComplex *)( pMatrixB->GetPointer(hPositionB) ), pMatrixB->GetWidth(),
										 &fBeta, (cuComplex*)( outMatrixC->GetPointer(outPositionC) ), outMatrixC->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddSymmetric<cuDoubleComplex>( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC, cuDoubleComplex fBeta,
													  const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, cuDoubleComplex fAlpha,
													  const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
													  CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixC->IsAllocated() );
	DebugAssert( outMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixC->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( outMatrixC->IsValidRegion(outPositionC, outRegionC) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( pMatrixB->IsAllocated() );
	DebugAssert( pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixB->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pMatrixB->IsValidRegion(hPositionB, hRegionB) );

	DebugAssert( hRegionB.iWidth == outRegionC.iWidth );
	DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasZsymm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, outRegionC.iWidth, outRegionC.iHeight,
										 &fAlpha, (const cuDoubleComplex *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const cuDoubleComplex *)( pMatrixB->GetPointer(hPositionB) ), pMatrixB->GetWidth(),
										 &fBeta, (cuDoubleComplex*)( outMatrixC->GetPointer(outPositionC) ), outMatrixC->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::MulAddHermitian<cuComplex>( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC, cuComplex fBeta,
												const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, cuComplex fAlpha,
												const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
												CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixC->IsAllocated() );
	DebugAssert( outMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixC->GetStride() == sizeof(cuComplex) );
	DebugAssert( outMatrixC->IsValidRegion(outPositionC, outRegionC) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(cuComplex) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( pMatrixB->IsAllocated() );
	DebugAssert( pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixB->GetStride() == sizeof(cuComplex) );
	DebugAssert( pMatrixB->IsValidRegion(hPositionB, hRegionB) );

	DebugAssert( hRegionB.iWidth == outRegionC.iWidth );
	DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasChemm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, outRegionC.iWidth, outRegionC.iHeight,
										 &fAlpha, (const cuComplex *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const cuComplex *)( pMatrixB->GetPointer(hPositionB) ), pMatrixB->GetWidth(),
										 &fBeta, (cuComplex*)( outMatrixC->GetPointer(outPositionC) ), outMatrixC->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddHermitian<cuDoubleComplex>( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC, cuDoubleComplex fBeta,
													  const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, cuDoubleComplex fAlpha,
													  const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
													  CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixC->IsAllocated() );
	DebugAssert( outMatrixC->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixC->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( outMatrixC->IsValidRegion(outPositionC, outRegionC) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	DebugAssert( pMatrixB->IsAllocated() );
	DebugAssert( pMatrixB->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixB->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pMatrixB->IsValidRegion(hPositionB, hRegionB) );

	DebugAssert( hRegionB.iWidth == outRegionC.iWidth );
	DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	
	cublasStatus_t iError = cublasZhemm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, outRegionC.iWidth, outRegionC.iHeight,
										 &fAlpha, (const cuDoubleComplex *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (const cuDoubleComplex *)( pMatrixB->GetPointer(hPositionB) ), pMatrixB->GetWidth(),
										 &fBeta, (cuDoubleComplex*)( outMatrixC->GetPointer(outPositionC) ), outMatrixC->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::MulAddBatched<Float>( SizeT iBatchCount, CUDADeviceMemory * outMatricesC, const CUDAMemoryPosition * outPositionsC, const CUDAMemoryRegion & outRegionC, Float fBeta,
										  const CUDADeviceMemory * arrMatricesA, const CUDAMemoryPosition * arrPositionsA, const CUDAMemoryRegion & hRegionA, Float fAlpha,
										  const CUDADeviceMemory * arrMatricesB, const CUDAMemoryPosition * arrPositionsB, const CUDAMemoryRegion & hRegionB,
										  CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( iBatchCount <= CUBLAS_BATCH_MAX_COUNT );

	// Empty Call
	if ( iBatchCount == 0 )
		return;

	// Prepare Batch Data
	Float * arrBatchMatricesC[CUBLAS_BATCH_MAX_COUNT];
	const Float * arrBatchMatricesA[CUBLAS_BATCH_MAX_COUNT];
	const Float * arrBatchMatricesB[CUBLAS_BATCH_MAX_COUNT];

	SizeT iReferenceWidthC = outMatricesC[0].GetWidth();
	SizeT iReferenceWidthA = arrMatricesA[0].GetWidth();
	SizeT iReferenceWidthB = arrMatricesB[0].GetWidth();

	for( UInt i = 0; i < iBatchCount; ++i ) {
		DebugAssert( outMatricesC[i].IsAllocated() );
		DebugAssert( outMatricesC[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( outMatricesC[i].GetStride() == sizeof(Float) );
		DebugAssert( outMatricesC[i].GetWidth() == iReferenceWidthC );
		DebugAssert( outMatricesC[i].IsValidRegion(outPositionsC[i], outRegionC) );

		DebugAssert( arrMatricesA[i].IsAllocated() );
		DebugAssert( arrMatricesA[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( arrMatricesA[i].GetStride() == sizeof(Float) );
		DebugAssert( arrMatricesA[i].GetWidth() == iReferenceWidthA );
		DebugAssert( arrMatricesA[i].IsValidRegion(arrPositionsA[i], hRegionA) );

		DebugAssert( arrMatricesB[i].IsAllocated() );
		DebugAssert( arrMatricesB[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( arrMatricesB[i].GetStride() == sizeof(Float) );
		DebugAssert( arrMatricesB[i].GetWidth() == iReferenceWidthB );
		DebugAssert( arrMatricesB[i].IsValidRegion(arrPositionsB[i], hRegionB) );

		arrBatchMatricesC[i] = (Float*)( outMatricesC[i].GetPointer(outPositionsC[i]) );
		arrBatchMatricesA[i] = (const Float *)( arrMatricesA[i].GetPointer(arrPositionsA[i]) );
		arrBatchMatricesB[i] = (const Float *)( arrMatricesB[i].GetPointer(arrPositionsB[i]) );
	}

	SizeT iM, iN, iK;
	if ( iTransOpA == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionA.iWidth == outRegionC.iWidth );
		iM = hRegionA.iWidth;
		iK = hRegionA.iHeight;
	} else {
		DebugAssert( hRegionA.iHeight == outRegionC.iWidth );
		iM = hRegionA.iHeight;
		iK = hRegionA.iWidth;
	}
	if ( iTransOpB == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
		iN = hRegionB.iHeight;
		DebugAssert( iK == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionB.iWidth == outRegionC.iHeight );
		iN = hRegionB.iWidth;
		DebugAssert( iK == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );
	
	cublasStatus_t iError = cublasSgemmBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
												&fAlpha, arrBatchMatricesA, iReferenceWidthA,
												arrBatchMatricesB, iReferenceWidthB,
												&fBeta, arrBatchMatricesC, iReferenceWidthC, iBatchCount );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddBatched<Double>( SizeT iBatchCount, CUDADeviceMemory * outMatricesC, const CUDAMemoryPosition * outPositionsC, const CUDAMemoryRegion & outRegionC, Double fBeta,
										   const CUDADeviceMemory * arrMatricesA, const CUDAMemoryPosition * arrPositionsA, const CUDAMemoryRegion & hRegionA, Double fAlpha,
										   const CUDADeviceMemory * arrMatricesB, const CUDAMemoryPosition * arrPositionsB, const CUDAMemoryRegion & hRegionB,
										   CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( iBatchCount <= CUBLAS_BATCH_MAX_COUNT );

	// Empty Call
	if ( iBatchCount == 0 )
		return;

	// Prepare Batch Data
	Double * arrBatchMatricesC[CUBLAS_BATCH_MAX_COUNT];
	const Double * arrBatchMatricesA[CUBLAS_BATCH_MAX_COUNT];
	const Double * arrBatchMatricesB[CUBLAS_BATCH_MAX_COUNT];

	SizeT iReferenceWidthC = outMatricesC[0].GetWidth();
	SizeT iReferenceWidthA = arrMatricesA[0].GetWidth();
	SizeT iReferenceWidthB = arrMatricesB[0].GetWidth();

	for( UInt i = 0; i < iBatchCount; ++i ) {
		DebugAssert( outMatricesC[i].IsAllocated() );
		DebugAssert( outMatricesC[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( outMatricesC[i].GetStride() == sizeof(Double) );
		DebugAssert( outMatricesC[i].GetWidth() == iReferenceWidthC );
		DebugAssert( outMatricesC[i].IsValidRegion(outPositionsC[i], outRegionC) );

		DebugAssert( arrMatricesA[i].IsAllocated() );
		DebugAssert( arrMatricesA[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( arrMatricesA[i].GetStride() == sizeof(Double) );
		DebugAssert( arrMatricesA[i].GetWidth() == iReferenceWidthA );
		DebugAssert( arrMatricesA[i].IsValidRegion(arrPositionsA[i], hRegionA) );

		DebugAssert( arrMatricesB[i].IsAllocated() );
		DebugAssert( arrMatricesB[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( arrMatricesB[i].GetStride() == sizeof(Double) );
		DebugAssert( arrMatricesB[i].GetWidth() == iReferenceWidthB );
		DebugAssert( arrMatricesB[i].IsValidRegion(arrPositionsB[i], hRegionB) );

		arrBatchMatricesC[i] = (Double*)( outMatricesC[i].GetPointer(outPositionsC[i]) );
		arrBatchMatricesA[i] = (const Double *)( arrMatricesA[i].GetPointer(arrPositionsA[i]) );
		arrBatchMatricesB[i] = (const Double *)( arrMatricesB[i].GetPointer(arrPositionsB[i]) );
	}

	SizeT iM, iN, iK;
	if ( iTransOpA == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionA.iWidth == outRegionC.iWidth );
		iM = hRegionA.iWidth;
		iK = hRegionA.iHeight;
	} else {
		DebugAssert( hRegionA.iHeight == outRegionC.iWidth );
		iM = hRegionA.iHeight;
		iK = hRegionA.iWidth;
	}
	if ( iTransOpB == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
		iN = hRegionB.iHeight;
		DebugAssert( iK == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionB.iWidth == outRegionC.iHeight );
		iN = hRegionB.iWidth;
		DebugAssert( iK == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );
	
	cublasStatus_t iError = cublasDgemmBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
												&fAlpha, arrBatchMatricesA, iReferenceWidthA,
												arrBatchMatricesB, iReferenceWidthB,
												&fBeta, arrBatchMatricesC, iReferenceWidthC, iBatchCount );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddBatched<cuComplex>( SizeT iBatchCount, CUDADeviceMemory * outMatricesC, const CUDAMemoryPosition * outPositionsC, const CUDAMemoryRegion & outRegionC, cuComplex fBeta,
											  const CUDADeviceMemory * arrMatricesA, const CUDAMemoryPosition * arrPositionsA, const CUDAMemoryRegion & hRegionA, cuComplex fAlpha,
											  const CUDADeviceMemory * arrMatricesB, const CUDAMemoryPosition * arrPositionsB, const CUDAMemoryRegion & hRegionB,
											  CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( iBatchCount <= CUBLAS_BATCH_MAX_COUNT );

	// Empty Call
	if ( iBatchCount == 0 )
		return;

	// Prepare Batch Data
	cuComplex * arrBatchMatricesC[CUBLAS_BATCH_MAX_COUNT];
	const cuComplex * arrBatchMatricesA[CUBLAS_BATCH_MAX_COUNT];
	const cuComplex * arrBatchMatricesB[CUBLAS_BATCH_MAX_COUNT];

	SizeT iReferenceWidthC = outMatricesC[0].GetWidth();
	SizeT iReferenceWidthA = arrMatricesA[0].GetWidth();
	SizeT iReferenceWidthB = arrMatricesB[0].GetWidth();

	for( UInt i = 0; i < iBatchCount; ++i ) {
		DebugAssert( outMatricesC[i].IsAllocated() );
		DebugAssert( outMatricesC[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( outMatricesC[i].GetStride() == sizeof(cuComplex) );
		DebugAssert( outMatricesC[i].GetWidth() == iReferenceWidthC );
		DebugAssert( outMatricesC[i].IsValidRegion(outPositionsC[i], outRegionC) );

		DebugAssert( arrMatricesA[i].IsAllocated() );
		DebugAssert( arrMatricesA[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( arrMatricesA[i].GetStride() == sizeof(cuComplex) );
		DebugAssert( arrMatricesA[i].GetWidth() == iReferenceWidthA );
		DebugAssert( arrMatricesA[i].IsValidRegion(arrPositionsA[i], hRegionA) );

		DebugAssert( arrMatricesB[i].IsAllocated() );
		DebugAssert( arrMatricesB[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( arrMatricesB[i].GetStride() == sizeof(cuComplex) );
		DebugAssert( arrMatricesB[i].GetWidth() == iReferenceWidthB );
		DebugAssert( arrMatricesB[i].IsValidRegion(arrPositionsB[i], hRegionB) );

		arrBatchMatricesC[i] = (cuComplex*)( outMatricesC[i].GetPointer(outPositionsC[i]) );
		arrBatchMatricesA[i] = (const cuComplex *)( arrMatricesA[i].GetPointer(arrPositionsA[i]) );
		arrBatchMatricesB[i] = (const cuComplex *)( arrMatricesB[i].GetPointer(arrPositionsB[i]) );
	}

	SizeT iM, iN, iK;
	if ( iTransOpA == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionA.iWidth == outRegionC.iWidth );
		iM = hRegionA.iWidth;
		iK = hRegionA.iHeight;
	} else {
		DebugAssert( hRegionA.iHeight == outRegionC.iWidth );
		iM = hRegionA.iHeight;
		iK = hRegionA.iWidth;
	}
	if ( iTransOpB == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
		iN = hRegionB.iHeight;
		DebugAssert( iK == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionB.iWidth == outRegionC.iHeight );
		iN = hRegionB.iWidth;
		DebugAssert( iK == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );
	
	cublasStatus_t iError = cublasCgemmBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
												&fAlpha, arrBatchMatricesA, iReferenceWidthA,
												arrBatchMatricesB, iReferenceWidthB,
												&fBeta, arrBatchMatricesC, iReferenceWidthC, iBatchCount );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddBatched<cuDoubleComplex>( SizeT iBatchCount, CUDADeviceMemory * outMatricesC, const CUDAMemoryPosition * outPositionsC, const CUDAMemoryRegion & outRegionC, cuDoubleComplex fBeta,
													const CUDADeviceMemory * arrMatricesA, const CUDAMemoryPosition * arrPositionsA, const CUDAMemoryRegion & hRegionA, cuDoubleComplex fAlpha,
													const CUDADeviceMemory * arrMatricesB, const CUDAMemoryPosition * arrPositionsB, const CUDAMemoryRegion & hRegionB,
													CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( iBatchCount <= CUBLAS_BATCH_MAX_COUNT );

	// Empty Call
	if ( iBatchCount == 0 )
		return;

	// Prepare Batch Data
	cuDoubleComplex * arrBatchMatricesC[CUBLAS_BATCH_MAX_COUNT];
	const cuDoubleComplex * arrBatchMatricesA[CUBLAS_BATCH_MAX_COUNT];
	const cuDoubleComplex * arrBatchMatricesB[CUBLAS_BATCH_MAX_COUNT];

	SizeT iReferenceWidthC = outMatricesC[0].GetWidth();
	SizeT iReferenceWidthA = arrMatricesA[0].GetWidth();
	SizeT iReferenceWidthB = arrMatricesB[0].GetWidth();

	for( UInt i = 0; i < iBatchCount; ++i ) {
		DebugAssert( outMatricesC[i].IsAllocated() );
		DebugAssert( outMatricesC[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( outMatricesC[i].GetStride() == sizeof(cuDoubleComplex) );
		DebugAssert( outMatricesC[i].GetWidth() == iReferenceWidthC );
		DebugAssert( outMatricesC[i].IsValidRegion(outPositionsC[i], outRegionC) );

		DebugAssert( arrMatricesA[i].IsAllocated() );
		DebugAssert( arrMatricesA[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( arrMatricesA[i].GetStride() == sizeof(cuDoubleComplex) );
		DebugAssert( arrMatricesA[i].GetWidth() == iReferenceWidthA );
		DebugAssert( arrMatricesA[i].IsValidRegion(arrPositionsA[i], hRegionA) );

		DebugAssert( arrMatricesB[i].IsAllocated() );
		DebugAssert( arrMatricesB[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( arrMatricesB[i].GetStride() == sizeof(cuDoubleComplex) );
		DebugAssert( arrMatricesB[i].GetWidth() == iReferenceWidthB );
		DebugAssert( arrMatricesB[i].IsValidRegion(arrPositionsB[i], hRegionB) );

		arrBatchMatricesC[i] = (cuDoubleComplex*)( outMatricesC[i].GetPointer(outPositionsC[i]) );
		arrBatchMatricesA[i] = (const cuDoubleComplex *)( arrMatricesA[i].GetPointer(arrPositionsA[i]) );
		arrBatchMatricesB[i] = (const cuDoubleComplex *)( arrMatricesB[i].GetPointer(arrPositionsB[i]) );
	}

	SizeT iM, iN, iK;
	if ( iTransOpA == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionA.iWidth == outRegionC.iWidth );
		iM = hRegionA.iWidth;
		iK = hRegionA.iHeight;
	} else {
		DebugAssert( hRegionA.iHeight == outRegionC.iWidth );
		iM = hRegionA.iHeight;
		iK = hRegionA.iWidth;
	}
	if ( iTransOpB == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
		iN = hRegionB.iHeight;
		DebugAssert( iK == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionB.iWidth == outRegionC.iHeight );
		iN = hRegionB.iWidth;
		DebugAssert( iK == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );
	
	cublasStatus_t iError = cublasZgemmBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
												&fAlpha, arrBatchMatricesA, iReferenceWidthA,
												arrBatchMatricesB, iReferenceWidthB,
												&fBeta, arrBatchMatricesC, iReferenceWidthC, iBatchCount );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::MulAddStrideBatched<Float>( SizeT iBatchCount, CUDADeviceMemory * outMatricesC, const CUDAMemoryPosition & outStartPositionC, const CUDAMemoryRegion & outRegionC, SizeT outStrideC, Float fBeta,
												const CUDADeviceMemory * arrMatricesA, const CUDAMemoryPosition & hStartPositionA, const CUDAMemoryRegion & hRegionA, SizeT iStrideA, Float fAlpha,
												const CUDADeviceMemory * arrMatricesB, const CUDAMemoryPosition & hStartPositionB, const CUDAMemoryRegion & hRegionB, SizeT iStrideB,
												CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( outStrideC >= outRegionC.iWidth * outRegionC.iHeight );
	DebugAssert( iStrideA >= hRegionA.iWidth * hRegionA.iHeight );
	DebugAssert( iStrideB >= hRegionB.iWidth * hRegionB.iHeight );

	DebugAssert( outMatricesC->IsAllocated() );
	DebugAssert( outMatricesC->GetShape() == CUDA_MEMORY_SHAPE_3D );
	DebugAssert( outMatricesC->GetStride() == sizeof(Float) );
	DebugAssert( outRegionC.iDepth == 1 );
	DebugAssert( outMatricesC->IsValidRegion(outStartPositionC, outRegionC) );

	DebugAssert( arrMatricesA->IsAllocated() );
	DebugAssert( arrMatricesA->GetShape() == CUDA_MEMORY_SHAPE_3D );
	DebugAssert( arrMatricesA->GetStride() == sizeof(Float) );
	DebugAssert( hRegionA.iDepth == 1 );
	DebugAssert( arrMatricesA->IsValidRegion(hStartPositionA, hRegionA) );

	DebugAssert( arrMatricesB->IsAllocated() );
	DebugAssert( arrMatricesB->GetShape() == CUDA_MEMORY_SHAPE_3D );
	DebugAssert( arrMatricesB->GetStride() == sizeof(Float) );
	DebugAssert( hRegionB.iDepth == 1 );
	DebugAssert( arrMatricesB->IsValidRegion(hStartPositionB, hRegionB) );

	// Empty Call
	if ( iBatchCount == 0 )
		return;

	DebugAssert( iBatchCount <= outMatricesC->GetDepth() );
	DebugAssert( iBatchCount <= arrMatricesA->GetDepth() );
	DebugAssert( iBatchCount <= arrMatricesB->GetDepth() );

	SizeT iM, iN, iK;
	if ( iTransOpA == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionA.iWidth == outRegionC.iWidth );
		iM = hRegionA.iWidth;
		iK = hRegionA.iHeight;
	} else {
		DebugAssert( hRegionA.iHeight == outRegionC.iWidth );
		iM = hRegionA.iHeight;
		iK = hRegionA.iWidth;
	}
	if ( iTransOpB == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
		iN = hRegionB.iHeight;
		DebugAssert( iK == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionB.iWidth == outRegionC.iHeight );
		iN = hRegionB.iWidth;
		DebugAssert( iK == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );
	
	cublasStatus_t iError = cublasSgemmStridedBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
													   &fAlpha, (const Float *)( arrMatricesA->GetPointer(hStartPositionA) ), arrMatricesA->GetWidth(), iStrideA,
													   (const Float *)( arrMatricesB->GetPointer(hStartPositionB) ), arrMatricesB->GetWidth(), iStrideB,
													   &fBeta, (Float*)( outMatricesC->GetPointer(outStartPositionC) ), outMatricesC->GetWidth(), outStrideC, iBatchCount );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddStrideBatched<Double>( SizeT iBatchCount, CUDADeviceMemory * outMatricesC, const CUDAMemoryPosition & outStartPositionC, const CUDAMemoryRegion & outRegionC, SizeT outStrideC, Double fBeta,
												 const CUDADeviceMemory * arrMatricesA, const CUDAMemoryPosition & hStartPositionA, const CUDAMemoryRegion & hRegionA, SizeT iStrideA, Double fAlpha,
												 const CUDADeviceMemory * arrMatricesB, const CUDAMemoryPosition & hStartPositionB, const CUDAMemoryRegion & hRegionB, SizeT iStrideB,
												 CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( outStrideC >= outRegionC.iWidth * outRegionC.iHeight );
	DebugAssert( iStrideA >= hRegionA.iWidth * hRegionA.iHeight );
	DebugAssert( iStrideB >= hRegionB.iWidth * hRegionB.iHeight );

	DebugAssert( outMatricesC->IsAllocated() );
	DebugAssert( outMatricesC->GetShape() == CUDA_MEMORY_SHAPE_3D );
	DebugAssert( outMatricesC->GetStride() == sizeof(Double) );
	DebugAssert( outRegionC.iDepth == 1 );
	DebugAssert( outMatricesC->IsValidRegion(outStartPositionC, outRegionC) );

	DebugAssert( arrMatricesA->IsAllocated() );
	DebugAssert( arrMatricesA->GetShape() == CUDA_MEMORY_SHAPE_3D );
	DebugAssert( arrMatricesA->GetStride() == sizeof(Double) );
	DebugAssert( hRegionA.iDepth == 1 );
	DebugAssert( arrMatricesA->IsValidRegion(hStartPositionA, hRegionA) );

	DebugAssert( arrMatricesB->IsAllocated() );
	DebugAssert( arrMatricesB->GetShape() == CUDA_MEMORY_SHAPE_3D );
	DebugAssert( arrMatricesB->GetStride() == sizeof(Double) );
	DebugAssert( hRegionB.iDepth == 1 );
	DebugAssert( arrMatricesB->IsValidRegion(hStartPositionB, hRegionB) );

	// Empty Call
	if ( iBatchCount == 0 )
		return;

	DebugAssert( iBatchCount <= outMatricesC->GetDepth() );
	DebugAssert( iBatchCount <= arrMatricesA->GetDepth() );
	DebugAssert( iBatchCount <= arrMatricesB->GetDepth() );

	SizeT iM, iN, iK;
	if ( iTransOpA == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionA.iWidth == outRegionC.iWidth );
		iM = hRegionA.iWidth;
		iK = hRegionA.iHeight;
	} else {
		DebugAssert( hRegionA.iHeight == outRegionC.iWidth );
		iM = hRegionA.iHeight;
		iK = hRegionA.iWidth;
	}
	if ( iTransOpB == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
		iN = hRegionB.iHeight;
		DebugAssert( iK == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionB.iWidth == outRegionC.iHeight );
		iN = hRegionB.iWidth;
		DebugAssert( iK == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );
	
	cublasStatus_t iError = cublasDgemmStridedBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
													   &fAlpha, (const Double *)( arrMatricesA->GetPointer(hStartPositionA) ), arrMatricesA->GetWidth(), iStrideA,
													   (const Double *)( arrMatricesB->GetPointer(hStartPositionB) ), arrMatricesB->GetWidth(), iStrideB,
													   &fBeta, (Double*)( outMatricesC->GetPointer(outStartPositionC) ), outMatricesC->GetWidth(), outStrideC, iBatchCount );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddStrideBatched<cuComplex>( SizeT iBatchCount, CUDADeviceMemory * outMatricesC, const CUDAMemoryPosition & outStartPositionC, const CUDAMemoryRegion & outRegionC, SizeT outStrideC, cuComplex fBeta,
													const CUDADeviceMemory * arrMatricesA, const CUDAMemoryPosition & hStartPositionA, const CUDAMemoryRegion & hRegionA, SizeT iStrideA, cuComplex fAlpha,
													const CUDADeviceMemory * arrMatricesB, const CUDAMemoryPosition & hStartPositionB, const CUDAMemoryRegion & hRegionB, SizeT iStrideB,
													CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( outStrideC >= outRegionC.iWidth * outRegionC.iHeight );
	DebugAssert( iStrideA >= hRegionA.iWidth * hRegionA.iHeight );
	DebugAssert( iStrideB >= hRegionB.iWidth * hRegionB.iHeight );

	DebugAssert( outMatricesC->IsAllocated() );
	DebugAssert( outMatricesC->GetShape() == CUDA_MEMORY_SHAPE_3D );
	DebugAssert( outMatricesC->GetStride() == sizeof(cuComplex) );
	DebugAssert( outRegionC.iDepth == 1 );
	DebugAssert( outMatricesC->IsValidRegion(outStartPositionC, outRegionC) );

	DebugAssert( arrMatricesA->IsAllocated() );
	DebugAssert( arrMatricesA->GetShape() == CUDA_MEMORY_SHAPE_3D );
	DebugAssert( arrMatricesA->GetStride() == sizeof(cuComplex) );
	DebugAssert( hRegionA.iDepth == 1 );
	DebugAssert( arrMatricesA->IsValidRegion(hStartPositionA, hRegionA) );

	DebugAssert( arrMatricesB->IsAllocated() );
	DebugAssert( arrMatricesB->GetShape() == CUDA_MEMORY_SHAPE_3D );
	DebugAssert( arrMatricesB->GetStride() == sizeof(cuComplex) );
	DebugAssert( hRegionB.iDepth == 1 );
	DebugAssert( arrMatricesB->IsValidRegion(hStartPositionB, hRegionB) );

	// Empty Call
	if ( iBatchCount == 0 )
		return;

	DebugAssert( iBatchCount <= outMatricesC->GetDepth() );
	DebugAssert( iBatchCount <= arrMatricesA->GetDepth() );
	DebugAssert( iBatchCount <= arrMatricesB->GetDepth() );

	SizeT iM, iN, iK;
	if ( iTransOpA == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionA.iWidth == outRegionC.iWidth );
		iM = hRegionA.iWidth;
		iK = hRegionA.iHeight;
	} else {
		DebugAssert( hRegionA.iHeight == outRegionC.iWidth );
		iM = hRegionA.iHeight;
		iK = hRegionA.iWidth;
	}
	if ( iTransOpB == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
		iN = hRegionB.iHeight;
		DebugAssert( iK == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionB.iWidth == outRegionC.iHeight );
		iN = hRegionB.iWidth;
		DebugAssert( iK == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );
	
	cublasStatus_t iError = cublasCgemmStridedBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
													   &fAlpha, (const cuComplex *)( arrMatricesA->GetPointer(hStartPositionA) ), arrMatricesA->GetWidth(), iStrideA,
													   (const cuComplex *)( arrMatricesB->GetPointer(hStartPositionB) ), arrMatricesB->GetWidth(), iStrideB,
													   &fBeta, (cuComplex*)( outMatricesC->GetPointer(outStartPositionC) ), outMatricesC->GetWidth(), outStrideC, iBatchCount );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::MulAddStrideBatched<cuDoubleComplex>( SizeT iBatchCount, CUDADeviceMemory * outMatricesC, const CUDAMemoryPosition & outStartPositionC, const CUDAMemoryRegion & outRegionC, SizeT outStrideC, cuDoubleComplex fBeta,
														  const CUDADeviceMemory * arrMatricesA, const CUDAMemoryPosition & hStartPositionA, const CUDAMemoryRegion & hRegionA, SizeT iStrideA, cuDoubleComplex fAlpha,
														  const CUDADeviceMemory * arrMatricesB, const CUDAMemoryPosition & hStartPositionB, const CUDAMemoryRegion & hRegionB, SizeT iStrideB,
														  CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( outStrideC >= outRegionC.iWidth * outRegionC.iHeight );
	DebugAssert( iStrideA >= hRegionA.iWidth * hRegionA.iHeight );
	DebugAssert( iStrideB >= hRegionB.iWidth * hRegionB.iHeight );

	DebugAssert( outMatricesC->IsAllocated() );
	DebugAssert( outMatricesC->GetShape() == CUDA_MEMORY_SHAPE_3D );
	DebugAssert( outMatricesC->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( outRegionC.iDepth == 1 );
	DebugAssert( outMatricesC->IsValidRegion(outStartPositionC, outRegionC) );

	DebugAssert( arrMatricesA->IsAllocated() );
	DebugAssert( arrMatricesA->GetShape() == CUDA_MEMORY_SHAPE_3D );
	DebugAssert( arrMatricesA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( hRegionA.iDepth == 1 );
	DebugAssert( arrMatricesA->IsValidRegion(hStartPositionA, hRegionA) );

	DebugAssert( arrMatricesB->IsAllocated() );
	DebugAssert( arrMatricesB->GetShape() == CUDA_MEMORY_SHAPE_3D );
	DebugAssert( arrMatricesB->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( hRegionB.iDepth == 1 );
	DebugAssert( arrMatricesB->IsValidRegion(hStartPositionB, hRegionB) );

	// Empty Call
	if ( iBatchCount == 0 )
		return;

	DebugAssert( iBatchCount <= outMatricesC->GetDepth() );
	DebugAssert( iBatchCount <= arrMatricesA->GetDepth() );
	DebugAssert( iBatchCount <= arrMatricesB->GetDepth() );

	SizeT iM, iN, iK;
	if ( iTransOpA == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionA.iWidth == outRegionC.iWidth );
		iM = hRegionA.iWidth;
		iK = hRegionA.iHeight;
	} else {
		DebugAssert( hRegionA.iHeight == outRegionC.iWidth );
		iM = hRegionA.iHeight;
		iK = hRegionA.iWidth;
	}
	if ( iTransOpB == CUBLAS_CONTEXT_TRANSOP_NONE ) {
		DebugAssert( hRegionB.iHeight == outRegionC.iHeight );
		iN = hRegionB.iHeight;
		DebugAssert( iK == hRegionB.iWidth );
	} else {
		DebugAssert( hRegionB.iWidth == outRegionC.iHeight );
		iN = hRegionB.iWidth;
		DebugAssert( iK == hRegionB.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	cublasOperation_t iCUBLASTransposeOpB = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpB] );
	
	cublasStatus_t iError = cublasZgemmStridedBatched( hCUBLASContext, iCUBLASTransposeOpA, iCUBLASTransposeOpB, iM, iN, iK,
													   &fAlpha, (const cuDoubleComplex *)( arrMatricesA->GetPointer(hStartPositionA) ), arrMatricesA->GetWidth(), iStrideA,
													   (const cuDoubleComplex *)( arrMatricesB->GetPointer(hStartPositionB) ), arrMatricesB->GetWidth(), iStrideB,
													   &fBeta, (cuDoubleComplex*)( outMatricesC->GetPointer(outStartPositionC) ), outMatricesC->GetWidth(), outStrideC, iBatchCount );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::SolveTriangular<Float>( CUDADeviceMemory * outMatrixX, const CUDAMemoryPosition & outPositionX, const CUDAMemoryRegion & outRegionX,
											const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, Float fAlpha,
											CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixX->IsAllocated() );
	DebugAssert( outMatrixX->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixX->GetStride() == sizeof(Float) );
	DebugAssert( outMatrixX->IsValidRegion(outPositionX, outRegionX) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(Float) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == outRegionX.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == outRegionX.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	
	cublasStatus_t iError = cublasStrsm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 outRegionX.iWidth, outRegionX.iHeight,
										 &fAlpha, (const Float *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (Float*)( outMatrixX->GetPointer(outPositionX) ), outMatrixX->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::SolveTriangular<Double>( CUDADeviceMemory * outMatrixX, const CUDAMemoryPosition & outPositionX, const CUDAMemoryRegion & outRegionX,
											 const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, Double fAlpha,
											 CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixX->IsAllocated() );
	DebugAssert( outMatrixX->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixX->GetStride() == sizeof(Double) );
	DebugAssert( outMatrixX->IsValidRegion(outPositionX, outRegionX) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(Double) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == outRegionX.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == outRegionX.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	
	cublasStatus_t iError = cublasDtrsm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 outRegionX.iWidth, outRegionX.iHeight,
										 &fAlpha, (const Double *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (Double*)( outMatrixX->GetPointer(outPositionX) ), outMatrixX->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::SolveTriangular<cuComplex>( CUDADeviceMemory * outMatrixX, const CUDAMemoryPosition & outPositionX, const CUDAMemoryRegion & outRegionX,
												const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, cuComplex fAlpha,
												CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixX->IsAllocated() );
	DebugAssert( outMatrixX->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixX->GetStride() == sizeof(cuComplex) );
	DebugAssert( outMatrixX->IsValidRegion(outPositionX, outRegionX) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(cuComplex) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == outRegionX.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == outRegionX.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	
	cublasStatus_t iError = cublasCtrsm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 outRegionX.iWidth, outRegionX.iHeight,
										 &fAlpha, (const cuComplex *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (cuComplex*)( outMatrixX->GetPointer(outPositionX) ), outMatrixX->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::SolveTriangular<cuDoubleComplex>( CUDADeviceMemory * outMatrixX, const CUDAMemoryPosition & outPositionX, const CUDAMemoryRegion & outRegionX,
													  const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, cuDoubleComplex fAlpha,
													  CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA ) const
{
	DebugAssert( m_hContext != NULL );

	DebugAssert( outMatrixX->IsAllocated() );
	DebugAssert( outMatrixX->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( outMatrixX->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( outMatrixX->IsValidRegion(outPositionX, outRegionX) );

	DebugAssert( pMatrixA->IsAllocated() );
	DebugAssert( pMatrixA->GetShape() == CUDA_MEMORY_SHAPE_2D );
	DebugAssert( pMatrixA->GetStride() == sizeof(cuDoubleComplex) );
	DebugAssert( pMatrixA->IsValidRegion(hPositionA, hRegionA) );

	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == outRegionX.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == outRegionX.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	
	cublasStatus_t iError = cublasZtrsm( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
										 outRegionX.iWidth, outRegionX.iHeight,
										 &fAlpha, (const cuDoubleComplex *)( pMatrixA->GetPointer(hPositionA) ), pMatrixA->GetWidth(),
										 (cuDoubleComplex*)( outMatrixX->GetPointer(outPositionX) ), outMatrixX->GetWidth() );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}

template<>
Void CUBLASContext::SolveTriangularBatched<Float>( SizeT iBatchCount, CUDADeviceMemory * outMatricesX, const CUDAMemoryPosition * outPositionsX, const CUDAMemoryRegion & outRegionX,
												   const CUDADeviceMemory * arrMatricesA, const CUDAMemoryPosition * arrPositionsA, const CUDAMemoryRegion & hRegionA, Float fAlpha,
												   CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( iBatchCount <= CUBLAS_BATCH_MAX_COUNT );

	// Empty Call
	if ( iBatchCount == 0 )
		return;

	// Prepare Batch Data
	Float * arrBatchMatricesX[CUBLAS_BATCH_MAX_COUNT];
	const Float * arrBatchMatricesA[CUBLAS_BATCH_MAX_COUNT];

	SizeT iReferenceWidthX = outMatricesX[0].GetWidth();
	SizeT iReferenceWidthA = arrMatricesA[0].GetWidth();

	for( UInt i = 0; i < iBatchCount; ++i ) {
		DebugAssert( outMatricesX[i].IsAllocated() );
		DebugAssert( outMatricesX[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( outMatricesX[i].GetStride() == sizeof(Float) );
		DebugAssert( outMatricesX[i].GetWidth() == iReferenceWidthX );
		DebugAssert( outMatricesX[i].IsValidRegion(outPositionsX[i], outRegionX) );

		DebugAssert( arrMatricesA[i].IsAllocated() );
		DebugAssert( arrMatricesA[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( arrMatricesA[i].GetStride() == sizeof(Float) );
		DebugAssert( arrMatricesA[i].GetWidth() == iReferenceWidthA );
		DebugAssert( arrMatricesA[i].IsValidRegion(arrPositionsA[i], hRegionA) );

		arrBatchMatricesX[i] = (Float*)( outMatricesX[i].GetPointer(outPositionsX[i]) );
		arrBatchMatricesA[i] = (const Float *)( arrMatricesA[i].GetPointer(arrPositionsA[i]) );
	}

	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == outRegionX.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == outRegionX.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	
	cublasStatus_t iError = cublasStrsmBatched( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
												outRegionX.iWidth, outRegionX.iHeight,
												&fAlpha, arrBatchMatricesA, iReferenceWidthA,
												arrBatchMatricesX, iReferenceWidthX, iBatchCount );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::SolveTriangularBatched<Double>( SizeT iBatchCount, CUDADeviceMemory * outMatricesX, const CUDAMemoryPosition * outPositionsX, const CUDAMemoryRegion & outRegionX,
													const CUDADeviceMemory * arrMatricesA, const CUDAMemoryPosition * arrPositionsA, const CUDAMemoryRegion & hRegionA, Double fAlpha,
													CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( iBatchCount <= CUBLAS_BATCH_MAX_COUNT );

	// Empty Call
	if ( iBatchCount == 0 )
		return;

	// Prepare Batch Data
	Double * arrBatchMatricesX[CUBLAS_BATCH_MAX_COUNT];
	const Double * arrBatchMatricesA[CUBLAS_BATCH_MAX_COUNT];

	SizeT iReferenceWidthX = outMatricesX[0].GetWidth();
	SizeT iReferenceWidthA = arrMatricesA[0].GetWidth();

	for( UInt i = 0; i < iBatchCount; ++i ) {
		DebugAssert( outMatricesX[i].IsAllocated() );
		DebugAssert( outMatricesX[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( outMatricesX[i].GetStride() == sizeof(Double) );
		DebugAssert( outMatricesX[i].GetWidth() == iReferenceWidthX );
		DebugAssert( outMatricesX[i].IsValidRegion(outPositionsX[i], outRegionX) );

		DebugAssert( arrMatricesA[i].IsAllocated() );
		DebugAssert( arrMatricesA[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( arrMatricesA[i].GetStride() == sizeof(Double) );
		DebugAssert( arrMatricesA[i].GetWidth() == iReferenceWidthA );
		DebugAssert( arrMatricesA[i].IsValidRegion(arrPositionsA[i], hRegionA) );

		arrBatchMatricesX[i] = (Double*)( outMatricesX[i].GetPointer(outPositionsX[i]) );
		arrBatchMatricesA[i] = (const Double *)( arrMatricesA[i].GetPointer(arrPositionsA[i]) );
	}

	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == outRegionX.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == outRegionX.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	
	cublasStatus_t iError = cublasDtrsmBatched( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
												outRegionX.iWidth, outRegionX.iHeight,
												&fAlpha, arrBatchMatricesA, iReferenceWidthA,
												arrBatchMatricesX, iReferenceWidthX, iBatchCount );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::SolveTriangularBatched<cuComplex>( SizeT iBatchCount, CUDADeviceMemory * outMatricesX, const CUDAMemoryPosition * outPositionsX, const CUDAMemoryRegion & outRegionX,
													   const CUDADeviceMemory * arrMatricesA, const CUDAMemoryPosition * arrPositionsA, const CUDAMemoryRegion & hRegionA, cuComplex fAlpha,
													   CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( iBatchCount <= CUBLAS_BATCH_MAX_COUNT );

	// Empty Call
	if ( iBatchCount == 0 )
		return;

	// Prepare Batch Data
	cuComplex * arrBatchMatricesX[CUBLAS_BATCH_MAX_COUNT];
	const cuComplex * arrBatchMatricesA[CUBLAS_BATCH_MAX_COUNT];

	SizeT iReferenceWidthX = outMatricesX[0].GetWidth();
	SizeT iReferenceWidthA = arrMatricesA[0].GetWidth();

	for( UInt i = 0; i < iBatchCount; ++i ) {
		DebugAssert( outMatricesX[i].IsAllocated() );
		DebugAssert( outMatricesX[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( outMatricesX[i].GetStride() == sizeof(cuComplex) );
		DebugAssert( outMatricesX[i].GetWidth() == iReferenceWidthX );
		DebugAssert( outMatricesX[i].IsValidRegion(outPositionsX[i], outRegionX) );

		DebugAssert( arrMatricesA[i].IsAllocated() );
		DebugAssert( arrMatricesA[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( arrMatricesA[i].GetStride() == sizeof(cuComplex) );
		DebugAssert( arrMatricesA[i].GetWidth() == iReferenceWidthA );
		DebugAssert( arrMatricesA[i].IsValidRegion(arrPositionsA[i], hRegionA) );

		arrBatchMatricesX[i] = (cuComplex*)( outMatricesX[i].GetPointer(outPositionsX[i]) );
		arrBatchMatricesA[i] = (const cuComplex *)( arrMatricesA[i].GetPointer(arrPositionsA[i]) );
	}

	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == outRegionX.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == outRegionX.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	
	cublasStatus_t iError = cublasCtrsmBatched( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
												outRegionX.iWidth, outRegionX.iHeight,
												&fAlpha, arrBatchMatricesA, iReferenceWidthA,
												arrBatchMatricesX, iReferenceWidthX, iBatchCount );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
template<>
Void CUBLASContext::SolveTriangularBatched<cuDoubleComplex>( SizeT iBatchCount, CUDADeviceMemory * outMatricesX, const CUDAMemoryPosition * outPositionsX, const CUDAMemoryRegion & outRegionX,
															 const CUDADeviceMemory * arrMatricesA, const CUDAMemoryPosition * arrPositionsA, const CUDAMemoryRegion & hRegionA, cuDoubleComplex fAlpha,
															 CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA ) const
{
	DebugAssert( m_hContext != NULL );
	DebugAssert( iBatchCount <= CUBLAS_BATCH_MAX_COUNT );

	// Empty Call
	if ( iBatchCount == 0 )
		return;

	// Prepare Batch Data
	cuDoubleComplex * arrBatchMatricesX[CUBLAS_BATCH_MAX_COUNT];
	const cuDoubleComplex * arrBatchMatricesA[CUBLAS_BATCH_MAX_COUNT];

	SizeT iReferenceWidthX = outMatricesX[0].GetWidth();
	SizeT iReferenceWidthA = arrMatricesA[0].GetWidth();

	for( UInt i = 0; i < iBatchCount; ++i ) {
		DebugAssert( outMatricesX[i].IsAllocated() );
		DebugAssert( outMatricesX[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( outMatricesX[i].GetStride() == sizeof(cuDoubleComplex) );
		DebugAssert( outMatricesX[i].GetWidth() == iReferenceWidthX );
		DebugAssert( outMatricesX[i].IsValidRegion(outPositionsX[i], outRegionX) );

		DebugAssert( arrMatricesA[i].IsAllocated() );
		DebugAssert( arrMatricesA[i].GetShape() == CUDA_MEMORY_SHAPE_2D );
		DebugAssert( arrMatricesA[i].GetStride() == sizeof(cuDoubleComplex) );
		DebugAssert( arrMatricesA[i].GetWidth() == iReferenceWidthA );
		DebugAssert( arrMatricesA[i].IsValidRegion(arrPositionsA[i], hRegionA) );

		arrBatchMatricesX[i] = (cuDoubleComplex*)( outMatricesX[i].GetPointer(outPositionsX[i]) );
		arrBatchMatricesA[i] = (const cuDoubleComplex *)( arrMatricesA[i].GetPointer(arrPositionsA[i]) );
	}

	if ( iSideMode == CUBLAS_CONTEXT_SIDEMODE_LEFT ) {
		DebugAssert( hRegionA.iHeight == outRegionX.iWidth );
	} else {
		DebugAssert( hRegionA.iWidth == outRegionX.iHeight );
	}

	cublasHandle_t hCUBLASContext = (cublasHandle_t)m_hContext;
	cublasSideMode_t iCUBLASSideMode = (cublasSideMode_t)( CUBLASContextSideModeToCUDA[iSideMode] );
	cublasFillMode_t iCUBLASFillMode = (cublasFillMode_t)( CUBLASContextFillModeToCUDA[iFillMode] );
	cublasOperation_t iCUBLASTransposeOpA = (cublasOperation_t)( CUBLASContextTransposeOpToCUDA[iTransOpA] );
	
	cublasStatus_t iError = cublasZtrsmBatched( hCUBLASContext, iCUBLASSideMode, iCUBLASFillMode, iCUBLASTransposeOpA, bMainDiagIsUnityA ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
												outRegionX.iWidth, outRegionX.iHeight,
												&fAlpha, arrBatchMatricesA, iReferenceWidthA,
												arrBatchMatricesX, iReferenceWidthX, iBatchCount );
	DebugAssert( iError == CUBLAS_STATUS_SUCCESS );
}
