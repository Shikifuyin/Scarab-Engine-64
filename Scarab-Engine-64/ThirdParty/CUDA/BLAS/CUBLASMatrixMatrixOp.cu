/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/BLAS/CUBLASMatrixMatrixOp.cpp
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
// Third-Party Includes
#include <typeinfo>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#pragma warning(disable:4005) // Macro redefinition
#pragma warning(disable:4739) // Reference exceed storage space
#include "CUBLASMatrixMatrixOp.h"

/////////////////////////////////////////////////////////////////////////////////
// Specialized Kernels
struct _Kernel_Matrix {
	size_t iRowCount;
	size_t iColCount;
	Void * arrElements;
};

__global__ void _Kernel_ComponentMul_RF( _Kernel_Matrix matOut, _Kernel_Matrix matIn ) {
	int iRow = blockIdx.x * blockDim.x + threadIdx.x;
	int iCol = blockIdx.y * blockDim.y + threadIdx.y;
	if ( iRow >= matOut.iRowCount || iCol >= matOut.iColCount )
		return;
	float * arrOut = (float*)( matOut.arrElements );
	float * arrIn = (float*)( matIn.arrElements );
	float * pOut = ( arrOut + matOut.iRowCount * iCol + iRow );
	float * pIn = ( arrIn + matIn.iRowCount * iCol + iRow );
	*pOut = (*pOut) * (*pIn);
}
__global__ void _Kernel_ComponentMul_RD( _Kernel_Matrix matOut, _Kernel_Matrix matIn ) {
	int iRow = blockIdx.x * blockDim.x + threadIdx.x;
	int iCol = blockIdx.y * blockDim.y + threadIdx.y;
	if ( iRow >= matOut.iRowCount || iCol >= matOut.iColCount )
		return;
	double * arrOut = (double*)( matOut.arrElements );
	double * arrIn = (double*)( matIn.arrElements );
	double * pOut = ( arrOut + matOut.iRowCount * iCol + iRow );
	double * pIn = ( arrIn + matIn.iRowCount * iCol + iRow );
	*pOut = (*pOut) * (*pIn);
}
__global__ void _Kernel_ComponentMul_CF( _Kernel_Matrix matOut, _Kernel_Matrix matIn ) {
	int iRow = blockIdx.x * blockDim.x + threadIdx.x;
	int iCol = blockIdx.y * blockDim.y + threadIdx.y;
	if ( iRow >= matOut.iRowCount || iCol >= matOut.iColCount )
		return;
	cuComplex * arrOut = (cuComplex*)( matOut.arrElements );
	cuComplex * arrIn = (cuComplex*)( matIn.arrElements );
	cuComplex * pOut = ( arrOut + matOut.iRowCount * iCol + iRow );
	cuComplex * pIn = ( arrIn + matIn.iRowCount * iCol + iRow );
	*pOut = cuCmulf( *pOut, *pIn );
}
__global__ void _Kernel_ComponentMul_CD( _Kernel_Matrix matOut, _Kernel_Matrix matIn ) {
	int iRow = blockIdx.x * blockDim.x + threadIdx.x;
	int iCol = blockIdx.y * blockDim.y + threadIdx.y;
	if ( iRow >= matOut.iRowCount || iCol >= matOut.iColCount )
		return;
	cuDoubleComplex * arrOut = (cuDoubleComplex*)( matOut.arrElements );
	cuDoubleComplex * arrIn = (cuDoubleComplex*)( matIn.arrElements );
	cuDoubleComplex * pOut = ( arrOut + matOut.iRowCount * iCol + iRow );
	cuDoubleComplex * pIn = ( arrIn + matIn.iRowCount * iCol + iRow );
	*pOut = cuCmul( *pOut, *pIn );
}

/////////////////////////////////////////////////////////////////////////////////
// CUBLASMatrixMatrixOp implementation
CUBLASMatrixMatrixOp::CUBLASMatrixMatrixOp( CUBLASContext * pCUBLASContext ):
	m_hMatrixPositionA(), m_hMatrixRegionA(),
	m_hMatrixPositionB(), m_hMatrixRegionB(),
	m_hMatrixPositionC(), m_hMatrixRegionC()
{
	DebugAssert( pCUBLASContext != NULL && pCUBLASContext->IsCreated() );

	m_pCUBLASContext = pCUBLASContext;

	m_pMatrixA = NULL;
	m_pMatrixB = NULL;
	m_pMatrixC = NULL;
}
CUBLASMatrixMatrixOp::~CUBLASMatrixMatrixOp()
{
	// nothing to do
}

Void CUBLASMatrixMatrixOp::_MulComponentRF_InvokeKernel()
{
	dim3 dimBlock(16, 16);
    dim3 dimGrid( (UInt)(m_hMatrixRegionA.iWidth) / dimBlock.x, (UInt)(m_hMatrixRegionA.iHeight) / dimBlock.y );

	_Kernel_Matrix matA;
	matA.iRowCount = m_hMatrixRegionA.iWidth;
	matA.iColCount = m_hMatrixRegionA.iHeight;
	matA.arrElements = (Void*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) );

	_Kernel_Matrix matC;
	matC.iRowCount = m_hMatrixRegionC.iWidth;
	matC.iColCount = m_hMatrixRegionC.iHeight;
	matC.arrElements = (Void*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) );

    _Kernel_ComponentMul_RF<<<dimGrid, dimBlock>>>( matC, matA );
}
Void CUBLASMatrixMatrixOp::_MulComponentRD_InvokeKernel()
{
	dim3 dimBlock(16, 16);
    dim3 dimGrid( (UInt)(m_hMatrixRegionA.iWidth) / dimBlock.x, (UInt)(m_hMatrixRegionA.iHeight) / dimBlock.y );

	_Kernel_Matrix matA;
	matA.iRowCount = m_hMatrixRegionA.iWidth;
	matA.iColCount = m_hMatrixRegionA.iHeight;
	matA.arrElements = (Void*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) );

	_Kernel_Matrix matC;
	matC.iRowCount = m_hMatrixRegionC.iWidth;
	matC.iColCount = m_hMatrixRegionC.iHeight;
	matC.arrElements = (Void*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) );

    _Kernel_ComponentMul_RD<<<dimGrid, dimBlock>>>( matC, matA );
}
Void CUBLASMatrixMatrixOp::_MulComponentCF_InvokeKernel()
{
	dim3 dimBlock(16, 16);
    dim3 dimGrid( (UInt)(m_hMatrixRegionA.iWidth) / dimBlock.x, (UInt)(m_hMatrixRegionA.iHeight) / dimBlock.y );

	_Kernel_Matrix matA;
	matA.iRowCount = m_hMatrixRegionA.iWidth;
	matA.iColCount = m_hMatrixRegionA.iHeight;
	matA.arrElements = (Void*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) );

	_Kernel_Matrix matC;
	matC.iRowCount = m_hMatrixRegionC.iWidth;
	matC.iColCount = m_hMatrixRegionC.iHeight;
	matC.arrElements = (Void*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) );

    _Kernel_ComponentMul_CF<<<dimGrid, dimBlock>>>( matC, matA );
}
Void CUBLASMatrixMatrixOp::_MulComponentCD_InvokeKernel()
{
	dim3 dimBlock(16, 16);
    dim3 dimGrid( (UInt)(m_hMatrixRegionA.iWidth) / dimBlock.x, (UInt)(m_hMatrixRegionA.iHeight) / dimBlock.y );

	_Kernel_Matrix matA;
	matA.iRowCount = m_hMatrixRegionA.iWidth;
	matA.iColCount = m_hMatrixRegionA.iHeight;
	matA.arrElements = (Void*)( m_pMatrixA->GetPointer(m_hMatrixPositionA) );

	_Kernel_Matrix matC;
	matC.iRowCount = m_hMatrixRegionC.iWidth;
	matC.iColCount = m_hMatrixRegionC.iHeight;
	matC.arrElements = (Void*)( m_pMatrixC->GetPointer(m_hMatrixPositionC) );

    _Kernel_ComponentMul_CD<<<dimGrid, dimBlock>>>( matC, matA );
}


