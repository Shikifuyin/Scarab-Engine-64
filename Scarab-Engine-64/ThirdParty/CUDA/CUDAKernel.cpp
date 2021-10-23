/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDAKernel.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA Kernels
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
#include <cuda_runtime.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUDAKernel.h"

/////////////////////////////////////////////////////////////////////////////////
// CUDAKernel implementation
CUDAKernel::CUDAKernel( Void * pKernelFunction, UInt iParameterCount )
{
	DebugAssert( pKernelFunction != NULL );
	DebugAssert( iParameterCount < CUDAKERNEL_MAX_PARAMETERS );

	m_iSharedMemorySize = 0;
	
	m_pStream = NULL;

	m_pKernelFunction = pKernelFunction;
	m_iParameterCount = iParameterCount;
	for( UInt i = 0; i < CUDAKERNEL_MAX_PARAMETERS; ++i )
		m_arrKernelParameters[i] = NULL;
}
CUDAKernel::~CUDAKernel()
{
	// nothing to do
}

Void CUDAKernel::Launch( const CUDAKernelDimension & hGridSize, const CUDAKernelDimension & hBlockSize, Bool bCooperative )
{
	dim3 hCUDAGridDim( hGridSize.iX, hGridSize.iY, hGridSize.iZ );
	dim3 hCUDABlockDim( hBlockSize.iX, hBlockSize.iY, hBlockSize.iZ );
	
	cudaStream_t hCUDAStream = NULL;
	if ( m_pStream != NULL ) {
		DebugAssert( m_pStream->IsCreated() );
		hCUDAStream = (cudaStream_t)( m_pStream->m_hStream );
	}
	
	if ( bCooperative ) {
		cudaError_t iError = cudaLaunchCooperativeKernel( m_pKernelFunction, hCUDAGridDim, hCUDABlockDim, m_arrKernelParameters, m_iSharedMemorySize, hCUDAStream );
		DebugAssert( iError == cudaSuccess );
	} else {
		cudaError_t iError = cudaLaunchKernel( m_pKernelFunction, hCUDAGridDim, hCUDABlockDim, m_arrKernelParameters, m_iSharedMemorySize, hCUDAStream );
		DebugAssert( iError == cudaSuccess );
	}
}


