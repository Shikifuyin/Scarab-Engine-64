/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUSolver/CUSolverDenseContext.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA Solver Context for Dense systems
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
#include <cusolverDn.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUSolverDenseContext.h"

/////////////////////////////////////////////////////////////////////////////////
// CUSolverDenseContext implementation
CUSolverDenseContext::CUSolverDenseContext()
{
	m_hContext = NULL;
}
CUSolverDenseContext::~CUSolverDenseContext()
{
	if ( IsCreated() )
		Destroy();
}

Void CUSolverDenseContext::Create()
{
	DebugAssert( m_hContext == NULL );
	
	cusolverDnHandle_t hCUSolverDnContext = NULL;
	
	cusolverStatus_t iError = cusolverDnCreate( &hCUSolverDnContext );
	DebugAssert( iError == CUSOLVER_STATUS_SUCCESS && hCUSolverDnContext != NULL );
	
	m_hContext = hCUSolverDnContext;
}
Void CUSolverDenseContext::Destroy()
{
	DebugAssert( m_hContext != NULL );
	
	cusolverDnHandle_t hCUSolverDnContext = (cusolverDnHandle_t)m_hContext;
	
	cusolverStatus_t iError = cusolverDnDestroy( hCUSolverDnContext );
	DebugAssert( iError == CUSOLVER_STATUS_SUCCESS );
	
	m_hContext = NULL;
}

Void CUSolverDenseContext::SetStream( CUDAStream * pStream ) const
{
	DebugAssert( m_hContext != NULL );

	cusolverDnHandle_t hCUSolverDnContext = (cusolverDnHandle_t)m_hContext;
	cudaStream_t hCUDAStream = NULL;
	if ( pStream != NULL ) {
		DebugAssert( pStream->IsCreated() );
		hCUDAStream = (cudaStream_t)( pStream->m_hStream );
	}
	
	cusolverStatus_t iError = cusolverDnSetStream( hCUSolverDnContext, hCUDAStream );
	DebugAssert( iError == CUSOLVER_STATUS_SUCCESS );

	syevjInfo_t a;
}

