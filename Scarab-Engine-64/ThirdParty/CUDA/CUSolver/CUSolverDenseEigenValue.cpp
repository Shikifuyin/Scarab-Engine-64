/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUSolver/CUSolverDenseEigenValue.cpp
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
// Third-Party Includes
#include <cuda_runtime.h>
#include <cusolverDn.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#pragma warning(disable:4244) // Conversion to smaller type
#pragma warning(disable:4739) // Reference exceed storage space
#include "CUSolverDenseEigenValue.h"

/////////////////////////////////////////////////////////////////////////////////
// CUSolverDenseEigenValue implementation
CUSolverDenseEigenValue::CUSolverDenseEigenValue( CUSolverDenseContext * pCUSolverDenseContext ):
	m_hMatrixPositionA(), m_hMatrixRegionA(),
	m_hVectorPositionX(), m_hVectorRegionX(),
	m_hWorkspace()
{
	DebugAssert( pCUSolverDenseContext != NULL && pCUSolverDenseContext->IsCreated() );

	m_pCUSolverDenseContext = pCUSolverDenseContext;

	m_pMatrixA = NULL;
	m_iFillModeA = CUBLAS_CONTEXT_FILLMODE_UPPER;

	m_pVectorX = NULL;

	m_bComputeEigenVectors = false;
	m_iAlgorithm = CUSOLVER_DENSE_EIGENVALUE_ALGORITHM_QR;
	m_fJacobiTolerance = 1.0e-8;
	m_iJacobiMaxSweeps = 100;

	m_hJacobiInfos = NULL;

	m_iSolverState = CUSOLVER_DENSE_EIGENVALUE_STATE_RESET;
	m_iSolverResult = 0;
}
CUSolverDenseEigenValue::~CUSolverDenseEigenValue()
{
	if ( m_hWorkspace.IsAllocated() )
		m_hWorkspace.Free();

	if ( m_hJacobiInfos != NULL ) {
		cusolverStatus_t iError = cusolverDnDestroySyevjInfo( (syevjInfo_t)m_hJacobiInfos );
		DebugAssert( iError == CUSOLVER_STATUS_SUCCESS );
		m_hJacobiInfos = NULL;
	}
}

Void CUSolverDenseEigenValue::Reset()
{
	DebugAssert( m_pCUSolverDenseContext != NULL );
	DebugAssert( m_iSolverState != CUSOLVER_DENSE_EIGENVALUE_STATE_RESET );

	if ( m_hWorkspace.IsAllocated() )
		m_hWorkspace.Free();

	if ( m_hJacobiInfos != NULL ) {
		cusolverStatus_t iError = cusolverDnDestroySyevjInfo( (syevjInfo_t)m_hJacobiInfos );
		DebugAssert( iError == CUSOLVER_STATUS_SUCCESS );
		m_hJacobiInfos = NULL;
	}

	m_iSolverState = CUSOLVER_DENSE_EIGENVALUE_STATE_RESET;
	m_iSolverResult = 0;
}



