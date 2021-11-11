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
#include <typeinfo>
#include <cuda_runtime.h>
#include <cusolverDn.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#pragma warning(disable:4739) // Reference exceed storage space
#include "CUSolverDenseEigenValue.h"

/////////////////////////////////////////////////////////////////////////////////
// CUSolverDenseEigenValue implementation
CUSolverDenseEigenValue::CUSolverDenseEigenValue( CUSolverDenseContext * pCUSolverDenseContext ):
	m_hMatrixPositionA(), m_hMatrixRegionA(),
	m_hVectorPositionX(),
	m_hWorkspace(), m_hDeviceInfo(), m_hDeviceInfoSaved()
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

	m_iWorkspaceSize = 0;
	m_hJacobiInfos = NULL;

	m_iSolverState = CUSOLVER_DENSE_EIGENVALUE_STATE_RESET;
	m_hDeviceInfo.Allocate( sizeof(int) );
	m_hDeviceInfoSaved.Allocate( sizeof(int) );
}
CUSolverDenseEigenValue::~CUSolverDenseEigenValue()
{
	if ( m_hWorkspace.IsAllocated() ) {
		m_hWorkspace.Free();
		m_iWorkspaceSize = 0;
	}

	if ( m_hJacobiInfos != NULL ) {
		cusolverStatus_t iError = cusolverDnDestroySyevjInfo( (syevjInfo_t)m_hJacobiInfos );
		DebugAssert( iError == CUSOLVER_STATUS_SUCCESS );
		m_hJacobiInfos = NULL;
	}

	m_hDeviceInfo.Free();
	m_hDeviceInfoSaved.Free();
}

Void CUSolverDenseEigenValue::UpdateStateAfterSync()
{
	DebugAssert( m_pCUSolverDenseContext != NULL );
	DebugAssert( m_iSolverState == CUSOLVER_DENSE_EIGENVALUE_STATE_RUNNING );

	m_hDeviceInfoSaved.Copy( &m_hDeviceInfo, sizeof(int) );
	Int * pDeviceInfos = (Int*)( m_hDeviceInfoSaved.GetPointer() );

	DebugAssert( *pDeviceInfos >= 0 );
	m_iSolverState = (*pDeviceInfos == 0) ? CUSOLVER_DENSE_EIGENVALUE_STATE_SUCCESS : CUSOLVER_DENSE_EIGENVALUE_STATE_FAILED;
}
Void CUSolverDenseEigenValue::Reset()
{
	DebugAssert( m_pCUSolverDenseContext != NULL );
	DebugAssert( m_iSolverState != CUSOLVER_DENSE_EIGENVALUE_STATE_RESET );

	if ( m_hWorkspace.IsAllocated() ) {
		m_hWorkspace.Free();
		m_iWorkspaceSize = 0;
	}

	if ( m_hJacobiInfos != NULL ) {
		cusolverStatus_t iError = cusolverDnDestroySyevjInfo( (syevjInfo_t)m_hJacobiInfos );
		DebugAssert( iError == CUSOLVER_STATUS_SUCCESS );
		m_hJacobiInfos = NULL;
	}

	m_iSolverState = CUSOLVER_DENSE_EIGENVALUE_STATE_RESET;
}



