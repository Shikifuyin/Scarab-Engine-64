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
#include <cublas_v2.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#pragma warning(disable:4739) // Reference exceed storage space
#include "CUBLASMatrixMatrixOp.h"

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
