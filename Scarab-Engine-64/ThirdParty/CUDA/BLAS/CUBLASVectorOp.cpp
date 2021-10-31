/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/BLAS/CUBLASVectorOp.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA BLAS : Vector Operations
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
#include "CUBLASVectorOp.h"

/////////////////////////////////////////////////////////////////////////////////
// CUBLASVectorOp implementation
CUBLASVectorOp::CUBLASVectorOp( CUBLASContext * pCUBLASContext ):
	m_hVectorPosition(), m_hVectorRegion()
{
	DebugAssert( pCUBLASContext != NULL && pCUBLASContext->IsCreated() );

	m_pCUBLASContext = pCUBLASContext;

	m_pVector = NULL;
}
CUBLASVectorOp::~CUBLASVectorOp()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// CUBLASVectorVectorOp implementation
CUBLASVectorVectorOp::CUBLASVectorVectorOp( CUBLASContext * pCUBLASContext ):
	m_hVectorPositionX(), m_hVectorPositionY(), m_hVectorRegion()
{
	DebugAssert( pCUBLASContext != NULL && pCUBLASContext->IsCreated() );

	m_pCUBLASContext = pCUBLASContext;

	m_pVectorX = NULL;
	m_pVectorY = NULL;
}
CUBLASVectorVectorOp::~CUBLASVectorVectorOp()
{
	// nothing to do
}
