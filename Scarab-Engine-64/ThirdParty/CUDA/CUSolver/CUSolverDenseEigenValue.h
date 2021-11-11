/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUSolver/CUSolverDenseEigenValue.h
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
// Header prelude
#ifndef SCARAB_THIRDPARTY_CUDA_CUSOLVER_CUSOLVERDENSEEIGENVALUE_H
#define SCARAB_THIRDPARTY_CUDA_CUSOLVER_CUSOLVERDENSEEIGENVALUE_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUSolverDenseContext.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// CUSolver, Dense, Eigen Value Decomposition Algorithm
enum CUSolverDenseEigenValueAlgorithm {
	// Better for large matrices (default)
	CUSOLVER_DENSE_EIGENVALUE_ALGORITHM_QR = 0,
	// Better for small/medium matrices, has precision control
	CUSOLVER_DENSE_EIGENVALUE_ALGORITHM_JACOBI
};

// CUSolver, Dense, Eigen Value Decomposition State
enum CUSolverDenseEigenValueState {
	CUSOLVER_DENSE_EIGENVALUE_STATE_RESET = 0,
	CUSOLVER_DENSE_EIGENVALUE_STATE_READY,
	CUSOLVER_DENSE_EIGENVALUE_STATE_RUNNING,
	CUSOLVER_DENSE_EIGENVALUE_STATE_SUCCESS,
	CUSOLVER_DENSE_EIGENVALUE_STATE_FAILED
};

/////////////////////////////////////////////////////////////////////////////////
// The CUSolverDenseEigenValue class
class CUSolverDenseEigenValue
{
public:
	CUSolverDenseEigenValue( CUSolverDenseContext * pCUSolverDenseContext );
    ~CUSolverDenseEigenValue();

	// Input-Output : Matrix A (Symmetric or Hermitian)
	inline Void SetMatrixA( CUDADeviceMemory * pMatrix, const CUDAMemoryPosition * pPosition = NULL, const CUDAMemoryRegion * pRegion = NULL );
	inline Void SetMatrixPositionA( const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetMatrixRegionA( const CUDAMemoryRegion * pRegion = NULL );

	inline CUDADeviceMemory * GetMatrixA( CUDAMemoryPosition * outPosition = NULL, CUDAMemoryRegion * outRegion = NULL ) const;

	inline Void SetMatrixFillModeA( CUBLASContextFillMode iFillMode );

	// Output : Vector X
	inline Void SetVectorX( CUDADeviceMemory * pVector, const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetVectorPositionX( const CUDAMemoryPosition * pPosition = NULL );

	inline CUDADeviceMemory * GetVectorX( CUDAMemoryPosition * outPosition = NULL, CUDAMemoryRegion * outRegion = NULL ) const;

	// Input Validation
	template<class T> Bool ValidateInput() const;

	// Solver Options
	inline Void ComputeEigenVectors( Bool bEnable );
	inline Void SetAlgorithm( CUSolverDenseEigenValueAlgorithm iAlgorithm );
	inline Void SetJacobiTolerance( Double fTolerance ); // Default = 1.0e-8
	inline Void SetJacobiMaxSweeps( UInt iMaxSweeps ); // Default = 100

	// Solver Routines
	inline CUSolverDenseEigenValueState GetSolverState() const;

	template<class T> Void Prepare();
	template<class T> Void Solve();
	Void UpdateStateAfterSync();
	Void Reset();

	inline UInt GetFailedToConvergeCount() const;
	inline UInt GetJacobiExecutedSweeps() const;
	inline Double GetJacobiResidual() const;

private:
	CUSolverDenseContext * m_pCUSolverDenseContext;

	CUDADeviceMemory * m_pMatrixA;
	CUDAMemoryPosition m_hMatrixPositionA;
	CUDAMemoryRegion m_hMatrixRegionA;
	CUBLASContextFillMode m_iFillModeA;

	CUDADeviceMemory * m_pVectorX;
	CUDAMemoryPosition m_hVectorPositionX;

	Bool m_bComputeEigenVectors;
	CUSolverDenseEigenValueAlgorithm m_iAlgorithm;
	Double m_fJacobiTolerance;
	UInt m_iJacobiMaxSweeps;

	CUDADeviceMemory m_hWorkspace;
	SizeT m_iWorkspaceSize;
	Void * m_hJacobiInfos;

	CUSolverDenseEigenValueState m_iSolverState;
	CUDADeviceMemory m_hDeviceInfo;
	CUDAHostMemory m_hDeviceInfoSaved;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUSolverDenseEigenValue.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_CUSOLVER_CUSOLVERDENSEEIGENVALUE_H

