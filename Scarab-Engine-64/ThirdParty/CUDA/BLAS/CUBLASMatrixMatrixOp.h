/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/BLAS/CUBLASMatrixMatrixOp.h
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
// Header prelude
#ifndef SCARAB_THIRDPARTY_CUDA_BLAS_REAL32_CUBLASMATRIXMATRIXOP_H
#define SCARAB_THIRDPARTY_CUDA_BLAS_REAL32_CUBLASMATRIXMATRIXOP_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUBLASContext.h"
#include "CUBLASVectorOp.h" // Needed for addition

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The CUBLASMatrixMatrixOp class
class CUBLASMatrixMatrixOp
{
public:
	CUBLASMatrixMatrixOp( CUBLASContext * pCUBLASContext );
	~CUBLASMatrixMatrixOp();

	// Input : Matrix A
	inline Void SetMatrixA( const CUDADeviceMemory * pMatrix, const CUDAMemoryPosition * pPosition = NULL, const CUDAMemoryRegion * pRegion = NULL );
	inline Void SetMatrixPositionA( const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetMatrixRegionA( const CUDAMemoryRegion * pRegion = NULL );

	// Input : Matrix B
	inline Void SetMatrixB( const CUDADeviceMemory * pMatrix, const CUDAMemoryPosition * pPosition = NULL, const CUDAMemoryRegion * pRegion = NULL );
	inline Void SetMatrixPositionB( const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetMatrixRegionB( const CUDAMemoryRegion * pRegion = NULL );

	// Output : Matrix C
	inline Void SetMatrixC( CUDADeviceMemory * pMatrix, const CUDAMemoryPosition * pPosition = NULL, const CUDAMemoryRegion * pRegion = NULL );
	inline Void SetMatrixPositionC( const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetMatrixRegionC( const CUDAMemoryRegion * pRegion = NULL );

	inline CUDADeviceMemory * GetMatrixC( CUDAMemoryPosition * outPosition = NULL, CUDAMemoryRegion * outRegion = NULL ) const;

	// Input Validation
	template<class T> inline Bool ValidateInputA() const;
	template<class T> inline Bool ValidateInputAB() const;

	template<class T> inline Bool ValidateInputBatchedA( SizeT iBatchIndex, Bool bStripMode ) const;
	template<class T> inline Bool ValidateInputBatchedAB( SizeT iBatchIndex, Bool bStripMode ) const;

	// Operations
		// C = fScaleA * A + C
	template<class T> Void Add( T fScaleA );

		// C = fScaleA * Op(A) * Op(B) + fScaleC * C
	template<class T> Void MulAdd( T fScaleA, T fScaleC, CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB, Bool bUseComplexGaussReduction = false );

		// C = fScaleA * A * B + fScaleC * C (CUBLAS_CONTEXT_SIDEMODE_LEFT)
		// C = fScaleA * B * A + fScaleC * C (CUBLAS_CONTEXT_SIDEMODE_RIGHT)
	template<class T> Void MulAddSymmetric( T fScaleA, T fScaleC, CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode );
	template<class T> Void MulAddHermitian( T fScaleA, T fScaleC, CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode );

		// C = fScaleA * Op(A) * B (CUBLAS_CONTEXT_SIDEMODE_LEFT)
		// C = fScaleA * B * Op(A) (CUBLAS_CONTEXT_SIDEMODE_RIGHT)
	template<class T> Void MulTriangular( T fScaleA, CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA );

		// C = fScaleA * Inverse(Op(A)) * C (CUBLAS_CONTEXT_SIDEMODE_LEFT)
		// C = fScaleA * C * Inverse(Op(A)) (CUBLAS_CONTEXT_SIDEMODE_RIGHT)
		// Solves Triangular System Op(A)*C = Alpha * B, B is given in the C parameter and gets overwritten. (CUBLAS_CONTEXT_SIDEMODE_LEFT)
		// Solves Triangular System C*Op(A) = Alpha * B, B is given in the C parameter and gets overwritten. (CUBLAS_CONTEXT_SIDEMODE_RIGHT)
		// Does NOT test for singularity or near-singularity !
	template<class T> Void SolveTriangular( T fScaleA, CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA );

	// Operations : Batched

		// Ci = fScaleA * Op(Ai) * Op(Bi) + fScaleC * Ci
		// Regular Mode : Matrices are presented in arrays of separated CUDADeviceMemory instances
		//				  Max Batch Count is CUBLAS_BATCH_MAX_COUNT, Better for small batch counts
		// Strip Mode : Matrices are presented in 3D-shaped CUDADeviceMemory instances as lists of matrices
		//			    Batch Count is only limited by matrix arrays depth, Better for large batch counts
	template<class T> Void MulAddBatched( T fScaleA, T fScaleC, CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB, SizeT iBatchCount, Bool bStripMode );

		// Ci = fScaleA * Inverse(Op(Ai)) * Ci (CUBLAS_CONTEXT_SIDEMODE_LEFT)
		// Ci = fScaleA * Ci * Inverse(Op(Ai)) (CUBLAS_CONTEXT_SIDEMODE_RIGHT)
		// Solves Triangular Systems Op(Ai)*Ci = Alpha * Bi, Bi is given in the Ci parameter and gets overwritten. (CUBLAS_CONTEXT_SIDEMODE_LEFT)
		// Solves Triangular Systems Ci*Op(Ai) = Alpha * Bi, Bi is given in the Ci parameter and gets overwritten. (CUBLAS_CONTEXT_SIDEMODE_RIGHT)
		// Does NOT test for singularity or near-singularity !
		// Regular Mode : Matrices are presented in arrays of separated CUDADeviceMemory instances
		//				  Max Batch Count is CUBLAS_BATCH_MAX_COUNT, Better for small batch counts
		// Strip Mode : Matrices are presented in 3D-shaped CUDADeviceMemory instances as lists of matrices
		//			    Batch Count is only limited by matrix arrays depth, Better for large batch counts
		//				STRIP MODE IS UNAVAILABLE FOR THIS ONE ! bStripMode MUST BE PASSED false
	template<class T> Void SolveTriangularBatched( T fScaleA, CUBLASContextSideMode iSideMode, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOpA, Bool bMainDiagIsUnityA, SizeT iBatchCount, Bool bStripMode );

	// UNIMPLEMENTED (for now ...) :
	// Rank-Update Operations : cublas<t>syrk, cublas<t>syr2k, cublas<t>syrkx, cublas<t>herk, cublas<t>her2k, cublas<t>herkx

private:
	CUBLASContext * m_pCUBLASContext;

	const CUDADeviceMemory * m_pMatrixA;
	CUDAMemoryPosition m_hMatrixPositionA;
	CUDAMemoryRegion m_hMatrixRegionA;

	const CUDADeviceMemory * m_pMatrixB;
	CUDAMemoryPosition m_hMatrixPositionB;
	CUDAMemoryRegion m_hMatrixRegionB;

	CUDADeviceMemory * m_pMatrixC;
	CUDAMemoryPosition m_hMatrixPositionC;
	CUDAMemoryRegion m_hMatrixRegionC;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUBLASMatrixMatrixOp.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_BLAS_REAL32_CUBLASMATRIXMATRIXOP_H


