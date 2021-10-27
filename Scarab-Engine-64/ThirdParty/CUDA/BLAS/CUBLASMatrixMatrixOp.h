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
	inline Void SetMatrixA( const CUDADeviceMemory * pMatrix );
	inline Void SetMatrixPositionA( const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetMatrixRegionA( const CUDAMemoryRegion * pRegion = NULL );
	inline Void SetMatrixA( const CUDADeviceMemory * pMatrix, const CUDAMemoryPosition * pPosition = NULL, const CUDAMemoryRegion * pRegion = NULL );

	// Input : Matrix B
	inline Void SetMatrixB( const CUDADeviceMemory * pMatrix );
	inline Void SetMatrixPositionB( const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetMatrixRegionB( const CUDAMemoryRegion * pRegion = NULL );
	inline Void SetMatrixB( const CUDADeviceMemory * pMatrix, const CUDAMemoryPosition * pPosition = NULL, const CUDAMemoryRegion * pRegion = NULL );

	// Output : Matrix C
	inline Void SetMatrixC( CUDADeviceMemory * pMatrix );
	inline Void SetMatrixPositionC( const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetMatrixRegionC( const CUDAMemoryRegion * pRegion = NULL );
	inline Void SetMatrixC( CUDADeviceMemory * pMatrix, const CUDAMemoryPosition * pPosition = NULL, const CUDAMemoryRegion * pRegion = NULL );

	inline CUDADeviceMemory * GetMatrixC( CUDAMemoryPosition * outPosition = NULL, CUDAMemoryRegion * outRegion = NULL ) const;

	// Input Validation
	template<class T> inline Bool ValidateInputA() const;
	template<class T> inline Bool ValidateInputAB() const;

	// Operations
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
// The CUBLASMatrixMatrixOpBatched class
class CUBLASMatrixMatrixOpBatched
{
public:
	CUBLASMatrixMatrixOpBatched( CUBLASContext * pCUBLASContext );
	~CUBLASMatrixMatrixOpBatched();

	// Batch Count
	inline SizeT GetBatchCount() const;
	inline Void SetBatchCount( SizeT iBatchCount );

	// Stride Mode (3D-shaped memory)
	inline Bool IsStrideMode() const;
	inline Void SetStrideMode( Bool bStrideMode );

	// Input : Matrix A
	inline Void SetMatrixA( const CUDADeviceMemory * arrMatrices );
	inline Void SetMatrixPositionA( const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetMatrixRegionA( const CUDAMemoryRegion * pRegion = NULL );
	inline Void SetMatrixA( const CUDADeviceMemory * arrMatrices, const CUDAMemoryPosition * pPosition, const CUDAMemoryRegion * pRegion = NULL );

	// Input : Matrix B
	inline Void SetMatrixB( const CUDADeviceMemory * arrMatrices );
	inline Void SetMatrixPositionB( const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetMatrixRegionB( const CUDAMemoryRegion * pRegion = NULL );
	inline Void SetMatrixB( const CUDADeviceMemory * arrMatrices, const CUDAMemoryPosition * pPosition = NULL, const CUDAMemoryRegion * pRegion = NULL );

	// Output : Matrix C
	inline Void SetMatrixC( CUDADeviceMemory * arrMatrices );
	inline Void SetMatrixPositionC( const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetMatrixRegionC( const CUDAMemoryRegion * pRegion = NULL );
	inline Void SetMatrixC( CUDADeviceMemory * arrMatrices, const CUDAMemoryPosition * pPosition = NULL, const CUDAMemoryRegion * pRegion = NULL );

	inline CUDADeviceMemory * GetMatrixC( SizeT iBatchIndex, CUDAMemoryPosition * outPosition = NULL, CUDAMemoryRegion * outRegion = NULL ) const;

	// Input Validation
	template<class T> inline Bool ValidateInputA() const;
	template<class T> inline Bool ValidateInputAB() const;

private:
	CUBLASContext * m_pCUBLASContext;

	SizeT m_iBatchCount;
	Bool m_bStrideMode;

	const CUDADeviceMemory * m_arrMatrixA;
	CUDAMemoryPosition m_hMatrixPositionA;
	CUDAMemoryRegion m_hMatrixRegionA;

	const CUDADeviceMemory * m_arrMatrixB;
	CUDAMemoryPosition m_hMatrixPositionB;
	CUDAMemoryRegion m_hMatrixRegionB;

	CUDADeviceMemory * m_arrMatrixC;
	CUDAMemoryPosition m_hMatrixPositionC;
	CUDAMemoryRegion m_hMatrixRegionC;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUBLASMatrixMatrixOp.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_BLAS_REAL32_CUBLASMATRIXMATRIXOP_H


