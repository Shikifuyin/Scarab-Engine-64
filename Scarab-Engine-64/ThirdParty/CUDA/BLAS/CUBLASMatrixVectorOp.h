/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/BLAS/CUBLASMatrixVectorOp.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA BLAS : Matrix-Vector Operations
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
#ifndef SCARAB_THIRDPARTY_CUDA_BLAS_REAL32_CUBLASMATRIXVECTOROP_H
#define SCARAB_THIRDPARTY_CUDA_BLAS_REAL32_CUBLASMATRIXVECTOROP_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUBLASContext.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The CUBLASMatrixVectorOp class
class CUBLASMatrixVectorOp
{
public:
	CUBLASMatrixVectorOp( CUBLASContext * pCUBLASContext );
	~CUBLASMatrixVectorOp();

	// Input : Matrix A
	inline Void SetMatrixA( const CUDADeviceMemory * pMatrix );
	inline Void SetMatrixPositionA( const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetMatrixRegionA( const CUDAMemoryRegion * pRegion = NULL );
	inline Void SetMatrixA( const CUDADeviceMemory * pMatrix, const CUDAMemoryPosition * pPosition = NULL, const CUDAMemoryRegion * pRegion = NULL );

	// Input-Output : Vector X
	inline Void SetVectorX( CUDADeviceMemory * pVector );
	inline Void SetVectorPositionX( const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetVectorX( CUDADeviceMemory * pVector, const CUDAMemoryPosition * pPosition = NULL );

	inline CUDADeviceMemory * GetVectorX( CUDAMemoryPosition * outPosition = NULL, CUDAMemoryRegion * outRegion = NULL ) const;

	// Input-Output : Vector Y
	inline Void SetVectorY( CUDADeviceMemory * pVector );
	inline Void SetVectorPositionY( const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetVectorY( CUDADeviceMemory * pVector, const CUDAMemoryPosition * pPosition = NULL );

	inline CUDADeviceMemory * GetVectorY( CUDAMemoryPosition * outPosition = NULL, CUDAMemoryRegion * outRegion = NULL ) const;

	// Input Validation
	template<class T> inline Bool ValidateInputX() const;
	template<class T> inline Bool ValidateInputXY() const;

	// Operations
		// Y = fScaleX * Op(A) * X + fScaleY * Y
	template<class T> Void MulAdd( T fScaleX, T fScaleY, CUBLASContextTransposeOp iTransOp );
		// Y = fScaleX * A * X + fScaleY * Y
	template<class T> Void MulAddSymmetric( T fScaleX, T fScaleY, CUBLASContextFillMode iFillMode );
	template<class T> Void MulAddHermitian( T fScaleX, T fScaleY, CUBLASContextFillMode iFillMode );

		// X = Op(A) * X
	template<class T> Void MulTriangular( CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity );

		// X = Inverse(Op(A)) * X
		// Solves AX = B, with B given in X parameter being overwritten
		// Does NOT test for singularity or near-singularity !
	template<class T> Void SolveTriangular( CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity );

	// Operations : Banded Matrices
		// Y = fScaleX * Op(A) * X + fScaleY * Y
		// Banded Matrix is stored column-wise with the following row order :
		// updiagN, ..., updiag2, updiag1, maindiag, lowdiag1, lowdiag2, ..., lowdiagM
		// A(i,j) is stored at memory location :
		// [k+i-j,j] with k = number of upper diagonals
	template<class T> Void MulAddBanded( T fScaleX, T fScaleY, SizeT iExpandedSizeA, SizeT iLowerDiagsCount, SizeT iUpperDiagsCount, CUBLASContextTransposeOp iTransOp );

		// Y = fScaleX * A * X + fScaleY * Y
		// Banded Matrix is stored column-wise with the following row order :
		// Fill Mode Lower : maindiag, subdiag1, subdiag2, ..., subdiagN
		// Fill Mode Upper : subdiagN, ..., subdiag2, subdiag1, maindiag
		// A(i,j) is stored at memory location :
		// Fill Mode Lower : [i-j,j]
		// Fill Mode Upper : [k+i-j,j] with k = number of upper diagonals
	template<class T> Void MulAddSymmetricBanded( T fScaleX, T fScaleY, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode );
	template<class T> Void MulAddHermitianBanded( T fScaleX, T fScaleY, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode );

		// X = Op(A) * X
		// Banded Matrix is stored column-wise with the following row order :
		// Fill Mode Lower : maindiag, subdiag1, subdiag2, ..., subdiagN
		// Fill Mode Upper : subdiagN, ..., subdiag2, subdiag1, maindiag
		// A(i,j) is stored at memory location :
		// Fill Mode Lower : [i-j,j]
		// Fill Mode Upper : [k+i-j,j] with k = number of upper diagonals
	template<class T> Void MulTriangularBanded( SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity );

		// X = Inverse(Op(A)) * X
		// Solves AX = B, with B given in X parameter being overwritten
		// Does NOT test for singularity or near-singularity !
		// Banded Matrix is stored column-wise with the following row order :
		// Fill Mode Lower : maindiag, subdiag1, subdiag2, ..., subdiagN
		// Fill Mode Upper : subdiagN, ..., subdiag2, subdiag1, maindiag
		// A(i,j) is stored at memory location :
		// Fill Mode Lower : [i-j,j]
		// Fill Mode Upper : [k+i-j,j] with k = number of upper diagonals
	template<class T> Void SolveTriangularBanded( SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity );

	// UNIMPLEMENTED (for now ...) :
	// Packed versions : cublas<t>spmv, cublas<t>tpmv, cublas<t>tpsv, cublas<t>hpmv
	// Rank-Update Operations : cublas<t>ger, cublas<t>spr, cublas<t>spr2, cublas<t>syr, cublas<t>syr2, cublas<t>her, cublas<t>her2, cublas<t>hpr, cublas<t>hpr2

private:
	CUBLASContext * m_pCUBLASContext;

	const CUDADeviceMemory * m_pMatrixA;
	CUDAMemoryPosition m_hMatrixPositionA;
	CUDAMemoryRegion m_hMatrixRegionA;

	CUDADeviceMemory * m_pVectorX;
	CUDAMemoryPosition m_hVectorPositionX;

	CUDADeviceMemory * m_pVectorY;
	CUDAMemoryPosition m_hVectorPositionY;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUBLASMatrixVectorOp.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_BLAS_REAL32_CUBLASMATRIXVECTOROP_H

