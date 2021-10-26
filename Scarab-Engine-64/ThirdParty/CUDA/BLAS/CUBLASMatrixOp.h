/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/BLAS/CUBLASMatrixOp.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA BLAS : Matrix Operations
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
#ifndef SCARAB_THIRDPARTY_CUDA_BLAS_REAL32_CUBLASMATRIXOP_H
#define SCARAB_THIRDPARTY_CUDA_BLAS_REAL32_CUBLASMATRIXOP_H

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
	template<class T> inline Bool ValidateInput() const;

	// Operations
		// X = Op(A) * X
	template<class T> Void MulTriangular( CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity );


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
// The CUBLASMatrixMatrixOp class
class CUBLASMatrixMatrixOp
{
public:
	CUBLASMatrixMatrixOp( CUBLASContext * pCUBLASContext );
	~CUBLASMatrixMatrixOp();

	// Input Validation
	//template<class T> inline Bool ValidateInput() const;

	// Operations

	// UNIMPLEMENTED (for now ...) :

private:
	CUBLASContext * m_pCUBLASContext;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUBLASMatrixOp.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_BLAS_REAL32_CUBLASMATRIXOP_H

