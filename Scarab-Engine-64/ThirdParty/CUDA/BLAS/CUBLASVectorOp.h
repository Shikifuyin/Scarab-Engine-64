/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/BLAS/CUBLASVectorOp.h
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
// Header prelude
#ifndef SCARAB_THIRDPARTY_CUDA_BLAS_REAL32_CUBLASVECTOROP_H
#define SCARAB_THIRDPARTY_CUDA_BLAS_REAL32_CUBLASVECTOROP_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUBLASContext.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The CUBLASVectorOp class
class CUBLASVectorOp
{
public:
	CUBLASVectorOp( CUBLASContext * pCUBLASContext );
	~CUBLASVectorOp();

	// Input-Output : Vector
	inline Void SetVector( CUDADeviceMemory * pVector );
	inline Void SetVectorPosition( const CUDAMemoryPosition * pPosition = NULL );
	inline Void SetVectorRegion( const CUDAMemoryRegion * pRegion = NULL );
	inline Void SetVector( CUDADeviceMemory * pVector, const CUDAMemoryPosition * pPosition = NULL, const CUDAMemoryRegion * pRegion = NULL );

	inline CUDADeviceMemory * GetVector( CUDAMemoryPosition * outPosition = NULL, CUDAMemoryRegion * outRegion = NULL ) const;

	// Input Validation
	template<class T> inline Bool ValidateInput() const;

	// Operations
		// Computes the minimal magnitude element and return its index
	template<class T> SizeT AbsMin() const;
		// Computes the maximal magnitude element and return its index
	template<class T> SizeT AbsMax() const;
		// Computes the sum of magnitudes of all elements
	template<class T> T AbsSum() const;
		// Computes the norm-2
	template<class T> T Norm() const;
		// Scales by the given factor
	template<class T> Void Scale( T fScale );

private:
	CUBLASContext * m_pCUBLASContext;

	CUDADeviceMemory * m_pVector;
	CUDAMemoryPosition m_hVectorPosition;
	CUDAMemoryRegion m_hVectorRegion;
};

/////////////////////////////////////////////////////////////////////////////////
// The CUBLASVectorVectorOp class
class CUBLASVectorVectorOp
{
public:
	CUBLASVectorVectorOp( CUBLASContext * pCUBLASContext );
	~CUBLASVectorVectorOp();

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

	// Input : Vector Region
	inline Void SetVectorRegion( const CUDAMemoryRegion * pRegion = NULL );

	// Input Validation
	template<class T> inline Bool ValidateInput() const;

	// Operations
		// Y = X
	template<class T> Void Copy();
		// Y <-> X
	template<class T> Void Swap();

		// Y = Y + fScaleX * X
	template<class T> Void MulAdd( T fScaleX );
		// Y = Y + X
	template<class T> inline Void Add();
		// Y = Y - X
	template<class T> inline Void Sub();

		// X . Y
	template<class T> T Dot( Bool bConjugateY = false ) const;

	// UNIMPLEMENTED (for now ...) :
	// Givens Rotations : cublas<t>rot, cublas<t>rotg, cublas<t>rotm, cublas<t>rotmg

private:
	CUBLASContext * m_pCUBLASContext;

	CUDADeviceMemory * m_pVectorX;
	CUDAMemoryPosition m_hVectorPositionX;

	CUDADeviceMemory * m_pVectorY;
	CUDAMemoryPosition m_hVectorPositionY;

	CUDAMemoryRegion m_hVectorRegion;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUBLASVectorOp.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_BLAS_REAL32_CUBLASVECTOROP_H

