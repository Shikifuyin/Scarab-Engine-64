/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/BLAS/CUBLASContext.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA BLAS Context management
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
// Usage : Users should create one CUBLASContext in each calling thread.
//         CUBLASContext lifetime should match its thread's lifetime.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_CUDA_BLAS_CUBLASCONTEXT_H
#define SCARAB_THIRDPARTY_CUDA_BLAS_CUBLASCONTEXT_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../CUDAMemory.h"
#include "../CUDAAsynchronous.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Log Callback
typedef Void (*CUBLASLogCallback)( const Char * strMessage );

// Batched Operations
#define CUBLAS_BATCH_MAX_COUNT 32768

/////////////////////////////////////////////////////////////////////////////////
// The CUBLASContext class
class CUBLASContext
{
public:
	CUBLASContext();
    ~CUBLASContext();
	
	// Deferred Creation/Destruction ////////////////////////////////////////////
	inline Bool IsCreated() const;
	Void Create();
	Void Destroy(); // Warning : Triggers Synchronization on current Device, use properly !
	
	// Runtime Infos ////////////////////////////////////////////////////////////
	Int GetVersion() const;
	
	// Pointer Mode /////////////////////////////////////////////////////////////
		// Default is Host pointers
		// - Host pointers are useful for final results that won't need
		//   further GPU processing
		// - Device pointers are usefull for intermediate results which
		//   do need further GPU processing (preferred for graphs/capture)
	Void SetPointerMode( CUBLASContextPointerMode iMode ) const;
	
	// Precision Mode ///////////////////////////////////////////////////////////
		// Default is CUBLAS_CONTEXT_PRECISION_MODE_DEFAULT
		// - CUBLAS_CONTEXT_PRECISION_MODE_DEFAULT is highest performance,
		//   tensor cores will be used whenever possible.
		//   Recommended for most usage.
		// - CUBLAS_CONTEXT_PRECISION_MODE_PRECISE forces internal precision
		//   to be the same precision as requested for output.
		//   Slower but more numerical robustness (good for testing/debugging).
		// - CUBLAS_CONTEXT_PRECISION_MODE_TF32 enables TF32 tensor cores for
		//   accelerated single precision routines.
		// - bDenyLowPrecisionReduction forces reductions during matrix multiplication
		//   to use the compute type rather than output type when output type has lower
		//   precision. Default is false (allow reduced precision).
	Void SetPrecisionMode( CUBLASContextPrecisionMode iMode, Bool bDenyLowPrecisionReduction = false ) const;
	
	// Atomics Mode /////////////////////////////////////////////////////////////
		// Default is disabled
		// Some routines have alternate implementation using atomics to provide more speed.
		// WARNING : Using atomics will generate results that may NOT be strictly
		// identical from one call to another ! Use carefully ! Bad when debugging !
		// Mathematically those discrepancies are non-significant.
	Void SetAtomicsMode( Bool bEnable ) const;
	
	// Logging Mode /////////////////////////////////////////////////////////////
		// Default is disabled
		// This is common to all contexts !
	static Void SetLoggingMode( CUBLASContextLoggingMode iMode, const Char * strLogFilename = NULL );
	static Void SetLogCallback( CUBLASLogCallback pfLogCallback ); // Auto-enables logging when given user callback
	
	// Stream Association ///////////////////////////////////////////////////////
		// Can pass NULL to use default stream again
		// Resets Memory to default workspace
	Void SetStream( CUDAStream * pStream ) const;
	
	// Memory Association ///////////////////////////////////////////////////////
		// Default CUBLAS Memory pool is 4Mb
		// User-provided memory should be at least 4Mb and shapeless
		// Most useful when using graphs & stream capture to avoid CUBLAS allocating
		// workspace memory for each new graph capturing CUBLAS routines.
	Void SetMemory( CUDADeviceMemory * pMemory ) const;
	
	// Memory Operations Helpers ////////////////////////////////////////////////
	Void GetVector( CUDAHostMemory * outHostVector, const CUDAMemoryPosition & outHostPosition, UInt iHostIncrement,
					const CUDADeviceMemory * pDeviceVector, const CUDAMemoryPosition & hDevicePosition, UInt iDeviceIncrement,
					SizeT iElementCount, CUDAStream * pStream = NULL );
	Void SetVector( CUDADeviceMemory * outDeviceVector, const CUDAMemoryPosition & outDevicePosition, UInt iDeviceIncrement,
					const CUDAHostMemory * pHostVector, const CUDAMemoryPosition & hHostPosition, UInt iHostIncrement,
					SizeT iElementCount, CUDAStream * pStream = NULL );

		// Note : Matrices are always stored column-wise
	Void GetMatrix( CUDAHostMemory * outHostMatrix, const CUDAMemoryPosition & outHostPosition,
					const CUDADeviceMemory * pDeviceMatrix, const CUDAMemoryPosition & hDevicePosition,
					const CUDAMemoryRegion & hCopyRegion, CUDAStream * pStream = NULL );
	Void SetMatrix( CUDADeviceMemory * outDeviceMatrix, const CUDAMemoryPosition & outDevicePosition,
					const CUDAHostMemory * pHostMatrix, const CUDAMemoryPosition & hHostPosition,
					const CUDAMemoryRegion & hCopyRegion, CUDAStream * pStream = NULL );

	// Vector-Scalar Functions //////////////////////////////////////////////////

	template<class T> Void Copy( CUDADeviceMemory * outDeviceVector, const CUDAMemoryPosition & outDevicePosition,
								 const CUDADeviceMemory * pDeviceVector, const CUDAMemoryPosition & hDevicePosition,
								 const CUDAMemoryRegion & hRegion );
	template<class T> inline Void Copy( CUDADeviceMemory * outDeviceVector, const CUDADeviceMemory * pDeviceVector ) const;

	template<class T> Void Swap( CUDADeviceMemory * pDeviceVectorA, const CUDAMemoryPosition & hDevicePositionA,
								 CUDADeviceMemory * pDeviceVectorB, const CUDAMemoryPosition & hDevicePositionB,
								 const CUDAMemoryRegion & hRegion );
	template<class T> inline Void Swap( CUDADeviceMemory * pDeviceVectorA, CUDADeviceMemory * pDeviceVectorB ) const;

	template<class T> SizeT AbsMin( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const;
	template<class T> inline SizeT AbsMin( const CUDADeviceMemory * pVector ) const;

	template<class T> SizeT AbsMax( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const;
	template<class T> inline SizeT AbsMax( const CUDADeviceMemory * pVector ) const;
	
	template<class T> T AbsSum( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const;
	template<class T> inline T AbsSum( const CUDADeviceMemory * pVector ) const;

	template<class T> T Dot( const CUDADeviceMemory * pVectorA, const CUDAMemoryPosition & hPositionA,
							 const CUDADeviceMemory * pVectorB, const CUDAMemoryPosition & hPositionB,
							 const CUDAMemoryRegion & hRegion, Bool bConjugateB = false ) const;
	template<class T> inline T Dot( const CUDADeviceMemory * pVectorA, const CUDADeviceMemory * pVectorB, Bool bConjugateB = false ) const;

	template<class T> T Norm( const CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion ) const;
	template<class T> inline T Norm( const CUDADeviceMemory * pVector ) const;

	template<class T> Void Scale( CUDADeviceMemory * pVector, const CUDAMemoryPosition & hPosition, const CUDAMemoryRegion & hRegion, T fAlpha ) const;
	template<class T> inline Void Scale( CUDADeviceMemory * pVector, T fAlpha ) const;

		// Y = Y + Alpha * X
	template<class T> Void MulAdd( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY,
								   const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX,
								   T fAlpha, const CUDAMemoryRegion & hRegion ) const;
	template<class T> inline Void MulAdd( CUDADeviceMemory * outVectorY, const CUDADeviceMemory * pVectorX, T fAlpha ) const;

		// Y = Y + X
	template<class T> inline Void Add( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY,
									   const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX,
									   const CUDAMemoryRegion & hRegion ) const;
	template<class T> inline Void Add( CUDADeviceMemory * outVectorY, const CUDADeviceMemory * pVectorX ) const;

		// Givens Rotations
	// cublas<t>rot, cublas<t>rotg, cublas<t>rotm, cublas<t>rotmg

	// Matrix-Vector Functions //////////////////////////////////////////////////

		// X = Op(A) * X
	template<class T> Void MulTriangular( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
										  const CUDADeviceMemory * pTriangularMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
										  CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const;
	template<class T> inline Void MulTriangular( CUDADeviceMemory * outVectorX, const CUDADeviceMemory * pTriangularMatrixA,
												 CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const;

		// X = Op(A) * X
		// Banded Matrix is stored column-wise with the following row order :
		// Fill Mode Lower : maindiag, subdiag1, subdiag2, ..., subdiagN
		// Fill Mode Upper : subdiagN, ..., subdiag2, subdiag1, maindiag
		// A(i,j) is stored at memory location :
		// Fill Mode Lower : [1+i-j,j]
		// Fill Mode Upper : [k+1+i-j,j]
	template<class T> Void MulTriangularBanded( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
												const CUDADeviceMemory * pTriangularBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
												SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const;
	template<class T> inline Void MulTriangularBanded( CUDADeviceMemory * outVectorX, const CUDADeviceMemory * pTriangularBandedMatrixA,
													   SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const;

		// Y = Alpha * Op(A) * X + Beta * Y
	template<class T> Void MulAdd( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, T fBeta,
								   const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, T fAlpha,
								   const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
								   CUBLASContextTransposeOp iTransOp ) const;
	template<class T> inline Void MulAdd( CUDADeviceMemory * outVectorY, T fBeta, const CUDADeviceMemory * pVectorX, T fAlpha,
										  const CUDADeviceMemory * pMatrixA, CUBLASContextTransposeOp iTransOp ) const;

		// Y = Alpha * A * X + Beta * Y
	template<class T> Void MulAddSymmetric( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, T fBeta,
											const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, T fAlpha,
											const CUDADeviceMemory * pSymmetricMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
											CUBLASContextFillMode iFillMode ) const;
	template<class T> inline Void MulAddSymmetric( CUDADeviceMemory * outVectorY, T fBeta, const CUDADeviceMemory * pVectorX, T fAlpha,
												   const CUDADeviceMemory * pSymmetricMatrixA, CUBLASContextFillMode iFillMode ) const;

		// Y = Alpha * A * X + Beta * Y
	template<class T> Void MulAddHermitian( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, T fBeta,
											const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, T fAlpha,
											const CUDADeviceMemory * pHermitianMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
											CUBLASContextFillMode iFillMode ) const;
	template<class T> inline Void MulAddHermitian( CUDADeviceMemory * outVectorY, T fBeta, const CUDADeviceMemory * pVectorX, T fAlpha,
												   const CUDADeviceMemory * pHermitianMatrixA, CUBLASContextFillMode iFillMode ) const;

		// Y = Alpha * Op(A) * X + Beta * Y
		// Banded Matrix is stored column-wise with the following row order :
		// updiagN, ..., updiag2, updiag1, maindiag, lowdiag1, lowdiag2, ..., lowdiagM
		// A(i,j) is stored at memory location :
		// [K+1+i-j,j] with K = number of upper diagonals
	template<class T> Void MulAddBanded( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, T fBeta,
										 const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, T fAlpha,
										 const CUDADeviceMemory * pBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
										 SizeT iExpandedSizeA, SizeT iLowerDiagsCount, SizeT iUpperDiagsCount, CUBLASContextTransposeOp iTransOp ) const;
	template<class T> inline Void MulAddBanded( CUDADeviceMemory * outVectorY, T fBeta, const CUDADeviceMemory * pVectorX, T fAlpha,
												const CUDADeviceMemory * pBandedMatrixA, SizeT iExpandedSizeA, SizeT iLowerDiagsCount, SizeT iUpperDiagsCount, CUBLASContextTransposeOp iTransOp ) const;

		// Y = Alpha * A * X + Beta * Y
		// Banded Matrix is stored column-wise with the following row order :
		// Fill Mode Lower : maindiag, subdiag1, subdiag2, ..., subdiagN
		// Fill Mode Upper : subdiagN, ..., subdiag2, subdiag1, maindiag
		// A(i,j) is stored at memory location :
		// Fill Mode Lower : [1+i-j,j]
		// Fill Mode Upper : [k+1+i-j,j]
	template<class T> Void MulAddSymmetricBanded( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, T fBeta,
												  const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, T fAlpha,
												  const CUDADeviceMemory * pSymmetricBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
												  SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode ) const;
	template<class T> inline Void MulAddSymmetricBanded( CUDADeviceMemory * outVectorY, T fBeta, const CUDADeviceMemory * pVectorX, T fAlpha,
														 const CUDADeviceMemory * pSymmetricBandedMatrixA, SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode ) const;

		// Y = Alpha * A * X + Beta * Y
		// Banded Matrix is stored column-wise with the following row order :
		// Fill Mode Lower : maindiag, subdiag1, subdiag2, ..., subdiagN
		// Fill Mode Upper : subdiagN, ..., subdiag2, subdiag1, maindiag
		// A(i,j) is stored at memory location :
		// Fill Mode Lower : [1+i-j,j]
		// Fill Mode Upper : [k+1+i-j,j]
	template<class T> Void MulAddHermitianBanded( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY, T fBeta,
												  const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX, T fAlpha,
												  const CUDADeviceMemory * pHermitianBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
												  SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode ) const;
	template<class T> inline Void MulAddHermitianBanded( CUDADeviceMemory * outVectorY, T fBeta, const CUDADeviceMemory * pVectorX, T fAlpha,
														 const CUDADeviceMemory * pHermitianBandedMatrixA, SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode ) const;

		// X = Inverse(Op(A)) * X
		// Solves Triangular System A*X = B, B is given in the X parameter and gets overwritten.
		// Does NOT test for singularity or near-singularity !
	template<class T> Void SolveTriangular( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
											const CUDADeviceMemory * pTriangularMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
											CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const;
	template<class T> inline Void SolveTriangular( CUDADeviceMemory * outVectorX, const CUDADeviceMemory * pTriangularMatrixA,
												   CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const;

		// X = Inverse(Op(A)) * X
		// Solves Triangular System A*X = B where B is given in the X parameter and gets overwritten.
		// Does NOT test for singularity or near-singularity !
		// Banded Matrix is stored column-wise with the following row order :
		// Fill Mode Lower : maindiag, subdiag1, subdiag2, ..., subdiagN
		// Fill Mode Upper : subdiagN, ..., subdiag2, subdiag1, maindiag
		// A(i,j) is stored at memory location :
		// Fill Mode Lower : [1+i-j,j]
		// Fill Mode Upper : [k+1+i-j,j]
	template<class T> Void SolveTriangularBanded( CUDADeviceMemory * outVectorX, const CUDAMemoryPosition & outPositionX,
												  const CUDADeviceMemory * pTriangularBandedMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA,
												  SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const;
	template<class T> inline Void SolveTriangularBanded( CUDADeviceMemory * outVectorX, const CUDADeviceMemory * pTriangularBandedMatrixA,
														 SizeT iExpandedSizeA, SizeT iSubDiagsCount, CUBLASContextFillMode iFillMode, CUBLASContextTransposeOp iTransOp, Bool bMainDiagIsUnity ) const;

		// UNIMPLEMENTED (for now ...) :
	// Packed versions : cublas<t>spmv, cublas<t>tpmv, cublas<t>tpsv, cublas<t>hpmv
	// Rank-Update Operations : cublas<t>ger, cublas<t>spr, cublas<t>spr2, cublas<t>syr, cublas<t>syr2, cublas<t>her, cublas<t>her2, cublas<t>hpr, cublas<t>hpr2

	// Matrix-Matrix Functions //////////////////////////////////////////////////

		// C = Alpha * Op(A) * Op(B) + Beta * C
	template<class T> Void MulAdd( CUDADeviceMemory * outMatrixC, const CUDAMemoryPosition & outPositionC, const CUDAMemoryRegion & outRegionC, T fBeta,
								   const CUDADeviceMemory * pMatrixA, const CUDAMemoryPosition & hPositionA, const CUDAMemoryRegion & hRegionA, T fAlpha,
								   const CUDADeviceMemory * pMatrixB, const CUDAMemoryPosition & hPositionB, const CUDAMemoryRegion & hRegionB,
								   CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB, Bool bUseComplexGaussReduction = false ) const;
	template<class T> inline Void MulAdd( CUDADeviceMemory * outMatrixC, T fBeta, const CUDADeviceMemory * pMatrixA, T fAlpha, const CUDADeviceMemory * pMatrixB,
										  CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB, Bool bUseComplexGaussReduction = false ) const;

		// Ci = Alpha * Op(Ai) * Op(Bi) + Beta * Ci
		// Matrices are presented in arrays of separated CUDADeviceMemory instances
		// A specific position can be given for each matrix inside its CUDADeviceMemory instance
		// Max Batch Count is CUBLAS_BATCH_MAX_COUNT
	template<class T> Void MulAddBatched( SizeT iBatchCount, CUDADeviceMemory * outMatricesC, const CUDAMemoryPosition * outPositionsC, const CUDAMemoryRegion & outRegionC, T fBeta,
										  const CUDADeviceMemory * arrMatricesA, const CUDAMemoryPosition * arrPositionsA, const CUDAMemoryRegion & hRegionA, T fAlpha,
										  const CUDADeviceMemory * arrMatricesB, const CUDAMemoryPosition * arrPositionsB, const CUDAMemoryRegion & hRegionB,
										  CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB ) const;

		// Ci = Alpha * Op(Ai) * Op(Bi) + Beta * Ci
		// Matrices are presented in 3D-shaped CUDADeviceMemory instances as lists of matrices
		// Stride values are in number of elements
		// Stride values must prevent overlap by being at least larger than a single matrix size (ie. Width * Height)
		// Batch Count is limited by matrix arrays depth
	template<class T> Void MulAddStrideBatched( SizeT iBatchCount, CUDADeviceMemory * outMatricesC, const CUDAMemoryPosition & outStartPositionC, const CUDAMemoryRegion & outRegionC, SizeT outStrideC, T fBeta,
												const CUDADeviceMemory * arrMatricesA, const CUDAMemoryPosition & hStartPositionA, const CUDAMemoryRegion & hRegionA, SizeT iStrideA, T fAlpha,
												const CUDADeviceMemory * arrMatricesB, const CUDAMemoryPosition & hStartPositionB, const CUDAMemoryRegion & hRegionB, SizeT iStrideB,
												CUBLASContextTransposeOp iTransOpA, CUBLASContextTransposeOp iTransOpB ) const;

private:
	Void * m_hContext;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUBLASContext.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_BLAS_CUBLASCONTEXT_H
