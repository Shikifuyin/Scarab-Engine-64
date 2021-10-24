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

/////////////////////////////////////////////////////////////////////////////////
// The CUBLASContext class
class CUBLASContext
{
public:
	CUBLASContext();
    ~CUBLASContext();
	
	// Deferred Creation/Destruction
	inline Bool IsCreated() const;
	Void Create();
	Void Destroy(); // Warning : Triggers Synchronization on current Device, use properly !
	
	// Runtime Infos
	Int GetVersion() const;
	
	// Pointer Mode
		// Default is Host pointers
		// - Host pointers are useful for final results that won't need
		//   further GPU processing
		// - Device pointers are usefull for intermediate results which
		//   do need further GPU processing (preferred for graphs/capture)
	Void SetPointerMode( CUBLASContextPointerMode iMode ) const;
	
	// Precision Mode
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
	
	// Atomics Mode
		// Default is disabled
		// Some routines have alternate implementation using atomics to provide more speed.
		// WARNING : Using atomics will generate results that may NOT be strictly
		// identical from one call to another ! Use carefully ! Bad when debugging !
		// Mathematically those discrepancies are non-significant.
	Void SetAtomicsMode( Bool bEnable ) const;
	
	// Logging Mode
		// Default is disabled
		// This is common to all contexts !
	static Void SetLoggingMode( CUBLASContextLoggingMode iMode, const Char * strLogFilename = NULL );
	static Void SetLogCallback( CUBLASLogCallback pfLogCallback ); // Auto-enables logging when given user callback
	
	// Stream Association
		// Can pass NULL to use default stream again
		// Resets Memory to default workspace
	Void SetStream( CUDAStream * pStream ) const;
	
	// Memory Association
		// Default CUBLAS Memory pool is 4Mb
		// User-provided memory should be at least 4Mb and shapeless
		// Most useful when using graphs & stream capture to avoid CUBLAS allocating
		// workspace memory for each new graph capturing CUBLAS routines.
	Void SetMemory( CUDADeviceMemory * pMemory ) const;
	
	// Memory Operations Helpers
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

	// Vector-Scalar Functions
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

		// Y += Alpha * X
	template<class T> Void MulAdd( CUDADeviceMemory * outVectorY, const CUDAMemoryPosition & outPositionY,
								   const CUDADeviceMemory * pVectorX, const CUDAMemoryPosition & hPositionX,
								   T fAlpha, const CUDAMemoryRegion & hRegion ) const;
	template<class T> inline Void MulAdd( CUDADeviceMemory * outVectorY, const CUDADeviceMemory * pVectorX, T fAlpha ) const;

		// Givens Rotations
	///////////////////////////////////////////

	// Matrix-Vector Functions


private:
	Void * m_hContext;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUBLASContext.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_BLAS_CUBLASCONTEXT_H
