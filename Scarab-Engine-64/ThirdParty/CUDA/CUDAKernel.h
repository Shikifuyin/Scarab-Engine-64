/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDAKernel.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA Kernels
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
// Usage :
// class MyKernel : public CUDAKernel
// {
// public:
//     MyKernel():CUDAKernel() { Initialize kernel data here ... }
//     virtual ~MyKernel() { Cleanup kernel data here ... }
// protected:
//     CUDAEXEC_GLOBAL virtual Void Execute( const CUDAKernelDimension & hGridSize, const CUDAKernelIndex & hBlockIndex,
//                           				 const CUDAKernelDimension & hBlockSize, const CUDAKernelIndex & hThreadIndex )
//     {
//          Kernel code here ...
//     }
//
//     Kernel data here ... (should be device-allocated !)
// };
//
// MyKernel hMyKernelInstance;
// hMyKernelInstance.SetSharedMemory( ... );
// hMyKernelInstance.SetStream( ... );
// hMyKernelInstance.Launch( hGridSize, hBlockSize, bCooperative );
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_CUDA_CUDAKERNEL_H
#define SCARAB_THIRDPARTY_CUDA_CUDAKERNEL_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUDABuiltins.h"
#include "CUDAMappings.h"

#include "CUDAAsynchronous.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define CUDAKERNEL_MAX_PARAMETERS 8

typedef struct _cuda_kernel_dimension {
	inline struct _cuda_kernel_dimension & operator=( const struct _cuda_kernel_dimension & rhs ) {
		iX = rhs.iX;
		iY = rhs.iY;
		iZ = rhs.iZ;
		return (*this);
	}
	
	UInt iX;
	UInt iY;
	UInt iZ;
} CUDAKernelDimension;

typedef struct _cuda_kernel_index {
	inline struct _cuda_kernel_index & operator=( const struct _cuda_kernel_index & rhs ) {
		iX = rhs.iX;
		iY = rhs.iY;
		iZ = rhs.iZ;
		return (*this);
	}
	
	UInt iX;
	UInt iY;
	UInt iZ;
} CUDAKernelIndex;

// Prototypes
class CUDAGraph;
class CUDANodeKernel;

/////////////////////////////////////////////////////////////////////////////////
// The CUDAKernel class
class CUDAKernel
{
public:
	CUDAKernel( Void * pKernelFunction, UInt iParameterCount );
	virtual ~CUDAKernel();
	
	// Settings
	inline Void SetSharedMemory( SizeT iSharedMemorySize );
	inline Void SetStream( CUDAStream * pStream );

	// Parameters
	inline Void SetParameter( UInt iIndex, Void * pParameter );
	
	// Entry Point
	Void Launch( const CUDAKernelDimension & hGridSize, const CUDAKernelDimension & hBlockSize, Bool bCooperative = false );

private:
	friend class CUDAGraph;
	friend class CUDANodeKernel;

	// Shared Memory
	SizeT m_iSharedMemorySize;
	
	// Stream
	CUDAStream * m_pStream;

	// Kernel Function
	Void * m_pKernelFunction;
	UInt m_iParameterCount;
	Void * m_arrKernelParameters[CUDAKERNEL_MAX_PARAMETERS];
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUDAKernel.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_CUDAKERNEL_H
