/////////////////////////////////////////////////////////////////////////////////
// File : Main.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Test Entry Point
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include <typeinfo>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "Main.h"

/////////////////////////////////////////////////////////////////////////////////
// Entry Point
Int main()
{
	CUBLASContext hCUBLAS;
	hCUBLAS.Create();

	CUDAHostMemory hHostMatrix;
	hHostMatrix.Allocate2D( sizeof(CUDAReal32), 3, 3 );

	CUDAMemoryPosition hPos;
	hPos.iZ = 0;

	hPos.iX = 0;
	hPos.iY = 0;
	hHostMatrix.Write<CUDAReal32>( hPos, 2.0f );
	hPos.iX = 1;
	hPos.iY = 0;
	hHostMatrix.Write<CUDAReal32>( hPos, 0.0f );
	hPos.iX = 2;
	hPos.iY = 0;
	hHostMatrix.Write<CUDAReal32>( hPos, 0.0f );

	hPos.iX = 0;
	hPos.iY = 1;
	hHostMatrix.Write<CUDAReal32>( hPos, 0.0f );
	hPos.iX = 1;
	hPos.iY = 1;
	hHostMatrix.Write<CUDAReal32>( hPos, 3.0f );
	hPos.iX = 2;
	hPos.iY = 1;
	hHostMatrix.Write<CUDAReal32>( hPos, 0.0f );

	hPos.iX = 0;
	hPos.iY = 2;
	hHostMatrix.Write<CUDAReal32>( hPos, 0.0f );
	hPos.iX = 1;
	hPos.iY = 2;
	hHostMatrix.Write<CUDAReal32>( hPos, 0.0f );
	hPos.iX = 2;
	hPos.iY = 2;
	hHostMatrix.Write<CUDAReal32>( hPos, 4.0f );

	CUDADeviceMemory hDeviceMatrixA;
	hDeviceMatrixA.Allocate2D( sizeof(CUDAReal32), 3, 3 );
	CUDADeviceMemory hDeviceMatrixB;
	hDeviceMatrixB.Allocate2D( sizeof(CUDAReal32), 3, 3 );
	CUDADeviceMemory hDeviceMatrixC;
	hDeviceMatrixC.Allocate2D( sizeof(CUDAReal32), 3, 3 );

	//hDeviceMatrixA.Copy( CUDAMemoryPosition(), &hHostMatrix, CUDAMemoryPosition(), CUDAMemoryRegion(3,3,0) );
	//hDeviceMatrixB.Copy( CUDAMemoryPosition(), &hHostMatrix, CUDAMemoryPosition(), CUDAMemoryRegion(3,3,0) );

	hCUBLAS.SetMatrix( &hDeviceMatrixA, CUDAMemoryPosition(), &hHostMatrix, CUDAMemoryPosition(), CUDAMemoryRegion(3, 3, 0) );
	hCUBLAS.SetMatrix( &hDeviceMatrixB, CUDAMemoryPosition(), &hHostMatrix, CUDAMemoryPosition(), CUDAMemoryRegion(3, 3, 0) );

	CUBLASMatrixMatrixOp hMatrixOp( &hCUBLAS );

	hMatrixOp.SetMatrixA( &hDeviceMatrixA );
	hMatrixOp.SetMatrixB( &hDeviceMatrixB );
	hMatrixOp.SetMatrixC( &hDeviceMatrixC );

	hMatrixOp.MulAddSymmetric<CUDAReal32>( 1.0f, 0.0f, CUBLAS_CONTEXT_SIDEMODE_LEFT, CUBLAS_CONTEXT_FILLMODE_UPPER );
	CUDAFn->WaitForCurrentDevice();

	//hHostMatrix.Copy( CUDAMemoryPosition(), &hDeviceMatrixC, CUDAMemoryPosition(), CUDAMemoryRegion(3,3,0) );

	hCUBLAS.GetMatrix( &hHostMatrix, CUDAMemoryPosition(), &hDeviceMatrixC, CUDAMemoryPosition(), CUDAMemoryRegion(3, 3, 0) );

	hDeviceMatrixC.Free();
	hDeviceMatrixB.Free();
	hDeviceMatrixA.Free();
	hHostMatrix.Free();
	hCUBLAS.Destroy();

	return 0;
}


