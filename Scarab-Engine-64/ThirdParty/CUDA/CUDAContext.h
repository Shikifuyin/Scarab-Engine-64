/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDAContext.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA Context management
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
#ifndef SCARAB_THIRDPARTY_CUDA_CUDACONTEXT_H
#define SCARAB_THIRDPARTY_CUDA_CUDACONTEXT_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUDAMappings.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define CUDAFn CUDAContext::GetInstance()

// Device-Management
typedef Int CUDADeviceID;

/////////////////////////////////////////////////////////////////////////////////
// The CUDAContext class
class CUDAContext
{
	// Discrete singleton interface
public:
    inline static CUDAContext * GetInstance();

private:
    CUDAContext();
    ~CUDAContext();

public:
	// CUDA Runtime /////////////////////////////////////////////////////////////////
	Int GetDriverVersion() const;
	Int GetRuntimeVersion() const;
	
	SizeT GetLimit( CUDALimit iLimit ) const;
	Void SetLimit( CUDALimit iLimit, SizeT iLimitValue ) const;
	
	// Error-Handling ///////////////////////////////////////////////////////////////
	DWord GetLastError() const;
	DWord PeekLastError() const; // Don't reset current error
	
	const Char * GetErrorName( DWord dwErrorCode ) const;
	const Char * GetErrorString( DWord dwErrorCode ) const;
	
	// Device-Management ////////////////////////////////////////////////////////////
	Int GetDeviceAttribute( CUDADeviceID iDeviceID, CUDADeviceAttribute iAttribute ) const;
	//Void GetDeviceProperties( CUDADeviceProperties * outProperties, CUDADeviceID iDeviceID ) const;
	
	const Char * GetDevicePCIBusID( CUDADeviceID iDeviceID ) const;
	CUDADeviceID GetDeviceByPCIBusID( const Char * strPCIBusID ) const;
	
	UInt GetDeviceInitFlags() const;                  // Uses CUDADeviceInitFlags
	Void SetDeviceInitFlags( UInt iInitFlags ) const; // Setup this before initializing current device !
	
	UInt GetDeviceCount() const;
	Void SetValidDevices( CUDADeviceID * arrDeviceIDs, UInt iDeviceCount ) const;
	
	//CUDADeviceID MatchDevice( const CUDADeviceProperties * pRequiredProperties );
	
	Void SetCurrentDevice( CUDADeviceID iDeviceID ) const;
	CUDADeviceID GetCurrentDevice() const;
	
	CUDADeviceCacheConfig GetCurrentDeviceCacheConfig() const;
	Void SetCurrentDeviceCacheConfig( CUDADeviceCacheConfig iCacheConfig ) const;
	
	CUDADeviceSharedMemoryConfig GetCurrentDeviceSharedMemoryConfig() const;
	Void SetCurrentDeviceSharedMemoryConfig( CUDADeviceSharedMemoryConfig iSharedMemoryConfig ) const;
	
	Void GetCurrentDeviceStreamPriorityRange( Int * outLowestPriority, Int * outHighestPriority ) const;
	
	Void ResetCurrentDevice() const;
	Void WaitForCurrentDevice() const;

	Void ResetPersistentL2Cache() const;

	// Inter-Process Communication (IPC)
	//Void * IPCOpenMemoryHandle( CUDASharedMemoryHandle hSharedMemory, CUDASharedMemoryFlags iFlags ) const;
	//Void IPCCloseMemoryHandle( Void * pDeviceMemory ) const;
	//CUDASharedMemoryHandle IPCGetMemoryHandle( Void * pDeviceMemory ) const;
	
	//CUDAEvent IPCOpenEventHandle( CUDAIPCEventHandle hIPCEvent ) const;
	//CUDAIPCEventHandle IPCGetEventHandle( CUDAEvent hEvent ) const;
	
	
	
	
		// Memory Pools
	//CUDAMemoryPool GetDeviceDefaultMemoryPool( CUDADeviceID iDeviceID ) const;
	//CUDAMemoryPool GetDeviceMemoryPool( CUDADeviceID iDeviceID ) const;
	//Void SetDeviceMemoryPool( CUDADeviceID iDeviceID, CUDAMemoryPool hMemoryPool ) const;
	
		// Arrays
	// Void DeviceMemFree( CUDAArray hArray ) const;
	// Void DeviceMemFree( CUDAMipMapArray hMipMapArray ) const;

	// Void GetArrayFormat( CUDAChannelFormat * outChannelFormat, CUDAArray hArray ) const;
	// Void GetArrayExtent( CUDAExtent * outExtent, CUDAArray hArray ) const;
	// Void GetArrayFlags( UInt * outArrayFlags, CUDAArray hArray ) const; // Uses CUDAArrayFlags
	
	// CUDAArray GetArrayPlane( CUDAArray hArray, UInt iPlane ) const;
	// Void GetArraySparseProperties( CUDAArraySparseProperties * outSparseProperties, CUDAArray hArray ) const;
	
	// CUDAArray GetMipMapArrayLevel( CUDAMipMapArray hMipMapArray, UInt iLevel ) const;
	

private:
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUDAContext.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_CUDACONTEXT_H
