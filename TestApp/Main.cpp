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
#include "Main.h"

/////////////////////////////////////////////////////////////////////////////////
// Entry Point
Int main()
{
    Int iDriverVersion = CUDAFn->GetDriverVersion();
    Int iRuntimeVersion = CUDAFn->GetRuntimeVersion();

    UInt iDeviceCount = CUDAFn->GetDeviceCount();
    CUDADeviceID iCurrentDevice = CUDAFn->GetCurrentDevice();

    const Char * strBusID = CUDAFn->GetDevicePCIBusID( iCurrentDevice );

    // Available
    Int iUnifiedAddressing = CUDAFn->GetDeviceAttribute( iCurrentDevice, CUDA_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING );
    Int iManagedMemory = CUDAFn->GetDeviceAttribute( iCurrentDevice, CUDA_DEVICE_ATTRIBUTE_MANAGED_MEMORY );
    Int iHostMemoryRegistration = CUDAFn->GetDeviceAttribute( iCurrentDevice, CUDA_DEVICE_ATTRIBUTE_HOST_MEMORY_REGISTRATION );
    Int iConcurrentKernels = CUDAFn->GetDeviceAttribute( iCurrentDevice, CUDA_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS );
    
    // Unavailable
    Int iPageableMemoryAccess = CUDAFn->GetDeviceAttribute( iCurrentDevice, CUDA_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS );
    Int iManagedMemoryConcurrent = CUDAFn->GetDeviceAttribute( iCurrentDevice, CUDA_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_MEMORY_ACCESS );
    Int iManagedMemoryHDA = CUDAFn->GetDeviceAttribute( iCurrentDevice, CUDA_DEVICE_ATTRIBUTE_MANAGED_MEMORY_HOST_DIRECT_ACCESS );
    Int iAsyncPooledMemory = CUDAFn->GetDeviceAttribute( iCurrentDevice, CUDA_DEVICE_ATTRIBUTE_ASYNC_AND_POOLED_MEMORY );
    Int iCoopKernels = CUDAFn->GetDeviceAttribute( iCurrentDevice, CUDA_DEVICE_ATTRIBUTE_COOPERATIVE_KERNELS );
    
    return 0;
}

