/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDABuiltIns.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA Built-In definitions
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_CUDA_CUDABUILTINS_H
#define SCARAB_THIRDPARTY_CUDA_CUDABUILTINS_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../System/Platform.h"

/////////////////////////////////////////////////////////////////////////////////
// Memory Space Specifiers
#define CUDAMEM_DEVICE   __device__
#define CUDAMEM_CONSTANT __constant__
#define CUDAMEM_SHARED   __shared__
#define CUDAMEM_MANAGED  __managed__

// For most CUDA purposes, avoid using restricted pointers because of register pressure ...
//#define CUDAMEM_RESTRICTPTR __restrict__

/////////////////////////////////////////////////////////////////////////////////
// Execution Specifiers
#define CUDAEXEC_GLOBAL   __global__
#define CUDAEXEC_DEVICE   __device__
#define CUDAEXEC_HOST     __host__
#define CUDAEXEC_INLINE   __forceinline__
#define CUDAEXEC_NOINLINE __noinline__

/////////////////////////////////////////////////////////////////////////////////
// Built-in type definitions

// Vector Types : Guaranteed bit-size types
typedef char1 VInt8_1;
typedef char2 VInt8_2;
typedef char3 VInt8_3;
typedef char4 VInt8_4;

typedef uchar1 VUInt8_1;
typedef uchar2 VUInt8_2;
typedef uchar3 VUInt8_3;
typedef uchar4 VUInt8_4;

typedef short1 VInt16_1;
typedef short2 VInt16_2;
typedef short3 VInt16_3;
typedef short4 VInt16_4;

typedef ushort1 VUInt16_1;
typedef ushort2 VUInt16_2;
typedef ushort3 VUInt16_3;
typedef ushort4 VUInt16_4;

typedef int1 VInt32_1;
typedef int2 VInt32_2;
typedef int3 VInt32_3;
typedef int4 VInt32_4;

typedef uint1 VUInt32_1;
typedef uint2 VUInt32_2;
typedef uint3 VUInt32_3;
typedef uint4 VUInt32_4;

typedef longlong1 VInt64_1;
typedef longlong2 VInt64_2;
typedef longlong3 VInt64_3;
typedef longlong4 VInt64_4;

typedef ulonglong1 VUInt64_1;
typedef ulonglong2 VUInt64_2;
typedef ulonglong3 VUInt64_3;
typedef ulonglong4 VUInt64_4;

typedef float1 VFloat32_1;
typedef float2 VFloat32_2;
typedef float3 VFloat32_3;
typedef float4 VFloat32_4;

typedef double1 VFloat64_1;
typedef double2 VFloat64_2;
typedef double3 VFloat64_3;
typedef double4 VFloat64_4;

// Vector Types : Conventional type names
typedef VInt8_1 VChar1;
typedef VInt8_2 VChar2;
typedef VInt8_3 VChar3;
typedef VInt8_4 VChar4;

typedef VUInt8_1 VByte1;
typedef VUInt8_2 VByte2;
typedef VUInt8_3 VByte3;
typedef VUInt8_4 VByte4;

typedef VInt16_1 VShort1;
typedef VInt16_2 VShort2;
typedef VInt16_3 VShort3;
typedef VInt16_4 VShort4;

typedef VUInt16_1 VUShort1;
typedef VUInt16_2 VUShort2;
typedef VUInt16_3 VUShort3;
typedef VUInt16_4 VUShort4;

typedef VInt32_1 VInt1;
typedef VInt32_2 VInt2;
typedef VInt32_3 VInt3;
typedef VInt32_4 VInt4;

typedef VUInt32_1 VUInt1;
typedef VUInt32_2 VUInt2;
typedef VUInt32_3 VUInt3;
typedef VUInt32_4 VUInt4;

typedef VInt32_1 VLong1;
typedef VInt32_2 VLong2;
typedef VInt32_3 VLong3;
typedef VInt32_4 VLong4;

typedef VUInt32_1 VULong1;
typedef VUInt32_2 VULong2;
typedef VUInt32_3 VULong3;
typedef VUInt32_4 VULong4;

typedef VFloat32_1 VFloat1;
typedef VFloat32_2 VFloat2;
typedef VFloat32_3 VFloat3;
typedef VFloat32_4 VFloat4;

typedef VFloat64_1 VDouble1;
typedef VFloat64_2 VDouble2;
typedef VFloat64_3 VDouble3;
typedef VFloat64_4 VDouble4;

// Vector Types : 

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUDABuiltIns.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_CUDABUILTINS_H
