/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Cast.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Cast operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : Compiler Hints only, No instruction generated, Free !
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDCAST_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDCAST_H

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <intrin.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../CPUID.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The SIMD::Cast namespace
namespace SIMD { namespace Cast {

	__forceinline __m128 ToFloat( __m128d mDouble );  // SSE2
    __forceinline __m128 ToFloat( __m128i mInteger ); // SSE2
    __forceinline __m256 ToFloat( __m256d mDouble );  // AVX
    __forceinline __m256 ToFloat( __m256i mInteger ); // AVX

    __forceinline __m128d ToDouble( __m128 mFloat );    // SSE2
    __forceinline __m128d ToDouble( __m128i mInteger ); // SSE2
    __forceinline __m256d ToDouble( __m256 mFloat );    // AVX
    __forceinline __m256d ToDouble( __m256i mInteger ); // AVX

    __forceinline __m128i ToInteger( __m128 mFloat );   // SSE2
    __forceinline __m128i ToInteger( __m128d mDouble ); // SSE2
    __forceinline __m256i ToInteger( __m256 mFloat );   // AVX
    __forceinline __m256i ToInteger( __m256d mDouble ); // AVX

    __forceinline __m128 Down( __m256 mFloat );     // AVX
    __forceinline __m128d Down( __m256d mDouble );  // AVX
    __forceinline __m128i Down( __m256i mInteger ); // AVX

    __forceinline __m256 Up( __m128 mFloat );     // AVX
    __forceinline __m256d Up( __m128d mDouble );  // AVX
    __forceinline __m256i Up( __m128i mInteger ); // AVX

}; };

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SIMD_Cast.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDCAST_H

