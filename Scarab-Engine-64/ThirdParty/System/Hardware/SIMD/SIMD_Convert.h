/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Convert.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Convert operations
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
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDCONVERT_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDCONVERT_H

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <intrin.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../CPUID.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The SIMD namespace
namespace SIMD {
    
    // Convert
    namespace Convert {

        // Lower Element
        __forceinline __m128 OneToFloat( __m128 mDst, __m128d mSrc );  // SSE2

        __forceinline __m128 OneToFloat( __m128 mDst, Int32 iSrc ); // SSE
        __forceinline __m128 OneToFloat( __m128 mDst, Int64 iSrc ); // SSE

        __forceinline __m128d OneToDouble( __m128d mDst, __m128 mSrc ); // SSE2

        __forceinline __m128d OneToDouble( __m128d mDst, Int32 iSrc ); // SSE2
        __forceinline __m128d OneToDouble( __m128d mDst, Int64 iSrc ); // SSE2

        __forceinline Int32 OneToInt32( __m128 mSrc );  // SSE
        __forceinline Int32 OneToInt32( __m128d mSrc ); // SSE2

        __forceinline Int64 OneToInt64( __m128 mSrc );  // SSE
        __forceinline Int64 OneToInt64( __m128d mSrc ); // SSE2

        // Convert
        __forceinline __m128 ToFloat128( __m128d mSrc ); // SSE2
        __forceinline __m128 ToFloat128( __m128i mSrc ); // SSE2

        __forceinline __m128 ToFloat128( __m256d mSrc ); // AVX
        __forceinline __m256 ToFloat256( __m256i mSrc ); // AVX

        __forceinline __m128d ToDouble128( __m128 mSrc );  // SSE2
        __forceinline __m128d ToDouble128( __m128i mSrc ); // SSE2

        __forceinline __m256d ToDouble256( __m128 mSrc );  // AVX
        __forceinline __m256d ToDouble256( __m128i mSrc ); // AVX

        __forceinline __m128i ToInt32( __m128 mSrc );  // SSE2
        __forceinline __m128i ToInt32( __m128d mSrc ); // SSE2

        __forceinline __m256i ToInt32( __m256 mSrc );  // AVX
        __forceinline __m128i ToInt32( __m256d mSrc ); // AVX

    };

    // Truncate
    namespace Truncate {

        // Lower Element
        __forceinline Int32 OneToInt32( __m128 mSrc );  // SSE
        __forceinline Int32 OneToInt32( __m128d mSrc ); // SSE2

        __forceinline Int64 OneToInt64( __m128 mSrc );  // SSE
        __forceinline Int64 OneToInt64( __m128d mSrc ); // SSE2

        // Truncate
        __forceinline __m128i ToInt32( __m128 mSrc );  // SSE2
        __forceinline __m128i ToInt32( __m128d mSrc ); // SSE2

        __forceinline __m256i ToInt32( __m256 mSrc );  // AVX
        __forceinline __m128i ToInt32( __m256d mSrc ); // AVX

    };

    // Sign-Extend 128-bits
    namespace SignExtend128 {

        __forceinline __m128i Int8To16( __m128i mSrc );  // SSE41
        __forceinline __m128i Int8To32( __m128i mSrc );  // SSE41
        __forceinline __m128i Int8To64( __m128i mSrc );  // SSE41
        __forceinline __m128i Int16To32( __m128i mSrc ); // SSE41
        __forceinline __m128i Int16To64( __m128i mSrc ); // SSE41
        __forceinline __m128i Int32To64( __m128i mSrc ); // SSE41

    };

    // Sign-Extend 256-bits
    namespace SignExtend256 {

        __forceinline __m256i Int8To16( __m128i mSrc );  // AVX2
        __forceinline __m256i Int8To32( __m128i mSrc );  // AVX2
        __forceinline __m256i Int8To64( __m128i mSrc );  // AVX2
        __forceinline __m256i Int16To32( __m128i mSrc ); // AVX2
        __forceinline __m256i Int16To64( __m128i mSrc ); // AVX2
        __forceinline __m256i Int32To64( __m128i mSrc ); // AVX2

    };

    // Zero-Extend 128-bits
    namespace ZeroExtend128 {

        __forceinline __m128i Int8To16( __m128i mSrc );  // SSE41
        __forceinline __m128i Int8To32( __m128i mSrc );  // SSE41
        __forceinline __m128i Int8To64( __m128i mSrc );  // SSE41
        __forceinline __m128i Int16To32( __m128i mSrc ); // SSE41
        __forceinline __m128i Int16To64( __m128i mSrc ); // SSE41
        __forceinline __m128i Int32To64( __m128i mSrc ); // SSE41

    };

    // Zero-Extend 256-bits
    namespace ZeroExtend256 {

        __forceinline __m256i Int8To16( __m128i mSrc );  // AVX2
        __forceinline __m256i Int8To32( __m128i mSrc );  // AVX2
        __forceinline __m256i Int8To64( __m128i mSrc );  // AVX2
        __forceinline __m256i Int16To32( __m128i mSrc ); // AVX2
        __forceinline __m256i Int16To64( __m128i mSrc ); // AVX2
        __forceinline __m256i Int32To64( __m128i mSrc ); // AVX2

    };

};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SIMD_Convert.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDCONVERT_H

