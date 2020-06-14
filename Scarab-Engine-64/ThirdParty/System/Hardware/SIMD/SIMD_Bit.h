/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Bit.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Bit operations
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
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDBIT_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDBIT_H

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <intrin.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../CPUID.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The SIMD::Bit namespace
namespace SIMD { namespace Bit {

	// And
	__forceinline __m128 And( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d And( __m128d mDst, __m128d mSrc ); // SSE2
    __forceinline __m128i And( __m128i mDst, __m128i mSrc ); // SSE2

    __forceinline __m256 And( __m256 mDst, __m256 mSrc );    // AVX
    __forceinline __m256d And( __m256d mDst, __m256d mSrc ); // AVX
    __forceinline __m256i And( __m256i mDst, __m256i mSrc ); // AVX2

    // And Not ( not(Dst) and Src ) --- THIS IS NOT A NAND ! ---
    __forceinline __m128 AndNot( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d AndNot( __m128d mDst, __m128d mSrc ); // SSE2
    __forceinline __m128i AndNot( __m128i mDst, __m128i mSrc ); // SSE2

    __forceinline __m256 AndNot( __m256 mDst, __m256 mSrc );    // AVX
    __forceinline __m256d AndNot( __m256d mDst, __m256d mSrc ); // AVX
    __forceinline __m256i AndNot( __m256i mDst, __m256i mSrc ); // AVX2

    // Or
    __forceinline __m128 Or( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d Or( __m128d mDst, __m128d mSrc ); // SSE2
    __forceinline __m128i Or( __m128i mDst, __m128i mSrc ); // SSE2

    __forceinline __m256 Or( __m256 mDst, __m256 mSrc );    // AVX
    __forceinline __m256d Or( __m256d mDst, __m256d mSrc ); // AVX
    __forceinline __m256i Or( __m256i mDst, __m256i mSrc ); // AVX2

    // Xor
    __forceinline __m128 Xor( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d Xor( __m128d mDst, __m128d mSrc ); // SSE2
    __forceinline __m128i Xor( __m128i mDst, __m128i mSrc ); // SSE2

    __forceinline __m256 Xor( __m256 mDst, __m256 mSrc );    // AVX
    __forceinline __m256d Xor( __m256d mDst, __m256d mSrc ); // AVX
    __forceinline __m256i Xor( __m256i mDst, __m256i mSrc ); // AVX2

    // Shift : 16-bits version
    namespace Shift16 {

        __forceinline __m128i Left( __m128i mDst, Int iCount );     // SSE2
        __forceinline __m128i Left( __m128i mDst, __m128i mCount ); // SSE2

        __forceinline __m256i Left( __m256i mDst, Int iCount );     // AVX2
        __forceinline __m256i Left( __m256i mDst, __m128i mCount ); // AVX2

        __forceinline __m128i Right( __m128i mDst, Int iCount );     // SSE2
        __forceinline __m128i Right( __m128i mDst, __m128i mCount ); // SSE2

        __forceinline __m256i Right( __m256i mDst, Int iCount );     // AVX2
        __forceinline __m256i Right( __m256i mDst, __m128i mCount ); // AVX2

        // Sign-Extend versions
        __forceinline __m128i RightSE( __m128i mDst, Int iCount );     // SSE2
        __forceinline __m128i RightSE( __m128i mDst, __m128i mCount ); // SSE2

        __forceinline __m256i RightSE( __m256i mDst, Int iCount );     // AVX2
        __forceinline __m256i RightSE( __m256i mDst, __m128i mCount ); // AVX2

    };

    // Shift : 32-bits version
    namespace Shift32 {

        __forceinline __m128i Left( __m128i mDst, Int iCount );     // SSE2
        __forceinline __m128i Left( __m128i mDst, __m128i mCount ); // SSE2

        __forceinline __m256i Left( __m256i mDst, Int iCount );     // AVX2
        __forceinline __m256i Left( __m256i mDst, __m128i mCount ); // AVX2

        __forceinline __m128i Right( __m128i mDst, Int iCount );     // SSE2
        __forceinline __m128i Right( __m128i mDst, __m128i mCount ); // SSE2

        __forceinline __m256i Right( __m256i mDst, Int iCount );     // AVX2
        __forceinline __m256i Right( __m256i mDst, __m128i mCount ); // AVX2

        // Vector versions
        __forceinline __m128i LeftV( __m128i mDst, __m128i mCounts ); // AVX2
        __forceinline __m256i LeftV( __m256i mDst, __m256i mCounts ); // AVX2

        __forceinline __m128i RightV( __m128i mDst, __m128i mCounts ); // AVX2
        __forceinline __m256i RightV( __m256i mDst, __m256i mCounts ); // AVX2

        // Sign-Extend versions
        __forceinline __m128i RightSE( __m128i mDst, Int iCount );     // SSE2
        __forceinline __m128i RightSE( __m128i mDst, __m128i mCount ); // SSE2

        __forceinline __m256i RightSE( __m256i mDst, Int iCount );     // AVX2
        __forceinline __m256i RightSE( __m256i mDst, __m128i mCount ); // AVX2

        __forceinline __m128i RightVSE( __m128i mDst, __m128i mCounts ); // AVX2
        __forceinline __m256i RightVSE( __m256i mDst, __m256i mCounts ); // AVX2
    };

    // Shift : 64-bits version
    namespace Shift64 {

        __forceinline __m128i Left( __m128i mDst, Int iCount );     // SSE2
        __forceinline __m128i Left( __m128i mDst, __m128i mCount ); // SSE2

        __forceinline __m256i Left( __m256i mDst, Int iCount );     // AVX2
        __forceinline __m256i Left( __m256i mDst, __m128i mCount ); // AVX2

        __forceinline __m128i Right( __m128i mDst, Int iCount );     // SSE2
        __forceinline __m128i Right( __m128i mDst, __m128i mCount ); // SSE2

        __forceinline __m256i Right( __m256i mDst, Int iCount );     // AVX2
        __forceinline __m256i Right( __m256i mDst, __m128i mCount ); // AVX2

        // Vector versions
        __forceinline __m128i LeftV( __m128i mDst, __m128i mCounts ); // AVX2
        __forceinline __m256i LeftV( __m256i mDst, __m256i mCounts ); // AVX2

        __forceinline __m128i RightV( __m128i mDst, __m128i mCounts ); // AVX2
        __forceinline __m256i RightV( __m256i mDst, __m256i mCounts ); // AVX2

    };

    // Shift : 128-bits version
    namespace Shift128 {

        //__forceinline __m128i Left( __m128i mDst, Int iCount ); // SSE2

        //__forceinline __m128i Right( __m128i mDst, Int iCount ); // SSE2

    };

    // Shift : 256-bits version
    namespace Shift256 {

        //__forceinline __m256i Left( __m256i mDst, Int iCount ); // AVX2

        //__forceinline __m256i Right( __m256i mDst, Int iCount ); // AVX2

    };

}; };

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SIMD_Bit.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDBIT_H

