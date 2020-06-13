/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Register.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Register operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : Because of STUPID FCKIN IMMEDIATE PARAMETERS, there is no clean
//              way to implement mask parameters for Shuffle & Blend ...
//              Macros are no good for a clean API !
//              Will implement specific routines as needed. For now, just start
//              with a strong basis of key combinations that are likely to
//              be the most useful ones ...
//              Shuffling variants that actually spread values are in the
//              Spread section, as the intuition suggests !
// @Microsoft : Please implement immediate specifier for function parameters,
//              and let us transmit immediate variables !!! (Should be easy !)
//              That would be the proper way to write wrappers for intrinsics !
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDREGISTER_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDREGISTER_H

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <intrin.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../CPUID.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Shuffle Control Mask
// Naming Convention : SIMD_SHUFFLE_MASK_NxP(_Z)
// N = Number of elements being picked.
// P = Number of elements to choose from.
// _Z = Has a Set to Zero flag.
#define SIMD_SHUFFLE_MASK_1x2( _element_0 ) \
    ( ((_element_0) & 0x01) << 1 )

#define SIMD_SHUFFLE_MASK_1x4( _element_0 ) \
    ( (_element_0) & 0x03 )

#define SIMD_SHUFFLE_MASK_1x8( _element_0 ) \
    ( (_element_0) & 0x07 )

#define SIMD_SHUFFLE_MASK_1x16_Z( _element_0, _zeroflag_0 ) \
    ( ((_element_0) & 0x0f) | (((_zeroflag_0) & 0x01) << 7) )

#define SIMD_SHUFFLE_MASK_2x2( _element_0, _element_1 ) \
    ( ((_element_0) & 0x01) | (((_element_1) & 0x01) << 1) )

#define SIMD_SHUFFLE_MASK_2x4_Z( _element_0, _zeroflag_0, _element_1, _zeroflag_1 ) \
    ( ((_element_0) & 0x03) | (((_zeroflag_0) & 0x01) << 3) | (((_element_1) & 0x03) << 4) | (((_zeroflag_1) & 0x01) << 7) )

#define SIMD_SHUFFLE_MASK_4x2( _element_0, _element_1, _element_2, _element_3 ) \
    ( ((_element_0) & 0x01) | (((_element_1) & 0x01) << 2) | (((_element_2) & 0x01) << 4) | (((_element_3) & 0x01) << 6) )

#define SIMD_SHUFFLE_MASK_4x4( _element_0, _element_1, _element_2, _element_3 ) \
    ( ((_element_0) & 0x03) | (((_element_1) & 0x03) << 2) | (((_element_2) & 0x03) << 4) | (((_element_3) & 0x03) << 6) )

// Blend Control Mask
#define SIMD_BLEND_MASK_2( _element_0, _element_1 ) \
    ( ((_element_0) & 0x01) | (((_element_1) & 0x01) << 1) )

#define SIMD_BLEND_MASK_4( _element_0, _element_1, _element_2, _element_3 ) \
    ( ((_element_0) & 0x01) | (((_element_1) & 0x01) << 1) | (((_element_2) & 0x01) << 2) | (((_element_3) & 0x01) << 3) )

#define SIMD_BLEND_MASK_8( _element_0, _element_1, _element_2, _element_3, _element_4, _element_5, _element_6, _element_7 ) \
    (        ((_element_0) & 0x01) | (((_element_1) & 0x01) << 1) | (((_element_2) & 0x01) << 2) | (((_element_3) & 0x01) << 3) | \
      (((_element_4) & 0x01) << 4) | (((_element_5) & 0x01) << 5) | (((_element_6) & 0x01) << 6) | (((_element_7) & 0x01) << 7) )



/////////////////////////////////////////////////////////////////////////////////
// The SIMD::Register namespace
namespace SIMD { namespace Register {

	// Move (generally less efficient than shuffling)
	namespace Move {

        // Float manipulation
        inline __m128 OneFloatLL( __m128 mDst, __m128 mSrc ); // SSE

        inline __m128 TwoFloatHL( __m128 mDst, __m128 mSrc ); // SSE
        inline __m128 TwoFloatLH( __m128 mDst, __m128 mSrc ); // SSE

        inline __m256 FourFloatL( __m256 mDst, __m128 mSrc ); // AVX
        inline __m256 FourFloatH( __m256 mDst, __m128 mSrc ); // AVX

        inline __m128 FourFloatL( __m256 mSrc ); // AVX
        inline __m128 FourFloatH( __m256 mSrc ); // AVX

        // Double manipulation
        inline __m128d OneDoubleLL( __m128d mDst, __m128d mSrc ); // SSE2

        inline __m256d TwoDoubleL( __m256d mDst, __m128d mSrc ); // AVX
        inline __m256d TwoDoubleH( __m256d mDst, __m128d mSrc ); // AVX

        inline __m128d TwoDoubleL( __m256d mSrc ); // AVX
        inline __m128d TwoDoubleH( __m256d mSrc ); // AVX

        // Int32 manipulation
        inline __m256i FourInt32L( __m256i mDst, __m128i mSrc ); // AVX2
        inline __m256i FourInt32H( __m256i mDst, __m128i mSrc ); // AVX2

        inline __m128i FourInt32L( __m256i mSrc ); // AVX2
        inline __m128i FourInt32H( __m256i mSrc ); // AVX2

        // Int64 manipulation
        inline __m128i OneInt64LL( __m128i mSrc ); // SSE2

	};

    // Pack
    namespace Pack {

        // Signed versions
        inline __m128i Int16To8( __m128i mSrcLow, __m128i mSrcHigh );   // SSE2
        inline __m128i Int32To16( __m128i mSrcLow, __m128i mSrcHigh );  // SSE2

        inline __m256i Int16To8( __m256i mSrcLow, __m256i mSrcHigh );   // AVX2
        inline __m256i Int32To16( __m256i mSrcLow, __m256i mSrcHigh );  // AVX2

        // Unsigned versions
        inline __m128i UInt16To8( __m128i mSrcLow, __m128i mSrcHigh );  // SSE2
        inline __m128i UInt32To16( __m128i mSrcLow, __m128i mSrcHigh ); // SSE41

        inline __m256i UInt16To8( __m256i mSrcLow, __m256i mSrcHigh );  // AVX2
        inline __m256i UInt32To16( __m256i mSrcLow, __m256i mSrcHigh ); // AVX2

    };

    // Unpack
    namespace Unpack {

        // Float manipulation
        inline __m128 UnpackFloatL( __m128 mSrcEven, __m128 mSrcOdd ); // SSE
        inline __m128 UnpackFloatH( __m128 mSrcEven, __m128 mSrcOdd ); // SSE

        inline __m256 UnpackFloatL( __m256 mSrcEven, __m256 mSrcOdd ); // AVX
        inline __m256 UnpackFloatH( __m256 mSrcEven, __m256 mSrcOdd ); // AVX

        // Double manipulation
        inline __m128d UnpackDoubleL( __m128d mSrcEven, __m128d mSrcOdd ); // SSE2
        inline __m128d UnpackDoubleH( __m128d mSrcEven, __m128d mSrcOdd ); // SSE2

        inline __m256d UnpackDoubleL( __m256d mSrcEven, __m256d mSrcOdd ); // AVX
        inline __m256d UnpackDoubleH( __m256d mSrcEven, __m256d mSrcOdd ); // AVX

        // Integer manipulation
        inline __m128i UnpackInt8L( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
        inline __m128i UnpackInt8H( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
        inline __m128i UnpackInt16L( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
        inline __m128i UnpackInt16H( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
        inline __m128i UnpackInt32L( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
        inline __m128i UnpackInt32H( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
        inline __m128i UnpackInt64L( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
        inline __m128i UnpackInt64H( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2

        inline __m256i UnpackInt8L( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2
        inline __m256i UnpackInt8H( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2
        inline __m256i UnpackInt16L( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2
        inline __m256i UnpackInt16H( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2
        inline __m256i UnpackInt32L( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2
        inline __m256i UnpackInt32H( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2
        inline __m256i UnpackInt64L( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2
        inline __m256i UnpackInt64H( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2

    };

    // Spread
    namespace Spread {

        // Float manipulation
        inline __m128 ABCD_AACC( __m128 mSrc ); // SSE3
        inline __m128 ABCD_BBDD( __m128 mSrc );  // SSE3
        inline __m128 ABCD_AAAA( __m128 mSrc ); // AVX2

        inline __m256 ABCDEFGH_AACCEEGG( __m256 mSrc ); // AVX
        inline __m256 ABCDEFGH_BBDDFFHH( __m256 mSrc );  // AVX
        inline __m256 ABCDEFGH_AAAAAAAA( __m256 mSrc ); // AVX2

        // Double manipulation
        inline __m128d AB_AA( __m128d mSrc ); // SSE3 or AVX2
        inline __m128d AB_BB( __m128d mSrc ); // AVX

        inline __m256d ABCD_AACC( __m256d mSrc ); // AVX
        inline __m256d ABCD_BBDD( __m256d mSrc ); // AVX2
        inline __m256d ABCD_AAAA( __m256d mSrc ); // AVX2

        // Integer manipulation
        inline __m128i Int8( __m128i mSrc ); // AVX2
        inline __m256i Int8( __m256i mSrc ); // AVX2

        inline __m128i Int16( __m128i mSrc ); // AVX2
        inline __m256i Int16( __m256i mSrc ); // AVX2

        inline __m128i Int32( __m128i mSrc ); // AVX2
        inline __m256i Int32( __m256i mSrc ); // AVX2

        inline __m128i Int64( __m128i mSrc ); // AVX2
        inline __m256i Int64( __m256i mSrc ); // AVX2

        inline __m256i Int128( __m256i mSrc ); // AVX2

    };

    // Shuffle
    namespace Shuffle {

    };

    // Blend
    namespace Blend {

    };

}; };

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SIMD_Register.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDREGISTER_H

