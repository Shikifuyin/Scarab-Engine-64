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
//              Spread sections, as the intuition suggests !
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
#include "SIMD_ImportMemory.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Shuffle Indices
#define SIMD_SHUFFLE_INDEX_2( _idx ) \
    (Int64)( ((_idx) & 0x01) << 1 )

#define SIMD_SHUFFLE_INDEX_4( _idx ) \
    (Int32)( (_idx) & 0x03 )

#define SIMD_SHUFFLE_INDEX_8( _idx ) \
    (Int32)( (_idx) & 0x07 )

#define SIMD_SHUFFLE_INDEX_16( _idx, _setzero ) \
    (Int8)( ((_idx) & 0x0f) | ((_setzero) << 7) )

//// Shuffle Control Mask
//// Naming Convention : SIMD_SHUFFLE_MASK_NxP(_Z)
//// N = Number of elements being picked.
//// P = Number of elements to choose from.
//// _Z = Has a Set to Zero flag.
//#define SIMD_SHUFFLE_MASK_1x2( _element_0 ) \
//    ( ((_element_0) & 0x01) << 1 )
//
//#define SIMD_SHUFFLE_MASK_1x4( _element_0 ) \
//    ( (_element_0) & 0x03 )
//
//#define SIMD_SHUFFLE_MASK_1x8( _element_0 ) \
//    ( (_element_0) & 0x07 )
//
//#define SIMD_SHUFFLE_MASK_1x16_Z( _element_0, _zeroflag_0 ) \
//    ( ((_element_0) & 0x0f) | (((_zeroflag_0) & 0x01) << 7) )
//
//#define SIMD_SHUFFLE_MASK_2x2( _element_0, _element_1 ) \
//    ( ((_element_0) & 0x01) | (((_element_1) & 0x01) << 1) )
//
//#define SIMD_SHUFFLE_MASK_2x4_Z( _element_0, _zeroflag_0, _element_1, _zeroflag_1 ) \
//    ( ((_element_0) & 0x03) | (((_zeroflag_0) & 0x01) << 3) | (((_element_1) & 0x03) << 4) | (((_zeroflag_1) & 0x01) << 7) )
//
//#define SIMD_SHUFFLE_MASK_4x2( _element_0, _element_1, _element_2, _element_3 ) \
//    ( ((_element_0) & 0x01) | (((_element_1) & 0x01) << 2) | (((_element_2) & 0x01) << 4) | (((_element_3) & 0x01) << 6) )
//
//#define SIMD_SHUFFLE_MASK_4x4( _element_0, _element_1, _element_2, _element_3 ) \
//    ( ((_element_0) & 0x03) | (((_element_1) & 0x03) << 2) | (((_element_2) & 0x03) << 4) | (((_element_3) & 0x03) << 6) )

//// Blend Control Mask
//#define SIMD_BLEND_MASK_2( _element_0, _element_1 ) \
//    ( ((_element_0) & 0x01) | (((_element_1) & 0x01) << 1) )
//
//#define SIMD_BLEND_MASK_4( _element_0, _element_1, _element_2, _element_3 ) \
//    ( ((_element_0) & 0x01) | (((_element_1) & 0x01) << 1) | (((_element_2) & 0x01) << 2) | (((_element_3) & 0x01) << 3) )
//
//#define SIMD_BLEND_MASK_8( _element_0, _element_1, _element_2, _element_3, _element_4, _element_5, _element_6, _element_7 ) \
//    (        ((_element_0) & 0x01) | (((_element_1) & 0x01) << 1) | (((_element_2) & 0x01) << 2) | (((_element_3) & 0x01) << 3) | \
//      (((_element_4) & 0x01) << 4) | (((_element_5) & 0x01) << 5) | (((_element_6) & 0x01) << 6) | (((_element_7) & 0x01) << 7) )

/////////////////////////////////////////////////////////////////////////////////
// The SIMD::Register namespace
namespace SIMD { namespace Register {

	// Move (generally less efficient than shuffling)
	namespace Move {
        
        // Float manipulation
        __forceinline __m128 OneFloatLL( __m128 mDst, __m128 mSrc ); // SSE

        __forceinline __m128 TwoFloatHL( __m128 mDst, __m128 mSrc ); // SSE
        __forceinline __m128 TwoFloatLH( __m128 mDst, __m128 mSrc ); // SSE

        __forceinline __m256 FourFloatL( __m256 mDst, __m128 mSrc ); // AVX
        __forceinline __m256 FourFloatH( __m256 mDst, __m128 mSrc ); // AVX

        __forceinline __m128 FourFloatL( __m256 mSrc ); // AVX
        __forceinline __m128 FourFloatH( __m256 mSrc ); // AVX

        // Double manipulation
        __forceinline __m128d OneDoubleLL( __m128d mDst, __m128d mSrc ); // SSE2

        __forceinline __m256d TwoDoubleL( __m256d mDst, __m128d mSrc ); // AVX
        __forceinline __m256d TwoDoubleH( __m256d mDst, __m128d mSrc ); // AVX

        __forceinline __m128d TwoDoubleL( __m256d mSrc ); // AVX
        __forceinline __m128d TwoDoubleH( __m256d mSrc ); // AVX

        // Int32 manipulation
        __forceinline __m256i FourInt32L( __m256i mDst, __m128i mSrc ); // AVX2
        __forceinline __m256i FourInt32H( __m256i mDst, __m128i mSrc ); // AVX2

        __forceinline __m128i FourInt32L( __m256i mSrc ); // AVX2
        __forceinline __m128i FourInt32H( __m256i mSrc ); // AVX2

        // Int64 manipulation
        __forceinline __m128i OneInt64LL( __m128i mSrc ); // SSE2

	};

    // Pack
    namespace Pack {

        // Signed versions
        __forceinline __m128i Int16To8( __m128i mSrcLow, __m128i mSrcHigh );   // SSE2
        __forceinline __m128i Int32To16( __m128i mSrcLow, __m128i mSrcHigh );  // SSE2

        __forceinline __m256i Int16To8( __m256i mSrcLow, __m256i mSrcHigh );   // AVX2
        __forceinline __m256i Int32To16( __m256i mSrcLow, __m256i mSrcHigh );  // AVX2

        // Unsigned versions
        __forceinline __m128i UInt16To8( __m128i mSrcLow, __m128i mSrcHigh );  // SSE2
        __forceinline __m128i UInt32To16( __m128i mSrcLow, __m128i mSrcHigh ); // SSE41

        __forceinline __m256i UInt16To8( __m256i mSrcLow, __m256i mSrcHigh );  // AVX2
        __forceinline __m256i UInt32To16( __m256i mSrcLow, __m256i mSrcHigh ); // AVX2

    };

    // Interleave
    namespace Interleave {

        // Float manipulation
        __forceinline __m128 Low( __m128 mSrcEven, __m128 mSrcOdd ); // SSE
        __forceinline __m128 High( __m128 mSrcEven, __m128 mSrcOdd ); // SSE

        __forceinline __m256 LowX2( __m256 mSrcEven, __m256 mSrcOdd ); // AVX
        __forceinline __m256 HighX2( __m256 mSrcEven, __m256 mSrcOdd ); // AVX

        // Double manipulation
        __forceinline __m128d Low( __m128d mSrcEven, __m128d mSrcOdd ); // SSE2
        __forceinline __m128d High( __m128d mSrcEven, __m128d mSrcOdd ); // SSE2

        __forceinline __m256d LowX2( __m256d mSrcEven, __m256d mSrcOdd ); // AVX
        __forceinline __m256d HighX2( __m256d mSrcEven, __m256d mSrcOdd ); // AVX

        // Integer manipulation
        __forceinline __m128i LowInt8( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
        __forceinline __m128i HighInt8( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
        __forceinline __m128i LowInt16( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
        __forceinline __m128i HighInt16( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
        __forceinline __m128i LowInt32( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
        __forceinline __m128i HighInt32( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
        __forceinline __m128i LowInt64( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
        __forceinline __m128i HighInt64( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2

        __forceinline __m256i LowInt8X2( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2
        __forceinline __m256i HighInt8X2( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2
        __forceinline __m256i LowInt16X2( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2
        __forceinline __m256i HighInt16X2( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2
        __forceinline __m256i LowInt32X2( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2
        __forceinline __m256i HighInt32X2( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2
        __forceinline __m256i LowInt64X2( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2
        __forceinline __m256i HighInt64X2( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2

    };

    // Spread over 2 Elements (AB)
    namespace Spread2 {

        // Double manipulation
        __forceinline __m128d AA( __m128d mSrc ); // SSE3 or AVX2
        __forceinline __m128d BB( __m128d mSrc ); // AVX

        // Int64 manipulation
        __forceinline __m128i AA( __m128i mSrc ); // AVX2

        // 128-bits Lane manipulation
        __forceinline __m256i AA( __m256i mSrc ); // AVX2

    };

    // Spread over 4 Elements (ABCD)
    namespace Spread4 {

        // Float manipulation
        __forceinline __m128 AAAA( __m128 mSrc ); // AVX2
        __forceinline __m128 BBBB( __m128 mSrc ); // AVX
        __forceinline __m128 CCCC( __m128 mSrc ); // AVX
        __forceinline __m128 DDDD( __m128 mSrc ); // AVX

        __forceinline __m128 AACC( __m128 mSrc ); // SSE3
        __forceinline __m128 BBDD( __m128 mSrc ); // SSE3

        // Double manipulation
        __forceinline __m256d AACC( __m256d mSrc ); // AVX
        __forceinline __m256d BBDD( __m256d mSrc ); // AVX2
        __forceinline __m256d AAAA( __m256d mSrc ); // AVX2

        // Int32 manipulation
        __forceinline __m128i AAAA( __m128i mSrc ); // AVX2

        // Int64 manipulation
        __forceinline __m256i AAAA( __m256i mSrc ); // AVX2

    };

    // Spread over 8 Elements (ABCDEFGH)
    namespace Spread8 {

        // Float manipulation
        __forceinline __m256 AACCEEGG( __m256 mSrc ); // AVX
        __forceinline __m256 BBDDFFHH( __m256 mSrc ); // AVX
        __forceinline __m256 AAAAAAAA( __m256 mSrc ); // AVX2

        // Int16 manipulation
        __forceinline __m128i AAAAAAAA( __m128i mSrc ); // AVX2

        // Int32 manipulation
        __forceinline __m256i AAAAAAAA( __m256i mSrc ); // AVX2

    };

    // Spread over 16 Elements
    namespace Spread16 {

        // Int8 manipulation
        //__forceinline __m128i Int8( __m128i mSrc );  // AVX2

        // Int16 manipulation
        //__forceinline __m256i Int16( __m256i mSrc ); // AVX2

    };

    // Spread over 32 Elements
    namespace Spread32 {

        // Int8 manipulation
        //__forceinline __m256i Int8( __m256i mSrc );  // AVX2

    };

    // Shuffle over 2 Elements (AB)
    namespace Shuffle2 {

        // Double manipulation
            // 2-Swap
        __forceinline __m128d BA( __m128d mSrc ); // AVX

        // Int64 manipulation
            // 2-Swap
        __forceinline __m128i BA( __m128i mSrc ); // AVX2

        // 128-bits Lane manipulation
            // 2-Swap
        __forceinline __m256 BA( __m256 mSrc );   // AVX
        __forceinline __m256d BA( __m256d mSrc ); // AVX
        __forceinline __m256i BA( __m256i mSrc ); // AVX2

            // Merge
        __forceinline __m256 AC( __m256 mSrcAB, __m256 mSrcCD ); // AVX
        __forceinline __m256 AD( __m256 mSrcAB, __m256 mSrcCD ); // AVX
        __forceinline __m256 BC( __m256 mSrcAB, __m256 mSrcCD ); // AVX
        __forceinline __m256 BD( __m256 mSrcAB, __m256 mSrcCD ); // AVX
        __forceinline __m256 CA( __m256 mSrcAB, __m256 mSrcCD ); // AVX
        __forceinline __m256 CB( __m256 mSrcAB, __m256 mSrcCD ); // AVX
        __forceinline __m256 DA( __m256 mSrcAB, __m256 mSrcCD ); // AVX
        __forceinline __m256 DB( __m256 mSrcAB, __m256 mSrcCD ); // AVX

    };

    // Shuffle over 4 Elements (ABCD)
    namespace Shuffle4 {

        // Float manipulation
            // 1 Swap
        __forceinline __m128 BACD( __m128 mSrc ); // AVX
        __forceinline __m128 ABDC( __m128 mSrc ); // AVX
        __forceinline __m128 ACBD( __m128 mSrc ); // AVX
        __forceinline __m128 DBCA( __m128 mSrc ); // AVX

            // Circular
        __forceinline __m128 BCDA( __m128 mSrc ); // AVX
        __forceinline __m128 CDAB( __m128 mSrc ); // AVX
        __forceinline __m128 DABC( __m128 mSrc ); // AVX

            // Mirror Circular
        __forceinline __m128 DCBA( __m128 mSrc ); // AVX
        __forceinline __m128 CBAD( __m128 mSrc ); // AVX
        __forceinline __m128 BADC( __m128 mSrc ); // AVX
        __forceinline __m128 ADCB( __m128 mSrc ); // AVX

        // Double manipulation
            // 1-Swap
        __forceinline __m256d BACD( __m256d mSrc ); // AVX
        __forceinline __m256d ABDC( __m256d mSrc ); // AVX
        __forceinline __m256d ACBD( __m256d mSrc ); // AVX2
        __forceinline __m256d DBCA( __m256d mSrc ); // AVX2

            // Circular
        __forceinline __m256d BCDA( __m256d mSrc ); // AVX2
        __forceinline __m256d CDAB( __m256d mSrc ); // AVX2
        __forceinline __m256d DABC( __m256d mSrc ); // AVX2

            // Mirror Circular
        __forceinline __m256d DCBA( __m256d mSrc ); // AVX2
        __forceinline __m256d CBAD( __m256d mSrc ); // AVX2
        __forceinline __m256d BADC( __m256d mSrc ); // AVX
        __forceinline __m256d ADCB( __m256d mSrc ); // AVX2

        // Int32 manipulation
            // 1-Swap
        __forceinline __m128i BACD( __m128i mSrc ); // SSE2
        __forceinline __m128i ABDC( __m128i mSrc ); // SSE2
        __forceinline __m128i ACBD( __m128i mSrc ); // SSE2
        __forceinline __m128i DBCA( __m128i mSrc ); // SSE2

            // Circular
        __forceinline __m128i BCDA( __m128i mSrc ); // SSE2
        __forceinline __m128i CDAB( __m128i mSrc ); // SSE2
        __forceinline __m128i DABC( __m128i mSrc ); // SSE2

            // Mirror Circular
        __forceinline __m128i DCBA( __m128i mSrc ); // SSE2
        __forceinline __m128i CBAD( __m128i mSrc ); // SSE2
        __forceinline __m128i BADC( __m128i mSrc ); // SSE2
        __forceinline __m128i ADCB( __m128i mSrc ); // SSE2

        // Int64 manipulation
            // 1-Swap
        __forceinline __m256i BACD( __m256i mSrc ); // AVX2
        __forceinline __m256i ABDC( __m256i mSrc ); // AVX2
        __forceinline __m256i ACBD( __m256i mSrc ); // AVX2
        __forceinline __m256i DBCA( __m256i mSrc ); // AVX2

            // Circular
        __forceinline __m256i BCDA( __m256i mSrc ); // AVX2
        __forceinline __m256i CDAB( __m256i mSrc ); // AVX2
        __forceinline __m256i DABC( __m256i mSrc ); // AVX2

            // Mirror Circular
        __forceinline __m256i DCBA( __m256i mSrc ); // AVX2
        __forceinline __m256i CBAD( __m256i mSrc ); // AVX2
        __forceinline __m256i BADC( __m256i mSrc ); // AVX2
        __forceinline __m256i ADCB( __m256i mSrc ); // AVX2

    };

    // Shuffle over 8 Elements (ABCDEFGH)
    namespace Shuffle8 {

        // Float manipulation
            // 1-Swap, Within 128-bits Lanes
        __forceinline __m256 BACDFEGH( __m256 mSrc ); // AVX
        __forceinline __m256 ABDCEFHG( __m256 mSrc ); // AVX
        __forceinline __m256 ACBDEGFH( __m256 mSrc ); // AVX
        __forceinline __m256 DBCAHFGE( __m256 mSrc ); // AVX

            // Circular, Within 128-bits Lanes
        __forceinline __m256 BCDAFGHE( __m256 mSrc ); // AVX
        __forceinline __m256 CDABGHEF( __m256 mSrc ); // AVX
        __forceinline __m256 DABCHEFG( __m256 mSrc ); // AVX

            // Mirror Circular, Within 128-bits Lanes
        __forceinline __m256 DCBAHGFE( __m256 mSrc ); // AVX
        __forceinline __m256 CBADGFEH( __m256 mSrc ); // AVX
        __forceinline __m256 BADCFEHG( __m256 mSrc ); // AVX
        __forceinline __m256 ADCBEHGF( __m256 mSrc ); // AVX

        // Int16 manipulation
            // 1-Swap
        __forceinline __m128i BACDEFGH( __m128i mSrc ); // SSE2
        __forceinline __m128i ABDCEFGH( __m128i mSrc ); // SSE2
        __forceinline __m128i ACBDEFGH( __m128i mSrc ); // SSE2
        __forceinline __m128i DBCAEFGH( __m128i mSrc ); // SSE2

        __forceinline __m128i ABCDFEGH( __m128i mSrc ); // SSE2
        __forceinline __m128i ABCDEFHG( __m128i mSrc ); // SSE2
        __forceinline __m128i ABCDEGFH( __m128i mSrc ); // SSE2
        __forceinline __m128i ABCDHFGE( __m128i mSrc ); // SSE2

            // Circular
        __forceinline __m128i BCDAEFGH( __m128i mSrc ); // SSE2
        __forceinline __m128i CDABEFGH( __m128i mSrc ); // SSE2
        __forceinline __m128i DABCEFGH( __m128i mSrc ); // SSE2

        __forceinline __m128i ABCDFGHE( __m128i mSrc ); // SSE2
        __forceinline __m128i ABCDGHEF( __m128i mSrc ); // SSE2
        __forceinline __m128i ABCDHEFG( __m128i mSrc ); // SSE2

            // Mirror Circular
        __forceinline __m128i DCBAEFGH( __m128i mSrc ); // SSE2
        __forceinline __m128i CBADEFGH( __m128i mSrc ); // SSE2
        __forceinline __m128i BADCEFGH( __m128i mSrc ); // SSE2
        __forceinline __m128i ADCBEFGH( __m128i mSrc ); // SSE2

        __forceinline __m128i ABCDHGFE( __m128i mSrc ); // SSE2
        __forceinline __m128i ABCDGFEH( __m128i mSrc ); // SSE2
        __forceinline __m128i ABCDFEHG( __m128i mSrc ); // SSE2
        __forceinline __m128i ABCDEHGF( __m128i mSrc ); // SSE2

        // Int32 manipulation
            // 1-Swap, Within 128-bits Lanes
        __forceinline __m256i BACDFEGH( __m256i mSrc ); // AVX2
        __forceinline __m256i ABDCEFHG( __m256i mSrc ); // AVX2
        __forceinline __m256i ACBDEGFH( __m256i mSrc ); // AVX2
        __forceinline __m256i DBCAHFGE( __m256i mSrc ); // AVX2

            // Circular, Within 128-bits Lanes
        __forceinline __m256i BCDAFGHE( __m256i mSrc ); // AVX2
        __forceinline __m256i CDABGHEF( __m256i mSrc ); // AVX2
        __forceinline __m256i DABCHEFG( __m256i mSrc ); // AVX2

            // Mirror Circular, Within 128-bits Lanes
        __forceinline __m256i DCBAHGFE( __m256i mSrc ); // AVX2
        __forceinline __m256i CBADGFEH( __m256i mSrc ); // AVX2
        __forceinline __m256i BADCFEHG( __m256i mSrc ); // AVX2
        __forceinline __m256i ADCBEHGF( __m256i mSrc ); // AVX2

    };

    // Shuffle over 16 Elements
    namespace Shuffle16 {

        // Int8 manipulation

        // Int16 manipulation

    };

    // Shuffle over 32 Elements
    namespace Shuffle32 {

        // Int8 manipulation

    };

    // Shuffle using Indices
    namespace ShuffleIndexed {

        __forceinline __m128i Make4Indices( const Int32 * arrIndices );
        __forceinline __m128 FourFloat( __m128 mSrc, __m128i mIndices4 ); // AVX

        __forceinline __m256i Make8Indices( const Int32 * arrIndices );
        __forceinline __m256 FourFloatX2( __m256 mSrc, __m256i mIndices4 );  // AVX
        __forceinline __m256 EightFloat( __m256 mSrc, __m256i mIndices8 );   // AVX2
        __forceinline __m256i EightInt32( __m256i mSrc, __m256i mIndices8 ); // AVX2

        // BEWARE THOSE MUST BE SHIFTED LEFT BY 1 !!!
        __forceinline __m128i Make2Indices( const Int64 * arrIndices );
        __forceinline __m128d TwoDouble( __m128d mSrc, __m128i mIndices2 ); // AVX

        // BEWARE THOSE MUST BE SHIFTED LEFT BY 1 !!!
        __forceinline __m256i Make4Indices( const Int64 * arrIndices );
        __forceinline __m256d TwoDoubleX2( __m256d mSrc, __m256i mIndices2 ); // AVX

        // Index Sign Bit = Set to Zero Flag
        __forceinline __m128i Make16Indices( const Int8 * arrIndices_Z );
        __forceinline __m128i SixteenInt8( __m128i mSrc, __m128i mIndices16_Z ); // SSSE3

        // Index Sign Bit = Set to Zero Flag
        __forceinline __m256i Make32Indices( const Int8 * arrIndices_Z );
        __forceinline __m256i SixteenInt8X2( __m256i mSrc, __m256i mIndices16_Z ); // AVX2

    };

    // Blend
    namespace Blend {
        //__forceinline __m128 Float( __m128 mDst, __m128 mSrc, Int iMask4 );    // SSE41
        __forceinline __m128 Float( __m128 mDst, __m128 mSrc, __m128 mSigns ); // SSE41

        //__forceinline __m128d Double( __m128d mDst, __m128d mSrc, Int iMask2 );     // SSE41
        __forceinline __m128d Double( __m128d mDst, __m128d mSrc, __m128d mSigns ); // SSE41

        __forceinline __m128i Int8( __m128i mDst, __m128i mSrc, __m128i mSigns ); // SSE41
        //__forceinline __m128i Int16( __m128i mDst, __m128i mSrc, Int iMask8 );    // SSE41
        //__forceinline __m128i Int32( __m128i mDst, __m128i mSrc, Int iMask4 );    // AVX2

        //__forceinline __m256 Float( __m256 mDst, __m256 mSrc, Int iMask8 );    // AVX
        __forceinline __m256 Float( __m256 mDst, __m256 mSrc, __m256 mSigns ); // AVX

        //__forceinline __m256d Double( __m256d mDst, __m256d mSrc, Int iMask4 );     // AVX
        __forceinline __m256d Double( __m256d mDst, __m256d mSrc, __m256d mSigns ); // AVX

        __forceinline __m256i Int8( __m256i mDst, __m256i mSrc, __m256i mSigns ); // AVX2
        //__forceinline __m256i Int16( __m256i mDst, __m256i mSrc, Int iMask8 );    // AVX2
        //__forceinline __m256i Int32( __m256i mDst, __m256i mSrc, Int iMask8 );    // AVX2
    };

}; };

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SIMD_Register.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDREGISTER_H

//    //__forceinline __m128 Shuffle128Float( __m128 mSrcLow, __m128 mSrcHigh, Int iMask4x4 ); // SSE
//#define SIMD_128_Shuffle128Float2( mSrcLow, mSrcHigh, iMask4x4 ) _mm_shuffle_ps( mSrcLow, mSrcHigh, (unsigned)iMask4x4 )
//    //__forceinline __m128 Shuffle128Float( __m128 mSrc, Int iMask4x4 );                     // AVX
//#define SIMD_128_Shuffle128Float( mSrc, iMask4x4 )               _mm_permute_ps( mSrc, iMask4x4 )
//
//    //__forceinline __m256 Shuffle128Float( __m256 mSrcLow, __m256 mSrcHigh, Int iMask4x4 ); // AVX
//#define SIMD_256_Shuffle128Float2( mSrcLow, mSrcHigh, iMask4x4 ) _mm256_shuffle_ps( mSrcLow, mSrcHigh, iMask4x4 )
//    //__forceinline __m256 Shuffle128Float( __m256 mSrc, Int iMask4x4 );                     // AVX
//#define SIMD_256_Shuffle128Float( mSrc, iMask4x4 )               _mm256_permute_ps( mSrc, iMask4x4 )
//
//
//    //__forceinline __m256 Shuffle512FourFloat( __m256 mSrc1, __m256 mSrc2, Int iMask2x4_Z ); // AVX
//#define SIMD_256_Shuffle512FourFloat( mSrc1, mSrc2, iMask2x4_Z ) _mm256_permute2f128_ps( mSrc1, mSrc2, iMask2x4_Z )
//
//    //__forceinline __m128d Shuffle128Double( __m128d mSrcLow, __m128d mSrcHigh, Int iMask2x2 ); // SSE2
//#define SIMD_128_Shuffle128Double( mSrcLow, mSrcHigh, iMask2x2 ) _mm_shuffle_pd( mSrcLow, mSrcHigh, iMask2x2 )
//
//    //__forceinline __m256d Shuffle128Double( __m256d mSrcLow, __m256d mSrcHigh, Int iMask4x2 ); // AVX
//#define SIMD_256_Shuffle128Double2( mSrcLow, mSrcHigh, iMask4x2 ) _mm256_shuffle_pd( mSrcLow, mSrcHigh, iMask4x2 )
//    //__forceinline __m256d Shuffle128Double( __m256d mSrc, Int iMask4x2 );                      // AVX
//#define SIMD_256_Shuffle128Double( mSrc, iMask4x2 )               _mm256_permute_pd( mSrc, iMask4x2 )
//
//    //__forceinline __m256d Shuffle256Double( __m256d mSrc, Int iMask4x4 ); // AVX2
//#define SIMD_256_Shuffle256Double( mSrc, iMask4x4 )              _mm256_permute4x64_pd( mSrc, iMask4x4 )
//    //__forceinline __m256d Shuffle512TwoDouble( __m256d mSrc1, __m256d mSrc2, Int iMask2x4_Z ); // AVX
//#define SIMD_256_Shuffle512TwoDouble( mSrc1, mSrc2, iMask2x4_Z ) _mm256_permute2f128_pd( mSrc1, mSrc2, iMask2x4_Z )
//
//
//    //__forceinline __m128i Shuffle64Int16L( __m128i mSrc, Int iMask4x4 ); // SSE2
//#define SIMD_128_Shuffle64Int16L( mSrc, iMask4x4 ) _mm_shufflelo_epi16( mSrc, iMask4x4 )
//    //__forceinline __m256i Shuffle64Int16L( __m256i mSrc, Int iMask4x4 ); // AVX2
//#define SIMD_256_Shuffle64Int16L( mSrc, iMask4x4 ) _mm256_shufflelo_epi16( mSrc, iMask4x4 )
//
//    //__forceinline __m128i Shuffle64Int16H( __m128i mSrc, Int iMask4x4 ); // SSE2
//#define SIMD_128_Shuffle64Int16H( mSrc, iMask4x4 ) _mm_shufflehi_epi16( mSrc, iMask4x4 )
//    //__forceinline __m256i Shuffle64Int16H( __m256i mSrc, Int iMask4x4 ); // AVX2
//#define SIMD_256_Shuffle64Int16H( mSrc, iMask4x4 ) _mm256_shufflehi_epi16( mSrc, iMask4x4 )
//
//    //__forceinline __m128i Shuffle128Int32( __m128i mSrc, Int iMask4x4 ); // SSE2
//#define SIMD_128_Shuffle128Int32( mSrc, iMask4x4 ) _mm_shuffle_epi32( mSrc, iMask4x4 )
//    //__forceinline __m256i Shuffle128Int32( __m256i mSrc, Int iMask4x4 ); // AVX2
//#define SIMD_256_Shuffle128Int32( mSrc, iMask4x4 ) _mm256_shuffle_epi32( mSrc, iMask4x4 )
//
//
//    //__forceinline __m256i Shuffle512FourInt32( __m256i mSrc1, __m256i mSrc2, Int iMask2x4_Z ); // AVX2
//#define SIMD_256_Shuffle512FourInt32( mSrc1, mSrc2, iMask2x4_Z ) _mm256_permute2x128_si256( mSrc1, mSrc2, iMask2x4_Z )
//
//    //__forceinline __m256i Shuffle256Int64( __m256i mSrc, Int iMask4x4 ); // AVX2
//#define SIMD_256_Shuffle256Int64( mSrc, iMask4x4 ) _mm256_permute4x64_epi64( mSrc, iMask4x4 )

//__forceinline __m128 SIMD::Shuffle128Float( __m128 mSrcLow, __m128 mSrcHigh, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasSSE() );
//    return _mm_shuffle_ps( mSrcLow, mSrcHigh, (unsigned)iMask4x4 );
//}
//__forceinline __m128 SIMD::Shuffle128Float( __m128 mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm_permute_ps( mSrc, iMask4x4 );
//}


//__forceinline __m256 SIMD::Shuffle128Float( __m256 mSrcLow, __m256 mSrcHigh, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_shuffle_ps( mSrcLow, mSrcHigh, iMask4x4 );
//}
//__forceinline __m256 SIMD::Shuffle128Float( __m256 mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_permute_ps( mSrc, iMask4x4 );
//}




//__forceinline __m256 SIMD::Shuffle512FourFloat( __m256 mSrc1, __m256 mSrc2, Int iMask2x4_Z ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_permute2f128_ps( mSrc1, mSrc2, iMask2x4_Z );
//}

//__forceinline __m128d SIMD::Shuffle128Double( __m128d mSrcLow, __m128d mSrcHigh, Int iMask2x2 ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_shuffle_pd( mSrcLow, mSrcHigh, iMask2x2 );
//}


//__forceinline __m256d SIMD::Shuffle128Double( __m256d mSrcLow, __m256d mSrcHigh, Int iMask4x2 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_shuffle_pd( mSrcLow, mSrcHigh, iMask4x2 );
//}
//__forceinline __m256d SIMD::Shuffle128Double( __m256d mSrc, Int iMask4x2 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_permute_pd( mSrc, iMask4x2 );
//}


//__forceinline __m256d SIMD::Shuffle256Double( __m256d mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_permute4x64_pd( mSrc, iMask4x4 );
//}

//__forceinline __m256d SIMD::Shuffle512TwoDouble( __m256d mSrc1, __m256d mSrc2, Int iMask2x4_Z ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_permute2f128_pd( mSrc1, mSrc2, iMask2x4_Z );
//}




//__forceinline __m128i SIMD::Shuffle64Int16L( __m128i mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_shufflelo_epi16( mSrc, iMask4x4 );
//}
//__forceinline __m256i SIMD::Shuffle64Int16L( __m256i mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_shufflelo_epi16( mSrc, iMask4x4 );
//}

//__forceinline __m128i SIMD::Shuffle64Int16H( __m128i mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_shufflehi_epi16( mSrc, iMask4x4 );
//}
//__forceinline __m256i SIMD::Shuffle64Int16H( __m256i mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_shufflehi_epi16( mSrc, iMask4x4 );
//}

//__forceinline __m128i SIMD::Shuffle128Int32( __m128i mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_shuffle_epi32( mSrc, iMask4x4 );
//}
//__forceinline __m256i SIMD::Shuffle128Int32( __m256i mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_shuffle_epi32( mSrc, iMask4x4 );
//}



//__forceinline __m256i SIMD::Shuffle512FourInt32( __m256i mSrc1, __m256i mSrc2, Int iMask2x4_Z ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_permute2x128_si256( mSrc1, mSrc2, iMask2x4_Z );
//}

//__forceinline __m256i SIMD::Shuffle256Int64( __m256i mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_permute4x64_epi64( mSrc, iMask4x4 );
//}


