/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Math.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Math operations
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
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDMATH_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDMATH_H

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <intrin.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../CPUID.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// SAD Control Mask
#define SIMD_SAD_MASK( _sel_src1, _sel_src2 ) \
    ( (((_sel_src1) & 0x01) << 2) | ((_sel_src2) & 0x03) )

// DotP Control Mask
#define SIMD_DOTP_MASK_2( _element_0, _element_1, _dest_0, _dest_1 ) \
    ( (((_element_0) & 0x01) << 4) | (((_element_1) & 0x01) << 5) | \
      ((_dest_0) & 0x01) | (((_dest_1) & 0x01) << 1) )

#define SIMD_DOTP_MASK_4( _element_0, _element_1, _element_2, _element_3, _dest_0, _dest_1, _dest_2, _dest_3 ) \
    ( (((_element_0) & 0x01) << 4) | (((_element_1) & 0x01) << 5) | (((_element_2) & 0x01) << 6) | (((_element_3) & 0x01) << 7) | \
      ((_dest_0) & 0x01) | (((_dest_1) & 0x01) << 1) | (((_dest_2) & 0x01) << 2) | (((_dest_3) & 0x01) << 3) )

/////////////////////////////////////////////////////////////////////////////////
// The SIMD::Math namespace
namespace SIMD { namespace Math {

    // Floor
    __forceinline __m128 FloorOne( __m128 mDst, __m128 mSrc );    // SSE41
    __forceinline __m128d FloorOne( __m128d mDst, __m128d mSrc ); // SSE41

    __forceinline __m128 Floor( __m128 mValue );   // SSE41
    __forceinline __m128d Floor( __m128d mValue ); // SSE41

    __forceinline __m256 Floor( __m256 mValue );   // AVX
    __forceinline __m256d Floor( __m256d mValue ); // AVX

    // Ceil
    __forceinline __m128 CeilOne( __m128 mDst, __m128 mSrc );    // SSE41
    __forceinline __m128d CeilOne( __m128d mDst, __m128d mSrc ); // SSE41

    __forceinline __m128 Ceil( __m128 mValue );   // SSE41
    __forceinline __m128d Ceil( __m128d mValue ); // SSE41

    __forceinline __m256 Ceil( __m256 mValue );   // AVX
    __forceinline __m256d Ceil( __m256d mValue ); // AVX

    // Round (Nearest)
    __forceinline __m128 RoundOne( __m128 mDst, __m128 mSrc );    // SSE41
    __forceinline __m128d RoundOne( __m128d mDst, __m128d mSrc ); // SSE41

    __forceinline __m128 Round( __m128 mValue );   // SSE41
    __forceinline __m128d Round( __m128d mValue ); // SSE41

    __forceinline __m256 Round( __m256 mValue );   // AVX
    __forceinline __m256d Round( __m256d mValue ); // AVX

    // Addition
    __forceinline __m128 AddOne( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d AddOne( __m128d mDst, __m128d mSrc ); // SSE2

    __forceinline __m128 Add( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d Add( __m128d mDst, __m128d mSrc ); // SSE2

    __forceinline __m256 Add( __m256 mDst, __m256 mSrc );    // AVX
    __forceinline __m256d Add( __m256d mDst, __m256d mSrc ); // AVX

    // Horizontal Addition (generally sub-optimal)
    __forceinline __m128 HAdd( __m128 mSrc1, __m128 mSrc2 );    // SSE3
    __forceinline __m128d HAdd( __m128d mSrc1, __m128d mSrc2 ); // SSE3

    __forceinline __m256 HAdd( __m256 mSrc1, __m256 mSrc2 );    // AVX
    __forceinline __m256d HAdd( __m256d mSrc1, __m256d mSrc2 ); // AVX

    // Substraction
    __forceinline __m128 SubOne( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d SubOne( __m128d mDst, __m128d mSrc ); // SSE2

    __forceinline __m128 Sub( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d Sub( __m128d mDst, __m128d mSrc ); // SSE2

    __forceinline __m256 Sub( __m256 mDst, __m256 mSrc );    // AVX
    __forceinline __m256d Sub( __m256d mDst, __m256d mSrc ); // AVX

    // Horizontal Substraction
    __forceinline __m128 HSub( __m128 mSrc1, __m128 mSrc2 );    // SSE3
    __forceinline __m128d HSub( __m128d mSrc1, __m128d mSrc2 ); // SSE3

    __forceinline __m256 HSub( __m256 mSrc1, __m256 mSrc2 );    // AVX
    __forceinline __m256d HSub( __m256d mSrc1, __m256d mSrc2 ); // AVX

    // Interleaved Add & Sub (Sub Even / Add Odd)
    __forceinline __m128 AddSub( __m128 mDst, __m128 mSrc );    // SSE3
    __forceinline __m128d AddSub( __m128d mDst, __m128d mSrc ); // SSE3

    __forceinline __m256 AddSub( __m256 mDst, __m256 mSrc );    // AVX
    __forceinline __m256d AddSub( __m256d mDst, __m256d mSrc ); // AVX

    // Multiplication
    __forceinline __m128 MulOne( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d MulOne( __m128d mDst, __m128d mSrc ); // SSE2

    __forceinline __m128 Mul( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d Mul( __m128d mDst, __m128d mSrc ); // SSE2

    __forceinline __m256 Mul( __m256 mDst, __m256 mSrc );    // AVX
    __forceinline __m256d Mul( __m256d mDst, __m256d mSrc ); // AVX

    // Dot Product
    __forceinline __m128 Dot2( __m128 mDst, __m128 mSrc ); // SSE41
    __forceinline __m128 Dot3( __m128 mDst, __m128 mSrc ); // SSE41
    __forceinline __m128 Dot4( __m128 mDst, __m128 mSrc ); // SSE41

    __forceinline __m128d Dot2( __m128d mDst, __m128d mSrc ); // SSE41

    __forceinline __m256 Dot2( __m256 mDst, __m256 mSrc ); // AVX
    __forceinline __m256 Dot3( __m256 mDst, __m256 mSrc ); // AVX
    __forceinline __m256 Dot4( __m256 mDst, __m256 mSrc ); // AVX

    //__forceinline __m128 DotP( __m128 mDst, __m128 mSrc, Int iMask4 );    // SSE41
    //__forceinline __m128d DotP( __m128d mDst, __m128d mSrc, Int iMask2 ); // SSE41

    //__forceinline __m256 DotP( __m256 mDst, __m256 mSrc, Int iMask4 ); // AVX

    // Division
    __forceinline __m128 DivOne( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d DivOne( __m128d mDst, __m128d mSrc ); // SSE2

    __forceinline __m128 Div( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d Div( __m128d mDst, __m128d mSrc ); // SSE2

    __forceinline __m256 Div( __m256 mDst, __m256 mSrc );    // AVX
    __forceinline __m256d Div( __m256d mDst, __m256d mSrc ); // AVX

    // Minimum Value
    __forceinline __m128 MinOne( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d MinOne( __m128d mDst, __m128d mSrc ); // SSE2

    __forceinline __m128 Min( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d Min( __m128d mDst, __m128d mSrc ); // SSE2

    __forceinline __m256 Min( __m256 mDst, __m256 mSrc );    // AVX
    __forceinline __m256d Min( __m256d mDst, __m256d mSrc ); // AVX

    // Maximum Value
    __forceinline __m128 MaxOne( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d MaxOne( __m128d mDst, __m128d mSrc ); // SSE2

    __forceinline __m128 Max( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d Max( __m128d mDst, __m128d mSrc ); // SSE2

    __forceinline __m256 Max( __m256 mDst, __m256 mSrc );    // AVX
    __forceinline __m256d Max( __m256d mDst, __m256d mSrc ); // AVX

    // Signed Integer Operations
    namespace Signed {
        // Absolute Value
        __forceinline __m128i Abs8( __m128i mValue );  // SSSE3
        __forceinline __m128i Abs16( __m128i mValue ); // SSSE3
        __forceinline __m128i Abs32( __m128i mValue ); // SSSE3
        __forceinline __m128i Abs64( __m128i mValue ); // SSSE3

        __forceinline __m256i Abs8( __m256i mValue );  // AVX2
        __forceinline __m256i Abs16( __m256i mValue ); // AVX2
        __forceinline __m256i Abs32( __m256i mValue ); // AVX2
        __forceinline __m256i Abs64( __m256i mValue ); // AVX2

        // Sign Change
        __forceinline __m128i Negate8( __m128i mValue, __m128i mSigns );  // SSSE3
        __forceinline __m128i Negate16( __m128i mValue, __m128i mSigns ); // SSSE3
        __forceinline __m128i Negate32( __m128i mValue, __m128i mSigns ); // SSSE3

        __forceinline __m256i Negate8( __m256i mValue, __m256i mSigns );  // AVX2
        __forceinline __m256i Negate16( __m256i mValue, __m256i mSigns ); // AVX2
        __forceinline __m256i Negate32( __m256i mValue, __m256i mSigns ); // AVX2

        // Addition
        __forceinline __m128i Add8( __m128i mDst, __m128i mSrc );  // SSE2
        __forceinline __m128i Add16( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Add32( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Add64( __m128i mDst, __m128i mSrc ); // SSE2

        __forceinline __m256i Add8( __m256i mDst, __m256i mSrc );  // AVX2
        __forceinline __m256i Add16( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Add32( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Add64( __m256i mDst, __m256i mSrc ); // AVX2

        // Addition, Saturated
        __forceinline __m128i AddSat8( __m128i mDst, __m128i mSrc );  // SSE2
        __forceinline __m128i AddSat16( __m128i mDst, __m128i mSrc ); // SSE2

        __forceinline __m256i AddSat8( __m256i mDst, __m256i mSrc );  // AVX2
        __forceinline __m256i AddSat16( __m256i mDst, __m256i mSrc ); // AVX2

        // Horizontal Addition (generally sub-optimal)
        __forceinline __m128i HAdd16( __m128i mSrc1, __m128i mSrc2 ); // SSSE3
        __forceinline __m128i HAdd32( __m128i mSrc1, __m128i mSrc2 ); // SSSE3

        __forceinline __m256i HAdd16( __m256i mSrc1, __m256i mSrc2 ); // AVX2
        __forceinline __m256i HAdd32( __m256i mSrc1, __m256i mSrc2 ); // AVX2

        // Horizontal Addition, Saturated (generally sub-optimal)
        __forceinline __m128i HAddSat16( __m128i mSrc1, __m128i mSrc2 ); // SSSE3

        __forceinline __m256i HAddSat16( __m256i mSrc1, __m256i mSrc2 ); // AVX2

        // Substraction
        __forceinline __m128i Sub8( __m128i mDst, __m128i mSrc );  // SSE2
        __forceinline __m128i Sub16( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Sub32( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Sub64( __m128i mDst, __m128i mSrc ); // SSE2

        __forceinline __m256i Sub8( __m256i mDst, __m256i mSrc );  // AVX2
        __forceinline __m256i Sub16( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Sub32( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Sub64( __m256i mDst, __m256i mSrc ); // AVX2

        // Substraction, Saturated
        __forceinline __m128i SubSat8( __m128i mDst, __m128i mSrc );  // SSE2
        __forceinline __m128i SubSat16( __m128i mDst, __m128i mSrc ); // SSE2

        __forceinline __m256i SubSat8( __m256i mDst, __m256i mSrc );  // AVX2
        __forceinline __m256i SubSat16( __m256i mDst, __m256i mSrc ); // AVX2

        // Horizontal Substraction (generally sub-optimal)
        __forceinline __m128i HSub16( __m128i mSrc1, __m128i mSrc2 ); // SSSE3
        __forceinline __m128i HSub32( __m128i mSrc1, __m128i mSrc2 ); // SSSE3

        __forceinline __m256i HSub16( __m256i mSrc1, __m256i mSrc2 ); // AVX2
        __forceinline __m256i HSub32( __m256i mSrc1, __m256i mSrc2 ); // AVX2

        // Horizontal Substraction, Saturated (generally sub-optimal)
        __forceinline __m128i HSubSat16( __m128i mSrc1, __m128i mSrc2 ); // SSSE3

        __forceinline __m256i HSubSat16( __m256i mSrc1, __m256i mSrc2 ); // AVX2

        // Multiplication
        __forceinline __m128i Mul16L( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Mul16H( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Mul32( __m128i mDst, __m128i mSrc );  // SSE41
        __forceinline __m128i Mul32L( __m128i mDst, __m128i mSrc ); // SSE41
        __forceinline __m128i Mul64L( __m128i mDst, __m128i mSrc ); // SSE41

        __forceinline __m256i Mul16L( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Mul16H( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Mul32( __m256i mDst, __m256i mSrc );  // AVX2
        __forceinline __m256i Mul32L( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Mul64L( __m256i mDst, __m256i mSrc ); // AVX2

        // Multiply and Add
            // Signed 16-bits -> Signed 32-bits
        __forceinline __m128i MAdd( __m128i mDst, __m128i mSrc ); // SSE2

        __forceinline __m256i MAdd( __m256i mDst, __m256i mSrc ); // AVX2

            // Unsigned-Signed 8-bits -> Signed 16-bits
        __forceinline __m128i MAddUS( __m128i mDst, __m128i mSrc ); // SSSE3

        __forceinline __m256i MAddUS( __m256i mDst, __m256i mSrc ); // AVX2

        // Division
        __forceinline __m128i Div8( __m128i mDst, __m128i mSrc );  // SSE2
        __forceinline __m128i Div16( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Div32( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Div64( __m128i mDst, __m128i mSrc ); // SSE2

        __forceinline __m256i Div8( __m256i mDst, __m256i mSrc );  // AVX
        __forceinline __m256i Div16( __m256i mDst, __m256i mSrc ); // AVX
        __forceinline __m256i Div32( __m256i mDst, __m256i mSrc ); // AVX
        __forceinline __m256i Div64( __m256i mDst, __m256i mSrc ); // AVX

        // Modulo
        __forceinline __m128i Mod8( __m128i mDst, __m128i mSrc );  // SSE2
        __forceinline __m128i Mod16( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Mod32( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Mod64( __m128i mDst, __m128i mSrc ); // SSE2

        __forceinline __m256i Mod8( __m256i mDst, __m256i mSrc );  // AVX
        __forceinline __m256i Mod16( __m256i mDst, __m256i mSrc ); // AVX
        __forceinline __m256i Mod32( __m256i mDst, __m256i mSrc ); // AVX
        __forceinline __m256i Mod64( __m256i mDst, __m256i mSrc ); // AVX

        // Division and Modulo
        __forceinline __m128i DivMod32( __m128i * outMod, __m128i mDst, __m128i mSrc ); // SSE2

        __forceinline __m256i DivMod32( __m256i * outMod, __m256i mDst, __m256i mSrc ); // AVX

        // Minimum Value
        __forceinline __m128i Min8( __m128i mDst, __m128i mSrc );  // SSE41
        __forceinline __m128i Min16( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Min32( __m128i mDst, __m128i mSrc ); // SSE41
        __forceinline __m128i Min64( __m128i mDst, __m128i mSrc ); // SSE41

        __forceinline __m256i Min8( __m256i mDst, __m256i mSrc );  // AVX2
        __forceinline __m256i Min16( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Min32( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Min64( __m256i mDst, __m256i mSrc ); // AVX2

        // Maximum Value
        __forceinline __m128i Max8( __m128i mDst, __m128i mSrc );  // SSE41
        __forceinline __m128i Max16( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Max32( __m128i mDst, __m128i mSrc ); // SSE41
        __forceinline __m128i Max64( __m128i mDst, __m128i mSrc ); // SSE41

        __forceinline __m256i Max8( __m256i mDst, __m256i mSrc );  // AVX2
        __forceinline __m256i Max16( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Max32( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Max64( __m256i mDst, __m256i mSrc ); // AVX2

    };

    // Unsigned Integer Operations
    namespace Unsigned {

        // Addition, Saturated
        __forceinline __m128i AddSat8( __m128i mDst, __m128i mSrc );  // SSE2
        __forceinline __m128i AddSat16( __m128i mDst, __m128i mSrc ); // SSE2

        __forceinline __m256i AddSat8( __m256i mDst, __m256i mSrc );  // AVX2
        __forceinline __m256i AddSat16( __m256i mDst, __m256i mSrc ); // AVX2

        // Substraction, Saturated
        __forceinline __m128i SubSat8( __m128i mDst, __m128i mSrc );  // SSE2
        __forceinline __m128i SubSat16( __m128i mDst, __m128i mSrc ); // SSE2

        __forceinline __m256i SubSat8( __m256i mDst, __m256i mSrc );  // AVX2
        __forceinline __m256i SubSat16( __m256i mDst, __m256i mSrc ); // AVX2

        // SAD (Sum Absolute Differences)
        __forceinline __m128i SAD( __m128i mSrc1, __m128i mSrc2 );            // SSE2
        //__forceinline __m128i SAD( __m128i mSrc1, __m128i mSrc2, Int iMask ); // SSE41

        __forceinline __m256i SAD( __m256i mSrc1, __m256i mSrc2 );            // AVX2
        //__forceinline __m256i SAD( __m256i mSrc1, __m256i mSrc2, Int iMask ); // AVX2

        // Multiplication
        __forceinline __m128i Mul16H( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Mul32( __m128i mDst, __m128i mSrc );  // SSE2

        __forceinline __m256i Mul16H( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Mul32( __m256i mDst, __m256i mSrc );  // AVX2

        // Division
        __forceinline __m128i Div8( __m128i mDst, __m128i mSrc );  // SSE2
        __forceinline __m128i Div16( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Div32( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Div64( __m128i mDst, __m128i mSrc ); // SSE2

        __forceinline __m256i Div8( __m256i mDst, __m256i mSrc );  // AVX
        __forceinline __m256i Div16( __m256i mDst, __m256i mSrc ); // AVX
        __forceinline __m256i Div32( __m256i mDst, __m256i mSrc ); // AVX
        __forceinline __m256i Div64( __m256i mDst, __m256i mSrc ); // AVX

        // Modulo
        __forceinline __m128i Mod8( __m128i mDst, __m128i mSrc );  // SSE2
        __forceinline __m128i Mod16( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Mod32( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Mod64( __m128i mDst, __m128i mSrc ); // SSE2

        __forceinline __m256i Mod8( __m256i mDst, __m256i mSrc );  // AVX
        __forceinline __m256i Mod16( __m256i mDst, __m256i mSrc ); // AVX
        __forceinline __m256i Mod32( __m256i mDst, __m256i mSrc ); // AVX
        __forceinline __m256i Mod64( __m256i mDst, __m256i mSrc ); // AVX

        // Division and Modulo
        __forceinline __m128i DivMod32( __m128i * outMod, __m128i mDst, __m128i mSrc ); // SSE2

        __forceinline __m256i DivMod32( __m256i * outMod, __m256i mDst, __m256i mSrc ); // AVX

        // Average
        __forceinline __m128i Avg8( __m128i mDst, __m128i mSrc ); // SSE2
        __forceinline __m128i Avg16( __m128i mDst, __m128i mSrc ); // SSE2

        __forceinline __m256i Avg8( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Avg16( __m256i mDst, __m256i mSrc ); // AVX2

        // Minimum Value
        __forceinline __m128i Min8( __m128i mDst, __m128i mSrc );  // SSE2
        __forceinline __m128i Min16( __m128i mDst, __m128i mSrc ); // SSE41
        __forceinline __m128i Min32( __m128i mDst, __m128i mSrc ); // SSE41
        __forceinline __m128i Min64( __m128i mDst, __m128i mSrc ); // SSE41

        __forceinline __m256i Min8( __m256i mDst, __m256i mSrc );  // AVX2
        __forceinline __m256i Min16( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Min32( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Min64( __m256i mDst, __m256i mSrc ); // AVX2

        // Maximum Value
        __forceinline __m128i Max8( __m128i mDst, __m128i mSrc );  // SSE2
        __forceinline __m128i Max16( __m128i mDst, __m128i mSrc ); // SSE41
        __forceinline __m128i Max32( __m128i mDst, __m128i mSrc ); // SSE41
        __forceinline __m128i Max64( __m128i mDst, __m128i mSrc ); // SSE41

        __forceinline __m256i Max8( __m256i mDst, __m256i mSrc );  // AVX2
        __forceinline __m256i Max16( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Max32( __m256i mDst, __m256i mSrc ); // AVX2
        __forceinline __m256i Max64( __m256i mDst, __m256i mSrc ); // AVX2

        // CRC32
        __forceinline UInt32 CRC32( UInt32 iCRC, UInt8 iValue );  // SSE42
        __forceinline UInt32 CRC32( UInt32 iCRC, UInt16 iValue ); // SSE42
        __forceinline UInt32 CRC32( UInt32 iCRC, UInt32 iValue ); // SSE42
        __forceinline UInt64 CRC32( UInt64 iCRC, UInt64 iValue ); // SSE42

    };

}; };

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SIMD_Math.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDMATH_H

