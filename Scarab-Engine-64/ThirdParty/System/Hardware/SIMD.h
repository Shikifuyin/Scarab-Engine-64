/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD low level abstraction layer
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : Unfortunately, there is no way to completely hide third-party
//              headers this time ... we have to make everything inline ...
//              Also compiler will require direct usage of intrinsic types.
//
// You should prefer AVX/AVX2 instructions when available !
//
// Also ... STUPID FCKIN IMMEDIATE PARAMETERS !!!
// Please implement a way to declare a function parameter as immediate !!!
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_H

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <intrin.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CPUID.h"

// General define for SIMD use in a lot of math code, comment accordingly
#define SIMD_ENABLE // Assumes AVX2 and SSE42

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
// The SIMD namespace
namespace SIMD
{        


    ////////////////////////////////////////////////////////////// Registers <-> Registers
        // Move
            // Dst argument : Unaffected elements are copied
    inline __m128 MoveOneFloatLL( __m128 mDst, __m128 mSrc ); // SSE

    inline __m128 MoveTwoFloatHL( __m128 mDst, __m128 mSrc ); // SSE
    inline __m128 MoveTwoFloatLH( __m128 mDst, __m128 mSrc ); // SSE

    inline __m256 MoveFourFloatL( __m256 mDst, __m128 mSrc ); // AVX
    inline __m256 MoveFourFloatH( __m256 mDst, __m128 mSrc ); // AVX

    inline __m128d MoveOneDoubleLL( __m128d mDst, __m128d mSrc ); // SSE2

    inline __m256d MoveTwoDoubleL( __m256d mDst, __m128d mSrc ); // AVX
    inline __m256d MoveTwoDoubleH( __m256d mDst, __m128d mSrc ); // AVX

    inline __m256i MoveFourIntL( __m256i mDst, __m128i mSrc ); // AVX2
    inline __m256i MoveFourIntH( __m256i mDst, __m128i mSrc ); // AVX2

            // No Dst argument : Unaffected elements are zeroed
    inline __m128 MoveFourFloatL( __m256 mSrc ); // AVX
    inline __m128 MoveFourFloatH( __m256 mSrc ); // AVX

    inline __m128d MoveTwoDoubleL( __m256d mSrc ); // AVX
    inline __m128d MoveTwoDoubleH( __m256d mSrc ); // AVX

    inline __m128i MoveFourIntL( __m256i mSrc ); // AVX2
    inline __m128i MoveFourIntH( __m256i mSrc ); // AVX2

    inline __m128i MoveOneInt64LL( __m128i mSrc ); // SSE2

        // Spread
    inline __m128 SpreadTwoFloatEven( __m128 mSrc ); // SSE3
    inline __m128 SpreadTwoFloatOdd( __m128 mSrc );  // SSE3
    inline __m128d SpreadOneDoubleL( __m128d mSrc ); // SSE3

    inline __m256 SpreadFourFloatEven( __m256 mSrc );   // AVX
    inline __m256 SpreadFourFloatOdd( __m256 mSrc );    // AVX
    inline __m256d SpreadTwoDoubleEven( __m256d mSrc ); // AVX

    inline __m128 Spread128Float( __m128 mSrc ); // AVX2
    inline __m256 Spread256Float( __m128 mSrc ); // AVX2

    inline __m128d Spread128Double( __m128d mSrc ); // AVX2
    inline __m256d Spread256Double( __m128d mSrc ); // AVX2

    inline __m128i Spread128Int8( __m128i mSrc ); // AVX2
    inline __m256i Spread256Int8( __m128i mSrc ); // AVX2

    inline __m128i Spread128Int16( __m128i mSrc ); // AVX2
    inline __m256i Spread256Int16( __m128i mSrc ); // AVX2

    inline __m128i Spread128Int32( __m128i mSrc ); // AVX2
    inline __m256i Spread256Int32( __m128i mSrc ); // AVX2

    inline __m128i Spread128Int64( __m128i mSrc ); // AVX2
    inline __m256i Spread256Int64( __m128i mSrc ); // AVX2

    inline __m256i Spread256Int128( __m128i mSrc ); // AVX2

    ////////////////////////////////////////////////////////////// Pack / Unpack
    inline __m128i PackSigned16To8( __m128i mSrcLow, __m128i mSrcHigh ); // SSE2
    inline __m256i PackSigned16To8( __m256i mSrcLow, __m256i mSrcHigh ); // AVX2

    inline __m128i PackSigned32To16( __m128i mSrcLow, __m128i mSrcHigh ); // SSE2
    inline __m256i PackSigned32To16( __m256i mSrcLow, __m256i mSrcHigh ); // AVX2

    inline __m128i PackUnsigned16To8( __m128i mSrcLow, __m128i mSrcHigh ); // SSE2
    inline __m256i PackUnsigned16To8( __m256i mSrcLow, __m256i mSrcHigh ); // AVX2

    inline __m128i PackUnsigned32To16( __m128i mSrcLow, __m128i mSrcHigh ); // SSE41
    inline __m256i PackUnsigned32To16( __m256i mSrcLow, __m256i mSrcHigh ); // AVX2

    inline __m128 UnpackFloatL( __m128 mSrcEven, __m128 mSrcOdd ); // SSE
    inline __m256 UnpackFloatL( __m256 mSrcEven, __m256 mSrcOdd ); // AVX

    inline __m128 UnpackFloatH( __m128 mSrcEven, __m128 mSrcOdd ); // SSE
    inline __m256 UnpackFloatH( __m256 mSrcEven, __m256 mSrcOdd ); // AVX

    inline __m128d UnpackDoubleL( __m128d mSrcEven, __m128d mSrcOdd ); // SSE2
    inline __m256d UnpackDoubleL( __m256d mSrcEven, __m256d mSrcOdd ); // AVX

    inline __m128d UnpackDoubleH( __m128d mSrcEven, __m128d mSrcOdd ); // SSE2
    inline __m256d UnpackDoubleH( __m256d mSrcEven, __m256d mSrcOdd ); // AVX

    inline __m128i UnpackInt8L( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
    inline __m256i UnpackInt8L( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2

    inline __m128i UnpackInt8H( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
    inline __m256i UnpackInt8H( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2

    inline __m128i UnpackInt16L( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
    inline __m256i UnpackInt16L( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2

    inline __m128i UnpackInt16H( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
    inline __m256i UnpackInt16H( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2

    inline __m128i UnpackInt32L( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
    inline __m256i UnpackInt32L( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2

    inline __m128i UnpackInt32H( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
    inline __m256i UnpackInt32H( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2

    inline __m128i UnpackInt64L( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
    inline __m256i UnpackInt64L( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2

    inline __m128i UnpackInt64H( __m128i mSrcEven, __m128i mSrcOdd ); // SSE2
    inline __m256i UnpackInt64H( __m256i mSrcEven, __m256i mSrcOdd ); // AVX2

    ////////////////////////////////////////////////////////////// Shuffle
    //inline __m128 Shuffle128Float( __m128 mSrcLow, __m128 mSrcHigh, Int iMask4x4 ); // SSE
#define SIMD_128_Shuffle128Float2( mSrcLow, mSrcHigh, iMask4x4 ) _mm_shuffle_ps( mSrcLow, mSrcHigh, (unsigned)iMask4x4 )
    //inline __m128 Shuffle128Float( __m128 mSrc, Int iMask4x4 );                     // AVX
#define SIMD_128_Shuffle128Float( mSrc, iMask4x4 )               _mm_permute_ps( mSrc, iMask4x4 )
    inline __m128 Shuffle128Float( __m128 mSrc, __m128i mMask1x4 );                 // AVX

    //inline __m256 Shuffle128Float( __m256 mSrcLow, __m256 mSrcHigh, Int iMask4x4 ); // AVX
#define SIMD_256_Shuffle128Float2( mSrcLow, mSrcHigh, iMask4x4 ) _mm256_shuffle_ps( mSrcLow, mSrcHigh, iMask4x4 )
    //inline __m256 Shuffle128Float( __m256 mSrc, Int iMask4x4 );                     // AVX
#define SIMD_256_Shuffle128Float( mSrc, iMask4x4 )               _mm256_permute_ps( mSrc, iMask4x4 )
    inline __m256 Shuffle128Float( __m256 mSrc, __m256i mMask1x4 );                 // AVX

    inline __m256 Shuffle256Float( __m256 mSrc, __m256i mMask1x8 ); // AVX2

    //inline __m256 Shuffle512FourFloat( __m256 mSrc1, __m256 mSrc2, Int iMask2x4_Z ); // AVX
#define SIMD_256_Shuffle512FourFloat( mSrc1, mSrc2, iMask2x4_Z ) _mm256_permute2f128_ps( mSrc1, mSrc2, iMask2x4_Z )

    //inline __m128d Shuffle128Double( __m128d mSrcLow, __m128d mSrcHigh, Int iMask2x2 ); // SSE2
#define SIMD_128_Shuffle128Double( mSrcLow, mSrcHigh, iMask2x2 ) _mm_shuffle_pd( mSrcLow, mSrcHigh, iMask2x2 )
    inline __m128d Shuffle128Double( __m128d mSrc, Int iMask2x2 );                      // AVX
    inline __m128d Shuffle128Double( __m128d mSrc, __m128i mMask1x2 );                  // AVX

    //inline __m256d Shuffle128Double( __m256d mSrcLow, __m256d mSrcHigh, Int iMask4x2 ); // AVX
#define SIMD_256_Shuffle128Double2( mSrcLow, mSrcHigh, iMask4x2 ) _mm256_shuffle_pd( mSrcLow, mSrcHigh, iMask4x2 )
    //inline __m256d Shuffle128Double( __m256d mSrc, Int iMask4x2 );                      // AVX
#define SIMD_256_Shuffle128Double( mSrc, iMask4x2 )               _mm256_permute_pd( mSrc, iMask4x2 )
    inline __m256d Shuffle128Double( __m256d mSrc, __m256i mMask1x2 );                  // AVX

    //inline __m256d Shuffle256Double( __m256d mSrc, Int iMask4x4 ); // AVX2
#define SIMD_256_Shuffle256Double( mSrc, iMask4x4 )              _mm256_permute4x64_pd( mSrc, iMask4x4 )
    //inline __m256d Shuffle512TwoDouble( __m256d mSrc1, __m256d mSrc2, Int iMask2x4_Z ); // AVX
#define SIMD_256_Shuffle512TwoDouble( mSrc1, mSrc2, iMask2x4_Z ) _mm256_permute2f128_pd( mSrc1, mSrc2, iMask2x4_Z )

    inline __m128i Shuffle128Int8( __m128i mSrc, __m128i mMask1x16_Z ); // SSSE3
    inline __m256i Shuffle128Int8( __m256i mSrc, __m256i mMask1x16_Z ); // AVX2

    //inline __m128i Shuffle64Int16L( __m128i mSrc, Int iMask4x4 ); // SSE2
#define SIMD_128_Shuffle64Int16L( mSrc, iMask4x4 ) _mm_shufflelo_epi16( mSrc, iMask4x4 )
    //inline __m256i Shuffle64Int16L( __m256i mSrc, Int iMask4x4 ); // AVX2
#define SIMD_256_Shuffle64Int16L( mSrc, iMask4x4 ) _mm256_shufflelo_epi16( mSrc, iMask4x4 )

    //inline __m128i Shuffle64Int16H( __m128i mSrc, Int iMask4x4 ); // SSE2
#define SIMD_128_Shuffle64Int16H( mSrc, iMask4x4 ) _mm_shufflehi_epi16( mSrc, iMask4x4 )
    //inline __m256i Shuffle64Int16H( __m256i mSrc, Int iMask4x4 ); // AVX2
#define SIMD_256_Shuffle64Int16H( mSrc, iMask4x4 ) _mm256_shufflehi_epi16( mSrc, iMask4x4 )

    //inline __m128i Shuffle128Int32( __m128i mSrc, Int iMask4x4 ); // SSE2
#define SIMD_128_Shuffle128Int32( mSrc, iMask4x4 ) _mm_shuffle_epi32( mSrc, iMask4x4 )
    //inline __m256i Shuffle128Int32( __m256i mSrc, Int iMask4x4 ); // AVX2
#define SIMD_256_Shuffle128Int32( mSrc, iMask4x4 ) _mm256_shuffle_epi32( mSrc, iMask4x4 )

    inline __m256i Shuffle256Int32( __m256i mSrc, __m256i mMask1x8 ); // AVX2

    //inline __m256i Shuffle512FourInt32( __m256i mSrc1, __m256i mSrc2, Int iMask2x4_Z ); // AVX2
#define SIMD_256_Shuffle512FourInt32( mSrc1, mSrc2, iMask2x4_Z ) _mm256_permute2x128_si256( mSrc1, mSrc2, iMask2x4_Z )

    //inline __m256i Shuffle256Int64( __m256i mSrc, Int iMask4x4 ); // AVX2
#define SIMD_256_Shuffle256Int64( mSrc, iMask4x4 ) _mm256_permute4x64_epi64( mSrc, iMask4x4 )

    ////////////////////////////////////////////////////////////// Blend
    //inline __m128 BlendFloat( __m128 mDst, __m128 mSrc, Int iMask4 );    // SSE41
    inline __m128 BlendFloat( __m128 mDst, __m128 mSrc, __m128 mSigns ); // SSE41
    //inline __m256 BlendFloat( __m256 mDst, __m256 mSrc, Int iMask8 );    // AVX
    inline __m256 BlendFloat( __m256 mDst, __m256 mSrc, __m256 mSigns ); // AVX

    //inline __m128d BlendDouble( __m128d mDst, __m128d mSrc, Int iMask2 );     // SSE41
    inline __m128d BlendDouble( __m128d mDst, __m128d mSrc, __m128d mSigns ); // SSE41
    //inline __m256d BlendDouble( __m256d mDst, __m256d mSrc, Int iMask4 );     // AVX
    inline __m256d BlendDouble( __m256d mDst, __m256d mSrc, __m256d mSigns ); // AVX

    inline __m128i BlendInt8( __m128i mDst, __m128i mSrc, __m128i mSigns ); // SSE41
    inline __m256i BlendInt8( __m256i mDst, __m256i mSrc, __m256i mSigns ); // AVX2

    //inline __m128i BlendInt16( __m128i mDst, __m128i mSrc, Int iMask8 ); // SSE41
    //inline __m256i BlendInt16( __m256i mDst, __m256i mSrc, Int iMask8 ); // AVX2

    //inline __m128i BlendInt32( __m128i mDst, __m128i mSrc, Int iMask4 ); // AVX2
    //inline __m256i BlendInt32( __m256i mDst, __m256i mSrc, Int iMask8 ); // AVX2

};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SIMD.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_H

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Design Choice : DO NOT USE SETR ! It's just confusing ... useless at best, bug mayhem in most cases.

// __m128 _mm_setr_ps(float, float, float, float)                                       - SSE
// __m256 _mm256_setr_ps(float, float, float, float, float, float, float, float)        - AVX
// __m128d _mm_setr_pd(double, double)                                                  - SSE2
// __m256d _mm256_setr_pd(double, double, double, double)                               - AVX
// __m128i _mm_setr_epi8(char, char, char, char, char, char, char, char,
//                       char, char, char, char, char, char, char, char)                - SSE2
// __m128i _mm_setr_epi16(short, short, short, short, short, short, short, short)       - SSE2
// __m128i _mm_setr_epi32(int, int, int, int)                                           - SSE2
// __m128i _mm_setr_epi64(int64, int64)                                                 - SSE2
// __m256i _mm256_setr_epi8(char, char, char, char, char, char, char, char,
//                          char, char, char, char, char, char, char, char,
//                          char, char, char, char, char, char, char, char,
//                          char, char, char, char, char, char, char, char)             - AVX
// __m256i _mm256_setr_epi16(short, short, short, short, short, short, short, short,
//                           short, short, short, short, short, short, short, short )   - AVX
// __m256i _mm256_setr_epi32(int, int, int, int, int, int, int, int)                    - AVX
// __m256i _mm256_setr_epi64(int64, int64, int64, int64)                                - AVX

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Deprecated stuff

// __m128i _mm_setl_epi64(__m128i)      - SSE2

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Leftover misc/weird instructions ... most likely introduced for very specific optimizations
// Will add them if ever needed ...

// void _mm_stream_si32(int*, int)                  - SSE2

// __m128i _mm_insert_si64(__m128i, __m128i)                - SSE4a
// __m128i _mm_inserti_si64(__m128i, __m128i, int, int)     - SSE4a
// __m128 _mm_insert_ps(__m128, __m128, const int)          - SSE41

// void _mm_maskmoveu_si128(__m128i, __m128i, char*)    - SSE2

// int _mm_movemask_ps(__m128)          - SSE
// int _mm_movemask_epi8(__m128i)       - SSE2
// int _mm_movemask_pd(__m128d)         - SSE2
// int _mm256_movemask_pd(__m256d)      - AVX
// int _mm256_movemask_ps(__m256)       - AVX
// int _mm256_movemask_epi8(__m256i)    - AVX2

// __m128i _mm_mulhrs_epi16(__m128i, __m128i)       - SSSE3
// __m256i _mm256_mulhrs_epi16(__m256i, __m256i)    - AVX2

// __m128i _mm_minpos_epu16(__m128i)    - SSE41

// __m128i _mm_alignr_epi8(__m128i, __m128i, int)         - SSSE3
// __m256i _mm256_alignr_epi8(__m256i, __m256i, const int)  - AVX2

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Unimplemented stuff ... Most likely will never use those ...

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPESTR
// int _mm_cmpestra(__m128i, int, __m128i, int, const int)      - SSE42
// int _mm_cmpestrc(__m128i, int, __m128i, int, const int)      - SSE42
// int _mm_cmpestri(__m128i, int, __m128i, int, const int)      - SSE42
// __m128i _mm_cmpestrm(__m128i, int, __m128i, int, const int)  - SSE42
// int _mm_cmpestro(__m128i, int, __m128i, int, const int)      - SSE42
// int _mm_cmpestrs(__m128i, int, __m128i, int, const int)      - SSE42
// int _mm_cmpestrz(__m128i, int, __m128i, int, const int)      - SSE42
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPISTR
// int _mm_cmpistra(__m128i, __m128i, const int)                - SSE42
// int _mm_cmpistrc(__m128i, __m128i, const int)                - SSE42
// int _mm_cmpistri(__m128i, __m128i, const int)                - SSE42
// __m128i _mm_cmpistrm(__m128i, __m128i, const int)            - SSE42
// int _mm_cmpistro(__m128i, __m128i, const int)                - SSE42
// int _mm_cmpistrs(__m128i, __m128i, const int)                - SSE42
// int _mm_cmpistrz(__m128i, __m128i, const int)                - SSE42

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// TESTC
// int _mm_testc_si128(__m128i, __m128i)        - SSE41
// int _mm_testc_pd(__m128d, __m128d)           - AVX
// int _mm_testc_ps(__m128, __m128)             - AVX
// int _mm256_testc_pd(__m256d, __m256d)        - AVX
// int _mm256_testc_ps(__m256, __m256)          - AVX
// int _mm256_testc_si256(__m256i, __m256i)     - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// TESTZ
// int _mm_testz_si128(__m128i, __m128i)        - SSE41
// int _mm_testz_pd(__m128d, __m128d)           - AVX
// int _mm_testz_ps(__m128, __m128)             - AVX
// int _mm256_testz_pd(__m256d, __m256d)        - AVX
// int _mm256_testz_ps(__m256, __m256)          - AVX
// int _mm256_testz_si256(__m256i, __m256i)     - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// TESTNZC
// int _mm_testnzc_si128(__m128i, __m128i)      - SSE41
// int _mm_testnzc_pd(__m128d, __m128d)         - AVX
// int _mm_testnzc_ps(__m128, __m128)           - AVX
// int _mm256_testnzc_pd(__m256d, __m256d)      - AVX
// int _mm256_testnzc_ps(__m256, __m256)        - AVX
// int _mm256_testnzc_si256(__m256i, __m256i)   - AVX

