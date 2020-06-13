/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD low level abstraction layer, Master Header
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
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "SIMD/SIMD_Control.h"      // General SIMD Control functions

#include "SIMD/SIMD_ImportValues.h" // Values -> Registers
#include "SIMD/SIMD_ImportMemory.h" // Memory -> Registers

#include "SIMD/SIMD_ExportValues.h" // Registers -> Values
#include "SIMD/SIMD_ExportMemory.h" // Registers -> Memory

#include "SIMD/SIMD_Register.h"     // Registers <-> Registers

#include "SIMD/SIMD_Cast.h"         // TypeCasting (No generated instruction)
#include "SIMD/SIMD_Convert.h"      // Type Conversion

#include "SIMD/SIMD_Compare.h"      // Comparisons

#include "SIMD/SIMD_Bit.h"          // Bit Manipulation

#include "SIMD/SIMD_Math.h"         // Basic Arithmetics
#include "SIMD/SIMD_Function.h"     // Math Functions

// General define for SIMD use in a lot of math code, comment accordingly
#define SIMD_ENABLE // Assumes AVX2 and SSE42

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The SIMD namespace
namespace SIMD
{        


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

/////////////////////////////////////////////////////////////////////////////////
// Macro Expansions for Immediate Parameters (not an optimal solution ...)
//#define _SIMD_ARGS( ... ) __VA_ARGS__
//
//#define _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _args, _argnames, _i ) \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##0( _args ) { return _funcname( _argnames, 0x##_i##0 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##1( _args ) { return _funcname( _argnames, 0x##_i##1 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##2( _args ) { return _funcname( _argnames, 0x##_i##2 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##3( _args ) { return _funcname( _argnames, 0x##_i##3 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##4( _args ) { return _funcname( _argnames, 0x##_i##4 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##5( _args ) { return _funcname( _argnames, 0x##_i##5 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##6( _args ) { return _funcname( _argnames, 0x##_i##6 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##7( _args ) { return _funcname( _argnames, 0x##_i##7 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##8( _args ) { return _funcname( _argnames, 0x##_i##8 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##9( _args ) { return _funcname( _argnames, 0x##_i##9 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##a( _args ) { return _funcname( _argnames, 0x##_i##a ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##b( _args ) { return _funcname( _argnames, 0x##_i##b ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##c( _args ) { return _funcname( _argnames, 0x##_i##c ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##d( _args ) { return _funcname( _argnames, 0x##_i##d ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##e( _args ) { return _funcname( _argnames, 0x##_i##e ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##f( _args ) { return _funcname( _argnames, 0x##_i##f ); }
//
//#define _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, _i ) \
//    _funcname##_imm##_0x##_i##0, \
//    _funcname##_imm##_0x##_i##1, \
//    _funcname##_imm##_0x##_i##2, \
//    _funcname##_imm##_0x##_i##3, \
//    _funcname##_imm##_0x##_i##4, \
//    _funcname##_imm##_0x##_i##5, \
//    _funcname##_imm##_0x##_i##6, \
//    _funcname##_imm##_0x##_i##7, \
//    _funcname##_imm##_0x##_i##8, \
//    _funcname##_imm##_0x##_i##9, \
//    _funcname##_imm##_0x##_i##a, \
//    _funcname##_imm##_0x##_i##b, \
//    _funcname##_imm##_0x##_i##c, \
//    _funcname##_imm##_0x##_i##d, \
//    _funcname##_imm##_0x##_i##e, \
//    _funcname##_imm##_0x##_i##f
//
//#define _SIMD_DECLARE_IMMEDIATE( _rettype, _funcname, _args, _argnames ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 0 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 1 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 2 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 3 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 4 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 5 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 6 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 7 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 8 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 9 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), a ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), b ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), c ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), d ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), e ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), f ) \
//    typedef _rettype (__fastcall * _functor_##_funcname)( _args ); \
//    static _functor_##_funcname s_arrFuncTable##_##_funcname[256] = { \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 0 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 1 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 2 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 3 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 4 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 5 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 6 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 7 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 8 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 9 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, a ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, b ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, c ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, d ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, e ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, f ), \
//    };
//
//#define _SIMD_CALL_IMMEDIATE( _funcname, _args, _immValue ) \
//     s_arrFuncTable##_##_funcname[_immValue]( _args )
//
//Example usage :
//
//_SIMD_DECLARE_IMMEDIATE( __m128i, _mm_insert_epi8, _SIMD_ARGS(__m128i mDst, Int8 iSrc), _SIMD_ARGS(mDst, iSrc) )
//
//_SIMD_CALL_IMMEDIATE( _mm_insert_epi8, _SIMD_ARGS(mDst, iSrc), iIndex );
//
