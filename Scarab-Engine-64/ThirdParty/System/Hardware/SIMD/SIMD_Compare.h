/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Compare.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Compare operations
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
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDCOMPARE_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDCOMPARE_H

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <intrin.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../CPUID.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Comparison Modes
//#define SIMD_CMP_TRUE_UQ    _CMP_TRUE_UQ    // True (unordered, quiet)
//#define SIMD_CMP_TRUE_US    _CMP_TRUE_US    // True (unordered, signaling)
//
//#define SIMD_CMP_FALSE_OQ   _CMP_FALSE_OQ   // False (ordered, quiet)
//#define SIMD_CMP_FALSE_OS   _CMP_FALSE_OS   // False (ordered, signaling)
//
//#define SIMD_CMP_ORD_Q      _CMP_ORD_Q      // Ordered (quiet)
//#define SIMD_CMP_ORD_S      _CMP_ORD_S      // Ordered (signaling)
//
//#define SIMD_CMP_UNORD_Q    _CMP_UNORD_Q    // Unordered (quiet)
//#define SIMD_CMP_UNORD_S    _CMP_UNORD_S    // Unordered (signaling)
//
//#define SIMD_CMP_EQ_UQ      _CMP_EQ_UQ      // Equal (unordered, quiet)
//#define SIMD_CMP_EQ_US      _CMP_EQ_US      // Equal (unordered, signaling)
//#define SIMD_CMP_EQ_OQ      _CMP_EQ_OQ      // Equal (ordered, quiet)
//#define SIMD_CMP_EQ_OS      _CMP_EQ_OS      // Equal (ordered, signaling)
//
//#define SIMD_CMP_NEQ_UQ     _CMP_NEQ_UQ     // Not Equal (unordered, quiet)
//#define SIMD_CMP_NEQ_US     _CMP_NEQ_US     // Not Equal (unordered, signaling)
//#define SIMD_CMP_NEQ_OQ     _CMP_NEQ_OQ     // Not Equal (ordered, quiet)
//#define SIMD_CMP_NEQ_OS     _CMP_NEQ_OS     // Not Equal (ordered, signaling)
//
//#define SIMD_CMP_LT_OQ      _CMP_LT_OQ      // Lesser-Than (ordered, quiet)
//#define SIMD_CMP_LT_OS      _CMP_LT_OS      // Lesser-Than (ordered, signaling)
//
//#define SIMD_CMP_NLT_UQ     _CMP_NLT_UQ     // Not Lesser-Than (unordered, quiet)
//#define SIMD_CMP_NLT_US     _CMP_NLT_US     // Not Lesser-Than (unordered, signaling)
//
//#define SIMD_CMP_LE_OQ      _CMP_LE_OQ      // Lesser-or-Equal (ordered, quiet)
//#define SIMD_CMP_LE_OS      _CMP_LE_OS      // Lesser-or-Equal (ordered, signaling)
//
//#define SIMD_CMP_NLE_UQ     _CMP_NLE_UQ     // Not Lesser-or-Equal (unordered, quiet)
//#define SIMD_CMP_NLE_US     _CMP_NLE_US     // Not Lesser-or-Equal (unordered, signaling)
//
//#define SIMD_CMP_GT_OQ      _CMP_GT_OQ      // Greater-Than (ordered, quiet)
//#define SIMD_CMP_GT_OS      _CMP_GT_OS      // Greater-Than (ordered, signaling)
//
//#define SIMD_CMP_NGT_UQ     _CMP_NGT_UQ     // Not Greater-Than (unordered, quiet)
//#define SIMD_CMP_NGT_US     _CMP_NGT_US     // Not Greater-Than (unordered, signaling)
//
//#define SIMD_CMP_GE_OQ      _CMP_GE_OQ      // Greater-or-Equal (ordered, quiet)
//#define SIMD_CMP_GE_OS      _CMP_GE_OS      // Greater-or-Equal (ordered, signaling)
//
//#define SIMD_CMP_NGE_UQ     _CMP_NGE_UQ     // Not Greater-or-Equal (unordererd, quiet)
//#define SIMD_CMP_NGE_US     _CMP_NGE_US     // Not Greater-or-Equal (unordererd, signaling)

/////////////////////////////////////////////////////////////////////////////////
// The SIMD::Compare namespace
namespace SIMD { namespace Compare {

    // Equal
    inline __m128 Equal( __m128 mDst, __m128 mSrc ); // SSE

    inline __m128d Equal( __m128d mDst, __m128d mSrc ); // SSE2

    inline __m128i Equal8( __m128i mDst, __m128i mSrc );  // SSE2
    inline __m128i Equal16( __m128i mDst, __m128i mSrc ); // SSE2
    inline __m128i Equal32( __m128i mDst, __m128i mSrc ); // SSE2
    inline __m128i Equal64( __m128i mDst, __m128i mSrc ); // SSE41

    inline __m256 Equal( __m256 mDst, __m256 mSrc ); // AVX

    inline __m256d Equal( __m256d mDst, __m256d mSrc ); // AVX

    inline __m256i Equal8( __m256i mDst, __m256i mSrc );  // AVX2
    inline __m256i Equal16( __m256i mDst, __m256i mSrc ); // AVX2
    inline __m256i Equal32( __m256i mDst, __m256i mSrc ); // AVX2
    inline __m256i Equal64( __m256i mDst, __m256i mSrc ); // AVX2

    // Not Equal
    inline __m128 NotEqual( __m128 mDst, __m128 mSrc ); // SSE

    inline __m128d NotEqual( __m128d mDst, __m128d mSrc ); // SSE2

    inline __m256 NotEqual( __m256 mDst, __m256 mSrc ); // AVX

    inline __m256d NotEqual( __m256d mDst, __m256d mSrc ); // AVX

    // Lesser
    inline __m128 Lesser( __m128 mDst, __m128 mSrc ); // SSE

    inline __m128d Lesser( __m128d mDst, __m128d mSrc ); // SSE2

    inline __m128i Lesser8( __m128i mDst, __m128i mSrc );  // SSE2
    inline __m128i Lesser16( __m128i mDst, __m128i mSrc ); // SSE2
    inline __m128i Lesser32( __m128i mDst, __m128i mSrc ); // SSE2

    inline __m256 Lesser( __m256 mDst, __m256 mSrc ); // AVX

    inline __m256d Lesser( __m256d mDst, __m256d mSrc ); // AVX

    // Not Lesser
    inline __m128 NotLesser( __m128 mDst, __m128 mSrc ); // SSE

    inline __m128d NotLesser( __m128d mDst, __m128d mSrc ); // SSE2

    inline __m256 NotLesser( __m256 mDst, __m256 mSrc ); // AVX

    inline __m256d NotLesser( __m256d mDst, __m256d mSrc ); // AVX

    // Lesser Equal
    inline __m128 LesserEqual( __m128 mDst, __m128 mSrc ); // SSE

    inline __m128d LesserEqual( __m128d mDst, __m128d mSrc ); // SSE2

    inline __m256 LesserEqual( __m256 mDst, __m256 mSrc ); // AVX

    inline __m256d LesserEqual( __m256d mDst, __m256d mSrc ); // AVX

    // Not Lesser Equal
    inline __m128 NotLesserEqual( __m128 mDst, __m128 mSrc ); // SSE

    inline __m128d NotLesserEqual( __m128d mDst, __m128d mSrc ); // SSE2

    inline __m256 NotLesserEqual( __m256 mDst, __m256 mSrc ); // AVX

    inline __m256d NotLesserEqual( __m256d mDst, __m256d mSrc ); // AVX

    // Greater
    inline __m128 Greater( __m128 mDst, __m128 mSrc ); // SSE

    inline __m128d Greater( __m128d mDst, __m128d mSrc ); // SSE2

    inline __m128i Greater8( __m128i mDst, __m128i mSrc );  // SSE2
    inline __m128i Greater16( __m128i mDst, __m128i mSrc ); // SSE2
    inline __m128i Greater32( __m128i mDst, __m128i mSrc ); // SSE2
    inline __m128i Greater64( __m128i mDst, __m128i mSrc ); // SSE42

    inline __m256 Greater( __m256 mDst, __m256 mSrc ); // AVX

    inline __m256d Greater( __m256d mDst, __m256d mSrc ); // AVX

    inline __m256i Greater8( __m256i mDst, __m256i mSrc );  // AVX2
    inline __m256i Greater16( __m256i mDst, __m256i mSrc ); // AVX2
    inline __m256i Greater32( __m256i mDst, __m256i mSrc ); // AVX2
    inline __m256i Greater64( __m256i mDst, __m256i mSrc ); // AVX2

    // Not Greater
    inline __m128 NotGreater( __m128 mDst, __m128 mSrc ); // SSE

    inline __m128d NotGreater( __m128d mDst, __m128d mSrc ); // SSE2

    inline __m256 NotGreater( __m256 mDst, __m256 mSrc ); // AVX

    inline __m256d NotGreater( __m256d mDst, __m256d mSrc ); // AVX

    // Greater Equal
    inline __m128 GreaterEqual( __m128 mDst, __m128 mSrc ); // SSE

    inline __m128d GreaterEqual( __m128d mDst, __m128d mSrc ); // SSE2

    inline __m256 GreaterEqual( __m256 mDst, __m256 mSrc ); // AVX

    inline __m256d GreaterEqual( __m256d mDst, __m256d mSrc ); // AVX

    // Not Greater Equal
    inline __m128 NotGreaterEqual( __m128 mDst, __m128 mSrc ); // SSE

    inline __m128d NotGreaterEqual( __m128d mDst, __m128d mSrc ); // SSE2

    inline __m256 NotGreaterEqual( __m256 mDst, __m256 mSrc ); // AVX

    inline __m256d NotGreaterEqual( __m256d mDst, __m256d mSrc ); // AVX

    // Ordered
    inline __m128 Ordered( __m128 mDst, __m128 mSrc ); // SSE

    inline __m128d Ordered( __m128d mDst, __m128d mSrc ); // SSE2

    inline __m256 Ordered( __m256 mDst, __m256 mSrc ); // AVX

    inline __m256d Ordered( __m256d mDst, __m256d mSrc ); // AVX

    // Unordered
    inline __m128 Unordered( __m128 mDst, __m128 mSrc ); // SSE

    inline __m128d Unordered( __m128d mDst, __m128d mSrc ); // SSE2

    inline __m256 Unordered( __m256 mDst, __m256 mSrc ); // AVX

    inline __m256d Unordered( __m256d mDst, __m256d mSrc ); // AVX

    // Lower Element
    namespace One {

        // Copy Signaling versions
        inline __m128 Equal( __m128 mDst, __m128 mSrc );    // SSE
        inline __m128d Equal( __m128d mDst, __m128d mSrc ); // SSE2

        inline __m128 NotEqual( __m128 mDst, __m128 mSrc );    // SSE
        inline __m128d NotEqual( __m128d mDst, __m128d mSrc ); // SSE2

        inline __m128 Lesser( __m128 mDst, __m128 mSrc );    // SSE
        inline __m128d Lesser( __m128d mDst, __m128d mSrc ); // SSE2

        inline __m128 NotLesser( __m128 mDst, __m128 mSrc );    // SSE
        inline __m128d NotLesser( __m128d mDst, __m128d mSrc ); // SSE2

        inline __m128 LesserEqual( __m128 mDst, __m128 mSrc );    // SSE
        inline __m128d LesserEqual( __m128d mDst, __m128d mSrc ); // SSE2

        inline __m128 NotLesserEqual( __m128 mDst, __m128 mSrc );    // SSE
        inline __m128d NotLesserEqual( __m128d mDst, __m128d mSrc ); // SSE2

        inline __m128 Greater( __m128 mDst, __m128 mSrc );    // SSE
        inline __m128d Greater( __m128d mDst, __m128d mSrc ); // SSE2

        inline __m128 NotGreater( __m128 mDst, __m128 mSrc );    // SSE
        inline __m128d NotGreater( __m128d mDst, __m128d mSrc ); // SSE2

        inline __m128 GreaterEqual( __m128 mDst, __m128 mSrc );    // SSE
        inline __m128d GreaterEqual( __m128d mDst, __m128d mSrc ); // SSE2

        inline __m128 NotGreaterEqual( __m128 mDst, __m128 mSrc );    // SSE
        inline __m128d NotGreaterEqual( __m128d mDst, __m128d mSrc ); // SSE2

        inline __m128 Ordered( __m128 mDst, __m128 mSrc );    // SSE
        inline __m128d Ordered( __m128d mDst, __m128d mSrc ); // SSE2

        inline __m128 Unordered( __m128 mDst, __m128 mSrc );    // SSE
        inline __m128d Unordered( __m128d mDst, __m128d mSrc ); // SSE2

        // Bool Signaling versions
        inline Int IsEqual( __m128 mLHS, __m128 mRHS );   // SSE
        inline Int IsEqual( __m128d mLHS, __m128d mRHS ); // SSE2

        inline Int IsNotEqual( __m128 mLHS, __m128 mRHS );   // SSE
        inline Int IsNotEqual( __m128d mLHS, __m128d mRHS ); // SSE2

        inline Int IsLesser( __m128 mLHS, __m128 mRHS );   // SSE
        inline Int IsLesser( __m128d mLHS, __m128d mRHS ); // SSE2

        inline Int IsLesserEqual( __m128 mLHS, __m128 mRHS );   // SSE
        inline Int IsLesserEqual( __m128d mLHS, __m128d mRHS ); // SSE2

        inline Int IsGreater( __m128 mLHS, __m128 mRHS );   // SSE
        inline Int IsGreater( __m128d mLHS, __m128d mRHS ); // SSE2

        inline Int IsGreaterEqual( __m128 mLHS, __m128 mRHS );   // SSE
        inline Int IsGreaterEqual( __m128d mLHS, __m128d mRHS ); // SSE2

        // Bool Quiet (Non-Signaling) versions
        inline Int IsEqualQ( __m128 mLHS, __m128 mRHS );   // SSE
        inline Int IsEqualQ( __m128d mLHS, __m128d mRHS ); // SSE2

        inline Int IsNotEqualQ( __m128 mLHS, __m128 mRHS );   // SSE
        inline Int IsNotEqualQ( __m128d mLHS, __m128d mRHS ); // SSE2

        inline Int IsLesserQ( __m128 mLHS, __m128 mRHS );   // SSE
        inline Int IsLesserQ( __m128d mLHS, __m128d mRHS ); // SSE2

        inline Int IsLesserEqualQ( __m128 mLHS, __m128 mRHS );   // SSE
        inline Int IsLesserEqualQ( __m128d mLHS, __m128d mRHS ); // SSE2

        inline Int IsGreaterQ( __m128 mLHS, __m128 mRHS );   // SSE
        inline Int IsGreaterQ( __m128d mLHS, __m128d mRHS ); // SSE2

        inline Int IsGreaterEqualQ( __m128 mLHS, __m128 mRHS );   // SSE
        inline Int IsGreaterEqualQ( __m128d mLHS, __m128d mRHS ); // SSE2

    };

    // Strings
    namespace String {

        // May implement those at some point ...
        /////////////////////////////////////////

    };

}; };

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SIMD_Compare.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDCOMPARE_H

