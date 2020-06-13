/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_ExportValues.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Export operations
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
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDEXPORTVALUES_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDEXPORTVALUES_H

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <intrin.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../CPUID.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The SIMD::Export::Values namespace
namespace SIMD { namespace Export { namespace Values {

    // Lower Element
	inline Float GetOne( __m128 mSrc ); // SSE

    inline Double GetOne( __m128d mSrc ); // SSE2

    inline Int32 GetOne32( __m128i mSrc ); // SSE2
    inline Int64 GetOne64( __m128i mSrc ); // SSE2

    inline Float GetOne( __m256 mSrc ); // AVX

    inline Double GetOne( __m256d mSrc ); // AVX

    inline Int32 GetOne32( __m256i mSrc ); // AVX
    inline Int64 GetOne64( __m256i mSrc ); // AVX

    // Indexed Access
    //inline Float Get( __m128 mSrc, Int32 iIndex ); // SSE41

    //inline Double Get( __m128d mSrc, Int32 iIndex ); // SSE41

    //inline Int32 Get8( __m128i mSrc, Int32 iIndex );  // SSE41
    //inline Int32 Get16( __m128i mSrc, Int32 iIndex ); // SSE2
    //inline Int32 Get32( __m128i mSrc, Int32 iIndex ); // SSE41
    //inline Int64 Get64( __m128i mSrc, Int32 iIndex ); // SSE41

    //inline Float Get( __m256 mSrc, Int32 iIndex ); // AVX

    //inline Double Get( __m256d mSrc, Int32 iIndex ); // AVX

    //inline Int32 Get8( __m256i mSrc, Int32 iIndex );  // AVX2
    //inline Int32 Get16( __m256i mSrc, Int32 iIndex ); // AVX2
    //inline Int32 Get32( __m256i mSrc, Int32 iIndex ); // AVX
    //inline Int64 Get64( __m256i mSrc, Int32 iIndex ); // AVX

}; }; };

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SIMD_ExportValues.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDEXPORTVALUES_H
