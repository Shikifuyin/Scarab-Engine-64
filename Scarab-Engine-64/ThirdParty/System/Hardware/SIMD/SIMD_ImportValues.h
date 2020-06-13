/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_ImportValues.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Import operations
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
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDIMPORTVALUES_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDIMPORTVALUES_H

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <intrin.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../CPUID.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The SIMD::Import::Values namespace
namespace SIMD { namespace Import { namespace Values {

	// Register Initialization
	inline __m128 Zero128F();  // SSE
	inline __m128d Zero128D(); // SSE2
	inline __m128i Zero128I(); // SSE2

	inline __m256 Zero256F();  // AVX
	inline __m256d Zero256D(); // AVX
	inline __m256i Zero256I(); // AVX

	// Lower Element
	inline __m128 SetOne( Float fValue );   // SSE
	inline __m128d SetOne( Double fValue ); // SSE2
	inline __m128i SetOne( Int32 iValue );  // SSE2
	inline __m128i SetOne( Int64 iValue );  // SSE2

	// Explicit Values
	inline __m128 Set( Float f0, Float f1, Float f2, Float f3 ); // SSE

    inline __m128d Set( Double f0, Double f1 ); // SSE2

    inline __m128i Set( Int8 i0, Int8 i1, Int8 i2, Int8 i3, Int8 i4, Int8 i5, Int8 i6, Int8 i7,
                        Int8 i8, Int8 i9, Int8 i10, Int8 i11, Int8 i12, Int8 i13, Int8 i14, Int8 i15 );   // SSE2
    inline __m128i Set( Int16 i0, Int16 i1, Int16 i2, Int16 i3, Int16 i4, Int16 i5, Int16 i6, Int16 i7 ); // SSE2
    inline __m128i Set( Int32 i0, Int32 i1, Int32 i2, Int32 i3 );                                         // SSE2
    inline __m128i Set( Int64 i0, Int64 i1 );                                                             // SSE2

    inline __m256 Set( Float f0, Float f1, Float f2, Float f3, Float f4, Float f5, Float f6, Float f7 ); // AVX

    inline __m256d Set( Double f0, Double f1, Double f2, Double f3 ); // AVX

    inline __m256i Set( Int8 i0, Int8 i1, Int8 i2, Int8 i3, Int8 i4, Int8 i5, Int8 i6, Int8 i7,
                        Int8 i8, Int8 i9, Int8 i10, Int8 i11, Int8 i12, Int8 i13, Int8 i14, Int8 i15,
                        Int8 i16, Int8 i17, Int8 i18, Int8 i19, Int8 i20, Int8 i21, Int8 i22, Int8 i23,
                        Int8 i24, Int8 i25, Int8 i26, Int8 i27, Int8 i28, Int8 i29, Int8 i30, Int8 i31 );       // AVX
    inline __m256i Set( Int16 i0, Int16 i1, Int16 i2, Int16 i3, Int16 i4, Int16 i5, Int16 i6, Int16 i7,
                        Int16 i8, Int16 i9, Int16 i10, Int16 i11, Int16 i12, Int16 i13, Int16 i14, Int16 i15 ); // AVX
    inline __m256i Set( Int32 i0, Int32 i1, Int32 i2, Int32 i3, Int32 i4, Int32 i5, Int32 i6, Int32 i7 );       // AVX
    inline __m256i Set( Int64 i0, Int64 i1, Int64 i2, Int64 i3 );                                               // AVX

    // Indexed Access
    //inline __m128 Set( __m128 mDst, Float fSrc, Int32 iIndex ); // SSE41

    //inline __m128d Set( __m128d mDst, Double fSrc, Int32 iIndex ); // SSE41

    //inline __m128i Set( __m128i mDst, Int8 iSrc, Int32 iIndex ); // SSE41
    //inline __m128i Set( __m128i mDst, Int16 iSrc, Int32 iIndex ); // SSE2
    //inline __m128i Set( __m128i mDst, Int32 iSrc, Int32 iIndex ); // SSE41
    //inline __m128i Set( __m128i mDst, Int64 iSrc, Int32 iIndex ); // SSE41

    //inline __m256 Set( __m256 mDst, Float fSrc, Int32 iIndex ); // AVX

    //inline __m256d Set( __m256d mDst, Double fSrc, Int32 iIndex ); // AVX

    //inline __m256i Set( __m256i mDst, Int8 iSrc, Int32 iIndex ); // AVX
    //inline __m256i Set( __m256i mDst, Int16 iSrc, Int32 iIndex ); // AVX
    //inline __m256i Set( __m256i mDst, Int32 iSrc, Int32 iIndex ); // AVX
    //inline __m256i Set( __m256i mDst, Int64 iSrc, Int32 iIndex ); // AVX

    // Spread Values
    inline __m128 Spread128( Float fValue ); // SSE

    inline __m128d Spread128( Double fValue ); // SSE2

    inline __m128i Spread128( Int8 iValue );  // SSE2
    inline __m128i Spread128( Int16 iValue ); // SSE2
    inline __m128i Spread128( Int32 iValue ); // SSE2
    inline __m128i Spread128( Int64 iValue ); // SSE2

    inline __m256 Spread256( Float fValue ); // AVX

    inline __m256d Spread256( Double fValue ); // AVX
    
    inline __m256i Spread256( Int8 iValue );  // AVX
    inline __m256i Spread256( Int16 iValue ); // AVX
    inline __m256i Spread256( Int32 iValue ); // AVX
    inline __m256i Spread256( Int64 iValue ); // AVX

}; }; };

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SIMD_ImportValues.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDIMPORTVALUES_H

