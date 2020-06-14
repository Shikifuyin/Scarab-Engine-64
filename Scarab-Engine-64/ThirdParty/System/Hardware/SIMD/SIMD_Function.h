/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Function.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Function operations
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
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDFUNCTION_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDFUNCTION_H

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <intrin.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../CPUID.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The SIMD::Function namespace
namespace SIMD { namespace Function {

	// Invert
    __forceinline __m128 InvertOne( __m128 mValue ); // SSE

    __forceinline __m128 Invert( __m128 mValue ); // SSE

    __forceinline __m256 Invert( __m256 mValue ); // AVX

    // Square Root
    __forceinline __m128 SqrtOne( __m128 mValue );   // SSE
    __forceinline __m128d SqrtOne( __m128d mValue ); // SSE2

    __forceinline __m128 Sqrt( __m128 mValue );   // SSE
    __forceinline __m128d Sqrt( __m128d mValue ); // SSE2

    __forceinline __m256 Sqrt( __m256 mValue );   // AVX
    __forceinline __m256d Sqrt( __m256d mValue ); // AVX

    // Inverted SquareRoot
    __forceinline __m128 InvSqrtOne( __m128 mValue ); // SSE

    __forceinline __m128 InvSqrt( __m128 mValue );   // SSE
    __forceinline __m128d InvSqrt( __m128d mValue ); // SSE2

    __forceinline __m256 InvSqrt( __m256 mValue );   // AVX
    __forceinline __m256d InvSqrt( __m256d mValue ); // AVX

    // Cube Root
    __forceinline __m128 Cbrt( __m128 mValue );   // SSE
    __forceinline __m128d Cbrt( __m128d mValue ); // SSE2

    __forceinline __m256 Cbrt( __m256 mValue );   // AVX
    __forceinline __m256d Cbrt( __m256d mValue ); // AVX

    // Inverted Cube Root
    __forceinline __m128 InvCbrt( __m128 mValue );   // SSE
    __forceinline __m128d InvCbrt( __m128d mValue ); // SSE2

    __forceinline __m256 InvCbrt( __m256 mValue );   // AVX
    __forceinline __m256d InvCbrt( __m256d mValue ); // AVX

    // Hypothenus (Square Root of summed products)
    __forceinline __m128 Hypot( __m128 mDst, __m128 mSrc );    // SSE
    __forceinline __m128d Hypot( __m128d mDst, __m128d mSrc ); // SSE2

    __forceinline __m256 Hypot( __m256 mDst, __m256 mSrc );    // AVX
    __forceinline __m256d Hypot( __m256d mDst, __m256d mSrc ); // AVX

    // Natural Logarithm
    __forceinline __m128 Ln( __m128 mValue );   // SSE
    __forceinline __m128d Ln( __m128d mValue ); // SSE2

    __forceinline __m256 Ln( __m256 mValue );   // AVX
    __forceinline __m256d Ln( __m256d mValue ); // AVX

    // Natural Logarithm of (1 + x)
    __forceinline __m128 Ln1P( __m128 mValue );   // SSE
    __forceinline __m128d Ln1P( __m128d mValue ); // SSE2

    __forceinline __m256 Ln1P( __m256 mValue );   // AVX
    __forceinline __m256d Ln1P( __m256d mValue ); // AVX

    // Logarithm Base 2
    __forceinline __m128 Log2( __m128 mValue );   // SSE
    __forceinline __m128d Log2( __m128d mValue ); // SSE2

    __forceinline __m256 Log2( __m256 mValue );   // AVX
    __forceinline __m256d Log2( __m256d mValue ); // AVX

    // Logarithm Base 10
    __forceinline __m128 Log10( __m128 mValue );   // SSE
    __forceinline __m128d Log10( __m128d mValue ); // SSE2

    __forceinline __m256 Log10( __m256 mValue );   // AVX
    __forceinline __m256d Log10( __m256d mValue ); // AVX

    // Natural Exponential
    __forceinline __m128 Exp( __m128 mValue );   // SSE
    __forceinline __m128d Exp( __m128d mValue ); // SSE2

    __forceinline __m256 Exp( __m256 mValue );   // AVX
    __forceinline __m256d Exp( __m256d mValue ); // AVX

    // (Natural Exponential of x) - 1
    __forceinline __m128 ExpM1( __m128 mValue );   // SSE
    __forceinline __m128d ExpM1( __m128d mValue ); // SSE2

    __forceinline __m256 ExpM1( __m256 mValue );   // AVX
    __forceinline __m256d ExpM1( __m256d mValue ); // AVX

    // Exponential Base 2
    __forceinline __m128 Exp2( __m128 mValue );   // SSE
    __forceinline __m128d Exp2( __m128d mValue ); // SSE2

    __forceinline __m256 Exp2( __m256 mValue );   // AVX
    __forceinline __m256d Exp2( __m256d mValue ); // AVX

    // Exponential Base 10
    __forceinline __m128 Exp10( __m128 mValue );   // SSE
    __forceinline __m128d Exp10( __m128d mValue ); // SSE2

    __forceinline __m256 Exp10( __m256 mValue );   // AVX
    __forceinline __m256d Exp10( __m256d mValue ); // AVX

    // Power
    __forceinline __m128 Pow( __m128 mBase, __m128 mExponent );    // SSE
    __forceinline __m128d Pow( __m128d mBase, __m128d mExponent ); // SSE2

    __forceinline __m256 Pow( __m256 mBase, __m256 mExponent );    // AVX
    __forceinline __m256d Pow( __m256d mBase, __m256d mExponent ); // AVX

    // Sine
    __forceinline __m128 Sin( __m128 mValue );   // SSE
    __forceinline __m128d Sin( __m128d mValue ); // SSE2

    __forceinline __m256 Sin( __m256 mValue );   // AVX
    __forceinline __m256d Sin( __m256d mValue ); // AVX

    // Cosine
    __forceinline __m128 Cos( __m128 mValue );   // SSE
    __forceinline __m128d Cos( __m128d mValue ); // SSE2

    __forceinline __m256 Cos( __m256 mValue );   // AVX
    __forceinline __m256d Cos( __m256d mValue ); // AVX

    // Sine and Cosine
    __forceinline __m128 SinCos( __m128 * outCos, __m128 mValue );    // SSE
    __forceinline __m128d SinCos( __m128d * outCos, __m128d mValue ); // SSE2

    __forceinline __m256 SinCos( __m256 * outCos, __m256 mValue );    // AVX
    __forceinline __m256d SinCos( __m256d * outCos, __m256d mValue ); // AVX

    // Tangent
    __forceinline __m128 Tan( __m128 mValue );   // SSE
    __forceinline __m128d Tan( __m128d mValue ); // SSE2

    __forceinline __m256 Tan( __m256 mValue );   // AVX
    __forceinline __m256d Tan( __m256d mValue ); // AVX

    // ArcSine
    __forceinline __m128 ArcSin( __m128 mValue );   // SSE
    __forceinline __m128d ArcSin( __m128d mValue ); // SSE2

    __forceinline __m256 ArcSin( __m256 mValue );   // AVX
    __forceinline __m256d ArcSin( __m256d mValue ); // AVX

    // ArcCosine
    __forceinline __m128 ArcCos( __m128 mValue );   // SSE
    __forceinline __m128d ArcCos( __m128d mValue ); // SSE2

    __forceinline __m256 ArcCos( __m256 mValue );   // AVX
    __forceinline __m256d ArcCos( __m256d mValue ); // AVX

    // ArcTangent
    __forceinline __m128 ArcTan( __m128 mValue );   // SSE
    __forceinline __m128d ArcTan( __m128d mValue ); // SSE2

    __forceinline __m256 ArcTan( __m256 mValue );   // AVX
    __forceinline __m256d ArcTan( __m256d mValue ); // AVX

    // ArcTangent2
    __forceinline __m128 ArcTan2( __m128 mNum, __m128 mDenom );    // SSE
    __forceinline __m128d ArcTan2( __m128d mNum, __m128d mDenom ); // SSE2

    __forceinline __m256 ArcTan2( __m256 mNum, __m256 mDenom );    // AVX
    __forceinline __m256d ArcTan2( __m256d mNum, __m256d mDenom ); // AVX

    // Hyperbolic Sine
    __forceinline __m128 SinH( __m128 mValue );   // SSE
    __forceinline __m128d SinH( __m128d mValue ); // SSE2

    __forceinline __m256 SinH( __m256 mValue );   // AVX
    __forceinline __m256d SinH( __m256d mValue ); // AVX

    // Hyperbolic Cosine
    __forceinline __m128 CosH( __m128 mValue );   // SSE
    __forceinline __m128d CosH( __m128d mValue ); // SSE2

    __forceinline __m256 CosH( __m256 mValue );   // AVX
    __forceinline __m256d CosH( __m256d mValue ); // AVX

    // Hyperbolic Tangent
    __forceinline __m128 TanH( __m128 mValue );   // SSE
    __forceinline __m128d TanH( __m128d mValue ); // SSE2

    __forceinline __m256 TanH( __m256 mValue );   // AVX
    __forceinline __m256d TanH( __m256d mValue ); // AVX

    // Hyperbolic ArcSine
    __forceinline __m128 ArgSinH( __m128 mValue );   // SSE
    __forceinline __m128d ArgSinH( __m128d mValue ); // SSE2

    __forceinline __m256 ArgSinH( __m256 mValue );   // AVX
    __forceinline __m256d ArgSinH( __m256d mValue ); // AVX

    // Hyperbolic ArcCosine
    __forceinline __m128 ArgCosH( __m128 mValue );   // SSE
    __forceinline __m128d ArgCosH( __m128d mValue ); // SSE2

    __forceinline __m256 ArgCosH( __m256 mValue );   // AVX
    __forceinline __m256d ArgCosH( __m256d mValue ); // AVX

    // Hyperbolic ArcTangent
    __forceinline __m128 ArgTanH( __m128 mValue );   // SSE
    __forceinline __m128d ArgTanH( __m128d mValue ); // SSE2

    __forceinline __m256 ArgTanH( __m256 mValue );   // AVX
    __forceinline __m256d ArgTanH( __m256d mValue ); // AVX

    // Gauss Error Function
    __forceinline __m128 Erf( __m128 mValue );   // SSE
    __forceinline __m128d Erf( __m128d mValue ); // SSE2

    __forceinline __m256 Erf( __m256 mValue );   // AVX
    __forceinline __m256d Erf( __m256d mValue ); // AVX

    // Inverted Gauss Error Function
    __forceinline __m128 InvErf( __m128 mValue );   // SSE
    __forceinline __m128d InvErf( __m128d mValue ); // SSE2

    __forceinline __m256 InvErf( __m256 mValue );   // AVX
    __forceinline __m256d InvErf( __m256d mValue ); // AVX

    // Complementary Gauss Error Function
    __forceinline __m128 ErfC( __m128 mValue );   // SSE
    __forceinline __m128d ErfC( __m128d mValue ); // SSE2

    __forceinline __m256 ErfC( __m256 mValue );   // AVX
    __forceinline __m256d ErfC( __m256d mValue ); // AVX

    // Inverted Complementary Gauss Error Function
    __forceinline __m128 InvErfC( __m128 mValue );   // SSE
    __forceinline __m128d InvErfC( __m128d mValue ); // SSE2

    __forceinline __m256 InvErfC( __m256 mValue );   // AVX
    __forceinline __m256d InvErfC( __m256d mValue ); // AVX

    // Normal Cumulative Distribution Function
    __forceinline __m128 CDFNorm( __m128 mValue );   // SSE
    __forceinline __m128d CDFNorm( __m128d mValue ); // SSE2

    __forceinline __m256 CDFNorm( __m256 mValue );   // AVX
    __forceinline __m256d CDFNorm( __m256d mValue ); // AVX

    // Inverted Normal Cumulative Distribution Function
    __forceinline __m128 InvCDFNorm( __m128 mValue );   // SSE
    __forceinline __m128d InvCDFNorm( __m128d mValue ); // SSE2

    __forceinline __m256 InvCDFNorm( __m256 mValue );   // AVX
    __forceinline __m256d InvCDFNorm( __m256d mValue ); // AVX

    // Complex Square Root
    __forceinline __m128 CSqrt( __m128 mValue ); // SSE

    __forceinline __m256 CSqrt( __m256 mValue ); // AVX

    // Complex Logarithm
    __forceinline __m128 CLog( __m128 mValue ); // SSE

    __forceinline __m256 CLog( __m256 mValue ); // AVX

    // Complex Exponential
    __forceinline __m128 CExp( __m128 mValue ); // SSE

    __forceinline __m256 CExp( __m256 mValue ); // AVX

}; };

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SIMD_Function.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDFUNCTION_H

