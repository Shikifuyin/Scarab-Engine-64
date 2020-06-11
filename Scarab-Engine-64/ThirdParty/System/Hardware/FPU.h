/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/FPU.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : FPU low level abstraction layer
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
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_FPU_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_FPU_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../Platform.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// IEEE 754 specifications
typedef Word FPURoundingMode;
#define FPU_ROUND_NEAREST  0 // floor if equally close
#define FPU_ROUND_FLOOR    1 // toward -inf
#define FPU_ROUND_CEIL     2 // toward +inf
#define FPU_ROUND_TRUNCATE 3 // toward 0 (abs round)

typedef DWord FPUClass;
#define FPU_CLASS_NAN         0
#define FPU_CLASS_INFINITY    1
#define FPU_CLASS_NORMAL      2
#define FPU_CLASS_DENORMAL    3
#define FPU_CLASS_ZERO        4

/////////////////////////////////////////////////////////////////////////////////
// The FPU namespace
namespace FPU
{
    inline FPURoundingMode GetRoundingMode();
    inline Void SetRoundingMode( FPURoundingMode iRoundingMode = FPU_ROUND_NEAREST );

    inline FPUClass Classify( Float fValue );
    inline FPUClass Classify( Double fValue );

    inline Bool IsNAN( Float fValue );
    inline Bool IsNAN( Double fValue );

    inline Bool IsFinite( Float fValue );
    inline Bool IsFinite( Double fValue );

    inline Float FloorF( Float fValue );
    inline Double FloorF( Double fValue );

    inline Float CeilF( Float fValue );
    inline Double CeilF( Double fValue );

    inline Float RoundF( Float fValue );
    inline Double RoundF( Double fValue );

    inline Float Abs( Float fValue );
    inline Double Abs( Double fValue );

    inline Float Mod( Float fValue, Float fMod );
    inline Double Mod( Double fValue, Double fMod );

    inline Float Split( Float fValue, Float * outIntPart ); // Splits fractional and integer parts
    inline Double Split( Double fValue, Double * outIntPart );

    inline Float Sqrt( Float fValue );
    inline Double Sqrt( Double fValue );

    inline Float Cbrt( Float fValue );
    inline Double Cbrt( Double fValue );

    inline Float Hypot( Float fX, Float fY );
    inline Double Hypot( Double fX, Double fY );

    inline Float Ln( Float fValue );
    inline Double Ln( Double fValue );

    inline Float Log2( Float fValue );
    inline Double Log2( Double fValue );

    inline Float Log10( Float fValue );
    inline Double Log10( Double fValue );

    inline Float LogN( Float fBase, Float fValue );
    inline Double LogN( Double fBase, Double fValue );

    inline Float Exp( Float fValue );
    inline Double Exp( Double fValue );

    inline Float Exp2( Float fValue );
    inline Double Exp2( Double fValue );

    inline Float Exp10( Float fValue );
    inline Double Exp10( Double fValue );

    inline Float ExpN( Float fBase, Float fValue );
    inline Double ExpN( Double fBase, Double fValue );

    inline Float Power2f( Int iExponent );
    inline Double Power2d( Int iExponent );

    inline Float Power10f( Int iExponent );
    inline Double Power10d( Int iExponent );

    inline Float PowerN( Float fBase, Int iExponent );
    inline Double PowerN( Double fBase, Int iExponent );

    inline Float Sin( Float fValue );
    inline Double Sin( Double fValue );

    inline Float Cos( Float fValue );
    inline Double Cos( Double fValue );

    inline Float Tan( Float fValue );
    inline Double Tan( Double fValue );

    inline Float ArcSin( Float fValue ); // returns in [-pi;pi], sign of fValue
    inline Double ArcSin( Double fValue );

    inline Float ArcCos( Float fValue ); // returns in [-pi;pi], sign of fValue
    inline Double ArcCos( Double fValue );

    inline Float ArcTan( Float fValue ); // = arctan2( f, 1.0f ) in [-pi/2;pi/2], sign of fValue
    inline Double ArcTan( Double fValue );

    inline Float ArcTan2( Float fNum, Float fDenom ); // returns in [-pi;pi], sign of fNum
    inline Double ArcTan2( Double fNum, Double fDenom );

    inline Float SinH( Float fValue );
    inline Double SinH( Double fValue );

    inline Float CosH( Float fValue );
    inline Double CosH( Double fValue );

    inline Float TanH( Float fValue );
    inline Double TanH( Double fValue );

    inline Float ArgSinH( Float fValue );
    inline Double ArgSinH( Double fValue );

    inline Float ArgCosH( Float fValue );
    inline Double ArgCosH( Double fValue );

    inline Float ArgTanH( Float fValue );
    inline Double ArgTanH( Double fValue );

    inline Float Erf( Float fValue ); // Gauss Error Function
    inline Double Erf( Double fValue );

    inline Float Gamma( Float fValue ); // Gamma Function, (x-1)! for positive integers
    inline Double Gamma( Double fValue );

    inline Float LnGamma( Float fValue ); // Ln of Gamma Function
    inline Double LnGamma( Double fValue );

    inline Double BesselJ( Double fValue, UInt iOrder ); // Bessel Function, 1st kind

    inline Double BesselY( Double fValue, UInt iOrder ); // Bessel Function, 2nd kind
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "FPU.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_FPU_H

