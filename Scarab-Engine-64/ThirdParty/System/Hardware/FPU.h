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
// Known Bugs : TODO = morph this to FPUStack, like SSE ...
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
#define FPUFn FPU::GetInstance()

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
// The FPU class
class FPU
{
    // Discrete singleton interface
public:
    inline static FPU * GetInstance();

private:
    FPU();
    virtual ~FPU();

public:
    virtual FPURoundingMode GetRoundingMode();
    virtual Void SetRoundingMode( FPURoundingMode iRoundingMode = FPU_ROUND_NEAREST );

    virtual FPUClass Classify( Float fValue );
    virtual FPUClass Classify( Double fValue );

    inline virtual Bool IsNAN( Float fValue );
    inline virtual Bool IsNAN( Double fValue );

    inline virtual Bool IsFinite( Float fValue );
    inline virtual Bool IsFinite( Double fValue );

    inline virtual Float FloorF( Float fValue );
    inline virtual Double FloorF( Double fValue );

    inline virtual Float CeilF( Float fValue );
    inline virtual Double CeilF( Double fValue );

    inline virtual Float RoundF( Float fValue );
    inline virtual Double RoundF( Double fValue );

    inline virtual Float Abs( Float fValue );
    inline virtual Double Abs( Double fValue );

    inline virtual Float Mod( Float fValue, Float fMod );
    inline virtual Double Mod( Double fValue, Double fMod );

    inline virtual Float Split( Float fValue, Float * outIntPart ); // Splits fractional and integer parts
    inline virtual Double Split( Double fValue, Double * outIntPart );

    inline virtual Float Sqrt( Float fValue );
    inline virtual Double Sqrt( Double fValue );

    inline virtual Float Cbrt( Float fValue );
    inline virtual Double Cbrt( Double fValue );

    inline virtual Float Hypot( Float fX, Float fY );
    inline virtual Double Hypot( Double fX, Double fY );

    inline virtual Float Ln( Float fValue );
    inline virtual Double Ln( Double fValue );

    inline virtual Float Log2( Float fValue );
    inline virtual Double Log2( Double fValue );

    inline virtual Float Log10( Float fValue );
    inline virtual Double Log10( Double fValue );

    inline virtual Float LogN( Float fBase, Float fValue );
    inline virtual Double LogN( Double fBase, Double fValue );

    inline virtual Float Exp( Float fValue );
    inline virtual Double Exp( Double fValue );

    inline virtual Float Exp2( Float fValue );
    inline virtual Double Exp2( Double fValue );

    inline virtual Float Exp10( Float fValue );
    inline virtual Double Exp10( Double fValue );

    inline virtual Float ExpN( Float fBase, Float fValue );
    inline virtual Double ExpN( Double fBase, Double fValue );

    inline virtual Float Power2f( Int iExponent );
    inline virtual Double Power2d( Int iExponent );

    inline virtual Float Power10f( Int iExponent );
    inline virtual Double Power10d( Int iExponent );

    inline virtual Float PowerN( Float fBase, Int iExponent );
    inline virtual Double PowerN( Double fBase, Int iExponent );

    inline virtual Float Sin( Float fValue );
    inline virtual Double Sin( Double fValue );

    inline virtual Float SinH( Float fValue );
    inline virtual Double SinH( Double fValue );

    inline virtual Float Cos( Float fValue );
    inline virtual Double Cos( Double fValue );

    inline virtual Float CosH( Float fValue );
    inline virtual Double CosH( Double fValue );

    inline virtual Float Tan( Float fValue );
    inline virtual Double Tan( Double fValue );

    inline virtual Float TanH( Float fValue );
    inline virtual Double TanH( Double fValue );

    inline virtual Float ArcSin( Float fValue ); // returns in [-pi;pi], sign of fValue
    inline virtual Double ArcSin( Double fValue );

    inline virtual Float ArcSinH( Float fValue );
    inline virtual Double ArcSinH( Double fValue );

    inline virtual Float ArcCos( Float fValue ); // returns in [-pi;pi], sign of fValue
    inline virtual Double ArcCos( Double fValue );

    inline virtual Float ArcCosH( Float fValue );
    inline virtual Double ArcCosH( Double fValue );

    inline virtual Float ArcTan( Float fValue ); // = arctan2( f, 1.0f ) in [-pi/2;pi/2], sign of fValue
    inline virtual Double ArcTan( Double fValue );

    inline virtual Float ArcTan2( Float fNum, Float fDenom ); // returns in [-pi;pi], sign of fNum
    inline virtual Double ArcTan2( Double fNum, Double fDenom );

    inline virtual Float ArcTanH( Float fValue );
    inline virtual Double ArcTanH( Double fValue );

    inline virtual Float Erf( Float fValue ); // Gauss Error Function
    inline virtual Double Erf( Double fValue );

    inline virtual Float Gamma( Float fValue ); // Gamma function, (x-1)! for positive integers
    inline virtual Double Gamma( Double fValue );

    inline virtual Float LnGamma( Float fValue ); // Ln of Gamma function
    inline virtual Double LnGamma( Double fValue );

    inline virtual Double BesselJ( Double fValue, UInt iOrder ); // Bessel Function, 1st kind

    inline virtual Double BesselY( Double fValue, UInt iOrder ); // Bessel Function, 2nd kind

private:
    const Float m_fHalfF;
    const Double m_fHalfD;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "FPU.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_FPU_H

