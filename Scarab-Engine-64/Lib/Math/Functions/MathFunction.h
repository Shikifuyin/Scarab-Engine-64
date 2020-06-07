/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Functions/MathFunction.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Basic functions
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : frndInt in Exp* functions rounds toward current rounding mode,
//				not 0 as wished ... should set mode accordingly ...
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_LIB_MATH_FUNCTIONS_MATHFUNCTION_H
#define SCARAB_LIB_MATH_FUNCTIONS_MATHFUNCTION_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../../../ThirdParty/System/Hardware/FPU.h"

#include "../Formats/Scalar.h"
#include "../Formats/Fixed.h"

/////////////////////////////////////////////////////////////////////////////////
// Conditional Compilation Settings
    // Toggle tricks usage (lower precision in most cases, sometimes more stability)
//#define MATHFUNCTION_USE_TRICKS

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define MathRealFn MathFunction<Real>::GetInstance()

#define MathFFn Mathf::GetInstance()
#define MathDFn Mathd::GetInstance()
#define MathFn  Math::GetInstance()

/////////////////////////////////////////////////////////////////////////////////
// The MathFunction template (constains almost only statics !)
template<typename Real>
class MathFunction
{
    // Discrete singleton interface
public:
    inline static MathFunction<Real> * GetInstance();

private:
    MathFunction();
    ~MathFunction();

public:
    // Constants
    static const Real Zero;
    static const Real Half;
    static const Real One;
    static const Real Two;
    static const Real Epsilon;
    static const Real Infinity;

	// Fundamentals
    inline Bool EqualsZero( Real f, Real fThreshold = (Real)SCALAR_ERROR ) const;
	inline Bool Equals( Real f, Real g, Real fThreshold = (Real)SCALAR_ERROR ) const;

    inline Bool IsNan( Real f ) const;
    inline Bool IsInfinite( Real f ) const;
    inline Bool IsZero( Real f ) const;
    
    inline Int Floor( Real f ) const;
	inline Int Ceil( Real f ) const;
	inline Int Round( Real f ) const;
	inline Real Floorf( Real f ) const;
	inline Real Ceilf( Real f ) const;
	inline Real Roundf( Real f ) const;

	inline Real FractPart( Real f ) const;
	inline Real TruncateDecimal( Real f, UInt iLastDecimal = SCALAR_DIGITS ) const;
	inline Real RoundDecimal( Real f, UInt iLastDecimal = SCALAR_DIGITS ) const;
    inline Real Split( Real f, Real * outIntPart ) const;

    inline Real Sign( Real f ) const;
    inline Int  SignI( Real f ) const;
    inline Real Abs( Real f ) const;
    inline Real Mod( Real f, Real g ) const;
	
	// Usual Functions
    inline Real Sqr( Real f ) const;
	inline Real Cube( Real f ) const;

    inline Real Invert( Real f ) const;
	inline Real Sqrt( Real f ) const;
	inline Real InvSqrt( Real f ) const;
    inline Real Cbrt( Real f ) const;
    inline Real RootN( Int n, Real f ) const;
    inline Real Hypot( Real fX, Real fY ) const;
	inline Real Ln( Real f ) const;
	inline Real Log2( Real f ) const;
	inline Real Log10( Real f ) const;
	inline Real LogN( Real n, Real f ) const;
	inline Real Exp( Real f ) const;
	inline Real Exp2( Real f ) const;
	inline Real Exp10( Real f ) const;
	inline Real ExpN( Real n, Real f ) const;

    inline Real Power2( Int iExponent ) const;
	inline Real Power10( Int iExponent ) const;
	inline Real PowerN( Real n, Int iExponent ) const;	

	// Trigonometrics
	inline Real NormalizeAngle( Real f ) const; // Force in [-pi;pi]

	inline Real Sin( Real f ) const;
	inline Real Cos( Real f ) const;
	inline Real Tan( Real f ) const;
	inline Real ArcSin( Real f ) const;
	inline Real ArcCos( Real f ) const;
	inline Real ArcTan( Real f ) const;
	inline Real ArcTan2( Real f, Real g ) const; // arctan( f / g )

    // Hyperbolics
	inline Real SinH( Real f ) const;
	inline Real CosH( Real f ) const;
	inline Real TanH( Real f ) const;
	inline Real ArgSinH( Real f ) const;
	inline Real ArgCosH( Real f ) const;
	inline Real ArgTanH( Real f ) const;

    // Specials
    inline Real Erf( Real f ) const; // Gauss Error Function

    inline Real Gamma( Real f ) const;   // Gamma Function, (x-1)! for positive integers
    inline Real LnGamma( Real f ) const; // Ln of Gamma Function

    inline Real BesselJ( Real f, UInt iOrder ) const; // Bessel Function, 1st kind
    inline Real BesselY( Real f, UInt iOrder ) const; // Bessel Function, 2nd kind

    // Misc & Tools
    inline Bool IsPower2( UInt uiValue ) const;
    inline UInt Log2OfPower2( UInt uiPower2Value ) const;

    inline UInt ScaleUnit( Real f, UInt iBits ) const;
    inline UInt64 ScaleUnit64( Real f, UInt iBits ) const;

private:
    // Functions : Tricked versions
    Real _Tricked_Invert( Real f ) const;
    Real _Tricked_InvSqrt( Real f ) const;

    UInt _Tricked_ScaleUnit( Float f, UInt iBits ) const;
    UInt64 _Tricked_ScaleUnit64( Double f, UInt iBits ) const;

    mutable FloatConverter m_FloatConverter;
    mutable DoubleConverter m_DoubleConverter;
};

// Explicit instanciation
typedef MathFunction<Float> Mathf;
typedef MathFunction<Double> Mathd;
typedef MathFunction<Scalar> Math;

// Specializations
template<>
Float MathFunction<Float>::_Tricked_Invert( Float f ) const;
template<>
Double MathFunction<Double>::_Tricked_Invert( Double f ) const;

template<>
Float MathFunction<Float>::_Tricked_InvSqrt( Float f ) const;
template<>
Double MathFunction<Double>::_Tricked_InvSqrt( Double f ) const;

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "MathFunction.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MATH_FUNCTIONS_MATHFUNCTION_H
