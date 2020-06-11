/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/FPU.cpp
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
// Third-Party Includes
#include <cmath>
#include <cfenv>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "FPU.h"

/////////////////////////////////////////////////////////////////////////////////
// FPU implementation
inline FPURoundingMode FPU::GetRoundingMode() {
	int iRoundingMode = fegetround();
	switch( iRoundingMode ) {
		case FE_TONEAREST:	return FPU_ROUND_NEAREST; break;
		case FE_DOWNWARD:	return FPU_ROUND_FLOOR; break;
		case FE_UPWARD:		return FPU_ROUND_CEIL; break;
		case FE_TOWARDZERO: return FPU_ROUND_TRUNCATE; break;
		default:			DebugAssert(false); break;
	}
	return (FPURoundingMode)INVALID_OFFSET;
}
inline Void FPU::SetRoundingMode( FPURoundingMode iRoundingMode ) {
	switch ( iRoundingMode ) {
		case FPU_ROUND_NEAREST:		fesetround( FE_TONEAREST ); break;
		case FPU_ROUND_FLOOR:		fesetround( FE_DOWNWARD ); break;
		case FPU_ROUND_CEIL:		fesetround( FE_UPWARD ); break;
		case FPU_ROUND_TRUNCATE:	fesetround( FE_TOWARDZERO ); break;
		default:					DebugAssert( false ); break;
	}
}

inline FPUClass FPU::Classify( Float fValue ) {
	int iClass = fpclassify( fValue );
	switch( iClass ) {
		case FP_NAN:		return FPU_CLASS_NAN; break;
		case FP_INFINITE:	return FPU_CLASS_INFINITY; break;
		case FP_NORMAL:		return FPU_CLASS_NORMAL; break;
		case FP_SUBNORMAL:	return FPU_CLASS_DENORMAL; break;
		case FP_ZERO:		return FPU_CLASS_ZERO; break;
		default:			DebugAssert( false ); break;
	}
	return (FPUClass)INVALID_OFFSET;
}
inline FPUClass FPU::Classify( Double fValue ) {
	int iClass = fpclassify( fValue );
	switch ( iClass ) {
		case FP_NAN:		return FPU_CLASS_NAN; break;
		case FP_INFINITE:	return FPU_CLASS_INFINITY; break;
		case FP_NORMAL:		return FPU_CLASS_NORMAL; break;
		case FP_SUBNORMAL:	return FPU_CLASS_DENORMAL; break;
		case FP_ZERO:		return FPU_CLASS_ZERO; break;
		default:			DebugAssert( false ); break;
	}
	return (FPUClass)INVALID_OFFSET;
}

inline Bool FPU::IsNAN( Float fValue ) {
	return isnan( fValue );
}
inline Bool FPU::IsNAN( Double fValue ) {
	return isnan( fValue );
}

inline Bool FPU::IsFinite( Float fValue ) {
	return isfinite( fValue );
}
inline Bool FPU::IsFinite( Double fValue ) {
	return isfinite( fValue );
}

inline Float FPU::FloorF( Float fValue ) {
	return floor( fValue );
}
inline Double FPU::FloorF( Double fValue ) {
	return floor( fValue );
}

inline Float FPU::CeilF( Float fValue ) {
	return ceil( fValue );
}
inline Double FPU::CeilF( Double fValue ) {
	return ceil( fValue );
}

inline Float FPU::RoundF( Float fValue ) {
	return round( fValue );
}
inline Double FPU::RoundF( Double fValue ) {
	return round( fValue );
}

inline Float FPU::Abs( Float fValue ) {
	return fabs( fValue );
}
inline Double FPU::Abs( Double fValue ) {
	return fabs( fValue );
}

inline Float FPU::Mod( Float fValue, Float fMod ) {
	return fmod( fValue, fMod );
}
inline Double FPU::Mod( Double fValue, Double fMod ) {
	return fmod( fValue, fMod );
}

inline Float FPU::Split( Float fValue, Float * outIntPart ) {
	return modf( fValue, outIntPart );
}
inline Double FPU::Split( Double fValue, Double * outIntPart ) {
	return modf( fValue, outIntPart );
}

inline Float FPU::Sqrt( Float fValue ) {
	return sqrt( fValue );
}
inline Double FPU::Sqrt( Double fValue ) {
	return sqrt( fValue );
}

inline Float FPU::Cbrt( Float fValue ) {
	return cbrt( fValue );
}
inline Double FPU::Cbrt( Double fValue ) {
	return cbrt( fValue );
}

inline Float FPU::Hypot( Float fX, Float fY ) {
	return hypot( fX, fY );
}
inline Double FPU::Hypot( Double fX, Double fY ) {
	return hypot( fX, fY );
}

inline Float FPU::Ln( Float fValue ) {
	return log( fValue );
}
inline Double FPU::Ln( Double fValue ) {
	return log( fValue );
}

inline Float FPU::Log2( Float fValue ) {
	return log2( fValue );
}
inline Double FPU::Log2( Double fValue ) {
	return log2( fValue );
}

inline Float FPU::Log10( Float fValue ) {
	return log10( fValue );
}
inline Double FPU::Log10( Double fValue ) {
	return log10( fValue );
}

inline Float FPU::LogN( Float fBase, Float fValue ) {
	return ( log(fValue) / log(fBase) );
}
inline Double FPU::LogN( Double fBase, Double fValue ) {
	return ( log(fValue) / log(fBase) );
}

inline Float FPU::Exp( Float fValue ) {
	return exp( fValue );
}
inline Double FPU::Exp( Double fValue ) {
	return exp( fValue );
}

inline Float FPU::Exp2( Float fValue ) {
	return exp2( fValue );
}
inline Double FPU::Exp2( Double fValue ) {
	return exp2( fValue );
}

inline Float FPU::Exp10( Float fValue ) {
	return pow( 10.0f, fValue );
}
inline Double FPU::Exp10( Double fValue ) {
	return pow( 10.0, fValue );
}

inline Float FPU::ExpN( Float fBase, Float fValue ) {
	return pow( fBase, fValue );
}
inline Double FPU::ExpN( Double fBase, Double fValue ) {
	return pow( fBase, fValue );
}

inline Float FPU::Power2f( Int iExponent ) {
	return pow( 2.0f, (Float)iExponent );
}
inline Double FPU::Power2d( Int iExponent ) {
	return pow( 2.0, (Double)iExponent );
}

inline Float FPU::Power10f( Int iExponent ) {
	return pow( 10.0f, (Float)iExponent );
}
inline Double FPU::Power10d( Int iExponent ) {
	return pow( 10.0, (Double)iExponent );
}

inline Float FPU::PowerN( Float fBase, Int iExponent ) {
	return pow( fBase, (Float)iExponent );
}
inline Double FPU::PowerN( Double fBase, Int iExponent ) {
	return pow( fBase, (Double)iExponent );
}

inline Float FPU::Sin( Float fValue ) {
	return sin( fValue );
}
inline Double FPU::Sin( Double fValue ) {
	return sin( fValue );
}

inline Float FPU::Cos( Float fValue ) {
	return cos( fValue );
}
inline Double FPU::Cos( Double fValue ) {
	return cos( fValue );
}

inline Float FPU::Tan( Float fValue ) {
	return tan( fValue );
}
inline Double FPU::Tan( Double fValue ) {
	return tan( fValue );
}

inline Float FPU::ArcSin( Float fValue ) {
	return asin( fValue );
}
inline Double FPU::ArcSin( Double fValue ) {
	return asin( fValue );
}

inline Float FPU::ArcCos( Float fValue ) {
	return acos( fValue );
}
inline Double FPU::ArcCos( Double fValue ) {
	return acos( fValue );
}

inline Float FPU::ArcTan( Float fValue ) {
	return atan( fValue );
}
inline Double FPU::ArcTan( Double fValue ) {
	return atan( fValue );
}

inline Float FPU::ArcTan2( Float fNum, Float fDenom ) {
	return atan2( fNum, fDenom );
}
inline Double FPU::ArcTan2( Double fNum, Double fDenom ) {
	return atan2( fNum, fDenom );
}

inline Float FPU::SinH( Float fValue ) {
	return sinh( fValue );
}
inline Double FPU::SinH( Double fValue ) {
	return sinh( fValue );
}

inline Float FPU::CosH( Float fValue ) {
	return cosh( fValue );
}
inline Double FPU::CosH( Double fValue ) {
	return cosh( fValue );
}

inline Float FPU::TanH( Float fValue ) {
	return tanh( fValue );
}
inline Double FPU::TanH( Double fValue ) {
	return tanh( fValue );
}

inline Float FPU::ArgSinH( Float fValue ) {
	return asinh( fValue );
}
inline Double FPU::ArgSinH( Double fValue ) {
	return asinh( fValue );
}

inline Float FPU::ArgCosH( Float fValue ) {
	return acosh( fValue );
}
inline Double FPU::ArgCosH( Double fValue ) {
	return acosh( fValue );
}

inline Float FPU::ArgTanH( Float fValue ) {
	return atanh( fValue );
}
inline Double FPU::ArgTanH( Double fValue ) {
	return atanh( fValue );
}

inline Float FPU::Erf( Float fValue ) {
	return erf( fValue  );
}
inline Double FPU::Erf( Double fValue ) {
	return erf( fValue  );
}

inline Float FPU::Gamma( Float fValue ) {
	return tgamma( fValue );
}
inline Double FPU::Gamma( Double fValue ) {
	return tgamma( fValue );
}

inline Float FPU::LnGamma( Float fValue ) {
	return lgamma( fValue );
}
inline Double FPU::LnGamma( Double fValue ) {
	return lgamma( fValue );
}

inline Double FPU::BesselJ( Double fValue, UInt iOrder ) {
	if ( iOrder == 0 )
		return _j0( fValue );
	if ( iOrder == 1 )
		return _j1( fValue );
	return _jn( iOrder, fValue );
}

inline Double FPU::BesselY( Double fValue, UInt iOrder ) {
	if ( iOrder == 0 )
		return _y0( fValue );
	if ( iOrder == 1 )
		return _y1( fValue );
	return _yn( iOrder, fValue );
}
