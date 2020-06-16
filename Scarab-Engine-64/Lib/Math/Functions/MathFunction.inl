/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Functions/MathFunction.inl
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
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// MathFunction implementation
template<typename Real>
inline MathFunction<Real> * MathFunction<Real>::GetInstance() {
    static MathFunction<Real> s_Instance;
    return &s_Instance;
}

template<typename Real>
MathFunction<Real>::MathFunction()
{
    // nothing to do
}
template<typename Real>
MathFunction<Real>::~MathFunction()
{
    // nothing to do
}

template<typename Real> const Real MathFunction<Real>::Zero = (Real)0;
template<typename Real> const Real MathFunction<Real>::Half = (Real)0.5f;
template<typename Real> const Real MathFunction<Real>::One = (Real)1;
template<typename Real> const Real MathFunction<Real>::Two = (Real)2;
template<typename Real> const Real MathFunction<Real>::Epsilon = (Real)FLOAT_EPSILON;
template<typename Real> const Real MathFunction<Real>::Infinity = (Real)FLOAT_INFINITE;

template<typename Real>
inline Bool MathFunction<Real>::EqualsZero( Real f, Real fThreshold ) const {
    return ( Abs(f) <= fThreshold );
}
template<typename Real>
inline Bool MathFunction<Real>::Equals( Real f, Real g, Real fThreshold ) const {
    return ( Abs(f - g) <= fThreshold );
}

template<typename Real>
inline Bool MathFunction<Real>::IsNan( Real f ) const {
    return ( f != f );
}
template<>
inline Bool MathFunction<Float>::IsNan( Float f ) const {
    return FPU::IsNAN(f);
}
template<>
inline Bool MathFunction<Double>::IsNan( Double f ) const {
    return FPU::IsNAN(f);
}

template<typename Real>
inline Bool MathFunction<Real>::IsInfinite( Real f ) const {
    return ( f == Infinity || f == (-Infinity) );
}
template<>
inline Bool MathFunction<Float>::IsInfinite( Float f ) const {
    return !( FPU::IsFinite(f) );
}
template<>
inline Bool MathFunction<Double>::IsInfinite( Double f ) const {
    return !( FPU::IsFinite(f) );
}

template<typename Real>
inline Bool MathFunction<Real>::IsZero( Real f ) const {
    return ( f == Zero || f == -Zero );
}
template<>
inline Bool MathFunction<Float>::IsZero( Float f ) const {
    return ( FPU::Classify(f) == FPU_CLASS_ZERO );
}
template<>
inline Bool MathFunction<Double>::IsZero( Double f ) const {
    return ( FPU::Classify(f) == FPU_CLASS_ZERO );
}

template<typename Real>
inline Int MathFunction<Real>::Floor( Real f ) const {
    return (Int)f;
}
template<>
inline Int MathFunction<Float>::Floor( Float f ) const {
    return (Int)( FPU::FloorF(f) );
}
template<>
inline Int MathFunction<Double>::Floor( Double f ) const {
    return (Int)( FPU::FloorF(f) );
}

template<typename Real>
inline Int MathFunction<Real>::Ceil( Real f ) const {
    Int iFloor = Floor(f);
    if ( (f - (Real)iFloor) > Zero )
        return iFloor + 1;
    return iFloor;
}
template<>
inline Int MathFunction<Float>::Ceil( Float f ) const {
    return (Int)( FPU::CeilF(f) );
}
template<>
inline Int MathFunction<Double>::Ceil( Double f ) const {
    return (Int)( FPU::CeilF(f) );
}

template<typename Real>
inline Int MathFunction<Real>::Round( Real f ) const {
    return Floor( f + Half );
}
template<>
inline Int MathFunction<Float>::Round( Float f ) const {
    return (Int)( FPU::RoundF(f) );
}
template<>
inline Int MathFunction<Double>::Round( Double f ) const {
    return (Int)( FPU::RoundF(f) );
}

template<typename Real>
inline Real MathFunction<Real>::Floorf( Real f ) const {
    return (Real)Floor(f);
}
template<>
inline Float MathFunction<Float>::Floorf( Float f ) const {
    return FPU::FloorF(f);
}
template<>
inline Double MathFunction<Double>::Floorf( Double f ) const {
    return FPU::FloorF(f);
}

template<typename Real>
inline Real MathFunction<Real>::Ceilf( Real f ) const {
    return (Real)Ceil(f);
}
template<>
inline Float MathFunction<Float>::Ceilf( Float f ) const {
    return FPU::CeilF(f);
}
template<>
inline Double MathFunction<Double>::Ceilf( Double f ) const {
    return FPU::CeilF(f);
}

template<typename Real>
inline Real MathFunction<Real>::Roundf( Real f ) const {
    return (Real)Round(f);
}
template<>
inline Float MathFunction<Float>::Roundf( Float f ) const {
    return FPU::RoundF(f);
}
template<>
inline Double MathFunction<Double>::Roundf( Double f ) const {
    return FPU::RoundF(f);
}

template<typename Real>
inline Real MathFunction<Real>::FractPart( Real f ) const {
    return ( f - Floorf(f) );
}
template<typename Real>
inline Real MathFunction<Real>::TruncateDecimal( Real f, UInt iLastDecimal ) const {
    Real fOrder = Power10( (Int)iLastDecimal );
    Real fScaled = (f * fOrder);
    return ( Floorf(fScaled) / fOrder );
}
template<typename Real>
inline Real MathFunction<Real>::RoundDecimal( Real f, UInt iLastDecimal ) const {
    Real fOrder = Power10( (Int)iLastDecimal );
    Real fScaled = (f * fOrder);
    return ( Roundf(fScaled) / fOrder );
}
template<typename Real>
inline Real MathFunction<Real>::Split( Real f, Real * outIntPart ) const {
    *outIntPart = Floorf(f);
    return ( f - *outIntPart );
}
template<>
inline Float MathFunction<Float>::Split( Float f, Float * outIntPart ) const {
    return FPU::Split( f, outIntPart );
}
template<>
inline Double MathFunction<Double>::Split( Double f, Double * outIntPart ) const {
    return FPU::Split( f, outIntPart );
}

template<typename Real>
inline Real MathFunction<Real>::Sign( Real f ) const {
    return ( f < Zero ) ? -One : One;
}
template<typename Real>
inline Int MathFunction<Real>::SignI( Real f ) const {
    return ( f < Zero ) ? -1 : +1;
}

template<typename Real>
inline Real MathFunction<Real>::Abs( Real f ) const {
    return ( f < Zero ) ? -f : f;
}
template<>
inline Float MathFunction<Float>::Abs( Float f ) const {
    return FPU::Abs(f);
}
template<>
inline Double MathFunction<Double>::Abs( Double f ) const {
    return FPU::Abs(f);
}

template<typename Real>
inline Real MathFunction<Real>::Mod( Real f, Real g ) const {
    return (Real)( MathFn->Mod( (Scalar)f, (Scalar)g ) );
}
template<>
inline Float MathFunction<Float>::Mod( Float f, Float g ) const {
    return FPU::Mod( f, g );
}
template<>
inline Double MathFunction<Double>::Mod( Double f, Double g ) const {
    return FPU::Mod( f, g );
}

template<typename Real>
inline Real MathFunction<Real>::Sqr( Real f ) const {
    return ( f * f );
}
template<typename Real>
inline Real MathFunction<Real>::Cube( Real f ) const {
    return ( f * f * f );
}

template<typename Real>
inline Real MathFunction<Real>::Invert( Real f ) const {
    return ( One / f );
}
template<>
inline Float MathFunction<Float>::Invert( Float f ) const {
#if defined(MATHFUNCTION_USE_TRICKS)
    return _Tricked_Invert(f);
#else
    return ( 1.0f / f );
#endif
}
template<>
inline Double MathFunction<Double>::Invert( Double f ) const {
#if defined(MATHFUNCTION_USE_TRICKS)
    return _Tricked_Invert(f);
#else
    return ( 1.0 / f );
#endif
}

template<typename Real>
inline Real MathFunction<Real>::Sqrt( Real f ) const {
    return (Real)( MathFn->Sqrt((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::Sqrt( Float f ) const {
    return FPU::Sqrt(f);
}
template<>
inline Double MathFunction<Double>::Sqrt( Double f ) const {
    return FPU::Sqrt(f);
}

template<typename Real>
inline Real MathFunction<Real>::Cbrt( Real f ) const {
    return (Real)( MathFn->Cbrt((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::Cbrt( Float f ) const {
    return FPU::Cbrt(f);
}
template<>
inline Double MathFunction<Double>::Cbrt( Double f ) const {
    return FPU::Cbrt(f);
}

template<typename Real>
inline Real MathFunction<Real>::InvSqrt( Real f ) const {
    return (Real)( MathFn->InvSqrt((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::InvSqrt( Float f ) const {
#if defined(MATHFUNCTION_USE_TRICKS)
    return _Tricked_InvSqrt(f);
#else
    return ( 1.0f / FPU::Sqrt(f) );
#endif
}
template<>
inline Double MathFunction<Double>::InvSqrt( Double f ) const {
#if defined(MATHFUNCTION_USE_TRICKS)
    return _Tricked_InvSqrt(f);
#else
    return ( 1.0 / FPU::Sqrt(f) );
#endif
}

template<typename Real>
inline Real MathFunction<Real>::RootN( Int n, Real f ) const {
    return ExpN( f, One / (Real)n );
}

template<typename Real>
inline Real MathFunction<Real>::Hypot( Real fX, Real fY ) const {
    return Sqrt( fX*fX + fY*fY );
}
template<>
inline Float MathFunction<Float>::Hypot( Float fX, Float fY ) const {
    return FPU::Hypot( fX, fY );
}
template<>
inline Double MathFunction<Double>::Hypot( Double fX, Double fY ) const {
    return FPU::Hypot( fX, fY );
}

template<typename Real>
inline Real MathFunction<Real>::Ln( Real f ) const {
    return (Real)( MathFn->Ln((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::Ln( Float f ) const {
    return FPU::Ln(f);
}
template<>
inline Double MathFunction<Double>::Ln( Double f ) const {
    return FPU::Ln(f);
}

template<typename Real>
inline Real MathFunction<Real>::Log2( Real f ) const {
    return (Real)( MathFn->Log2((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::Log2( Float f ) const {
    return FPU::Log2(f);
}
template<>
inline Double MathFunction<Double>::Log2( Double f ) const {
    return FPU::Log2(f);
}

template<typename Real>
inline Real MathFunction<Real>::Log10( Real f ) const {
    return (Real)( MathFn->Log10((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::Log10( Float f ) const {
    return FPU::Log10(f);
}
template<>
inline Double MathFunction<Double>::Log10( Double f ) const {
    return FPU::Log10(f);
}

template<typename Real>
inline Real MathFunction<Real>::LogN( Real n, Real f ) const {
    return (Real)( MathFn->LogN((Scalar)n, (Scalar)f) );
}
template<>
inline Float MathFunction<Float>::LogN( Float n, Float f ) const {
    return FPU::LogN( n, f );
}
template<>
inline Double MathFunction<Double>::LogN( Double n, Double f ) const {
    return FPU::LogN( n, f );
}

template<typename Real>
inline Real MathFunction<Real>::Exp( Real f ) const {
    return (Real)( MathFn->Exp((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::Exp( Float f ) const {
    return FPU::Exp(f);
}
template<>
inline Double MathFunction<Double>::Exp( Double f ) const {
    return FPU::Exp(f);
}

template<typename Real>
inline Real MathFunction<Real>::Exp2( Real f ) const {
    return (Real)( MathFn->Exp2((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::Exp2( Float f ) const {
    return FPU::Exp2(f);
}
template<>
inline Double MathFunction<Double>::Exp2( Double f ) const {
    return FPU::Exp2(f);
}

template<typename Real>
inline Real MathFunction<Real>::Exp10( Real f ) const {
    return (Real)( MathFn->Exp10((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::Exp10( Float f ) const {
    return FPU::Exp10(f);
}
template<>
inline Double MathFunction<Double>::Exp10( Double f ) const {
    return FPU::Exp10(f);
}

template<typename Real>
inline Real MathFunction<Real>::ExpN( Real n, Real f ) const {
    return (Real)( MathFn->ExpN((Scalar)n, (Scalar)f) );
}
template<>
inline Float MathFunction<Float>::ExpN( Float n, Float f ) const {
    return FPU::ExpN( n, f );
}
template<>
inline Double MathFunction<Double>::ExpN( Double n, Double f ) const {
    return FPU::ExpN( n, f );
}

template<typename Real>
inline Real MathFunction<Real>::Power2( Int iExponent ) const {
    return (Real)( MathFn->Power2(iExponent) );
}
template<>
inline Float MathFunction<Float>::Power2( Int iExponent ) const {
    return FPU::Power2f(iExponent);
}
template<>
inline Double MathFunction<Double>::Power2( Int iExponent ) const {
    return FPU::Power2d(iExponent);
}

template<typename Real>
inline Real MathFunction<Real>::Power10( Int iExponent ) const {
    return (Real)( MathFn->Power10(iExponent) );
}
template<>
inline Float MathFunction<Float>::Power10( Int iExponent ) const {
    return FPU::Power10f(iExponent);
}
template<>
inline Double MathFunction<Double>::Power10( Int iExponent ) const {
    return FPU::Power10d(iExponent);
}

template<typename Real>
inline Real MathFunction<Real>::PowerN( Real n, Int iExponent ) const {
    return (Real)( MathFn->PowerN((Scalar)n, iExponent) );
}
template<>
inline Float MathFunction<Float>::PowerN( Float n, Int iExponent ) const {
    return FPU::PowerN( n, iExponent );
}
template<>
inline Double MathFunction<Double>::PowerN( Double n, Int iExponent ) const {
    return FPU::PowerN( n, iExponent );
}

template<typename Real>
inline Real MathFunction<Real>::NormalizeAngle( Real f ) const {
    f = Mod( f, SCALAR_2PI );
    if ( f < -SCALAR_PI )
        return f + SCALAR_2PI;
    if ( f > SCALAR_PI )
        return f - SCALAR_2PI;
    return f;
}

template<typename Real>
inline Real MathFunction<Real>::Sin( Real f ) const {
    return (Real)( MathFn->Sin((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::Sin( Float f ) const {
    return FPU::Sin(f);
}
template<>
inline Double MathFunction<Double>::Sin( Double f ) const {
    return FPU::Sin(f);
}

template<typename Real>
inline Real MathFunction<Real>::Cos( Real f ) const {
    return (Real)( MathFn->Cos((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::Cos( Float f ) const {
    return FPU::Cos(f);
}
template<>
inline Double MathFunction<Double>::Cos( Double f ) const {
    return FPU::Cos(f);
}

template<typename Real>
inline Real MathFunction<Real>::Tan( Real f ) const {
    return (Real)( MathFn->Tan((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::Tan( Float f ) const {
    return FPU::Tan(f);
}
template<>
inline Double MathFunction<Double>::Tan( Double f ) const {
    return FPU::Tan(f);
}

template<typename Real>
inline Real MathFunction<Real>::ArcSin( Real f ) const {
    return (Real)( MathFn->ArcSin((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::ArcSin( Float f ) const {
    return FPU::ArcSin(f);
}
template<>
inline Double MathFunction<Double>::ArcSin( Double f ) const {
    return FPU::ArcSin(f);
}

template<typename Real>
inline Real MathFunction<Real>::ArcCos( Real f ) const {
    return (Real)( MathFn->ArcCos((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::ArcCos( Float f ) const {
    return FPU::ArcCos(f);
}
template<>
inline Double MathFunction<Double>::ArcCos( Double f ) const {
    return FPU::ArcCos(f);
}

template<typename Real>
inline Real MathFunction<Real>::ArcTan( Real f ) const {
    return (Real)( MathFn->ArcTan((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::ArcTan( Float f ) const {
    return FPU::ArcTan(f);
}
template<>
inline Double MathFunction<Double>::ArcTan( Double f ) const {
    return FPU::ArcTan(f);
}

template<typename Real>
inline Real MathFunction<Real>::ArcTan2( Real f, Real g ) const {
    return (Real)( MathFn->ArcTan2((Scalar)f, (Scalar)g) );
}
template<>
inline Float MathFunction<Float>::ArcTan2( Float f, Float g ) const {
    return FPU::ArcTan2( f, g );
}
template<>
inline Double MathFunction<Double>::ArcTan2( Double f, Double g ) const {
    return FPU::ArcTan2( f, g );
}

template<typename Real>
inline Real MathFunction<Real>::SinH( Real f ) const {
    Real ef = Exp(f);
    return Half * ( ef - Invert(ef) );
}
template<>
inline Float MathFunction<Float>::SinH( Float f ) const {
    return FPU::SinH(f);
}
template<>
inline Double MathFunction<Double>::SinH( Double f ) const {
    return FPU::SinH(f);
}

template<typename Real>
inline Real MathFunction<Real>::CosH( Real f ) const {
    Real ef = Exp(f);
    return Half * ( ef + Invert(ef) );
}
template<>
inline Float MathFunction<Float>::CosH( Float f ) const {
    return FPU::CosH(f);
}
template<>
inline Double MathFunction<Double>::CosH( Double f ) const {
    return FPU::CosH(f);
}

template<typename Real>
inline Real MathFunction<Real>::TanH( Real f) const {
    Real e2f = Exp(Two * f);
    return ( (e2f - One) / (e2f + One) );
}
template<>
inline Float MathFunction<Float>::TanH( Float f ) const {
    return FPU::TanH(f);
}
template<>
inline Double MathFunction<Double>::TanH( Double f ) const {
    return FPU::TanH(f);
}

template<typename Real>
inline Real MathFunction<Real>::ArgSinH( Real f ) const {
    return Ln( f + Sqrt( (f*f) + One ) );
}
template<>
inline Float MathFunction<Float>::ArgSinH( Float f ) const {
    return FPU::ArgSinH(f);
}
template<>
inline Double MathFunction<Double>::ArgSinH( Double f ) const {
    return FPU::ArgSinH(f);
}

template<typename Real>
inline Real MathFunction<Real>::ArgCosH( Real f ) const {
    return Ln( f + Sqrt( (f*f) - One ) );
}
template<>
inline Float MathFunction<Float>::ArgCosH( Float f ) const {
    return FPU::ArgCosH(f);
}
template<>
inline Double MathFunction<Double>::ArgCosH( Double f ) const {
    return FPU::ArgCosH(f);
}

template<typename Real>
inline Real MathFunction<Real>::ArgTanH( Real f ) const {
    return Half * Ln( (One + f) / (One - f) );
}
template<>
inline Float MathFunction<Float>::ArgTanH( Float f ) const {
    return FPU::ArgTanH(f);
}
template<>
inline Double MathFunction<Double>::ArgTanH( Double f ) const {
    return FPU::ArgTanH(f);
}

template<typename Real>
inline Real MathFunction<Real>::Erf( Real f ) const {
    return (Real)( MathFn->Erf((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::Erf( Float f ) const {
    return FPU::Erf(f);
}
template<>
inline Double MathFunction<Double>::Erf( Double f ) const {
    return FPU::Erf(f);
}

template<typename Real>
inline Real MathFunction<Real>::Gamma( Real f ) const {
    return (Real)( MathFn->Gamma((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::Gamma( Float f ) const {
    return FPU::Gamma(f);
}
template<>
inline Double MathFunction<Double>::Gamma( Double f ) const {
    return FPU::Gamma(f);
}

template<typename Real>
inline Real MathFunction<Real>::LnGamma( Real f ) const {
    return (Real)( MathFn->LnGamma((Scalar)f) );
}
template<>
inline Float MathFunction<Float>::LnGamma( Float f ) const {
    return FPU::LnGamma(f);
}
template<>
inline Double MathFunction<Double>::LnGamma( Double f ) const {
    return FPU::LnGamma(f);
}

template<typename Real>
inline Real MathFunction<Real>::BesselJ( Real f, UInt iOrder ) const {
    return (Real)( MathFn->BesselJ((Scalar)f, iOrder) );
}
template<>
inline Float MathFunction<Float>::BesselJ( Float f, UInt iOrder ) const {
    return (Float)( FPU::BesselJ( (Double)f, iOrder ) );
}
template<>
inline Double MathFunction<Double>::BesselJ( Double f, UInt iOrder ) const {
    return FPU::BesselJ(f, iOrder);
}

template<typename Real>
inline Real MathFunction<Real>::BesselY( Real f, UInt iOrder ) const {
    return (Real)( MathFn->BesselY((Scalar)f, iOrder) );
}
template<>
inline Float MathFunction<Float>::BesselY( Float f, UInt iOrder ) const {
    return (Float)( FPU::BesselY( (Double)f, iOrder ) );
}
template<>
inline Double MathFunction<Double>::BesselY( Double f, UInt iOrder ) const {
    return FPU::BesselY(f, iOrder);
}

template<typename Real>
inline Bool MathFunction<Real>::IsPower2( UInt uiValue ) const {
    return ( (uiValue > 0) && ((uiValue & (uiValue-1)) == 0) );
}
template<typename Real>
inline UInt MathFunction<Real>::Log2OfPower2( UInt uiPower2Value ) const {
    UInt uiLog2 = ( uiPower2Value & 0xAAAAAAAA ) != 0;
    uiLog2 |= ( (uiPower2Value & 0xCCCCCCCC) != 0 ) << 1;
    uiLog2 |= ( (uiPower2Value & 0xF0F0F0F0) != 0 ) << 2;
    uiLog2 |= ( (uiPower2Value & 0xFF00FF00) != 0 ) << 3;
    uiLog2 |= ( (uiPower2Value & 0xFFFF0000) != 0 ) << 4;
    return uiLog2;
}

template<typename Real>
inline UInt MathFunction<Real>::ScaleUnit( Real f, UInt iBits ) const {
    UInt iMask = ( (1ul << iBits) - 1ul );
    return ( ((UInt)(f * (Real)iMask)) & iMask );
}
template<>
inline UInt MathFunction<Float>::ScaleUnit( Float f, UInt iBits ) const {
#if defined(MATHFUNCTION_USE_TRICKS)
    return _Tricked_ScaleUnit( f, iBits );
#else
    UInt iMask = ( (1ul << iBits) - 1ul );
    return ( ( (UInt)(f * (Float)iMask) ) & iMask );
#endif
}
template<>
inline UInt MathFunction<Double>::ScaleUnit( Double f, UInt iBits ) const {
#if defined(MATHFUNCTION_USE_TRICKS)
    return _Tricked_ScaleUnit( f, iBits );
#else
    UInt iMask = ( (1ul << iBits) - 1ul );
    return ( ( (UInt)(f * (Double)iMask) ) & iMask );
#endif
}

template<typename Real>
inline UInt64 MathFunction<Real>::ScaleUnit64( Real f, UInt iBits ) const {
    UInt64 iMask = ( (1ui64 << iBits) - 1ui64 );
    return ( ( (UInt64)(f * (Real)iMask) ) & iMask );
}
template<>
inline UInt64 MathFunction<Float>::ScaleUnit64( Float f, UInt iBits ) const {
#if defined(MATHFUNCTION_USE_TRICKS)
    return _Tricked_ScaleUnit64( f, iBits );
#else
    UInt64 iMask = ( (1ui64 << iBits) - 1ui64 );
    return ( ( (UInt64)(f * (Float)iMask) ) & iMask );
#endif
}
template<>
inline UInt64 MathFunction<Double>::ScaleUnit64( Double f, UInt iBits ) const {
#if defined(MATHFUNCTION_USE_TRICKS)
    return _Tricked_ScaleUnit64( f, iBits );
#else
    UInt64 iMask = ( (1ui64 << iBits) - 1ui64 );
    return ( ( (UInt64)(f * (Double)iMask) ) & iMask );
#endif
}

/////////////////////////////////////////////////////////////////////////////////

template<typename Real>
Real MathFunction<Real>::_Tricked_Invert( Real f ) const
{
    // Dummy stub, never called
    return Infinity;
}
template<typename Real>
Real MathFunction<Real>::_Tricked_InvSqrt( Real f ) const
{
    // Dummy stub, never called
    return Infinity;
}

template<typename Real>
UInt MathFunction<Real>::_Tricked_ScaleUnit( Float f, UInt iBits ) const
{
    // Untested //
    m_FloatConverter.f = f;
    DWord dwRepr = m_FloatConverter.i;
    Int iShift = ( FLOAT_EXP_BIAS + FLOAT_MANT_BITS - iBits - ((dwRepr >> FLOAT_MANT_BITS) & 0xff) );
    if ( iShift < FLOAT_MANT_BITS + 1 ) {
        DWord dwMask = ( 0x00000001ul << FLOAT_MANT_BITS );
	    Int iRes = ( (dwRepr & (dwMask-1)) | dwMask ) >> iShift;
	    if ( iRes == (1 << iBits) )
		    --iRes;
	    return (unsigned)iRes;
    } else
	    return 0;
}
template<typename Real>
UInt64 MathFunction<Real>::_Tricked_ScaleUnit64( Double f, UInt iBits ) const
{
    // Untested //
    m_DoubleConverter.f = f;
    QWord qwRepr = m_DoubleConverter.i;
    Int iShift = ( DOUBLE_EXP_BIAS + DOUBLE_MANT_BITS - iBits - ((qwRepr >> DOUBLE_MANT_BITS) & 0x7ff) );
    if ( iShift < DOUBLE_MANT_BITS + 1 ) {
        QWord qwMask = ( 0x0000000000000001ui64 << DOUBLE_MANT_BITS );
	    Int64 iRes = ( (qwRepr & (qwMask-1)) | qwMask ) >> iShift;
	    if ( iRes == (1i64 << iBits) )
		    --iRes;
	    return (unsigned)iRes;
    } else
	    return 0ui64;
}

