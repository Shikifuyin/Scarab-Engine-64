/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Complex/Complex.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Complex numbers
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// TComplex implementation
template<typename Real> const TComplex<Real> TComplex<Real>::Null = TComplex<Real>( MathFunction<Real>::Zero, MathFunction<Real>::Zero );

template<typename Real> TComplex<Real>::TComplex()                                   {}
template<typename Real> TComplex<Real>::TComplex( const Real & fR, const Real & fI ) { R = fR; I = fI; }
template<typename Real> TComplex<Real>::TComplex( const Real vArr[2] )               { R = vArr[0]; I = vArr[1]; }
template<typename Real> TComplex<Real>::~TComplex()                                  {}

template<typename Real> inline TComplex<Real> & TComplex<Real>::operator=( const TComplex<Real> & rhs ) { R = rhs.R; I = rhs.I; return (*this); }

template<typename Real> inline TComplex<Real>::operator Real*() const       { return (Real*)this; }
template<typename Real> inline TComplex<Real>::operator const Real*() const { return (const Real*)this; }

template<typename Real> inline TComplex<Real> TComplex<Real>::operator+() const { return TComplex<Real>( R, I ); }
template<typename Real> inline TComplex<Real> TComplex<Real>::operator-() const { return TComplex<Real>( -R, -I ); }

template<typename Real>
inline Bool TComplex<Real>::operator==( const TComplex<Real> & rhs ) const {
    return ( MathRealFn->Equals(R,rhs.R) && MathRealFn->Equals(I,rhs.I) );
}
template<typename Real>
inline Bool TComplex<Real>::operator!=( const TComplex<Real> & rhs ) const {
    return ( !(MathRealFn->Equals(R,rhs.R)) || !(MathRealFn->Equals(I,rhs.I)) );
}

template<typename Real> inline TComplex<Real> TComplex<Real>::operator+( const Real & rhs ) const { return TComplex<Real>( R + rhs, I ); }
template<typename Real> inline TComplex<Real> TComplex<Real>::operator-( const Real & rhs ) const { return TComplex<Real>( R - rhs, I ); }
template<typename Real> inline TComplex<Real> TComplex<Real>::operator*( const Real & rhs ) const { return TComplex<Real>( R * rhs, I * rhs ); }
template<typename Real> inline TComplex<Real> TComplex<Real>::operator/( const Real & rhs ) const { return TComplex<Real>( R / rhs, I / rhs ); }

template<typename Real> inline TComplex<Real> & TComplex<Real>::operator+=( const Real & rhs ) { R += rhs; return (*this); }
template<typename Real> inline TComplex<Real> & TComplex<Real>::operator-=( const Real & rhs ) { R -= rhs; return (*this); }
template<typename Real> inline TComplex<Real> & TComplex<Real>::operator*=( const Real & rhs ) { R *= rhs; I *= rhs; return (*this); }
template<typename Real> inline TComplex<Real> & TComplex<Real>::operator/=( const Real & rhs ) { R /= rhs; I /= rhs; return (*this); }

template<typename Real> inline TComplex<Real> TComplex<Real>::operator+( const TComplex<Real> & rhs ) const { return TComplex<Real>( R + rhs.R, I + rhs.I ); }
template<typename Real> inline TComplex<Real> TComplex<Real>::operator-( const TComplex<Real> & rhs ) const { return TComplex<Real>( R - rhs.R, I - rhs.I ); }
template<typename Real> inline
TComplex<Real> TComplex<Real>::operator*( const TComplex<Real> & rhs ) const {
    return TComplex<Real>( (R * rhs.R) - (I * rhs.I), (I * rhs.R) + (R * rhs.I) );
}
template<typename Real>
inline TComplex<Real> TComplex<Real>::operator/( const TComplex<Real> & rhs ) const {
    Real fInvDenom = MathRealFn->Invert( (rhs.R * rhs.R) + (rhs.I * rhs.I) );
    return TComplex<Real>( ( (R * rhs.R) + (I * rhs.I) ) * fInvDenom,
                           ( (I * rhs.R) - (R * rhs.I) ) * fInvDenom );
}

template<typename Real> inline TComplex<Real> & TComplex<Real>::operator+=( const TComplex<Real> & rhs ) { R += rhs.R; I += rhs.I; return (*this); }
template<typename Real> inline TComplex<Real> & TComplex<Real>::operator-=( const TComplex<Real> & rhs ) { R -= rhs.R; I -= rhs.I; return (*this); }
template<typename Real>
inline TComplex<Real> & TComplex<Real>::operator*=( const TComplex<Real> & rhs ) {
    Real fOldR = R;
    R = ( (R * rhs.R) - (I * rhs.I) );
    I = ( (I * rhs.R) + (fOldR * rhs.I) );
    return (*this);
}
template<typename Real>
inline TComplex<Real> & TComplex<Real>::operator/=( const TComplex<Real> & rhs ) {
    Real fInvDenom = MathRealFn->Invert( (rhs.R * rhs.R) + (rhs.I * rhs.I) );
    Real fOldR = R;
    R = ( (R * rhs.R) + (I * rhs.I) ) * fInvDenom;
    I = ( (I * rhs.R) - (fOldR * rhs.I) ) * fInvDenom;
    return (*this);
}

template<typename Real>
inline Real TComplex<Real>::ModulusSqr() const {
    return ( R*R + I*I );
}
template<typename Real>
inline Real TComplex<Real>::Modulus() const {
    return MathRealFn->Sqrt( ModulusSqr() );
}
template<typename Real>
inline Real TComplex<Real>::InvModulusSqr() const {
    return MathRealFn->Invert( ModulusSqr() );
}
template<typename Real>
inline Real TComplex<Real>::InvModulus() const {
    return MathRealFn->InvSqrt( ModulusSqr() );
}

template<typename Real>
inline Real TComplex<Real>::Argument() const {
    return MathRealFn->ArcTan2( I, R );
}

template<typename Real>
inline Void TComplex<Real>::FromPolar( const Real & fRadius, const Real & fTheta ) {
	R = fRadius * MathRealFn->Cos( fTheta );
	I = fRadius * MathRealFn->Sin( fTheta );
}
template<typename Real>
inline Void TComplex<Real>::FromPolar( const TComplex<Real> & vPolar ) {
    FromPolar( vPolar.R, vPolar.I );
}
template<typename Real>
inline Void TComplex<Real>::ToPolar( Real & outRadius, Real & outTheta ) const {
    outRadius = Modulus();
	outTheta = Argument();
}
template<typename Real>
inline Void TComplex<Real>::ToPolar( TComplex<Real> & outPolar ) const {
    ToPolar( outPolar.R, outPolar.I );
}

template<typename Real>
inline TMatrix2<Real> TComplex<Real>::GetMatrix() const {
    return TMatrix2<Real>( R, -I, I, R );
}

template<typename Real>
inline TComplex<Real> TComplex<Real>::Conjugate() const {
    return TComplex<Real>( R, -I );
}
template<typename Real>
inline TComplex<Real> TComplex<Real>::Sqrt() const {
    Real fModulus = Modulus();
    Real fLambda = MathRealFn->Sqrt( (fModulus + R) * MathFunction<Real>::Half );
    Real fDelta = MathRealFn->Sqrt( (fModulus - R) * MathFunction<Real>::Half );
    if ( I < MathFunction<Real>::Zero )
        fDelta = -fDelta;
    return TComplex<Real>( fLambda, fDelta );
}
template<typename Real>
inline TComplex<Real> TComplex<Real>::Ln() const {
    Real fRadius = Modulus();
    Real fAngle = Argument();
    return TComplex<Real>( MathRealFn->Ln(fRadius), fAngle );
}
template<typename Real>
inline TComplex<Real> TComplex<Real>::Exp() const {
    Real fRadius = MathRealFn->Exp( R );
    Real fSin = MathRealFn->Sin( I );
    Real fCos = MathRealFn->Cos( I );
    return TComplex<Real>( fRadius * fCos, fRadius * fSin );
}

