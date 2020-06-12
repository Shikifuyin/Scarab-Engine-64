/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Vector/Vector2.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : 2D vector
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// TVector2 implementation
template<typename Real> const TVector2<Real> TVector2<Real>::Null = TVector2<Real>( (Real)0, (Real)0 );
template<typename Real> const TVector2<Real> TVector2<Real>::eI   = TVector2<Real>( (Real)1, (Real)0 );
template<typename Real> const TVector2<Real> TVector2<Real>::eJ   = TVector2<Real>( (Real)0, (Real)1 );

template<typename Real> TVector2<Real>::TVector2()                                 {}
template<typename Real> TVector2<Real>::TVector2( const Real & x, const Real & y ) { X = x; Y = y; }
template<typename Real> TVector2<Real>::TVector2( const Real vArr[2] )             { X = vArr[0]; Y = vArr[1]; }
template<typename Real> TVector2<Real>::TVector2( const TVector2<Real> & rhs )     { X = rhs.X; Y = rhs.Y; }
template<typename Real> TVector2<Real>::TVector2( const TVector3<Real> & rhs )     { X = rhs.X; Y = rhs.Y; }
template<typename Real> TVector2<Real>::TVector2( const TVector4<Real> & rhs )     { X = rhs.X; Y = rhs.Y; }
template<typename Real> TVector2<Real>::~TVector2()                                {}

template<typename Real> inline TVector2<Real> & TVector2<Real>::operator=( const TVector2<Real> & rhs ) { X = rhs.X; Y = rhs.Y; return (*this); }

template<typename Real> inline TVector2<Real>::operator Real*() const       { return (Real*)this; }
template<typename Real> inline TVector2<Real>::operator const Real*() const { return (const Real*)this; }

template<typename Real> inline Real & TVector2<Real>::operator[]( Int i )              { return *( ((Real*)this) + i ); }
template<typename Real> inline const Real & TVector2<Real>::operator[]( Int i ) const  { return *( ((const Real*)this) + i ); }
template<typename Real> inline Real & TVector2<Real>::operator[]( UInt i )             { return *( ((Real*)this) + i ); }
template<typename Real> inline const Real & TVector2<Real>::operator[]( UInt i ) const { return *( ((const Real*)this) + i ); }

template<typename Real> inline TVector2<Real> TVector2<Real>::operator+() const { return TVector2<Real>( X, Y ); }
template<typename Real> inline TVector2<Real> TVector2<Real>::operator-() const { return TVector2<Real>( -X, -Y ); }

template<typename Real>
inline Bool TVector2<Real>::operator==(const TVector2<Real> & rhs) const {
    return ( MathRealFn->Equals(X,rhs.X) && MathRealFn->Equals(Y,rhs.Y) );
}
template<typename Real>
inline Bool TVector2<Real>::operator!=(const TVector2<Real> & rhs) const {
    return ( !(MathRealFn->Equals(X,rhs.X)) || !(MathRealFn->Equals(Y,rhs.Y)) );
}

template<typename Real> inline TVector2<Real> TVector2<Real>::operator+( const Real & rhs ) const { return TVector2<Real>( X + rhs, Y + rhs ); }
template<typename Real> inline TVector2<Real> TVector2<Real>::operator-( const Real & rhs ) const { return TVector2<Real>( X - rhs, Y - rhs ); }
template<typename Real> inline TVector2<Real> TVector2<Real>::operator*( const Real & rhs ) const { return TVector2<Real>( X * rhs, Y * rhs ); }
template<typename Real> inline TVector2<Real> TVector2<Real>::operator/( const Real & rhs ) const { return TVector2<Real>( X / rhs, Y / rhs ); }

template<typename Real> inline TVector2<Real> & TVector2<Real>::operator+=( const Real & rhs ) { X += rhs; Y += rhs; return (*this); }
template<typename Real> inline TVector2<Real> & TVector2<Real>::operator-=( const Real & rhs ) { X -= rhs; Y -= rhs; return (*this); }
template<typename Real> inline TVector2<Real> & TVector2<Real>::operator*=( const Real & rhs ) { X *= rhs; Y *= rhs; return (*this); }
template<typename Real> inline TVector2<Real> & TVector2<Real>::operator/=( const Real & rhs ) { X /= rhs; Y /= rhs; return (*this); }

template<typename Real> inline TVector2<Real> TVector2<Real>::operator+( const TVector2<Real> & rhs ) const { return TVector2<Real>( X + rhs.X, Y + rhs.Y ); }
template<typename Real> inline TVector2<Real> TVector2<Real>::operator-( const TVector2<Real> & rhs ) const { return TVector2<Real>( X - rhs.X, Y - rhs.Y ); }

template<typename Real> inline TVector2<Real> & TVector2<Real>::operator+=( const TVector2<Real> & rhs ) { X += rhs.X; Y += rhs.Y; return (*this); }
template<typename Real> inline TVector2<Real> & TVector2<Real>::operator-=( const TVector2<Real> & rhs ) { X -= rhs.X; Y -= rhs.Y; return (*this); }

template<typename Real>
inline Real TVector2<Real>::operator*( const TVector2<Real> & rhs ) const {
    return ( (X * rhs.X) + (Y * rhs.Y) );
}

template<typename Real>
inline Real TVector2<Real>::operator^( const TVector2<Real> & rhs ) const {
    return ( (X * rhs.Y) - (Y * rhs.X) );
}

template<typename Real>
inline Real TVector2<Real>::NormSqr() const {
    return ( X*X + Y*Y );
}
template<typename Real>
inline Real TVector2<Real>::Norm() const {
    return MathRealFn->Sqrt( NormSqr() );
}
template<typename Real>
inline Real TVector2<Real>::InvNormSqr() const {
    return MathRealFn->Invert( NormSqr() );
}
template<typename Real>
inline Real TVector2<Real>::InvNorm() const {
    return MathRealFn->InvSqrt( NormSqr() );
}

template<typename Real>
inline Real TVector2<Real>::Normalize() {
    Real fNorm = Norm(), fInvNorm;
    if ( MathRealFn->EqualsZero(fNorm) ) {
        X = MathFunction<Real>::Zero;
        Y = MathFunction<Real>::Zero;
        return MathFunction<Real>::Zero;
    } else {
        fInvNorm = MathRealFn->Invert(fNorm);
        X *= fInvNorm;
        Y *= fInvNorm;
        return fNorm;
    }
}

template<typename Real>
inline TVector2<Real> TVector2<Real>::Perp() const {
    return TVector2<Real>(Y, -X);
}

template<typename Real>
inline TVector2<Real> TVector2<Real>::ProjectToNormal(const TVector2<Real> & vNormal) const {
    return ( vNormal * ( (*this) * vNormal ) );
}
template<typename Real>
inline TVector2<Real> TVector2<Real>::ProjectToPlane(const TVector2<Real> & vNormal) const {
    return ( (*this) - ProjectToNormal(vNormal) );
}

template<typename Real>
Void TVector2<Real>::OrthoNormalize( TVector2<Real> & vI, TVector2<Real> & vJ )
{
    // Gram-Schmidt OrthoNormalization
    vI.Normalize();

    Real fDotI = (vI * vJ);
    vJ -= (vI * fDotI);
    vJ.Normalize();
}
template<typename Real>
inline Void TVector2<Real>::MakeComplementBasis( TVector2<Real> & vI, const TVector2<Real> & vJ ) {
    vI = vJ.Perp();
}

