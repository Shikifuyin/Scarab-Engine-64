/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Vector/Vector4.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Homogeneous 4D vector
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// TVector4 implementation
template<typename Real> const TVector4<Real> TVector4<Real>::Null = TVector4<Real>( (Real)0, (Real)0, (Real)0, (Real)0 );
template<typename Real> const TVector4<Real> TVector4<Real>::eI   = TVector4<Real>( (Real)1, (Real)0, (Real)0, (Real)0 );
template<typename Real> const TVector4<Real> TVector4<Real>::eJ   = TVector4<Real>( (Real)0, (Real)1, (Real)0, (Real)0 );
template<typename Real> const TVector4<Real> TVector4<Real>::eK   = TVector4<Real>( (Real)0, (Real)0, (Real)1, (Real)0 );

template<typename Real> TVector4<Real>::TVector4()                                                                 {}
template<typename Real> TVector4<Real>::TVector4( const Real & x, const Real & y, const Real & z, const Real & w ) { X = x; Y = y; Z = z; W = w; }
template<typename Real> TVector4<Real>::TVector4( const Real vArr[4] )                                             { X = vArr[0]; Y = vArr[1]; Z = vArr[2]; W = vArr[3]; }
template<typename Real> TVector4<Real>::TVector4( const TVector2<Real> & rhs )                                     { X = rhs.X; Y = rhs.Y; Z = (Real)0; W = (Real)0; }
template<typename Real> TVector4<Real>::TVector4( const TVector3<Real> & rhs )                                     { X = rhs.X; Y = rhs.Y; Z = rhs.Z; W = (Real)0; }
template<typename Real> TVector4<Real>::TVector4( const TVector4<Real> & rhs )                                     { X = rhs.X; Y = rhs.Y; Z = rhs.Z; W = rhs.W; }
template<typename Real> TVector4<Real>::~TVector4()                                                                {}

template<typename Real> inline TVector4<Real> & TVector4<Real>::operator=( const TVector4<Real> & rhs ) { X = rhs.X; Y = rhs.Y; Z = rhs.Z; W = rhs.W; return (*this); }

template<typename Real> inline TVector4<Real>::operator Real*() const       { return (Real*)this; }
template<typename Real> inline TVector4<Real>::operator const Real*() const { return (const Real*)this; }

template<typename Real> inline Real & TVector4<Real>::operator[]( Int i )              { return *( ((Real*)this) + i ); }
template<typename Real> inline const Real & TVector4<Real>::operator[]( Int i ) const  { return *( ((const Real*)this) + i ); }
template<typename Real> inline Real & TVector4<Real>::operator[]( UInt i )             { return *( ((Real*)this) + i ); }
template<typename Real> inline const Real & TVector4<Real>::operator[]( UInt i ) const { return *( ((const Real*)this) + i ); }

template<typename Real> inline TVector4<Real> TVector4<Real>::operator+() const { return TVector4<Real>( X, Y, Z, W ); }
template<typename Real> inline TVector4<Real> TVector4<Real>::operator-() const { return TVector4<Real>( -X, -Y, -Z, W ); }

template<typename Real>
inline Bool TVector4<Real>::operator==( const TVector4<Real> & rhs ) const {
    return ( MathRealFn->Equals(X,rhs.X) && MathRealFn->Equals(Y,rhs.Y) && MathRealFn->Equals(Z,rhs.Z) && MathRealFn->Equals(W,rhs.W) );
}
template<typename Real>
inline Bool TVector4<Real>::operator!=( const TVector4<Real> & rhs ) const {
    return ( !(MathRealFn->Equals(X,rhs.X)) || !(MathRealFn->Equals(Y,rhs.Y)) || !(MathRealFn->Equals(Z,rhs.Z)) || !(MathRealFn->Equals(W,rhs.W)) );
}

template<typename Real> inline TVector4<Real> TVector4<Real>::operator+( const Real & rhs ) const { return TVector4<Real>( X + rhs, Y + rhs, Z + rhs, W ); }
template<typename Real> inline TVector4<Real> TVector4<Real>::operator-( const Real & rhs ) const { return TVector4<Real>( X - rhs, Y - rhs, Z - rhs, W ); }
template<typename Real> inline TVector4<Real> TVector4<Real>::operator*( const Real & rhs ) const { return TVector4<Real>( X * rhs, Y * rhs, Z * rhs, W ); }
template<typename Real> inline TVector4<Real> TVector4<Real>::operator/( const Real & rhs ) const { return TVector4<Real>( X / rhs, Y / rhs, Z / rhs, W ); }

template<typename Real> inline TVector4<Real> & TVector4<Real>::operator+=( const Real & rhs ) { X += rhs; Y += rhs; Z += rhs; return (*this); }
template<typename Real> inline TVector4<Real> & TVector4<Real>::operator-=( const Real & rhs ) { X -= rhs; Y -= rhs; Z -= rhs; return (*this); }
template<typename Real> inline TVector4<Real> & TVector4<Real>::operator*=( const Real & rhs ) { X *= rhs; Y *= rhs; Z *= rhs; return (*this); }
template<typename Real> inline TVector4<Real> & TVector4<Real>::operator/=( const Real & rhs ) { X /= rhs; Y /= rhs; Z /= rhs; return (*this); }

template<typename Real> inline TVector4<Real> TVector4<Real>::operator+( const TVector4<Real> & rhs ) const { return TVector4<Real>( X + rhs.X, Y + rhs.Y, Z + rhs.Z, W + rhs.W ); }
template<typename Real> inline TVector4<Real> TVector4<Real>::operator-( const TVector4<Real> & rhs ) const { return TVector4<Real>( X - rhs.X, Y - rhs.Y, Z - rhs.Z, W - rhs.W ); }

template<typename Real> inline TVector4<Real> & TVector4<Real>::operator+=( const TVector4<Real> & rhs ) { X += rhs.X; Y += rhs.Y; Z += rhs.Z; W += rhs.W; return (*this); }
template<typename Real> inline TVector4<Real> & TVector4<Real>::operator-=( const TVector4<Real> & rhs ) { X -= rhs.X; Y -= rhs.Y; Z -= rhs.Z; W -= rhs.W; return (*this); }

template<typename Real>
inline Real TVector4<Real>::operator*( const TVector4<Real> & rhs ) const {
    return ( (X * rhs.X) + (Y * rhs.Y) + (Z * rhs.Z) + (W * rhs.W) );
}

template<typename Real>
inline TVector4<Real> TVector4<Real>::operator^( const TVector4<Real> & rhs ) const {
    return TVector4<Real> (
        (Y * rhs.Z) - (Z * rhs.Y),
        (Z * rhs.X) - (X * rhs.Z),
        (X * rhs.Y) - (Y * rhs.X),
        MathFunction<Real>::Zero
    );
}

template<typename Real>
inline Real TVector4<Real>::NormSqr() const {
    return ( X*X + Y*Y + Z*Z + W*W );
}
template<typename Real>
inline Real TVector4<Real>::Norm() const {
    return MathRealFn->Sqrt( NormSqr() );
}
template<typename Real>
inline Real TVector4<Real>::InvNormSqr() const {
    return MathRealFn->Invert( NormSqr() );
}
template<typename Real>
inline Real TVector4<Real>::InvNorm() const {
    return MathRealFn->InvSqrt( NormSqr() );
}

template<typename Real>
inline Real TVector4<Real>::Normalize() {
    Real fNorm = Norm(), fInvNorm;
    if ( MathRealFn->EqualsZero(fNorm) ) {
        X = MathFunction<Real>::Zero;
        Y = MathFunction<Real>::Zero;
        Z = MathFunction<Real>::Zero;
        W = MathFunction<Real>::Zero;
        return MathFunction<Real>::Zero;
    } else {
        fInvNorm = MathRealFn->Invert(fNorm);
        X *= fInvNorm;
        Y *= fInvNorm;
        Z *= fInvNorm;
        W *= fInvNorm;
        return fNorm;
    }
}

template<typename Real>
inline TVector4<Real> TVector4<Real>::ProjectToNormal( const TVector4<Real> & vNormal ) const {
    return ( vNormal * ( (*this) * vNormal ) );
}
template<typename Real>
inline TVector4<Real> TVector4<Real>::ProjectToPlane( const TVector4<Real> & vNormal ) const {
    return ( (*this) - ProjectToNormal(vNormal) );
}

template<typename Real>
Void TVector4<Real>::OrthoNormalize( TVector4<Real> & vI, TVector4<Real> & vJ, TVector4<Real> & vK )
{
    // Gram-Schmidt OrthoNormalization
    vI.Normalize();

    Real fDotI = (vI * vJ);
    vJ -= (vI * fDotI);
    vJ.Normalize();

    Real fDotJ = (vJ * vK);
    fDotI = (vI * vK);
    vK -= ( (vI * fDotI) + (vJ * fDotJ) );
    vK.Normalize();
}
template<typename Real>
Void TVector4<Real>::MakeComplementBasis( TVector4<Real> & vI, TVector4<Real> & vJ, const TVector4<Real> & vK )
{
    Real fInvNorm;
    if ( MathRealFn->Abs(vK.X) >= MathRealFn->Abs(vK.Y) ) {
        fInvNorm = MathRealFn->InvSqrt( (vK.X * vK.X) + (vK.Z * vK.Z) );
        vI.X = -(vK.Z * fInvNorm);
        vI.Y = MathFunction<Real>::Zero;
        vI.Z = +(vK.X * fInvNorm);
        vJ.X = (vK.Y * vI.Z);
        vJ.Y = ( (vK.Z * vI.X) - (vK.X * vI.Z) );
        vJ.Z = -(vK.Y * vI.X);
    } else {
        fInvNorm = MathRealFn->InvSqrt( (vK.Y * vK.Y) + (vK.Z * vK.Z) );
        vI.X = MathFunction<Real>::Zero;
        vI.Y = +(vK.Z * fInvNorm);
        vI.Z = -(vK.Y * fInvNorm);
        vJ.X = ( (vK.Y * vI.Z) - (vK.Z * vI.Y) );
        vJ.Y = -(vK.X * vI.Z);
        vJ.Z = (vK.X * vI.Y);
    }
}

