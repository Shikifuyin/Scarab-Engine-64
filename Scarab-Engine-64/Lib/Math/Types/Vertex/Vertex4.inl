/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Vertex/Vertex4.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Homogeneous 4D vertex
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// TVertex4 implementation
template<typename Real> const TVertex4<Real> TVertex4<Real>::Null = TVertex4<Real>( MathFunction<Real>::Zero, MathFunction<Real>::Zero, MathFunction<Real>::Zero, MathFunction<Real>::One );

template<typename Real> TVertex4<Real>::TVertex4()                                                                 {}
template<typename Real> TVertex4<Real>::TVertex4( const Real & x, const Real & y, const Real & z, const Real & w ) { X = x; Y = y; Z = z; W = w; }
template<typename Real> TVertex4<Real>::TVertex4( const Real vArr[4] )                                             { X = vArr[0]; Y = vArr[1]; Z = vArr[2]; W = vArr[3]; }
template<typename Real> TVertex4<Real>::TVertex4( const TVertex2<Real> & rhs )                                     { X = rhs.X; Y = rhs.Y; Z = MathFunction<Real>::Zero; W = MathFunction<Real>::One; }
template<typename Real> TVertex4<Real>::TVertex4( const TVertex3<Real> & rhs )                                     { X = rhs.X; Y = rhs.Y; Z = rhs.Z; W = MathFunction<Real>::One; }
template<typename Real> TVertex4<Real>::TVertex4( const TVertex4<Real> & rhs )                                     { X = rhs.X; Y = rhs.Y; Z = rhs.Z; W = rhs.W; }
template<typename Real> TVertex4<Real>::~TVertex4()                                                                {}

template<typename Real> inline TVertex4<Real> & TVertex4<Real>::operator=( const TVertex4<Real> & rhs ) { X = rhs.X; Y = rhs.Y; Z = rhs.Z; W = rhs.W; return (*this); }

template<typename Real> inline TVertex4<Real>::operator Real*() const       { return (Real*)this; }
template<typename Real> inline TVertex4<Real>::operator const Real*() const { return (const Real*)this; }

template<typename Real> inline TVector4<Real> TVertex4<Real>::ToVector() const { return TVector4<Real>( X, Y, Z, MathFunction<Real>::Zero ); }

template<typename Real> inline Real & TVertex4<Real>::operator[]( Int i )              { return *( ((Real*)this) + i ); }
template<typename Real> inline const Real & TVertex4<Real>::operator[]( Int i ) const  { return *( ((const Real*)this) + i ); }
template<typename Real> inline Real & TVertex4<Real>::operator[]( UInt i )             { return *( ((Real*)this) + i ); }
template<typename Real> inline const Real & TVertex4<Real>::operator[]( UInt i ) const { return *( ((const Real*)this) + i ); }

template<typename Real> inline TVertex4<Real> TVertex4<Real>::operator+() const { return TVertex4<Real>( X, Y, Z, W ); }
template<typename Real> inline TVertex4<Real> TVertex4<Real>::operator-() const { return TVertex4<Real>( -X, -Y, -Z, W ); }

template<typename Real>
inline Bool TVertex4<Real>::operator==( const TVertex4<Real> & rhs ) const {
    return ( MathRealFn->Equals(X,rhs.X) && MathRealFn->Equals(Y,rhs.Y) && MathRealFn->Equals(Z,rhs.Z) && MathRealFn->Equals(W,rhs.W) );
}
template<typename Real>
inline Bool TVertex4<Real>::operator!=( const TVertex4<Real> & rhs ) const {
    return ( !(MathRealFn->Equals(X,rhs.X)) || !(MathRealFn->Equals(Y,rhs.Y)) || !(MathRealFn->Equals(Z,rhs.Z)) || !(MathRealFn->Equals(W,rhs.W)) );
}

template<typename Real> inline TVertex4<Real> TVertex4<Real>::operator+( const Real & rhs ) const { return TVertex4<Real>( X + rhs, Y + rhs, Z + rhs, W ); }
template<typename Real> inline TVertex4<Real> TVertex4<Real>::operator-( const Real & rhs ) const { return TVertex4<Real>( X - rhs, Y - rhs, Z - rhs, W ); }
template<typename Real> inline TVertex4<Real> TVertex4<Real>::operator*( const Real & rhs ) const { return TVertex4<Real>( X * rhs, Y * rhs, Z * rhs, W ); }
template<typename Real> inline TVertex4<Real> TVertex4<Real>::operator/( const Real & rhs ) const { return TVertex4<Real>( X / rhs, Y / rhs, Z / rhs, W ); }

template<typename Real> inline TVertex4<Real> & TVertex4<Real>::operator+=( const Real & rhs ) { X += rhs; Y += rhs; Z += rhs; return (*this); }
template<typename Real> inline TVertex4<Real> & TVertex4<Real>::operator-=( const Real & rhs ) { X -= rhs; Y -= rhs; Z -= rhs; return (*this); }
template<typename Real> inline TVertex4<Real> & TVertex4<Real>::operator*=( const Real & rhs ) { X *= rhs; Y *= rhs; Z *= rhs; return (*this); }
template<typename Real> inline TVertex4<Real> & TVertex4<Real>::operator/=( const Real & rhs ) { X /= rhs; Y /= rhs; Z /= rhs; return (*this); }

template<typename Real> inline TVector4<Real> TVertex4<Real>::operator-( const TVertex4<Real> & rhs ) const { return TVector4<Real>( X - rhs.X, Y - rhs.Y, Z - rhs.Z, W - rhs.W ); }

template<typename Real> inline TVertex4<Real> TVertex4<Real>::operator+( const TVector4<Real> & rhs ) const { return TVertex4<Real>( X + rhs.X, Y + rhs.Y, Z + rhs.Z, W + rhs.W ); }
template<typename Real> inline TVertex4<Real> TVertex4<Real>::operator-( const TVector4<Real> & rhs ) const { return TVertex4<Real>( X - rhs.X, Y - rhs.Y, Z - rhs.Z, W - rhs.W ); }

template<typename Real> inline TVertex4<Real> & TVertex4<Real>::operator+=( const TVector4<Real> & rhs ) { X += rhs.X; Y += rhs.Y; Z += rhs.Z; W += rhs.W; return (*this); }
template<typename Real> inline TVertex4<Real> & TVertex4<Real>::operator-=( const TVector4<Real> & rhs ) { X -= rhs.X; Y -= rhs.Y; Z -= rhs.Z; W -= rhs.W; return (*this); }

template<typename Real>
inline Real TVertex4<Real>::DistSqr() const {
    return ( X*X + Y*Y + Z*Z );
}
template<typename Real>
inline Real TVertex4<Real>::Dist() const {
    return MathRealFn->Sqrt( DistSqr() );
}
template<typename Real>
inline Real TVertex4<Real>::InvDistSqr() const {
    return MathRealFn->Invert( DistSqr() );
}
template<typename Real>
inline Real TVertex4<Real>::InvDist() const {
    return MathRealFn->InvSqrt( DistSqr() );
}
template<typename Real>
inline Void TVertex4<Real>::NormalizeW() {
    Real fInvW = MathRealFn->Invert(W);
    X *= fInvW;
    Y *= fInvW;
    Z *= fInvW;
    W = MathFunction<Real>::One;
}


