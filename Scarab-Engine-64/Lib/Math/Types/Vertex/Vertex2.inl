/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Vertex/Vertex2.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : 2D vertex
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// TVertex2 implementation
template<typename Real> const TVertex2<Real> TVertex2<Real>::Null = TVertex2<Real>( MathFunction<Real>::Zero, MathFunction<Real>::Zero );

template<typename Real> TVertex2<Real>::TVertex2()                                 {}
template<typename Real> TVertex2<Real>::TVertex2( const Real & x, const Real & y ) { X = x; Y = y; }
template<typename Real> TVertex2<Real>::TVertex2( const Real vArr[2] )             { X = vArr[0]; Y = vArr[1]; }
template<typename Real> TVertex2<Real>::TVertex2( const TVertex2<Real> & rhs )     { X = rhs.X; Y = rhs.Y; }
template<typename Real> TVertex2<Real>::TVertex2( const TVertex3<Real> & rhs )     { X = rhs.X; Y = rhs.Y; }
template<typename Real> TVertex2<Real>::TVertex2( const TVertex4<Real> & rhs )     { X = rhs.X; Y = rhs.Y; }
template<typename Real> TVertex2<Real>::~TVertex2()                                {}

template<typename Real> inline TVertex2<Real> & TVertex2<Real>::operator=( const TVertex2<Real> & rhs ) { X = rhs.X; Y = rhs.Y; return (*this); }

template<typename Real> inline TVertex2<Real>::operator Real*() const       { return (Real*)this; }
template<typename Real> inline TVertex2<Real>::operator const Real*() const { return (const Real*)this; }

template<typename Real> inline TVector2<Real> TVertex2<Real>::ToVector() const { return TVector2<Real>( X, Y ); }

template<typename Real> inline Real & TVertex2<Real>::operator[]( Int i )              { return *( ((Real*)this) + i ); }
template<typename Real> inline const Real & TVertex2<Real>::operator[]( Int i ) const  { return *( ((const Real*)this) + i ); }
template<typename Real> inline Real & TVertex2<Real>::operator[]( UInt i )             { return *( ((Real*)this) + i ); }
template<typename Real> inline const Real & TVertex2<Real>::operator[]( UInt i ) const { return *( ((const Real*)this) + i ); }

template<typename Real> inline TVertex2<Real> TVertex2<Real>::operator+() const { return TVertex2<Real>( X, Y ); }
template<typename Real> inline TVertex2<Real> TVertex2<Real>::operator-() const { return TVertex2<Real>( -X, -Y ); }

template<typename Real>
inline Bool TVertex2<Real>::operator==( const TVertex2<Real> & rhs ) const {
    return ( MathRealFn->Equals(X,rhs.X) && MathRealFn->Equals(Y,rhs.Y) );
}
template<typename Real>
inline Bool TVertex2<Real>::operator!=( const TVertex2<Real> & rhs ) const {
    return ( !(MathRealFn->Equals(X,rhs.X)) || !(MathRealFn->Equals(Y,rhs.Y)) );
}

template<typename Real> inline TVertex2<Real> TVertex2<Real>::operator+( const Real & rhs ) const { return TVertex2<Real>( X + rhs, Y + rhs ); }
template<typename Real> inline TVertex2<Real> TVertex2<Real>::operator-( const Real & rhs ) const { return TVertex2<Real>( X - rhs, Y - rhs ); }
template<typename Real> inline TVertex2<Real> TVertex2<Real>::operator*( const Real & rhs ) const { return TVertex2<Real>( X * rhs, Y * rhs ); }
template<typename Real> inline TVertex2<Real> TVertex2<Real>::operator/( const Real & rhs ) const { return TVertex2<Real>( X / rhs, Y / rhs ); }

template<typename Real> inline TVertex2<Real> & TVertex2<Real>::operator+=( const Real & rhs ) { X += rhs; Y += rhs; return (*this); }
template<typename Real> inline TVertex2<Real> & TVertex2<Real>::operator-=( const Real & rhs ) { X -= rhs; Y -= rhs; return (*this); }
template<typename Real> inline TVertex2<Real> & TVertex2<Real>::operator*=( const Real & rhs ) { X *= rhs; Y *= rhs; return (*this); }
template<typename Real> inline TVertex2<Real> & TVertex2<Real>::operator/=( const Real & rhs ) { X /= rhs; Y /= rhs; return (*this); }

template<typename Real> inline TVector2<Real> TVertex2<Real>::operator-( const TVertex2<Real> & rhs ) const { return TVector2<Real>( X - rhs.X, Y - rhs.Y ); }

template<typename Real> inline TVertex2<Real> TVertex2<Real>::operator+( const TVector2<Real> & rhs ) const { return TVertex2<Real>( X + rhs.X, Y + rhs.Y ); }
template<typename Real> inline TVertex2<Real> TVertex2<Real>::operator-( const TVector2<Real> & rhs ) const { return TVertex2<Real>( X - rhs.X, Y - rhs.Y ); }

template<typename Real> inline TVertex2<Real> & TVertex2<Real>::operator+=( const TVector2<Real> & rhs ) { X += rhs.X; Y += rhs.Y; return (*this); }
template<typename Real> inline TVertex2<Real> & TVertex2<Real>::operator-=( const TVector2<Real> & rhs ) { X -= rhs.X; Y -= rhs.Y; return (*this); }

template<typename Real>
inline Real TVertex2<Real>::DistSqr() const {
    return ( X*X + Y*Y );
}
template<typename Real>
inline Real TVertex2<Real>::Dist() const {
    return MathRealFn->Sqrt( DistSqr() );
}
template<typename Real>
inline Real TVertex2<Real>::InvDistSqr() const {
    return MathRealFn->Invert( DistSqr() );
}
template<typename Real>
inline Real TVertex2<Real>::InvDist() const {
    return MathRealFn->InvSqrt( DistSqr() );
}

template<typename Real>
inline Void TVertex2<Real>::FromPolar( const Real & fRadius, const Real & fTheta ) {
	X = fRadius * MathRealFn->Cos( fTheta );
	Y = fRadius * MathRealFn->Sin( fTheta );
}
template<typename Real>
inline Void TVertex2<Real>::FromPolar( const TVertex2<Real> & vPolar ) {
    FromPolar( vPolar.X, vPolar.Y );
}
template<typename Real>
inline Void TVertex2<Real>::ToPolar( Real & outRadius, Real & outTheta ) const {
    outRadius = Dist();
	outTheta = MathRealFn->ArcTan2(Y, X);
}
template<typename Real>
inline Void TVertex2<Real>::ToPolar( TVertex2<Real> & outPolar ) const {
    ToPolar( outPolar.X, outPolar.Y );
}

