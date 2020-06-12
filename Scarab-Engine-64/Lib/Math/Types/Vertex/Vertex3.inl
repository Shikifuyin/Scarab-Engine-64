/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Vertex/Vertex3.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : 3D vertex
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// TVertex3 implementation
template<typename Real> const TVertex3<Real> TVertex3<Real>::Null = TVertex3<Real>( MathFunction<Real>::Zero, MathFunction<Real>::Zero, MathFunction<Real>::Zero );

template<typename Real> TVertex3<Real>::TVertex3()                                                 {}
template<typename Real> TVertex3<Real>::TVertex3( const Real & x, const Real & y, const Real & z ) { X = x; Y = y; Z = z; }
template<typename Real> TVertex3<Real>::TVertex3( const Real vArr[3] )                             { X = vArr[0]; Y = vArr[1]; Z = vArr[2]; }
template<typename Real> TVertex3<Real>::TVertex3( const TVertex2<Real> & rhs )                     { X = rhs.X; Y = rhs.Y; Z = MathFunction<Real>::Zero; }
template<typename Real> TVertex3<Real>::TVertex3( const TVertex3<Real> & rhs )                     { X = rhs.X; Y = rhs.Y; Z = rhs.Z; }
template<typename Real> TVertex3<Real>::TVertex3( const TVertex4<Real> & rhs )                     { X = rhs.X; Y = rhs.Y; Z = rhs.Z; }
template<typename Real> TVertex3<Real>::~TVertex3()                                                {}

template<typename Real> inline TVertex3<Real> & TVertex3<Real>::operator=( const TVertex3<Real> & rhs ) { X = rhs.X; Y = rhs.Y; Z = rhs.Z; return (*this); }

template<typename Real> inline TVertex3<Real>::operator Real*() const       { return (Real*)this; }
template<typename Real> inline TVertex3<Real>::operator const Real*() const { return (const Real*)this; }

template<typename Real> inline TVector3<Real> TVertex3<Real>::ToVector() const { return TVector3<Real>( X, Y, Z ); }

template<typename Real> inline Real & TVertex3<Real>::operator[]( Int i )              { return *( ((Real*)this) + i ); }
template<typename Real> inline const Real & TVertex3<Real>::operator[]( Int i ) const  { return *( ((const Real*)this) + i ); }
template<typename Real> inline Real & TVertex3<Real>::operator[]( UInt i )             { return *( ((Real*)this) + i ); }
template<typename Real> inline const Real & TVertex3<Real>::operator[]( UInt i ) const { return *( ((const Real*)this) + i ); }

template<typename Real> inline TVertex3<Real> TVertex3<Real>::operator+() const { return TVertex3<Real>( X, Y, Z ); }
template<typename Real> inline TVertex3<Real> TVertex3<Real>::operator-() const { return TVertex3<Real>( -X, -Y, -Z ); }

template<typename Real>
inline Bool TVertex3<Real>::operator==( const TVertex3<Real> & rhs ) const {
    return ( MathRealFn->Equals(X,rhs.X) && MathRealFn->Equals(Y,rhs.Y) && MathRealFn->Equals(Z,rhs.Z) );
}
template<typename Real>
inline Bool TVertex3<Real>::operator!=( const TVertex3<Real> & rhs ) const {
    return ( !(MathRealFn->Equals(X,rhs.X)) || !(MathRealFn->Equals(Y,rhs.Y)) || !(MathRealFn->Equals(Z,rhs.Z)) );
}

template<typename Real> inline TVertex3<Real> TVertex3<Real>::operator+( const Real & rhs ) const { return TVertex3<Real>( X + rhs, Y + rhs, Z + rhs ); }
template<typename Real> inline TVertex3<Real> TVertex3<Real>::operator-( const Real & rhs ) const { return TVertex3<Real>( X - rhs, Y - rhs, Z - rhs ); }
template<typename Real> inline TVertex3<Real> TVertex3<Real>::operator*( const Real & rhs ) const { return TVertex3<Real>( X * rhs, Y * rhs, Z * rhs ); }
template<typename Real> inline TVertex3<Real> TVertex3<Real>::operator/( const Real & rhs ) const { return TVertex3<Real>( X / rhs, Y / rhs, Z / rhs ); }

template<typename Real> inline TVertex3<Real> & TVertex3<Real>::operator+=( const Real & rhs ) { X += rhs; Y += rhs; Z += rhs; return (*this); }
template<typename Real> inline TVertex3<Real> & TVertex3<Real>::operator-=( const Real & rhs ) { X -= rhs; Y -= rhs; Z -= rhs; return (*this); }
template<typename Real> inline TVertex3<Real> & TVertex3<Real>::operator*=( const Real & rhs ) { X *= rhs; Y *= rhs; Z *= rhs; return (*this); }
template<typename Real> inline TVertex3<Real> & TVertex3<Real>::operator/=( const Real & rhs ) { X /= rhs; Y /= rhs; Z /= rhs; return (*this); }

template<typename Real> inline TVector3<Real> TVertex3<Real>::operator-( const TVertex3<Real> & rhs ) const { return TVector3<Real>( X - rhs.X, Y - rhs.Y, Z - rhs.Z ); }

template<typename Real> inline TVertex3<Real> TVertex3<Real>::operator+( const TVector3<Real> & rhs ) const { return TVertex3<Real>( X + rhs.X, Y + rhs.Y, Z + rhs.Z ); }
template<typename Real> inline TVertex3<Real> TVertex3<Real>::operator-( const TVector3<Real> & rhs ) const { return TVertex3<Real>( X - rhs.X, Y - rhs.Y, Z - rhs.Z ); }

template<typename Real> inline TVertex3<Real> & TVertex3<Real>::operator+=( const TVector3<Real> & rhs ) { X += rhs.X; Y += rhs.Y; Z += rhs.Z; return (*this); }
template<typename Real> inline TVertex3<Real> & TVertex3<Real>::operator-=( const TVector3<Real> & rhs ) { X -= rhs.X; Y -= rhs.Y; Z -= rhs.Z; return (*this); }

template<typename Real>
inline Real TVertex3<Real>::DistSqr() const {
    return ( X*X + Y*Y + Z*Z );
}
template<typename Real>
inline Real TVertex3<Real>::Dist() const {
    return MathRealFn->Sqrt( DistSqr() );
}
template<typename Real>
inline Real TVertex3<Real>::InvDistSqr() const {
    return MathRealFn->Invert( DistSqr() );
}
template<typename Real>
inline Real TVertex3<Real>::InvDist() const {
    return MathRealFn->InvSqrt( DistSqr() );
}

template<typename Real>
inline Void TVertex3<Real>::FromCylindric( const Real & fRadius, const Real & fTheta, const Real & fZ ) {
    Real fSin = MathRealFn->Sin( fTheta );
    Real fCos = MathRealFn->Cos( fTheta );
	X = fRadius * fCos;
	Y = fRadius * fSin;
	Z = fZ;
}
template<typename Real>
inline Void TVertex3<Real>::FromCylindric( const TVertex3<Real> & vCylindric ) {
    FromCylindric( vCylindric.X, vCylindric.Y, vCylindric.Z );
}
template<typename Real>
inline Void TVertex3<Real>::ToCylindric( Real & outRadius, Real & outTheta, Real & outZ ) const {
    outRadius = Dist();
	outTheta = MathRealFn->ArcTan2(Y, X);
	outZ = Z;
}
template<typename Real>
inline Void TVertex3<Real>::ToCylindric( TVertex3<Real> & outCylindric ) const {
    ToCylindric( outCylindric.X, outCylindric.Y, outCylindric.Z );
}

template<typename Real>
inline Void TVertex3<Real>::FromSpherical( const Real & fRadius, const Real & fTheta, const Real & fPhi ) {
	Real fSinTheta = MathRealFn->Sin( fTheta );
    Real fCosTheta = MathRealFn->Cos( fTheta );
	Real fSinPhi = MathRealFn->Sin( fPhi );
    Real fCosPhi = MathRealFn->Cos( fPhi );
    Real fTmp = fRadius * fSinPhi;
	X = fTmp * fCosTheta;
	Y = fTmp * fSinTheta;
	Z = fRadius * fCosPhi;
}
template<typename Real>
inline Void TVertex3<Real>::FromSpherical( const TVertex3<Real> & vSpherical ) {
    FromSpherical( vSpherical.X, vSpherical.Y, vSpherical.Z );
}
template<typename Real>
inline Void TVertex3<Real>::ToSpherical( Real & outRadius, Real & outTheta, Real & outPhi ) const {
    outRadius = Dist();
	outTheta = MathRealFn->ArcTan2(Y, X);
	outPhi = MathRealFn->ArcCos(Z / outRadius);
}
template<typename Real>
inline Void TVertex3<Real>::ToSpherical( TVertex3<Real> & outSpherical ) const {
    ToSpherical( outSpherical.X, outSpherical.Y, outSpherical.Z );
}


