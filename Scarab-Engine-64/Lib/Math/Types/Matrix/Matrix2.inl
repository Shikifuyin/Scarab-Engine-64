/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Matrix/Matrix2.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : 2D matrix
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// TMatrix2 implementation
template<typename Real>
const TMatrix2<Real> TMatrix2<Real>::Null = TMatrix2<Real>( MathFunction<Real>::Zero, MathFunction<Real>::Zero,
                                                            MathFunction<Real>::Zero, MathFunction<Real>::Zero );
template<typename Real>
const TMatrix2<Real> TMatrix2<Real>::Identity = TMatrix2<Real>( MathFunction<Real>::One,  MathFunction<Real>::Zero,
                                                                MathFunction<Real>::Zero, MathFunction<Real>::One );

template<typename Real>
TMatrix2<Real>::TMatrix2()
{
    // nothing to do
}
template<typename Real>
TMatrix2<Real>::TMatrix2( const Real & a00, const Real & a01,
                          const Real & a10, const Real & a11 )
{
    m00 = a00; m01 = a01;
    m10 = a10; m11 = a11;
}
template<typename Real>
TMatrix2<Real>::TMatrix2( const Real v0[2], const Real v1[2], Bool bRows )
{
	if ( bRows ) {
        m00 = v0[0]; m01 = v0[1];
        m10 = v1[0]; m11 = v1[1];
    } else {
        m00 = v0[0]; m01 = v1[0];
        m10 = v0[1]; m11 = v1[1];
    }
}
template<typename Real>
TMatrix2<Real>::TMatrix2( const Real arrMat[4], Bool bRows )
{
	if ( bRows ) {
        m00 = arrMat[0];  m01 = arrMat[1];
        m10 = arrMat[2];  m11 = arrMat[3];
    } else {
        m00 = arrMat[0];  m01 = arrMat[2];
        m10 = arrMat[1];  m11 = arrMat[3];
    }
}
template<typename Real>
TMatrix2<Real>::TMatrix2( const TVector2<Real> & v0, const TVector2<Real> & v1, Bool bRows )
{
	if ( bRows ) {
        m00 = v0.X; m01 = v0.Y;
        m10 = v1.X; m11 = v1.Y;
    } else {
        m00 = v0.X; m01 = v1.X;
        m10 = v0.Y; m11 = v1.Y;
    }
}
template<typename Real>
TMatrix2<Real>::TMatrix2( const TVector2<Real> vMat[2], Bool bRows )
{
	if ( bRows ) {
        m00 = vMat[0].X; m01 = vMat[0].Y;
        m10 = vMat[1].X; m11 = vMat[1].Y;
    } else {
        m00 = vMat[0].X; m01 = vMat[1].X;
        m10 = vMat[0].Y; m11 = vMat[1].Y;
    }
}
template<typename Real>
TMatrix2<Real>::TMatrix2( const TMatrix2<Real> & rhs )
{
    m00 = rhs.m00; m01 = rhs.m01;
    m10 = rhs.m10; m11 = rhs.m11;
}
template<typename Real>
TMatrix2<Real>::TMatrix2( const TMatrix3<Real> & rhs )
{
    m00 = rhs.m00; m01 = rhs.m01;
    m10 = rhs.m10; m11 = rhs.m11;
}
template<typename Real>
TMatrix2<Real>::TMatrix2( const TMatrix4<Real> & rhs )
{
    m00 = rhs.m00; m01 = rhs.m01;
    m10 = rhs.m10; m11 = rhs.m11;
}
template<typename Real>
TMatrix2<Real>::~TMatrix2()
{
    // nothing to do
}

template<typename Real>
inline TMatrix2<Real> & TMatrix2<Real>::operator=( const TMatrix2<Real> & rhs ) {
    m00 = rhs.m00; m01 = rhs.m01;
    m10 = rhs.m10; m11 = rhs.m11;
    return (*this);
}

template<typename Real> inline TMatrix2<Real>::operator Real*() const       { return (Real*)this; }
template<typename Real> inline TMatrix2<Real>::operator const Real*() const { return (const Real*)this; }

template<typename Real> inline Real & TMatrix2<Real>::operator[]( Int i )              { return *( ((Real*)this) + i ); }
template<typename Real> inline const Real & TMatrix2<Real>::operator[]( Int i ) const  { return *( ((const Real*)this) + i ); }
template<typename Real> inline Real & TMatrix2<Real>::operator[]( UInt i )             { return *( ((Real*)this) + i ); }
template<typename Real> inline const Real & TMatrix2<Real>::operator[]( UInt i ) const { return *( ((const Real*)this) + i ); }

template<typename Real> inline Real & TMatrix2<Real>::operator()( Int iRow, Int iColumn )               { return *( ((Real*)this) + ((iColumn<<1) + iRow) ); }
template<typename Real> inline const Real & TMatrix2<Real>::operator()( Int iRow, Int iColumn ) const   { return *( ((const Real*)this) + ((iColumn<<1) + iRow) ); }
template<typename Real> inline Real & TMatrix2<Real>::operator()( UInt iRow, UInt iColumn )             { return *( ((Real*)this) + ((iColumn<<1) + iRow) ); }
template<typename Real> inline const Real & TMatrix2<Real>::operator()( UInt iRow, UInt iColumn ) const { return *( ((const Real*)this) + ((iColumn<<1) + iRow) ); }

template<typename Real>
inline TMatrix2<Real> TMatrix2<Real>::operator+() const {
    return TMatrix2<Real>( m00, m01,
                           m10, m11 );
}
template<typename Real>
inline TMatrix2<Real> TMatrix2<Real>::operator-() const {
    return TMatrix2<Real>( -m00, -m01,
                           -m10, -m11 );
}

template<typename Real>
Bool TMatrix2<Real>::operator==( const TMatrix2<Real> & rhs ) const
{
    return ( MathRealFn->Equals(m00, rhs.m00) && MathRealFn->Equals(m01, rhs.m01) &&
             MathRealFn->Equals(m10, rhs.m10) && MathRealFn->Equals(m11, rhs.m11) );
}
template<typename Real>
Bool TMatrix2<Real>::operator!=( const TMatrix2<Real> & rhs ) const
{
    return ( !(MathRealFn->Equals(m00, rhs.m00)) || !(MathRealFn->Equals(m01, rhs.m01)) ||
             !(MathRealFn->Equals(m10, rhs.m10)) || !(MathRealFn->Equals(m11, rhs.m11)) );
}

template<typename Real>
TMatrix2<Real> TMatrix2<Real>::operator*( const Real & rhs ) const
{
    return TMatrix2<Real>( m00 * rhs, m01 * rhs,
                           m10 * rhs, m11 * rhs );
}
template<typename Real>
TMatrix2<Real> TMatrix2<Real>::operator/( const Real & rhs ) const
{
    // You should never use this !
    return TMatrix2<Real>( m00 / rhs, m01 / rhs,
                           m10 / rhs, m11 / rhs );
}

template<typename Real>
TMatrix2<Real> & TMatrix2<Real>::operator*=( const Real & rhs )
{
    m00 *= rhs; m01 *= rhs;
    m10 *= rhs; m11 *= rhs;
    return (*this);
}
template<typename Real>
TMatrix2<Real> & TMatrix2<Real>::operator/=( const Real & rhs )
{
    // You should never use this !
    m00 /= rhs; m01 /= rhs;
    m10 /= rhs; m11 /= rhs;
    return (*this);
}

template<typename Real>
TVertex2<Real> TMatrix2<Real>::operator*( const TVertex2<Real> & rhs ) const
{
    return TVertex2<Real>( (m00 * rhs.X + m01 * rhs.Y),
                           (m10 * rhs.X + m11 * rhs.Y) );
}

template<typename Real>
TVector2<Real> TMatrix2<Real>::operator*( const TVector2<Real> & rhs ) const
{
    return TVector2<Real>( (m00 * rhs.X + m01 * rhs.Y),
                           (m10 * rhs.X + m11 * rhs.Y) );
}

template<typename Real>
TMatrix2<Real> TMatrix2<Real>::operator+( const TMatrix2<Real> & rhs ) const
{
    return TMatrix2<Real>( m00 + rhs.m00, m01 + rhs.m01,
                           m10 + rhs.m10, m11 + rhs.m11 );
}
template<typename Real>
TMatrix2<Real> TMatrix2<Real>::operator-( const TMatrix2<Real> & rhs ) const
{
    return TMatrix2<Real>( m00 - rhs.m00, m01 - rhs.m01,
                           m10 - rhs.m10, m11 - rhs.m11 );
}
template<typename Real>
TMatrix2<Real> TMatrix2<Real>::operator*( const TMatrix2<Real> & rhs ) const
{
    return TMatrix2<Real> (
        (m00 * rhs.m00 + m01 * rhs.m10), (m00 * rhs.m01 + m01 * rhs.m11),
	    (m10 * rhs.m00 + m11 * rhs.m10), (m10 * rhs.m01 + m11 * rhs.m11)
    );
}

template<typename Real>
TMatrix2<Real> & TMatrix2<Real>::operator+=( const TMatrix2<Real> & rhs )
{
    m00 += rhs.m00; m01 += rhs.m01;
	m10 += rhs.m10; m11 += rhs.m11;
    return (*this);
}
template<typename Real>
TMatrix2<Real> & TMatrix2<Real>::operator-=( const TMatrix2<Real> & rhs )
{
    m00 -= rhs.m00; m01 -= rhs.m01;
	m10 -= rhs.m10; m11 -= rhs.m11;
    return (*this);
}
template<typename Real>
TMatrix2<Real> & TMatrix2<Real>::operator*=( const TMatrix2<Real> & rhs )
{
    Real f0 = (m00 * rhs.m00 + m01 * rhs.m10);
    Real f1 = (m00 * rhs.m01 + m01 * rhs.m11);
    m00 = f0; m01 = f1;

    f0 = (m10 * rhs.m00 + m11 * rhs.m10);
    f1 = (m10 * rhs.m01 + m11 * rhs.m11);
    m10 = f0; m11 = f1;

    return (*this);
}

template<typename Real>
inline Void TMatrix2<Real>::GetRow( TVector2<Real> & outRow, UInt iRow ) const {
    Assert( iRow < 2 );
    const Real * Values = ( ((const Real*)this) + iRow );
    outRow.X = Values[0];
    outRow.Y = Values[2];
}
template<typename Real>
inline Void TMatrix2<Real>::SetRow( UInt iRow, const Real & fRow0, const Real & fRow1 ) {
    Assert( iRow < 2 );
    Real * Values = ( ((Real*)this) + iRow );
    Values[0] = fRow0;
    Values[2] = fRow1;
}
template<typename Real>
inline Void TMatrix2<Real>::SetRow( UInt iRow, const Real vRow[2] ) {
    Assert( iRow < 2 );
    Real * Values = ( ((Real*)this) + iRow );
    Values[0] = vRow[0];
    Values[2] = vRow[1];
}
template<typename Real>
inline Void TMatrix2<Real>::SetRow( UInt iRow, const TVector2<Real> & vRow ) {
    Assert( iRow < 2 );
    Real * Values = ( ((Real*)this) + iRow );
    Values[0] = vRow.X;
    Values[2] = vRow.Y;
}

template<typename Real>
inline Void TMatrix2<Real>::GetColumn( TVector2<Real> & outColumn, UInt iColumn ) const {
    Assert( iColumn < 2 );
    const Real * Values = ( ((const Real*)this) + (iColumn<<1) );
    outColumn.X = Values[0];
    outColumn.Y = Values[1];
}
template<typename Real>
inline Void TMatrix2<Real>::SetColumn( UInt iColumn, const Real & fCol0, const Real & fCol1 ) {
    Assert( iColumn < 2 );
    Real * Values = ( ((Real*)this) + (iColumn<<1) );
    Values[0] = fCol0;
    Values[1] = fCol1;
}
template<typename Real>
inline Void TMatrix2<Real>::SetColumn( UInt iColumn, const Real vCol[2] ) {
    Assert( iColumn < 2 );
    Real * Values = ( ((Real*)this) + (iColumn<<1) );
    Values[0] = vCol[0];
    Values[1] = vCol[1];
}
template<typename Real>
inline Void TMatrix2<Real>::SetColumn( UInt iColumn, const TVector2<Real> & vCol ) {
    Assert( iColumn < 2 );
    Real * Values = ( ((Real*)this) + (iColumn<<1) );
    Values[0] = vCol.X;
    Values[1] = vCol.Y;
}

template<typename Real>
inline Void TMatrix2<Real>::GetDiagonal( TVector2<Real> & outDiag ) const {
    outDiag.X = m00;
    outDiag.Y = m11;
}
template<typename Real>
inline Void TMatrix2<Real>::SetDiagonal( const Real & fDiag0, const Real & fDiag1 ) {
    m00 = fDiag0;
    m11 = fDiag1;
}
template<typename Real>
inline Void TMatrix2<Real>::SetDiagonal( const Real vDiag[2] ) {
    m00 = vDiag[0];
    m11 = vDiag[1];
}
template<typename Real>
inline Void TMatrix2<Real>::SetDiagonal( const TVector2<Real> & vDiag ) {
    m00 = vDiag.X;
    m11 = vDiag.Y;
}

template<typename Real>
inline Void TMatrix2<Real>::MakeNull() {
    m00 = MathFunction<Real>::Zero; m01 = MathFunction<Real>::Zero;
    m10 = MathFunction<Real>::Zero; m11 = MathFunction<Real>::Zero;
}
template<typename Real>
inline Void TMatrix2<Real>::MakeIdentity() {
    m00 = MathFunction<Real>::One;  m01 = MathFunction<Real>::Zero;
    m10 = MathFunction<Real>::Zero; m11 = MathFunction<Real>::One;
}

template<typename Real>
inline Void TMatrix2<Real>::MakeDiagonal( const Real & fDiag0, const Real & fDiag1 ) {
    m00 = fDiag0;                   m01 = MathFunction<Real>::Zero;
    m10 = MathFunction<Real>::Zero; m11 = fDiag1;
}
template<typename Real>
inline Void TMatrix2<Real>::MakeDiagonal( const Real vDiag[2] ) {
    m00 = vDiag[0];                 m01 = MathFunction<Real>::Zero;
    m10 = MathFunction<Real>::Zero; m11 = vDiag[1];
}
template<typename Real>
inline Void TMatrix2<Real>::MakeDiagonal( const TVector2<Real> & vDiagonal ) {
    m00 = vDiagonal.X;              m01 = MathFunction<Real>::Zero;
    m10 = MathFunction<Real>::Zero; m11 = vDiagonal.Y;
}

template<typename Real>
inline Void TMatrix2<Real>::MakeScale( const TVector2<Real> & vScale ) {
    m00 = vScale.X;                 m01 = MathFunction<Real>::Zero;
    m10 = MathFunction<Real>::Zero; m11 = vScale.Y;
}

template<typename Real>
inline Void TMatrix2<Real>::MakeBasis( const TVector2<Real> & vI, const TVector2<Real> & vJ ) {
    m00 = vI.X; m01 = vJ.X;
    m10 = vI.Y; m11 = vJ.Y;
}

template<typename Real>
inline Void TMatrix2<Real>::MakeRotate( const Real & fAngle ) {
    Real fSin = MathRealFn->Sin( fAngle );
    Real fCos = MathRealFn->Cos( fAngle );
    m00 = fCos;	m01 = -fSin;
	m10 = fSin;	m11 = fCos;
}

template<typename Real>
inline Void TMatrix2<Real>::MakeTensorProduct( const TVector2<Real> & vU, const TVector2<Real> & vV ) {
    m00 = (vU.X * vV.X); m01 = (vU.X * vV.Y);
    m10 = (vU.Y * vV.X); m11 = (vU.Y * vV.Y);
}

template<typename Real>
inline Void TMatrix2<Real>::GetAngle( Real & outAngle ) const {
    outAngle = MathRealFn->ArcTan2(m10, m00);
}

template<typename Real>
inline Void TMatrix2<Real>::Transpose( TMatrix2<Real> & outTransposedMatrix ) const {
    outTransposedMatrix.m00 = m00; outTransposedMatrix.m01 = m10;
    outTransposedMatrix.m10 = m01; outTransposedMatrix.m11 = m11;
}

template<typename Real>
inline Real TMatrix2<Real>::Trace() const {
    return ( m00 + m11 );
}
template<typename Real>
inline Real TMatrix2<Real>::Determinant() const {
    return ( m00 * m11 - m10 * m01 );
}

template<typename Real>
inline Void TMatrix2<Real>::Minor( Real & outMinor, UInt iRow, UInt iColumn ) const {
    const Real * Values = ( (const Real*)this );
    outMinor = Values[((1-iColumn) << 1) + (1-iRow)];
}

template<typename Real>
inline Void TMatrix2<Real>::Adjoint( TMatrix2<Real> & outAdjointMatrix ) const {
    outAdjointMatrix.m00 = +m11; outAdjointMatrix.m01 = -m01;
    outAdjointMatrix.m10 = -m10; outAdjointMatrix.m11 = +m00;
}

template<typename Real>
inline Bool TMatrix2<Real>::IsInvertible( Real fZeroTolerance ) const {
    return ( MathRealFn->Abs(Determinant()) >= fZeroTolerance );
}
template<typename Real>
Bool TMatrix2<Real>::Invert( TMatrix2<Real> & outInvMatrix, Real fZeroTolerance ) const
{
    Real fInvDet = ( m00 * m11 - m10 * m01 );
    if ( MathRealFn->Abs(fInvDet) < fZeroTolerance )
        return false;
    fInvDet = MathRealFn->Invert(fInvDet);

    outInvMatrix.m00 = +m11 * fInvDet; outInvMatrix.m01 = -m01 * fInvDet;
    outInvMatrix.m10 = -m10 * fInvDet; outInvMatrix.m11 = +m00 * fInvDet;

    return true;
}

template<typename Real>
Void TMatrix2<Real>::OrthoNormalize()
{
    Real fInvLen = MathRealFn->InvSqrt( (m00 * m00) + (m10 * m10) );
    m00 *= fInvLen;
    m10 *= fInvLen;

    Real fDot0 = ( (m00 * m01) + (m10 * m11) );
    m01 -= ( fDot0 * m00 );
    m11 -= ( fDot0 * m10 );

    fInvLen = MathRealFn->InvSqrt( (m01 * m01) + (m11 * m11) );
    m01 *= fInvLen;
    m11 *= fInvLen;
}

template<typename Real>
inline Real TMatrix2<Real>::QuadraticForm( const TVector2<Real> & v0, const TVector2<Real> & v1 ) const {
    return ( v0 * ( (*this) * v1 ) );
}



