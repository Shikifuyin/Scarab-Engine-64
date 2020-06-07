/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Formats/Fixed.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Fixed provides fixed point arithmetic support
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Fixed implementation
template<Int FIXED_BITS> const Int Fixed<FIXED_BITS>::IntBits	= ( FIXED_BITS );
template<Int FIXED_BITS> const Int Fixed<FIXED_BITS>::FractBits	= ( 32 - FIXED_BITS );

template<Int FIXED_BITS> const Fixed<FIXED_BITS> Fixed<FIXED_BITS>::Zero		= Fixed<FIXED_BITS>( 0, 0 );
template<Int FIXED_BITS> const Fixed<FIXED_BITS> Fixed<FIXED_BITS>::One			= Fixed<FIXED_BITS>( 0, FP_POW );
template<Int FIXED_BITS> const Fixed<FIXED_BITS> Fixed<FIXED_BITS>::Min			= Fixed<FIXED_BITS>( 0, 0xffffffff );
template<Int FIXED_BITS> const Fixed<FIXED_BITS> Fixed<FIXED_BITS>::Max			= Fixed<FIXED_BITS>( 0, 0x7fffffff );
template<Int FIXED_BITS> const Fixed<FIXED_BITS> Fixed<FIXED_BITS>::Epsilon		= Fixed<FIXED_BITS>( 0, 1 );
template<Int FIXED_BITS> const Fixed<FIXED_BITS> Fixed<FIXED_BITS>::Infinity	= Fixed<FIXED_BITS>( 0, 0x7fffffff );

template<Int FIXED_BITS> const Int Fixed<FIXED_BITS>::FP			= ( 32 - FIXED_BITS );
template<Int FIXED_BITS> const Int Fixed<FIXED_BITS>::FP_POW		= ( 1 << FP );
template<Int FIXED_BITS> const Double Fixed<FIXED_BITS>::FP_INVPOW	= ( 1.0 / ((Double)FP_POW) );
template<Int FIXED_BITS> const Int Fixed<FIXED_BITS>::FP_DBL		= ( FP << 1 );
template<Int FIXED_BITS> const Int Fixed<FIXED_BITS>::FP_HALF		= ( FP >> 1 );

template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::Fixed( Int /*dummyarg*/, Int fpValue ) { m_fpValue = fpValue; }

template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::Fixed()									{}
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::Fixed( const Fixed<FIXED_BITS> & rhs )	{ m_fpValue = rhs.m_fpValue; }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::Fixed( UInt8 iValue )					{ m_fpValue = ( ((Int)iValue) << FP ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::Fixed( Int8 iValue )						{ m_fpValue = ( ((Int)iValue) << FP ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::Fixed( UInt16 iValue )					{ m_fpValue = ( ((Int)iValue) << FP ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::Fixed( Int16 iValue )					{ m_fpValue = ( ((Int)iValue) << FP ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::Fixed( UInt32 iValue )					{ m_fpValue = ( ((signed)iValue) << FP ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::Fixed( Int32 iValue )					{ m_fpValue = ( iValue << FP ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::Fixed( Float fValue )					{ m_fpValue = (Int)( fValue * (Float)FP_POW ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::Fixed( Double fValue )					{ m_fpValue = (Int)( fValue * (Double)FP_POW ); }


template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::operator UInt8() const	{ return (UChar)( m_fpValue >> FP ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::operator Int8() const	{ return (Char)( m_fpValue >> FP ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::operator UInt16() const	{ return (UShort)( m_fpValue >> FP ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::operator Int16() const	{ return (Short)( m_fpValue >> FP ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::operator UInt32() const	{ return (unsigned)( m_fpValue >> FP ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::operator Int32() const	{ return ( m_fpValue >> FP ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::operator Float() const	{ return ( m_fpValue * (Float)FP_INVPOW ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS>::operator Double() const	{ return ( m_fpValue * FP_INVPOW ); }

template<Int FIXED_BITS> inline Fixed<FIXED_BITS> & Fixed<FIXED_BITS>::operator=( const Fixed<FIXED_BITS> & rhs ) { m_fpValue = rhs.m_fpValue; return (*this); }

template<Int FIXED_BITS> inline Fixed<FIXED_BITS> Fixed<FIXED_BITS>::operator>>( const Int rhs ) const { return Fixed<FIXED_BITS>( 0, m_fpValue >> rhs ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS> Fixed<FIXED_BITS>::operator<<( const Int rhs ) const { return Fixed<FIXED_BITS>( 0, m_fpValue << rhs ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS> & Fixed<FIXED_BITS>::operator>>=( const Int rhs )    { m_fpValue >>= rhs; return (*this); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS> & Fixed<FIXED_BITS>::operator<<=( const Int rhs )    { m_fpValue <<= rhs; return (*this); }

template<Int FIXED_BITS> inline Fixed<FIXED_BITS> Fixed<FIXED_BITS>::operator+() const { return Fixed<FIXED_BITS>( 0, m_fpValue ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS> Fixed<FIXED_BITS>::operator-() const { return Fixed<FIXED_BITS>( 0, -m_fpValue ); }

template<Int FIXED_BITS> inline Fixed<FIXED_BITS> Fixed<FIXED_BITS>::operator+( const Fixed<FIXED_BITS> & rhs ) const { return Fixed<FIXED_BITS>( 0, m_fpValue + rhs.m_fpValue ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS> Fixed<FIXED_BITS>::operator-( const Fixed<FIXED_BITS> & rhs ) const { return Fixed<FIXED_BITS>( 0, m_fpValue - rhs.m_fpValue ); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS> Fixed<FIXED_BITS>::operator*( const Fixed<FIXED_BITS> & rhs ) const {
	return Fixed( 0, (m_fpValue >> FP_HALF) * (rhs.m_fpValue >> FP_HALF) );
}
template<Int FIXED_BITS> inline Fixed<FIXED_BITS> Fixed<FIXED_BITS>::operator/( const Fixed<FIXED_BITS> & rhs ) const {
	return Fixed( 0, (Int)( (((Int64)m_fpValue) << FP) / (Int64)(rhs.m_fpValue) ) );
}

template<Int FIXED_BITS> inline Fixed<FIXED_BITS> & Fixed<FIXED_BITS>::operator+=( const Fixed<FIXED_BITS> & rhs )	{ m_fpValue += rhs.m_fpValue; return (*this); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS> & Fixed<FIXED_BITS>::operator-=( const Fixed<FIXED_BITS> & rhs )	{ m_fpValue -= rhs.m_fpValue; return (*this); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS> & Fixed<FIXED_BITS>::operator*=( const Fixed<FIXED_BITS> & rhs )	{ *this = (*this) * rhs; return (*this); }
template<Int FIXED_BITS> inline Fixed<FIXED_BITS> & Fixed<FIXED_BITS>::operator/=( const Fixed<FIXED_BITS> & rhs )	{ *this = (*this) / rhs; return (*this); }

template<Int FIXED_BITS> inline Bool Fixed<FIXED_BITS>::operator==( const Fixed<FIXED_BITS> & rhs ) const	{ return ( m_fpValue == rhs.m_fpValue ); }
template<Int FIXED_BITS> inline Bool Fixed<FIXED_BITS>::operator!=( const Fixed<FIXED_BITS> & rhs ) const	{ return ( m_fpValue != rhs.m_fpValue ); }
template<Int FIXED_BITS> inline Bool Fixed<FIXED_BITS>::operator<=( const Fixed<FIXED_BITS> & rhs ) const	{ return ( m_fpValue <= rhs.m_fpValue ); }
template<Int FIXED_BITS> inline Bool Fixed<FIXED_BITS>::operator>=( const Fixed<FIXED_BITS> & rhs ) const	{ return ( m_fpValue >= rhs.m_fpValue ); }
template<Int FIXED_BITS> inline Bool Fixed<FIXED_BITS>::operator<( const Fixed<FIXED_BITS> & rhs ) const	{ return ( m_fpValue < rhs.m_fpValue ); }
template<Int FIXED_BITS> inline Bool Fixed<FIXED_BITS>::operator>( const Fixed<FIXED_BITS> & rhs ) const	{ return ( m_fpValue > rhs.m_fpValue ); }

template<Int FIXED_BITS>
inline Fixed<FIXED_BITS> Fixed<FIXED_BITS>::Abs( const Fixed<FIXED_BITS> & fpValue ) {
    return (fpValue < Zero) ? -fpValue : fpValue;
}
template<Int FIXED_BITS>
Fixed<FIXED_BITS> Fixed<FIXED_BITS>::Sqrt( const Fixed<FIXED_BITS> & fpValue )
{
    Int64 iRoot = 0;
    Int64 iLeft = ( ((Int64)fpValue.m_fpValue) << FP );
    Int64 iMask = ( 1i64 << ((sizeof(Int64) << 3) - 2) );
    while( iMask != 0 ) {
        if ( (iLeft & (-iMask)) > iRoot ) {
            iRoot += iMask;
            iLeft -= iRoot;
            iRoot += iMask;
        }
        iRoot >>= 1;
        iMask >>= 2;
    }
    return Fixed<FIXED_BITS>( 0, (Int)iRoot );
}
template<Int FIXED_BITS>
Fixed<FIXED_BITS> Fixed<FIXED_BITS>::ArcTan2( const Fixed<FIXED_BITS> & fNum, const Fixed<FIXED_BITS> & fDenom )
{
	Fixed fpAbsY = Abs(fNum) + Epsilon;
	Fixed fpR, fpTheta;
	if ( fDenom >= Zero ) {
		fpR = (fDenom - fpAbsY) / (fDenom + fpAbsY);
		fpTheta = Fixed<FIXED_BITS>(SCALAR_PI_4);
	} else {
		fpR = (fpAbsY + fDenom) / (fpAbsY - fDenom);
		fpTheta = Fixed<FIXED_BITS>(SCALAR_3PI_4);
	}
	fpTheta += ( Fixed<FIXED_BITS>(0.1963f) * (fpR*fpR*fpR) - Fixed<FIXED_BITS>(0.9817f) * fpR );
	return (fNum < Zero) ? -fpTheta : fpTheta;
}

