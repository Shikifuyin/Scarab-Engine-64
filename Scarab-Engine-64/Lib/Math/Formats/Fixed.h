/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Formats/Fixed.h
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
// Header prelude
#ifndef SCARAB_LIB_MATH_FORMATS_FIXED_H
#define SCARAB_LIB_MATH_FORMATS_FIXED_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "Scalar.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The Fixed class
template<Int FIXED_BITS>
class Fixed
{
public:
	static const Int IntBits;
	static const Int FractBits;

    static const Fixed<FIXED_BITS> Zero;
    static const Fixed<FIXED_BITS> One;
    static const Fixed<FIXED_BITS> Min;
    static const Fixed<FIXED_BITS> Max;
	static const Fixed<FIXED_BITS> Epsilon;
    static const Fixed<FIXED_BITS> Infinity;

	// constructors & destructor
	inline Fixed(); // uninitialized
	inline Fixed( const Fixed<FIXED_BITS> & rhs );
    inline Fixed( UInt8 iValue );
	inline Fixed( Int8 iValue );
	inline Fixed( UInt16 iValue );
	inline Fixed( Int16 iValue );
	inline Fixed( UInt32 iValue );
	inline Fixed( Int32 iValue );
	inline Fixed( Float fValue );
	inline Fixed( Double fValue );
    inline ~Fixed() {}

	// operators : casts
	inline operator UInt8() const;
	inline operator Int8() const;
    inline operator UInt16() const;
	inline operator Int16() const;
	inline operator UInt32() const;
	inline operator Int32() const;
	inline operator Float() const;
	inline operator Double() const;

	// operators : affectations
	inline Fixed<FIXED_BITS> & operator=( const Fixed<FIXED_BITS> & rhs );

	// operators : shifts
	inline Fixed<FIXED_BITS> operator>>( const Int rhs ) const;
	inline Fixed<FIXED_BITS> operator<<( const Int rhs ) const;
	inline Fixed<FIXED_BITS> & operator>>=( const Int rhs );
	inline Fixed<FIXED_BITS> & operator<<=( const Int rhs );

	// operators : arithmetic, unary
	inline Fixed<FIXED_BITS> operator+() const;
	inline Fixed<FIXED_BITS> operator-() const;

	// operators : arithmetic, binary
	inline Fixed<FIXED_BITS> operator+( const Fixed<FIXED_BITS> & rhs ) const;
	inline Fixed<FIXED_BITS> operator-( const Fixed<FIXED_BITS> & rhs ) const;
	inline Fixed<FIXED_BITS> operator*( const Fixed<FIXED_BITS> & rhs ) const;
	inline Fixed<FIXED_BITS> operator/( const Fixed<FIXED_BITS> & rhs ) const;

	// operators : arithmetic & affect, binary
	inline Fixed<FIXED_BITS> & operator+=( const Fixed<FIXED_BITS> & rhs );
	inline Fixed<FIXED_BITS> & operator-=( const Fixed<FIXED_BITS> & rhs );
	inline Fixed<FIXED_BITS> & operator*=( const Fixed<FIXED_BITS> & rhs );
	inline Fixed<FIXED_BITS> & operator/=( const Fixed<FIXED_BITS> & rhs );

	// operators : booleans
	inline Bool operator==( const Fixed<FIXED_BITS> & rhs ) const;
	inline Bool operator!=( const Fixed<FIXED_BITS> & rhs ) const;
	inline Bool operator<=( const Fixed<FIXED_BITS> & rhs ) const;
	inline Bool operator>=( const Fixed<FIXED_BITS> & rhs ) const;
	inline Bool operator<( const Fixed<FIXED_BITS> & rhs ) const;
	inline Bool operator>( const Fixed<FIXED_BITS> & rhs ) const;

    // Functions
    inline static Fixed<FIXED_BITS> Abs( const Fixed<FIXED_BITS> & fpValue );
    static Fixed<FIXED_BITS> Sqrt( const Fixed<FIXED_BITS> & fpValue );
    static Fixed<FIXED_BITS> ArcTan2( const Fixed<FIXED_BITS> & fpNum, const Fixed<FIXED_BITS> & fpDenom );

private:
	// dummyarg needed to avoid conflict with public Fixed(Int)
	inline Fixed( Int dummyarg, Int fpValue );

	static const Int FP;			// fixed point bit position
	static const Int FP_POW;		// (1 << FP) = 2^FP
	static const Double FP_INVPOW;	// 1 / FP_POW
	static const Int FP_DBL;		// (FP << 1) = FP * 2
	static const Int FP_HALF;		// (FP >> 1) = FP / 2

	Int m_fpValue;
};

// Explicit instanciation
typedef Fixed<8>	Fixed8;
typedef Fixed<16>	Fixed16;
typedef Fixed<24>	Fixed24;

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "Fixed.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MATH_FORMATS_FIXED_H
