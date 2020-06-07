/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Formats/Integer.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Integer provides exact integer arithmetic support
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : INT_BYTES MUST be at least 8
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_LIB_MATH_FORMATS_INTEGER_H
#define SCARAB_LIB_MATH_FORMATS_INTEGER_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../../../ThirdParty/System/Platform.h"

#include "../../Error/ErrorManager.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
    // prototypes
template<UInt INT_BYTES>
class Rational;

/////////////////////////////////////////////////////////////////////////////////
// The Integer class
template<UInt INT_BYTES>
class Integer
{
public:
    static const Integer<INT_BYTES> Zero;
    static const Integer<INT_BYTES> One;
    static const Integer<INT_BYTES> Infinity;

	// constructors & destructor
	Integer(); // uninitialized
	Integer( const Integer<INT_BYTES> & rhs );
    Integer( UInt8 iValue );
	Integer( Int8 iValue );
	Integer( UInt16 iValue );
	Integer( Int16 iValue );
	Integer( UInt32 iValue );
	Integer( Int32 iValue );
    Integer( UInt64 iValue );
	Integer( Int64 iValue );
	Integer( Float fValue );
	Integer( Double fValue );
	~Integer();

	// operators : casts
    inline operator UInt8() const;
	inline operator Int8() const;
	inline operator UInt16() const;
	inline operator Int16() const;
	inline operator UInt32() const;
	inline operator Int32() const;
    inline operator UInt64() const;
	inline operator Int64() const;
	operator Float() const;
	operator Double() const;

	// operators : affectations
	Integer<INT_BYTES> & operator=( const Integer<INT_BYTES> & rhs );

	// operators : shifts
	Integer<INT_BYTES> operator>>( const Int rhs ) const;
	Integer<INT_BYTES> operator<<( const Int rhs ) const;
	Integer<INT_BYTES> & operator>>=( const Int rhs );
	Integer<INT_BYTES> & operator<<=( const Int rhs );

	// operators : arithmetic, unary
	inline Integer<INT_BYTES> operator+() const;
	inline Integer<INT_BYTES> operator-() const;

	// operators : arithmetic, binary
	Integer<INT_BYTES> operator+( const Integer<INT_BYTES> & rhs ) const;
	Integer<INT_BYTES> operator-( const Integer<INT_BYTES> & rhs ) const;
	Integer<INT_BYTES> operator*( const Integer<INT_BYTES> & rhs ) const;
	Integer<INT_BYTES> operator/( const Integer<INT_BYTES> & rhs ) const;
    Integer<INT_BYTES> operator%( const Integer<INT_BYTES> & rhs ) const;

	// operators : arithmetic & affect, binary
	inline Integer<INT_BYTES> & operator+=( const Integer<INT_BYTES> & rhs );
	inline Integer<INT_BYTES> & operator-=( const Integer<INT_BYTES> & rhs );
	inline Integer<INT_BYTES> & operator*=( const Integer<INT_BYTES> & rhs );
	inline Integer<INT_BYTES> & operator/=( const Integer<INT_BYTES> & rhs );
    inline Integer<INT_BYTES> & operator%=( const Integer<INT_BYTES> & rhs );

	// operators : booleans
	inline Bool operator==( const Integer<INT_BYTES> & rhs ) const;
	inline Bool operator!=( const Integer<INT_BYTES> & rhs ) const;
	inline Bool operator<=( const Integer<INT_BYTES> & rhs ) const;
	inline Bool operator>=( const Integer<INT_BYTES> & rhs ) const;
	inline Bool operator<( const Integer<INT_BYTES> & rhs ) const;
	inline Bool operator>( const Integer<INT_BYTES> & rhs ) const;

private:
    // Give access to Rational
    friend class Rational<INT_BYTES>;

    // Comparison helper
    static Int _Compare( const Integer<INT_BYTES> & lhs, const Integer<INT_BYTES> & rhs );

    // Division helpers
    static Bool _DivMod( const Integer<INT_BYTES> & iNumerator, const Integer<INT_BYTES> & iDenominator,
                         Integer<INT_BYTES> & outQuotient, Integer<INT_BYTES> & outRemainder );
    static Void _DivSingle( const Integer<INT_BYTES> & iNumerator, UInt iDenominator,
                            Integer<INT_BYTES> & outQuotient, Integer<INT_BYTES> & outRemainder );
    static Void _DivMultiple( const Integer<INT_BYTES> & iNumerator, const Integer<INT_BYTES> & iDenominator,
                              Integer<INT_BYTES> & outQuotient, Integer<INT_BYTES> & outRemainder );

    // Digits / Bits access
    inline Int _GetSign() const;
    inline Void _SetSign( Bool bBitValue );
    inline UInt _GetDigit( UInt iDigit ) const;
    inline UInt _GetDigit( UInt iLowDigit, UInt iHighDigit ) const;
    inline Void _SetDigit( UInt iDigit, UInt iValue );
    inline Bool _GetBit( UInt iBit ) const;
    inline Void _SetBit( UInt iBit, Bool bBitValue );

    UInt _LeadingDigit() const;
    UInt _TrailingDigit() const;
    UInt _LeadingBit( UInt iDigit ) const;
    UInt _TrailingBit( UInt iDigit ) const;
    inline UInt _LeadingBit() const;
    inline UInt _TrailingBit() const;

    // Constants
    static const UInt INT_BITS			= ( INT_BYTES << 3 );
    static const UInt INT_DIGIT_COUNT	= ( INT_BYTES / sizeof(UShort) );
    static const UInt INT_DIGIT_LAST	= ( INT_DIGIT_COUNT - 1 );

    // Data
	UShort m_arrDigits[INT_DIGIT_COUNT]; // LSD at 0, MSD at INT_DIGIT_LAST
};

// Explicit instanciation
typedef Integer<16> Integer128;
typedef Integer<32> Integer256;

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "Integer.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MATH_FORMATS_INTEGER_H
