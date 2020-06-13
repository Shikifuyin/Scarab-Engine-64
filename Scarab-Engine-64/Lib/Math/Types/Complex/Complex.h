/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Complex/Complex.h
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
// Header prelude
#ifndef SCARAB_LIB_MATH_TYPES_COMPLEX_COMPLEX_H
#define SCARAB_LIB_MATH_TYPES_COMPLEX_COMPLEX_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../../Functions/MathFunction.h"

#include "../Matrix/Matrix2.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The TComplex class
template<typename Real>
class TComplex
{
public:
    // Constant values
    static const TComplex<Real> Null; // Null value

	// Constructors
	TComplex(); // uninitialized
	TComplex( const Real & fR, const Real & fI );
	TComplex( const Real vArr[2] );
	~TComplex();

	// Assignment operator
	inline TComplex<Real> & operator=( const TComplex<Real> & rhs );

	// Casting operations
	inline operator Real*() const;
    inline operator const Real*() const;

	// Unary operations
	inline TComplex<Real> operator+() const;
	inline TComplex<Real> operator-() const;

	// Boolean operations
	inline Bool operator==( const TComplex<Real> & rhs ) const;
	inline Bool operator!=( const TComplex<Real> & rhs ) const;

	// Real operations
	inline TComplex<Real> operator+( const Real & rhs ) const;
	inline TComplex<Real> operator-( const Real & rhs ) const;
	inline TComplex<Real> operator*( const Real & rhs ) const;
	inline TComplex<Real> operator/( const Real & rhs ) const;

	inline TComplex<Real> & operator+=( const Real & rhs );
	inline TComplex<Real> & operator-=( const Real & rhs );
	inline TComplex<Real> & operator*=( const Real & rhs );
	inline TComplex<Real> & operator/=( const Real & rhs );

	// Complex operations
	inline TComplex<Real> operator+( const TComplex<Real> & rhs ) const;
	inline TComplex<Real> operator-( const TComplex<Real> & rhs ) const;
	inline TComplex<Real> operator*( const TComplex<Real> & rhs ) const;
	inline TComplex<Real> operator/( const TComplex<Real> & rhs ) const;

	inline TComplex<Real> & operator+=( const TComplex<Real> & rhs );
	inline TComplex<Real> & operator-=( const TComplex<Real> & rhs );
	inline TComplex<Real> & operator*=( const TComplex<Real> & rhs );
	inline TComplex<Real> & operator/=( const TComplex<Real> & rhs );

	// Methods
	inline Real ModulusSqr() const;
	inline Real Modulus() const;
	inline Real InvModulusSqr() const;
    inline Real InvModulus() const;

	inline Real Argument() const;

    inline Void FromPolar( const Real & fRadius, const Real & fTheta );
    inline Void FromPolar( const TComplex<Real> & vPolar );
    inline Void ToPolar( Real & outRadius, Real & outTheta ) const;
    inline Void ToPolar( TComplex<Real> & outPolar ) const;

    inline TMatrix2<Real> GetMatrix() const;

    inline TComplex<Real> Conjugate() const;
    inline TComplex<Real> Sqrt() const;
    inline TComplex<Real> Ln() const;  // cartesian => polar
    inline TComplex<Real> Exp() const; // polar => cartesians

	// Data
	Real R, I;
};

// Explicit instanciation
typedef TComplex<Float> Complexf;
typedef TComplex<Double> Complexd;
typedef TComplex<Scalar> Complex;

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "Complex.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MATH_TYPES_COMPLEX_COMPLEX_H
