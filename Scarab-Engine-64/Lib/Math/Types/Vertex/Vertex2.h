/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Vertex/Vertex2.h
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
// Header prelude
#ifndef SCARAB_LIB_MATH_TYPES_VERTEX_VERTEX2_H
#define SCARAB_LIB_MATH_TYPES_VERTEX_VERTEX2_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../Vector/Vector2.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
    // prototypes
template<typename Real> class TVertex3;
template<typename Real> class TVertex4;

/////////////////////////////////////////////////////////////////////////////////
// The TVertex2 class
template<typename Real>
class TVertex2
{
public:
    // Constant values
    static const TVertex2<Real> Null; // Null vertex

	// Constructors
	TVertex2(); // uninitialized
	TVertex2( const Real & x, const Real & y );
	TVertex2( const Real vArr[2] );
	TVertex2( const TVertex2<Real> & rhs );
	TVertex2( const TVertex3<Real> & rhs );
	TVertex2( const TVertex4<Real> & rhs );
	~TVertex2();

	// Assignment operator
	inline TVertex2<Real> & operator=( const TVertex2<Real> & rhs );

	// Casting operations
	inline operator Real*() const;
    inline operator const Real*() const;

    inline TVector2<Real> ToVector() const;

	// Index operations
	inline Real & operator[]( Int i );
    inline const Real & operator[]( Int i ) const;
	inline Real & operator[]( UInt i );
	inline const Real & operator[]( UInt i ) const;

	// Unary operations
	inline TVertex2<Real> operator+() const;
	inline TVertex2<Real> operator-() const;

	// Boolean operations
	inline Bool operator==( const TVertex2<Real> & rhs ) const;
	inline Bool operator!=( const TVertex2<Real> & rhs ) const;

	// Real operations
	inline TVertex2<Real> operator+( const Real & rhs ) const;
	inline TVertex2<Real> operator-( const Real & rhs ) const;
	inline TVertex2<Real> operator*( const Real & rhs ) const;
	inline TVertex2<Real> operator/( const Real & rhs ) const;

	inline TVertex2<Real> & operator+=( const Real & rhs );
	inline TVertex2<Real> & operator-=( const Real & rhs );
	inline TVertex2<Real> & operator*=( const Real & rhs );
	inline TVertex2<Real> & operator/=( const Real & rhs );

    // Vertex operations
	inline TVector2<Real> operator-( const TVertex2<Real> & rhs ) const;

	// Vector operations
	inline TVertex2<Real> operator+( const TVector2<Real> & rhs ) const;
	inline TVertex2<Real> operator-( const TVector2<Real> & rhs ) const;

	inline TVertex2<Real> & operator+=( const TVector2<Real> & rhs );
	inline TVertex2<Real> & operator-=( const TVector2<Real> & rhs );

	// Methods
    inline Real DistSqr() const;
	inline Real Dist() const;
	inline Real InvDistSqr() const;
    inline Real InvDist() const;

    inline Void FromPolar( const Real & fRadius, const Real & fTheta );
    inline Void FromPolar( const TVertex2<Real> & vPolar );
    inline Void ToPolar( Real & outRadius, Real & outTheta ) const;
    inline Void ToPolar( TVertex2<Real> & outPolar ) const;

	// Data
	Real X, Y;
};

// Explicit instanciation
typedef TVertex2<Float> Vertex2f;
typedef TVertex2<Double> Vertex2d;
typedef TVertex2<Scalar> Vertex2;

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "Vertex2.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MATH_TYPES_VERTEX_VERTEX2_H
