/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Vertex/Vertex4.h
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
// Header prelude
#ifndef SCARAB_LIB_MATH_TYPES_VERTEX_VERTEX4_H
#define SCARAB_LIB_MATH_TYPES_VERTEX_VERTEX4_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../Vector/Vector4.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
    // prototypes
template<typename Real> class TVertex2;
template<typename Real> class TVertex3;

/////////////////////////////////////////////////////////////////////////////////
// The TVertex4 class
template<typename Real>
class alignas(32) TVertex4
{
public:
    // Constant values
    static const TVertex4<Real> Null; // Null vertex

	// Constructors
	TVertex4(); // uninitialized
	TVertex4( const Real & x, const Real & y, const Real & z, const Real & w = (Real)1 );
	TVertex4( const Real vArr[4] );
	TVertex4( const TVertex2<Real> & rhs );
	TVertex4( const TVertex3<Real> & rhs );
	TVertex4( const TVertex4<Real> & rhs );
	~TVertex4();

	// Assignment operator
	inline TVertex4<Real> & operator=( const TVertex4<Real> & rhs );

	// Casting operations
	inline operator Real*() const;
    inline operator const Real*() const;

    inline TVector4<Real> ToVector() const;

	// Index operations
	inline Real & operator[]( Int i );
    inline const Real & operator[]( Int i ) const;
	inline Real & operator[]( UInt i );
	inline const Real & operator[]( UInt i ) const;

	// Unary operations
	inline TVertex4<Real> operator+() const;
	inline TVertex4<Real> operator-() const;

	// Boolean operations
	inline Bool operator==( const TVertex4<Real> & rhs ) const;
	inline Bool operator!=( const TVertex4<Real> & rhs ) const;

	// Real operations
	inline TVertex4<Real> operator+( const Real & rhs ) const;
	inline TVertex4<Real> operator-( const Real & rhs ) const;
	inline TVertex4<Real> operator*( const Real & rhs ) const;
	inline TVertex4<Real> operator/( const Real & rhs ) const;

	inline TVertex4<Real> & operator+=( const Real & rhs );
	inline TVertex4<Real> & operator-=( const Real & rhs );
	inline TVertex4<Real> & operator*=( const Real & rhs );
	inline TVertex4<Real> & operator/=( const Real & rhs );

    // Vertex operations
	inline TVector4<Real> operator-( const TVertex4<Real> & rhs ) const;

	// Vector operations
	inline TVertex4<Real> operator+( const TVector4<Real> & rhs ) const;
	inline TVertex4<Real> operator-( const TVector4<Real> & rhs ) const;

	inline TVertex4<Real> & operator+=( const TVector4<Real> & rhs );
	inline TVertex4<Real> & operator-=( const TVector4<Real> & rhs );

	// Methods
    inline Real DistSqr() const;
	inline Real Dist() const;
	inline Real InvDistSqr() const;
    inline Real InvDist() const;
	inline Void NormalizeW();

	// Data
	Real X, Y, Z, W;
};

// Explicit instanciation
typedef TVertex4<Float> Vertex4f;
typedef TVertex4<Double> Vertex4d;
typedef TVertex4<Scalar> Vertex4;

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "Vertex4.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MATH_TYPES_VERTEX_VERTEX4_H
