/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Vertex/Vertex3.h
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
// Header prelude
#ifndef SCARAB_LIB_MATH_TYPES_VERTEX_VERTEX3_H
#define SCARAB_LIB_MATH_TYPES_VERTEX_VERTEX3_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../Vector/Vector3.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
    // prototypes
template<typename Real> class TVertex2;
template<typename Real> class TVertex4;

/////////////////////////////////////////////////////////////////////////////////
// The TVertex3 class
template<typename Real>
class alignas(32) TVertex3
{
public:
    // Constant values
    static const TVertex3<Real> Null; // Null vertex

	// Constructors
	TVertex3(); // uninitialized
	TVertex3( const Real & x, const Real & y, const Real & z );
	TVertex3( const Real vArr[3] );
	TVertex3( const TVertex2<Real> & rhs );
	TVertex3( const TVertex3<Real> & rhs );
	TVertex3( const TVertex4<Real> & rhs );
	~TVertex3();

	// Assignment operator
	inline TVertex3<Real> & operator=( const TVertex3<Real> & rhs );

	// Casting operations
	inline operator Real*() const;
    inline operator const Real*() const;

    inline TVector3<Real> ToVector() const;

	// Index operations
	inline Real & operator[]( Int i );
    inline const Real & operator[]( Int i ) const;
	inline Real & operator[]( UInt i );
	inline const Real & operator[]( UInt i ) const;

	// Unary operations
	inline TVertex3<Real> operator+() const;
	inline TVertex3<Real> operator-() const;

	// Boolean operations
	inline Bool operator==( const TVertex3<Real> & rhs ) const;
	inline Bool operator!=( const TVertex3<Real> & rhs ) const;

	// Real operations
	inline TVertex3<Real> operator+( const Real & rhs ) const;
	inline TVertex3<Real> operator-( const Real & rhs ) const;
	inline TVertex3<Real> operator*( const Real & rhs ) const;
	inline TVertex3<Real> operator/( const Real & rhs ) const;

	inline TVertex3<Real> & operator+=( const Real & rhs );
	inline TVertex3<Real> & operator-=( const Real & rhs );
	inline TVertex3<Real> & operator*=( const Real & rhs );
	inline TVertex3<Real> & operator/=( const Real & rhs );

    // Vertex operations
	inline TVector3<Real> operator-( const TVertex3<Real> & rhs ) const;

	// Vector operations
	inline TVertex3<Real> operator+( const TVector3<Real> & rhs ) const;
	inline TVertex3<Real> operator-( const TVector3<Real> & rhs ) const;

	inline TVertex3<Real> & operator+=( const TVector3<Real> & rhs );
	inline TVertex3<Real> & operator-=( const TVector3<Real> & rhs );

	// Methods
    inline Real DistSqr() const;
	inline Real Dist() const;
	inline Real InvDistSqr() const;
    inline Real InvDist() const;

    inline Void FromCylindric( const Real & fRadius, const Real & fTheta, const Real & fZ );
    inline Void FromCylindric( const TVertex3<Real> & vCylindric );
    inline Void ToCylindric( Real & outRadius, Real & outTheta, Real & outZ ) const;
    inline Void ToCylindric( TVertex3<Real> & outCylindric ) const;

    inline Void FromSpherical( const Real & fRadius, const Real & fTheta, const Real & fPhi );
    inline Void FromSpherical( const TVertex3<Real> & vSpherical );
    inline Void ToSpherical( Real & outRadius, Real & outTheta, Real & outPhi ) const;
    inline Void ToSpherical( TVertex3<Real> & outSpherical ) const;

	// Data
	Real X, Y, Z;
};

// Explicit instanciation
typedef TVertex3<Float> Vertex3f;
typedef TVertex3<Double> Vertex3d;
typedef TVertex3<Scalar> Vertex3;

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "Vertex3.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MATH_TYPES_VERTEX_VERTEX3_H
