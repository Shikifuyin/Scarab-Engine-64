/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Matrix/Matrix2.h
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
// Known Bugs : We use row-wise matrix representation, vertices & vectors are
//              considered as column tuples and transformations use right-hand
//              products such that u' = M*u.
//              Let M be a n*n matrix (n = 2,3,4), and Mij be the value at
//              row i, column j.
//              Representation as individual values sets mIJ = Mij.
//              As an example, m12 is value at row 1, column 2, in a 4*4 matrix.
//              In 4D, translations are stored in last column (m03,m13,m23,m33).
//
// STORED COLUMN-WISE IN MEMORY FOR PERFORMANCE !
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_LIB_MATH_TYPES_MATRIX_MATRIX2_H
#define SCARAB_LIB_MATH_TYPES_MATRIX_MATRIX2_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../../../../ThirdParty/System/Hardware/SIMD.h"

#include "../../../Error/ErrorManager.h"

#include "../Vector/Vector2.h"
#include "../Vertex/Vertex2.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#ifndef _MATRIX_AXIS_EULER_ANGLES_DECLARED
#define _MATRIX_AXIS_EULER_ANGLES_DECLARED
    enum Axis
    {
	    AXIS_X = 0,
	    AXIS_Y = 1,
	    AXIS_Z = 2
    };
    enum EulerAngles
    {
        // udv
	    EULER_ANGLES_XYZ = ( (AXIS_X << 4) | (AXIS_Y << 2) | AXIS_Z ),
	    EULER_ANGLES_XZY = ( (AXIS_X << 4) | (AXIS_Z << 2) | AXIS_Y ),
	    EULER_ANGLES_YXZ = ( (AXIS_Y << 4) | (AXIS_X << 2) | AXIS_Z ),
	    EULER_ANGLES_YZX = ( (AXIS_Y << 4) | (AXIS_Z << 2) | AXIS_X ),
	    EULER_ANGLES_ZXY = ( (AXIS_Z << 4) | (AXIS_X << 2) | AXIS_Y ),
	    EULER_ANGLES_ZYX = ( (AXIS_Z << 4) | (AXIS_Y << 2) | AXIS_X ),
        // udu
	    EULER_ANGLES_XYX = ( (AXIS_X << 4) | (AXIS_Y << 2) | AXIS_X ),
	    EULER_ANGLES_XZX = ( (AXIS_X << 4) | (AXIS_Z << 2) | AXIS_X ),
	    EULER_ANGLES_YXY = ( (AXIS_Y << 4) | (AXIS_X << 2) | AXIS_Y ),
	    EULER_ANGLES_YZY = ( (AXIS_Y << 4) | (AXIS_Z << 2) | AXIS_Y ),
	    EULER_ANGLES_ZXZ = ( (AXIS_Z << 4) | (AXIS_X << 2) | AXIS_Z ),
	    EULER_ANGLES_ZYZ = ( (AXIS_Z << 4) | (AXIS_Y << 2) | AXIS_Z )
    };
#endif

    // prototypes
template<typename Real> class TMatrix3;
template<typename Real> class TMatrix4;

/////////////////////////////////////////////////////////////////////////////////
// The TMatrix2 class
template<typename Real>
class alignas(32) TMatrix2
{
public:
    // Constant values
    static const TMatrix2<Real> Null;	  // Null matrix
    static const TMatrix2<Real> Identity; // Identity matrix

	// Constructors
	TMatrix2(); // uninitialized
	TMatrix2( const Real & a00, const Real & a01,
			  const Real & a10, const Real & a11 );
	TMatrix2( const Real v0[2], const Real v1[2], Bool bRows = true );
	TMatrix2( const Real arrMat[4], Bool bRows = true );
	TMatrix2( const TVector2<Real> & v0, const TVector2<Real> & v1, Bool bRows = true );
	TMatrix2( const TVector2<Real> vMat[2], Bool bRows = true );
	TMatrix2( const TMatrix2<Real> & rhs );
	TMatrix2( const TMatrix3<Real> & rhs );
	TMatrix2( const TMatrix4<Real> & rhs );
	~TMatrix2();

	// Assignment operator
	inline TMatrix2<Real> & operator=( const TMatrix2<Real> & rhs );

	// Casting operations
	inline operator Real*() const;
    inline operator const Real*() const;

	// Index operations
		// flat index 0-15 (Row-wise !)
	inline Real & operator[]( Int i );
    inline const Real & operator[]( Int i ) const;
	inline Real & operator[]( UInt i );
	inline const Real & operator[]( UInt i ) const;
		// (row,col) index
	inline Real & operator()( Int iRow, Int iColumn );
    inline const Real & operator()( Int iRow, Int iColumn ) const;
	inline Real & operator()( UInt iRow, UInt iColumn );
	inline const Real & operator()( UInt iRow, UInt iColumn ) const;

	// Unary operations
	inline TMatrix2<Real> operator+() const;
	inline TMatrix2<Real> operator-() const;

	// Boolean operations
	Bool operator==( const TMatrix2<Real> & rhs ) const;
	Bool operator!=( const TMatrix2<Real> & rhs ) const;

    // Real operations
	TMatrix2<Real> operator*( const Real & rhs ) const;
	TMatrix2<Real> operator/( const Real & rhs ) const;

    TMatrix2<Real> & operator*=( const Real & rhs );
	TMatrix2<Real> & operator/=( const Real & rhs );

	// Vertex operations
	TVertex2<Real> operator*( const TVertex2<Real> & rhs ) const;

	// Vector operations
	TVector2<Real> operator*( const TVector2<Real> & rhs ) const;

	// Matrix operations
	TMatrix2<Real> operator+( const TMatrix2<Real> & rhs ) const;
	TMatrix2<Real> operator-( const TMatrix2<Real> & rhs ) const;
	TMatrix2<Real> operator*( const TMatrix2<Real> & rhs ) const;

	TMatrix2<Real> & operator+=( const TMatrix2<Real> & rhs );
	TMatrix2<Real> & operator-=( const TMatrix2<Real> & rhs );
	TMatrix2<Real> & operator*=( const TMatrix2<Real> & rhs );

	// Methods : Row & Column access
	inline Void GetRow( TVector2<Real> & outRow, UInt iRow ) const;
	inline Void SetRow( UInt iRow, const Real & fRow0, const Real & fRow1 );
	inline Void SetRow( UInt iRow, const Real vRow[2] );
	inline Void SetRow( UInt iRow, const TVector2<Real> & vRow );

    inline Void GetColumn( TVector2<Real> & outColumn, UInt iColumn ) const;
	inline Void SetColumn( UInt iColumn, const Real & fCol0, const Real & fCol1 );
	inline Void SetColumn( UInt iColumn, const Real vCol[2] );
	inline Void SetColumn( UInt iColumn, const TVector2<Real> & vCol );

	inline Void GetDiagonal( TVector2<Real> & outDiag ) const;
	inline Void SetDiagonal( const Real & fDiag0, const Real & fDiag1 );
	inline Void SetDiagonal( const Real vDiag[2] );
	inline Void SetDiagonal( const TVector2<Real> & vDiag );

    // Methods : Makers
	inline Void MakeNull();
	inline Void MakeIdentity();

	inline Void MakeDiagonal( const Real & fDiag0, const Real & fDiag1 );
	inline Void MakeDiagonal( const Real vDiag[2] );
	inline Void MakeDiagonal( const TVector2<Real> & vDiag );

	inline Void MakeScale( const TVector2<Real> & vScale );

    inline Void MakeBasis( const TVector2<Real> & vI, const TVector2<Real> & vJ );

	inline Void MakeRotate( const Real & fAngle );

    inline Void GetAngle( Real & outAngle ) const;

    inline Void MakeTensorProduct( const TVector2<Real> & vU, const TVector2<Real> & vV );

    // Methods : Linear algebra stuff
    inline Void Transpose( TMatrix2<Real> & outTransposedMatrix ) const;
    //inline TMatrix2<Real> TransposeMul(const TMatrix2<Real> & rhs) const; // Transpose(M) * rhs
    //inline TMatrix2<Real> MulTranspose(const TMatrix2<Real> & rhs) const; // M * Transpose(rhs)
    //inline TMatrix2<Real> TransposeMulTranspose(const TMatrix2<Real> & rhs) const; // Transpose(M) * Transpose(rhs)
    //inline TMatrix2<Real> DiagonalMul(const TVector2<Real> & rhs) const; // Diag(rhs) * M
    //inline TMatrix2<Real> MulDiagonal(const TVector2<Real> & rhs) const; // M * Diag(rhs)

	inline Real Trace() const;
	inline Real Determinant() const;

    inline Void Minor( Real & outMinor, UInt iRow, UInt iColumn ) const;

    inline Void Adjoint( TMatrix2<Real> & outAdjointMatrix ) const; // Transposed Co-Matrix

	inline Bool IsInvertible( Real fZeroTolerance = MathFunction<Real>::Epsilon ) const;
	Bool Invert( TMatrix2<Real> & outInvMatrix, Real fZeroTolerance = MathFunction<Real>::Epsilon ) const;

    Void OrthoNormalize(); // Gram-Schmidt

    inline Real QuadraticForm( const TVector2<Real> & v0, const TVector2<Real> & v1 ) const; // Transpose(v0) * M * v1

	// Data, stored column-wise for optimisation
	Real m00, m10; // Column 0
	Real m01, m11; // Column 1
};

// Explicit instanciation
typedef TMatrix2<Float> Matrix2f;
typedef TMatrix2<Double> Matrix2d;
typedef TMatrix2<Scalar> Matrix2;

// Specializations
#ifdef SIMD_ENABLE

template<>
TMatrix2<Float> TMatrix2<Float>::operator*( const Float & rhs ) const;
template<>
TMatrix2<Double> TMatrix2<Double>::operator*( const Double & rhs ) const;
template<>
TMatrix2<Float> TMatrix2<Float>::operator/( const Float & rhs ) const;
template<>
TMatrix2<Double> TMatrix2<Double>::operator/( const Double & rhs ) const;

template<>
TMatrix2<Float> & TMatrix2<Float>::operator*=( const Float & rhs );
template<>
TMatrix2<Double> & TMatrix2<Double>::operator*=( const Double & rhs );
template<>
TMatrix2<Float> & TMatrix2<Float>::operator/=( const Float & rhs );
template<>
TMatrix2<Double> & TMatrix2<Double>::operator/=( const Double & rhs );

template<>
TMatrix2<Float> TMatrix2<Float>::operator+( const TMatrix2<Float> & rhs ) const;
template<>
TMatrix2<Double> TMatrix2<Double>::operator+( const TMatrix2<Double> & rhs ) const;
template<>
TMatrix2<Float> TMatrix2<Float>::operator-( const TMatrix2<Float> & rhs ) const;
template<>
TMatrix2<Double> TMatrix2<Double>::operator-( const TMatrix2<Double> & rhs ) const;
template<>
TMatrix2<Float> TMatrix2<Float>::operator*( const TMatrix2<Float> & rhs ) const;
template<>
TMatrix2<Double> TMatrix2<Double>::operator*( const TMatrix2<Double> & rhs ) const;

template<>
TMatrix2<Float> & TMatrix2<Float>::operator+=( const TMatrix2<Float> & rhs );
template<>
TMatrix2<Double> & TMatrix2<Double>::operator+=( const TMatrix2<Double> & rhs );
template<>
TMatrix2<Float> & TMatrix2<Float>::operator-=( const TMatrix2<Float> & rhs );
template<>
TMatrix2<Double> & TMatrix2<Double>::operator-=( const TMatrix2<Double> & rhs );
template<>
TMatrix2<Float> & TMatrix2<Float>::operator*=( const TMatrix2<Float> & rhs );
template<>
TMatrix2<Double> & TMatrix2<Double>::operator*=( const TMatrix2<Double> & rhs );

#endif // SIMD_ENABLE

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "Matrix2.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MATH_TYPES_MATRIX_MATRIX2_H

