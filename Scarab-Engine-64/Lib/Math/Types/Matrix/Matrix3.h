/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Matrix/Matrix3.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : 3D matrix
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
#ifndef SCARAB_LIB_MATH_TYPES_MATRIX_MATRIX3_H
#define SCARAB_LIB_MATH_TYPES_MATRIX_MATRIX3_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../../../../ThirdParty/System/Hardware/SIMD.h"

#include "../../../Error/ErrorManager.h"

#include "../Vector/Vector2.h"
#include "../Vector/Vector3.h"
#include "../Vertex/Vertex2.h"
#include "../Vertex/Vertex3.h"

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
template<typename Real> class TMatrix2;
template<typename Real> class TMatrix4;

/////////////////////////////////////////////////////////////////////////////////
// The TMatrix3 class
template<typename Real>
class alignas(32) TMatrix3
{
public:
    // Constant values
    static const TMatrix3<Real> Null;	  // Null matrix
    static const TMatrix3<Real> Identity; // Identity matrix

	// Constructors
	TMatrix3(); // uninitialized
	TMatrix3( const Real & a00, const Real & a01, const Real & a02,
			  const Real & a10, const Real & a11, const Real & a12,
			  const Real & a20, const Real & a21, const Real & a22 );
	TMatrix3( const Real v0[3], const Real v1[3], const Real v2[3], Bool bRows = true );
	TMatrix3( const Real arrMat[9], Bool bRows = true );
	TMatrix3( const TVector3<Real> & v0, const TVector3<Real> & v1, const TVector3<Real> & v2, Bool bRows = true );
	TMatrix3( const TVector3<Real> vMat[3], Bool bRows = true );
	TMatrix3( const TMatrix2<Real> & rhs );
	TMatrix3( const TMatrix3<Real> & rhs );
	TMatrix3( const TMatrix4<Real> & rhs );
	~TMatrix3();

	// Assignment operator
	inline TMatrix3<Real> & operator=( const TMatrix3<Real> & rhs );

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
	inline TMatrix3<Real> operator+() const;
	inline TMatrix3<Real> operator-() const;

	// Boolean operations
	Bool operator==( const TMatrix3<Real> & rhs ) const;
	Bool operator!=( const TMatrix3<Real> & rhs ) const;

    // Real operations
	TMatrix3<Real> operator*( const Real & rhs ) const;
	TMatrix3<Real> operator/( const Real & rhs ) const;

    TMatrix3<Real> & operator*=( const Real & rhs );
	TMatrix3<Real> & operator/=( const Real & rhs );

	// Vertex operations
	TVertex3<Real> operator*( const TVertex3<Real> & rhs ) const;

	// Vector operations
	TVector3<Real> operator*( const TVector3<Real> & rhs ) const;

	// Matrix operations
	TMatrix3<Real> operator+( const TMatrix3<Real> & rhs ) const;
	TMatrix3<Real> operator-( const TMatrix3<Real> & rhs ) const;
	TMatrix3<Real> operator*( const TMatrix3<Real> & rhs ) const;

	TMatrix3<Real> & operator+=( const TMatrix3<Real> & rhs );
	TMatrix3<Real> & operator-=( const TMatrix3<Real> & rhs );
	TMatrix3<Real> & operator*=( const TMatrix3<Real> & rhs );

	// Methods : Row & Column access
	inline Void GetRow( TVector3<Real> & outRow, UInt iRow ) const;
	inline Void SetRow( UInt iRow, const Real & fRow0, const Real & fRow1, const Real & fRow2 );
	inline Void SetRow( UInt iRow, const Real vRow[3] );
	inline Void SetRow( UInt iRow, const TVector3<Real> & vRow );

    inline Void GetColumn( TVector3<Real> & outColumn, UInt iColumn ) const;
	inline Void SetColumn( UInt iColumn, const Real & fCol0, const Real & fCol1, const Real & fCol2 );
	inline Void SetColumn( UInt iColumn, const Real vCol[3] );
	inline Void SetColumn( UInt iColumn, const TVector3<Real> & vCol );

	inline Void GetDiagonal( TVector3<Real> & outDiag ) const;
	inline Void SetDiagonal( const Real & fDiag0, const Real & fDiag1, const Real & fDiag2 );
	inline Void SetDiagonal( const Real vDiag[3] );
	inline Void SetDiagonal( const TVector3<Real> & vDiag );

    // Methods : Makers
	inline Void MakeNull();
	inline Void MakeIdentity();

	inline Void MakeDiagonal( const Real & fDiag0, const Real & fDiag1, const Real & fDiag2 );
	inline Void MakeDiagonal( const Real vDiag[3] );
	inline Void MakeDiagonal( const TVector3<Real> & vDiag );

	inline Void MakeTranslate( const TVector2<Real> & vTranslate );
	inline Void SetTranslate( const TVector2<Real> & vTranslate );

	inline Void MakeScale( const TVector2<Real> & vScale );
	inline Void MakeScale( const TVector3<Real> & vScale );

    inline Void MakeBasis( const TVertex2<Real> & vOrigin, const TVector2<Real> & vI, const TVector2<Real> & vJ );
    inline Void MakeBasis( const TVector3<Real> & vI, const TVector3<Real> & vJ, const TVector3<Real> & vK );

    inline Void MakeSkewSymmetric( const TVector3<Real> & vSkew );

	Void MakeRotate( Axis iAxis, const Real & fAngle );
	Void MakeRotate( const TVector3<Real> & vAxis, const Real & fAngle );
    Void MakeRotate( const Real & fYaw, const Real & fPitch, const Real & fRoll, EulerAngles eulerAnglesOrder );

    Void GetAxisAngle( TVector3<Real> & outAxis, Real & outAngle ) const;

    Void MakeTensorProduct( const TVector3<Real> & vU, const TVector3<Real> & vV );

    // Methods : Linear algebra stuff
    inline Void Transpose( TMatrix3<Real> & outTransposedMatrix ) const;
    //inline TMatrix3<Real> TransposeMul(const TMatrix3<Real> & rhs) const; // Transpose(M) * rhs
    //inline TMatrix3<Real> MulTranspose(const TMatrix3<Real> & rhs) const; // M * Transpose(rhs)
    //inline TMatrix3<Real> TransposeMulTranspose(const TMatrix3<Real> & rhs) const; // Transpose(M) * Transpose(rhs)
    //inline TMatrix3<Real> DiagonalMul(const TVector3<Real> & rhs) const; // Diag(rhs) * M
    //inline TMatrix3<Real> MulDiagonal(const TVector3<Real> & rhs) const; // M * Diag(rhs)

	inline Real Trace() const;
	Real Determinant() const;

    inline Void Minor( TMatrix2<Real> & outMinor, UInt iRow, UInt iColumn ) const;

    Void Adjoint( TMatrix3<Real> & outAdjointMatrix ) const; // Transposed Co-Matrix

	inline Bool IsInvertible( Real fZeroTolerance = MathFunction<Real>::Epsilon ) const;
	Bool Invert( TMatrix3<Real> & outInvMatrix, Real fZeroTolerance = MathFunction<Real>::Epsilon ) const;

    Void OrthoNormalize(); // Gram-Schmidt

    Void PolarDecomposition( TMatrix3<Real> & outRotate, TMatrix3<Real> & outScale ) const;

    inline Real QuadraticForm( const TVector3<Real> & v0, const TVector3<Real> & v1 ) const; // Transpose(v0) * M * v1

    // Spherical interpolation
    Void SLerp( const TMatrix3<Real> & matSource, const TMatrix3<Real> & matTarget, const Real & fT );

	// Data, stored column-wise for optimisation
	Real m00, m10, m20; // Column 0
	Real m01, m11, m21; // Column 1
	Real m02, m12, m22; // Column 2
};

// Explicit instanciation
typedef TMatrix3<Float> Matrix3f;
typedef TMatrix3<Double> Matrix3d;
typedef TMatrix3<Scalar> Matrix3;

// Specializations
#ifdef SIMD_ENABLE

//template<>
//TMatrix3<Float> TMatrix3<Float>::operator*( const Float & rhs ) const;
//template<>
//TMatrix3<Double> TMatrix3<Double>::operator*( const Double & rhs ) const;

//template<>
//TMatrix3<Float> TMatrix3<Float>::operator/( const Float & rhs ) const;
//template<>
//TMatrix3<Double> TMatrix3<Double>::operator/( const Double & rhs ) const;
//
//template<>
//TMatrix3<Float> & TMatrix3<Float>::operator*=( const Float & rhs );
//template<>
//TMatrix3<Double> & TMatrix3<Double>::operator*=( const Double & rhs );

//template<>
//TMatrix3<Float> & TMatrix3<Float>::operator/=( const Float & rhs );
//template<>
//TMatrix3<Double> & TMatrix3<Double>::operator/=( const Double & rhs );

template<>
TVertex3<Float> TMatrix3<Float>::operator*( const TVertex3<Float> & rhs ) const;
template<>
TVertex3<Double> TMatrix3<Double>::operator*( const TVertex3<Double> & rhs ) const;

template<>
TVector3<Float> TMatrix3<Float>::operator*( const TVector3<Float> & rhs ) const;
template<>
TVector3<Double> TMatrix3<Double>::operator*( const TVector3<Double> & rhs ) const;

//template<>
//TMatrix3<Float> TMatrix3<Float>::operator+( const TMatrix3<Float> & rhs ) const;
//template<>
//TMatrix3<Double> TMatrix3<Double>::operator+( const TMatrix3<Double> & rhs ) const;
//template<>
//TMatrix3<Float> TMatrix3<Float>::operator-( const TMatrix3<Float> & rhs ) const;
//template<>
//TMatrix3<Double> TMatrix3<Double>::operator-( const TMatrix3<Double> & rhs ) const;
template<>
TMatrix3<Float> TMatrix3<Float>::operator*( const TMatrix3<Float> & rhs ) const;
template<>
TMatrix3<Double> TMatrix3<Double>::operator*( const TMatrix3<Double> & rhs ) const;

//template<>
//TMatrix3<Float> & TMatrix3<Float>::operator+=( const TMatrix3<Float> & rhs );
//template<>
//TMatrix3<Double> & TMatrix3<Double>::operator+=( const TMatrix3<Double> & rhs );
//template<>
//TMatrix3<Float> & TMatrix3<Float>::operator-=( const TMatrix3<Float> & rhs );
//template<>
//TMatrix3<Double> & TMatrix3<Double>::operator-=( const TMatrix3<Double> & rhs );
template<>
TMatrix3<Float> & TMatrix3<Float>::operator*=( const TMatrix3<Float> & rhs );
template<>
TMatrix3<Double> & TMatrix3<Double>::operator*=( const TMatrix3<Double> & rhs );

//template<>
//Void TMatrix3<Float>::MakeRotate( const TVector3<Float> & vAxis, const Float & fAngle );
//template<>
//Void TMatrix3<Double>::MakeRotate( const TVector3<Double> & vAxis, const Double & fAngle );

template<>
Float TMatrix3<Float>::Determinant() const;
template<>
Double TMatrix3<Double>::Determinant() const;

//template<>
//Void TMatrix3<Float>::Adjoint( TMatrix3<Float> & outAdjointMatrix ) const;
//template<>
//Void TMatrix3<Double>::Adjoint( TMatrix3<Double> & outAdjointMatrix ) const;

template<>
Bool TMatrix3<Float>::Invert( TMatrix3<Float> & outInvertMatrix, Float fZeroTolerance ) const;
template<>
Bool TMatrix3<Double>::Invert( TMatrix3<Double> & outInvertMatrix, Double fZeroTolerance ) const;

#endif // SIMD_ENABLE

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "Matrix3.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MATH_TYPES_MATRIX_MATRIX3_H

