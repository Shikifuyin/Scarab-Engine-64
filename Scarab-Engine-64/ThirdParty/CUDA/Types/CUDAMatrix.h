/////////////////////////////////////////////////////////////////////////////////
// File : CUDAMatrix.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA-Optimized Large Matrices
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef MATRIXSIMULATION_CUDAMATRIX_H
#define MATRIXSIMULATION_CUDAMATRIX_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUDAVector.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The TCUDAMatrix class
template<typename Number>
class TCUDAMatrix
{
public:
	// Constants
	static const TCUDAMatrix<Number> Null;
	static const TCUDAMatrix<Number> Identity;

	// Constructors
	TCUDAMatrix( UInt iRowCount, UInt iColumnCount, const Number * arrData = NULL, Bool bColumnWise = false );
	TCUDAMatrix( const TCUDAMatrix<Number> & rhs );
	~TCUDAMatrix();

	// Assignment operator
	TCUDAMatrix<Number> & operator=( const TCUDAMatrix<Number> & rhs );
	
	// Casting operators
	operator Number*() const;
	operator const Number*() const;
	
	// Index operators
		// Row-Wise flat index
	Number & operator[]( Int i );
	const Number & operator[]( Int i ) const;
	Number & operator[]( UInt i );
	const Number & operator[]( UInt i ) const;
	
		// (Row,Col) index
	Number & operator()( Int iRow, Int iColumn );
	const Number & operator()( Int iRow, Int iColumn ) const;
	Number & operator()( UInt iRow, UInt iColumn );
	const Number & operator()( UInt iRow, UInt iColumn ) const;
	
	// Unary operators
	TCUDAMatrix<Number> operator+() const;
	TCUDAMatrix<Number> operator-() const;
	
	// Boolean operators
	Bool operator==( const TCUDAMatrix<Number> & rhs ) const;
	Bool operator!=( const TCUDAMatrix<Number> & rhs ) const;
	
	// Number operations
	TCUDAMatrix<Number> operator*( const Number & rhs ) const;
	TCUDAMatrix<Number> operator/( const Number & rhs ) const; // Already works by inverse, safe to call
	
	TCUDAMatrix<Number> & operator*=( const Number & rhs );
	TCUDAMatrix<Number> & operator/=( const Number & rhs ); // Already works by inverse, safe to call
	
	// Vector operations
	TCUDAVector<Number> operator*( const TCUDAVector<Number> & rhs ) const;
	
	// Matrix operations
	TCUDAMatrix<Number> operator+( const TCUDAMatrix<Number> & rhs ) const;
	TCUDAMatrix<Number> operator-( const TCUDAMatrix<Number> & rhs ) const;
	TCUDAMatrix<Number> operator*( const TCUDAMatrix<Number> & rhs ) const;

	TCUDAMatrix<Number> & operator+=( const TCUDAMatrix<Number> & rhs );
	TCUDAMatrix<Number> & operator-=( const TCUDAMatrix<Number> & rhs );
	TCUDAMatrix<Number> & operator*=( const TCUDAMatrix<Number> & rhs );
	
	// Row & Column access
	Void GetRow( TCUDAVector<Number> & outRow, UInt iRow ) const;
	Void SetRow( UInt iRow, const TCUDAVector<Number> & vRow );
	
	Void GetColumn( TCUDAVector<Number> & outColumn, UInt iColumn ) const;
	Void SetColumn( UInt iColumn, const TCUDAVector<Number> & vColumn );
	
	Void GetDiagonal( TCUDAVector<Number> & outDiag ) const;
	Void SetDiagonal( const TCUDAVector<Number> & vDiag );
	
	// Make methods
	Void MakeNull();
	Void MakeIdentity();
	Void MakeDiagonal( const TCUDAVector<Number> & vDiag );
	
	Void MakeScale( const TCUDAVector<Number> & vScale );
	
	Void MakeBasis( const TCUDAVector<Number> * arrBasisVectors );
	
	// Simple methods
	Number Trace() const;
	
	Void Transpose( TCUDAMatrix<Number> & outTransposedMatrix ) const;
	
	// Advanced methods
	Number Determinant() const;
	
	Bool IsInvertible( const Number & fZeroTolerance ) const;
	Bool Invert( TCUDAMatrix<Number> & outInvertedMatrix, const Number & fZeroTolerance ) const;
	
private:
	UInt m_iRowCount;
	UInt m_iColumnCount;
	Number * m_arrData;
};

// Template Specializations
typedef TCUDAMatrix<Float> CUDAMatrixF;
typedef TCUDAMatrix<Double> CUDAMatrixD;
typedef TCUDAMatrix<ComplexFloat> CUDAMatrixCF;
typedef TCUDAMatrix<ComplexDouble> CUDAMatrixCD;

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUDAMatrix.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // MATRIXSIMULATION_CUDAMATRIX_H

