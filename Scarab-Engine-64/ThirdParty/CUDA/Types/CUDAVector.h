/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/Types/CUDAVector.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA-Optimized Large Vectors
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_CUDA_TYPES_CUDAVECTOR_H
#define SCARAB_THIRDPARTY_CUDA_TYPES_CUDAVECTOR_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../CUDAMemory.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The TCUDAVector class
template<typename Number>
class TCUDAVector
{
public:
	// Constructors
	TCUDAVector( UInt iSize, Number * arrData = NULL ); // Not optimal, beware data must be wrapped as non-const !
	TCUDAVector( const CUDAHostMemory * pHostMemory );  // Better, one should always prepare page-locked memory beforehand.
	TCUDAVector( const TCUDAVector<Number> & rhs );
	~TCUDAVector();

	// Assignment operator
	TCUDAVector<Number> & operator=( const TCUDAVector<Number> & rhs );
	
	// Casting operators
	// DISALLOWED, Bad practice !
	
	// Index operators
	Number & operator[]( Int i );
	const Number & operator[]( Int i ) const;
	Number & operator[]( UInt i );
	const Number & operator[]( UInt i ) const;
	
	// Unary operators
	TCUDAVector<Number> operator+() const;
	TCUDAVector<Number> operator-() const;
	
	// Boolean operators
	Bool operator==( const TCUDAVector<Number> & rhs ) const;
	Bool operator!=( const TCUDAVector<Number> & rhs ) const;
	
	// Number operations
	TCUDAVector<Number> operator+( const Number & rhs ) const;
	TCUDAVector<Number> operator-( const Number & rhs ) const;
	TCUDAVector<Number> operator*( const Number & rhs ) const;
	TCUDAVector<Number> operator/( const Number & rhs ) const; // Already works by inverse, safe to call
	
	TCUDAVector<Number> & operator+=( const Number & rhs );
	TCUDAVector<Number> & operator-=( const Number & rhs );
	TCUDAVector<Number> & operator*=( const Number & rhs );
	TCUDAVector<Number> & operator/=( const Number & rhs ); // Already works by inverse, safe to call
	
	// Vector operations
	TCUDAVector<Number> operator+( const TCUDAVector<Number> & rhs ) const;
	TCUDAVector<Number> operator-( const TCUDAVector<Number> & rhs ) const;

	TCUDAVector<Number> & operator+=( const TCUDAVector<Number> & rhs );
	TCUDAVector<Number> & operator-=( const TCUDAVector<Number> & rhs );
	
	Number operator*( const TCUDAVector<Number> & rhs ) const; // DOT product
	
	// Normalization
	Number NormSqr() const;
	Number Norm() const;
	Number InvNormSqr() const;
	Number InvNorm() const;
	Number Normalize();
	
private:
	CUDADeviceMemory m_hDeviceVector;
};

// Template Specializations
typedef TCUDAVector<Float> CUDAVectorF;
typedef TCUDAVector<Double> CUDAVectorD;
typedef TCUDAVector<ComplexFloat> CUDAVectorCF;
typedef TCUDAVector<ComplexDouble> CUDAVectorCD;

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUDAVector.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_TYPES_CUDAVECTOR_H

