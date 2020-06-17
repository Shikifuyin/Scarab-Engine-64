/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Transform/Transform2.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Generic transformations in 2D
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_LIB_MATH_TYPES_TRANSFORM_TRANSFORM2_H
#define SCARAB_LIB_MATH_TYPES_TRANSFORM_TRANSFORM2_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../../../Error/ErrorManager.h"

#include "../Matrix/Matrix2.h"
#include "../Matrix/Matrix3.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The TTransform2 class
template<typename Real>
class TTransform2
{
public:
    // Constant values
    static const TTransform2<Real> Identity;

    // Constructors
    TTransform2();
    TTransform2( const TMatrix2<Real> & matRotate );
    TTransform2( const TMatrix2<Real> & matRotate, const TVector2<Real> & vTranslate );
    TTransform2( const TMatrix2<Real> & matRotate, const TVector2<Real> & vTranslate, const Real & fUniformScale );
    TTransform2( const TMatrix2<Real> & matRotate, const TVector2<Real> & vTranslate, const TVector2<Real> & vScale );
    TTransform2( const TTransform2<Real> & rhs );
    ~TTransform2();

    // Assignment operator
    TTransform2<Real> & operator=( const TTransform2<Real> & rhs );

    // Vertex operations
	inline TVertex3<Real> operator*( const TVertex3<Real> & rhs ) const;
	inline TVertex2<Real> operator*( const TVertex2<Real> & rhs ) const;

	// Vector operations
	inline TVector3<Real> operator*( const TVector3<Real> & rhs ) const;
	inline TVector2<Real> operator*( const TVector2<Real> & rhs ) const;

    // Transform operations
	TTransform2<Real> operator*( const TTransform2<Real> & rhs ) const;

    // Construction
    Void SetRotate( const TMatrix2<Real> & matRotate );
    Void SetTranslate( const TVector2<Real> & vTranslate );
    Void SetUniformScale( const Real & fUniformScale );
    Void SetScale( const TVector2<Real> & vScale );

    // Getters
    inline Bool IsIdentity() const;
    inline Bool HasScale() const;
    inline Bool IsUniformScale() const;

    const TMatrix2<Real> & GetRotate() const;
    inline const TVector2<Real> & GetTranslate() const;
    inline const Real & GetUniformScale() const;
    inline const TVector2<Real> & GetScale() const;

    inline const TMatrix3<Real> & GetMatrix() const;

    // Methods
    Void MakeIdentity();
    Void MakeUnitScale();

    Real GetMaxScale() const; // ie. Spectral Norm

    Void Invert( TTransform2<Real> & outInvTransform ) const;
    Void Invert();

private:
    Void _UpdateInverse() const;

    // Flags
    Bool m_bIsIdentity;
    Bool m_bHasScale;
    Bool m_bIsUniformScale;

    // Components
    TVector2<Real> m_vScale;
    TVector2<Real> m_vTranslate;
    TMatrix3<Real> m_matTransform;
    
    // Invert support
    mutable Bool m_bUpdateInverse;
    mutable TVector2<Real> m_vInvScale;
    mutable TVector2<Real> m_vInvTranslate;
    mutable TMatrix3<Real> m_matInvTransform;
};

// Explicit instanciation
typedef TTransform2<Float> Transform2f;
typedef TTransform2<Double> Transform2d;
typedef TTransform2<Scalar> Transform2;

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "Transform2.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MATH_TYPES_TRANSFORM_TRANSFORM2_H


