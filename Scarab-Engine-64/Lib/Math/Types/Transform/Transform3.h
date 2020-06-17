/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Transform/Transform3.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Generic transformations in 3D
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
#ifndef SCARAB_LIB_MATH_TYPES_TRANSFORM_TRANSFORM3_H
#define SCARAB_LIB_MATH_TYPES_TRANSFORM_TRANSFORM3_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../../../Error/ErrorManager.h"

#include "../Matrix/Matrix3.h"
#include "../Matrix/Matrix4.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The TTransform3 class
template<typename Real>
class TTransform3
{
public:
    // Constant values
    static const TTransform3<Real> Identity;

    // Constructors
    TTransform3();
    TTransform3( const TMatrix3<Real> & matRotate );
    TTransform3( const TMatrix3<Real> & matRotate, const TVector3<Real> & vTranslate );
    TTransform3( const TMatrix3<Real> & matRotate, const TVector3<Real> & vTranslate, const Real & fUniformScale );
    TTransform3( const TMatrix3<Real> & matRotate, const TVector3<Real> & vTranslate, const TVector3<Real> & vScale );
    TTransform3( const TTransform3<Real> & rhs );
    ~TTransform3();

    // Assignment operator
    TTransform3<Real> & operator=( const TTransform3<Real> & rhs );

    // Vertex operations
	inline TVertex4<Real> operator*( const TVertex4<Real> & rhs ) const;
	inline TVertex3<Real> operator*( const TVertex3<Real> & rhs ) const;

	// Vector operations
	inline TVector4<Real> operator*( const TVector4<Real> & rhs ) const;
	inline TVector3<Real> operator*( const TVector3<Real> & rhs ) const;

    // Transform operations
	TTransform3<Real> operator*( const TTransform3<Real> & rhs ) const;

    // Construction
    Void SetRotate( const TMatrix3<Real> & matRotate );
    Void SetTranslate( const TVector3<Real> & vTranslate );
    Void SetUniformScale( const Real & fUniformScale );
    Void SetScale( const TVector3<Real> & vScale );

    // Getters
    inline Bool IsIdentity() const;
    inline Bool HasScale() const;
    inline Bool IsUniformScale() const;

    const TMatrix3<Real> & GetRotate() const;
    inline const TVector3<Real> & GetTranslate() const;
    inline const Real & GetUniformScale() const;
    inline const TVector3<Real> & GetScale() const;

    inline const TMatrix4<Real> & GetMatrix() const;

    // Methods
    Void MakeIdentity();
    Void MakeUnitScale();

    Real GetMaxScale() const; // ie. Spectral Norm

    Void Invert( TTransform3<Real> & outInvTransform ) const;
    Void Invert();

private:
    // Helpers
    Void _UpdateInverse() const;

    // Flags
    Bool m_bIsIdentity;
    Bool m_bHasScale;
    Bool m_bIsUniformScale;

    // Components
    TVector3<Real> m_vScale;
    TVector3<Real> m_vTranslate;
    TMatrix4<Real> m_matTransform;

    // Inverse support
    mutable Bool m_bUpdateInverse;
    mutable TVector3<Real> m_vInvScale;
    mutable TVector3<Real> m_vInvTranslate;
    mutable TMatrix4<Real> m_matInvTransform;
};

// Explicit instanciation
typedef TTransform3<Float> Transform3f;
typedef TTransform3<Double> Transform3d;
typedef TTransform3<Scalar> Transform3;

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "Transform3.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MATH_TYPES_TRANSFORM_TRANSFORM3_H


