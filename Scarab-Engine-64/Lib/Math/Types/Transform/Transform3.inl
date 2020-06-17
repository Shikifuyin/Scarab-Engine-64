/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Transform/Transform3.inl
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
// TTransform3 implementation
template<typename Real> const TTransform3<Real> TTransform3<Real>::Identity;

template<typename Real>
TTransform3<Real>::TTransform3():
    m_vScale( MathFunction<Real>::One, MathFunction<Real>::One, MathFunction<Real>::One ),
    m_vTranslate( TVector3<Real>::Null ),
    m_matTransform( TMatrix4<Real>::Identity ),
    m_vInvScale( MathFunction<Real>::One, MathFunction<Real>::One, MathFunction<Real>::One ),
    m_vInvTranslate( TVector3<Real>::Null ),
    m_matInvTransform( TMatrix4<Real>::Identity )
{
    m_bIsIdentity = true;
    m_bHasScale = false;
    m_bIsUniformScale = true;

    m_bUpdateInverse = false;
}
template<typename Real>
TTransform3<Real>::TTransform3( const TMatrix3<Real> & matRotate ):
    m_vScale( MathFunction<Real>::One, MathFunction<Real>::One, MathFunction<Real>::One ),
    m_vTranslate( TVector3<Real>::Null ),
    m_matTransform( TMatrix4<Real>::Identity ),
    m_vInvScale( MathFunction<Real>::One, MathFunction<Real>::One, MathFunction<Real>::One ),
    m_vInvTranslate( TVector3<Real>::Null ),
    m_matInvTransform( TMatrix4<Real>::Identity )
{
    m_bIsIdentity = false;
    m_bHasScale = false;
    m_bIsUniformScale = true;

    // Setup Transform
    m_matTransform = TMatrix4<Real>( matRotate );

    // Update Inverse
    m_bUpdateInverse = true;
}
template<typename Real>
TTransform3<Real>::TTransform3( const TMatrix3<Real> & matRotate, const TVector3<Real> & vTranslate ):
    m_vScale( MathFunction<Real>::One, MathFunction<Real>::One, MathFunction<Real>::One ),
    m_vTranslate( TVector3<Real>::Null ),
    m_matTransform( TMatrix4<Real>::Identity ),
    m_vInvScale( MathFunction<Real>::One, MathFunction<Real>::One, MathFunction<Real>::One ),
    m_vInvTranslate( TVector3<Real>::Null ),
    m_matInvTransform( TMatrix4<Real>::Identity )
{
    m_bIsIdentity = false;
    m_bHasScale = false;
    m_bIsUniformScale = true;

    // Setup Transform
    m_vTranslate = vTranslate;

    m_matTransform = TMatrix4<Real>( matRotate );
    m_matTransform.SetTranslate( m_vTranslate );

    // Update Inverse
    m_bUpdateInverse = true;
}
template<typename Real>
TTransform3<Real>::TTransform3( const TMatrix3<Real> & matRotate, const TVector3<Real> & vTranslate, const Real & fUniformScale ):
    m_vScale( MathFunction<Real>::One, MathFunction<Real>::One, MathFunction<Real>::One ),
    m_vTranslate( TVector3<Real>::Null ),
    m_matTransform( TMatrix4<Real>::Identity ),
    m_vInvScale( MathFunction<Real>::One, MathFunction<Real>::One, MathFunction<Real>::One ),
    m_vInvTranslate( TVector3<Real>::Null ),
    m_matInvTransform( TMatrix4<Real>::Identity )
{
    m_bIsIdentity = false;
    m_bHasScale = true;
    m_bIsUniformScale = true;

    // Setup Transform
    m_vScale = TVector3<Real>( fUniformScale, fUniformScale, fUniformScale );

    m_vTranslate = vTranslate;

    m_matTransform = TMatrix4<Real>( matRotate * fUniformScale );
    m_matTransform.SetTranslate( m_vTranslate );

    // Update Inverse
    m_bUpdateInverse = true;
}
template<typename Real>
TTransform3<Real>::TTransform3( const TMatrix3<Real> & matRotate, const TVector3<Real> & vTranslate, const TVector3<Real> & vScale ):
    m_vScale( MathFunction<Real>::One, MathFunction<Real>::One, MathFunction<Real>::One ),
    m_vTranslate( TVector3<Real>::Null ),
    m_matTransform( TMatrix4<Real>::Identity ),
    m_vInvScale( MathFunction<Real>::One, MathFunction<Real>::One, MathFunction<Real>::One ),
    m_vInvTranslate( TVector3<Real>::Null ),
    m_matInvTransform( TMatrix4<Real>::Identity )
{
    m_bIsIdentity = false;
    m_bHasScale = true;
    m_bIsUniformScale = false;

    // Setup Transform
    m_vScale = vScale;
    TMatrix4<Real> matScale;
    matScale.MakeScale( m_vScale );

    m_vTranslate = vTranslate;

    m_matTransform = ( TMatrix4<Real>(matRotate) * matScale );
    m_matTransform.SetTranslate( m_vTranslate );

    // Update Inverse
    m_bUpdateInverse = true;
}
template<typename Real>
TTransform3<Real>::TTransform3( const TTransform3<Real> & rhs ):
    m_vScale( rhs.m_vScale ),
    m_vTranslate( rhs.m_vTranslate ),
    m_matTransform( rhs.m_matTransform ),
    m_vInvScale( rhs.m_vInvScale ),
    m_vInvTranslate( rhs.m_vInvTranslate ),
    m_matInvTransform( rhs.m_matInvTransform )
{
    m_bIsIdentity = rhs.m_bIsIdentity;
    m_bHasScale = rhs.m_bHasScale;
    m_bIsUniformScale = rhs.m_bIsUniformScale;

    m_bUpdateInverse = rhs.m_bUpdateInverse;
}
template<typename Real>
TTransform3<Real>::~TTransform3()
{
    // nothing to do
}

template<typename Real>
TTransform3<Real> & TTransform3<Real>::operator=( const TTransform3<Real> & rhs )
{
    m_bIsIdentity = rhs.m_bIsIdentity;
    m_bHasScale = rhs.m_bHasScale;
    m_bIsUniformScale = rhs.m_bIsUniformScale;

    m_vScale = rhs.m_vScale;
    m_vTranslate = rhs.m_vTranslate;
    m_matTransform = rhs.m_matTransform;

    m_vInvScale = rhs.m_vInvScale;
    m_vInvTranslate = rhs.m_vInvTranslate;
    m_matInvTransform = rhs.m_matInvTransform;

    m_bUpdateInverse = rhs.m_bUpdateInverse;

    return (*this);
}

template<typename Real>
inline TVertex4<Real> TTransform3<Real>::operator*( const TVertex4<Real> & rhs ) const {
    return m_matTransform * rhs;
}
template<typename Real>
inline TVertex3<Real> TTransform3<Real>::operator*( const TVertex3<Real> & rhs ) const {
    return TVertex3<Real>( m_matTransform * TVertex4<Real>(rhs) );
}

template<typename Real>
inline TVector4<Real> TTransform3<Real>::operator*( const TVector4<Real> & rhs ) const {
    return m_matTransform * rhs;
}
template<typename Real>
inline TVector3<Real> TTransform3<Real>::operator*( const TVector3<Real> & rhs ) const {
    return TVector3<Real>( m_matTransform * TVector4<Real>(rhs) );
}

template<typename Real>
TTransform3<Real> TTransform3<Real>::operator*( const TTransform3<Real> & rhs ) const
{
    if ( m_bIsIdentity )
        return rhs;
    if ( rhs.m_bIsIdentity )
        return (*this);

    TTransform3<Real> trComposed;

    trComposed.m_bIsIdentity = false;
    trComposed.m_bHasScale = ( m_bHasScale || rhs.m_bHasScale );
    trComposed.m_bIsUniformScale = ( m_bIsUniformScale && rhs.m_bIsUniformScale );

    trComposed.m_vScale.X = m_vScale.X * rhs.m_vScale.X;
    trComposed.m_vScale.Y = m_vScale.Y * rhs.m_vScale.Y;
    trComposed.m_vScale.Z = m_vScale.Z * rhs.m_vScale.Z;

    TMatrix3<Real> matRotScale = TMatrix3<Real>( m_matTransform );
    trComposed.m_vTranslate = ( matRotScale * rhs.m_vTranslate ) + m_vTranslate;

    trComposed.m_matTransform = m_matTransform * rhs.m_matTransform;

    trComposed.m_bUpdateInverse = true;

    return trComposed;
}

template<typename Real>
Void TTransform3<Real>::SetRotate( const TMatrix3<Real> & matRotate )
{
    m_bIsIdentity = false;

    if ( m_bHasScale ) {
        if ( m_bIsUniformScale )
            m_matTransform = TMatrix4<Real>( matRotate  * m_vScale.X );
        else {
            TMatrix4<Real> matScale;
            matScale.MakeScale( m_vScale );
            m_matTransform = ( TMatrix4<Real>(matRotate) * matScale );
        }
    } else
        m_matTransform = TMatrix4<Real>( matRotate );

    m_matTransform.SetTranslate( m_vTranslate );

    // Update Inverse
    m_bUpdateInverse = true;
}
template<typename Real>
Void TTransform3<Real>::SetTranslate( const TVector3<Real> & vTranslate )
{
    m_bIsIdentity = false;

    m_vTranslate = vTranslate;
    m_matTransform.SetTranslate( m_vTranslate );

    // Update Inverse
    m_bUpdateInverse = true;
}
template<typename Real>
Void TTransform3<Real>::SetUniformScale( const Real & fUniformScale )
{
    TMatrix3<Real> matRotate = GetRotate();

    m_bIsIdentity = false;
    m_bHasScale = true;
    m_bIsUniformScale = true;

    m_vScale = TVector3<Real>( fUniformScale, fUniformScale, fUniformScale );

    m_matTransform = TMatrix4<Real>( matRotate * fUniformScale );
    m_matTransform.SetTranslate( m_vTranslate );

    // Update Inverse
    m_bUpdateInverse = true;
}
template<typename Real>
Void TTransform3<Real>::SetScale( const TVector3<Real> & vScale )
{
    TMatrix3<Real> matRotate = GetRotate();

    m_bIsIdentity = false;
    m_bHasScale = true;
    m_bIsUniformScale = false;
    
    m_vScale = vScale;
    TMatrix4<Real> matScale;
    matScale.MakeScale( m_vScale );

    m_matTransform = ( TMatrix4<Real>(matRotate) * matScale );
    m_matTransform.SetTranslate( m_vTranslate );

    // Update Inverse
    m_bUpdateInverse = true;
}

template<typename Real>
inline Bool TTransform3<Real>::IsIdentity() const {
    return m_bIsIdentity;
}
template<typename Real>
inline Bool TTransform3<Real>::HasScale() const {
    return m_bHasScale;
}
template<typename Real>
inline Bool TTransform3<Real>::IsUniformScale() const {
    return ( m_bHasScale ? m_bIsUniformScale : true );
}

template<typename Real>
const TMatrix3<Real> & TTransform3<Real>::GetRotate() const
{
    if ( !m_bHasScale )
        return TMatrix3<Real>( m_matTransform );

    TMatrix3<Real> matRotScale = TMatrix3<Real>( m_matTransform );

    if ( m_bIsUniformScale ) {
        Real fInvUniformScale;
        if ( m_bUpdateInverse )
            fInvUniformScale = MathRealFn->Invert( m_vScale.X );
        else
            fInvUniformScale = m_vInvScale.X;

        return ( matRotScale * fInvUniformScale );
    }

    TMatrix3<Real> matInvScale;
    if ( m_bUpdateInverse )
        matInvScale.MakeDiagonal( MathRealFn->Invert( m_vScale.X ), MathRealFn->Invert( m_vScale.Y ), MathRealFn->Invert( m_vScale.Z ) );
    else
        matInvScale.MakeScale( m_vInvScale );

    return ( matRotScale * matInvScale );
}
template<typename Real>
inline const TVector3<Real> & TTransform3<Real>::GetTranslate() const {
    return m_vTranslate;
}
template<typename Real>
inline const Real & TTransform3<Real>::GetUniformScale() const {
    Assert( IsUniformScale() );
    return m_vScale.X;
}
template<typename Real>
inline const TVector3<Real> & TTransform3<Real>::GetScale() const {
    return m_vScale;
}

template<typename Real>
inline const TMatrix4<Real> & TTransform3<Real>::GetMatrix() const {
    return m_matTransform;
}

template<typename Real>
Void TTransform3<Real>::MakeIdentity()
{
    m_bIsIdentity = true;
    m_bHasScale = false;
    m_bIsUniformScale = true;

    m_vScale = TVector3<Real>( MathFunction<Real>::One, MathFunction<Real>::One, MathFunction<Real>::One );
    m_vTranslate = TVector3<Real>::Null;
    m_matTransform = TMatrix4<Real>::Identity;

    // Update Inverse
    m_bUpdateInverse = true;
}
template<typename Real>
Void TTransform3<Real>::MakeUnitScale()
{
    TMatrix3<Real> matRotate = GetRotate();

    m_bHasScale = false;
    m_bIsUniformScale = true;

    m_vScale = TVector3<Real>( MathFunction<Real>::One, MathFunction<Real>::One, MathFunction<Real>::One );

    m_matTransform = TMatrix4<Real>( matRotate );
    m_matTransform.SetTranslate( m_vTranslate );

    // Update Inverse
    m_bUpdateInverse = true;
}

template<typename Real>
Real TTransform3<Real>::GetMaxScale() const
{
    // Basically the spectral norm in our case ...
    if ( !m_bHasScale )
        return MathFunction<Real>::One;

    if ( m_bIsUniformScale )
        return MathRealFn->Abs( m_vScale.X );

    Real fMaxScale = MathRealFn->Abs( m_vScale.X );
    Real fScale = MathRealFn->Abs( m_vScale.Y );
    if ( fScale > fMaxScale )
        fMaxScale = fScale;
    fScale = MathRealFn->Abs( m_vScale.Z );
    if ( fScale > fMaxScale )
        fMaxScale = fScale;
    return fMaxScale;
}

template<typename Real>
Void TTransform3<Real>::Invert( TTransform3<Real> & outInvTransform ) const
{
    if ( m_bUpdateInverse )
        _UpdateInverse();

    if ( m_bIsIdentity ) {
        outInvTransform.MakeIdentity();
        return;
    }

    outInvTransform.m_bIsIdentity = false;
    outInvTransform.m_bHasScale = m_bHasScale;
    outInvTransform.m_bIsUniformScale = m_bIsUniformScale;

    outInvTransform.m_vScale = m_vInvScale;
    outInvTransform.m_vTranslate = m_vInvTranslate;
    outInvTransform.m_matTransform = m_matInvTransform;

    outInvTransform.m_bUpdateInverse = false;
    outInvTransform.m_vInvScale = m_vScale;
    outInvTransform.m_vInvTranslate = m_vTranslate;
    outInvTransform.m_matInvTransform = m_matTransform;
}
template<typename Real>
Void TTransform3<Real>::Invert()
{
    if ( m_bUpdateInverse )
        _UpdateInverse();

    if ( m_bIsIdentity )
        return;

    Swap<TVector3<Real> >( &m_vScale, &m_vInvScale );
    Swap<TVector3<Real> >( &m_vTranslate, &m_vInvTranslate );
    Swap<TMatrix4<Real> >( &m_matTransform, &m_matInvTransform );
}

/////////////////////////////////////////////////////////////////////////////////

template<typename Real>
Void TTransform3<Real>::_UpdateInverse() const
{
    if ( !m_bUpdateInverse )
        return;

    if ( m_bIsIdentity ) {
        m_vInvScale = TVector3<Real>( MathFunction<Real>::One, MathFunction<Real>::One, MathFunction<Real>::One );
        m_vInvTranslate = TVector3<Real>::Null;
        m_matInvTransform = TMatrix4<Real>::Identity;

        m_bUpdateInverse = false;
        return;
    }

    if ( !m_bHasScale ) {
        m_vInvScale = TVector3<Real>( MathFunction<Real>::One, MathFunction<Real>::One, MathFunction<Real>::One );

        TMatrix3<Real> matRotate = TMatrix3<Real>( m_matTransform );
        TMatrix3<Real> matInvRotate;
        matRotate.Transpose( matInvRotate );

        m_vInvTranslate = -( matInvRotate * m_vTranslate );

        m_matInvTransform = TMatrix4<Real>( matInvRotate );
        m_matInvTransform.SetTranslate( m_vInvTranslate );

        m_bUpdateInverse = false;
        return;
    }

    TMatrix3<Real> matRotScale = TMatrix3<Real>( m_matTransform );

    if ( m_bIsUniformScale ) {
        Real fInvUniformScale = MathRealFn->Invert( m_vScale.X );
        m_vInvScale.X = fInvUniformScale;
        m_vInvScale.Y = fInvUniformScale;
        m_vInvScale.Z = fInvUniformScale;

        matRotScale *= fInvUniformScale;

        TMatrix3<Real> matInvRotate;
        matRotScale.Transpose( matInvRotate );

        m_vInvTranslate = -( (matInvRotate * m_vTranslate) * fInvUniformScale );

        m_matInvTransform = TMatrix4<Real>( matInvRotate * fInvUniformScale );
        m_matInvTransform.SetTranslate( m_vInvTranslate );

        m_bUpdateInverse = false;
        return;
    }

    m_vInvScale.X = MathRealFn->Invert( m_vScale.X );
    m_vInvScale.Y = MathRealFn->Invert( m_vScale.Y );
    m_vInvScale.Z = MathRealFn->Invert( m_vScale.Z );

    TMatrix3<Real> matInvScale;
    matInvScale.MakeScale( m_vInvScale );

    matRotScale *= matInvScale;

    TMatrix3<Real> matInvRotScale;
    matRotScale.Transpose( matInvRotScale );
    matInvRotScale *= matInvScale;

    m_vInvTranslate = -( matInvRotScale * m_vTranslate );

    m_matInvTransform = TMatrix4<Real>( matInvRotScale );
    m_matInvTransform.SetTranslate( m_vInvTranslate );

    m_bUpdateInverse = false;
}


