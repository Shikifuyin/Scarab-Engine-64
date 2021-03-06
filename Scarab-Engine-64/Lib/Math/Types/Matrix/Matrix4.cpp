/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Matrix/Matrix4.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Homogeneous 4D matrix
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "Matrix2.h"
#include "Matrix3.h"
#include "Matrix4.h"

/////////////////////////////////////////////////////////////////////////////////
// TMatrix4 implementation
#ifdef SIMD_ENABLE

template<>
TMatrix4<Float> TMatrix4<Float>::operator*( const Float & rhs ) const
{
    TMatrix4<Float> matRes;

    __m256 mRHS = SIMD::Import::Memory::Spread256( &rhs );

    __m256 mTwoCols = SIMD::Import::Memory::Aligned::Load256( &m00 );
    mTwoCols = SIMD::Math::Mul( mTwoCols, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mTwoCols );

    mTwoCols = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mTwoCols = SIMD::Math::Mul( mTwoCols, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m02), mTwoCols );

    return matRes;
}
template<>
TMatrix4<Double> TMatrix4<Double>::operator*( const Double & rhs ) const
{
    TMatrix4<Double> matRes;

    __m256d mRHS = SIMD::Import::Memory::Spread256( &rhs );
    
    __m256d mCol = SIMD::Import::Memory::Aligned::Load256( &m00 );
    mCol = SIMD::Math::Mul( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mCol );

    mCol = SIMD::Import::Memory::Aligned::Load256( &m01 );
    mCol = SIMD::Math::Mul( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m01), mCol );

    mCol = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mCol = SIMD::Math::Mul( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m02), mCol );

    mCol = SIMD::Import::Memory::Aligned::Load256( &m03 );
    mCol = SIMD::Math::Mul( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m03), mCol );

    return matRes;
}

template<>
TMatrix4<Float> TMatrix4<Float>::operator/( const Float & rhs ) const
{
    TMatrix4<Float> matRes;

    __m256 mRHS = SIMD::Import::Memory::Spread256( &rhs );

    __m256 mTwoCols = SIMD::Import::Memory::Aligned::Load256( &m00 );
    mTwoCols = SIMD::Math::Div( mTwoCols, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mTwoCols );

    mTwoCols = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mTwoCols = SIMD::Math::Div( mTwoCols, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m02), mTwoCols );

    return matRes;
}
template<>
TMatrix4<Double> TMatrix4<Double>::operator/( const Double & rhs ) const
{
    TMatrix4<Double> matRes;

    __m256d mRHS = SIMD::Import::Memory::Spread256( &rhs );
    
    __m256d mCol = SIMD::Import::Memory::Aligned::Load256( &m00 );
    mCol = SIMD::Math::Div( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mCol );

    mCol = SIMD::Import::Memory::Aligned::Load256( &m01 );
    mCol = SIMD::Math::Div( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m01), mCol );

    mCol = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mCol = SIMD::Math::Div( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m02), mCol );

    mCol = SIMD::Import::Memory::Aligned::Load256( &m03 );
    mCol = SIMD::Math::Div( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m03), mCol );

    return matRes;
}

template<>
TMatrix4<Float> & TMatrix4<Float>::operator*=( const Float & rhs )
{
    __m256 mRHS = SIMD::Import::Memory::Spread256( &rhs );

    __m256 mTwoCols = SIMD::Import::Memory::Aligned::Load256( &m00 );
    mTwoCols = SIMD::Math::Mul( mTwoCols, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m00, mTwoCols );

    mTwoCols = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mTwoCols = SIMD::Math::Mul( mTwoCols, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m02, mTwoCols );

    return (*this);
}
template<>
TMatrix4<Double> & TMatrix4<Double>::operator*=( const Double & rhs )
{
    __m256d mRHS = SIMD::Import::Memory::Spread256( &rhs );
    
    __m256d mCol = SIMD::Import::Memory::Aligned::Load256( &m00 );
    mCol = SIMD::Math::Mul( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m00, mCol );

    mCol = SIMD::Import::Memory::Aligned::Load256( &m01 );
    mCol = SIMD::Math::Mul( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m01, mCol );

    mCol = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mCol = SIMD::Math::Mul( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m02, mCol );

    mCol = SIMD::Import::Memory::Aligned::Load256( &m03 );
    mCol = SIMD::Math::Mul( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m03, mCol );

    return (*this);
}

template<>
TMatrix4<Float> & TMatrix4<Float>::operator/=( const Float & rhs )
{
    __m256 mRHS = SIMD::Import::Memory::Spread256( &rhs );

    __m256 mTwoCols = SIMD::Import::Memory::Aligned::Load256( &m00 );
    mTwoCols = SIMD::Math::Div( mTwoCols, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m00, mTwoCols );

    mTwoCols = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mTwoCols = SIMD::Math::Div( mTwoCols, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m02, mTwoCols );

    return (*this);
}
template<>
TMatrix4<Double> & TMatrix4<Double>::operator/=( const Double & rhs )
{
    __m256d mRHS = SIMD::Import::Memory::Spread256( &rhs );
    
    __m256d mCol = SIMD::Import::Memory::Aligned::Load256( &m00 );
    mCol = SIMD::Math::Div( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m00, mCol );

    mCol = SIMD::Import::Memory::Aligned::Load256( &m01 );
    mCol = SIMD::Math::Div( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m01, mCol );

    mCol = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mCol = SIMD::Math::Div( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m02, mCol );

    mCol = SIMD::Import::Memory::Aligned::Load256( &m03 );
    mCol = SIMD::Math::Div( mCol, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m03, mCol );

    return (*this);
}

template<>
TVertex4<Float> TMatrix4<Float>::operator*( const TVertex4<Float> & rhs ) const
{
    // This is one of the most important ones ! Very careful optimisation here !
    TVertex4<Float> vRes;

    // Load Data
    __m256 m256Tmp0 = SIMD::Import::Memory::Aligned::Load256( &m00 );     // m256Tmp0 = ( m00, m10, m20, m30, m01, m11, m21, m31 )
    __m256 m256Tmp1 = SIMD::Import::Memory::Aligned::Load256( &m02 );     // m256Tmp1 = ( m02, m12, m22, m32, m03, m13, m23, m33 )
    __m128 m128Tmp0 = SIMD::Import::Memory::Aligned::Load128( &(rhs.X) ); // m128Tmp0 = ( v0, v1, v2, v3 )

    // Build Matrix columns
    __m256 mCols02 = SIMD::Register::Shuffle2::AC( m256Tmp0, m256Tmp1 ); // mCols02 = ( m00, m10, m20, m30, m02, m12, m22, m32 )
    __m256 mCols13 = SIMD::Register::Shuffle2::BD( m256Tmp0, m256Tmp1 ); // mCols13 = ( m01, m11, m21, m31, m03, m13, m23, m33 )

    // Build RHS vectors
    __m256 mRHS02 = SIMD::Cast::Up( SIMD::Register::Spread4::AAAA(m128Tmp0) ); // mRHS02 = ( v0, v0, v0, v0, ?, ?, ?, ? )
    __m256 mRHS13 = SIMD::Cast::Up( SIMD::Register::Spread4::BBBB(m128Tmp0) ); // mRHS13 = ( v1, v1, v1, v1, ?, ?, ?, ? )

    __m128 m128Tmp1 = SIMD::Register::Spread4::CCCC( m128Tmp0 );   // m128Tmp1 = ( v2, v2, v2, v2 )
    mRHS02 = SIMD::Register::Move::FourFloatH( mRHS02, m128Tmp1 ); // mRHS02 = ( v0, v0, v0, v0, v2, v2, v2, v2 )

    m128Tmp1 = SIMD::Register::Spread4::DDDD( m128Tmp0 );          // m128Tmp1 = ( v3, v3, v3, v3 )
    mRHS13 = SIMD::Register::Move::FourFloatH( mRHS13, m128Tmp1 ); // mRHS13 = ( v1, v1, v1, v1, v3, v3, v3, v3 )    

    // Perform Product
    m256Tmp0 = SIMD::Math::Mul( mCols02, mRHS02 ); // m256Tmp0 = ( m00 * v0, m10 * v0, m20 * v0, m30 * v0, m02 * v2, m12 * v2, m22 * v2, m32 * v2 )
    m256Tmp1 = SIMD::Math::Mul( mCols13, mRHS13 ); // m256Tmp1 = ( m01 * v1, m11 * v1, m21 * v1, m31 * v1, m03 * v3, m13 * v3, m23 * v3, m33 * v3 )

    m256Tmp0 = SIMD::Math::Add( m256Tmp0, m256Tmp1 ); // m256Tmp0 = ( m00*v0 + m01*v1, m10*v0 + m11*v1, m20*v0 + m21*v1, m30*v0 + m31*v1,
                                                      //              m02*v2 + m03*v3, m12*v2 + m13*v3, m22*v2 + m23*v3, m32*v2 + m33*v3 )

    m128Tmp0 = SIMD::Register::Move::FourFloatL( m256Tmp0 ); // m128Tmp0 = ( m00*v0 + m01*v1, m10*v0 + m11*v1, m20*v0 + m21*v1, m30*v0 + m31*v1 )
    m128Tmp1 = SIMD::Register::Move::FourFloatH( m256Tmp0 ); // m128Tmp1 = ( m02*v2 + m03*v3, m12*v2 + m13*v3, m22*v2 + m23*v3, m32*v2 + m33*v3 )

    m128Tmp0 = SIMD::Math::Add( m128Tmp0, m128Tmp1 ); // m128Tmp0 = ( m00*v0 + m01*v1 + m02*v2 + m03*v3, m10*v0 + m11*v1 + m12*v2 + m13*v3,
                                                      //              m20*v0 + m21*v1 + m22*v2 + m23*v3, m30*v0 + m31*v1 + m32*v2 + m33*v3 )

    // Save Data
    SIMD::Export::Memory::Aligned::Save128( &(vRes.X), m128Tmp0 );

    // Done
    return vRes;
}
template<>
TVertex4<Double> TMatrix4<Double>::operator*( const TVertex4<Double> & rhs ) const
{
    // This is one of the most important ones ! Very careful optimisation here !
    TVertex4<Double> vRes;

    // Load Data
    __m256d mTmp0 = SIMD::Import::Memory::Aligned::Load256( &(rhs.X) ); // mTmp0 = ( v0, v1, v2, v3 )
    __m256d mCol0 = SIMD::Import::Memory::Aligned::Load256( &m00 );     // mCol0 = ( m00, m10, m20, m30 )
    __m256d mCol1 = SIMD::Import::Memory::Aligned::Load256( &m01 );     // mCol1 = ( m01, m11, m21, m31 )
    __m256d mCol2 = SIMD::Import::Memory::Aligned::Load256( &m02 );     // mCol2 = ( m02, m12, m22, m32 )
    __m256d mCol3 = SIMD::Import::Memory::Aligned::Load256( &m03 );     // mCol3 = ( m03, m13, m23, m33 )
    
    // Build RHS vectors
    __m256d mRHS0 = SIMD::Register::Spread4::AAAA( mTmp0 ); // mRHS0 = ( v0, v0, v0, v0 )
    __m256d mRHS1 = SIMD::Register::Spread4::BBBB( mTmp0 ); // mRHS1 = ( v1, v1, v1, v1 )
    __m256d mRHS2 = SIMD::Register::Spread4::CCCC( mTmp0 ); // mRHS2 = ( v2, v2, v2, v2 )
    __m256d mRHS3 = SIMD::Register::Spread4::DDDD( mTmp0 ); // mRHS3 = ( v3, v3, v3, v3 )

    // Perform Product
    mTmp0 = SIMD::Math::Mul( mCol0, mRHS0 );         // mTmp0 = ( m00 * v0, m10 * v0, m20 * v0, m30 * v0 )
    __m256d mTmp1 = SIMD::Math::Mul( mCol1, mRHS1 ); // mTmp1 = ( m01 * v1, m11 * v1, m21 * v1, m31 * v1 )

    mTmp0 = SIMD::Math::Add( mTmp0, mTmp1 ); // mTmp0 = ( m00*v0 + m01*v1, m10*v0 + m11*v1, m20*v0 + m21*v1, m30*v0 + m31*v1 )

    mTmp1 = SIMD::Math::Mul( mCol2, mRHS2 ); // mTmp1 = ( m02 * v2, m12 * v2, m22 * v2, m32 * v2 )

    mTmp0 = SIMD::Math::Add( mTmp0, mTmp1 ); // mTmp0 = ( m00*v0 + m01*v1 + m02*v2, m10*v0 + m11*v1 + m12*v2,
                                             //           m20*v0 + m21*v1 + m22*v2, m30*v0 + m31*v1 + m32*v2 )

    mTmp1 = SIMD::Math::Mul( mCol3, mRHS3 ); // mTmp1 = ( m03 * v3, m13 * v3, m23 * v3, m33 * v3 )

    mTmp0 = SIMD::Math::Add( mTmp0, mTmp1 ); // mTmp0 = ( m00*v0 + m01*v1 + m02*v2 + m03*v3, m10*v0 + m11*v1 + m12*v2 + m13*v3,
                                             //           m20*v0 + m21*v1 + m22*v2 + m23*v3, m30*v0 + m31*v1 + m32*v2 + m33*v3 )

    // Save Data
    SIMD::Export::Memory::Aligned::Save256( &(vRes.X), mTmp0 );

    // Done
    return vRes;
}

template<>
TVector4<Float> TMatrix4<Float>::operator*( const TVector4<Float> & rhs ) const
{
    // This is one of the most important ones ! Very careful optimisation here !
    TVector4<Float> vRes;

    // Load Data
    __m256 m256Tmp0 = SIMD::Import::Memory::Aligned::Load256( &m00 );     // m256Tmp0 = ( m00, m10, m20, m30, m01, m11, m21, m31 )
    __m256 m256Tmp1 = SIMD::Import::Memory::Aligned::Load256( &m02 );     // m256Tmp1 = ( m02, m12, m22, m32, m03, m13, m23, m33 )
    __m128 m128Tmp0 = SIMD::Import::Memory::Aligned::Load128( &(rhs.X) ); // m128Tmp0 = ( v0, v1, v2, v3 )

    // Build Matrix Columns
    __m256 mCols02 = SIMD::Register::Shuffle2::AC( m256Tmp0, m256Tmp1 ); // mCols02 = ( m00, m10, m20, m30, m02, m12, m22, m32 )
    __m256 mCols13 = SIMD::Register::Shuffle2::BD( m256Tmp0, m256Tmp1 ); // mCols13 = ( m01, m11, m21, m31, m03, m13, m23, m33 )

    // Build RHS vectors
    __m256 mRHS02 = SIMD::Cast::Up( SIMD::Register::Spread4::AAAA(m128Tmp0) ); // mRHS02 = ( v0, v0, v0, v0, ?, ?, ?, ? )
    __m256 mRHS13 = SIMD::Cast::Up( SIMD::Register::Spread4::BBBB(m128Tmp0) ); // mRHS13 = ( v1, v1, v1, v1, ?, ?, ?, ? )

    __m128 m128Tmp1 = SIMD::Register::Spread4::CCCC( m128Tmp0 );   // m128Tmp1 = ( v2, v2, v2, v2 )
    mRHS02 = SIMD::Register::Move::FourFloatH( mRHS02, m128Tmp1 ); // mRHS02 = ( v0, v0, v0, v0, v2, v2, v2, v2 )

    m128Tmp1 = SIMD::Register::Spread4::DDDD( m128Tmp0 );          // m128Tmp1 = ( v3, v3, v3, v3 )
    mRHS13 = SIMD::Register::Move::FourFloatH( mRHS13, m128Tmp1 ); // mRHS13 = ( v1, v1, v1, v1, v3, v3, v3, v3 )

    // Perform Product
    m256Tmp0 = SIMD::Math::Mul( mCols02, mRHS02 ); // m256Tmp0 = ( m00 * v0, m10 * v0, m20 * v0, m30 * v0, m02 * v2, m12 * v2, m22 * v2, m32 * v2 )
    m256Tmp1 = SIMD::Math::Mul( mCols13, mRHS13 ); // m256Tmp1 = ( m01 * v1, m11 * v1, m21 * v1, m31 * v1, m03 * v3, m13 * v3, m23 * v3, m33 * v3 )

    m256Tmp0 = SIMD::Math::Add( m256Tmp0, m256Tmp1 ); // m256Tmp0 = ( m00*v0 + m01*v1, m10*v0 + m11*v1, m20*v0 + m21*v1, m30*v0 + m31*v1,
                                                      //              m02*v2 + m03*v3, m12*v2 + m13*v3, m22*v2 + m23*v3, m32*v2 + m33*v3 )

    m128Tmp0 = SIMD::Register::Move::FourFloatL( m256Tmp0 ); // m128Tmp0 = ( m00*v0 + m01*v1, m10*v0 + m11*v1, m20*v0 + m21*v1, m30*v0 + m31*v1 )
    m128Tmp1 = SIMD::Register::Move::FourFloatH( m256Tmp0 ); // m128Tmp1 = ( m02*v2 + m03*v3, m12*v2 + m13*v3, m22*v2 + m23*v3, m32*v2 + m33*v3 )

    m128Tmp0 = SIMD::Math::Add( m128Tmp0, m128Tmp1 ); // m128Tmp0 = ( m00*v0 + m01*v1 + m02*v2 + m03*v3, m10*v0 + m11*v1 + m12*v2 + m13*v3,
                                                      //              m20*v0 + m21*v1 + m22*v2 + m23*v3, m30*v0 + m31*v1 + m32*v2 + m33*v3 )

    // Save Data
    SIMD::Export::Memory::Aligned::Save128( &(vRes.X), m128Tmp0 );

    // Done
    return vRes;
}
template<>
TVector4<Double> TMatrix4<Double>::operator*( const TVector4<Double> & rhs ) const
{
    // This is one of the most important ones ! Very careful optimisation here !
    TVector4<Double> vRes;

    // Load Data
    __m256d mTmp0 = SIMD::Import::Memory::Aligned::Load256( &(rhs.X) ); // mTmp0 = ( v0, v1, v2, v3 )
    __m256d mCol0 = SIMD::Import::Memory::Aligned::Load256( &m00 );     // mCol0 = ( m00, m10, m20, m30 )
    __m256d mCol1 = SIMD::Import::Memory::Aligned::Load256( &m01 );     // mCol1 = ( m01, m11, m21, m31 )
    __m256d mCol2 = SIMD::Import::Memory::Aligned::Load256( &m02 );     // mCol2 = ( m02, m12, m22, m32 )
    __m256d mCol3 = SIMD::Import::Memory::Aligned::Load256( &m03 );     // mCol3 = ( m03, m13, m23, m33 )
    
    // Build RHS vectors
    __m256d mRHS0 = SIMD::Register::Spread4::AAAA( mTmp0 ); // mRHS0 = ( v0, v0, v0, v0 )
    __m256d mRHS1 = SIMD::Register::Spread4::BBBB( mTmp0 ); // mRHS1 = ( v1, v1, v1, v1 )
    __m256d mRHS2 = SIMD::Register::Spread4::CCCC( mTmp0 ); // mRHS2 = ( v2, v2, v2, v2 )
    __m256d mRHS3 = SIMD::Register::Spread4::DDDD( mTmp0 ); // mRHS3 = ( v3, v3, v3, v3 )

    // Perform Product
    mTmp0 = SIMD::Math::Mul( mCol0, mRHS0 );         // mTmp0 = ( m00 * v0, m10 * v0, m20 * v0, m30 * v0 )
    __m256d mTmp1 = SIMD::Math::Mul( mCol1, mRHS1 ); // mTmp1 = ( m01 * v1, m11 * v1, m21 * v1, m31 * v1 )

    mTmp0 = SIMD::Math::Add( mTmp0, mTmp1 ); // mTmp0 = ( m00*v0 + m01*v1, m10*v0 + m11*v1, m20*v0 + m21*v1, m30*v0 + m31*v1 )

    mTmp1 = SIMD::Math::Mul( mCol2, mRHS2 ); // mTmp1 = ( m02 * v2, m12 * v2, m22 * v2, m32 * v2 )

    mTmp0 = SIMD::Math::Add( mTmp0, mTmp1 ); // mTmp0 = ( m00*v0 + m01*v1 + m02*v2, m10*v0 + m11*v1 + m12*v2,
                                             //           m20*v0 + m21*v1 + m22*v2, m30*v0 + m31*v1 + m32*v2 )

    mTmp1 = SIMD::Math::Mul( mCol3, mRHS3 ); // mTmp1 = ( m03 * v3, m13 * v3, m23 * v3, m33 * v3 )

    mTmp0 = SIMD::Math::Add( mTmp0, mTmp1 ); // mTmp0 = ( m00*v0 + m01*v1 + m02*v2 + m03*v3, m10*v0 + m11*v1 + m12*v2 + m13*v3,
                                             //           m20*v0 + m21*v1 + m22*v2 + m23*v3, m30*v0 + m31*v1 + m32*v2 + m33*v3 )

    // Save Data
    SIMD::Export::Memory::Aligned::Save256( &(vRes.X), mTmp0 );

    // Done
    return vRes;
}

template<>
TMatrix4<Float> TMatrix4<Float>::operator+( const TMatrix4<Float> & rhs ) const
{
    TMatrix4<Float> matRes;

    __m256 mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256 mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) );
    mLHS = SIMD::Math::Add( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m02) );
    mLHS = SIMD::Math::Add( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m02), mLHS );

    return matRes;
}
template<>
TMatrix4<Double> TMatrix4<Double>::operator+( const TMatrix4<Double> & rhs ) const
{
    TMatrix4<Double> matRes;

    __m256d mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256d mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) );
    mLHS = SIMD::Math::Add( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m01 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m01) );
    mLHS = SIMD::Math::Add( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m01), mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m02) );
    mLHS = SIMD::Math::Add( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m02), mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m03 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m03) );
    mLHS = SIMD::Math::Add( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m03), mLHS );

    return matRes;
}

template<>
TMatrix4<Float> TMatrix4<Float>::operator-( const TMatrix4<Float> & rhs ) const
{
    TMatrix4<Float> matRes;

    __m256 mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256 mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) );
    mLHS = SIMD::Math::Sub( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m02) );
    mLHS = SIMD::Math::Sub( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m02), mLHS );

    return matRes;
}
template<>
TMatrix4<Double> TMatrix4<Double>::operator-( const TMatrix4<Double> & rhs ) const
{
    TMatrix4<Double> matRes;

    __m256d mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256d mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) );
    mLHS = SIMD::Math::Sub( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m01 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m01) );
    mLHS = SIMD::Math::Sub( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m01), mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m02) );
    mLHS = SIMD::Math::Sub( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m02), mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m03 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m03) );
    mLHS = SIMD::Math::Sub( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m03), mLHS );

    return matRes;
}

template<>
TMatrix4<Float> TMatrix4<Float>::operator*( const TMatrix4<Float> & rhs ) const
{
    // This is one of the most important ones ! Very careful optimisation here !
    TMatrix4<Float> matRes;

    // Load Data
    __m256 mLHSCol01 = SIMD::Import::Memory::Aligned::Load256( &m00 ); // mLHSCol01 = ( l00, l10, l20, l30, l01, l11, l21, l31 )
    __m256 mLHSCol23 = SIMD::Import::Memory::Aligned::Load256( &m02 ); // mLHSCol23 = ( l02, l12, l22, l32, l03, l13, l23, l33 )

    __m256 mRHSCol01 = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) ); // mRHSCol01 = ( r00, r10, r20, r30, r01, r11, r21, r31 )
    __m256 mRHSCol23 = SIMD::Import::Memory::Aligned::Load256( &(rhs.m02) ); // mRHSCol23 = ( r02, r12, r22, r32, r03, r13, r23, r33 )

    __m256 mResCol01, mResCol23;
    __m256 mTmp0, mTmp1;

    // Perform Product
        // LHS Column 0
    mTmp0 = SIMD::Register::Spread2::AA( mLHSCol01 ); // mTmp0 = ( l00, l10, l20, l30, l00, l10, l20, l30 )

    mTmp1 = SIMD::Register::Spread8::AAAAEEEE( mRHSCol01 ); // mTmp1 = ( r00, r00, r00, r00, r01, r01, r01, r01 )
    mResCol01 = SIMD::Math::Mul( mTmp0, mTmp1 );            // mResCol01 = ( LCol0*r00, LCol0*r01 )

    mTmp1 = SIMD::Register::Spread8::AAAAEEEE( mRHSCol23 ); // mTmp1 = ( r02, r02, r02, r02, r03, r03, r03, r03 )
    mResCol23 = SIMD::Math::Mul( mTmp0, mTmp1 );            // mResCol23 = ( LCol0*r02, LCol0*r03 )

        // LHS Column 1
    mTmp0 = SIMD::Register::Spread2::BB( mLHSCol01 ); // mTmp0 = ( l01, l11, l21, l31, l01, l11, l21, l31 )

    mTmp1 = SIMD::Register::Spread8::BBBBFFFF( mRHSCol01 ); // mTmp1 = ( r10, r10, r10, r10, r11, r11, r11, r11 )
    mTmp1 = SIMD::Math::Mul( mTmp1, mTmp0 );                // mTmp1 = ( LCol1*r10, LCol1*r11 )
    mResCol01 = SIMD::Math::Add( mResCol01, mTmp1 );        // mResCol01 = ( LCol0*r00 + LCol1*r10, LCol0*r01 + LCol1*r11 )

    mTmp1 = SIMD::Register::Spread8::BBBBFFFF( mRHSCol23 ); // mTmp1 = ( r12, r12, r12, r12, r13, r13, r13, r13 )
    mTmp1 = SIMD::Math::Mul( mTmp1, mTmp0 );                // mTmp1 = ( LCol1*r12, LCol1*r13 )
    mResCol23 = SIMD::Math::Add( mResCol23, mTmp1 );        // mResCol23 = ( LCol0*r02 + LCol1*r12, LCol0*r03 + LCol1*r13 )

        // LHS Column 2
    mTmp0 = SIMD::Register::Spread2::AA( mLHSCol23 ); // mTmp0 = ( l02, l12, l22, l32, l02, l12, l22, l32 )

    mTmp1 = SIMD::Register::Spread8::CCCCGGGG( mRHSCol01 ); // mTmp1 = ( r20, r20, r20, r20, r21, r21, r21, r21 )
    mTmp1 = SIMD::Math::Mul( mTmp1, mTmp0 );                // mTmp1 = ( LCol2*r20, LCol2*r21 )
    mResCol01 = SIMD::Math::Add( mResCol01, mTmp1 );        // mResCol01 = ( LCol0*r00 + LCol1*r10 + LCol2*r20, LCol0*r01 + LCol1*r11 + LCol2*r21 )

    mTmp1 = SIMD::Register::Spread8::CCCCGGGG( mRHSCol23 ); // mTmp1 = ( r22, r22, r22, r22, r23, r23, r23, r23 )
    mTmp1 = SIMD::Math::Mul( mTmp1, mTmp0 );                // mTmp1 = ( LCol2*r22, LCol2*r23 )
    mResCol23 = SIMD::Math::Add( mResCol23, mTmp1 );        // mResCol23 = ( LCol0*r02 + LCol1*r12 + LCol2*r22, LCol0*r03 + LCol1*r13 + LCol2*r23 )

        // LHS Column 3
    mTmp0 = SIMD::Register::Spread2::BB( mLHSCol23 ); // mTmp0 = ( l03, l13, l23, l33, l03, l13, l23, l33 )

    mTmp1 = SIMD::Register::Spread8::DDDDHHHH( mRHSCol01 ); // mTmp1 = ( r30, r30, r30, r30, r31, r31, r31, r31 )
    mTmp1 = SIMD::Math::Mul( mTmp1, mTmp0 );                // mTmp1 = ( LCol3*r30, LCol3*r31 )
    mResCol01 = SIMD::Math::Add( mResCol01, mTmp1 );        // mResCol01 = ( LCol0*r00 + LCol1*r10 + LCol2*r20 + LCol3*r30, LCol0*r01 + LCol1*r11 + LCol2*r21 + LCol3*r31 )

    mTmp1 = SIMD::Register::Spread8::DDDDHHHH( mRHSCol23 ); // mTmp1 = ( r32, r32, r32, r32, r33, r33, r33, r33 )
    mTmp1 = SIMD::Math::Mul( mTmp1, mTmp0 );                // mTmp1 = ( LCol3*r32, LCol3*r33 )
    mResCol23 = SIMD::Math::Add( mResCol23, mTmp1 );        // mResCol23 = ( LCol0*r02 + LCol1*r12 + LCol2*r22 + LCol3*r32, LCol0*r03 + LCol1*r13 + LCol2*r23 + LCol3*r33 )

    // Save Data
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mResCol01 );
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m02), mResCol23 );

    // Done
    return matRes;
}
template<>
TMatrix4<Double> TMatrix4<Double>::operator*( const TMatrix4<Double> & rhs ) const
{
    // This is one of the most important ones ! Very careful optimisation here !
    TMatrix4<Double> matRes;

    // Load Data
    __m256d mLHSCol0 = SIMD::Import::Memory::Aligned::Load256( &m00 ); // mLHSCol0 = ( l00, l10, l20, l30 )
    __m256d mLHSCol1 = SIMD::Import::Memory::Aligned::Load256( &m01 ); // mLHSCol1 = ( l01, l11, l21, l31 )
    __m256d mLHSCol2 = SIMD::Import::Memory::Aligned::Load256( &m02 ); // mLHSCol2 = ( l02, l12, l22, l32 )
    __m256d mLHSCol3 = SIMD::Import::Memory::Aligned::Load256( &m03 ); // mLHSCol3 = ( l03, l13, l23, l33 )

    __m256d mRHSCol, mResCol;
    __m256d mTmp;

    // Perform Product
        // RHS Column 0
    mRHSCol = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) ); // mRHSCol = ( r00, r10, r20, r30 )

    mTmp = SIMD::Register::Spread4::AAAA( mRHSCol ); // mTmp = ( r00, r00, r00, r00 )
    mResCol = SIMD::Math::Mul( mTmp, mLHSCol0 );     // mResCol = ( LCol0*r00 )

    mTmp = SIMD::Register::Spread4::BBBB( mRHSCol ); // mTmp = ( r10, r10, r10, r10 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol1 );        // mTmp = ( LCol1*r10 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r00 + LCol1*r10 )

    mTmp = SIMD::Register::Spread4::CCCC( mRHSCol ); // mTmp = ( r20, r20, r20, r20 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol2 );        // mTmp = ( LCol2*r20 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r00 + LCol1*r10 + LCol2*r20 )

    mTmp = SIMD::Register::Spread4::DDDD( mRHSCol ); // mTmp = ( r30, r30, r30, r30 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol3 );        // mTmp = ( LCol3*r30 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r00 + LCol1*r10 + LCol2*r20 + LCol3*r30 )

    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mResCol );

        // RHS Column 1
    mRHSCol = SIMD::Import::Memory::Aligned::Load256( &(rhs.m01) ); // mRHSCol = ( r01, r11, r21, r31 )

    mTmp = SIMD::Register::Spread4::AAAA( mRHSCol ); // mTmp = ( r01, r01, r01, r01 )
    mResCol = SIMD::Math::Mul( mTmp, mLHSCol0 );     // mResCol = ( LCol0*r01 )

    mTmp = SIMD::Register::Spread4::BBBB( mRHSCol ); // mTmp = ( r11, r11, r11, r11 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol1 );        // mTmp = ( LCol1*r11 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r01 + LCol1*r11 )

    mTmp = SIMD::Register::Spread4::CCCC( mRHSCol ); // mTmp = ( r21, r21, r21, r21 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol2 );        // mTmp = ( LCol2*r21 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r01 + LCol1*r11 + LCol2*r21 )

    mTmp = SIMD::Register::Spread4::DDDD( mRHSCol ); // mTmp = ( r31, r31, r31, r31 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol3 );        // mTmp = ( LCol3*r31 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r01 + LCol1*r11 + LCol2*r21 + LCol3*r31 )

    SIMD::Export::Memory::Aligned::Save256( &(matRes.m01), mResCol );

        // RHS Column 2
    mRHSCol = SIMD::Import::Memory::Aligned::Load256( &(rhs.m02) ); // mRHSCol = ( r02, r12, r22, r32 )

    mTmp = SIMD::Register::Spread4::AAAA( mRHSCol ); // mTmp = ( r02, r02, r02, r02 )
    mResCol = SIMD::Math::Mul( mTmp, mLHSCol0 );     // mResCol = ( LCol0*r02 )

    mTmp = SIMD::Register::Spread4::BBBB( mRHSCol ); // mTmp = ( r12, r12, r12, r12 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol1 );        // mTmp = ( LCol1*r12 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r02 + LCol1*r12 )

    mTmp = SIMD::Register::Spread4::CCCC( mRHSCol ); // mTmp = ( r22, r22, r22, r22 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol2 );        // mTmp = ( LCol2*r22 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r02 + LCol1*r12 + LCol2*r22 )

    mTmp = SIMD::Register::Spread4::DDDD( mRHSCol ); // mTmp = ( r32, r32, r32, r32 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol3 );        // mTmp = ( LCol3*r32 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r02 + LCol1*r12 + LCol2*r22 + LCol3*r32 )

    SIMD::Export::Memory::Aligned::Save256( &(matRes.m02), mResCol );

        // RHS Column 3
    mRHSCol = SIMD::Import::Memory::Aligned::Load256( &(rhs.m03) ); // mRHSCol = ( r03, r13, r23, r33 )

    mTmp = SIMD::Register::Spread4::AAAA( mRHSCol ); // mTmp = ( r03, r03, r03, r03 )
    mResCol = SIMD::Math::Mul( mTmp, mLHSCol0 );     // mResCol = ( LCol0*r03 )

    mTmp = SIMD::Register::Spread4::BBBB( mRHSCol ); // mTmp = ( r13, r13, r13, r13 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol1 );        // mTmp = ( LCol1*r13 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r03 + LCol1*r13 )

    mTmp = SIMD::Register::Spread4::CCCC( mRHSCol ); // mTmp = ( r23, r23, r23, r23 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol2 );        // mTmp = ( LCol2*r23 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r03 + LCol1*r13 + LCol2*r23 )

    mTmp = SIMD::Register::Spread4::DDDD( mRHSCol ); // mTmp = ( r33, r33, r33, r33 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol3 );        // mTmp = ( LCol3*r33 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r03 + LCol1*r13 + LCol2*r23 + LCol3*r33 )

    SIMD::Export::Memory::Aligned::Save256( &(matRes.m03), mResCol );

    // Done
    return matRes;
}

template<>
TMatrix4<Float> & TMatrix4<Float>::operator+=( const TMatrix4<Float> & rhs )
{
    __m256 mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256 mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) );
    mLHS = SIMD::Math::Add( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m00, mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m02) );
    mLHS = SIMD::Math::Add( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m02, mLHS );

    return (*this);
}
template<>
TMatrix4<Double> & TMatrix4<Double>::operator+=( const TMatrix4<Double> & rhs )
{
    __m256d mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256d mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) );
    mLHS = SIMD::Math::Add( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m00, mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m01 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m01) );
    mLHS = SIMD::Math::Add( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m01, mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m02) );
    mLHS = SIMD::Math::Add( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m02, mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m03 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m03) );
    mLHS = SIMD::Math::Add( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m03, mLHS );

    return (*this);
}

template<>
TMatrix4<Float> & TMatrix4<Float>::operator-=( const TMatrix4<Float> & rhs )
{
    __m256 mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256 mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) );
    mLHS = SIMD::Math::Sub( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m00, mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m02) );
    mLHS = SIMD::Math::Sub( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m02, mLHS );

    return (*this);
}
template<>
TMatrix4<Double> & TMatrix4<Double>::operator-=( const TMatrix4<Double> & rhs )
{
    __m256d mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256d mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) );
    mLHS = SIMD::Math::Sub( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m00, mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m01 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m01) );
    mLHS = SIMD::Math::Sub( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m01, mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m02 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m02) );
    mLHS = SIMD::Math::Sub( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m02, mLHS );

    mLHS = SIMD::Import::Memory::Aligned::Load256( &m03 );
    mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m03) );
    mLHS = SIMD::Math::Sub( mLHS, mRHS );
    SIMD::Export::Memory::Aligned::Save256( &m03, mLHS );

    return (*this);
}

template<>
TMatrix4<Float> & TMatrix4<Float>::operator*=( const TMatrix4<Float> & rhs )
{
    // This is one of the most important ones ! Very careful optimisation here !
    
    // Load Data
    __m256 mLHSCol01 = SIMD::Import::Memory::Aligned::Load256( &m00 ); // mLHSCol01 = ( l00, l10, l20, l30, l01, l11, l21, l31 )
    __m256 mLHSCol23 = SIMD::Import::Memory::Aligned::Load256( &m02 ); // mLHSCol23 = ( l02, l12, l22, l32, l03, l13, l23, l33 )

    __m256 mRHSCol01 = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) ); // mRHSCol01 = ( r00, r10, r20, r30, r01, r11, r21, r31 )
    __m256 mRHSCol23 = SIMD::Import::Memory::Aligned::Load256( &(rhs.m02) ); // mRHSCol23 = ( r02, r12, r22, r32, r03, r13, r23, r33 )

    __m256 mResCol01, mResCol23;
    __m256 mTmp0, mTmp1;

    // Perform Product
        // LHS Column 0
    mTmp0 = SIMD::Register::Spread2::AA( mLHSCol01 ); // mTmp0 = ( l00, l10, l20, l30, l00, l10, l20, l30 )

    mTmp1 = SIMD::Register::Spread8::AAAAEEEE( mRHSCol01 ); // mTmp1 = ( r00, r00, r00, r00, r01, r01, r01, r01 )
    mResCol01 = SIMD::Math::Mul( mTmp0, mTmp1 );            // mResCol01 = ( LCol0*r00, LCol0*r01 )

    mTmp1 = SIMD::Register::Spread8::AAAAEEEE( mRHSCol23 ); // mTmp1 = ( r02, r02, r02, r02, r03, r03, r03, r03 )
    mResCol23 = SIMD::Math::Mul( mTmp0, mTmp1 );            // mResCol23 = ( LCol0*r02, LCol0*r03 )

        // LHS Column 1
    mTmp0 = SIMD::Register::Spread2::BB( mLHSCol01 ); // mTmp0 = ( l01, l11, l21, l31, l01, l11, l21, l31 )

    mTmp1 = SIMD::Register::Spread8::BBBBFFFF( mRHSCol01 ); // mTmp1 = ( r10, r10, r10, r10, r11, r11, r11, r11 )
    mTmp1 = SIMD::Math::Mul( mTmp1, mTmp0 );                // mTmp1 = ( LCol1*r10, LCol1*r11 )
    mResCol01 = SIMD::Math::Add( mResCol01, mTmp1 );        // mResCol01 = ( LCol0*r00 + LCol1*r10, LCol0*r01 + LCol1*r11 )

    mTmp1 = SIMD::Register::Spread8::BBBBFFFF( mRHSCol23 ); // mTmp1 = ( r12, r12, r12, r12, r13, r13, r13, r13 )
    mTmp1 = SIMD::Math::Mul( mTmp1, mTmp0 );                // mTmp1 = ( LCol1*r12, LCol1*r13 )
    mResCol23 = SIMD::Math::Add( mResCol23, mTmp1 );        // mResCol23 = ( LCol0*r02 + LCol1*r12, LCol0*r03 + LCol1*r13 )

        // LHS Column 2
    mTmp0 = SIMD::Register::Spread2::AA( mLHSCol23 ); // mTmp0 = ( l02, l12, l22, l32, l02, l12, l22, l32 )

    mTmp1 = SIMD::Register::Spread8::CCCCGGGG( mRHSCol01 ); // mTmp1 = ( r20, r20, r20, r20, r21, r21, r21, r21 )
    mTmp1 = SIMD::Math::Mul( mTmp1, mTmp0 );                // mTmp1 = ( LCol2*r20, LCol2*r21 )
    mResCol01 = SIMD::Math::Add( mResCol01, mTmp1 );        // mResCol01 = ( LCol0*r00 + LCol1*r10 + LCol2*r20, LCol0*r01 + LCol1*r11 + LCol2*r21 )

    mTmp1 = SIMD::Register::Spread8::CCCCGGGG( mRHSCol23 ); // mTmp1 = ( r22, r22, r22, r22, r23, r23, r23, r23 )
    mTmp1 = SIMD::Math::Mul( mTmp1, mTmp0 );                // mTmp1 = ( LCol2*r22, LCol2*r23 )
    mResCol23 = SIMD::Math::Add( mResCol23, mTmp1 );        // mResCol23 = ( LCol0*r02 + LCol1*r12 + LCol2*r22, LCol0*r03 + LCol1*r13 + LCol2*r23 )

        // LHS Column 3
    mTmp0 = SIMD::Register::Spread2::BB( mLHSCol23 ); // mTmp0 = ( l03, l13, l23, l33, l03, l13, l23, l33 )

    mTmp1 = SIMD::Register::Spread8::DDDDHHHH( mRHSCol01 ); // mTmp1 = ( r30, r30, r30, r30, r31, r31, r31, r31 )
    mTmp1 = SIMD::Math::Mul( mTmp1, mTmp0 );                // mTmp1 = ( LCol3*r30, LCol3*r31 )
    mResCol01 = SIMD::Math::Add( mResCol01, mTmp1 );        // mResCol01 = ( LCol0*r00 + LCol1*r10 + LCol2*r20 + LCol3*r30, LCol0*r01 + LCol1*r11 + LCol2*r21 + LCol3*r31 )

    mTmp1 = SIMD::Register::Spread8::DDDDHHHH( mRHSCol23 ); // mTmp1 = ( r32, r32, r32, r32, r33, r33, r33, r33 )
    mTmp1 = SIMD::Math::Mul( mTmp1, mTmp0 );                // mTmp1 = ( LCol3*r32, LCol3*r33 )
    mResCol23 = SIMD::Math::Add( mResCol23, mTmp1 );        // mResCol23 = ( LCol0*r02 + LCol1*r12 + LCol2*r22 + LCol3*r32, LCol0*r03 + LCol1*r13 + LCol2*r23 + LCol3*r33 )

    // Save Data
    SIMD::Export::Memory::Aligned::Save256( &m00, mResCol01 );
    SIMD::Export::Memory::Aligned::Save256( &m02, mResCol23 );

    return (*this);
}
template<>
TMatrix4<Double> & TMatrix4<Double>::operator*=( const TMatrix4<Double> & rhs )
{
    // This is one of the most important ones ! Very careful optimisation here !
    
    // Load Data
    __m256d mLHSCol0 = SIMD::Import::Memory::Aligned::Load256( &m00 ); // mLHSCol0 = ( l00, l10, l20, l30 )
    __m256d mLHSCol1 = SIMD::Import::Memory::Aligned::Load256( &m01 ); // mLHSCol1 = ( l01, l11, l21, l31 )
    __m256d mLHSCol2 = SIMD::Import::Memory::Aligned::Load256( &m02 ); // mLHSCol2 = ( l02, l12, l22, l32 )
    __m256d mLHSCol3 = SIMD::Import::Memory::Aligned::Load256( &m03 ); // mLHSCol3 = ( l03, l13, l23, l33 )

    __m256d mRHSCol, mResCol;
    __m256d mTmp;

    // Perform Product
        // RHS Column 0
    mRHSCol = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) ); // mRHSCol = ( r00, r10, r20, r30 )

    mTmp = SIMD::Register::Spread4::AAAA( mRHSCol ); // mTmp = ( r00, r00, r00, r00 )
    mResCol = SIMD::Math::Mul( mTmp, mLHSCol0 );     // mResCol = ( LCol0*r00 )

    mTmp = SIMD::Register::Spread4::BBBB( mRHSCol ); // mTmp = ( r10, r10, r10, r10 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol1 );        // mTmp = ( LCol1*r10 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r00 + LCol1*r10 )

    mTmp = SIMD::Register::Spread4::CCCC( mRHSCol ); // mTmp = ( r20, r20, r20, r20 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol2 );        // mTmp = ( LCol2*r20 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r00 + LCol1*r10 + LCol2*r20 )

    mTmp = SIMD::Register::Spread4::DDDD( mRHSCol ); // mTmp = ( r30, r30, r30, r30 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol3 );        // mTmp = ( LCol3*r30 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r00 + LCol1*r10 + LCol2*r20 + LCol3*r30 )

    SIMD::Export::Memory::Aligned::Save256( &m00, mResCol );

        // RHS Column 1
    mRHSCol = SIMD::Import::Memory::Aligned::Load256( &(rhs.m01) ); // mRHSCol = ( r01, r11, r21, r31 )

    mTmp = SIMD::Register::Spread4::AAAA( mRHSCol ); // mTmp = ( r01, r01, r01, r01 )
    mResCol = SIMD::Math::Mul( mTmp, mLHSCol0 );     // mResCol = ( LCol0*r01 )

    mTmp = SIMD::Register::Spread4::BBBB( mRHSCol ); // mTmp = ( r11, r11, r11, r11 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol1 );        // mTmp = ( LCol1*r11 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r01 + LCol1*r11 )

    mTmp = SIMD::Register::Spread4::CCCC( mRHSCol ); // mTmp = ( r21, r21, r21, r21 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol2 );        // mTmp = ( LCol2*r21 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r01 + LCol1*r11 + LCol2*r21 )

    mTmp = SIMD::Register::Spread4::DDDD( mRHSCol ); // mTmp = ( r31, r31, r31, r31 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol3 );        // mTmp = ( LCol3*r31 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r01 + LCol1*r11 + LCol2*r21 + LCol3*r31 )

    SIMD::Export::Memory::Aligned::Save256( &m01, mResCol );

        // RHS Column 2
    mRHSCol = SIMD::Import::Memory::Aligned::Load256( &(rhs.m02) ); // mRHSCol = ( r02, r12, r22, r32 )

    mTmp = SIMD::Register::Spread4::AAAA( mRHSCol ); // mTmp = ( r02, r02, r02, r02 )
    mResCol = SIMD::Math::Mul( mTmp, mLHSCol0 );     // mResCol = ( LCol0*r02 )

    mTmp = SIMD::Register::Spread4::BBBB( mRHSCol ); // mTmp = ( r12, r12, r12, r12 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol1 );        // mTmp = ( LCol1*r12 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r02 + LCol1*r12 )

    mTmp = SIMD::Register::Spread4::CCCC( mRHSCol ); // mTmp = ( r22, r22, r22, r22 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol2 );        // mTmp = ( LCol2*r22 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r02 + LCol1*r12 + LCol2*r22 )

    mTmp = SIMD::Register::Spread4::DDDD( mRHSCol ); // mTmp = ( r32, r32, r32, r32 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol3 );        // mTmp = ( LCol3*r32 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r02 + LCol1*r12 + LCol2*r22 + LCol3*r32 )

    SIMD::Export::Memory::Aligned::Save256( &m02, mResCol );

        // RHS Column 3
    mRHSCol = SIMD::Import::Memory::Aligned::Load256( &(rhs.m03) ); // mRHSCol = ( r03, r13, r23, r33 )

    mTmp = SIMD::Register::Spread4::AAAA( mRHSCol ); // mTmp = ( r03, r03, r03, r03 )
    mResCol = SIMD::Math::Mul( mTmp, mLHSCol0 );     // mResCol = ( LCol0*r03 )

    mTmp = SIMD::Register::Spread4::BBBB( mRHSCol ); // mTmp = ( r13, r13, r13, r13 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol1 );        // mTmp = ( LCol1*r13 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r03 + LCol1*r13 )

    mTmp = SIMD::Register::Spread4::CCCC( mRHSCol ); // mTmp = ( r23, r23, r23, r23 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol2 );        // mTmp = ( LCol2*r23 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r03 + LCol1*r13 + LCol2*r23 )

    mTmp = SIMD::Register::Spread4::DDDD( mRHSCol ); // mTmp = ( r33, r33, r33, r33 )
    mTmp = SIMD::Math::Mul( mTmp, mLHSCol3 );        // mTmp = ( LCol3*r33 )
    mResCol = SIMD::Math::Add( mResCol, mTmp );      // mResCol = ( LCol0*r03 + LCol1*r13 + LCol2*r23 + LCol3*r33 )

    SIMD::Export::Memory::Aligned::Save256( &m03, mResCol );

    return (*this);
}

template<>
Float TMatrix4<Float>::Determinant() const
{
    alignas(16) static Float arrTmp[4];

    // Load Data
    __m256 mTmp0 = SIMD::Import::Memory::Aligned::Load256( &m00 ); // mTmp0 = ( m00, m10, m20, m30, m01, m11, m21, m31 )
    __m256 mTmp1 = SIMD::Import::Memory::Aligned::Load256( &m02 ); // mTmp1 = ( m02, m12, m22, m32, m03, m13, m23, m33 )

    __m256 mCol02 = SIMD::Register::Shuffle2::AC( mTmp0, mTmp1 ); // mCol02 = ( m00, m10, m20, m30, m02, m12, m22, m32 )
    __m256 mCol13 = SIMD::Register::Shuffle2::BD( mTmp0, mTmp1 ); // mCol13 = ( m01, m11, m21, m31, m03, m13, m23, m33 )

    __m256 mCombine, mPart1, mPart2;

    // Part 1
    mTmp0 = SIMD::Register::Spread8::AAAAEEEE( mCol02 );  // mTmp0 = ( m00, m00, m00, X, m02, m02, m02, X )
    mTmp1 = SIMD::Register::Shuffle8::BCDAFGHE( mCol13 ); // mTmp1 = ( m11, m21, m31, X, m13, m23, m33, X )
    mPart1 = SIMD::Math::Mul( mTmp0, mTmp1 );             // mPart1 = ( m00*m11, m00*m21, m00*m31, X, m02*m13, m02*m23, m02*m33, X )

    mTmp0 = SIMD::Register::Spread8::AAAAEEEE( mCol13 );  // mTmp0 = ( m01, m01, m01, X, m03, m03, m03, X )
    mTmp1 = SIMD::Register::Shuffle8::BCDAFGHE( mCol02 ); // mTmp1 = ( m10, m20, m30, X, m12, m22, m32, X )
    mCombine = SIMD::Math::Mul( mTmp0, mTmp1 );           // mCombine = ( m01*m10, m01*m20, m01*m30, X, m03*m12, m03*m22, m03*m32, X )

    mPart1 = SIMD::Math::Sub( mPart1, mCombine ); // mPart1 = ( A1, B1, C1, X, D1, E1, F1, X )

    // Part 2
    mTmp0 = SIMD::Register::Spread8::CBBBGFFF( mCol02 ); // mTmp0 = ( m20, m10, m10, X, m22, m12, m12, X )
    mTmp1 = SIMD::Register::Spread8::DDCCHHGG( mCol13 ); // mTmp1 = ( m31, m31, m21, X, m33, m33, m23, X )
    mPart2 = SIMD::Math::Mul( mTmp0, mTmp1 );            // mPart2 = ( m20*m31, m10*m31, m10*m21, X, m22*m33, m12*m33, m12*m23, X )

    mTmp0 = SIMD::Register::Spread8::CBBBGFFF( mCol13 ); // mTmp0 = ( m21, m11, m11, X, m23, m13, m13, X )
    mTmp1 = SIMD::Register::Spread8::DDCCHHGG( mCol02 ); // mTmp1 = ( m30, m30, m20, X, m32, m32, m22, X )
    mCombine = SIMD::Math::Mul( mTmp0, mTmp1 );          // mCombine = ( m21*m30, m11*m30, m11*m20, X, m23*m32, m13*m32, m13*m22, X )

    mPart2 = SIMD::Math::Sub( mPart2, mCombine ); // mPart2 = ( C2, B2, A2, X, F2, E2, D2, X )

    // Combine Parts
    mPart2 = SIMD::Register::Shuffle2::BA( mPart2 ); // mPart2 = ( F2, E2, D2, X, C2, B2, A2, X )
    mCombine = SIMD::Math::Mul( mPart1, mPart2 );    // mCombine = ( A1*F2, B1*E2, C1*D2, X, D1*C2, E1*B2, F1*A2, X )

    __m128 mResult = SIMD::Register::Move::FourFloatH( mCombine );      // mResult = ( D1*C2, E1*B2, F1*A2, X )
    mResult = SIMD::Math::Add( mResult, SIMD::Cast::Down( mCombine ) ); // mResult = ( A1*F2 + D1*C2, B1*E2 + E1*B2, C1*D2 + F1*A2, X )

    // Save Data
    SIMD::Export::Memory::Aligned::Save128( arrTmp, mResult );

    // Done
    return ( arrTmp[0] - arrTmp[1] + arrTmp[2] );

}
template<>
Double TMatrix4<Double>::Determinant() const
{
    alignas(32) static Double arrTmp[4];
    
    // Load Data
    __m256d mCol0 = SIMD::Import::Memory::Aligned::Load256( &m00 ); // mCol0 = ( m00, m10, m20, m30 )
    __m256d mCol1 = SIMD::Import::Memory::Aligned::Load256( &m01 ); // mCol1 = ( m01, m11, m21, m31 )
    __m256d mCol2 = SIMD::Import::Memory::Aligned::Load256( &m02 ); // mCol2 = ( m02, m12, m22, m32 )
    __m256d mCol3 = SIMD::Import::Memory::Aligned::Load256( &m03 ); // mCol3 = ( m03, m13, m23, m33 )

    __m256d mCombine, mPart1, mPart2, mPart3;
    __m256d mTmp0, mTmp1;

    // Part 1
    mTmp0 = SIMD::Register::Spread4::AAAC( mCol0 ); // mTmp0 = ( m00, m00, m00, m20 )
    mTmp1 = SIMD::Register::Spread4::BCDD( mCol1 ); // mTmp1 = ( m11, m21, m31, m31 )
    mPart1 = SIMD::Math::Mul( mTmp0, mTmp1 );       // mPart1 = ( m00*m11, m00*m21, m00*m31, m20*m31 )

    mTmp0 = SIMD::Register::Spread4::AAAC( mCol1 ); // mTmp0 = ( m01, m01, m01, m21 )
    mTmp1 = SIMD::Register::Spread4::BCDD( mCol0 ); // mTmp1 = ( m10, m20, m30, m30 )
    mCombine = SIMD::Math::Mul( mTmp0, mTmp1 );     // mCombine = ( m01*m10, m01*m20, m01*m30, m21*m30 )

    mPart1 = SIMD::Math::Sub( mPart1, mCombine ); // mPart1 = ( A1, B1, C1, C2 )

    // Part 2
    mTmp0 = SIMD::Register::Spread4::AAAC( mCol2 ); // mTmp0 = ( m02, m02, m02, m22 )
    mTmp1 = SIMD::Register::Spread4::BCDD( mCol3 ); // mTmp1 = ( m13, m23, m33, m33 )
    mPart2 = SIMD::Math::Mul( mTmp0, mTmp1 );       // mPart2 = ( m02*m13, m02*m23, m02*m33, m22*m33 )

    mTmp0 = SIMD::Register::Spread4::AAAC( mCol3 ); // mTmp0 = ( m03, m03, m03, m23 )
    mTmp1 = SIMD::Register::Spread4::BCDD( mCol2 ); // mTmp1 = ( m12, m22, m32, m32 )
    mCombine = SIMD::Math::Mul( mTmp0, mTmp1 );     // mCombine = ( m03*m12, m03*m22, m03*m32, m23*m32 )

    mPart2 = SIMD::Math::Sub( mPart2, mCombine ); // mPart2 = ( D1, E1, F1, F2 )

    // Part 3
    mTmp0 = SIMD::Register::Shuffle2::AC( mCol0, mCol2 ); // mTmp0 = ( m00, m10, m02, m12 )
    mTmp0 = SIMD::Register::Spread4::BBDD( mTmp0 );       // mTmp0 = ( m10, m10, m12, m12 )
    mTmp1 = SIMD::Register::Shuffle2::BD( mCol1, mCol3 ); // mTmp1 = ( m21, m31, m23, m33 )
    mPart3 = SIMD::Math::Mul( mTmp0, mTmp1 );             // mPart3 = ( m10*m21, m10*m31, m12*m23, m12*m33 )

    mTmp0 = SIMD::Register::Shuffle2::AC( mCol1, mCol3 ); // mTmp0 = ( m01, m11, m03, m13 )
    mTmp0 = SIMD::Register::Spread4::BBDD( mTmp0 );       // mTmp0 = ( m11, m11, m13, m13 )
    mTmp1 = SIMD::Register::Shuffle2::BD( mCol0, mCol2 ); // mTmp1 = ( m20, m30, m22, m32 )
    mCombine = SIMD::Math::Mul( mTmp0, mTmp1 );           // mCombine = ( m11*m20, m11*m30, m13*m22, m13*m32 )

    mPart3 = SIMD::Math::Sub( mPart3, mCombine ); // mPart3 = ( A2, B2, D2, E2 )

    // Combine Parts
    mPart3 = SIMD::Register::Shuffle4::ACBD( mPart3 ); // mPart3 = ( A2, D2, B2, E2 )

    mTmp0 = SIMD::Register::Shuffle4::AECH( mPart3, mPart1 ); // mTmp0 = ( A2, X, B2, C2 )
    mTmp1 = SIMD::Register::Shuffle4::BFDH( mPart3, mPart2 ); // mTmp1 = ( D2, X, E2, F2 )

    mTmp0 = SIMD::Register::Shuffle4::DCAB( mTmp0 ); // mTmp0 = ( C2, B2, A2, X )
    mTmp1 = SIMD::Register::Shuffle4::DCAB( mTmp1 ); // mTmp1 = ( F2, E2, D2, X )

    mPart1 = SIMD::Math::Mul( mPart1, mTmp1 ); // mPart1 = ( A1*F2, B1*E2, C1*D2, X )
    mPart2 = SIMD::Math::Mul( mPart2, mTmp0 ); // mPart2 = ( D1*C2, E1*B2, F1*A2, X )

    mCombine = SIMD::Math::Add( mPart1, mPart2 ); // mCombine = ( A1*F2 + D1*C2, B1*E2 + E1*B2, C1*D2 + F1*A2, X )

    // Save Data
    SIMD::Export::Memory::Aligned::Save256( arrTmp, mCombine );

    // Done
    return ( arrTmp[0] - arrTmp[1] + arrTmp[2] );
}

//template<>
//Void TMatrix4<Float>::Adjoint( TMatrix4<Float> & outAdjointMatrix ) const
//{
//    
//}
//template<>
//Void TMatrix4<Double>::Adjoint( TMatrix4<Double> & outAdjointMatrix ) const
//{
//    
//}
//
//template<>
//Bool TMatrix4<Float>::Invert( TMatrix4<Float> & outInvMatrix, Float fZeroTolerance ) const
//{
//    
//
//    return true;
//}
//template<>
//Bool TMatrix4<Double>::Invert( TMatrix4<Double> & outInvMatrix, Double fZeroTolerance ) const
//{
//    
//
//    return true;
//}

#endif // SIMD_ENABLE


