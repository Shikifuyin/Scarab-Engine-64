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

    __m256 mRHS = SIMD::Spread256( rhs );

    __m256 mTwoRows = SIMD::Load256( &m00 );
    mTwoRows = SIMD::Mul( mTwoRows, mRHS );
    SIMD::Store256( &(matRes.m00), mTwoRows );

    mTwoRows = SIMD::Load256( &m20 );
    mTwoRows = SIMD::Mul( mTwoRows, mRHS );
    SIMD::Store256( &(matRes.m20), mTwoRows );

    return matRes;
}
template<>
TMatrix4<Double> TMatrix4<Double>::operator*( const Double & rhs ) const
{
    TMatrix4<Double> matRes;

    __m256d mRHS = SIMD::Spread256( rhs );
    
    __m256d mRow = SIMD::Load256( &m00 );
    mRow = SIMD::Mul( mRow, mRHS );
    SIMD::Store256( &(matRes.m00), mRow );

    mRow = SIMD::Load256( &m10 );
    mRow = SIMD::Mul( mRow, mRHS );
    SIMD::Store256( &(matRes.m10), mRow );

    mRow = SIMD::Load256( &m20 );
    mRow = SIMD::Mul( mRow, mRHS );
    SIMD::Store256( &(matRes.m20), mRow );

    mRow = SIMD::Load256( &m30 );
    mRow = SIMD::Mul( mRow, mRHS );
    SIMD::Store256( &(matRes.m30), mRow );

    return matRes;
}

template<>
TMatrix4<Float> TMatrix4<Float>::operator/( const Float & rhs ) const
{
    TMatrix4<Float> matRes;

    __m256 mRHS = SIMD::Spread256( rhs );

    __m256 mTwoRows = SIMD::Load256( &m00 );
    mTwoRows = SIMD::Div( mTwoRows, mRHS );
    SIMD::Store256( &(matRes.m00), mTwoRows );

    mTwoRows = SIMD::Load256( &m20 );
    mTwoRows = SIMD::Div( mTwoRows, mRHS );
    SIMD::Store256( &(matRes.m20), mTwoRows );

    return matRes;
}
template<>
TMatrix4<Double> TMatrix4<Double>::operator/( const Double & rhs ) const
{
    TMatrix4<Double> matRes;

    __m256d mRHS = SIMD::Spread256( rhs );
    
    __m256d mRow = SIMD::Load256( &m00 );
    mRow = SIMD::Div( mRow, mRHS );
    SIMD::Store256( &(matRes.m00), mRow );

    mRow = SIMD::Load256( &m10 );
    mRow = SIMD::Div( mRow, mRHS );
    SIMD::Store256( &(matRes.m10), mRow );

    mRow = SIMD::Load256( &m20 );
    mRow = SIMD::Div( mRow, mRHS );
    SIMD::Store256( &(matRes.m20), mRow );

    mRow = SIMD::Load256( &m30 );
    mRow = SIMD::Div( mRow, mRHS );
    SIMD::Store256( &(matRes.m30), mRow );

    return matRes;
}

template<>
TMatrix4<Float> & TMatrix4<Float>::operator*=( const Float & rhs )
{
    __m256 mRHS = SIMD::Spread256( rhs );

    __m256 mTwoRows = SIMD::Load256( &m00 );
    mTwoRows = SIMD::Mul( mTwoRows, mRHS );
    SIMD::Store256( &m00, mTwoRows );

    mTwoRows = SIMD::Load256( &m20 );
    mTwoRows = SIMD::Mul( mTwoRows, mRHS );
    SIMD::Store256( &m20, mTwoRows );

    return (*this);
}
template<>
TMatrix4<Double> & TMatrix4<Double>::operator*=( const Double & rhs )
{
    __m256d mRHS = SIMD::Spread256( rhs );
    
    __m256d mRow = SIMD::Load256( &m00 );
    mRow = SIMD::Mul( mRow, mRHS );
    SIMD::Store256( &m00, mRow );

    mRow = SIMD::Load256( &m10 );
    mRow = SIMD::Mul( mRow, mRHS );
    SIMD::Store256( &m10, mRow );

    mRow = SIMD::Load256( &m20 );
    mRow = SIMD::Mul( mRow, mRHS );
    SIMD::Store256( &m20, mRow );

    mRow = SIMD::Load256( &m30 );
    mRow = SIMD::Mul( mRow, mRHS );
    SIMD::Store256( &m30, mRow );

    return (*this);
}

template<>
TMatrix4<Float> & TMatrix4<Float>::operator/=( const Float & rhs )
{
    __m256 mRHS = SIMD::Spread256( rhs );

    __m256 mTwoRows = SIMD::Load256( &m00 );
    mTwoRows = SIMD::Div( mTwoRows, mRHS );
    SIMD::Store256( &m00, mTwoRows );

    mTwoRows = SIMD::Load256( &m20 );
    mTwoRows = SIMD::Div( mTwoRows, mRHS );
    SIMD::Store256( &m20, mTwoRows );

    return (*this);
}
template<>
TMatrix4<Double> & TMatrix4<Double>::operator/=( const Double & rhs )
{
    __m256d mRHS = SIMD::Spread256( rhs );
    
    __m256d mRow = SIMD::Load256( &m00 );
    mRow = SIMD::Div( mRow, mRHS );
    SIMD::Store256( &m00, mRow );

    mRow = SIMD::Load256( &m10 );
    mRow = SIMD::Div( mRow, mRHS );
    SIMD::Store256( &m10, mRow );

    mRow = SIMD::Load256( &m20 );
    mRow = SIMD::Div( mRow, mRHS );
    SIMD::Store256( &m20, mRow );

    mRow = SIMD::Load256( &m30 );
    mRow = SIMD::Div( mRow, mRHS );
    SIMD::Store256( &m30, mRow );

    return (*this);
}

template<>
TVertex4<Float> TMatrix4<Float>::operator*( const TVertex4<Float> & rhs ) const
{
    TVertex4<Float> vRes;

    __m128 mTmp = SIMD::Load128( &(rhs.X) );
    __m256 mRHS = SIMD::Zero256F();
    mRHS = SIMD::MoveFourFloatL( mRHS, mTmp );
    mRHS = SIMD::MoveFourFloatH( mRHS, mTmp );

    __m256 mTwoDots = SIMD::Load256( &m00 );
    mTwoDots = SIMD::Dot4( mTwoDots, mRHS );

    mTmp = SIMD::MoveFourFloatL( mTwoDots );
    SIMD::StoreLower( &(vRes.X), mTmp );

    mTmp = SIMD::MoveFourFloatH( mTwoDots );
    SIMD::StoreLower( &(vRes.Y), mTmp );

    mTwoDots = SIMD::Load256( &m20 );
    mTwoDots = SIMD::Dot4( mTwoDots, mRHS );

    mTmp = SIMD::MoveFourFloatL( mTwoDots );
    SIMD::StoreLower( &(vRes.Z), mTmp );

    mTmp = SIMD::MoveFourFloatH( mTwoDots );
    SIMD::StoreLower( &(vRes.W), mTmp );

    return vRes;
}
template<>
TVertex4<Double> TMatrix4<Double>::operator*( const TVertex4<Double> & rhs ) const
{
    TVertex4<Double> vRes;

    __m256d mRHS = SIMD::Load256( &(rhs.X) );
    
    __m256d mDot = SIMD::Load256( &m00 );
    mDot = SIMD::Mul( mDot, mRHS );
    mDot = SIMD::HAdd( mDot, mDot );
    mDot = SIMD_256_Shuffle256Double( mDot, SIMD_SHUFFLE_MASK_4x4(0,2,1,3) );
    mDot = SIMD::HAdd( mDot, mDot );
    SIMD::StoreLower( &(vRes.X), SIMD::CastDown(mDot) );

    mDot = SIMD::Load256( &m10 );
    mDot = SIMD::Mul( mDot, mRHS );
    mDot = SIMD::HAdd( mDot, mDot );
    mDot = SIMD_256_Shuffle256Double( mDot, SIMD_SHUFFLE_MASK_4x4(0,2,1,3) );
    mDot = SIMD::HAdd( mDot, mDot );
    SIMD::StoreLower( &(vRes.Y), SIMD::CastDown(mDot) );

    mDot = SIMD::Load256( &m20 );
    mDot = SIMD::Mul( mDot, mRHS );
    mDot = SIMD::HAdd( mDot, mDot );
    mDot = SIMD_256_Shuffle256Double( mDot, SIMD_SHUFFLE_MASK_4x4(0,2,1,3) );
    mDot = SIMD::HAdd( mDot, mDot );
    SIMD::StoreLower( &(vRes.Z), SIMD::CastDown(mDot) );

    mDot = SIMD::Load256( &m30 );
    mDot = SIMD::Mul( mDot, mRHS );
    mDot = SIMD::HAdd( mDot, mDot );
    mDot = SIMD_256_Shuffle256Double( mDot, SIMD_SHUFFLE_MASK_4x4(0,2,1,3) );
    mDot = SIMD::HAdd( mDot, mDot );
    SIMD::StoreLower( &(vRes.W), SIMD::CastDown(mDot) );

    return vRes;
}

template<>
TVector4<Float> TMatrix4<Float>::operator*( const TVector4<Float> & rhs ) const
{
    TVector4<Float> vRes;

    __m128 mTmp = SIMD::Load128( &(rhs.X) );
    __m256 mRHS = SIMD::Zero256F();
    mRHS = SIMD::MoveFourFloatL( mRHS, mTmp );
    mRHS = SIMD::MoveFourFloatH( mRHS, mTmp );

    __m256 mTwoDots = SIMD::Load256( &m00 );
    mTwoDots = SIMD::Dot4( mTwoDots, mRHS );

    mTmp = SIMD::MoveFourFloatL( mTwoDots );
    SIMD::StoreLower( &(vRes.X), mTmp );

    mTmp = SIMD::MoveFourFloatH( mTwoDots );
    SIMD::StoreLower( &(vRes.Y), mTmp );

    mTwoDots = SIMD::Load256( &m20 );
    mTwoDots = SIMD::Dot4( mTwoDots, mRHS );

    mTmp = SIMD::MoveFourFloatL( mTwoDots );
    SIMD::StoreLower( &(vRes.Z), mTmp );

    mTmp = SIMD::MoveFourFloatH( mTwoDots );
    SIMD::StoreLower( &(vRes.W), mTmp );

    return vRes;
}
template<>
TVector4<Double> TMatrix4<Double>::operator*( const TVector4<Double> & rhs ) const
{
    TVector4<Double> vRes;

    __m256d mRHS = SIMD::Load256( &(rhs.X) );
    
    __m256d mDot = SIMD::Load256( &m00 );
    mDot = SIMD::Mul( mDot, mRHS );
    mDot = SIMD::HAdd( mDot, mDot );
    mDot = SIMD_256_Shuffle256Double( mDot, SIMD_SHUFFLE_MASK_4x4(0,2,1,3) );
    mDot = SIMD::HAdd( mDot, mDot );
    SIMD::StoreLower( &(vRes.X), SIMD::CastDown(mDot) );

    mDot = SIMD::Load256( &m10 );
    mDot = SIMD::Mul( mDot, mRHS );
    mDot = SIMD::HAdd( mDot, mDot );
    mDot = SIMD_256_Shuffle256Double( mDot, SIMD_SHUFFLE_MASK_4x4(0,2,1,3) );
    mDot = SIMD::HAdd( mDot, mDot );
    SIMD::StoreLower( &(vRes.Y), SIMD::CastDown(mDot) );

    mDot = SIMD::Load256( &m20 );
    mDot = SIMD::Mul( mDot, mRHS );
    mDot = SIMD::HAdd( mDot, mDot );
    mDot = SIMD_256_Shuffle256Double( mDot, SIMD_SHUFFLE_MASK_4x4(0,2,1,3) );
    mDot = SIMD::HAdd( mDot, mDot );
    SIMD::StoreLower( &(vRes.Z), SIMD::CastDown(mDot) );

    mDot = SIMD::Load256( &m30 );
    mDot = SIMD::Mul( mDot, mRHS );
    mDot = SIMD::HAdd( mDot, mDot );
    mDot = SIMD_256_Shuffle256Double( mDot, SIMD_SHUFFLE_MASK_4x4(0,2,1,3) );
    mDot = SIMD::HAdd( mDot, mDot );
    SIMD::StoreLower( &(vRes.W), SIMD::CastDown(mDot) );

    return vRes;
}

template<>
TMatrix4<Float> TMatrix4<Float>::operator+( const TMatrix4<Float> & rhs ) const
{
    TMatrix4<Float> matRes;

    __m256 mLHS = SIMD::Load256( &m00 );
    __m256 mRHS = SIMD::Load256( &(rhs.m00) );
    mLHS = SIMD::Add( mLHS, mRHS );
    SIMD::Store256( &(matRes.m00), mLHS );

    mLHS = SIMD::Load256( &m20 );
    mRHS = SIMD::Load256( &(rhs.m20) );
    mLHS = SIMD::Add( mLHS, mRHS );
    SIMD::Store256( &(matRes.m20), mLHS );

    return matRes;
}
template<>
TMatrix4<Double> TMatrix4<Double>::operator+( const TMatrix4<Double> & rhs ) const
{
    TMatrix4<Double> matRes;

    __m256d mLHS = SIMD::Load256( &m00 );
    __m256d mRHS = SIMD::Load256( &(rhs.m00) );
    mLHS = SIMD::Add( mLHS, mRHS );
    SIMD::Store256( &(matRes.m00), mLHS );

    mLHS = SIMD::Load256( &m10 );
    mRHS = SIMD::Load256( &(rhs.m10) );
    mLHS = SIMD::Add( mLHS, mRHS );
    SIMD::Store256( &(matRes.m10), mLHS );

    mLHS = SIMD::Load256( &m20 );
    mRHS = SIMD::Load256( &(rhs.m20) );
    mLHS = SIMD::Add( mLHS, mRHS );
    SIMD::Store256( &(matRes.m20), mLHS );

    mLHS = SIMD::Load256( &m30 );
    mRHS = SIMD::Load256( &(rhs.m30) );
    mLHS = SIMD::Add( mLHS, mRHS );
    SIMD::Store256( &(matRes.m30), mLHS );

    return matRes;
}

template<>
TMatrix4<Float> TMatrix4<Float>::operator-( const TMatrix4<Float> & rhs ) const
{
    TMatrix4<Float> matRes;

    __m256 mLHS = SIMD::Load256( &m00 );
    __m256 mRHS = SIMD::Load256( &(rhs.m00) );
    mLHS = SIMD::Sub( mLHS, mRHS );
    SIMD::Store256( &(matRes.m00), mLHS );

    mLHS = SIMD::Load256( &m20 );
    mRHS = SIMD::Load256( &(rhs.m20) );
    mLHS = SIMD::Sub( mLHS, mRHS );
    SIMD::Store256( &(matRes.m20), mLHS );

    return matRes;
}
template<>
TMatrix4<Double> TMatrix4<Double>::operator-( const TMatrix4<Double> & rhs ) const
{
    TMatrix4<Double> matRes;

    __m256d mLHS = SIMD::Load256( &m00 );
    __m256d mRHS = SIMD::Load256( &(rhs.m00) );
    mLHS = SIMD::Sub( mLHS, mRHS );
    SIMD::Store256( &(matRes.m00), mLHS );

    mLHS = SIMD::Load256( &m10 );
    mRHS = SIMD::Load256( &(rhs.m10) );
    mLHS = SIMD::Sub( mLHS, mRHS );
    SIMD::Store256( &(matRes.m10), mLHS );

    mLHS = SIMD::Load256( &m20 );
    mRHS = SIMD::Load256( &(rhs.m20) );
    mLHS = SIMD::Sub( mLHS, mRHS );
    SIMD::Store256( &(matRes.m20), mLHS );

    mLHS = SIMD::Load256( &m30 );
    mRHS = SIMD::Load256( &(rhs.m30) );
    mLHS = SIMD::Sub( mLHS, mRHS );
    SIMD::Store256( &(matRes.m30), mLHS );

    return matRes;
}

template<>
TMatrix4<Float> TMatrix4<Float>::operator*( const TMatrix4<Float> & rhs ) const
{
    // That's the funny one xD
    TMatrix4<Float> matRes;

    __m256 mRHS = SIMD::Load256( &(rhs.m00) );

    __m256 mTwoDots = SIMD::Load256( &m00 );
    mTwoDots = SIMD::Dot4( mTwoDots, mRHS );

    mTmp = SIMD::MoveFourFloatL( mTwoDots );
    SIMD::StoreLower( &(vRes.X), mTmp );

    mTmp = SIMD::MoveFourFloatH( mTwoDots );
    SIMD::StoreLower( &(vRes.Y), mTmp );

    mTwoDots = SIMD::Load256( &m20 );
    mTwoDots = SIMD::Dot4( mTwoDots, mRHS );

    mTmp = SIMD::MoveFourFloatL( mTwoDots );
    SIMD::StoreLower( &(vRes.Z), mTmp );

    mTmp = SIMD::MoveFourFloatH( mTwoDots );
    SIMD::StoreLower( &(vRes.W), mTmp );

    // Row 0
    MathSSEFn->Push( m00, m00, m00, m00 );
    MathSSEFn->Push( rhs.m00, rhs.m01, rhs.m02, rhs.m03 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m01, m01, m01, m01 );
    MathSSEFn->Push( rhs.m10, rhs.m11, rhs.m12, rhs.m13 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m02, m02, m02, m02 );
    MathSSEFn->Push( rhs.m20, rhs.m21, rhs.m22, rhs.m23 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m03, m03, m03, m03 );
    MathSSEFn->Push( rhs.m30, rhs.m31, rhs.m32, rhs.m33 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Pop( matRes.m00, matRes.m01, matRes.m02, matRes.m03 );

    // Row 1
    MathSSEFn->Push( m10, m10, m10, m10 );
    MathSSEFn->Push( rhs.m00, rhs.m01, rhs.m02, rhs.m03 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m11, m11, m11, m11 );
    MathSSEFn->Push( rhs.m10, rhs.m11, rhs.m12, rhs.m13 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m12, m12, m12, m12 );
    MathSSEFn->Push( rhs.m20, rhs.m21, rhs.m22, rhs.m23 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m13, m13, m13, m13 );
    MathSSEFn->Push( rhs.m30, rhs.m31, rhs.m32, rhs.m33 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Pop( matRes.m10, matRes.m11, matRes.m12, matRes.m13 );

    // Row 2
    MathSSEFn->Push( m20, m20, m20, m20 );
    MathSSEFn->Push( rhs.m00, rhs.m01, rhs.m02, rhs.m03 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m21, m21, m21, m21 );
    MathSSEFn->Push( rhs.m10, rhs.m11, rhs.m12, rhs.m13 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m22, m22, m22, m22 );
    MathSSEFn->Push( rhs.m20, rhs.m21, rhs.m22, rhs.m23 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m23, m23, m23, m23 );
    MathSSEFn->Push( rhs.m30, rhs.m31, rhs.m32, rhs.m33 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Pop( matRes.m20, matRes.m21, matRes.m22, matRes.m23 );

    // Row 3
    MathSSEFn->Push( m30, m30, m30, m30 );
    MathSSEFn->Push( rhs.m00, rhs.m01, rhs.m02, rhs.m03 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m31, m31, m31, m31 );
    MathSSEFn->Push( rhs.m10, rhs.m11, rhs.m12, rhs.m13 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m32, m32, m32, m32 );
    MathSSEFn->Push( rhs.m20, rhs.m21, rhs.m22, rhs.m23 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m33, m33, m33, m33 );
    MathSSEFn->Push( rhs.m30, rhs.m31, rhs.m32, rhs.m33 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Pop( matRes.m30, matRes.m31, matRes.m32, matRes.m33 );

    return matRes;
}
template<>
TMatrix4<Double> TMatrix4<Double>::operator*( const TMatrix4<Double> & rhs ) const
{
    // That's the even more funny one xD
    TMatrix4<Double> matRes;

    // Row 0
    MathSSEFn->Push( m00, m00 );
    MathSSEFn->Push( rhs.m00, rhs.m01 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m01, m01 );
    MathSSEFn->Push( rhs.m10, rhs.m11 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m02, m02 );
    MathSSEFn->Push( rhs.m20, rhs.m21 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m03, m03 );
    MathSSEFn->Push( rhs.m30, rhs.m31 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( matRes.m00, matRes.m01 );

    MathSSEFn->Push( m00, m00 );
    MathSSEFn->Push( rhs.m02, rhs.m03 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m01, m01 );
    MathSSEFn->Push( rhs.m12, rhs.m13 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m02, m02 );
    MathSSEFn->Push( rhs.m22, rhs.m23 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m03, m03 );
    MathSSEFn->Push( rhs.m32, rhs.m33 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( matRes.m02, matRes.m03 );

    // Row 1
    MathSSEFn->Push( m10, m10 );
    MathSSEFn->Push( rhs.m00, rhs.m01 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m11, m11 );
    MathSSEFn->Push( rhs.m10, rhs.m11 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m12, m12 );
    MathSSEFn->Push( rhs.m20, rhs.m21 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m13, m13 );
    MathSSEFn->Push( rhs.m30, rhs.m31 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( matRes.m10, matRes.m11 );

    MathSSEFn->Push( m10, m10 );
    MathSSEFn->Push( rhs.m02, rhs.m03 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m11, m11 );
    MathSSEFn->Push( rhs.m12, rhs.m13 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m12, m12 );
    MathSSEFn->Push( rhs.m22, rhs.m23 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m13, m13 );
    MathSSEFn->Push( rhs.m32, rhs.m33 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( matRes.m12, matRes.m13 );

    // Row 2
    MathSSEFn->Push( m20, m20 );
    MathSSEFn->Push( rhs.m00, rhs.m01 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m21, m21 );
    MathSSEFn->Push( rhs.m10, rhs.m11 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m22, m22 );
    MathSSEFn->Push( rhs.m20, rhs.m21 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m23, m23 );
    MathSSEFn->Push( rhs.m30, rhs.m31 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( matRes.m20, matRes.m21 );

    MathSSEFn->Push( m20, m20 );
    MathSSEFn->Push( rhs.m02, rhs.m03 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m21, m21 );
    MathSSEFn->Push( rhs.m12, rhs.m13 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m22, m22 );
    MathSSEFn->Push( rhs.m22, rhs.m23 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m23, m23 );
    MathSSEFn->Push( rhs.m32, rhs.m33 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( matRes.m22, matRes.m23 );

    // Row 3
    MathSSEFn->Push( m30, m30 );
    MathSSEFn->Push( rhs.m00, rhs.m01 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m31, m31 );
    MathSSEFn->Push( rhs.m10, rhs.m11 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m32, m32 );
    MathSSEFn->Push( rhs.m20, rhs.m21 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m33, m33 );
    MathSSEFn->Push( rhs.m30, rhs.m31 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( matRes.m30, matRes.m31 );

    MathSSEFn->Push( m30, m30 );
    MathSSEFn->Push( rhs.m02, rhs.m03 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m31, m31 );
    MathSSEFn->Push( rhs.m12, rhs.m13 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m32, m32 );
    MathSSEFn->Push( rhs.m22, rhs.m23 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m33, m33 );
    MathSSEFn->Push( rhs.m32, rhs.m33 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( matRes.m32, matRes.m33 );

    return matRes;
}

template<>
TMatrix4<Float> & TMatrix4<Float>::operator+=( const TMatrix4<Float> & rhs )
{
    MathSSEFn->Push( m00, m01, m02, m03 );
    MathSSEFn->Push( rhs.m00, rhs.m01, rhs.m02, rhs.m03 );
    MathSSEFn->AddPF();
    MathSSEFn->Pop( m00, m01, m02, m03 );

    MathSSEFn->Push( m10, m11, m12, m13 );
    MathSSEFn->Push( rhs.m10, rhs.m11, rhs.m12, rhs.m13 );
    MathSSEFn->AddPF();
    MathSSEFn->Pop( m10, m11, m12, m13 );

    MathSSEFn->Push( m20, m21, m22, m23 );
    MathSSEFn->Push( rhs.m20, rhs.m21, rhs.m22, rhs.m23 );
    MathSSEFn->AddPF();
    MathSSEFn->Pop( m20, m21, m22, m23 );

    MathSSEFn->Push( m30, m31, m32, m33 );
    MathSSEFn->Push( rhs.m30, rhs.m31, rhs.m32, rhs.m33 );
    MathSSEFn->AddPF();
    MathSSEFn->Pop( m30, m31, m32, m33 );

    return (*this);
}
template<>
TMatrix4<Double> & TMatrix4<Double>::operator+=( const TMatrix4<Double> & rhs )
{
    MathSSEFn->Push( m00, m01 );
    MathSSEFn->Push( rhs.m00, rhs.m01 );
    MathSSEFn->AddPD();
    MathSSEFn->Pop( m00, m01 );
    MathSSEFn->Push( m02, m03 );
    MathSSEFn->Push( rhs.m02, rhs.m03 );
    MathSSEFn->AddPD();
    MathSSEFn->Pop( m02, m03 );

    MathSSEFn->Push( m10, m11 );
    MathSSEFn->Push( rhs.m10, rhs.m11 );
    MathSSEFn->AddPD();
    MathSSEFn->Pop( m10, m11 );
    MathSSEFn->Push( m12, m13 );
    MathSSEFn->Push( rhs.m12, rhs.m13 );
    MathSSEFn->AddPD();
    MathSSEFn->Pop( m12, m13 );

    MathSSEFn->Push( m20, m21 );
    MathSSEFn->Push( rhs.m20, rhs.m21 );
    MathSSEFn->AddPD();
    MathSSEFn->Pop( m20, m21 );
    MathSSEFn->Push( m22, m23 );
    MathSSEFn->Push( rhs.m22, rhs.m23 );
    MathSSEFn->AddPD();
    MathSSEFn->Pop( m22, m23 );

    MathSSEFn->Push( m30, m31 );
    MathSSEFn->Push( rhs.m30, rhs.m31 );
    MathSSEFn->AddPD();
    MathSSEFn->Pop( m30, m31 );
    MathSSEFn->Push( m32, m33 );
    MathSSEFn->Push( rhs.m32, rhs.m33 );
    MathSSEFn->AddPD();
    MathSSEFn->Pop( m32, m33 );

    return (*this);
}

template<>
TMatrix4<Float> & TMatrix4<Float>::operator-=( const TMatrix4<Float> & rhs )
{
    MathSSEFn->Push( m00, m01, m02, m03 );
    MathSSEFn->Push( rhs.m00, rhs.m01, rhs.m02, rhs.m03 );
    MathSSEFn->SubPF();
    MathSSEFn->Pop( m00, m01, m02, m03 );

    MathSSEFn->Push( m10, m11, m12, m13 );
    MathSSEFn->Push( rhs.m10, rhs.m11, rhs.m12, rhs.m13 );
    MathSSEFn->SubPF();
    MathSSEFn->Pop( m10, m11, m12, m13 );

    MathSSEFn->Push( m20, m21, m22, m23 );
    MathSSEFn->Push( rhs.m20, rhs.m21, rhs.m22, rhs.m23 );
    MathSSEFn->SubPF();
    MathSSEFn->Pop( m20, m21, m22, m23 );

    MathSSEFn->Push( m30, m31, m32, m33 );
    MathSSEFn->Push( rhs.m30, rhs.m31, rhs.m32, rhs.m33 );
    MathSSEFn->SubPF();
    MathSSEFn->Pop( m30, m31, m32, m33 );

    return (*this);
}
template<>
TMatrix4<Double> & TMatrix4<Double>::operator-=( const TMatrix4<Double> & rhs )
{
    MathSSEFn->Push( m00, m01 );
    MathSSEFn->Push( rhs.m00, rhs.m01 );
    MathSSEFn->SubPD();
    MathSSEFn->Pop( m00, m01 );
    MathSSEFn->Push( m02, m03 );
    MathSSEFn->Push( rhs.m02, rhs.m03 );
    MathSSEFn->SubPD();
    MathSSEFn->Pop( m02, m03 );

    MathSSEFn->Push( m10, m11 );
    MathSSEFn->Push( rhs.m10, rhs.m11 );
    MathSSEFn->SubPD();
    MathSSEFn->Pop( m10, m11 );
    MathSSEFn->Push( m12, m13 );
    MathSSEFn->Push( rhs.m12, rhs.m13 );
    MathSSEFn->SubPD();
    MathSSEFn->Pop( m12, m13 );

    MathSSEFn->Push( m20, m21 );
    MathSSEFn->Push( rhs.m20, rhs.m21 );
    MathSSEFn->SubPD();
    MathSSEFn->Pop( m20, m21 );
    MathSSEFn->Push( m22, m23 );
    MathSSEFn->Push( rhs.m22, rhs.m23 );
    MathSSEFn->SubPD();
    MathSSEFn->Pop( m22, m23 );

    MathSSEFn->Push( m30, m31 );
    MathSSEFn->Push( rhs.m30, rhs.m31 );
    MathSSEFn->SubPD();
    MathSSEFn->Pop( m30, m31 );
    MathSSEFn->Push( m32, m33 );
    MathSSEFn->Push( rhs.m32, rhs.m33 );
    MathSSEFn->SubPD();
    MathSSEFn->Pop( m32, m33 );

    return (*this);
}

template<>
TMatrix4<Float> & TMatrix4<Float>::operator*=( const TMatrix4<Float> & rhs )
{
    // That's the funny one xD

    // Row 0
    MathSSEFn->Push( m00, m00, m00, m00 );
    MathSSEFn->Push( rhs.m00, rhs.m01, rhs.m02, rhs.m03 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m01, m01, m01, m01 );
    MathSSEFn->Push( rhs.m10, rhs.m11, rhs.m12, rhs.m13 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m02, m02, m02, m02 );
    MathSSEFn->Push( rhs.m20, rhs.m21, rhs.m22, rhs.m23 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m03, m03, m03, m03 );
    MathSSEFn->Push( rhs.m30, rhs.m31, rhs.m32, rhs.m33 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Pop( m00, m01, m02, m03 );

    // Row 1
    MathSSEFn->Push( m10, m10, m10, m10 );
    MathSSEFn->Push( rhs.m00, rhs.m01, rhs.m02, rhs.m03 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m11, m11, m11, m11 );
    MathSSEFn->Push( rhs.m10, rhs.m11, rhs.m12, rhs.m13 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m12, m12, m12, m12 );
    MathSSEFn->Push( rhs.m20, rhs.m21, rhs.m22, rhs.m23 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m13, m13, m13, m13 );
    MathSSEFn->Push( rhs.m30, rhs.m31, rhs.m32, rhs.m33 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Pop( m10, m11, m12, m13 );

    // Row 2
    MathSSEFn->Push( m20, m20, m20, m20 );
    MathSSEFn->Push( rhs.m00, rhs.m01, rhs.m02, rhs.m03 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m21, m21, m21, m21 );
    MathSSEFn->Push( rhs.m10, rhs.m11, rhs.m12, rhs.m13 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m22, m22, m22, m22 );
    MathSSEFn->Push( rhs.m20, rhs.m21, rhs.m22, rhs.m23 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m23, m23, m23, m23 );
    MathSSEFn->Push( rhs.m30, rhs.m31, rhs.m32, rhs.m33 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Pop( m20, m21, m22, m23 );

    // Row 3
    MathSSEFn->Push( m30, m30, m30, m30 );
    MathSSEFn->Push( rhs.m00, rhs.m01, rhs.m02, rhs.m03 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m31, m31, m31, m31 );
    MathSSEFn->Push( rhs.m10, rhs.m11, rhs.m12, rhs.m13 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m32, m32, m32, m32 );
    MathSSEFn->Push( rhs.m20, rhs.m21, rhs.m22, rhs.m23 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( m33, m33, m33, m33 );
    MathSSEFn->Push( rhs.m30, rhs.m31, rhs.m32, rhs.m33 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Pop( m30, m31, m32, m33 );

    return (*this);
}
template<>
TMatrix4<Double> & TMatrix4<Double>::operator*=( const TMatrix4<Double> & rhs )
{
    // That's the even more funny one xD

    // Row 0
    MathSSEFn->Push( m00, m00 );
    MathSSEFn->Push( rhs.m00, rhs.m01 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m01, m01 );
    MathSSEFn->Push( rhs.m10, rhs.m11 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m02, m02 );
    MathSSEFn->Push( rhs.m20, rhs.m21 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m03, m03 );
    MathSSEFn->Push( rhs.m30, rhs.m31 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();

    MathSSEFn->Push( m00, m00 );
    MathSSEFn->Push( rhs.m02, rhs.m03 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m01, m01 );
    MathSSEFn->Push( rhs.m12, rhs.m13 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m02, m02 );
    MathSSEFn->Push( rhs.m22, rhs.m23 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m03, m03 );
    MathSSEFn->Push( rhs.m32, rhs.m33 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();

    MathSSEFn->Pop( m02, m03 );
    MathSSEFn->Pop( m00, m01 );

    // Row 1
    MathSSEFn->Push( m10, m10 );
    MathSSEFn->Push( rhs.m00, rhs.m01 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m11, m11 );
    MathSSEFn->Push( rhs.m10, rhs.m11 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m12, m12 );
    MathSSEFn->Push( rhs.m20, rhs.m21 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m13, m13 );
    MathSSEFn->Push( rhs.m30, rhs.m31 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();

    MathSSEFn->Push( m10, m10 );
    MathSSEFn->Push( rhs.m02, rhs.m03 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m11, m11 );
    MathSSEFn->Push( rhs.m12, rhs.m13 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m12, m12 );
    MathSSEFn->Push( rhs.m22, rhs.m23 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m13, m13 );
    MathSSEFn->Push( rhs.m32, rhs.m33 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();

    MathSSEFn->Pop( m12, m13 );
    MathSSEFn->Pop( m10, m11 );

    // Row 2
    MathSSEFn->Push( m20, m20 );
    MathSSEFn->Push( rhs.m00, rhs.m01 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m21, m21 );
    MathSSEFn->Push( rhs.m10, rhs.m11 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m22, m22 );
    MathSSEFn->Push( rhs.m20, rhs.m21 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m23, m23 );
    MathSSEFn->Push( rhs.m30, rhs.m31 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();

    MathSSEFn->Push( m20, m20 );
    MathSSEFn->Push( rhs.m02, rhs.m03 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m21, m21 );
    MathSSEFn->Push( rhs.m12, rhs.m13 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m22, m22 );
    MathSSEFn->Push( rhs.m22, rhs.m23 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m23, m23 );
    MathSSEFn->Push( rhs.m32, rhs.m33 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();

    MathSSEFn->Pop( m22, m23 );
    MathSSEFn->Pop( m20, m21 );

    // Row 3
    MathSSEFn->Push( m30, m30 );
    MathSSEFn->Push( rhs.m00, rhs.m01 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m31, m31 );
    MathSSEFn->Push( rhs.m10, rhs.m11 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m32, m32 );
    MathSSEFn->Push( rhs.m20, rhs.m21 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m33, m33 );
    MathSSEFn->Push( rhs.m30, rhs.m31 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();

    MathSSEFn->Push( m30, m30 );
    MathSSEFn->Push( rhs.m02, rhs.m03 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m31, m31 );
    MathSSEFn->Push( rhs.m12, rhs.m13 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m32, m32 );
    MathSSEFn->Push( rhs.m22, rhs.m23 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( m33, m33 );
    MathSSEFn->Push( rhs.m32, rhs.m33 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();

    MathSSEFn->Pop( m32, m33 );
    MathSSEFn->Pop( m30, m31 );

    return (*this);
}

template<>
Float TMatrix4<Float>::Determinant() const
{
    Float fA0, fA1, fA2, fA3, fA4, fA5;
    Float fB0, fB1, fB2, fB3, fB4, fB5, fDet;

    MathSSEFn->Push( m00, m00, m00, m01 );
    MathSSEFn->Push( m11, m12, m13, m12 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m01, m02, m03, m02 );
    MathSSEFn->Push( m10, m10, m10, m11 );
    MathSSEFn->MulPF();
    MathSSEFn->SubPF();
    MathSSEFn->Pop( fA0, fA1, fA2, fA3 );

    MathSSEFn->Push( m01, m02, m20, m20 );
    MathSSEFn->Push( m13, m13, m31, m32 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m03, m03, m21, m22 );
    MathSSEFn->Push( m11, m12, m30, m30 );
    MathSSEFn->MulPF();
    MathSSEFn->SubPF();
    MathSSEFn->Pop( fA4, fA5, fB0, fB1 );

    MathSSEFn->Push( m20, m21, m21, m22 );
    MathSSEFn->Push( m33, m32, m33, m33 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m23, m22, m23, m23 );
    MathSSEFn->Push( m30, m31, m31, m32 );
    MathSSEFn->MulPF();
    MathSSEFn->SubPF();
    MathSSEFn->Pop( fB2, fB3, fB4, fB5 );

    MathSSEFn->Push( fA0, -fA1, fA2, 0.0f );
    MathSSEFn->Push( fB5, fB4, fB3, 0.0f );
    MathSSEFn->MulPF();
    MathSSEFn->Push( fA3, -fA4, fA5, 0.0f );
    MathSSEFn->Push( fB2, fB1, fB0, 0.0f );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->HAddF(0);
    MathSSEFn->HAddF(0);
    MathSSEFn->Pop(fDet);

    return fDet;
}
template<>
Double TMatrix4<Double>::Determinant() const
{
    Double fA0, fA1, fA2, fA3, fA4, fA5;
    Double fB0, fB1, fB2, fB3, fB4, fB5, fDet;

    MathSSEFn->Push( m00, m00 );
    MathSSEFn->Push( m11, m12 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m01, m02 );
    MathSSEFn->Push( m10, m10 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fA0, fA1 );

    MathSSEFn->Push( m00, m01 );
    MathSSEFn->Push( m13, m12 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m03, m02 );
    MathSSEFn->Push( m10, m11 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fA2, fA3 );

    MathSSEFn->Push( m01, m02 );
    MathSSEFn->Push( m13, m13 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m03, m03 );
    MathSSEFn->Push( m11, m12 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fA4, fA5 );

    MathSSEFn->Push( m20, m20 );
    MathSSEFn->Push( m31, m32 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m21, m22 );
    MathSSEFn->Push( m30, m30 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fB0, fB1 );

    MathSSEFn->Push( m20, m21 );
    MathSSEFn->Push( m33, m32 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m23, m22 );
    MathSSEFn->Push( m30, m31 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fB2, fB3 );

    MathSSEFn->Push( m21, m22 );
    MathSSEFn->Push( m33, m33 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m23, m23 );
    MathSSEFn->Push( m31, m32 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fB4, fB5 );

    MathSSEFn->Push( fA0, -fA1 );
    MathSSEFn->Push( fB5, fB4 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( fA2, fA3 );
    MathSSEFn->Push( fB3, fB2 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( -fA4, fA5 );
    MathSSEFn->Push( fB1, fB0 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->HAddD(0);
    MathSSEFn->Pop(fDet);

    return fDet;
}

template<>
Void TMatrix4<Float>::Adjoint( TMatrix4<Float> & outAdjointMatrix ) const
{
    Float fA0, fA1, fA2, fA3, fA4, fA5;
    Float fB0, fB1, fB2, fB3, fB4, fB5;

    MathSSEFn->Push( m00, m00, m00, m01 );
    MathSSEFn->Push( m11, m12, m13, m12 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m01, m02, m03, m02 );
    MathSSEFn->Push( m10, m10, m10, m11 );
    MathSSEFn->MulPF();
    MathSSEFn->SubPF();
    MathSSEFn->Pop( fA0, fA1, fA2, fA3 );

    MathSSEFn->Push( m01, m02, m20, m20 );
    MathSSEFn->Push( m13, m13, m31, m32 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m03, m03, m21, m22 );
    MathSSEFn->Push( m11, m12, m30, m30 );
    MathSSEFn->MulPF();
    MathSSEFn->SubPF();
    MathSSEFn->Pop( fA4, fA5, fB0, fB1 );

    MathSSEFn->Push( m20, m21, m21, m22 );
    MathSSEFn->Push( m33, m32, m33, m33 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m23, m22, m23, m23 );
    MathSSEFn->Push( m30, m31, m31, m32 );
    MathSSEFn->MulPF();
    MathSSEFn->SubPF();
    MathSSEFn->Pop( fB2, fB3, fB4, fB5 );

    MathSSEFn->Push( +m11, -m01, +m31, -m21 );
    MathSSEFn->Push( fB5, fB5, fA5, fA5 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( -m12, +m02, -m32, +m22 );
    MathSSEFn->Push( fB4, fB4, fA4, fA4 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( +m13, -m03, +m33, -m23 );
    MathSSEFn->Push( fB3, fB3, fA3, fA3 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Pop( outAdjointMatrix.m00, outAdjointMatrix.m01, outAdjointMatrix.m02, outAdjointMatrix.m03 );

    MathSSEFn->Push( -m10, +m00, -m30, +m20 );
    MathSSEFn->Push( fB5, fB5, fA5, fA5 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( +m12, -m02, +m32, -m22 );
    MathSSEFn->Push( fB2, fB2, fA2, fA2 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( -m13, +m03, -m33, +m23 );
    MathSSEFn->Push( fB1, fB1, fA1, fA1 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Pop( outAdjointMatrix.m10, outAdjointMatrix.m11, outAdjointMatrix.m12, outAdjointMatrix.m13 );

    MathSSEFn->Push( +m10, -m00, +m30, -m20 );
    MathSSEFn->Push( fB4, fB4, fA4, fA4 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( -m11, +m01, -m31, +m21 );
    MathSSEFn->Push( fB2, fB2, fA2, fA2 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( +m13, -m03, +m33, -m23 );
    MathSSEFn->Push( fB0, fB0, fA0, fA0 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Pop( outAdjointMatrix.m20, outAdjointMatrix.m21, outAdjointMatrix.m22, outAdjointMatrix.m23 );

    MathSSEFn->Push( -m10, +m00, -m30, +m20 );
    MathSSEFn->Push( fB3, fB3, fA3, fA3 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( +m11, -m01, +m31, -m21 );
    MathSSEFn->Push( fB1, fB1, fA1, fA1 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( -m12, +m02, -m32, +m22 );
    MathSSEFn->Push( fB0, fB0, fA0, fA0 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Pop( outAdjointMatrix.m30, outAdjointMatrix.m31, outAdjointMatrix.m32, outAdjointMatrix.m33 );
}
template<>
Void TMatrix4<Double>::Adjoint( TMatrix4<Double> & outAdjointMatrix ) const
{
    Double fA0, fA1, fA2, fA3, fA4, fA5;
    Double fB0, fB1, fB2, fB3, fB4, fB5;

    MathSSEFn->Push( m00, m00 );
    MathSSEFn->Push( m11, m12 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m01, m02 );
    MathSSEFn->Push( m10, m10 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fA0, fA1 );

    MathSSEFn->Push( m00, m01 );
    MathSSEFn->Push( m13, m12 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m03, m02 );
    MathSSEFn->Push( m10, m11 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fA2, fA3 );

    MathSSEFn->Push( m01, m02 );
    MathSSEFn->Push( m13, m13 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m03, m03 );
    MathSSEFn->Push( m11, m12 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fA4, fA5 );

    MathSSEFn->Push( m20, m20 );
    MathSSEFn->Push( m31, m32 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m21, m22 );
    MathSSEFn->Push( m30, m30 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fB0, fB1 );

    MathSSEFn->Push( m20, m21 );
    MathSSEFn->Push( m33, m32 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m23, m22 );
    MathSSEFn->Push( m30, m31 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fB2, fB3 );

    MathSSEFn->Push( m21, m22 );
    MathSSEFn->Push( m33, m33 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m23, m23 );
    MathSSEFn->Push( m31, m32 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fB4, fB5 );

    MathSSEFn->Push( +m11, -m01 );
    MathSSEFn->Push( fB5, fB5 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( -m12, +m02 );
    MathSSEFn->Push( fB4, fB4 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( +m13, -m03 );
    MathSSEFn->Push( fB3, fB3 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( outAdjointMatrix.m00, outAdjointMatrix.m01 );

    MathSSEFn->Push( +m31, -m21 );
    MathSSEFn->Push( fA5, fA5 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( -m32, +m22 );
    MathSSEFn->Push( fA4, fA4 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( +m33, -m23 );
    MathSSEFn->Push( fA3, fA3 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( outAdjointMatrix.m02, outAdjointMatrix.m03 );

    MathSSEFn->Push( -m10, +m00 );
    MathSSEFn->Push( fB5, fB5 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( +m12, -m02 );
    MathSSEFn->Push( fB2, fB2 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( -m13, +m03 );
    MathSSEFn->Push( fB1, fB1 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( outAdjointMatrix.m10, outAdjointMatrix.m11 );

    MathSSEFn->Push( -m30, +m20 );
    MathSSEFn->Push( fA5, fA5 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( +m32, -m22 );
    MathSSEFn->Push( fA2, fA2 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( -m33, +m23 );
    MathSSEFn->Push( fA1, fA1 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( outAdjointMatrix.m12, outAdjointMatrix.m13 );

    MathSSEFn->Push( +m10, -m00 );
    MathSSEFn->Push( fB4, fB4 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( -m11, +m01 );
    MathSSEFn->Push( fB2, fB2 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( +m13, -m03 );
    MathSSEFn->Push( fB0, fB0 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( outAdjointMatrix.m20, outAdjointMatrix.m21 );

    MathSSEFn->Push( +m30, -m20 );
    MathSSEFn->Push( fA4, fA4 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( -m31, +m21 );
    MathSSEFn->Push( fA2, fA2 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( +m33, -m23 );
    MathSSEFn->Push( fA0, fA0 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( outAdjointMatrix.m22, outAdjointMatrix.m23 );

    MathSSEFn->Push( -m10, +m00 );
    MathSSEFn->Push( fB3, fB3 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( +m11, -m01 );
    MathSSEFn->Push( fB1, fB1 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( -m12, +m02 );
    MathSSEFn->Push( fB0, fB0 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( outAdjointMatrix.m30, outAdjointMatrix.m31 );

    MathSSEFn->Push( -m30, +m20 );
    MathSSEFn->Push( fA3, fA3 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( +m31, -m21 );
    MathSSEFn->Push( fA1, fA1 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( -m32, +m22 );
    MathSSEFn->Push( fA0, fA0 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Pop( outAdjointMatrix.m32, outAdjointMatrix.m33 );
}

template<>
Bool TMatrix4<Float>::Invert( TMatrix4<Float> & outInvMatrix, Float fZeroTolerance ) const
{
    Float fA0, fA1, fA2, fA3, fA4, fA5;
    Float fB0, fB1, fB2, fB3, fB4, fB5, fInvDet;

    MathSSEFn->Push( m00, m00, m00, m01 );
    MathSSEFn->Push( m11, m12, m13, m12 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m01, m02, m03, m02 );
    MathSSEFn->Push( m10, m10, m10, m11 );
    MathSSEFn->MulPF();
    MathSSEFn->SubPF();
    MathSSEFn->Pop( fA0, fA1, fA2, fA3 );

    MathSSEFn->Push( m01, m02, m20, m20 );
    MathSSEFn->Push( m13, m13, m31, m32 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m03, m03, m21, m22 );
    MathSSEFn->Push( m11, m12, m30, m30 );
    MathSSEFn->MulPF();
    MathSSEFn->SubPF();
    MathSSEFn->Pop( fA4, fA5, fB0, fB1 );

    MathSSEFn->Push( m20, m21, m21, m22 );
    MathSSEFn->Push( m33, m32, m33, m33 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( m23, m22, m23, m23 );
    MathSSEFn->Push( m30, m31, m31, m32 );
    MathSSEFn->MulPF();
    MathSSEFn->SubPF();
    MathSSEFn->Pop( fB2, fB3, fB4, fB5 );

    MathSSEFn->Push( fA0, -fA1, fA2, 0.0f );
    MathSSEFn->Push( fB5, fB4, fB3, 0.0f );
    MathSSEFn->MulPF();
    MathSSEFn->Push( fA3, -fA4, fA5, 0.0f );
    MathSSEFn->Push( fB2, fB1, fB0, 0.0f );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->HAddF(0);
    MathSSEFn->HAddF(0);
    MathSSEFn->Pop(fInvDet);

    if ( MathFFn->Abs(fInvDet) < fZeroTolerance )
        return false;
    fInvDet = MathFFn->Invert(fInvDet);

    MathSSEFn->Push( fInvDet, fInvDet, fInvDet, fInvDet );

    MathSSEFn->Push( +m11, -m01, +m31, -m21 );
    MathSSEFn->Push( fB5, fB5, fA5, fA5 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( -m12, +m02, -m32, +m22 );
    MathSSEFn->Push( fB4, fB4, fA4, fA4 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( +m13, -m03, +m33, -m23 );
    MathSSEFn->Push( fB3, fB3, fA3, fA3 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->MulF(0, 1);
    MathSSEFn->Pop( outInvMatrix.m00, outInvMatrix.m01, outInvMatrix.m02, outInvMatrix.m03 );

    MathSSEFn->Push( -m10, +m00, -m30, +m20 );
    MathSSEFn->Push( fB5, fB5, fA5, fA5 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( +m12, -m02, +m32, -m22 );
    MathSSEFn->Push( fB2, fB2, fA2, fA2 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( -m13, +m03, -m33, +m23 );
    MathSSEFn->Push( fB1, fB1, fA1, fA1 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->MulF(0, 1);
    MathSSEFn->Pop( outInvMatrix.m10, outInvMatrix.m11, outInvMatrix.m12, outInvMatrix.m13 );

    MathSSEFn->Push( +m10, -m00, +m30, -m20 );
    MathSSEFn->Push( fB4, fB4, fA4, fA4 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( -m11, +m01, -m31, +m21 );
    MathSSEFn->Push( fB2, fB2, fA2, fA2 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( +m13, -m03, +m33, -m23 );
    MathSSEFn->Push( fB0, fB0, fA0, fA0 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->MulF(0, 1);
    MathSSEFn->Pop( outInvMatrix.m20, outInvMatrix.m21, outInvMatrix.m22, outInvMatrix.m23 );

    MathSSEFn->Push( -m10, +m00, -m30, +m20 );
    MathSSEFn->Push( fB3, fB3, fA3, fA3 );
    MathSSEFn->MulPF();
    MathSSEFn->Push( +m11, -m01, +m31, -m21 );
    MathSSEFn->Push( fB1, fB1, fA1, fA1 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->Push( -m12, +m02, -m32, +m22 );
    MathSSEFn->Push( fB0, fB0, fA0, fA0 );
    MathSSEFn->MulPF();
    MathSSEFn->AddPF();
    MathSSEFn->MulF(0, 1);
    MathSSEFn->Pop( outInvMatrix.m30, outInvMatrix.m31, outInvMatrix.m32, outInvMatrix.m33 );

    MathSSEFn->Pop( fInvDet, fInvDet, fInvDet, fInvDet );

    return true;
}
template<>
Bool TMatrix4<Double>::Invert( TMatrix4<Double> & outInvMatrix, Double fZeroTolerance ) const
{
    Double fA0, fA1, fA2, fA3, fA4, fA5;
    Double fB0, fB1, fB2, fB3, fB4, fB5, fInvDet;

    MathSSEFn->Push( m00, m00 );
    MathSSEFn->Push( m11, m12 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m01, m02 );
    MathSSEFn->Push( m10, m10 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fA0, fA1 );

    MathSSEFn->Push( m00, m01 );
    MathSSEFn->Push( m13, m12 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m03, m02 );
    MathSSEFn->Push( m10, m11 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fA2, fA3 );

    MathSSEFn->Push( m01, m02 );
    MathSSEFn->Push( m13, m13 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m03, m03 );
    MathSSEFn->Push( m11, m12 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fA4, fA5 );

    MathSSEFn->Push( m20, m20 );
    MathSSEFn->Push( m31, m32 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m21, m22 );
    MathSSEFn->Push( m30, m30 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fB0, fB1 );

    MathSSEFn->Push( m20, m21 );
    MathSSEFn->Push( m33, m32 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m23, m22 );
    MathSSEFn->Push( m30, m31 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fB2, fB3 );

    MathSSEFn->Push( m21, m22 );
    MathSSEFn->Push( m33, m33 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( m23, m23 );
    MathSSEFn->Push( m31, m32 );
    MathSSEFn->MulPD();
    MathSSEFn->SubPD();
    MathSSEFn->Pop( fB4, fB5 );

    MathSSEFn->Push( fA0, -fA1 );
    MathSSEFn->Push( fB5, fB4 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( fA2, fA3 );
    MathSSEFn->Push( fB3, fB2 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( -fA4, fA5 );
    MathSSEFn->Push( fB1, fB0 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->HAddD(0);
    MathSSEFn->Pop(fInvDet);

    if ( MathDFn->Abs(fInvDet) < fZeroTolerance )
        return false;
    fInvDet = MathDFn->Invert(fInvDet);

    MathSSEFn->Push( fInvDet, fInvDet );

    MathSSEFn->Push( +m11, -m01 );
    MathSSEFn->Push( fB5, fB5 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( -m12, +m02 );
    MathSSEFn->Push( fB4, fB4 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( +m13, -m03 );
    MathSSEFn->Push( fB3, fB3 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->MulD(0, 1);
    MathSSEFn->Pop( outInvMatrix.m00, outInvMatrix.m01 );

    MathSSEFn->Push( +m31, -m21 );
    MathSSEFn->Push( fA5, fA5 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( -m32, +m22 );
    MathSSEFn->Push( fA4, fA4 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( +m33, -m23 );
    MathSSEFn->Push( fA3, fA3 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->MulD(0, 1);
    MathSSEFn->Pop( outInvMatrix.m02, outInvMatrix.m03 );

    MathSSEFn->Push( -m10, +m00 );
    MathSSEFn->Push( fB5, fB5 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( +m12, -m02 );
    MathSSEFn->Push( fB2, fB2 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( -m13, +m03 );
    MathSSEFn->Push( fB1, fB1 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->MulD(0, 1);
    MathSSEFn->Pop( outInvMatrix.m10, outInvMatrix.m11 );

    MathSSEFn->Push( -m30, +m20 );
    MathSSEFn->Push( fA5, fA5 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( +m32, -m22 );
    MathSSEFn->Push( fA2, fA2 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( -m33, +m23 );
    MathSSEFn->Push( fA1, fA1 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->MulD(0, 1);
    MathSSEFn->Pop( outInvMatrix.m12, outInvMatrix.m13 );

    MathSSEFn->Push( +m10, -m00 );
    MathSSEFn->Push( fB4, fB4 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( -m11, +m01 );
    MathSSEFn->Push( fB2, fB2 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( +m13, -m03 );
    MathSSEFn->Push( fB0, fB0 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->MulD(0, 1);
    MathSSEFn->Pop( outInvMatrix.m20, outInvMatrix.m21 );

    MathSSEFn->Push( +m30, -m20 );
    MathSSEFn->Push( fA4, fA4 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( -m31, +m21 );
    MathSSEFn->Push( fA2, fA2 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( +m33, -m23 );
    MathSSEFn->Push( fA0, fA0 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->MulD(0, 1);
    MathSSEFn->Pop( outInvMatrix.m22, outInvMatrix.m23 );

    MathSSEFn->Push( -m10, +m00 );
    MathSSEFn->Push( fB3, fB3 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( +m11, -m01 );
    MathSSEFn->Push( fB1, fB1 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( -m12, +m02 );
    MathSSEFn->Push( fB0, fB0 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->MulD(0, 1);
    MathSSEFn->Pop( outInvMatrix.m30, outInvMatrix.m31 );

    MathSSEFn->Push( -m30, +m20 );
    MathSSEFn->Push( fA3, fA3 );
    MathSSEFn->MulPD();
    MathSSEFn->Push( +m31, -m21 );
    MathSSEFn->Push( fA1, fA1 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->Push( -m32, +m22 );
    MathSSEFn->Push( fA0, fA0 );
    MathSSEFn->MulPD();
    MathSSEFn->AddPD();
    MathSSEFn->MulD(0, 1);
    MathSSEFn->Pop( outInvMatrix.m32, outInvMatrix.m33 );

    MathSSEFn->Pop( fInvDet, fInvDet );

    return true;
}

#endif // SIMD_ENABLE


