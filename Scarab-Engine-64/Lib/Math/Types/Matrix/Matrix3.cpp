/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Matrix/Matrix3.cpp
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
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "Matrix2.h"
#include "Matrix3.h"
#include "Matrix4.h"

/////////////////////////////////////////////////////////////////////////////////
// TMatrix3 implementation
#ifdef SIMD_ENABLE

//template<>
//TMatrix3<Float> TMatrix3<Float>::operator*( const Float & rhs ) const
//{
//    
//}
//template<>
//TMatrix3<Double> TMatrix3<Double>::operator*( const Double & rhs ) const
//{
//    
//}
//
//template<>
//TMatrix3<Float> TMatrix3<Float>::operator/( const Float & rhs ) const
//{
//    
//}
//template<>
//TMatrix3<Double> TMatrix3<Double>::operator/( const Double & rhs ) const
//{
//    
//}

//template<>
//TMatrix3<Float> & TMatrix3<Float>::operator*=( const Float & rhs )
//{
//   
//    return (*this);
//}
//template<>
//TMatrix3<Double> & TMatrix3<Double>::operator*=( const Double & rhs )
//{
//    
//    return (*this);
//}
//
//template<>
//TMatrix3<Float> & TMatrix3<Float>::operator/=( const Float & rhs )
//{
//    
//    return (*this);
//}
//template<>
//TMatrix3<Double> & TMatrix3<Double>::operator/=( const Double & rhs )
//{
//    
//    return (*this);
//}

template<>
TVertex3<Float> TMatrix3<Float>::operator*( const TVertex3<Float> & rhs ) const
{
    // Faster to just cast ...
    TMatrix4<Float> mat4( *this );
    TVertex4<Float> vRes = mat4 * TVertex4<Float>( rhs );
    return TVertex3<Float>( vRes );
}
template<>
TVertex3<Double> TMatrix3<Double>::operator*( const TVertex3<Double> & rhs ) const
{
    // Faster to just cast ...
    TMatrix4<Double> mat4( *this );
    TVertex4<Double> vRes = mat4 * TVertex4<Double>( rhs );
    return TVertex3<Double>( vRes );
}

template<>
TVector3<Float> TMatrix3<Float>::operator*( const TVector3<Float> & rhs ) const
{
    // Faster to just cast ...
    TMatrix4<Float> mat4( *this );
    TVector4<Float> vRes = mat4 * TVector4<Float>( rhs );
    return TVector3<Float>( vRes );
}
template<>
TVector3<Double> TMatrix3<Double>::operator*( const TVector3<Double> & rhs ) const
{
    // Faster to just cast ...
    TMatrix4<Double> mat4( *this );
    TVector4<Double> vRes = mat4 * TVector4<Double>( rhs );
    return TVector3<Double>( vRes );
}

//template<>
//TMatrix3<Float> TMatrix3<Float>::operator+( const TMatrix3<Float> & rhs ) const
//{
//    
//}
//template<>
//TMatrix3<Double> TMatrix3<Double>::operator+( const TMatrix3<Double> & rhs ) const
//{
//    
//}
//
//template<>
//TMatrix3<Float> TMatrix3<Float>::operator-( const TMatrix3<Float> & rhs ) const
//{
//    
//}
//template<>
//TMatrix3<Double> TMatrix3<Double>::operator-( const TMatrix3<Double> & rhs ) const
//{
//    
//}

template<>
TMatrix3<Float> TMatrix3<Float>::operator*( const TMatrix3<Float> & rhs ) const
{
    // Faster to just cast ...
    TMatrix4<Float> matRes( *this );
    matRes *= TMatrix4<Float>( rhs );
    return matRes;
}
template<>
TMatrix3<Double> TMatrix3<Double>::operator*( const TMatrix3<Double> & rhs ) const
{
    // Faster to just cast ...
    TMatrix4<Double> matRes( *this );
    matRes *= TMatrix4<Double>( rhs );
    return matRes;
    
}

//template<>
//TMatrix3<Float> & TMatrix3<Float>::operator+=( const TMatrix3<Float> & rhs )
//{
//    
//    return (*this);
//}
//template<>
//TMatrix3<Double> & TMatrix3<Double>::operator+=( const TMatrix3<Double> & rhs )
//{
//    
//    return (*this);
//}
//
//template<>
//TMatrix3<Float> & TMatrix3<Float>::operator-=( const TMatrix3<Float> & rhs )
//{
//    
//    return (*this);
//}
//template<>
//TMatrix3<Double> & TMatrix3<Double>::operator-=( const TMatrix3<Double> & rhs )
//{
//   
//    return (*this);
//}

template<>
TMatrix3<Float> & TMatrix3<Float>::operator*=( const TMatrix3<Float> & rhs )
{
    // Faster to just cast ...
    TMatrix4<Float> matRes( *this );
    matRes *= TMatrix4<Float>( rhs );
    *this = TMatrix3<Float>( matRes );
    return (*this);
}
template<>
TMatrix3<Double> & TMatrix3<Double>::operator*=( const TMatrix3<Double> & rhs )
{
    // Faster to just cast ...
    TMatrix4<Double> matRes( *this );
    matRes *= TMatrix4<Double>( rhs );
    *this = TMatrix3<Double>( matRes );
    return (*this);
}

template<>
Float TMatrix3<Float>::Determinant() const
{
    alignas(16) static Float arrTmp[4];

    // Load Data
    __m256 mLHS = SIMD::Import::Values::Set( m11, m01, m01, 0.0f, m21, m21, m11, 0.0f );
    __m256 mRHS = SIMD::Import::Values::Set( m22, m22, m12, 0.0f, m12, m02, m02, 0.0f );
    __m128 mTmp = SIMD::Import::Values::Set( m00, m10, m20, 0.0f );

    // Perform Operations
    __m256 mCombine = SIMD::Math::Mul( mLHS, mRHS );

    __m128 mResult = SIMD::Cast::Down( mCombine );
    __m128 mA = SIMD::Register::Move::FourFloatH( mCombine );

    mResult = SIMD::Math::Sub( mResult, mA );
    mResult = SIMD::Math::Mul( mResult, mTmp );

    // Save Data
    SIMD::Export::Memory::Aligned::Save128( arrTmp, mResult );

    // Done
    return ( arrTmp[0] - arrTmp[1] + arrTmp[2] );
}
template<>
Double TMatrix3<Double>::Determinant() const
{
    alignas(32) static Double arrTmp[4];

    // Load Data
    __m256d mLHS0 = SIMD::Import::Values::Set( m11, m01, m01, 0.0 );
    __m256d mRHS0 = SIMD::Import::Values::Set( m22, m22, m12, 0.0 );

    __m256d mLHS1 = SIMD::Import::Values::Set( m21, m21, m11, 0.0 );
    __m256d mRHS1 = SIMD::Import::Values::Set( m12, m02, m02, 0.0 );

    __m256d mTmp = SIMD::Import::Values::Set( m00, m10, m20, 0.0 );

    // Perform Operations
    __m256d mResult = SIMD::Math::Mul( mLHS0, mRHS0 );
    __m256d mCombine = SIMD::Math::Mul( mLHS1, mRHS1 );
    
    mResult = SIMD::Math::Sub( mResult, mCombine );
    mResult = SIMD::Math::Mul( mResult, mTmp );

    // Save Data
    SIMD::Export::Memory::Aligned::Save256( arrTmp, mResult );

    // Done
    return ( arrTmp[0] - arrTmp[1] + arrTmp[2] );
}

//template<>
//Void TMatrix3<Float>::Adjoint( TMatrix3<Float> & outAdjointMatrix ) const
//{
//    
//}
//template<>
//Void TMatrix3<Double>::Adjoint( TMatrix3<Double> & outAdjointMatrix ) const
//{
//    
//}

template<>
Bool TMatrix3<Float>::Invert( TMatrix3<Float> & outInvMatrix, Float fZeroTolerance ) const
{
    alignas(16) static Float arrTmpDet[4];
    alignas(32) static Float arrTmpValues[12];
    
    // Load Data
    __m256 mLHS0 = SIMD::Import::Values::Set( m11, m01, m01, 0.0f, m21, m21, m11, 0.0f );
    __m256 mRHS0 = SIMD::Import::Values::Set( m22, m22, m12, 0.0f, m12, m02, m02, 0.0f );

    __m256 mLHS1 = SIMD::Import::Values::Set( m10, m00, m00, 0.0f, m10, m00, m00, 0.0f );
    __m256 mRHS1 = SIMD::Import::Values::Set( m22, m22, m12, 0.0f, m21, m21, m11, 0.0f );

    __m256 mLHS2 = SIMD::Import::Values::Set( m20, m20, m10, 0.0f, m20, m20, m10, 0.0f );
    __m256 mRHS2 = SIMD::Import::Values::Set( m12, m02, m02, 0.0f, m11, m01, m01, 0.0f );

    __m128 mDet = SIMD::Import::Values::Set( m00, m10, m20, 0.0f );

    // Part 1
    __m256 mCombine = SIMD::Math::Mul( mLHS0, mRHS0 ); // mCombine = ( m11*m22, m01*m22, m01*m12, 0, m21*m12, m21*m02, m11*m02, 0 )

    __m128 mA = SIMD::Cast::Down( mCombine );                   // mA = ( m11*m22, m01*m22, m01*m12, 0 )
    __m128 mTmp = SIMD::Register::Move::FourFloatH( mCombine ); // mTmp = ( m21*m12, m21*m02, m11*m02, 0 )

    mA = SIMD::Math::Sub( mA, mTmp );// mTmp = ( m11*m22 - m21*m12, m01*m22 - m21*m02, m01*m12 - m11*m02, 0 )

    // Check determinant
    mDet = SIMD::Math::Mul( mDet, mA );
    SIMD::Export::Memory::Aligned::Save128( arrTmpDet, mDet );

    Float fInvDet = ( arrTmpDet[0] - arrTmpDet[1] + arrTmpDet[2] );
    if ( MathFFn->Abs(fInvDet) < fZeroTolerance )
        return false;
    fInvDet = MathFFn->Invert( fInvDet );

    // Part 2 & 3
    __m256 mBC = SIMD::Math::Mul( mLHS1, mRHS1 ); // mBC = ( m10*m22, m00*m22, m00*m12, 0, m10*m21, m00*m21, m00*m11, 0 )
    mCombine = SIMD::Math::Mul( mLHS2, mRHS2 );   // mCombine = ( m20*m12, m20*m02, m10*m02, 0, m20*m11, m20*m01, m10*m01, 0 )
    mBC = SIMD::Math::Sub( mBC, mCombine );       // mBC = ( m10*m22 - m20*m12, m00*m22 - m20*m02, m00*m12 - m10*m02, 0,
                                                  //         m10*m21 - m20*m11, m00*m21 - m20*m01, m00*m11 - m10*m01, 0 )

    // Apply Det factor
    mCombine = SIMD::Import::Memory::Spread256( &fInvDet );
    mA = SIMD::Math::Mul( mA, SIMD::Cast::Down(mCombine) );
    mBC = SIMD::Math::Mul( mBC, mCombine );

    // Save Data
    SIMD::Export::Memory::Aligned::Save128( arrTmpValues, mA );
    SIMD::Export::Memory::Aligned::Save256( arrTmpValues + 4, mBC );

    // Transpose
    outInvMatrix.m00 = +arrTmpValues[0]; outInvMatrix.m01 = -arrTmpValues[1]; outInvMatrix.m02 = +arrTmpValues[2];
    outInvMatrix.m10 = -arrTmpValues[4]; outInvMatrix.m11 = +arrTmpValues[5]; outInvMatrix.m12 = -arrTmpValues[6];
    outInvMatrix.m20 = +arrTmpValues[8]; outInvMatrix.m21 = -arrTmpValues[9]; outInvMatrix.m22 = +arrTmpValues[10];

    // Done
    return true;
}
template<>
Bool TMatrix3<Double>::Invert( TMatrix3<Double> & outInvMatrix, Double fZeroTolerance ) const
{
    alignas(32) static Double arrTmpDet[4];
    alignas(32) static Double arrTmpValues[12];

    // Load Data
    __m256d mLHS0 = SIMD::Import::Values::Set( m11, m01, m01, 0.0 );
    __m256d mRHS0 = SIMD::Import::Values::Set( m22, m22, m12, 0.0 );

    __m256d mLHS1 = SIMD::Import::Values::Set( m21, m21, m11, 0.0 );
    __m256d mRHS1 = SIMD::Import::Values::Set( m12, m02, m02, 0.0 );

    __m256d mLHS2 = SIMD::Import::Values::Set( m10, m00, m00, 0.0 );
    __m256d mRHS2 = SIMD::Import::Values::Set( m22, m22, m12, 0.0 );

    __m256d mLHS3 = SIMD::Import::Values::Set( m20, m20, m10, 0.0 );
    __m256d mRHS3 = SIMD::Import::Values::Set( m12, m02, m02, 0.0 );

    __m256d mLHS4 = SIMD::Import::Values::Set( m10, m00, m00, 0.0 );
    __m256d mRHS4 = SIMD::Import::Values::Set( m21, m21, m11, 0.0 );

    __m256d mLHS5 = SIMD::Import::Values::Set( m20, m20, m10, 0.0 );
    __m256d mRHS5 = SIMD::Import::Values::Set( m11, m01, m01, 0.0 );

    __m256d mDet = SIMD::Import::Values::Set( m00, m10, m20, 0.0 );

    // Part 1
    __m256d mA = SIMD::Math::Mul( mLHS0, mRHS0 );       // mA = ( m11*m22, m01*m22, m01*m12, 0 )
    __m256d mCombine = SIMD::Math::Mul( mLHS1, mRHS1 ); // mCombine = ( m21*m12, m21*m02, m11*m02, 0 )

    mA = SIMD::Math::Sub( mA, mCombine ); // mA = ( m11*m22 - m21*m12, m01*m22 - m21*m02, m01*m12 - m11*m02, 0 )

    // Check determinant
    mDet = SIMD::Math::Mul( mDet, mA );
    SIMD::Export::Memory::Aligned::Save256( arrTmpDet, mDet );

    Double fInvDet = ( arrTmpDet[0] - arrTmpDet[1] + arrTmpDet[2] );
    if ( MathFFn->Abs(fInvDet) < fZeroTolerance )
        return false;
    fInvDet = MathFFn->Invert( fInvDet );

    // Part 2
    __m256d mB = SIMD::Math::Mul( mLHS2, mRHS2 ); // mB = ( m10*m22, m00*m22, m00*m12, 0 )
    mCombine = SIMD::Math::Mul( mLHS3, mRHS3 );   // mCombine = ( m20*m12, m20*m02, m10*m02, 0 )

    mB = SIMD::Math::Sub( mB, mCombine ); // mB = ( m10*m22 - m20*m12, m00*m22 - m20*m02, m00*m12 - m10*m02, 0 )

    // Part 3
    __m256d mC = SIMD::Math::Mul( mLHS4, mRHS4 ); // mC = ( m10*m21, m00*m21, m00*m11, 0 )
    mCombine = SIMD::Math::Mul( mLHS5, mRHS5 );   // mCombine = ( m20*m11, m20*m01, m10*m01, 0 )

    mC = SIMD::Math::Sub( mC, mCombine ); // mC = ( m10*m21 - m20*m11, m00*m21 - m20*m01, m00*m11 - m10*m01, 0 )

    // Apply Det factor
    mCombine = SIMD::Import::Memory::Spread256( &fInvDet );
    mA = SIMD::Math::Mul( mA, mCombine );
    mB = SIMD::Math::Mul( mB, mCombine );
    mC = SIMD::Math::Mul( mC, mCombine );

    // Save Data
    SIMD::Export::Memory::Aligned::Save256( arrTmpValues, mA );
    SIMD::Export::Memory::Aligned::Save256( arrTmpValues + 4, mB );
    SIMD::Export::Memory::Aligned::Save256( arrTmpValues + 8, mC );

    // Transpose
    outInvMatrix.m00 = +arrTmpValues[0]; outInvMatrix.m01 = -arrTmpValues[1]; outInvMatrix.m02 = +arrTmpValues[2];
    outInvMatrix.m10 = -arrTmpValues[4]; outInvMatrix.m11 = +arrTmpValues[5]; outInvMatrix.m12 = -arrTmpValues[6];
    outInvMatrix.m20 = +arrTmpValues[8]; outInvMatrix.m21 = -arrTmpValues[9]; outInvMatrix.m22 = +arrTmpValues[10];

    // Done
    return true;
}

#endif // SIMD_ENABLE

