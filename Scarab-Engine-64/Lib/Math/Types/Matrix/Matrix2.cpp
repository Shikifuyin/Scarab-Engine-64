/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Types/Matrix/Matrix2.cpp
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
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "Matrix2.h"
#include "Matrix3.h"
#include "Matrix4.h"

/////////////////////////////////////////////////////////////////////////////////
// TMatrix2 implementation
#ifdef SIMD_ENABLE

template<>
TMatrix2<Float> TMatrix2<Float>::operator*( const Float & rhs ) const
{
    TMatrix2<Float> matRes;

    __m128 mLHS = SIMD::Import::Memory::Aligned::Load128( &m00 );
    __m128 mRHS = SIMD::Import::Memory::Spread128( &rhs );

    mLHS = SIMD::Math::Mul( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save128( &(matRes.m00), mLHS );

    return matRes;
}
template<>
TMatrix2<Double> TMatrix2<Double>::operator*( const Double & rhs ) const
{
    TMatrix2<Double> matRes;

    __m256d mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256d mRHS = SIMD::Import::Memory::Spread256( &rhs );

    mLHS = SIMD::Math::Mul( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mLHS );

    return matRes;
}

template<>
TMatrix2<Float> TMatrix2<Float>::operator/( const Float & rhs ) const
{
   TMatrix2<Float> matRes;

    __m128 mLHS = SIMD::Import::Memory::Aligned::Load128( &m00 );
    __m128 mRHS = SIMD::Import::Memory::Spread128( &rhs );

    mLHS = SIMD::Math::Div( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save128( &(matRes.m00), mLHS );

    return matRes;
}
template<>
TMatrix2<Double> TMatrix2<Double>::operator/( const Double & rhs ) const
{
    TMatrix2<Double> matRes;

    __m256d mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256d mRHS = SIMD::Import::Memory::Spread256( &rhs );

    mLHS = SIMD::Math::Div( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mLHS );

    return matRes;
}

template<>
TMatrix2<Float> & TMatrix2<Float>::operator*=( const Float & rhs )
{
    __m128 mLHS = SIMD::Import::Memory::Aligned::Load128( &m00 );
    __m128 mRHS = SIMD::Import::Memory::Spread128( &rhs );

    mLHS = SIMD::Math::Mul( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save128( &m00, mLHS );

    return (*this);
}
template<>
TMatrix2<Double> & TMatrix2<Double>::operator*=( const Double & rhs )
{
    __m256d mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256d mRHS = SIMD::Import::Memory::Spread256( &rhs );

    mLHS = SIMD::Math::Mul( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save256( &m00, mLHS );

    return (*this);
}

template<>
TMatrix2<Float> & TMatrix2<Float>::operator/=( const Float & rhs )
{
    __m128 mLHS = SIMD::Import::Memory::Aligned::Load128( &m00 );
    __m128 mRHS = SIMD::Import::Memory::Spread128( &rhs );

    mLHS = SIMD::Math::Div( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save128( &m00, mLHS );

    return (*this);
}
template<>
TMatrix2<Double> & TMatrix2<Double>::operator/=( const Double & rhs )
{
    __m256d mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256d mRHS = SIMD::Import::Memory::Spread256( &rhs );

    mLHS = SIMD::Math::Div( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save256( &m00, mLHS );

    return (*this);
}

template<>
TMatrix2<Float> TMatrix2<Float>::operator+( const TMatrix2<Float> & rhs ) const
{
    TMatrix2<Float> matRes;

    __m128 mLHS = SIMD::Import::Memory::Aligned::Load128( &m00 );
    __m128 mRHS = SIMD::Import::Memory::Aligned::Load128( &(rhs.m00) );

    mLHS = SIMD::Math::Add( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save128( &(matRes.m00), mLHS );

    return matRes;
}
template<>
TMatrix2<Double> TMatrix2<Double>::operator+( const TMatrix2<Double> & rhs ) const
{
    TMatrix2<Double> matRes;

    __m256d mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256d mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) );

    mLHS = SIMD::Math::Add( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mLHS );

    return matRes;
}

template<>
TMatrix2<Float> TMatrix2<Float>::operator-( const TMatrix2<Float> & rhs ) const
{
    TMatrix2<Float> matRes;

    __m128 mLHS = SIMD::Import::Memory::Aligned::Load128( &m00 );
    __m128 mRHS = SIMD::Import::Memory::Aligned::Load128( &(rhs.m00) );

    mLHS = SIMD::Math::Sub( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save128( &(matRes.m00), mLHS );

    return matRes;
}
template<>
TMatrix2<Double> TMatrix2<Double>::operator-( const TMatrix2<Double> & rhs ) const
{
    TMatrix2<Double> matRes;

    __m256d mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256d mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) );

    mLHS = SIMD::Math::Sub( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mLHS );

    return matRes;
}

template<>
TMatrix2<Float> TMatrix2<Float>::operator*( const TMatrix2<Float> & rhs ) const
{
    TMatrix2<Float> matRes;

    // Load Data
    __m128 mLHS = SIMD::Import::Memory::Aligned::Load128( &m00 );
    __m128 mRHS = SIMD::Import::Memory::Aligned::Load128( &(rhs.m00) );

    __m256 mCombine0, mCombine1;
    __m128 mTmp0, mTmp1;

    // Perform Product
    mTmp0 = SIMD::Register::Spread4::ABAB( mLHS ); // mTmp0 = ( l00, l10, l00, l10 )
    mTmp1 = SIMD::Register::Spread4::CDCD( mLHS ); // mTmp1 = ( l01, l11, l01, l11 )

    mCombine0 = SIMD::Cast::Up( mTmp0 );
    mCombine0 = SIMD::Register::Move::FourFloatH( mCombine0, mTmp1 ); // mCombine0 = ( l00, l10, l00, l10, l01, l11, l01, l11 )

    mTmp0 = SIMD::Register::Spread4::AACC( mRHS ); // mTmp0 = ( r00, r00, r01, r01 )
    mTmp1 = SIMD::Register::Spread4::BBDD( mRHS ); // mTmp1 = ( r10, r10, r11, r11 )

    mCombine1 = SIMD::Cast::Up( mTmp0 );
    mCombine1 = SIMD::Register::Move::FourFloatH( mCombine1, mTmp1 ); // mCombine1 = ( r00, r00, r01, r01, r10, r10, r11, r11 )

    mCombine0 = SIMD::Math::Mul( mCombine0, mCombine1 ); // mCombine0 = ( l00*r00, l10*r00, l00*r01, l10*r01, l01*r10, l11*r10, l01*r11, l11*r11 )

    mTmp0 = SIMD::Cast::Down( mCombine0 );                 // mTmp0 = ( l00*r00, l10*r00, l00*r01, l10*r01 )
    mTmp1 = SIMD::Register::Move::FourFloatH( mCombine0 ); // mTmp1 = ( l01*r10, l11*r10, l01*r11, l11*r11 )
    mTmp0 = SIMD::Math::Add( mTmp0, mTmp1 );               // mTmp0 = ( l00*r00 + l01*r10, l10*r00 + l11*r10, l00*r01 + l01*r11, l10*r01 + l11*r11 )

    // Save Data
    SIMD::Export::Memory::Aligned::Save128( &(matRes.m00), mTmp0 );

    // Done
    return matRes;
}
template<>
TMatrix2<Double> TMatrix2<Double>::operator*( const TMatrix2<Double> & rhs ) const
{
    TMatrix2<Double> matRes;

    // Load Data
    __m256d mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256d mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) );

    __m256d mCombine0, mCombine1;
    __m256d mTmp0, mTmp1;

    // Perform Product
    mCombine0 = SIMD::Register::Spread2::AA( mLHS ); // mCombine0 = ( l00, l10, l00, l10 )
    mCombine1 = SIMD::Register::Spread2::BB( mLHS ); // mCombine1 = ( l01, l11, l01, l11 )

    mTmp0 = SIMD::Register::Spread4::AACC( mRHS ); // mTmp0 = ( r00, r00, r01, r01 )
    mTmp1 = SIMD::Register::Spread4::BBDD( mRHS ); // mTmp1 = ( r10, r10, r11, r11 )

    mCombine0 = SIMD::Math::Mul( mCombine0, mTmp0 );     // mCombine0 = ( l00*r00, l10*r00, l00*r01, l10*r01 )
    mCombine1 = SIMD::Math::Mul( mCombine1, mTmp1 );     // mCombine1 = ( l01*r10, l11*r10, l01*r11, l11*r11 )
    mCombine0 = SIMD::Math::Add( mCombine0, mCombine1 ); // mCombine0 = ( l00*r00 + l01*r10, l10*r00 + l11*r10, l00*r01 + l01*r11, l10*r01 + l11*r11 )

    // Save Data
    SIMD::Export::Memory::Aligned::Save256( &(matRes.m00), mCombine0 );

    // Done
    return matRes;
}

template<>
TMatrix2<Float> & TMatrix2<Float>::operator+=( const TMatrix2<Float> & rhs )
{
    __m128 mLHS = SIMD::Import::Memory::Aligned::Load128( &m00 );
    __m128 mRHS = SIMD::Import::Memory::Aligned::Load128( &(rhs.m00) );

    mLHS = SIMD::Math::Add( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save128( &m00, mLHS );

    return (*this);
}
template<>
TMatrix2<Double> & TMatrix2<Double>::operator+=( const TMatrix2<Double> & rhs )
{
    __m256d mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256d mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) );

    mLHS = SIMD::Math::Add( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save256( &m00, mLHS );

    return (*this);
}

template<>
TMatrix2<Float> & TMatrix2<Float>::operator-=( const TMatrix2<Float> & rhs )
{
    __m128 mLHS = SIMD::Import::Memory::Aligned::Load128( &m00 );
    __m128 mRHS = SIMD::Import::Memory::Aligned::Load128( &(rhs.m00) );

    mLHS = SIMD::Math::Sub( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save128( &m00, mLHS );

    return (*this);
}
template<>
TMatrix2<Double> & TMatrix2<Double>::operator-=( const TMatrix2<Double> & rhs )
{
    __m256d mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256d mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) );

    mLHS = SIMD::Math::Sub( mLHS, mRHS );

    SIMD::Export::Memory::Aligned::Save256( &m00, mLHS );

    return (*this);
}

template<>
TMatrix2<Float> & TMatrix2<Float>::operator*=( const TMatrix2<Float> & rhs )
{
    // Load Data
    __m128 mLHS = SIMD::Import::Memory::Aligned::Load128( &m00 );
    __m128 mRHS = SIMD::Import::Memory::Aligned::Load128( &(rhs.m00) );

    __m256 mCombine0, mCombine1;
    __m128 mTmp0, mTmp1;

    // Perform Product
    mTmp0 = SIMD::Register::Spread4::ABAB( mLHS ); // mTmp0 = ( l00, l10, l00, l10 )
    mTmp1 = SIMD::Register::Spread4::CDCD( mLHS ); // mTmp1 = ( l01, l11, l01, l11 )

    mCombine0 = SIMD::Cast::Up( mTmp0 );
    mCombine0 = SIMD::Register::Move::FourFloatH( mCombine0, mTmp1 ); // mCombine0 = ( l00, l10, l00, l10, l01, l11, l01, l11 )

    mTmp0 = SIMD::Register::Spread4::AACC( mRHS ); // mTmp0 = ( r00, r00, r01, r01 )
    mTmp1 = SIMD::Register::Spread4::BBDD( mRHS ); // mTmp1 = ( r10, r10, r11, r11 )

    mCombine1 = SIMD::Cast::Up( mTmp0 );
    mCombine1 = SIMD::Register::Move::FourFloatH( mCombine1, mTmp1 ); // mCombine1 = ( r00, r00, r01, r01, r10, r10, r11, r11 )

    mCombine0 = SIMD::Math::Mul( mCombine0, mCombine1 ); // mCombine0 = ( l00*r00, l10*r00, l00*r01, l10*r01, l01*r10, l11*r10, l01*r11, l11*r11 )

    mTmp0 = SIMD::Cast::Down( mCombine0 );                 // mTmp0 = ( l00*r00, l10*r00, l00*r01, l10*r01 )
    mTmp1 = SIMD::Register::Move::FourFloatH( mCombine0 ); // mTmp1 = ( l01*r10, l11*r10, l01*r11, l11*r11 )
    mTmp0 = SIMD::Math::Add( mTmp0, mTmp1 );               // mTmp0 = ( l00*r00 + l01*r10, l10*r00 + l11*r10, l00*r01 + l01*r11, l10*r01 + l11*r11 )

    // Save Data
    SIMD::Export::Memory::Aligned::Save128( &m00, mTmp0 );
    
    return (*this);
}
template<>
TMatrix2<Double> & TMatrix2<Double>::operator*=( const TMatrix2<Double> & rhs )
{
    // Load Data
    __m256d mLHS = SIMD::Import::Memory::Aligned::Load256( &m00 );
    __m256d mRHS = SIMD::Import::Memory::Aligned::Load256( &(rhs.m00) );

    __m256d mCombine0, mCombine1;
    __m256d mTmp0, mTmp1;

    // Perform Product
    mCombine0 = SIMD::Register::Spread2::AA( mLHS ); // mCombine0 = ( l00, l10, l00, l10 )
    mCombine1 = SIMD::Register::Spread2::BB( mLHS ); // mCombine1 = ( l01, l11, l01, l11 )

    mTmp0 = SIMD::Register::Spread4::AACC( mRHS ); // mTmp0 = ( r00, r00, r01, r01 )
    mTmp1 = SIMD::Register::Spread4::BBDD( mRHS ); // mTmp1 = ( r10, r10, r11, r11 )

    mCombine0 = SIMD::Math::Mul( mCombine0, mTmp0 );     // mCombine0 = ( l00*r00, l10*r00, l00*r01, l10*r01 )
    mCombine1 = SIMD::Math::Mul( mCombine1, mTmp1 );     // mCombine1 = ( l01*r10, l11*r10, l01*r11, l11*r11 )
    mCombine0 = SIMD::Math::Add( mCombine0, mCombine1 ); // mCombine0 = ( l00*r00 + l01*r10, l10*r00 + l11*r10, l00*r01 + l01*r11, l10*r01 + l11*r11 )

    // Save Data
    SIMD::Export::Memory::Aligned::Save256( &m00, mCombine0 );
    
    return (*this);
}

#endif // SIMD_ENABLE
