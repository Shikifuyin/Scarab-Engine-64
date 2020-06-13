/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Bit.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Bit operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Bit implementation
inline __m128 SIMD::Bit::And( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_and_ps( mDst, mSrc );
}
inline __m128d SIMD::Bit::And( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_and_pd( mDst, mSrc );
}
inline __m128i SIMD::Bit::And( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_and_si128( mDst, mSrc );
}

inline __m256 SIMD::Bit::And( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_and_ps( mDst, mSrc );
}
inline __m256d SIMD::Bit::And( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_and_pd( mDst, mSrc );
}
inline __m256i SIMD::Bit::And( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_and_si256( mDst, mSrc );
}

inline __m128 SIMD::Bit::AndNot( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_andnot_ps( mDst, mSrc );
}
inline __m128d SIMD::Bit::AndNot( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_andnot_pd( mDst, mSrc );
}
inline __m128i SIMD::Bit::AndNot( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_andnot_si128( mDst, mSrc );
}

inline __m256 SIMD::Bit::AndNot( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_andnot_ps( mDst, mSrc );
}
inline __m256d SIMD::Bit::AndNot( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_andnot_pd( mDst, mSrc );
}
inline __m256i SIMD::Bit::AndNot( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_andnot_si256( mDst, mSrc );
}

inline __m128 SIMD::Bit::Or( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_or_ps( mDst, mSrc );
}
inline __m128d SIMD::Bit::Or( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_or_pd( mDst, mSrc );
}
inline __m128i SIMD::Bit::Or( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_or_si128( mDst, mSrc );
}

inline __m256 SIMD::Bit::Or( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_or_ps( mDst, mSrc );
}
inline __m256d SIMD::Bit::Or( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_or_pd( mDst, mSrc );
}
inline __m256i SIMD::Bit::Or( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_or_si256( mDst, mSrc );
}

inline __m128 SIMD::Bit::Xor( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_xor_ps( mDst, mSrc );
}
inline __m128d SIMD::Bit::Xor( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_xor_pd( mDst, mSrc );
}
inline __m128i SIMD::Bit::Xor( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_xor_si128( mDst, mSrc );
}

inline __m256 SIMD::Bit::Xor( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_xor_ps( mDst, mSrc );
}
inline __m256d SIMD::Bit::Xor( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_xor_pd( mDst, mSrc );
}
inline __m256i SIMD::Bit::Xor( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_xor_si256( mDst, mSrc );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Bit::Shift16 implementation
inline __m128i SIMD::Bit::Shift16::Left( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_slli_epi16( mDst, iCount );
}
inline __m128i SIMD::Bit::Shift16::Left( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sll_epi16( mDst, mCount );
}

inline __m256i SIMD::Bit::Shift16::Left( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_slli_epi16( mDst, iCount );
}
inline __m256i SIMD::Bit::Shift16::Left( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sll_epi16( mDst, mCount );
}

inline __m128i SIMD::Bit::Shift16::Right( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srli_epi16( mDst, iCount );
}
inline __m128i SIMD::Bit::Shift16::Right( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srl_epi16( mDst, mCount );
}

inline __m256i SIMD::Bit::Shift16::Right( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srli_epi16( mDst, iCount );
}
inline __m256i SIMD::Bit::Shift16::Right( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srl_epi16( mDst, mCount );
}

inline __m128i SIMD::Bit::Shift16::RightSE( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srai_epi16( mDst, iCount );
}
inline __m128i SIMD::Bit::Shift16::RightSE( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sra_epi16( mDst, mCount );
}

inline __m256i SIMD::Bit::Shift16::RightSE( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srai_epi16( mDst, iCount );
}
inline __m256i SIMD::Bit::Shift16::RightSE( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sra_epi16( mDst, mCount );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Bit::Shift32 implementation
inline __m128i SIMD::Bit::Shift32::Left( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_slli_epi32( mDst, iCount );
}
inline __m128i SIMD::Bit::Shift32::Left( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sll_epi32( mDst, mCount );
}

inline __m256i SIMD::Bit::Shift32::Left( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_slli_epi32( mDst, iCount );
}
inline __m256i SIMD::Bit::Shift32::Left( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sll_epi32( mDst, mCount );
}

inline __m128i SIMD::Bit::Shift32::Right( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srli_epi32( mDst, iCount );
}
inline __m128i SIMD::Bit::Shift32::Right( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srl_epi32( mDst, mCount );
}

inline __m256i SIMD::Bit::Shift32::Right( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srli_epi32( mDst, iCount );
}
inline __m256i SIMD::Bit::Shift32::Right( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srl_epi32( mDst, mCount );
}


inline __m128i SIMD::Bit::Shift32::LeftV( __m128i mDst, __m128i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_sllv_epi32( mDst, mCounts );
}
inline __m256i SIMD::Bit::Shift32::LeftV( __m256i mDst, __m256i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sllv_epi32( mDst, mCounts );
}

inline __m128i SIMD::Bit::Shift32::RightV( __m128i mDst, __m128i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_srlv_epi32( mDst, mCounts );
}
inline __m256i SIMD::Bit::Shift32::RightV( __m256i mDst, __m256i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srlv_epi32( mDst, mCounts );
}

inline __m128i SIMD::Bit::Shift32::RightSE( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srai_epi32( mDst, iCount );
}
inline __m128i SIMD::Bit::Shift32::RightSE( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sra_epi32( mDst, mCount );
}

inline __m256i SIMD::Bit::Shift32::RightSE( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srai_epi32( mDst, iCount );
}
inline __m256i SIMD::Bit::Shift32::RightSE( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sra_epi32( mDst, mCount );
}

inline __m128i SIMD::Bit::Shift32::RightVSE( __m128i mDst, __m128i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_srav_epi32( mDst, mCounts );
}
inline __m256i SIMD::Bit::Shift32::RightVSE( __m256i mDst, __m256i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srav_epi32( mDst, mCounts );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Bit::Shift64 implementation
inline __m128i SIMD::Bit::Shift64::Left( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_slli_epi64( mDst, iCount );
}
inline __m128i SIMD::Bit::Shift64::Left( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sll_epi64( mDst, mCount );
}

inline __m256i SIMD::Bit::Shift64::Left( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_slli_epi64( mDst, iCount );
}
inline __m256i SIMD::Bit::Shift64::Left( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sll_epi64( mDst, mCount );
}

inline __m128i SIMD::Bit::Shift64::Right( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srli_epi64( mDst, iCount );
}
inline __m128i SIMD::Bit::Shift64::Right( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srl_epi64( mDst, mCount );
}

inline __m256i SIMD::Bit::Shift64::Right( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srli_epi64( mDst, iCount );
}
inline __m256i SIMD::Bit::Shift64::Right( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srl_epi64( mDst, mCount );
}

inline __m128i SIMD::Bit::Shift64::LeftV( __m128i mDst, __m128i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_sllv_epi64( mDst, mCounts );
}
inline __m256i SIMD::Bit::Shift64::LeftV( __m256i mDst, __m256i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sllv_epi64( mDst, mCounts );
}

inline __m128i SIMD::Bit::Shift64::RightV( __m128i mDst, __m128i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_srlv_epi64( mDst, mCounts );
}
inline __m256i SIMD::Bit::Shift64::RightV( __m256i mDst, __m256i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srlv_epi64( mDst, mCounts );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Bit::Shift128 implementation

//inline __m128i SIMD::Bit::Shift128::Left( __m128i mDst, Int iCount ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_slli_si128( mDst, iCount );
//}

//inline __m128i SIMD::Bit::Shift128::Right( __m128i mDst, Int iCount ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_srli_si128( mDst, iCount );
//}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Bit::Shift256 implementation

//inline __m256i SIMD::Bit::Shift256::Left( __m256i mDst, Int iCount ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_slli_si256( mDst, iCount );
//}

//inline __m256i SIMD::Bit::Shift256::Right( __m256i mDst, Int iCount ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_srli_si256( mDst, iCount );
//}
