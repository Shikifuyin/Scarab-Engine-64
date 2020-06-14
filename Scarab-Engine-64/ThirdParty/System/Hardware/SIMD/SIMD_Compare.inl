/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Compare.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Compare operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Compare implementation
__forceinline __m128 SIMD::Compare::Equal( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpeq_ps( mDst, mSrc );
}

__forceinline __m128d SIMD::Compare::Equal( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_pd( mDst, mSrc );
}

__forceinline __m128i SIMD::Compare::Equal8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_epi8( mDst, mSrc );
}
__forceinline __m128i SIMD::Compare::Equal16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_epi16( mDst, mSrc );
}
__forceinline __m128i SIMD::Compare::Equal32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_epi32( mDst, mSrc );
}
__forceinline __m128i SIMD::Compare::Equal64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cmpeq_epi64( mDst, mSrc );
}

__forceinline __m256 SIMD::Compare::Equal( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, _CMP_EQ_OS );
}

__forceinline __m256d SIMD::Compare::Equal( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, _CMP_EQ_OS );
}

__forceinline __m256i SIMD::Compare::Equal8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpeq_epi8( mDst, mSrc );
}
__forceinline __m256i SIMD::Compare::Equal16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpeq_epi16( mDst, mSrc );
}
__forceinline __m256i SIMD::Compare::Equal32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpeq_epi32( mDst, mSrc );
}
__forceinline __m256i SIMD::Compare::Equal64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpeq_epi64( mDst, mSrc );
}

__forceinline __m128 SIMD::Compare::NotEqual( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpneq_ps( mDst, mSrc );
}

__forceinline __m128d SIMD::Compare::NotEqual( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpneq_pd( mDst, mSrc );
}

__forceinline __m256 SIMD::Compare::NotEqual( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, _CMP_NEQ_OS );
}

__forceinline __m256d SIMD::Compare::NotEqual( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, _CMP_NEQ_OS );
}

__forceinline __m128 SIMD::Compare::Lesser( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmplt_ps( mDst, mSrc );
}

__forceinline __m128d SIMD::Compare::Lesser( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_pd( mDst, mSrc );
}

__forceinline __m128i SIMD::Compare::Lesser8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_epi8( mDst, mSrc );
}
__forceinline __m128i SIMD::Compare::Lesser16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_epi16( mDst, mSrc );
}
__forceinline __m128i SIMD::Compare::Lesser32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_epi32( mDst, mSrc );
}

__forceinline __m256 SIMD::Compare::Lesser( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, _CMP_LT_OS );
}

__forceinline __m256d SIMD::Compare::Lesser( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, _CMP_LT_OS );
}

__forceinline __m128 SIMD::Compare::NotLesser( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnlt_ps( mDst, mSrc );
}

__forceinline __m128d SIMD::Compare::NotLesser( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnlt_pd( mDst, mSrc );
}

__forceinline __m256 SIMD::Compare::NotLesser( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, _CMP_NLT_US );
}

__forceinline __m256d SIMD::Compare::NotLesser( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, _CMP_NLT_US );
}

__forceinline __m128 SIMD::Compare::LesserEqual( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmple_ps( mDst, mSrc );
}

__forceinline __m128d SIMD::Compare::LesserEqual( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmple_pd( mDst, mSrc );
}

__forceinline __m256 SIMD::Compare::LesserEqual( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, _CMP_LE_OS );
}

__forceinline __m256d SIMD::Compare::LesserEqual( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, _CMP_LE_OS );
}

__forceinline __m128 SIMD::Compare::NotLesserEqual( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnle_ps( mDst, mSrc );
}

__forceinline __m128d SIMD::Compare::NotLesserEqual( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnle_pd( mDst, mSrc );
}

__forceinline __m256 SIMD::Compare::NotLesserEqual( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, _CMP_NLE_US );
}

__forceinline __m256d SIMD::Compare::NotLesserEqual( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, _CMP_NLE_US );
}

__forceinline __m128 SIMD::Compare::Greater( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpgt_ps( mDst, mSrc );
}

__forceinline __m128d SIMD::Compare::Greater( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_pd( mDst, mSrc );
}

__forceinline __m128i SIMD::Compare::Greater8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_epi8( mDst, mSrc );
}
__forceinline __m128i SIMD::Compare::Greater16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_epi16( mDst, mSrc );
}
__forceinline __m128i SIMD::Compare::Greater32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_epi32( mDst, mSrc );
}
__forceinline __m128i SIMD::Compare::Greater64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cmpgt_epi64( mDst, mSrc );
}

__forceinline __m256 SIMD::Compare::Greater( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, _CMP_GT_OS );
}

__forceinline __m256d SIMD::Compare::Greater( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, _CMP_GT_OS );
}

__forceinline __m256i SIMD::Compare::Greater8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpgt_epi8( mDst, mSrc );
}
__forceinline __m256i SIMD::Compare::Greater16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpgt_epi16( mDst, mSrc );
}
__forceinline __m256i SIMD::Compare::Greater32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpgt_epi32( mDst, mSrc );
}
__forceinline __m256i SIMD::Compare::Greater64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpgt_epi64( mDst, mSrc );
}

__forceinline __m128 SIMD::Compare::NotGreater( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpngt_ps( mDst, mSrc );
}

__forceinline __m128d SIMD::Compare::NotGreater( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpngt_pd( mDst, mSrc );
}

__forceinline __m256 SIMD::Compare::NotGreater( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, _CMP_NGT_US );
}

__forceinline __m256d SIMD::Compare::NotGreater( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, _CMP_NGT_US );
}

__forceinline __m128 SIMD::Compare::GreaterEqual( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpge_ps( mDst, mSrc );
}

__forceinline __m128d SIMD::Compare::GreaterEqual( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpge_pd( mDst, mSrc );
}

__forceinline __m256 SIMD::Compare::GreaterEqual( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, _CMP_GE_OS );
}

__forceinline __m256d SIMD::Compare::GreaterEqual( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, _CMP_GE_OS );
}

__forceinline __m128 SIMD::Compare::NotGreaterEqual( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnge_ps( mDst, mSrc );
}

__forceinline __m128d SIMD::Compare::NotGreaterEqual( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnge_pd( mDst, mSrc );
}

__forceinline __m256 SIMD::Compare::NotGreaterEqual( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, _CMP_NGE_US );
}

__forceinline __m256d SIMD::Compare::NotGreaterEqual( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, _CMP_NGE_US );
}

__forceinline __m128 SIMD::Compare::Ordered( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpord_ps( mDst, mSrc );
}

__forceinline __m128d SIMD::Compare::Ordered( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpord_pd( mDst, mSrc );
}

__forceinline __m256 SIMD::Compare::Ordered( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, _CMP_ORD_S );
}

__forceinline __m256d SIMD::Compare::Ordered( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, _CMP_ORD_S );
}

__forceinline __m128 SIMD::Compare::Unordered( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpunord_ps( mDst, mSrc );
}

__forceinline __m128d SIMD::Compare::Unordered( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpunord_pd( mDst, mSrc );
}

__forceinline __m256 SIMD::Compare::Unordered( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, _CMP_UNORD_S );
}

__forceinline __m256d SIMD::Compare::Unordered( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, _CMP_UNORD_S );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Compare::One implementation
__forceinline __m128 SIMD::Compare::One::Equal( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpeq_ss( mDst, mSrc );
}
__forceinline __m128d SIMD::Compare::One::Equal( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_sd( mDst, mSrc );
}

__forceinline __m128 SIMD::Compare::One::NotEqual( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpneq_ss( mDst, mSrc );
}
__forceinline __m128d SIMD::Compare::One::NotEqual( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpneq_sd( mDst, mSrc );
}

__forceinline __m128 SIMD::Compare::One::Lesser( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmplt_ss( mDst, mSrc );
}
__forceinline __m128d SIMD::Compare::One::Lesser( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_sd( mDst, mSrc );
}

__forceinline __m128 SIMD::Compare::One::NotLesser( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnlt_ss( mDst, mSrc );
}
__forceinline __m128d SIMD::Compare::One::NotLesser( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnlt_sd( mDst, mSrc );
}

__forceinline __m128 SIMD::Compare::One::LesserEqual( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmple_ss( mDst, mSrc );
}
__forceinline __m128d SIMD::Compare::One::LesserEqual( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmple_sd( mDst, mSrc );
}

__forceinline __m128 SIMD::Compare::One::NotLesserEqual( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnle_ss( mDst, mSrc );
}
__forceinline __m128d SIMD::Compare::One::NotLesserEqual( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnle_sd( mDst, mSrc );
}

__forceinline __m128 SIMD::Compare::One::Greater( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpgt_ss( mDst, mSrc );
}
__forceinline __m128d SIMD::Compare::One::Greater( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_sd( mDst, mSrc );
}

__forceinline __m128 SIMD::Compare::One::NotGreater( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpngt_ss( mDst, mSrc );
}
__forceinline __m128d SIMD::Compare::One::NotGreater( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpngt_sd( mDst, mSrc );
}

__forceinline __m128 SIMD::Compare::One::GreaterEqual( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpge_ss( mDst, mSrc );
}
__forceinline __m128d SIMD::Compare::One::GreaterEqual( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpge_sd( mDst, mSrc );
}

__forceinline __m128 SIMD::Compare::One::NotGreaterEqual( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnge_ss( mDst, mSrc );
}
__forceinline __m128d SIMD::Compare::One::NotGreaterEqual( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnge_sd( mDst, mSrc );
}

__forceinline __m128 SIMD::Compare::One::Ordered( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpord_ss( mDst, mSrc );
}
__forceinline __m128d SIMD::Compare::One::Ordered( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpord_sd( mDst, mSrc );
}

__forceinline __m128 SIMD::Compare::One::Unordered( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpunord_ss( mDst, mSrc );
}
__forceinline __m128d SIMD::Compare::One::Unordered( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpunord_sd( mDst, mSrc );
}

__forceinline Int SIMD::Compare::One::IsEqual( __m128 mLHS, __m128 mRHS ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comieq_ss( mLHS, mRHS );
}
__forceinline Int SIMD::Compare::One::IsEqual( __m128d mLHS, __m128d mRHS ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comieq_sd( mLHS, mRHS );
}

__forceinline Int SIMD::Compare::One::IsNotEqual( __m128 mLHS, __m128 mRHS ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comineq_ss( mLHS, mRHS );
}
__forceinline Int SIMD::Compare::One::IsNotEqual( __m128d mLHS, __m128d mRHS ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comineq_sd( mLHS, mRHS );
}

__forceinline Int SIMD::Compare::One::IsLesser( __m128 mLHS, __m128 mRHS ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comilt_ss( mLHS, mRHS );
}
__forceinline Int SIMD::Compare::One::IsLesser( __m128d mLHS, __m128d mRHS ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comilt_sd( mLHS, mRHS );
}

__forceinline Int SIMD::Compare::One::IsLesserEqual( __m128 mLHS, __m128 mRHS ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comile_ss( mLHS, mRHS );
}
__forceinline Int SIMD::Compare::One::IsLesserEqual( __m128d mLHS, __m128d mRHS ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comile_sd( mLHS, mRHS );
}

__forceinline Int SIMD::Compare::One::IsGreater( __m128 mLHS, __m128 mRHS ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comigt_ss( mLHS, mRHS );
}
__forceinline Int SIMD::Compare::One::IsGreater( __m128d mLHS, __m128d mRHS ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comigt_sd( mLHS, mRHS );
}

__forceinline Int SIMD::Compare::One::IsGreaterEqual( __m128 mLHS, __m128 mRHS ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comige_ss( mLHS, mRHS );
}
__forceinline Int SIMD::Compare::One::IsGreaterEqual( __m128d mLHS, __m128d mRHS ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comige_sd( mLHS, mRHS );
}

__forceinline Int SIMD::Compare::One::IsEqualQ( __m128 mLHS, __m128 mRHS ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomieq_ss( mLHS, mRHS );
}
__forceinline Int SIMD::Compare::One::IsEqualQ( __m128d mLHS, __m128d mRHS ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomieq_sd( mLHS, mRHS );
}

__forceinline Int SIMD::Compare::One::IsNotEqualQ( __m128 mLHS, __m128 mRHS ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomineq_ss( mLHS, mRHS );
}
__forceinline Int SIMD::Compare::One::IsNotEqualQ( __m128d mLHS, __m128d mRHS ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomineq_sd( mLHS, mRHS );
}

__forceinline Int SIMD::Compare::One::IsLesserQ( __m128 mLHS, __m128 mRHS ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomilt_ss( mLHS, mRHS );
}
__forceinline Int SIMD::Compare::One::IsLesserQ( __m128d mLHS, __m128d mRHS ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomilt_sd( mLHS, mRHS );
}

__forceinline Int SIMD::Compare::One::IsLesserEqualQ( __m128 mLHS, __m128 mRHS ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomile_ss( mLHS, mRHS );
}
__forceinline Int SIMD::Compare::One::IsLesserEqualQ( __m128d mLHS, __m128d mRHS ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomile_sd( mLHS, mRHS );
}

__forceinline Int SIMD::Compare::One::IsGreaterQ( __m128 mLHS, __m128 mRHS ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomigt_ss( mLHS, mRHS );
}
__forceinline Int SIMD::Compare::One::IsGreaterQ( __m128d mLHS, __m128d mRHS ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomigt_sd( mLHS, mRHS );
}

__forceinline Int SIMD::Compare::One::IsGreaterEqualQ( __m128 mLHS, __m128 mRHS ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomige_ss( mLHS, mRHS );
}
__forceinline Int SIMD::Compare::One::IsGreaterEqualQ( __m128d mLHS, __m128d mRHS ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomige_sd( mLHS, mRHS );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Compare::String implementation

