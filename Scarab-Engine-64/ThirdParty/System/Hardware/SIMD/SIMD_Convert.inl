/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Convert.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Convert operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Convert implementation
inline __m128 SIMD::Convert::OneToFloat( __m128 mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsd_ss( mDst, mSrc );
}

inline __m128 SIMD::Convert::OneToFloat( __m128 mDst, Int32 iSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtsi32_ss( mDst, iSrc );
}
inline __m128 SIMD::Convert::OneToFloat( __m128 mDst, Int64 iSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtsi64_ss( mDst, iSrc );
}

inline __m128d SIMD::Convert::OneToDouble( __m128d mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtss_sd( mDst, mSrc );
}

inline __m128d SIMD::Convert::OneToDouble( __m128d mDst, Int32 iSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi32_sd( mDst, iSrc );
}
inline __m128d SIMD::Convert::OneToDouble( __m128d mDst, Int64 iSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi64_sd( mDst, iSrc );
}

inline Int32 SIMD::Convert::OneToInt32( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtss_si32( mSrc );
}
inline Int32 SIMD::Convert::OneToInt32( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsd_si32( mSrc );
}

inline Int64 SIMD::Convert::OneToInt64( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtss_si64( mSrc );
}
inline Int64 SIMD::Convert::OneToInt64( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsd_si64( mSrc );
}

inline __m128 SIMD::Convert::ToFloat128( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtpd_ps( mSrc );
}
inline __m128 SIMD::Convert::ToFloat128( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtepi32_ps( mSrc );
}

inline __m128 SIMD::Convert::ToFloat128( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtpd_ps( mSrc );
}
inline __m256 SIMD::Convert::ToFloat256( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtepi32_ps( mSrc );
}

inline __m128d SIMD::Convert::ToDouble128( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtps_pd( mSrc );
}
inline __m128d SIMD::Convert::ToDouble128( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtepi32_pd( mSrc );
}

inline __m256d SIMD::Convert::ToDouble256( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtps_pd( mSrc );
}
inline __m256d SIMD::Convert::ToDouble256( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtepi32_pd( mSrc );
}

inline __m128i SIMD::Convert::ToInt32( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtps_epi32( mSrc );
}
inline __m128i SIMD::Convert::ToInt32( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtpd_epi32( mSrc );
}

inline __m256i SIMD::Convert::ToInt32( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtps_epi32( mSrc );
}
inline __m128i SIMD::Convert::ToInt32( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtpd_epi32( mSrc );
}


/////////////////////////////////////////////////////////////////////////////////
// SIMD::Truncate implementation
inline Int32 SIMD::Truncate::OneToInt32( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvttss_si32( mSrc );
}
inline Int32 SIMD::Truncate::OneToInt32( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvttsd_si32( mSrc );
}

inline Int64 SIMD::Truncate::OneToInt64( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvttss_si64( mSrc );
}
inline Int64 SIMD::Truncate::OneToInt64( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvttsd_si64( mSrc );
}

inline __m128i SIMD::Truncate::ToInt32( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvttps_epi32( mSrc );
}
inline __m128i SIMD::Truncate::ToInt32( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvttpd_epi32( mSrc );
}

inline __m256i SIMD::Truncate::ToInt32( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvttps_epi32( mSrc );
}
inline __m128i SIMD::Truncate::ToInt32( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvttpd_epi32( mSrc );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::SignExtend128 implementation
inline __m128i SIMD::SignExtend128::Int8To16( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi8_epi16( mSrc );
}
inline __m128i SIMD::SignExtend128::Int8To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi8_epi32( mSrc );
}
inline __m128i SIMD::SignExtend128::Int8To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi8_epi64( mSrc );
}
inline __m128i SIMD::SignExtend128::Int16To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi16_epi32( mSrc );
}
inline __m128i SIMD::SignExtend128::Int16To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi16_epi64( mSrc );
}
inline __m128i SIMD::SignExtend128::Int32To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi32_epi64( mSrc );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::SignExted256 implementation
inline __m256i SIMD::SignExtend256::Int8To16( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi8_epi16( mSrc );
}
inline __m256i SIMD::SignExtend256::Int8To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi8_epi32( mSrc );
}
inline __m256i SIMD::SignExtend256::Int8To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi8_epi64( mSrc );
}
inline __m256i SIMD::SignExtend256::Int16To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi16_epi32( mSrc );
}
inline __m256i SIMD::SignExtend256::Int16To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi16_epi64( mSrc );
}
inline __m256i SIMD::SignExtend256::Int32To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi32_epi64( mSrc );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::ZeroExtend128 implementation
inline __m128i SIMD::ZeroExtend128::Int8To16( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu8_epi16( mSrc );
}
inline __m128i SIMD::ZeroExtend128::Int8To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu8_epi32( mSrc );
}
inline __m128i SIMD::ZeroExtend128::Int8To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu8_epi64( mSrc );
}
inline __m128i SIMD::ZeroExtend128::Int16To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu16_epi32( mSrc );
}
inline __m128i SIMD::ZeroExtend128::Int16To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu16_epi64( mSrc );
}
inline __m128i SIMD::ZeroExtend128::Int32To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu32_epi64( mSrc );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::ZeroExtend256 implementation
inline __m256i SIMD::ZeroExtend256::Int8To16( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu8_epi16( mSrc );
}
inline __m256i SIMD::ZeroExtend256::Int8To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu8_epi32( mSrc );
}
inline __m256i SIMD::ZeroExtend256::Int8To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu8_epi64( mSrc );
}
inline __m256i SIMD::ZeroExtend256::Int16To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu16_epi32( mSrc );
}
inline __m256i SIMD::ZeroExtend256::Int16To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu16_epi64( mSrc );
}
inline __m256i SIMD::ZeroExtend256::Int32To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu32_epi64( mSrc );
}
