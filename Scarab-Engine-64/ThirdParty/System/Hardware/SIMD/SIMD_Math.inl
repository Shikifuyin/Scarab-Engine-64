/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Math.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Math operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Math implementation
inline __m128 SIMD::Math::FloorOne( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_floor_ss( mDst, mSrc );
}
inline __m128d SIMD::Math::FloorOne( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_floor_sd( mDst, mSrc );
}

inline __m128 SIMD::Math::Floor( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_floor_ps( mValue );
}
inline __m128d SIMD::Math::Floor( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_floor_pd( mValue );
}

inline __m256 SIMD::Math::Floor( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_floor_ps( mValue );
}
inline __m256d SIMD::Math::Floor( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_floor_pd( mValue );
}

inline __m128 SIMD::Math::CeilOne( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_ceil_ss( mDst, mSrc );
}
inline __m128d SIMD::Math::CeilOne( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_ceil_sd( mDst, mSrc );
}

inline __m128 SIMD::Math::Ceil( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_ceil_ps( mValue );
}
inline __m128d SIMD::Math::Ceil( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_ceil_pd( mValue );
}

inline __m256 SIMD::Math::Ceil( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_ceil_ps( mValue );
}
inline __m256d SIMD::Math::Ceil( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_ceil_pd( mValue );
}

inline __m128 SIMD::Math::RoundOne( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_round_ss( mDst, mSrc, _MM_FROUND_NINT );
}
inline __m128d SIMD::Math::RoundOne( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_round_sd( mDst, mSrc, _MM_FROUND_NINT );
}

inline __m128 SIMD::Math::Round( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_round_ps( mValue, _MM_FROUND_NINT );
}
inline __m128d SIMD::Math::Round( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_round_pd( mValue, _MM_FROUND_NINT );
}

inline __m256 SIMD::Math::Round( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_round_ps( mValue, _MM_FROUND_NINT );
}
inline __m256d SIMD::Math::Round( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_round_pd( mValue, _MM_FROUND_NINT );
}

inline __m128 SIMD::Math::AddOne( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_add_ss( mDst, mSrc );
}
inline __m128d SIMD::Math::AddOne( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_sd( mDst, mSrc );
}

inline __m128 SIMD::Math::Add( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_add_ps( mDst, mSrc );
}
inline __m128d SIMD::Math::Add( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_pd( mDst, mSrc );
}

inline __m256 SIMD::Math::Add( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_add_ps( mDst, mSrc );
}
inline __m256d SIMD::Math::Add( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_add_pd( mDst, mSrc );
}

inline __m128 SIMD::Math::HAdd( __m128 mSrc1, __m128 mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_hadd_ps( mSrc1, mSrc2 );
}
inline __m128d SIMD::Math::HAdd( __m128d mSrc1, __m128d mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_hadd_pd( mSrc1, mSrc2 );
}

inline __m256 SIMD::Math::HAdd( __m256 mSrc1, __m256 mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hadd_ps( mSrc1, mSrc2 );
}
inline __m256d SIMD::Math::HAdd( __m256d mSrc1, __m256d mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hadd_pd( mSrc1, mSrc2 );
}

inline __m128 SIMD::Math::SubOne( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sub_ss( mDst, mSrc );
}
inline __m128d SIMD::Math::SubOne( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_sd( mDst, mSrc );
}

inline __m128 SIMD::Math::Sub( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sub_ps( mDst, mSrc );
}
inline __m128d SIMD::Math::Sub( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_pd( mDst, mSrc );
}

inline __m256 SIMD::Math::Sub( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sub_ps( mDst, mSrc );
}
inline __m256d SIMD::Math::Sub( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sub_pd( mDst, mSrc );
}

inline __m128 SIMD::Math::HSub( __m128 mSrc1, __m128 mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_hsub_ps( mSrc1, mSrc2 );
}
inline __m128d SIMD::Math::HSub( __m128d mSrc1, __m128d mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_hsub_pd( mSrc1, mSrc2 );
}

inline __m256 SIMD::Math::HSub( __m256 mSrc1, __m256 mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hsub_ps( mSrc1, mSrc2 );
}
inline __m256d SIMD::Math::HSub( __m256d mSrc1, __m256d mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hsub_pd( mSrc1, mSrc2 );
}

inline __m128 SIMD::Math::AddSub( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_addsub_ps( mDst, mSrc );
}
inline __m128d SIMD::Math::AddSub( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_addsub_pd( mDst, mSrc );
}

inline __m256 SIMD::Math::AddSub( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_addsub_ps( mDst, mSrc );
}
inline __m256d SIMD::Math::AddSub( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_addsub_pd( mDst, mSrc );
}

inline __m128 SIMD::Math::MulOne( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_mul_ss( mDst, mSrc );
}
inline __m128d SIMD::Math::MulOne( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mul_sd( mDst, mSrc );
}

inline __m128 SIMD::Math::Mul( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_mul_ps( mDst, mSrc );
}
inline __m128d SIMD::Math::Mul( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mul_pd( mDst, mSrc );
}

inline __m256 SIMD::Math::Mul( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_mul_ps( mDst, mSrc );
}
inline __m256d SIMD::Math::Mul( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_mul_pd( mDst, mSrc );
}


inline __m128 SIMD::Math::Dot2( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_dp_ps( mDst, mSrc, SIMD_DOTP_MASK_4(1,1,0,0,1,0,0,0) );
}
inline __m128 SIMD::Math::Dot3( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_dp_ps( mDst, mSrc, SIMD_DOTP_MASK_4(1,1,1,0,1,0,0,0) );
}
inline __m128 SIMD::Math::Dot4( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_dp_ps( mDst, mSrc, SIMD_DOTP_MASK_4(1,1,1,1,1,0,0,0) );
}

inline __m128d SIMD::Math::Dot2( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_dp_pd( mDst, mSrc, SIMD_DOTP_MASK_2(1,1,1,0) );
}

inline __m256 SIMD::Math::Dot2( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_dp_ps( mDst, mSrc, SIMD_DOTP_MASK_4(1,1,0,0,1,0,0,0) );
}
inline __m256 SIMD::Math::Dot3( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_dp_ps( mDst, mSrc, SIMD_DOTP_MASK_4(1,1,1,0,1,0,0,0) );
}
inline __m256 SIMD::Math::Dot4( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_dp_ps( mDst, mSrc, SIMD_DOTP_MASK_4(1,1,1,1,1,0,0,0) );
}

//inline __m128 SIMD::Math::DotP( __m128 mDst, __m128 mSrc, Int iMask4 ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_dp_ps( mDst, mSrc, iMask4 );
//}
//inline __m256 SIMD::Math::DotP( __m256 mDst, __m256 mSrc, Int iMask4 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_dp_ps( mDst, mSrc, iMask4 );
//}

//inline __m128d SIMD::Math::DotP( __m128d mDst, __m128d mSrc, Int iMask2 ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_dp_pd( mDst, mSrc, iMask2 );
//}

inline __m128 SIMD::Math::DivOne( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_div_ss( mDst, mSrc );
}
inline __m128d SIMD::Math::DivOne( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_sd( mDst, mSrc );
}

inline __m128 SIMD::Math::Div( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_div_ps( mDst, mSrc );
}
inline __m128d SIMD::Math::Div( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_pd( mDst, mSrc );
}

inline __m256 SIMD::Math::Div( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_ps( mDst, mSrc );
}
inline __m256d SIMD::Math::Div( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_pd( mDst, mSrc );
}

inline __m128 SIMD::Math::MinOne( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_min_ss( mDst, mSrc );
}
inline __m128d SIMD::Math::MinOne( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_min_sd( mDst, mSrc );
}

inline __m128 SIMD::Math::Min( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_min_ps( mDst, mSrc );
}
inline __m128d SIMD::Math::Min( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_min_pd( mDst, mSrc );
}

inline __m256 SIMD::Math::Min( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_min_ps( mDst, mSrc );
}
inline __m256d SIMD::Math::Min( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_min_pd( mDst, mSrc );
}

inline __m128 SIMD::Math::MaxOne( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_max_ss( mDst, mSrc );
}
inline __m128d SIMD::Math::MaxOne( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_max_sd( mDst, mSrc );
}

inline __m128 SIMD::Math::Max( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_max_ps( mDst, mSrc );
}
inline __m128d SIMD::Math::Max( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_max_pd( mDst, mSrc );
}

inline __m256 SIMD::Math::Max( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_max_ps( mDst, mSrc );
}
inline __m256d SIMD::Math::Max( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_max_pd( mDst, mSrc );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Math::Signed implementation
inline __m128i SIMD::Math::Signed::Abs8( __m128i mValue ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_abs_epi8( mValue );
}
inline __m128i SIMD::Math::Signed::Abs16( __m128i mValue ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_abs_epi16( mValue );
}
inline __m128i SIMD::Math::Signed::Abs32( __m128i mValue ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_abs_epi32( mValue );
}
inline __m128i SIMD::Math::Signed::Abs64( __m128i mValue ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_abs_epi64( mValue );
}

inline __m256i SIMD::Math::Signed::Abs8( __m256i mValue ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_abs_epi8( mValue );
}
inline __m256i SIMD::Math::Signed::Abs16( __m256i mValue ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_abs_epi16( mValue );
}
inline __m256i SIMD::Math::Signed::Abs32( __m256i mValue ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_abs_epi32( mValue );
}
inline __m256i SIMD::Math::Signed::Abs64( __m256i mValue ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_abs_epi64( mValue );
}

inline __m128i SIMD::Math::Signed::Negate8( __m128i mValue, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_sign_epi8( mValue, mSigns );
}
inline __m128i SIMD::Math::Signed::Negate16( __m128i mValue, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_sign_epi16( mValue, mSigns );
}
inline __m128i SIMD::Math::Signed::Negate32( __m128i mValue, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_sign_epi32( mValue, mSigns );
}

inline __m256i SIMD::Math::Signed::Negate8( __m256i mValue, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sign_epi8( mValue, mSigns );
}
inline __m256i SIMD::Math::Signed::Negate16( __m256i mValue, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sign_epi16( mValue, mSigns );
}
inline __m256i SIMD::Math::Signed::Negate32( __m256i mValue, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sign_epi32( mValue, mSigns );
}

inline __m128i SIMD::Math::Signed::Add8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_epi8( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Add16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_epi16( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Add32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_epi32( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Add64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_epi64( mDst, mSrc );
}

inline __m256i SIMD::Math::Signed::Add8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_add_epi8( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Add16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_add_epi16( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Add32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_add_epi32( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Add64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_add_epi64( mDst, mSrc );
}

inline __m128i SIMD::Math::Signed::AddSat8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_adds_epi8( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::AddSat16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_adds_epi16( mDst, mSrc );
}

inline __m256i SIMD::Math::Signed::AddSat8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_adds_epi8( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::AddSat16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_adds_epi16( mDst, mSrc );
}

inline __m128i SIMD::Math::Signed::HAdd16( __m128i mSrc1, __m128i mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hadd_epi16( mSrc1, mSrc2 );
}
inline __m128i SIMD::Math::Signed::HAdd32( __m128i mSrc1, __m128i mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hadd_epi32( mSrc1, mSrc2 );
}

inline __m256i SIMD::Math::Signed::HAdd16( __m256i mSrc1, __m256i mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hadd_epi16( mSrc1, mSrc2 );
}
inline __m256i SIMD::Math::Signed::HAdd32( __m256i mSrc1, __m256i mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hadd_epi32( mSrc1, mSrc2 );
}

inline __m128i SIMD::Math::Signed::HAddSat16( __m128i mSrc1, __m128i mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hadds_epi16( mSrc1, mSrc2 );
}

inline __m256i SIMD::Math::Signed::HAddSat16( __m256i mSrc1, __m256i mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hadds_epi16( mSrc1, mSrc2 );
}

inline __m128i SIMD::Math::Signed::Sub8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_epi8( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Sub16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_epi16( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Sub32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_epi32( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Sub64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_epi64( mDst, mSrc );
}

inline __m256i SIMD::Math::Signed::Sub8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sub_epi8( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Sub16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sub_epi16( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Sub32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sub_epi32( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Sub64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sub_epi64( mDst, mSrc );
}

inline __m128i SIMD::Math::Signed::SubSat8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_subs_epi8( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::SubSat16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_subs_epi16( mDst, mSrc );
}

inline __m256i SIMD::Math::Signed::SubSat8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_subs_epi8( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::SubSat16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_subs_epi16( mDst, mSrc );
}

inline __m128i SIMD::Math::Signed::HSub16( __m128i mSrc1, __m128i mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hsub_epi16( mSrc1, mSrc2 );
}
inline __m128i SIMD::Math::Signed::HSub32( __m128i mSrc1, __m128i mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hsub_epi32( mSrc1, mSrc2 );
}

inline __m256i SIMD::Math::Signed::HSub16( __m256i mSrc1, __m256i mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hsub_epi16( mSrc1, mSrc2 );
}
inline __m256i SIMD::Math::Signed::HSub32( __m256i mSrc1, __m256i mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hsub_epi32( mSrc1, mSrc2 );
}

inline __m128i SIMD::Math::Signed::HSubSat16( __m128i mSrc1, __m128i mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hsubs_epi16( mSrc1, mSrc2 );
}

inline __m256i SIMD::Math::Signed::HSubSat16( __m256i mSrc1, __m256i mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hsubs_epi16( mSrc1, mSrc2 );
}

inline __m128i SIMD::Math::Signed::Mul16L( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mullo_epi16( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Mul16H( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mulhi_epi16( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Mul32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_mul_epi32( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Mul32L( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_mullo_epi32( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Mul64L( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_mullo_epi64( mDst, mSrc );
}

inline __m256i SIMD::Math::Signed::Mul16L( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mullo_epi16( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Mul16H( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mulhi_epi16( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Mul32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mul_epi32( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Mul32L( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mullo_epi32( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Mul64L( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mullo_epi64( mDst, mSrc );
}

inline __m128i SIMD::Math::Signed::MAdd( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_madd_epi16( mDst, mSrc );
}

inline __m256i SIMD::Math::Signed::MAdd( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_madd_epi16( mDst, mSrc );
}

inline __m128i SIMD::Math::Signed::MAddUS( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_maddubs_epi16( mDst, mSrc );
}

inline __m256i SIMD::Math::Signed::MAddUS( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_maddubs_epi16( mDst, mSrc );
}

inline __m128i SIMD::Math::Signed::Div8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epi8( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Div16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epi16( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Div32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epi32( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Div64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epi64( mDst, mSrc );
}

inline __m256i SIMD::Math::Signed::Div8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epi8( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Div16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epi16( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Div32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epi32( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Div64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epi64( mDst, mSrc );
}

inline __m128i SIMD::Math::Signed::Mod8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epi8( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Mod16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epi16( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Mod32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epi32( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Mod64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epi64( mDst, mSrc );
}

inline __m256i SIMD::Math::Signed::Mod8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epi8( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Mod16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epi16( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Mod32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epi32( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Mod64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epi64( mDst, mSrc );
}

inline __m128i SIMD::Math::Signed::DivMod32( __m128i * outMod, __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_divrem_epi32( outMod, mDst, mSrc );
}

inline __m256i SIMD::Math::Signed::DivMod32( __m256i * outMod, __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_divrem_epi32( outMod, mDst, mSrc );
}

inline __m128i SIMD::Math::Signed::Min8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epi8( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Min16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_min_epi16( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Min32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epi32( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Min64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epi64( mDst, mSrc );
}

inline __m256i SIMD::Math::Signed::Min8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epi8( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Min16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epi16( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Min32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epi32( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Min64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epi64( mDst, mSrc );
}

inline __m128i SIMD::Math::Signed::Max8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epi8( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Max16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_max_epi16( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Max32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epi32( mDst, mSrc );
}
inline __m128i SIMD::Math::Signed::Max64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epi64( mDst, mSrc );
}

inline __m256i SIMD::Math::Signed::Max8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epi8( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Max16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epi16( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Max32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epi32( mDst, mSrc );
}
inline __m256i SIMD::Math::Signed::Max64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epi64( mDst, mSrc );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Math::Unsigned implementation
inline __m128i SIMD::Math::Unsigned::AddSat8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_adds_epu8( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::AddSat16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_adds_epu16( mDst, mSrc );
}

inline __m256i SIMD::Math::Unsigned::AddSat8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_adds_epu8( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::AddSat16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_adds_epu16( mDst, mSrc );
}

inline __m128i SIMD::Math::Unsigned::SubSat8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_subs_epu8( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::SubSat16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_subs_epu16( mDst, mSrc );
}

inline __m256i SIMD::Math::Unsigned::SubSat8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_subs_epu8( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::SubSat16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_subs_epu16( mDst, mSrc );
}

inline __m128i SIMD::Math::Unsigned::SAD( __m128i mSrc1, __m128i mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sad_epu8( mSrc1, mSrc2 );
}
//inline __m128i SIMD::Math::Unsigned::SAD( __m128i mSrc1, __m128i mSrc2, Int iMask ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_mpsadbw_epu8( mSrc1, mSrc2, iMask );
//}

inline __m256i SIMD::Math::Unsigned::SAD( __m256i mSrc1, __m256i mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sad_epu8( mSrc1, mSrc2 );
}
//inline __m256i SIMD::Math::Unsigned::SAD( __m256i mSrc1, __m256i mSrc2, Int iMask ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_mpsadbw_epu8( mSrc1, mSrc2, iMask );
//}

inline __m128i SIMD::Math::Unsigned::Mul16H( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mulhi_epu16( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::Mul32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mul_epu32( mDst, mSrc );
}

inline __m256i SIMD::Math::Unsigned::Mul16H( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mulhi_epu16( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::Mul32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mul_epu32( mDst, mSrc );
}

inline __m128i SIMD::Math::Unsigned::Div8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epu8( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::Div16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epu16( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::Div32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epu32( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::Div64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epu64( mDst, mSrc );
}

inline __m256i SIMD::Math::Unsigned::Div8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epu8( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::Div16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epu16( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::Div32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epu32( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::Div64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epu64( mDst, mSrc );
}

inline __m128i SIMD::Math::Unsigned::Mod8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epu8( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::Mod16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epu16( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::Mod32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epu32( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::Mod64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epu64( mDst, mSrc );
}

inline __m256i SIMD::Math::Unsigned::Mod8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epu8( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::Mod16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epu16( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::Mod32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epu32( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::Mod64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epu64( mDst, mSrc );
}

inline __m128i SIMD::Math::Unsigned::DivMod32( __m128i * outMod, __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_divrem_epu32( outMod, mDst, mSrc );
}

inline __m256i SIMD::Math::Unsigned::DivMod32( __m256i * outMod, __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_divrem_epu32( outMod, mDst, mSrc );
}

inline __m128i SIMD::Math::Unsigned::Avg8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_avg_epu8( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::Avg16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_avg_epu16( mDst, mSrc );
}

inline __m256i SIMD::Math::Unsigned::Avg8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_avg_epu8( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::Avg16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_avg_epu16( mDst, mSrc );
}

inline __m128i SIMD::Math::Unsigned::Min8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_min_epu8( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::Min16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epu16( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::Min32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epu32( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::Min64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epu64( mDst, mSrc );
}

inline __m256i SIMD::Math::Unsigned::Min8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epu8( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::Min16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epu16( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::Min32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epu32( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::Min64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epu64( mDst, mSrc );
}

inline __m128i SIMD::Math::Unsigned::Max8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_max_epu8( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::Max16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epu16( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::Max32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epu32( mDst, mSrc );
}
inline __m128i SIMD::Math::Unsigned::Max64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epu64( mDst, mSrc );
}

inline __m256i SIMD::Math::Unsigned::Max8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epu8( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::Max16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epu16( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::Max32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epu32( mDst, mSrc );
}
inline __m256i SIMD::Math::Unsigned::Max64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epu64( mDst, mSrc );
}

inline UInt32 SIMD::Math::Unsigned::CRC32( UInt32 iCRC, UInt8 iValue ) {
    DebugAssert( CPUIDFn->HasSSE42() );
    return _mm_crc32_u8( iCRC, iValue );
}
inline UInt32 SIMD::Math::Unsigned::CRC32( UInt32 iCRC, UInt16 iValue ) {
    DebugAssert( CPUIDFn->HasSSE42() );
    return _mm_crc32_u16( iCRC, iValue );
}
inline UInt32 SIMD::Math::Unsigned::CRC32( UInt32 iCRC, UInt32 iValue ) {
    DebugAssert( CPUIDFn->HasSSE42() );
    return _mm_crc32_u32( iCRC, iValue );
}
inline UInt64 SIMD::Math::Unsigned::CRC32( UInt64 iCRC, UInt64 iValue ) {
    DebugAssert( CPUIDFn->HasSSE42() );
    return _mm_crc32_u64( iCRC, iValue );
}
