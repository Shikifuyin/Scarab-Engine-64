/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Register.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Register operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register implementation

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Move implementation
inline __m128 SIMD::Register::Move::OneFloatLL( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_move_ss( mDst, mSrc );
}

inline __m128 SIMD::Register::Move::TwoFloatLH( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_movehl_ps( mDst, mSrc );
}
inline __m128 SIMD::Register::Move::TwoFloatHL( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_movelh_ps( mDst, mSrc );
}

inline __m256 SIMD::Register::Move::FourFloatL( __m256 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insertf128_ps( mDst, mSrc, 0 );
}
inline __m256 SIMD::Register::Move::FourFloatH( __m256 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insertf128_ps( mDst, mSrc, 1 );
}

inline __m128 SIMD::Register::Move::FourFloatL( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extractf128_ps( mSrc, 0 );
}
inline __m128 SIMD::Register::Move::FourFloatH( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extractf128_ps( mSrc, 1 );
}

inline __m128d SIMD::Register::Move::OneDoubleLL( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_move_sd( mDst, mSrc );
}

inline __m256d SIMD::Register::Move::TwoDoubleL( __m256d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insertf128_pd( mDst, mSrc, 0 );
}
inline __m256d SIMD::Register::Move::TwoDoubleH( __m256d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insertf128_pd( mDst, mSrc, 1 );
}

inline __m128d SIMD::Register::Move::TwoDoubleL( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extractf128_pd( mSrc, 0 );
}
inline __m128d SIMD::Register::Move::TwoDoubleH( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extractf128_pd( mSrc, 1 );
}

inline __m256i SIMD::Register::Move::FourInt32L( __m256i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_inserti128_si256( mDst, mSrc, 0 );
}
inline __m256i SIMD::Register::Move::FourInt32H( __m256i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_inserti128_si256( mDst, mSrc, 1 );
}

inline __m128i SIMD::Register::Move::FourInt32L( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_extracti128_si256( mSrc, 0 );
}
inline __m128i SIMD::Register::Move::FourInt32H( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_extracti128_si256( mSrc, 1 );
}

inline __m128i SIMD::Register::Move::OneInt64LL( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_move_epi64( mSrc );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Pack implementation
inline __m128i SIMD::Register::Pack::Int16To8( __m128i mSrcLow, __m128i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_packs_epi16( mSrcLow, mSrcHigh );
}
inline __m128i SIMD::Register::Pack::Int32To16( __m128i mSrcLow, __m128i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_packs_epi32( mSrcLow, mSrcHigh );
}

inline __m256i SIMD::Register::Pack::Int16To8( __m256i mSrcLow, __m256i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packs_epi16( mSrcLow, mSrcHigh );
}
inline __m256i SIMD::Register::Pack::Int32To16( __m256i mSrcLow, __m256i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packs_epi32( mSrcLow, mSrcHigh );
}

inline __m128i SIMD::Register::Pack::UInt16To8( __m128i mSrcLow, __m128i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_packus_epi16( mSrcLow, mSrcHigh );
}
inline __m128i SIMD::Register::Pack::UInt32To16( __m128i mSrcLow, __m128i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_packus_epi32( mSrcLow, mSrcHigh );
}

inline __m256i SIMD::Register::Pack::UInt16To8( __m256i mSrcLow, __m256i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packus_epi16( mSrcLow, mSrcHigh );
}
inline __m256i SIMD::Register::Pack::UInt32To16( __m256i mSrcLow, __m256i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packus_epi32( mSrcLow, mSrcHigh );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Unpack implementation
inline __m128 SIMD::Register::Unpack::UnpackFloatL( __m128 mSrcEven, __m128 mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_unpacklo_ps( mSrcEven, mSrcOdd );
}
inline __m128 SIMD::Register::Unpack::UnpackFloatH( __m128 mSrcEven, __m128 mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_unpackhi_ps( mSrcEven, mSrcOdd );
}

inline __m256 SIMD::Register::Unpack::UnpackFloatL( __m256 mSrcEven, __m256 mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpacklo_ps( mSrcEven, mSrcOdd );
}
inline __m256 SIMD::Register::Unpack::UnpackFloatH( __m256 mSrcEven, __m256 mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpackhi_ps( mSrcEven, mSrcOdd );
}

inline __m128d SIMD::Register::Unpack::UnpackDoubleL( __m128d mSrcEven, __m128d mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_pd( mSrcEven, mSrcOdd );
}
inline __m128d SIMD::Register::Unpack::UnpackDoubleH( __m128d mSrcEven, __m128d mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_pd( mSrcEven, mSrcOdd );
}

inline __m256d SIMD::Register::Unpack::UnpackDoubleL( __m256d mSrcEven, __m256d mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpacklo_pd( mSrcEven, mSrcOdd );
}
inline __m256d SIMD::Register::Unpack::UnpackDoubleH( __m256d mSrcEven, __m256d mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpackhi_pd( mSrcEven, mSrcOdd );
}

inline __m128i SIMD::Register::Unpack::UnpackInt8L( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi8( mSrcEven, mSrcOdd );
}
inline __m128i SIMD::Register::Unpack::UnpackInt8H( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi8( mSrcEven, mSrcOdd );
}
inline __m128i SIMD::Register::Unpack::UnpackInt16L( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi16( mSrcEven, mSrcOdd );
}
inline __m128i SIMD::Register::Unpack::UnpackInt16H( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi16( mSrcEven, mSrcOdd );
}
inline __m128i SIMD::Register::Unpack::UnpackInt32L( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi32( mSrcEven, mSrcOdd );
}
inline __m128i SIMD::Register::Unpack::UnpackInt32H( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi32( mSrcEven, mSrcOdd );
}
inline __m128i SIMD::Register::Unpack::UnpackInt64L( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi64( mSrcEven, mSrcOdd );
}
inline __m128i SIMD::Register::Unpack::UnpackInt64H( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi64( mSrcEven, mSrcOdd );
}

inline __m256i SIMD::Register::Unpack::UnpackInt8L( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi8( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::Register::Unpack::UnpackInt8H( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi8( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::Register::Unpack::UnpackInt16L( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi16( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::Register::Unpack::UnpackInt16H( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi16( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::Register::Unpack::UnpackInt32L( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi32( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::Register::Unpack::UnpackInt32H( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi32( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::Register::Unpack::UnpackInt64L( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi64( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::Register::Unpack::UnpackInt64H( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi64( mSrcEven, mSrcOdd );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Spread implementation
inline __m128 SIMD::Register::Spread::ABCD_AACC( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_moveldup_ps( mSrc );
}
inline __m128 SIMD::Register::Spread::ABCD_BBDD( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_movehdup_ps( mSrc );
}
inline __m128 SIMD::Register::Spread::ABCD_AAAA( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastss_ps( mSrc );
}

inline __m256 SIMD::Register::Spread::ABCDEFGH_AACCEEGG( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_moveldup_ps( mSrc );
}
inline __m256 SIMD::Register::Spread::ABCDEFGH_BBDDFFHH( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_movehdup_ps( mSrc );
}
inline __m256 SIMD::Register::Spread::ABCDEFGH_AAAAAAAA( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastss_ps( _mm256_castps256_ps128(mSrc) );
}

//inline __m128d SIMD::Register::Spread::AB_AA( __m128d mSrc ) {
//    DebugAssert( CPUIDFn->HasSSE3() );
//    return _mm_movedup_pd( mSrc );
//}
inline __m128d SIMD::Register::Spread::AB_AA( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastsd_pd( mSrc );
}
inline __m128d SIMD::Register::Spread::AB_BB( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_pd( mSrc, 0x03 );
}

inline __m256d SIMD::Register::Spread::ABCD_AACC( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_movedup_pd( mSrc );
}
inline __m256d SIMD::Register::Spread::ABCD_BBDD( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_pd( mSrc, 0xf5 );
}
inline __m256d SIMD::Register::Spread::ABCD_AAAA( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastsd_pd( _mm256_castpd256_pd128(mSrc) );
}

inline __m128i SIMD::Register::Spread::Int8( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastb_epi8( mSrc );
}
inline __m256i SIMD::Register::Spread::Int8( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastb_epi8( _mm256_castsi256_si128(mSrc) );
}

inline __m128i SIMD::Register::Spread::Int16( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastw_epi16( mSrc );
}
inline __m256i SIMD::Register::Spread::Int16( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastw_epi16( _mm256_castsi256_si128(mSrc) );
}

inline __m128i SIMD::Register::Spread::Int32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastd_epi32( mSrc );
}
inline __m256i SIMD::Register::Spread::Int32( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastd_epi32( _mm256_castsi256_si128(mSrc) );
}

inline __m128i SIMD::Register::Spread::Int64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastq_epi64( mSrc );
}
inline __m256i SIMD::Register::Spread::Int64( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastq_epi64( _mm256_castsi256_si128(mSrc) );
}

inline __m256i SIMD::Register::Spread::Int128( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastsi128_si256( _mm256_castsi256_si128(mSrc) );
}



