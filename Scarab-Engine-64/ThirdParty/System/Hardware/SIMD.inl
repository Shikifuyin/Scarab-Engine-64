/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD low level abstraction layer
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// SIMD implementation





////////////////////////////////////////////////////////////// Registers <-> Registers
inline __m128 SIMD::MoveOneFloatLL( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_move_ss( mDst, mSrc );
}

inline __m128 SIMD::MoveTwoFloatLH( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_movehl_ps( mDst, mSrc );
}
inline __m128 SIMD::MoveTwoFloatHL( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_movelh_ps( mDst, mSrc );
}

inline __m256 SIMD::MoveFourFloatL( __m256 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insertf128_ps( mDst, mSrc, 0 );
}
inline __m256 SIMD::MoveFourFloatH( __m256 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insertf128_ps( mDst, mSrc, 1 );
}

inline __m128d SIMD::MoveOneDoubleLL( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_move_sd( mDst, mSrc );
}

inline __m256d SIMD::MoveTwoDoubleL( __m256d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insertf128_pd( mDst, mSrc, 0 );
}
inline __m256d SIMD::MoveTwoDoubleH( __m256d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insertf128_pd( mDst, mSrc, 1 );
}

inline __m256i SIMD::MoveFourIntL( __m256i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_inserti128_si256( mDst, mSrc, 0 );
}
inline __m256i SIMD::MoveFourIntH( __m256i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_inserti128_si256( mDst, mSrc, 1 );
}

inline __m128 SIMD::MoveFourFloatL( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extractf128_ps( mSrc, 0 );
}
inline __m128 SIMD::MoveFourFloatH( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extractf128_ps( mSrc, 1 );
}

inline __m128d SIMD::MoveTwoDoubleL( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extractf128_pd( mSrc, 0 );
}
inline __m128d SIMD::MoveTwoDoubleH( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extractf128_pd( mSrc, 1 );
}

inline __m128i SIMD::MoveFourIntL( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_extracti128_si256( mSrc, 0 );
}
inline __m128i SIMD::MoveFourIntH( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_extracti128_si256( mSrc, 1 );
}

inline __m128i SIMD::MoveOneInt64LL( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_move_epi64( mSrc );
}

inline __m128 SIMD::SpreadTwoFloatEven( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_moveldup_ps( mSrc );
}
inline __m128 SIMD::SpreadTwoFloatOdd( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_movehdup_ps( mSrc );
}
inline __m128d SIMD::SpreadOneDoubleL( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_movedup_pd( mSrc );
}

inline __m256 SIMD::SpreadFourFloatEven( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_moveldup_ps( mSrc );
}
inline __m256 SIMD::SpreadFourFloatOdd( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_movehdup_ps( mSrc );
}
inline __m256d SIMD::SpreadTwoDoubleEven( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_movedup_pd( mSrc );
}

inline __m128 SIMD::Spread128Float( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastss_ps( mSrc );
}
inline __m256 SIMD::Spread256Float( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastss_ps( mSrc );
}

inline __m128d SIMD::Spread128Double( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastsd_pd( mSrc );
}
inline __m256d SIMD::Spread256Double( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastsd_pd( mSrc );
}

inline __m128i SIMD::Spread128Int8( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastb_epi8( mSrc );
}
inline __m256i SIMD::Spread256Int8( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastb_epi8( mSrc );
}

inline __m128i SIMD::Spread128Int16( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastw_epi16( mSrc );
}
inline __m256i SIMD::Spread256Int16( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastw_epi16( mSrc );
}

inline __m128i SIMD::Spread128Int32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastd_epi32( mSrc );
}
inline __m256i SIMD::Spread256Int32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastd_epi32( mSrc );
}

inline __m128i SIMD::Spread128Int64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastq_epi64( mSrc );
}
inline __m256i SIMD::Spread256Int64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastq_epi64( mSrc );
}

inline __m256i SIMD::Spread256Int128( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastsi128_si256( mSrc );
}

////////////////////////////////////////////////////////////// Pack / Unpack
inline __m128i SIMD::PackSigned16To8( __m128i mSrcLow, __m128i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_packs_epi16( mSrcLow, mSrcHigh );
}
inline __m256i SIMD::PackSigned16To8( __m256i mSrcLow, __m256i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packs_epi16( mSrcLow, mSrcHigh );
}

inline __m128i SIMD::PackSigned32To16( __m128i mSrcLow, __m128i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_packs_epi32( mSrcLow, mSrcHigh );
}
inline __m256i SIMD::PackSigned32To16( __m256i mSrcLow, __m256i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packs_epi32( mSrcLow, mSrcHigh );
}

inline __m128i SIMD::PackUnsigned16To8( __m128i mSrcLow, __m128i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_packus_epi16( mSrcLow, mSrcHigh );
}
inline __m256i SIMD::PackUnsigned16To8( __m256i mSrcLow, __m256i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packus_epi16( mSrcLow, mSrcHigh );
}

inline __m128i SIMD::PackUnsigned32To16( __m128i mSrcLow, __m128i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_packus_epi32( mSrcLow, mSrcHigh );
}
inline __m256i SIMD::PackUnsigned32To16( __m256i mSrcLow, __m256i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packus_epi32( mSrcLow, mSrcHigh );
}

inline __m128 SIMD::UnpackFloatL( __m128 mSrcEven, __m128 mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_unpacklo_ps( mSrcEven, mSrcOdd );
}
inline __m256 SIMD::UnpackFloatL( __m256 mSrcEven, __m256 mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpacklo_ps( mSrcEven, mSrcOdd );
}

inline __m128 SIMD::UnpackFloatH( __m128 mSrcEven, __m128 mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_unpackhi_ps( mSrcEven, mSrcOdd );
}
inline __m256 SIMD::UnpackFloatH( __m256 mSrcEven, __m256 mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpackhi_ps( mSrcEven, mSrcOdd );
}

inline __m128d SIMD::UnpackDoubleL( __m128d mSrcEven, __m128d mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_pd( mSrcEven, mSrcOdd );
}
inline __m256d SIMD::UnpackDoubleL( __m256d mSrcEven, __m256d mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpacklo_pd( mSrcEven, mSrcOdd );
}

inline __m128d SIMD::UnpackDoubleH( __m128d mSrcEven, __m128d mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_pd( mSrcEven, mSrcOdd );
}
inline __m256d SIMD::UnpackDoubleH( __m256d mSrcEven, __m256d mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpackhi_pd( mSrcEven, mSrcOdd );
}

inline __m128i SIMD::UnpackInt8L( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi8( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::UnpackInt8L( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi8( mSrcEven, mSrcOdd );
}

inline __m128i SIMD::UnpackInt8H( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi8( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::UnpackInt8H( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi8( mSrcEven, mSrcOdd );
}

inline __m128i SIMD::UnpackInt16L( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi16( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::UnpackInt16L( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi16( mSrcEven, mSrcOdd );
}

inline __m128i SIMD::UnpackInt16H( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi16( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::UnpackInt16H( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi16( mSrcEven, mSrcOdd );
}

inline __m128i SIMD::UnpackInt32L( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi32( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::UnpackInt32L( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi32( mSrcEven, mSrcOdd );
}

inline __m128i SIMD::UnpackInt32H( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi32( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::UnpackInt32H( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi32( mSrcEven, mSrcOdd );
}

inline __m128i SIMD::UnpackInt64L( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi64( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::UnpackInt64L( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi64( mSrcEven, mSrcOdd );
}

inline __m128i SIMD::UnpackInt64H( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi64( mSrcEven, mSrcOdd );
}
inline __m256i SIMD::UnpackInt64H( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi64( mSrcEven, mSrcOdd );
}

////////////////////////////////////////////////////////////// Shuffle
//inline __m128 SIMD::Shuffle128Float( __m128 mSrcLow, __m128 mSrcHigh, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasSSE() );
//    return _mm_shuffle_ps( mSrcLow, mSrcHigh, (unsigned)iMask4x4 );
//}
//inline __m128 SIMD::Shuffle128Float( __m128 mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm_permute_ps( mSrc, iMask4x4 );
//}
inline __m128 SIMD::Shuffle128Float( __m128 mSrc, __m128i mMask1x4 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permutevar_ps( mSrc, mMask1x4 );
}

//inline __m256 SIMD::Shuffle128Float( __m256 mSrcLow, __m256 mSrcHigh, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_shuffle_ps( mSrcLow, mSrcHigh, iMask4x4 );
//}
//inline __m256 SIMD::Shuffle128Float( __m256 mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_permute_ps( mSrc, iMask4x4 );
//}
inline __m256 SIMD::Shuffle128Float( __m256 mSrc, __m256i mMask1x4 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permutevar_ps( mSrc, mMask1x4 );
}

inline __m256 SIMD::Shuffle256Float( __m256 mSrc, __m256i mMask1x8 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permutevar8x32_ps( mSrc, mMask1x8 );
}

//inline __m256 SIMD::Shuffle512FourFloat( __m256 mSrc1, __m256 mSrc2, Int iMask2x4_Z ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_permute2f128_ps( mSrc1, mSrc2, iMask2x4_Z );
//}

//inline __m128d SIMD::Shuffle128Double( __m128d mSrcLow, __m128d mSrcHigh, Int iMask2x2 ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_shuffle_pd( mSrcLow, mSrcHigh, iMask2x2 );
//}
inline __m128d SIMD::Shuffle128Double( __m128d mSrc, Int iMask2x2 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_pd( mSrc, iMask2x2 );
}
inline __m128d SIMD::Shuffle128Double( __m128d mSrc, __m128i mMask1x2 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permutevar_pd( mSrc, mMask1x2 );
}

//inline __m256d SIMD::Shuffle128Double( __m256d mSrcLow, __m256d mSrcHigh, Int iMask4x2 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_shuffle_pd( mSrcLow, mSrcHigh, iMask4x2 );
//}
//inline __m256d SIMD::Shuffle128Double( __m256d mSrc, Int iMask4x2 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_permute_pd( mSrc, iMask4x2 );
//}
inline __m256d SIMD::Shuffle128Double( __m256d mSrc, __m256i mMask1x2 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permutevar_pd( mSrc, mMask1x2 );
}

//inline __m256d SIMD::Shuffle256Double( __m256d mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_permute4x64_pd( mSrc, iMask4x4 );
//}

//inline __m256d SIMD::Shuffle512TwoDouble( __m256d mSrc1, __m256d mSrc2, Int iMask2x4_Z ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_permute2f128_pd( mSrc1, mSrc2, iMask2x4_Z );
//}

inline __m128i SIMD::Shuffle128Int8( __m128i mSrc, __m128i mMask1x16_Z ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_shuffle_epi8( mSrc, mMask1x16_Z );
}
inline __m256i SIMD::Shuffle128Int8( __m256i mSrc, __m256i mMask1x16_Z ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi8( mSrc, mMask1x16_Z );
}

//inline __m128i SIMD::Shuffle64Int16L( __m128i mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_shufflelo_epi16( mSrc, iMask4x4 );
//}
//inline __m256i SIMD::Shuffle64Int16L( __m256i mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_shufflelo_epi16( mSrc, iMask4x4 );
//}

//inline __m128i SIMD::Shuffle64Int16H( __m128i mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_shufflehi_epi16( mSrc, iMask4x4 );
//}
//inline __m256i SIMD::Shuffle64Int16H( __m256i mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_shufflehi_epi16( mSrc, iMask4x4 );
//}

//inline __m128i SIMD::Shuffle128Int32( __m128i mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_shuffle_epi32( mSrc, iMask4x4 );
//}
//inline __m256i SIMD::Shuffle128Int32( __m256i mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_shuffle_epi32( mSrc, iMask4x4 );
//}

inline __m256i SIMD::Shuffle256Int32( __m256i mSrc, __m256i mMask1x8 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permutevar8x32_epi32( mSrc, mMask1x8 );
}

//inline __m256i SIMD::Shuffle512FourInt32( __m256i mSrc1, __m256i mSrc2, Int iMask2x4_Z ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_permute2x128_si256( mSrc1, mSrc2, iMask2x4_Z );
//}

//inline __m256i SIMD::Shuffle256Int64( __m256i mSrc, Int iMask4x4 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_permute4x64_epi64( mSrc, iMask4x4 );
//}

////////////////////////////////////////////////////////////// Blend
//inline __m128 SIMD::BlendFloat( __m128 mDst, __m128 mSrc, Int iMask4 ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_blend_ps( mDst, mSrc, iMask4 );
//}
inline __m128 SIMD::BlendFloat( __m128 mDst, __m128 mSrc, __m128 mSigns ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_blendv_ps( mDst, mSrc, mSigns );
}
//inline __m256 SIMD::BlendFloat( __m256 mDst, __m256 mSrc, Int iMask8 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_blend_ps( mDst, mSrc, iMask8 );
//}
inline __m256 SIMD::BlendFloat( __m256 mDst, __m256 mSrc, __m256 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_blendv_ps( mDst, mSrc, mSigns );
}

//inline __m128d SIMD::BlendDouble( __m128d mDst, __m128d mSrc, Int iMask2 ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_blend_pd( mDst, mSrc, iMask2 );
//}
inline __m128d SIMD::BlendDouble( __m128d mDst, __m128d mSrc, __m128d mSigns ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_blendv_pd( mDst, mSrc, mSigns );
}
//inline __m256d SIMD::BlendDouble( __m256d mDst, __m256d mSrc, Int iMask4 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_blend_pd( mDst, mSrc, iMask4 );
//}
inline __m256d SIMD::BlendDouble( __m256d mDst, __m256d mSrc, __m256d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_blendv_pd( mDst, mSrc, mSigns );
}

inline __m128i SIMD::BlendInt8( __m128i mDst, __m128i mSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_blendv_epi8( mDst, mSrc, mSigns );
}
inline __m256i SIMD::BlendInt8( __m256i mDst, __m256i mSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_blendv_epi8( mDst, mSrc, mSigns );
}

//inline __m128i SIMD::BlendInt16( __m128i mDst, __m128i mSrc, Int iMask8 ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_blend_epi16( mDst, mSrc, iMask8 );
//}
//inline __m256i SIMD::BlendInt16( __m256i mDst, __m256i mSrc, Int iMask8 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_blend_epi16( mDst, mSrc, iMask8 );
//}

//inline __m128i SIMD::BlendInt32( __m128i mDst, __m128i mSrc, Int iMask4 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_blend_epi32( mDst, mSrc, iMask4 );
//}
//inline __m256i SIMD::BlendInt32( __m256i mDst, __m256i mSrc, Int iMask8 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_blend_epi32( mDst, mSrc, iMask8 );
//}


/////////////////////////////////////////////////////////////////////////////////
// Macro Expansions for Immediate Parameters (not an optimal solution ...)
//#define _SIMD_ARGS( ... ) __VA_ARGS__
//
//#define _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _args, _argnames, _i ) \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##0( _args ) { return _funcname( _argnames, 0x##_i##0 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##1( _args ) { return _funcname( _argnames, 0x##_i##1 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##2( _args ) { return _funcname( _argnames, 0x##_i##2 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##3( _args ) { return _funcname( _argnames, 0x##_i##3 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##4( _args ) { return _funcname( _argnames, 0x##_i##4 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##5( _args ) { return _funcname( _argnames, 0x##_i##5 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##6( _args ) { return _funcname( _argnames, 0x##_i##6 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##7( _args ) { return _funcname( _argnames, 0x##_i##7 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##8( _args ) { return _funcname( _argnames, 0x##_i##8 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##9( _args ) { return _funcname( _argnames, 0x##_i##9 ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##a( _args ) { return _funcname( _argnames, 0x##_i##a ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##b( _args ) { return _funcname( _argnames, 0x##_i##b ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##c( _args ) { return _funcname( _argnames, 0x##_i##c ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##d( _args ) { return _funcname( _argnames, 0x##_i##d ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##e( _args ) { return _funcname( _argnames, 0x##_i##e ); } \
//    inline _rettype __fastcall _funcname##_imm##_0x##_i##f( _args ) { return _funcname( _argnames, 0x##_i##f ); }
//
//#define _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, _i ) \
//    _funcname##_imm##_0x##_i##0, \
//    _funcname##_imm##_0x##_i##1, \
//    _funcname##_imm##_0x##_i##2, \
//    _funcname##_imm##_0x##_i##3, \
//    _funcname##_imm##_0x##_i##4, \
//    _funcname##_imm##_0x##_i##5, \
//    _funcname##_imm##_0x##_i##6, \
//    _funcname##_imm##_0x##_i##7, \
//    _funcname##_imm##_0x##_i##8, \
//    _funcname##_imm##_0x##_i##9, \
//    _funcname##_imm##_0x##_i##a, \
//    _funcname##_imm##_0x##_i##b, \
//    _funcname##_imm##_0x##_i##c, \
//    _funcname##_imm##_0x##_i##d, \
//    _funcname##_imm##_0x##_i##e, \
//    _funcname##_imm##_0x##_i##f
//
//#define _SIMD_DECLARE_IMMEDIATE( _rettype, _funcname, _args, _argnames ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 0 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 1 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 2 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 3 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 4 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 5 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 6 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 7 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 8 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), 9 ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), a ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), b ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), c ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), d ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), e ) \
//    _SIMD_DECLARE_IMMEDIATE_EXPAND( _rettype, _funcname, _SIMD_ARGS(_args), _SIMD_ARGS(_argnames), f ) \
//    typedef _rettype (__fastcall * _functor_##_funcname)( _args ); \
//    static _functor_##_funcname s_arrFuncTable##_##_funcname[256] = { \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 0 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 1 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 2 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 3 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 4 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 5 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 6 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 7 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 8 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, 9 ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, a ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, b ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, c ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, d ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, e ), \
//        _SIMD_POPULATE_IMMEDIATE_EXPAND( _funcname, f ), \
//    };
//
//#define _SIMD_CALL_IMMEDIATE( _funcname, _args, _immValue ) \
//     s_arrFuncTable##_##_funcname[_immValue]( _args )
//
//_SIMD_DECLARE_IMMEDIATE( __m128i, _mm_insert_epi8, _SIMD_ARGS(__m128i mDst, Int8 iSrc), _SIMD_ARGS(mDst, iSrc) )
