/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD low level abstraction layer, Master Header
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


