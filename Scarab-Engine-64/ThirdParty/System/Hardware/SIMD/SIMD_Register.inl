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
__forceinline __m128 SIMD::Register::Move::OneFloatLL( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_move_ss( mDst, mSrc );
}

__forceinline __m128 SIMD::Register::Move::TwoFloatLH( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_movehl_ps( mDst, mSrc );
}
__forceinline __m128 SIMD::Register::Move::TwoFloatHL( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_movelh_ps( mDst, mSrc );
}

__forceinline __m256 SIMD::Register::Move::FourFloatL( __m256 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insertf128_ps( mDst, mSrc, 0 );
}
__forceinline __m256 SIMD::Register::Move::FourFloatH( __m256 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insertf128_ps( mDst, mSrc, 1 );
}

__forceinline __m128 SIMD::Register::Move::FourFloatL( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extractf128_ps( mSrc, 0 );
}
__forceinline __m128 SIMD::Register::Move::FourFloatH( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extractf128_ps( mSrc, 1 );
}

__forceinline __m128d SIMD::Register::Move::OneDoubleLL( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_move_sd( mDst, mSrc );
}

__forceinline __m256d SIMD::Register::Move::TwoDoubleL( __m256d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insertf128_pd( mDst, mSrc, 0 );
}
__forceinline __m256d SIMD::Register::Move::TwoDoubleH( __m256d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insertf128_pd( mDst, mSrc, 1 );
}

__forceinline __m128d SIMD::Register::Move::TwoDoubleL( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extractf128_pd( mSrc, 0 );
}
__forceinline __m128d SIMD::Register::Move::TwoDoubleH( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extractf128_pd( mSrc, 1 );
}

__forceinline __m256i SIMD::Register::Move::FourInt32L( __m256i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_inserti128_si256( mDst, mSrc, 0 );
}
__forceinline __m256i SIMD::Register::Move::FourInt32H( __m256i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_inserti128_si256( mDst, mSrc, 1 );
}

__forceinline __m128i SIMD::Register::Move::FourInt32L( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_extracti128_si256( mSrc, 0 );
}
__forceinline __m128i SIMD::Register::Move::FourInt32H( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_extracti128_si256( mSrc, 1 );
}

__forceinline __m128i SIMD::Register::Move::OneInt64LL( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_move_epi64( mSrc );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Pack implementation
__forceinline __m128i SIMD::Register::Pack::Int16To8( __m128i mSrcLow, __m128i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_packs_epi16( mSrcLow, mSrcHigh );
}
__forceinline __m128i SIMD::Register::Pack::Int32To16( __m128i mSrcLow, __m128i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_packs_epi32( mSrcLow, mSrcHigh );
}

__forceinline __m256i SIMD::Register::Pack::Int16To8( __m256i mSrcLow, __m256i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packs_epi16( mSrcLow, mSrcHigh );
}
__forceinline __m256i SIMD::Register::Pack::Int32To16( __m256i mSrcLow, __m256i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packs_epi32( mSrcLow, mSrcHigh );
}

__forceinline __m128i SIMD::Register::Pack::UInt16To8( __m128i mSrcLow, __m128i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_packus_epi16( mSrcLow, mSrcHigh );
}
__forceinline __m128i SIMD::Register::Pack::UInt32To16( __m128i mSrcLow, __m128i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_packus_epi32( mSrcLow, mSrcHigh );
}

__forceinline __m256i SIMD::Register::Pack::UInt16To8( __m256i mSrcLow, __m256i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packus_epi16( mSrcLow, mSrcHigh );
}
__forceinline __m256i SIMD::Register::Pack::UInt32To16( __m256i mSrcLow, __m256i mSrcHigh ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packus_epi32( mSrcLow, mSrcHigh );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Interleave implementation
__forceinline __m128 SIMD::Register::Interleave::Low( __m128 mSrcEven, __m128 mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_unpacklo_ps( mSrcEven, mSrcOdd );
}
__forceinline __m128 SIMD::Register::Interleave::High( __m128 mSrcEven, __m128 mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_unpackhi_ps( mSrcEven, mSrcOdd );
}

__forceinline __m256 SIMD::Register::Interleave::LowX2( __m256 mSrcEven, __m256 mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpacklo_ps( mSrcEven, mSrcOdd );
}
__forceinline __m256 SIMD::Register::Interleave::HighX2( __m256 mSrcEven, __m256 mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpackhi_ps( mSrcEven, mSrcOdd );
}

__forceinline __m128d SIMD::Register::Interleave::Low( __m128d mSrcEven, __m128d mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_pd( mSrcEven, mSrcOdd );
}
__forceinline __m128d SIMD::Register::Interleave::High( __m128d mSrcEven, __m128d mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_pd( mSrcEven, mSrcOdd );
}

__forceinline __m256d SIMD::Register::Interleave::LowX2( __m256d mSrcEven, __m256d mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpacklo_pd( mSrcEven, mSrcOdd );
}
__forceinline __m256d SIMD::Register::Interleave::HighX2( __m256d mSrcEven, __m256d mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpackhi_pd( mSrcEven, mSrcOdd );
}

__forceinline __m128i SIMD::Register::Interleave::LowInt8( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi8( mSrcEven, mSrcOdd );
}
__forceinline __m128i SIMD::Register::Interleave::HighInt8( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi8( mSrcEven, mSrcOdd );
}
__forceinline __m128i SIMD::Register::Interleave::LowInt16( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi16( mSrcEven, mSrcOdd );
}
__forceinline __m128i SIMD::Register::Interleave::HighInt16( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi16( mSrcEven, mSrcOdd );
}
__forceinline __m128i SIMD::Register::Interleave::LowInt32( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi32( mSrcEven, mSrcOdd );
}
__forceinline __m128i SIMD::Register::Interleave::HighInt32( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi32( mSrcEven, mSrcOdd );
}
__forceinline __m128i SIMD::Register::Interleave::LowInt64( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi64( mSrcEven, mSrcOdd );
}
__forceinline __m128i SIMD::Register::Interleave::HighInt64( __m128i mSrcEven, __m128i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi64( mSrcEven, mSrcOdd );
}

__forceinline __m256i SIMD::Register::Interleave::LowInt8X2( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi8( mSrcEven, mSrcOdd );
}
__forceinline __m256i SIMD::Register::Interleave::HighInt8X2( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi8( mSrcEven, mSrcOdd );
}
__forceinline __m256i SIMD::Register::Interleave::LowInt16X2( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi16( mSrcEven, mSrcOdd );
}
__forceinline __m256i SIMD::Register::Interleave::HighInt16X2( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi16( mSrcEven, mSrcOdd );
}
__forceinline __m256i SIMD::Register::Interleave::LowInt32X2( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi32( mSrcEven, mSrcOdd );
}
__forceinline __m256i SIMD::Register::Interleave::HighInt32X2( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi32( mSrcEven, mSrcOdd );
}
__forceinline __m256i SIMD::Register::Interleave::LowInt64X2( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi64( mSrcEven, mSrcOdd );
}
__forceinline __m256i SIMD::Register::Interleave::HighInt64X2( __m256i mSrcEven, __m256i mSrcOdd ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi64( mSrcEven, mSrcOdd );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Spread2 implementation
__forceinline __m128d SIMD::Register::Spread2::AA( __m128d mSrc ) {
    //DebugAssert( CPUIDFn->HasSSE3() );
    //return _mm_movedup_pd( mSrc );
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastsd_pd( mSrc );
}
__forceinline __m128d SIMD::Register::Spread2::BB( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_pd( mSrc, 0x03 );
}

__forceinline __m128i SIMD::Register::Spread2::AA( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastq_epi64( mSrc );
}

__forceinline __m256 SIMD::Register::Spread2::AA( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_ps( mSrc, mSrc, 0x00 );
}
__forceinline __m256 SIMD::Register::Spread2::BB( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_ps( mSrc, mSrc, 0x11 );
}

__forceinline __m256d SIMD::Register::Spread2::AA( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_pd( mSrc, mSrc, 0x00 );
}
__forceinline __m256d SIMD::Register::Spread2::BB( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_pd( mSrc, mSrc, 0x11 );
}

__forceinline __m256i SIMD::Register::Spread2::AA( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastsi128_si256( _mm256_castsi256_si128(mSrc) );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Spread4 implementation
__forceinline __m128 SIMD::Register::Spread4::AAAA( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastss_ps( mSrc );
}
__forceinline __m128 SIMD::Register::Spread4::BBBB( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0x55 );
}
__forceinline __m128 SIMD::Register::Spread4::CCCC( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0xaa );
}
__forceinline __m128 SIMD::Register::Spread4::DDDD( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0xff );
}

__forceinline __m128 SIMD::Register::Spread4::AACC( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_moveldup_ps( mSrc );
}
__forceinline __m128 SIMD::Register::Spread4::BBDD( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_movehdup_ps( mSrc );
}

__forceinline __m128 SIMD::Register::Spread4::ABAB( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0x44 );
}
__forceinline __m128 SIMD::Register::Spread4::CDCD( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0xee );
}

__forceinline __m256d SIMD::Register::Spread4::AAAA( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastsd_pd( _mm256_castpd256_pd128(mSrc) );
}
__forceinline __m256d SIMD::Register::Spread4::BBBB( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute4x64_pd( mSrc, 0x55 );
}
__forceinline __m256d SIMD::Register::Spread4::CCCC( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute4x64_pd( mSrc, 0xaa );
}
__forceinline __m256d SIMD::Register::Spread4::DDDD( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute4x64_pd( mSrc, 0xff );
}

__forceinline __m256d SIMD::Register::Spread4::AACC( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_movedup_pd( mSrc );
}
__forceinline __m256d SIMD::Register::Spread4::BBDD( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_pd( mSrc, 0x0f );
}

__forceinline __m256d SIMD::Register::Spread4::AAAC( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_pd( mSrc, 0x80 );
}
__forceinline __m256d SIMD::Register::Spread4::AABB( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_pd( mSrc, 0x50 );
}
__forceinline __m256d SIMD::Register::Spread4::BCDD( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_pd( mSrc, 0xf9 );
}

__forceinline __m128i SIMD::Register::Spread4::AAAA( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastd_epi32( mSrc );
}

__forceinline __m256i SIMD::Register::Spread4::AAAA( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastq_epi64( _mm256_castsi256_si128(mSrc) );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Spread8 implementation
__forceinline __m256 SIMD::Register::Spread8::AAAAAAAA( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastss_ps( _mm256_castps256_ps128(mSrc) );
}

__forceinline __m256 SIMD::Register::Spread8::AACCEEGG( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_moveldup_ps( mSrc );
}
__forceinline __m256 SIMD::Register::Spread8::BBDDFFHH( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_movehdup_ps( mSrc );
}

__forceinline __m256 SIMD::Register::Spread8::AAAAEEEE( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0x00 );
}
__forceinline __m256 SIMD::Register::Spread8::BBBBFFFF( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0x55 );
}
__forceinline __m256 SIMD::Register::Spread8::CCCCGGGG( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0xaa );
}
__forceinline __m256 SIMD::Register::Spread8::DDDDHHHH( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0xff );
}

__forceinline __m256 SIMD::Register::Spread8::CBBBGFFF( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0x56 );
}
__forceinline __m256 SIMD::Register::Spread8::DDCCHHGG( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0xaf );
}

__forceinline __m128i SIMD::Register::Spread8::AAAAAAAA( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastw_epi16( mSrc );
}

__forceinline __m256i SIMD::Register::Spread8::AAAAAAAA( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastd_epi32( _mm256_castsi256_si128(mSrc) );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Spread16 implementation
//__forceinline __m128i SIMD::Register::Spread::Int8( __m128i mSrc ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_broadcastb_epi8( mSrc );
//}

//__forceinline __m256i SIMD::Register::Spread::Int16( __m256i mSrc ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_broadcastw_epi16( _mm256_castsi256_si128(mSrc) );
//}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Spread32 implementation
//__forceinline __m256i SIMD::Register::Spread::Int8( __m256i mSrc ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_broadcastb_epi8( _mm256_castsi256_si128(mSrc) );
//}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Shuffle2 implementation
__forceinline __m128d SIMD::Register::Shuffle2::BA( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_pd( mSrc, 0x01 );
}

__forceinline __m128i SIMD::Register::Shuffle2::BA( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_castsi256_si128( _mm256_permute4x64_epi64( _mm256_castsi128_si256(mSrc), 0x01 ) );
}

__forceinline __m256 SIMD::Register::Shuffle2::BA( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_ps( mSrc, mSrc, 0x01 );
}
__forceinline __m256d SIMD::Register::Shuffle2::BA( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_pd( mSrc, mSrc, 0x01 );
}
__forceinline __m256i SIMD::Register::Shuffle2::BA( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute2x128_si256( mSrc, mSrc, 0x01 );
}

__forceinline __m256 SIMD::Register::Shuffle2::AC( __m256 mSrcAB, __m256 mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_ps( mSrcAB, mSrcCD, 0x20 );
}
__forceinline __m256 SIMD::Register::Shuffle2::AD( __m256 mSrcAB, __m256 mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_ps( mSrcAB, mSrcCD, 0x30 );
}
__forceinline __m256 SIMD::Register::Shuffle2::BC( __m256 mSrcAB, __m256 mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_ps( mSrcAB, mSrcCD, 0x21 );
}
__forceinline __m256 SIMD::Register::Shuffle2::BD( __m256 mSrcAB, __m256 mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_ps( mSrcAB, mSrcCD, 0x31 );
}
__forceinline __m256 SIMD::Register::Shuffle2::CA( __m256 mSrcAB, __m256 mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_ps( mSrcAB, mSrcCD, 0x02 );
}
__forceinline __m256 SIMD::Register::Shuffle2::CB( __m256 mSrcAB, __m256 mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_ps( mSrcAB, mSrcCD, 0x12 );
}
__forceinline __m256 SIMD::Register::Shuffle2::DA( __m256 mSrcAB, __m256 mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_ps( mSrcAB, mSrcCD, 0x03 );
}
__forceinline __m256 SIMD::Register::Shuffle2::DB( __m256 mSrcAB, __m256 mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_ps( mSrcAB, mSrcCD, 0x13 );
}

__forceinline __m256d SIMD::Register::Shuffle2::AC( __m256d mSrcAB, __m256d mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_pd( mSrcAB, mSrcCD, 0x20 );
}
__forceinline __m256d SIMD::Register::Shuffle2::AD( __m256d mSrcAB, __m256d mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_pd( mSrcAB, mSrcCD, 0x30 );
}
__forceinline __m256d SIMD::Register::Shuffle2::BC( __m256d mSrcAB, __m256d mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_pd( mSrcAB, mSrcCD, 0x21 );
}
__forceinline __m256d SIMD::Register::Shuffle2::BD( __m256d mSrcAB, __m256d mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_pd( mSrcAB, mSrcCD, 0x31 );
}
__forceinline __m256d SIMD::Register::Shuffle2::CA( __m256d mSrcAB, __m256d mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_pd( mSrcAB, mSrcCD, 0x02 );
}
__forceinline __m256d SIMD::Register::Shuffle2::CB( __m256d mSrcAB, __m256d mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_pd( mSrcAB, mSrcCD, 0x12 );
}
__forceinline __m256d SIMD::Register::Shuffle2::DA( __m256d mSrcAB, __m256d mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_pd( mSrcAB, mSrcCD, 0x03 );
}
__forceinline __m256d SIMD::Register::Shuffle2::DB( __m256d mSrcAB, __m256d mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_pd( mSrcAB, mSrcCD, 0x13 );
}

__forceinline __m256i SIMD::Register::Shuffle2::AC( __m256i mSrcAB, __m256i mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute2x128_si256( mSrcAB, mSrcCD, 0x20 );
}
__forceinline __m256i SIMD::Register::Shuffle2::AD( __m256i mSrcAB, __m256i mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute2x128_si256( mSrcAB, mSrcCD, 0x30 );
}
__forceinline __m256i SIMD::Register::Shuffle2::BC( __m256i mSrcAB, __m256i mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute2x128_si256( mSrcAB, mSrcCD, 0x21 );
}
__forceinline __m256i SIMD::Register::Shuffle2::BD( __m256i mSrcAB, __m256i mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute2x128_si256( mSrcAB, mSrcCD, 0x31 );
}
__forceinline __m256i SIMD::Register::Shuffle2::CA( __m256i mSrcAB, __m256i mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute2x128_si256( mSrcAB, mSrcCD, 0x02 );
}
__forceinline __m256i SIMD::Register::Shuffle2::CB( __m256i mSrcAB, __m256i mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute2x128_si256( mSrcAB, mSrcCD, 0x12 );
}
__forceinline __m256i SIMD::Register::Shuffle2::DA( __m256i mSrcAB, __m256i mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute2x128_si256( mSrcAB, mSrcCD, 0x03 );
}
__forceinline __m256i SIMD::Register::Shuffle2::DB( __m256i mSrcAB, __m256i mSrcCD ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute2x128_si256( mSrcAB, mSrcCD, 0x13 );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Shuffle4 implementation
__forceinline __m128 SIMD::Register::Shuffle4::BACD( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0xe1 );
}
__forceinline __m128 SIMD::Register::Shuffle4::ABDC( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0xb4 );
}
__forceinline __m128 SIMD::Register::Shuffle4::ACBD( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0xd8 );
}
__forceinline __m128 SIMD::Register::Shuffle4::DBCA( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0x27 );
}

__forceinline __m128 SIMD::Register::Shuffle4::BCDA( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0x39 );
}
__forceinline __m128 SIMD::Register::Shuffle4::CDAB( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0x4e );
}
__forceinline __m128 SIMD::Register::Shuffle4::DABC( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0x93 );
}

__forceinline __m128 SIMD::Register::Shuffle4::DCBA( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0x1b );
}
__forceinline __m128 SIMD::Register::Shuffle4::CBAD( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0xc6 );
}
__forceinline __m128 SIMD::Register::Shuffle4::BADC( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0xb1 );
}
__forceinline __m128 SIMD::Register::Shuffle4::ADCB( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, 0x6c );
}

__forceinline __m256d SIMD::Register::Shuffle4::BACD( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_pd( mSrc, 0x09 );
}
__forceinline __m256d SIMD::Register::Shuffle4::ABDC( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_pd( mSrc, 0x06 );
}
__forceinline __m256d SIMD::Register::Shuffle4::ACBD( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_pd( mSrc, 0xd8 );
}
__forceinline __m256d SIMD::Register::Shuffle4::DBCA( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_pd( mSrc, 0x27 );
}

__forceinline __m256d SIMD::Register::Shuffle4::BCDA( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_pd( mSrc, 0x39 );
}
__forceinline __m256d SIMD::Register::Shuffle4::CDAB( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_pd( mSrc, 0x4e );
}
__forceinline __m256d SIMD::Register::Shuffle4::DABC( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_pd( mSrc, 0x93 );
}

__forceinline __m256d SIMD::Register::Shuffle4::DCBA( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_pd( mSrc, 0x1b );
}
__forceinline __m256d SIMD::Register::Shuffle4::CBAD( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_pd( mSrc, 0xc6 );
}
__forceinline __m256d SIMD::Register::Shuffle4::BADC( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_pd( mSrc, 0x05 );
}
__forceinline __m256d SIMD::Register::Shuffle4::ADCB( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_pd( mSrc, 0x6c );
}

__forceinline __m256d SIMD::Register::Shuffle4::DCAB( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_pd( mSrc, 0x4b );
}

__forceinline __m256d SIMD::Register::Shuffle4::AECH( __m256d mSrcABCD, __m256d mSrcEFGH ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_shuffle_pd( mSrcABCD, mSrcEFGH, 0x08 );
}
__forceinline __m256d SIMD::Register::Shuffle4::BFDH( __m256d mSrcABCD, __m256d mSrcEFGH ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_shuffle_pd( mSrcABCD, mSrcEFGH, 0x0f );
}

__forceinline __m128i SIMD::Register::Shuffle4::BACD( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shuffle_epi32( mSrc, 0xe1 );
}
__forceinline __m128i SIMD::Register::Shuffle4::ABDC( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shuffle_epi32( mSrc, 0xb4 );
}
__forceinline __m128i SIMD::Register::Shuffle4::ACBD( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shuffle_epi32( mSrc, 0xd8 );
}
__forceinline __m128i SIMD::Register::Shuffle4::DBCA( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shuffle_epi32( mSrc, 0x27 );
}

__forceinline __m128i SIMD::Register::Shuffle4::BCDA( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shuffle_epi32( mSrc, 0x39 );
}
__forceinline __m128i SIMD::Register::Shuffle4::CDAB( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shuffle_epi32( mSrc, 0x4e );
}
__forceinline __m128i SIMD::Register::Shuffle4::DABC( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shuffle_epi32( mSrc, 0x93 );
}

__forceinline __m128i SIMD::Register::Shuffle4::DCBA( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shuffle_epi32( mSrc, 0x1b );
}
__forceinline __m128i SIMD::Register::Shuffle4::CBAD( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shuffle_epi32( mSrc, 0xc6 );
}
__forceinline __m128i SIMD::Register::Shuffle4::BADC( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shuffle_epi32( mSrc, 0xb1 );
}
__forceinline __m128i SIMD::Register::Shuffle4::ADCB( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shuffle_epi32( mSrc, 0x6c );
}

__forceinline __m256i SIMD::Register::Shuffle4::BACD( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_epi64( mSrc, 0xe1 );
}
__forceinline __m256i SIMD::Register::Shuffle4::ABDC( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_epi64( mSrc, 0xb4 );
}
__forceinline __m256i SIMD::Register::Shuffle4::ACBD( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_epi64( mSrc, 0xd8 );
}
__forceinline __m256i SIMD::Register::Shuffle4::DBCA( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_epi64( mSrc, 0x27 );
}

__forceinline __m256i SIMD::Register::Shuffle4::BCDA( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_epi64( mSrc, 0x39 );
}
__forceinline __m256i SIMD::Register::Shuffle4::CDAB( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_epi64( mSrc, 0x4e );
}
__forceinline __m256i SIMD::Register::Shuffle4::DABC( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_epi64( mSrc, 0x93 );
}

__forceinline __m256i SIMD::Register::Shuffle4::DCBA( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_epi64( mSrc, 0x1b );
}
__forceinline __m256i SIMD::Register::Shuffle4::CBAD( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_epi64( mSrc, 0xc6 );
}
__forceinline __m256i SIMD::Register::Shuffle4::BADC( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_epi64( mSrc, 0xb1 );
}
__forceinline __m256i SIMD::Register::Shuffle4::ADCB( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_epi64( mSrc, 0x6c );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Shuffle8 implementation
__forceinline __m256 SIMD::Register::Shuffle8::BACDFEGH( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0xe1 );
}
__forceinline __m256 SIMD::Register::Shuffle8::ABDCEFHG( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0xb4 );
}
__forceinline __m256 SIMD::Register::Shuffle8::ACBDEGFH( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0xd8 );
}
__forceinline __m256 SIMD::Register::Shuffle8::DBCAHFGE( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0x27 );
}

__forceinline __m256 SIMD::Register::Shuffle8::BCDAFGHE( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0x39 );
}
__forceinline __m256 SIMD::Register::Shuffle8::CDABGHEF( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0x4e );
}
__forceinline __m256 SIMD::Register::Shuffle8::DABCHEFG( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0x93 );
}

__forceinline __m256 SIMD::Register::Shuffle8::DCBAHGFE( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0x1b );
}
__forceinline __m256 SIMD::Register::Shuffle8::CBADGFEH( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0xc6 );
}
__forceinline __m256 SIMD::Register::Shuffle8::BADCFEHG( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0xb1 );
}
__forceinline __m256 SIMD::Register::Shuffle8::ADCBEHGF( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, 0x6c );
}

__forceinline __m128i SIMD::Register::Shuffle8::BACDEFGH( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflelo_epi16( mSrc, 0xe1 );
}
__forceinline __m128i SIMD::Register::Shuffle8::ABDCEFGH( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflelo_epi16( mSrc, 0xb4 );
}
__forceinline __m128i SIMD::Register::Shuffle8::ACBDEFGH( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflelo_epi16( mSrc, 0xd8 );
}
__forceinline __m128i SIMD::Register::Shuffle8::DBCAEFGH( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflelo_epi16( mSrc, 0x27 );
}

__forceinline __m128i SIMD::Register::Shuffle8::ABCDFEGH( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflehi_epi16( mSrc, 0xe1 );
}
__forceinline __m128i SIMD::Register::Shuffle8::ABCDEFHG( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflehi_epi16( mSrc, 0xb4 );
}
__forceinline __m128i SIMD::Register::Shuffle8::ABCDEGFH( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflehi_epi16( mSrc, 0xd8 );
}
__forceinline __m128i SIMD::Register::Shuffle8::ABCDHFGE( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflehi_epi16( mSrc, 0x27 );
}

__forceinline __m128i SIMD::Register::Shuffle8::BCDAEFGH( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflelo_epi16( mSrc, 0x39 );
}
__forceinline __m128i SIMD::Register::Shuffle8::CDABEFGH( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflelo_epi16( mSrc, 0x4e );
}
__forceinline __m128i SIMD::Register::Shuffle8::DABCEFGH( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflelo_epi16( mSrc, 0x93 );
}

__forceinline __m128i SIMD::Register::Shuffle8::ABCDFGHE( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflehi_epi16( mSrc, 0x39 );
}
__forceinline __m128i SIMD::Register::Shuffle8::ABCDGHEF( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflehi_epi16( mSrc, 0x4e );
}
__forceinline __m128i SIMD::Register::Shuffle8::ABCDHEFG( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflehi_epi16( mSrc, 0x93 );
}

__forceinline __m128i SIMD::Register::Shuffle8::DCBAEFGH( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflelo_epi16( mSrc, 0x1b );
}
__forceinline __m128i SIMD::Register::Shuffle8::CBADEFGH( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflelo_epi16( mSrc, 0xc6 );
}
__forceinline __m128i SIMD::Register::Shuffle8::BADCEFGH( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflelo_epi16( mSrc, 0xb1 );
}
__forceinline __m128i SIMD::Register::Shuffle8::ADCBEFGH( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflelo_epi16( mSrc, 0x6c );
}

__forceinline __m128i SIMD::Register::Shuffle8::ABCDHGFE( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflehi_epi16( mSrc, 0x1b );
}
__forceinline __m128i SIMD::Register::Shuffle8::ABCDGFEH( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflehi_epi16( mSrc, 0xc6 );
}
__forceinline __m128i SIMD::Register::Shuffle8::ABCDFEHG( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflehi_epi16( mSrc, 0xb1 );
}
__forceinline __m128i SIMD::Register::Shuffle8::ABCDEHGF( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflehi_epi16( mSrc, 0x6c );
}

__forceinline __m256i SIMD::Register::Shuffle8::BACDFEGH( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi32( mSrc, 0xe1 );
}
__forceinline __m256i SIMD::Register::Shuffle8::ABDCEFHG( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi32( mSrc, 0xb4 );
}
__forceinline __m256i SIMD::Register::Shuffle8::ACBDEGFH( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi32( mSrc, 0xd8 );
}
__forceinline __m256i SIMD::Register::Shuffle8::DBCAHFGE( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi32( mSrc, 0x27 );
}

__forceinline __m256i SIMD::Register::Shuffle8::BCDAFGHE( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi32( mSrc, 0x39 );
}
__forceinline __m256i SIMD::Register::Shuffle8::CDABGHEF( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi32( mSrc, 0x4e );
}
__forceinline __m256i SIMD::Register::Shuffle8::DABCHEFG( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi32( mSrc, 0x93 );
}

__forceinline __m256i SIMD::Register::Shuffle8::DCBAHGFE( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi32( mSrc, 0x1b );
}
__forceinline __m256i SIMD::Register::Shuffle8::CBADGFEH( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi32( mSrc, 0xc6 );
}
__forceinline __m256i SIMD::Register::Shuffle8::BADCFEHG( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi32( mSrc, 0xb1 );
}
__forceinline __m256i SIMD::Register::Shuffle8::ADCBEHGF( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi32( mSrc, 0x6c );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Shuffle16 implementation

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Shuffle32 implementation

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::ShuffleIndexed implementation
__forceinline __m128i SIMD::Register::ShuffleIndexed::Make4Indices( const Int32 * arrIndices ) {
    return ::SIMD::Import::Memory::Load128( arrIndices );
}
__forceinline __m128 SIMD::Register::ShuffleIndexed::FourFloat( __m128 mSrc, __m128i mIndices4 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permutevar_ps( mSrc, mIndices4 );
}

__forceinline __m256i SIMD::Register::ShuffleIndexed::Make8Indices( const Int32 * arrIndices ) {
    return ::SIMD::Import::Memory::Load256( arrIndices );
}
__forceinline __m256 SIMD::Register::ShuffleIndexed::FourFloatX2( __m256 mSrc, __m256i mIndices4 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permutevar_ps( mSrc, mIndices4 );
}
__forceinline __m256 SIMD::Register::ShuffleIndexed::EightFloat( __m256 mSrc, __m256i mIndices8 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permutevar8x32_ps( mSrc, mIndices8 );
}
__forceinline __m256i SIMD::Register::ShuffleIndexed::EightInt32( __m256i mSrc, __m256i mIndices8 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permutevar8x32_epi32( mSrc, mIndices8 );
}

__forceinline __m128i SIMD::Register::ShuffleIndexed::Make2Indices( const Int64 * arrIndices ) {
    return ::SIMD::Import::Memory::Load128( arrIndices );
}
__forceinline __m128d SIMD::Register::ShuffleIndexed::TwoDouble( __m128d mSrc, __m128i mIndices2 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permutevar_pd( mSrc, mIndices2 );
}

__forceinline __m256i SIMD::Register::ShuffleIndexed::Make4Indices( const Int64 * arrIndices ) {
    return ::SIMD::Import::Memory::Load256( arrIndices );
}
__forceinline __m256d SIMD::Register::ShuffleIndexed::TwoDoubleX2( __m256d mSrc, __m256i mIndices2 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permutevar_pd( mSrc, mIndices2 );
}

__forceinline __m128i SIMD::Register::ShuffleIndexed::Make16Indices( const Int8 * arrIndices_Z ) {
    return ::SIMD::Import::Memory::Load128( arrIndices_Z );
}
__forceinline __m128i SIMD::Register::ShuffleIndexed::SixteenInt8( __m128i mSrc, __m128i mIndices16_Z ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_shuffle_epi8( mSrc, mIndices16_Z );
}

__forceinline __m256i SIMD::Register::ShuffleIndexed::Make32Indices( const Int8 * arrIndices_Z ) {
    return ::SIMD::Import::Memory::Load256( arrIndices_Z );
}
__forceinline __m256i SIMD::Register::ShuffleIndexed::SixteenInt8X2( __m256i mSrc, __m256i mIndices16_Z ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi8( mSrc, mIndices16_Z );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Register::Blend implementation
//__forceinline __m128 SIMD::Register::Blend::Float( __m128 mDst, __m128 mSrc, Int iMask4 ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_blend_ps( mDst, mSrc, iMask4 );
//}
__forceinline __m128 SIMD::Register::Blend::Float( __m128 mDst, __m128 mSrc, __m128 mSigns ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_blendv_ps( mDst, mSrc, mSigns );
}

//__forceinline __m128d SIMD::Register::Blend::Double( __m128d mDst, __m128d mSrc, Int iMask2 ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_blend_pd( mDst, mSrc, iMask2 );
//}
__forceinline __m128d SIMD::Register::Blend::Double( __m128d mDst, __m128d mSrc, __m128d mSigns ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_blendv_pd( mDst, mSrc, mSigns );
}

__forceinline __m128i SIMD::Register::Blend::Int8( __m128i mDst, __m128i mSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_blendv_epi8( mDst, mSrc, mSigns );
}
//__forceinline __m128i SIMD::Register::Blend::Int16( __m128i mDst, __m128i mSrc, Int iMask8 ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_blend_epi16( mDst, mSrc, iMask8 );
//}
//__forceinline __m128i SIMD::Register::Blend::Int32( __m128i mDst, __m128i mSrc, Int iMask4 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_blend_epi32( mDst, mSrc, iMask4 );
//}

//__forceinline __m256 SIMD::Register::Blend::Float( __m256 mDst, __m256 mSrc, Int iMask8 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_blend_ps( mDst, mSrc, iMask8 );
//}
__forceinline __m256 SIMD::Register::Blend::Float( __m256 mDst, __m256 mSrc, __m256 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_blendv_ps( mDst, mSrc, mSigns );
}

//__forceinline __m256d SIMD::Register::Blend::Double( __m256d mDst, __m256d mSrc, Int iMask4 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_blend_pd( mDst, mSrc, iMask4 );
//}
__forceinline __m256d SIMD::Register::Blend::Double( __m256d mDst, __m256d mSrc, __m256d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_blendv_pd( mDst, mSrc, mSigns );
}

__forceinline __m256i SIMD::Register::Blend::Int8( __m256i mDst, __m256i mSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_blendv_epi8( mDst, mSrc, mSigns );
}
//__forceinline __m256i SIMD::Register::Blend::Int16( __m256i mDst, __m256i mSrc, Int iMask8 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_blend_epi16( mDst, mSrc, iMask8 );
//}
//__forceinline __m256i SIMD::Register::Blend::Int32( __m256i mDst, __m256i mSrc, Int iMask8 ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_blend_epi32( mDst, mSrc, iMask8 );
//}


