/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_ExportValues.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Export operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Export::Values implementation
__forceinline Float SIMD::Export::Values::GetOne( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtss_f32( mSrc );
}

__forceinline Double SIMD::Export::Values::GetOne( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsd_f64( mSrc );
}

__forceinline Int32 SIMD::Export::Values::GetOne32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi128_si32( mSrc );
}
__forceinline Int64 SIMD::Export::Values::GetOne64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi128_si64( mSrc );
}

__forceinline Float SIMD::Export::Values::GetOne( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtss_f32( mSrc );
}

__forceinline Double SIMD::Export::Values::GetOne( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtsd_f64( mSrc );
}

__forceinline Int32 SIMD::Export::Values::GetOne32( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtsi256_si32( mSrc );
}
__forceinline Int64 SIMD::Export::Values::GetOne64( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtsi256_si64( mSrc );
}

//__forceinline Float SIMD::Export::Values::Get( __m128 mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    FloatConverter hConv;
//    hConv.i = _mm_extract_ps( mSrc, iIndex );
//    return hConv.f;
//}

//__forceinline Double SIMD::Export::Values::Get( __m128d mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    FloatConverter hConv;
//    hConv.i = _mm_extract_pd( mSrc, iIndex );
//    return hConv.f;
//}

//__forceinline Int32 SIMD::Export::Values::Get8( __m128i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_extract_epi8( mSrc, iIndex );
//}
//__forceinline Int32 SIMD::Export::Values::Get16( __m128i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_extract_epi16( mSrc, iIndex );
//}
//__forceinline Int32 SIMD::Export::Values::Get32( __m128i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_extract_epi32( mSrc, iIndex );
//}
//__forceinline Int64 SIMD::Export::Values::Get64( __m128i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_extract_epi64( mSrc, iIndex );
//}

//__forceinline Float SIMD::Export::Values::Get( __m256 mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    FloatConverter hConv;
//    hConv.i = _mm256_extract_ps( mSrc, iIndex );
//    return hConv.f;
//}

//__forceinline Double SIMD::Export::Values::Get( __m256d mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    FloatConverter hConv;
//    hConv.i = _mm256_extract_pd( mSrc, iIndex );
//    return hConv.f;
//}

//__forceinline Int32 SIMD::Export::Values::Get8( __m256i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_extract_epi8( mSrc, iIndex );
//}
//__forceinline Int32 SIMD::Export::Values::Get16( __m256i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_extract_epi16( mSrc, iIndex );
//}
//__forceinline Int32 SIMD::Export::Values::Get32( __m256i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_extract_epi32( mSrc, iIndex );
//}
//__forceinline Int64 SIMD::Export::Values::Get64( __m256i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_extract_epi64( mSrc, iIndex );
//}

