/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Cast.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Cast operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Cast implementation
__forceinline __m128 SIMD::Cast::ToFloat( __m128d mDouble ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castpd_ps( mDouble );
}
__forceinline __m128 SIMD::Cast::ToFloat( __m128i mInteger ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castsi128_ps( mInteger );
}
__forceinline __m256 SIMD::Cast::ToFloat( __m256d mDouble ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castpd_ps( mDouble );
}
__forceinline __m256 SIMD::Cast::ToFloat( __m256i mInteger ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castsi256_ps( mInteger );
}

__forceinline __m128d SIMD::Cast::ToDouble( __m128 mFloat ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castps_pd( mFloat );
}
__forceinline __m128d SIMD::Cast::ToDouble( __m128i mInteger ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castsi128_pd( mInteger );
}
__forceinline __m256d SIMD::Cast::ToDouble( __m256 mFloat ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castps_pd( mFloat );
}
__forceinline __m256d SIMD::Cast::ToDouble( __m256i mInteger ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castsi256_pd( mInteger );
}

__forceinline __m128i SIMD::Cast::ToInteger( __m128 mFloat ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castps_si128( mFloat );
}
__forceinline __m128i SIMD::Cast::ToInteger( __m128d mDouble ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castpd_si128( mDouble );
}
__forceinline __m256i SIMD::Cast::ToInteger( __m256 mFloat ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castps_si256( mFloat );
}
__forceinline __m256i SIMD::Cast::ToInteger( __m256d mDouble ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castpd_si256( mDouble );
}

__forceinline __m128 SIMD::Cast::Down( __m256 mFloat ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castps256_ps128( mFloat );
}
__forceinline __m128d SIMD::Cast::Down( __m256d mDouble ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castpd256_pd128( mDouble );
}
__forceinline __m128i SIMD::Cast::Down( __m256i mInteger ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castsi256_si128( mInteger );
}

__forceinline __m256 SIMD::Cast::Up( __m128 mFloat ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castps128_ps256( mFloat );
}
__forceinline __m256d SIMD::Cast::Up( __m128d mDouble ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castpd128_pd256( mDouble );
}
__forceinline __m256i SIMD::Cast::Up( __m128i mInteger ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castsi128_si256( mInteger );
}

