/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_ImportMemory.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Import operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Import::Memory implementation
inline __m128 SIMD::Import::Memory::LoadOne( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_load_ss( pSrc );
}
inline __m128d SIMD::Import::Memory::LoadOne( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_sd( pSrc );
}

inline __m128d SIMD::Import::Memory::LoadOneDoubleL( __m128d mDst, const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadl_pd( mDst, pSrc );
}
inline __m128d SIMD::Import::Memory::LoadOneDoubleH( __m128d mDst, const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadh_pd( mDst, pSrc );
}

inline __m128i SIMD::Import::Memory::LoadOneInt64L( const __m128i * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadl_epi64( pSrc );
}

inline __m128 SIMD::Import::Memory::Load128( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_loadu_ps( pSrc );
}

inline __m128d SIMD::Import::Memory::Load128( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadu_pd( pSrc );
}

inline __m128i SIMD::Import::Memory::Load128( const Int8 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_lddqu_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Load128( const Int16 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_lddqu_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Load128( const Int32 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_lddqu_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Load128( const Int64 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_lddqu_si128( (const __m128i *)pSrc );
}

inline __m128i SIMD::Import::Memory::Load128( const UInt8 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_lddqu_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Load128( const UInt16 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_lddqu_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Load128( const UInt32 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_lddqu_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Load128( const UInt64 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_lddqu_si128( (const __m128i *)pSrc );
}

inline __m256 SIMD::Import::Memory::Load256( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_loadu_ps( pSrc );
}

inline __m256d SIMD::Import::Memory::Load256( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_loadu_pd( pSrc );
}

inline __m256i SIMD::Import::Memory::Load256( const Int8 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_lddqu_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Load256( const Int16 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_lddqu_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Load256( const Int32 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_lddqu_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Load256( const Int64 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_lddqu_si256( (const __m256i *)pSrc );
}

inline __m256i SIMD::Import::Memory::Load256( const UInt8 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_lddqu_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Load256( const UInt16 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_lddqu_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Load256( const UInt32 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_lddqu_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Load256( const UInt64 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_lddqu_si256( (const __m256i *)pSrc );
}

inline __m128 SIMD::Import::Memory::Load128( const Float * pSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_maskload_ps( pSrc, mSigns );
}

inline __m128d SIMD::Import::Memory::Load128( const Double * pSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_maskload_pd( pSrc, mSigns );
}

inline __m128i SIMD::Import::Memory::Load128( const Int32 * pSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_maskload_epi32( pSrc, mSigns );
}
inline __m128i SIMD::Import::Memory::Load128( const Int64 * pSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_maskload_epi64( pSrc, mSigns );
}

inline __m256 SIMD::Import::Memory::Load256( const Float * pSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_maskload_ps( pSrc, mSigns );
}

inline __m256d SIMD::Import::Memory::Load256( const Double * pSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_maskload_pd( pSrc, mSigns );
}

inline __m256i SIMD::Import::Memory::Load256( const Int32 * pSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_maskload_epi32( pSrc, mSigns );
}
inline __m256i SIMD::Import::Memory::Load256( const Int64 * pSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_maskload_epi64( pSrc, mSigns );
}

//inline __m128 SIMD::Import::Memory::Spread128( const Float * pSrc ) {
//    DebugAssert( CPUIDFn->HasSSE() );
//    return _mm_load1_ps( pSrc );
//}
inline __m128 SIMD::Import::Memory::Spread128( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_broadcast_ss( pSrc );
}

inline __m128d SIMD::Import::Memory::Spread128( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_loaddup_pd( pSrc );
}

inline __m256 SIMD::Import::Memory::Spread256( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_broadcast_ss( pSrc );
}
inline __m256 SIMD::Import::Memory::Spread256( const __m128 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_broadcast_ps( pSrc );
}

inline __m256d SIMD::Import::Memory::Spread256( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_broadcast_sd( pSrc );
}
inline __m256d SIMD::Import::Memory::Spread256( const __m128d * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_broadcast_pd( pSrc );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Import::Memory::Aligned implementation
inline __m128 SIMD::Import::Memory::Aligned::Load128( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_load_ps( pSrc );
}

inline __m128d SIMD::Import::Memory::Aligned::Load128( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_pd( pSrc );
}

inline __m128i SIMD::Import::Memory::Aligned::Load128( const Int8 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Aligned::Load128( const Int16 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Aligned::Load128( const Int32 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Aligned::Load128( const Int64 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_si128( (const __m128i *)pSrc );
}

inline __m128i SIMD::Import::Memory::Aligned::Load128( const UInt8 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Aligned::Load128( const UInt16 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Aligned::Load128( const UInt32 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Aligned::Load128( const UInt64 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_si128( (const __m128i *)pSrc );
}

inline __m256 SIMD::Import::Memory::Aligned::Load256( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_ps( pSrc );
}

inline __m256d SIMD::Import::Memory::Aligned::Load256( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_pd( pSrc );
}

inline __m256i SIMD::Import::Memory::Aligned::Load256( const Int8 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Aligned::Load256( const Int16 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Aligned::Load256( const Int32 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Aligned::Load256( const Int64 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_si256( (const __m256i *)pSrc );
}

inline __m256i SIMD::Import::Memory::Aligned::Load256( const UInt8 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Aligned::Load256( const UInt16 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Aligned::Load256( const UInt32 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Aligned::Load256( const UInt64 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_si256( (const __m256i *)pSrc );
}

inline __m128 SIMD::Import::Memory::Aligned::Load128R( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_loadr_ps( pSrc );
}
inline __m128d SIMD::Import::Memory::Aligned::Load128R( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadr_pd( pSrc );
}

inline __m128 SIMD::Import::Memory::Aligned::Load128NT( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_castsi128_ps( _mm_stream_load_si128( (const __m128i *)pSrc ) );
}

inline __m128d SIMD::Import::Memory::Aligned::Load128NT( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_castsi128_pd( _mm_stream_load_si128( (const __m128i *)pSrc ) );
}

inline __m128i SIMD::Import::Memory::Aligned::Load128NT( const Int8 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_stream_load_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Aligned::Load128NT( const Int16 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_stream_load_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Aligned::Load128NT( const Int32 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_stream_load_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Aligned::Load128NT( const Int64 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_stream_load_si128( (const __m128i *)pSrc );
}

inline __m128i SIMD::Import::Memory::Aligned::Load128NT( const UInt8 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_stream_load_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Aligned::Load128NT( const UInt16 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_stream_load_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Aligned::Load128NT( const UInt32 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_stream_load_si128( (const __m128i *)pSrc );
}
inline __m128i SIMD::Import::Memory::Aligned::Load128NT( const UInt64 * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_stream_load_si128( (const __m128i *)pSrc );
}

inline __m256 SIMD::Import::Memory::Aligned::Load256NT( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_castsi256_ps( _mm256_stream_load_si256( (const __m256i *)pSrc ) );
}

inline __m256d SIMD::Import::Memory::Aligned::Load256NT( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_castsi256_pd( _mm256_stream_load_si256( (const __m256i *)pSrc ) );
}

inline __m256i SIMD::Import::Memory::Aligned::Load256NT( const Int8 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_stream_load_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Aligned::Load256NT( const Int16 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_stream_load_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Aligned::Load256NT( const Int32 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_stream_load_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Aligned::Load256NT( const Int64 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_stream_load_si256( (const __m256i *)pSrc );
}

inline __m256i SIMD::Import::Memory::Aligned::Load256NT( const UInt8 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_stream_load_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Aligned::Load256NT( const UInt16 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_stream_load_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Aligned::Load256NT( const UInt32 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_stream_load_si256( (const __m256i *)pSrc );
}
inline __m256i SIMD::Import::Memory::Aligned::Load256NT( const UInt64 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_stream_load_si256( (const __m256i *)pSrc );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Import::Memory::Sparse32::Stride1 implementation
inline __m128 SIMD::Import::Memory::Sparse32::Stride1::Load128( const Float * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_ps( pSrc, mIndices, 1 );
}

inline __m128d SIMD::Import::Memory::Sparse32::Stride1::Load128( const Double * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_pd( pSrc, mIndices, 1 );
}

inline __m128i SIMD::Import::Memory::Sparse32::Stride1::Load128( const Int32 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_epi32( pSrc, mIndices, 1 );
}
inline __m128i SIMD::Import::Memory::Sparse32::Stride1::Load128( const Int64 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_epi64( pSrc, mIndices, 1 );
}

inline __m256 SIMD::Import::Memory::Sparse32::Stride1::Load256( const Float * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_ps( pSrc, mIndices, 1 );
}

inline __m256d SIMD::Import::Memory::Sparse32::Stride1::Load256( const Double * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_pd( pSrc, mIndices, 1 );
}

inline __m256i SIMD::Import::Memory::Sparse32::Stride1::Load256( const Int32 * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_epi32( pSrc, mIndices, 1 );
}
inline __m256i SIMD::Import::Memory::Sparse32::Stride1::Load256( const Int64 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_epi64( pSrc, mIndices, 1 );
}

inline __m128 SIMD::Import::Memory::Sparse32::Stride1::Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_ps( mDst, pSrc, mIndices, mSigns, 1 );
}

inline __m128d SIMD::Import::Memory::Sparse32::Stride1::Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_pd( mDst, pSrc, mIndices, mSigns, 1 );
}

inline __m128i SIMD::Import::Memory::Sparse32::Stride1::Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_epi32( mDst, pSrc, mIndices, mSigns, 1 );
}
inline __m128i SIMD::Import::Memory::Sparse32::Stride1::Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_epi64( mDst, pSrc, mIndices, mSigns, 1 );
}

inline __m256 SIMD::Import::Memory::Sparse32::Stride1::Load256( __m256 mDst, const Float * pSrc, __m256i mIndices, __m256 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_ps( mDst, pSrc, mIndices, mSigns, 1 );
}

inline __m256d SIMD::Import::Memory::Sparse32::Stride1::Load256( __m256d mDst, const Double * pSrc, __m128i mIndices, __m256d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_pd( mDst, pSrc, mIndices, mSigns, 1 );
}

inline __m256i SIMD::Import::Memory::Sparse32::Stride1::Load256( __m256i mDst, const Int32 * pSrc, __m256i mIndices, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_epi32( mDst, pSrc, mIndices, mSigns, 1 );
}
inline __m256i SIMD::Import::Memory::Sparse32::Stride1::Load256( __m256i mDst, const Int64 * pSrc, __m128i mIndices, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_epi64( mDst, pSrc, mIndices, mSigns, 1 );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Import::Memory::Sparse32::Stride2 implementation
inline __m128 SIMD::Import::Memory::Sparse32::Stride2::Load128( const Float * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_ps( pSrc, mIndices, 2 );
}

inline __m128d SIMD::Import::Memory::Sparse32::Stride2::Load128( const Double * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_pd( pSrc, mIndices, 2 );
}

inline __m128i SIMD::Import::Memory::Sparse32::Stride2::Load128( const Int32 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_epi32( pSrc, mIndices, 2 );
}
inline __m128i SIMD::Import::Memory::Sparse32::Stride2::Load128( const Int64 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_epi64( pSrc, mIndices, 2 );
}

inline __m256 SIMD::Import::Memory::Sparse32::Stride2::Load256( const Float * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_ps( pSrc, mIndices, 2 );
}

inline __m256d SIMD::Import::Memory::Sparse32::Stride2::Load256( const Double * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_pd( pSrc, mIndices, 2 );
}

inline __m256i SIMD::Import::Memory::Sparse32::Stride2::Load256( const Int32 * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_epi32( pSrc, mIndices, 2 );
}
inline __m256i SIMD::Import::Memory::Sparse32::Stride2::Load256( const Int64 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_epi64( pSrc, mIndices, 2 );
}

inline __m128 SIMD::Import::Memory::Sparse32::Stride2::Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_ps( mDst, pSrc, mIndices, mSigns, 2 );
}

inline __m128d SIMD::Import::Memory::Sparse32::Stride2::Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_pd( mDst, pSrc, mIndices, mSigns, 2 );
}

inline __m128i SIMD::Import::Memory::Sparse32::Stride2::Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_epi32( mDst, pSrc, mIndices, mSigns, 2 );
}
inline __m128i SIMD::Import::Memory::Sparse32::Stride2::Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_epi64( mDst, pSrc, mIndices, mSigns, 2 );
}

inline __m256 SIMD::Import::Memory::Sparse32::Stride2::Load256( __m256 mDst, const Float * pSrc, __m256i mIndices, __m256 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_ps( mDst, pSrc, mIndices, mSigns, 2 );
}

inline __m256d SIMD::Import::Memory::Sparse32::Stride2::Load256( __m256d mDst, const Double * pSrc, __m128i mIndices, __m256d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_pd( mDst, pSrc, mIndices, mSigns, 2 );
}

inline __m256i SIMD::Import::Memory::Sparse32::Stride2::Load256( __m256i mDst, const Int32 * pSrc, __m256i mIndices, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_epi32( mDst, pSrc, mIndices, mSigns, 2 );
}
inline __m256i SIMD::Import::Memory::Sparse32::Stride2::Load256( __m256i mDst, const Int64 * pSrc, __m128i mIndices, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_epi64( mDst, pSrc, mIndices, mSigns, 2 );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Import::Memory::Sparse32::Stride4 implementation
inline __m128 SIMD::Import::Memory::Sparse32::Stride4::Load128( const Float * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_ps( pSrc, mIndices, 4 );
}

inline __m128d SIMD::Import::Memory::Sparse32::Stride4::Load128( const Double * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_pd( pSrc, mIndices, 4 );
}

inline __m128i SIMD::Import::Memory::Sparse32::Stride4::Load128( const Int32 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_epi32( pSrc, mIndices, 4 );
}
inline __m128i SIMD::Import::Memory::Sparse32::Stride4::Load128( const Int64 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_epi64( pSrc, mIndices, 4 );
}

inline __m256 SIMD::Import::Memory::Sparse32::Stride4::Load256( const Float * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_ps( pSrc, mIndices, 4 );
}

inline __m256d SIMD::Import::Memory::Sparse32::Stride4::Load256( const Double * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_pd( pSrc, mIndices, 4 );
}

inline __m256i SIMD::Import::Memory::Sparse32::Stride4::Load256( const Int32 * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_epi32( pSrc, mIndices, 4 );
}
inline __m256i SIMD::Import::Memory::Sparse32::Stride4::Load256( const Int64 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_epi64( pSrc, mIndices, 4 );
}

inline __m128 SIMD::Import::Memory::Sparse32::Stride4::Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_ps( mDst, pSrc, mIndices, mSigns, 4 );
}

inline __m128d SIMD::Import::Memory::Sparse32::Stride4::Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_pd( mDst, pSrc, mIndices, mSigns, 4 );
}

inline __m128i SIMD::Import::Memory::Sparse32::Stride4::Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_epi32( mDst, pSrc, mIndices, mSigns, 4 );
}
inline __m128i SIMD::Import::Memory::Sparse32::Stride4::Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_epi64( mDst, pSrc, mIndices, mSigns, 4 );
}

inline __m256 SIMD::Import::Memory::Sparse32::Stride4::Load256( __m256 mDst, const Float * pSrc, __m256i mIndices, __m256 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_ps( mDst, pSrc, mIndices, mSigns, 4 );
}

inline __m256d SIMD::Import::Memory::Sparse32::Stride4::Load256( __m256d mDst, const Double * pSrc, __m128i mIndices, __m256d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_pd( mDst, pSrc, mIndices, mSigns, 4 );
}

inline __m256i SIMD::Import::Memory::Sparse32::Stride4::Load256( __m256i mDst, const Int32 * pSrc, __m256i mIndices, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_epi32( mDst, pSrc, mIndices, mSigns, 4 );
}
inline __m256i SIMD::Import::Memory::Sparse32::Stride4::Load256( __m256i mDst, const Int64 * pSrc, __m128i mIndices, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_epi64( mDst, pSrc, mIndices, mSigns, 4 );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Import::Memory::Sparse32::Stride8 implementation
inline __m128 SIMD::Import::Memory::Sparse32::Stride8::Load128( const Float * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_ps( pSrc, mIndices, 8 );
}

inline __m128d SIMD::Import::Memory::Sparse32::Stride8::Load128( const Double * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_pd( pSrc, mIndices, 8 );
}

inline __m128i SIMD::Import::Memory::Sparse32::Stride8::Load128( const Int32 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_epi32( pSrc, mIndices, 8 );
}
inline __m128i SIMD::Import::Memory::Sparse32::Stride8::Load128( const Int64 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_epi64( pSrc, mIndices, 8 );
}

inline __m256 SIMD::Import::Memory::Sparse32::Stride8::Load256( const Float * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_ps( pSrc, mIndices, 8 );
}

inline __m256d SIMD::Import::Memory::Sparse32::Stride8::Load256( const Double * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_pd( pSrc, mIndices, 8 );
}

inline __m256i SIMD::Import::Memory::Sparse32::Stride8::Load256( const Int32 * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_epi32( pSrc, mIndices, 8 );
}
inline __m256i SIMD::Import::Memory::Sparse32::Stride8::Load256( const Int64 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_epi64( pSrc, mIndices, 8 );
}

inline __m128 SIMD::Import::Memory::Sparse32::Stride8::Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_ps( mDst, pSrc, mIndices, mSigns, 8 );
}

inline __m128d SIMD::Import::Memory::Sparse32::Stride8::Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_pd( mDst, pSrc, mIndices, mSigns, 8 );
}

inline __m128i SIMD::Import::Memory::Sparse32::Stride8::Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_epi32( mDst, pSrc, mIndices, mSigns, 8 );
}
inline __m128i SIMD::Import::Memory::Sparse32::Stride8::Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_epi64( mDst, pSrc, mIndices, mSigns, 8 );
}

inline __m256 SIMD::Import::Memory::Sparse32::Stride8::Load256( __m256 mDst, const Float * pSrc, __m256i mIndices, __m256 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_ps( mDst, pSrc, mIndices, mSigns, 8 );
}

inline __m256d SIMD::Import::Memory::Sparse32::Stride8::Load256( __m256d mDst, const Double * pSrc, __m128i mIndices, __m256d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_pd( mDst, pSrc, mIndices, mSigns, 8 );
}

inline __m256i SIMD::Import::Memory::Sparse32::Stride8::Load256( __m256i mDst, const Int32 * pSrc, __m256i mIndices, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_epi32( mDst, pSrc, mIndices, mSigns, 8 );
}
inline __m256i SIMD::Import::Memory::Sparse32::Stride8::Load256( __m256i mDst, const Int64 * pSrc, __m128i mIndices, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_epi64( mDst, pSrc, mIndices, mSigns, 8 );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Import::Memory::Sparse64::Stride1 implementation
inline __m128 SIMD::Import::Memory::Sparse64::Stride1::Load128( const Float * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_ps( pSrc, mIndices, 1 );
}

inline __m128d SIMD::Import::Memory::Sparse64::Stride1::Load128( const Double * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_pd( pSrc, mIndices, 1 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride1::Load128( const Int32 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_epi32( pSrc, mIndices, 1 );
}
inline __m128i SIMD::Import::Memory::Sparse64::Stride1::Load128( const Int64 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_epi64( pSrc, mIndices, 1 );
}

inline __m128 SIMD::Import::Memory::Sparse64::Stride1::Load256( const Float * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_ps( pSrc, mIndices, 1 );
}

inline __m256d SIMD::Import::Memory::Sparse64::Stride1::Load256( const Double * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_pd( pSrc, mIndices, 1 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride1::Load256( const Int32 * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_epi32( pSrc, mIndices, 1 );
}
inline __m256i SIMD::Import::Memory::Sparse64::Stride1::Load256( const Int64 * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_epi64( pSrc, mIndices, 1 );
}

inline __m128 SIMD::Import::Memory::Sparse64::Stride1::Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_ps( mDst, pSrc, mIndices, mSigns, 1 );
}

inline __m128d SIMD::Import::Memory::Sparse64::Stride1::Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_pd( mDst, pSrc, mIndices, mSigns, 1 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride1::Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_epi32( mDst, pSrc, mIndices, mSigns, 1 );
}
inline __m128i SIMD::Import::Memory::Sparse64::Stride1::Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_epi64( mDst, pSrc, mIndices, mSigns, 1 );
}

inline __m128 SIMD::Import::Memory::Sparse64::Stride1::Load256( __m128 mDst, const Float * pSrc, __m256i mIndices, __m128 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_ps( mDst, pSrc, mIndices, mSigns, 1 );
}

inline __m256d SIMD::Import::Memory::Sparse64::Stride1::Load256( __m256d mDst, const Double * pSrc, __m256i mIndices, __m256d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_pd( mDst, pSrc, mIndices, mSigns, 1 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride1::Load256( __m128i mDst, const Int32 * pSrc, __m256i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_epi32( mDst, pSrc, mIndices, mSigns, 1 );
}
inline __m256i SIMD::Import::Memory::Sparse64::Stride1::Load256( __m256i mDst, const Int64 * pSrc, __m256i mIndices, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_epi64( mDst, pSrc, mIndices, mSigns, 1 );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Import::Memory::Sparse64::Stride2 implementation
inline __m128 SIMD::Import::Memory::Sparse64::Stride2::Load128( const Float * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_ps( pSrc, mIndices, 2 );
}

inline __m128d SIMD::Import::Memory::Sparse64::Stride2::Load128( const Double * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_pd( pSrc, mIndices, 2 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride2::Load128( const Int32 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_epi32( pSrc, mIndices, 2 );
}
inline __m128i SIMD::Import::Memory::Sparse64::Stride2::Load128( const Int64 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_epi64( pSrc, mIndices, 2 );
}

inline __m128 SIMD::Import::Memory::Sparse64::Stride2::Load256( const Float * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_ps( pSrc, mIndices, 2 );
}

inline __m256d SIMD::Import::Memory::Sparse64::Stride2::Load256( const Double * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_pd( pSrc, mIndices, 2 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride2::Load256( const Int32 * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_epi32( pSrc, mIndices, 2 );
}
inline __m256i SIMD::Import::Memory::Sparse64::Stride2::Load256( const Int64 * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_epi64( pSrc, mIndices, 2 );
}

inline __m128 SIMD::Import::Memory::Sparse64::Stride2::Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_ps( mDst, pSrc, mIndices, mSigns, 2 );
}

inline __m128d SIMD::Import::Memory::Sparse64::Stride2::Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_pd( mDst, pSrc, mIndices, mSigns, 2 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride2::Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_epi32( mDst, pSrc, mIndices, mSigns, 2 );
}
inline __m128i SIMD::Import::Memory::Sparse64::Stride2::Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_epi64( mDst, pSrc, mIndices, mSigns, 2 );
}

inline __m128 SIMD::Import::Memory::Sparse64::Stride2::Load256( __m128 mDst, const Float * pSrc, __m256i mIndices, __m128 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_ps( mDst, pSrc, mIndices, mSigns, 2 );
}

inline __m256d SIMD::Import::Memory::Sparse64::Stride2::Load256( __m256d mDst, const Double * pSrc, __m256i mIndices, __m256d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_pd( mDst, pSrc, mIndices, mSigns, 2 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride2::Load256( __m128i mDst, const Int32 * pSrc, __m256i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_epi32( mDst, pSrc, mIndices, mSigns, 2 );
}
inline __m256i SIMD::Import::Memory::Sparse64::Stride2::Load256( __m256i mDst, const Int64 * pSrc, __m256i mIndices, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_epi64( mDst, pSrc, mIndices, mSigns, 2 );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Import::Memory::Sparse64::Stride4 implementation
inline __m128 SIMD::Import::Memory::Sparse64::Stride4::Load128( const Float * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_ps( pSrc, mIndices, 4 );
}

inline __m128d SIMD::Import::Memory::Sparse64::Stride4::Load128( const Double * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_pd( pSrc, mIndices, 4 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride4::Load128( const Int32 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_epi32( pSrc, mIndices, 4 );
}
inline __m128i SIMD::Import::Memory::Sparse64::Stride4::Load128( const Int64 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_epi64( pSrc, mIndices, 4 );
}

inline __m128 SIMD::Import::Memory::Sparse64::Stride4::Load256( const Float * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_ps( pSrc, mIndices, 4 );
}

inline __m256d SIMD::Import::Memory::Sparse64::Stride4::Load256( const Double * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_pd( pSrc, mIndices, 4 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride4::Load256( const Int32 * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_epi32( pSrc, mIndices, 4 );
}
inline __m256i SIMD::Import::Memory::Sparse64::Stride4::Load256( const Int64 * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_epi64( pSrc, mIndices, 4 );
}

inline __m128 SIMD::Import::Memory::Sparse64::Stride4::Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_ps( mDst, pSrc, mIndices, mSigns, 4 );
}

inline __m128d SIMD::Import::Memory::Sparse64::Stride4::Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_pd( mDst, pSrc, mIndices, mSigns, 4 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride4::Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_epi32( mDst, pSrc, mIndices, mSigns, 4 );
}
inline __m128i SIMD::Import::Memory::Sparse64::Stride4::Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_epi64( mDst, pSrc, mIndices, mSigns, 4 );
}

inline __m128 SIMD::Import::Memory::Sparse64::Stride4::Load256( __m128 mDst, const Float * pSrc, __m256i mIndices, __m128 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_ps( mDst, pSrc, mIndices, mSigns, 4 );
}

inline __m256d SIMD::Import::Memory::Sparse64::Stride4::Load256( __m256d mDst, const Double * pSrc, __m256i mIndices, __m256d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_pd( mDst, pSrc, mIndices, mSigns, 4 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride4::Load256( __m128i mDst, const Int32 * pSrc, __m256i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_epi32( mDst, pSrc, mIndices, mSigns, 4 );
}
inline __m256i SIMD::Import::Memory::Sparse64::Stride4::Load256( __m256i mDst, const Int64 * pSrc, __m256i mIndices, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_epi64( mDst, pSrc, mIndices, mSigns, 4 );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Import::Memory::Sparse64::Stride8 implementation
inline __m128 SIMD::Import::Memory::Sparse64::Stride8::Load128( const Float * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_ps( pSrc, mIndices, 8 );
}

inline __m128d SIMD::Import::Memory::Sparse64::Stride8::Load128( const Double * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_pd( pSrc, mIndices, 8 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride8::Load128( const Int32 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_epi32( pSrc, mIndices, 8 );
}
inline __m128i SIMD::Import::Memory::Sparse64::Stride8::Load128( const Int64 * pSrc, __m128i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_epi64( pSrc, mIndices, 8 );
}

inline __m128 SIMD::Import::Memory::Sparse64::Stride8::Load256( const Float * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_ps( pSrc, mIndices, 8 );
}

inline __m256d SIMD::Import::Memory::Sparse64::Stride8::Load256( const Double * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_pd( pSrc, mIndices, 8 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride8::Load256( const Int32 * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_epi32( pSrc, mIndices, 8 );
}
inline __m256i SIMD::Import::Memory::Sparse64::Stride8::Load256( const Int64 * pSrc, __m256i mIndices ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_epi64( pSrc, mIndices, 8 );
}

inline __m128 SIMD::Import::Memory::Sparse64::Stride8::Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_ps( mDst, pSrc, mIndices, mSigns, 8 );
}

inline __m128d SIMD::Import::Memory::Sparse64::Stride8::Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_pd( mDst, pSrc, mIndices, mSigns, 8 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride8::Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_epi32( mDst, pSrc, mIndices, mSigns, 8 );
}
inline __m128i SIMD::Import::Memory::Sparse64::Stride8::Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_epi64( mDst, pSrc, mIndices, mSigns, 8 );
}

inline __m128 SIMD::Import::Memory::Sparse64::Stride8::Load256( __m128 mDst, const Float * pSrc, __m256i mIndices, __m128 mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_ps( mDst, pSrc, mIndices, mSigns, 8 );
}

inline __m256d SIMD::Import::Memory::Sparse64::Stride8::Load256( __m256d mDst, const Double * pSrc, __m256i mIndices, __m256d mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_pd( mDst, pSrc, mIndices, mSigns, 8 );
}

inline __m128i SIMD::Import::Memory::Sparse64::Stride8::Load256( __m128i mDst, const Int32 * pSrc, __m256i mIndices, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_epi32( mDst, pSrc, mIndices, mSigns, 8 );
}
inline __m256i SIMD::Import::Memory::Sparse64::Stride8::Load256( __m256i mDst, const Int64 * pSrc, __m256i mIndices, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_epi64( mDst, pSrc, mIndices, mSigns, 8 );
}

