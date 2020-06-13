/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_ExportMemory.inl
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
// SIMD::Export::Memory implementation
inline Void SIMD::Export::Memory::SaveOne( Float * outDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_store_ss( outDst, mSrc );
}
inline Void SIMD::Export::Memory::SaveOne( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_sd( outDst, mSrc );
}

inline Void SIMD::Export::Memory::SaveOneDoubleL( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storel_pd( outDst, mSrc );
}
inline Void SIMD::Export::Memory::SaveOneDoubleH( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeh_pd( outDst, mSrc );
}

inline Void SIMD::Export::Memory::SaveOneInt64L( __m128i * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storel_epi64( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Save128( Float * outDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_storeu_ps( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Save128( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeu_pd( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Save128( Int8 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeu_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Save128( Int16 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeu_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Save128( Int32 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeu_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Save128( Int64 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeu_si128( (__m128i*)outDst, mSrc );
}

inline Void SIMD::Export::Memory::Save128( UInt8 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeu_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Save128( UInt16 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeu_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Save128( UInt32 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeu_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Save128( UInt64 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeu_si128( (__m128i*)outDst, mSrc );
}

inline Void SIMD::Export::Memory::Save256( Float * outDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_ps( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Save256( Double * outDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_pd( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Save256( Int8 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Save256( Int16 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Save256( Int32 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Save256( Int64 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_si256( (__m256i*)outDst, mSrc );
}

inline Void SIMD::Export::Memory::Save256( UInt8 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Save256( UInt16 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Save256( UInt32 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Save256( UInt64 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_si256( (__m256i*)outDst, mSrc );
}

inline Void SIMD::Export::Memory::Save128( Float * outDst, __m128 mSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm_maskstore_ps( outDst, mSigns, mSrc );
}

inline Void SIMD::Export::Memory::Save128( Double * outDst, __m128d mSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm_maskstore_pd( outDst, mSigns, mSrc );
}

inline Void SIMD::Export::Memory::Save128( Int32 * outDst, __m128i mSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    _mm_maskstore_epi32( outDst, mSigns, mSrc );
}
inline Void SIMD::Export::Memory::Save128( Int64 * outDst, __m128i mSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    _mm_maskstore_epi64( outDst, mSigns, mSrc );
}

inline Void SIMD::Export::Memory::Save256( Float * outDst, __m256 mSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_maskstore_ps( outDst, mSigns, mSrc );
}

inline Void SIMD::Export::Memory::Save256( Double * outDst, __m256d mSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_maskstore_pd( outDst, mSigns, mSrc );
}

inline Void SIMD::Export::Memory::Save256( Int32 * outDst, __m256i mSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    _mm256_maskstore_epi32( outDst, mSigns, mSrc );
}
inline Void SIMD::Export::Memory::Save256( Int64 * outDst, __m256i mSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    _mm256_maskstore_epi64( outDst, mSigns, mSrc );
}

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Export::Memory::Aligned implementation
inline Void SIMD::Export::Memory::Aligned::Save128( Float * outDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_store_ps( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save128( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_pd( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save128( Int8 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save128( Int16 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save128( Int32 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save128( Int64 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_si128( (__m128i*)outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save128( UInt8 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save128( UInt16 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save128( UInt32 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save128( UInt64 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_si128( (__m128i*)outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save256( Float * outDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_ps( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save256( Double * outDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_pd( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save256( Int8 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save256( Int16 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save256( Int32 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save256( Int64 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_si256( (__m256i*)outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save256( UInt8 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save256( UInt16 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save256( UInt32 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save256( UInt64 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_si256( (__m256i*)outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save128R( Float * outDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_storer_ps( outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save128R( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storer_pd( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::SaveOneNT( Float * outDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    _mm_stream_ss( outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::SaveOneNT( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    _mm_stream_sd( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save128NT( Float * outDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_stream_ps( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save128NT( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_stream_pd( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save128NT( Int8 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_stream_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save128NT( Int16 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_stream_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save128NT( Int32 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_stream_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save128NT( Int64 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_stream_si128( (__m128i*)outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save128NT( UInt8 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_stream_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save128NT( UInt16 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_stream_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save128NT( UInt32 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_stream_si128( (__m128i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save128NT( UInt64 * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_stream_si128( (__m128i*)outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save256NT( Float * outDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_stream_ps( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save256NT( Double * outDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_stream_pd( outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save256NT( Int8 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_stream_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save256NT( Int16 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_stream_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save256NT( Int32 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_stream_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save256NT( Int64 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_stream_si256( (__m256i*)outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Save256NT( UInt8 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_stream_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save256NT( UInt16 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_stream_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save256NT( UInt32 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_stream_si256( (__m256i*)outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Save256NT( UInt64 * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_stream_si256( (__m256i*)outDst, mSrc );
}

inline Void SIMD::Export::Memory::Aligned::Spread128( Float * outDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_store1_ps( outDst, mSrc );
}
inline Void SIMD::Export::Memory::Aligned::Spread128( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store1_pd( outDst, mSrc );
}

