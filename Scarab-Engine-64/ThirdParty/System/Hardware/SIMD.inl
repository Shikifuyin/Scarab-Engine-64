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

////////////////////////////////////////////////////////////// Control instructions
inline UInt32 SIMD::GetCSR() {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_getcsr();
}
inline Void SIMD::SetCSR( UInt32 iValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_setcsr( iValue );
}

inline Void SIMD::ClearAndFlushCacheLine( Void * pAddress ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_clflush( pAddress );
}

inline Void SIMD::Pause() {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_pause();
}

////////////////////////////////////////////////////////////// Serializing instruction (makes sure everything is flushed)
inline Void SIMD::SerializeMemoryStore() {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_sfence();
}
inline Void SIMD::SerializeMemoryLoad() {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_lfence();
}
inline Void SIMD::SerializeMemory() {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_mfence();
}

////////////////////////////////////////////////////////////// Register Initialization
inline __m128 SIMD::Zero128F() {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_setzero_ps();
}
inline __m256 SIMD::Zero256F() {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_setzero_ps();
}

inline __m128d SIMD::Zero128D() {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_setzero_pd();
}
inline __m256d SIMD::Zero256D() {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_setzero_pd();
}

inline __m128i SIMD::Zero128I() {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_setzero_si128();
}
inline __m256i SIMD::Zero256I() {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_setzero_si256();
}

inline Void SIMD::ZeroUpper128() {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_zeroupper();
}
inline Void SIMD::Zero256() {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_zeroall();
}

////////////////////////////////////////////////////////////// Values -> Registers
inline __m128 SIMD::SetLower( Float f0 ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_set_ss( f0 );
}
inline __m128d SIMD::SetLower( Double f0 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_sd( f0 );
}

inline __m128i SIMD::SetLower( Int32 i0 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi32_si128( i0 );
}
inline __m128i SIMD::SetLower( Int64 i0 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi64_si128( i0 );
}

inline __m128 SIMD::Set128( Float f0, Float f1, Float f2, Float f3 ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_set_ps( f0, f1, f2, f3 );
}
inline __m256 SIMD::Set256( Float f0, Float f1, Float f2, Float f3, Float f4, Float f5, Float f6, Float f7 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_ps( f7, f6, f5, f4, f3, f2, f1, f0 );
}

inline __m128d SIMD::Set128( Double f0, Double f1 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_pd( f0, f1 );
}
inline __m256d SIMD::Set256( Double f0, Double f1, Double f2, Double f3 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_pd( f3, f2, f1, f0 );
}

inline __m128i SIMD::Set128( Int8 i0, Int8 i1, Int8 i2, Int8 i3, Int8 i4, Int8 i5, Int8 i6, Int8 i7,
                                 Int8 i8, Int8 i9, Int8 i10, Int8 i11, Int8 i12, Int8 i13, Int8 i14, Int8 i15 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_epi8( i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0 );
}
inline __m256i SIMD::Set256( Int8 i0, Int8 i1, Int8 i2, Int8 i3, Int8 i4, Int8 i5, Int8 i6, Int8 i7,
                                 Int8 i8, Int8 i9, Int8 i10, Int8 i11, Int8 i12, Int8 i13, Int8 i14, Int8 i15,
                                 Int8 i16, Int8 i17, Int8 i18, Int8 i19, Int8 i20, Int8 i21, Int8 i22, Int8 i23,
                                 Int8 i24, Int8 i25, Int8 i26, Int8 i27, Int8 i28, Int8 i29, Int8 i30, Int8 i31 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_epi8( i31, i30, i29, i28, i27, i26, i25, i24, i23, i22, i21, i20, i19, i18, i17, i16,
                            i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0 );
}

inline __m128i SIMD::Set128( Int16 i0, Int16 i1, Int16 i2, Int16 i3, Int16 i4, Int16 i5, Int16 i6, Int16 i7 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_epi16( i7, i6, i5, i4, i3, i2, i1, i0 );
}
inline __m256i SIMD::Set256( Int16 i0, Int16 i1, Int16 i2, Int16 i3, Int16 i4, Int16 i5, Int16 i6, Int16 i7,
                                 Int16 i8, Int16 i9, Int16 i10, Int16 i11, Int16 i12, Int16 i13, Int16 i14, Int16 i15 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_epi16( i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0 );
}

inline __m128i SIMD::Set128( Int32 i0, Int32 i1, Int32 i2, Int32 i3 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_epi32( i3, i2, i1, i0 );
}
inline __m256i SIMD::Set256( Int32 i0, Int32 i1, Int32 i2, Int32 i3, Int32 i4, Int32 i5, Int32 i6, Int32 i7 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_epi32( i7, i6, i5, i4, i3, i2, i1, i0 );
}

inline __m128i SIMD::Set128( Int64 i0, Int64 i1 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_epi64x( i1, i0 );
}
inline __m256i SIMD::Set256( Int64 i0, Int64 i1, Int64 i2, Int64 i3 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_epi64x( i3, i2, i1, i0 );
}

//inline __m128 SIMD::SetFloat( __m128 mDst, Float fSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_insert_ps( mDst, fSrc, iIndex );
//}
//inline __m256 SIMD::SetFloat( __m256 mDst, Float fSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_insert_ps( mDst, fSrc, iIndex );
//}
//
//inline __m128d SIMD::SetDouble( __m128d mDst, Double fSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_insert_pd( mDst, fSrc, iIndex );
//}
//inline __m256d SIMD::SetDouble( __m256d mDst, Double fSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_insert_pd( mDst, fSrc, iIndex );
//}

//inline __m128i SIMD::SetInt8( __m128i mDst, Int8 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_insert_epi8( mDst, iSrc, iIndex );
//}
//inline __m256i SIMD::SetInt8( __m256i mDst, Int8 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_insert_epi8( mDst, iSrc, iIndex );
//}

//inline __m128i SIMD::SetInt16( __m128i mDst, Int16 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_insert_epi16( mDst, iSrc, iIndex );
//}
//inline __m256i SIMD::SetInt16( __m256i mDst, Int16 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_insert_epi16( mDst, iSrc, iIndex );
//}

//inline __m128i SIMD::SetInt32( __m128i mDst, Int32 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_insert_epi32( mDst, iSrc, iIndex );
//}
//inline __m256i SIMD::SetInt32( __m256i mDst, Int32 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_insert_epi32( mDst, iSrc, iIndex );
//}

//inline __m128i SIMD::SetInt64( __m128i mDst, Int64 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_insert_epi64( mDst, iSrc, iIndex );
//}
//inline __m256i SIMD::SetInt64( __m256i mDst, Int64 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_insert_epi64( mDst, iSrc, iIndex );
//}

inline __m128 SIMD::Spread128( Float fValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_set1_ps( fValue );
}
inline __m256 SIMD::Spread256( Float fValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_ps( fValue );
}

inline __m128d SIMD::Spread128( Double fValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_pd( fValue );
}
inline __m256d SIMD::Spread256( Double fValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_pd( fValue );
}

inline __m128i SIMD::Spread128( Int8 iValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_epi8( iValue );
}
inline __m256i SIMD::Spread256( Int8 iValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_epi8( iValue );
}

inline __m128i SIMD::Spread128( Int16 iValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_epi16( iValue );
}
inline __m256i SIMD::Spread256( Int16 iValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_epi16( iValue );
}

inline __m128i SIMD::Spread128( Int32 iValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_epi32( iValue );
}
inline __m256i SIMD::Spread256( Int32 iValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_epi32( iValue );
}

inline __m128i SIMD::Spread128( Int64 iValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_epi64x( iValue );
}
inline __m256i SIMD::Spread256( Int64 iValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_epi64x( iValue );
}

////////////////////////////////////////////////////////////// Registers -> Values
inline Float SIMD::GetLower( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtss_f32( mSrc );
}
inline Float SIMD::GetLower( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtss_f32( mSrc );
}

inline Double SIMD::GetLower( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsd_f64( mSrc );
}
inline Double SIMD::GetLower( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtsd_f64( mSrc );
}

inline Int32 SIMD::GetLower32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi128_si32( mSrc );
}
inline Int32 SIMD::GetLower32( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtsi256_si32( mSrc );
}

inline Int64 SIMD::GetLower64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi128_si64( mSrc );
}
inline Int64 SIMD::GetLower64( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtsi256_si64( mSrc );
}

//inline Float SIMD::GetFloat( __m128 mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    FloatConverter hConv;
//    hConv.i = _mm_extract_ps( mSrc, iIndex );
//    return hConv.f;
//}
//inline Float SIMD::GetFloat( __m256 mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    FloatConverter hConv;
//    hConv.i = _mm256_extract_ps( mSrc, iIndex );
//    return hConv.f;
//}

//inline Double SIMD::GetDouble( __m128d mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    FloatConverter hConv;
//    hConv.i = _mm_extract_pd( mSrc, iIndex );
//    return hConv.f;
//}
//inline Double SIMD::GetDouble( __m256d mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    FloatConverter hConv;
//    hConv.i = _mm256_extract_pd( mSrc, iIndex );
//    return hConv.f;
//}

//inline Int32 SIMD::GetInt8( __m128i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_extract_epi8( mSrc, iIndex );
//}
//inline Int32 SIMD::GetInt8( __m256i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_extract_epi8( mSrc, iIndex );
//}

//inline Int32 SIMD::GetInt16( __m128i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_extract_epi16( mSrc, iIndex );
//}
//inline Int32 SIMD::GetInt16( __m256i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_extract_epi16( mSrc, iIndex );
//}

//inline Int32 SIMD::GetInt32( __m128i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_extract_epi32( mSrc, iIndex );
//}
//inline Int32 SIMD::GetInt32( __m256i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_extract_epi32( mSrc, iIndex );
//}

//inline Int64 SIMD::GetInt64( __m128i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_extract_epi64( mSrc, iIndex );
//}
//inline Int64 SIMD::GetInt64( __m256i mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_extract_epi64( mSrc, iIndex );
//}

////////////////////////////////////////////////////////////// Memory -> Registers
inline __m128 SIMD::LoadLower( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_load_ss( pSrc );
}
inline __m128d SIMD::LoadLower( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_sd( pSrc );
}

inline __m128 SIMD::Load128Aligned( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_load_ps( pSrc );
}
inline __m256 SIMD::Load256Aligned( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_ps( pSrc );
}

inline __m128d SIMD::Load128Aligned( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_pd( pSrc );
}
inline __m256d SIMD::Load256Aligned( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_pd( pSrc );
}

inline __m128i SIMD::Load128Aligned( const __m128i * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_si128( pSrc );
}
inline __m256i SIMD::Load256Aligned( const __m256i * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_si256( pSrc );
}

inline __m128 SIMD::Load128( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_loadu_ps( pSrc );
}
inline __m128 SIMD::Load128( const Float * pSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_maskload_ps( pSrc, mSigns );
}
inline __m256 SIMD::Load256( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_loadu_ps( pSrc );
}
inline __m256 SIMD::Load256( const Float * pSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_maskload_ps( pSrc, mSigns );
}

inline __m128d SIMD::Load128( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadu_pd( pSrc );
}
inline __m128d SIMD::Load128( const Double * pSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_maskload_pd( pSrc, mSigns );
}
inline __m256d SIMD::Load256( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_loadu_pd( pSrc );
}
inline __m256d SIMD::Load256( const Double * pSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_maskload_pd( pSrc, mSigns );
}

inline __m128i SIMD::Load128( const Int32 * pSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_maskload_epi32( pSrc, mSigns );
}
inline __m256i SIMD::Load256( const Int32 * pSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_maskload_epi32( pSrc, mSigns );
}

inline __m128i SIMD::Load128( const Int64 * pSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_maskload_epi64( pSrc, mSigns );
}
inline __m256i SIMD::Load256( const Int64 * pSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_maskload_epi64( pSrc, mSigns );
}

inline __m128i SIMD::Load128( const __m128i * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_lddqu_si128( pSrc );
}
inline __m256i SIMD::Load256( const __m256i * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_lddqu_si256( pSrc );
}

inline __m128 SIMD::Load128AlignedR( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_loadr_ps( pSrc );
}
inline __m128d SIMD::Load128AlignedR( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadr_pd( pSrc );
}

inline __m128d SIMD::LoadOneDoubleL( __m128d mDst, const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadl_pd( mDst, pSrc );
}
inline __m128d SIMD::LoadOneDoubleH( __m128d mDst, const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadh_pd( mDst, pSrc );
}

inline __m128i SIMD::LoadOneInt64L( const __m128i * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadl_epi64( pSrc );
}

//inline __m128 SIMD::Spread128( const Float * pSrc ) {
//    DebugAssert( CPUIDFn->HasSSE() );
//    return _mm_load1_ps( pSrc );
//}
inline __m128 SIMD::Spread128( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_broadcast_ss( pSrc );
}
inline __m256 SIMD::Spread256( const Float * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_broadcast_ss( pSrc );
}

inline __m128d SIMD::Spread128( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_loaddup_pd( pSrc );
}
inline __m256d SIMD::Spread256( const Double * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_broadcast_sd( pSrc );
}

inline __m256 SIMD::Spread256( const __m128 * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_broadcast_ps( pSrc );
}
inline __m256d SIMD::Spread256( const __m128d * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_broadcast_pd( pSrc );
}

inline __m128i SIMD::LoadNT128Aligned( const __m128i * pSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_stream_load_si128( pSrc );
}
inline __m256i SIMD::LoadNT256Aligned( const __m256i * pSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_stream_load_si256( pSrc );
}

//inline __m128 SIMD::Load32FourFloat( const Float * pSrc, __m128i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_i32gather_ps( pSrc, mIndices, iStride );
//}
//inline __m128 SIMD::Load32FourFloat( __m128 mDst, const Float * pSrc, __m128i mIndices, Int32 iStride, __m128 mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_mask_i32gather_ps( mDst, pSrc, mIndices, mSigns, iStride );
//}
//inline __m256 SIMD::Load32EightFloat( const Float * pSrc, __m256i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_i32gather_ps( pSrc, mIndices, iStride );
//}
//inline __m256 SIMD::Load32EightFloat( __m256 mDst, const Float * pSrc, __m256i mIndices, Int32 iStride, __m256 mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_mask_i32gather_ps( mDst, pSrc, mIndices, mSigns, iStride );
//}

//inline __m128d SIMD::Load32TwoDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_i32gather_pd( pSrc, mIndices, iStride );
//}
//inline __m128d SIMD::Load32TwoDouble( __m128d mDst, const Double * pSrc, __m128i mIndices, Int32 iStride, __m128d mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_mask_i32gather_pd( mDst, pSrc, mIndices, mSigns, iStride );
//}
//inline __m256d SIMD::Load32FourDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_i32gather_pd( pSrc, mIndices, iStride );
//}
//inline __m256d SIMD::Load32FourDouble( __m256d mDst, const Double * pSrc, __m128i mIndices, Int32 iStride, __m256d mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_mask_i32gather_pd( mDst, pSrc, mIndices, mSigns, iStride );
//}

//inline __m128i SIMD::Load32FourInt32( const Int32 * pSrc, __m128i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_i32gather_epi32( pSrc, mIndices, iStride );
//}
//inline __m128i SIMD::Load32FourInt32( __m128i mDst, const Int32 * pSrc, __m128i mIndices, Int32 iStride, __m128i mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_mask_i32gather_epi32( mDst, pSrc, mIndices, mSigns, iStride );
//}
//inline __m256i SIMD::Load32EightInt32( const Int32 * pSrc, __m256i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_i32gather_epi32( pSrc, mIndices, iStride );
//}
//inline __m256i SIMD::Load32EightInt32( __m256i mDst, const Int32 * pSrc, __m256i mIndices, Int32 iStride, __m256i mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_mask_i32gather_epi32( mDst, pSrc, mIndices, mSigns, iStride );
//}

//inline __m128i SIMD::Load32TwoInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_i32gather_epi64( pSrc, mIndices, iStride );
//}
//inline __m128i SIMD::Load32TwoInt64( __m128i mDst, const Int64 * pSrc, __m128i mIndices, Int32 iStride, __m128i mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_mask_i32gather_epi64( mDst, pSrc, mIndices, mSigns, iStride );
//}
//inline __m256i SIMD::Load32FourInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_i32gather_epi64( pSrc, mIndices, iStride );
//}
//inline __m256i SIMD::Load32FourInt64( __m256i mDst, const Int64 * pSrc, __m128i mIndices, Int32 iStride, __m256i mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_mask_i32gather_epi64( mDst, pSrc, mIndices, mSigns, iStride );
//}

//inline __m128 SIMD::Load64TwoFloat( const Float * pSrc, __m128i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_i64gather_ps( pSrc, mIndices, iStride );
//}
//inline __m128 SIMD::Load64TwoFloat( __m128 mDst, const Float * pSrc, __m128i mIndices, Int32 iStride, __m128 mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_mask_i64gather_ps( mDst, pSrc, mIndices, mSigns, iStride );
//}
//inline __m128 SIMD::Load64FourFloat( const Float * pSrc, __m256i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_i64gather_ps( pSrc, mIndices, iStride );
//}
//inline __m128 SIMD::Load64FourFloat( __m128 mDst, const Float * pSrc, __m256i mIndices, Int32 iStride, __m128 mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_mask_i64gather_ps( mDst, pSrc, mIndices, mSigns, iStride );
//}

//inline __m128d SIMD::Load64TwoDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_i64gather_pd( pSrc, mIndices, iStride );
//}
//inline __m128d SIMD::Load64TwoDouble( __m128d mDst, const Double * pSrc, __m128i mIndices, Int32 iStride, __m128d mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_mask_i64gather_pd( mDst, pSrc, mIndices, mSigns, iStride );
//}
//inline __m256d SIMD::Load64FourDouble( const Double * pSrc, __m256i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_i64gather_pd( pSrc, mIndices, iStride );
//}
//inline __m256d SIMD::Load64FourDouble( __m256d mDst, const Double * pSrc, __m256i mIndices, Int32 iStride, __m256d mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_mask_i64gather_pd( mDst, pSrc, mIndices, mSigns, iStride );
//}

//inline __m128i SIMD::Load64TwoInt32( const Int32 * pSrc, __m128i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_i64gather_epi32( pSrc, mIndices, iStride );
//}
//inline __m128i SIMD::Load64TwoInt32( __m128i mDst, const Int32 * pSrc, __m128i mIndices, Int32 iStride, __m128i mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_mask_i64gather_epi32( mDst, pSrc, mIndices, mSigns, iStride );
//}
//inline __m128i SIMD::Load64FourInt32( const Int32 * pSrc, __m256i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_i64gather_epi32( pSrc, mIndices, iStride );
//}
//inline __m128i SIMD::Load64FourInt32( __m128i mDst, const Int32 * pSrc, __m256i mIndices, Int32 iStride, __m128i mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_mask_i64gather_epi32( mDst, pSrc, mIndices, mSigns, iStride );
//}

//inline __m128i SIMD::Load64TwoInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_i64gather_epi64( pSrc, mIndices, iStride );
//}
//inline __m128i SIMD::Load64TwoInt64( __m128i mDst, const Int64 * pSrc, __m128i mIndices, Int32 iStride, __m128i mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm_mask_i64gather_epi64( mDst, pSrc, mIndices, mSigns, iStride );
//}
//inline __m256i SIMD::Load64FourInt64( const Int64 * pSrc, __m256i mIndices, Int32 iStride ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_i64gather_epi64( pSrc, mIndices, iStride );
//}
//inline __m256i SIMD::Load64FourInt64( __m256i mDst, const Int64 * pSrc, __m256i mIndices, Int32 iStride, __m256i mSigns ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_mask_i64gather_epi64( mDst, pSrc, mIndices, mSigns, iStride );
//}

////////////////////////////////////////////////////////////// Registers -> Memory
inline Void SIMD::StoreLower( Float * outDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_store_ss( outDst, mSrc );
}
inline Void SIMD::StoreLower( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_sd( outDst, mSrc );
}

inline Void SIMD::Store128Aligned( Float * outDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_store_ps( outDst, mSrc );
}
inline Void SIMD::Store256Aligned( Float * outDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_ps( outDst, mSrc );
}

inline Void SIMD::Store128Aligned( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_pd( outDst, mSrc );
}
inline Void SIMD::Store256Aligned( Double * outDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_pd( outDst, mSrc );
}

inline Void SIMD::Store128Aligned( __m128i * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_si128( outDst, mSrc );
}
inline Void SIMD::Store256Aligned( __m256i * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_si256( outDst, mSrc );
}

inline Void SIMD::Store128( Float * outDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_storeu_ps( outDst, mSrc );
}
inline Void SIMD::Store128( Float * outDst, __m128 mSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm_maskstore_ps( outDst, mSigns, mSrc );
}
inline Void SIMD::Store256( Float * outDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_ps( outDst, mSrc );
}
inline Void SIMD::Store256( Float * outDst, __m256 mSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_maskstore_ps( outDst, mSigns, mSrc );
}

inline Void SIMD::Store128( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeu_pd( outDst, mSrc );
}
inline Void SIMD::Store128( Double * outDst, __m128d mSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm_maskstore_pd( outDst, mSigns, mSrc );
}
inline Void SIMD::Store256( Double * outDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_pd( outDst, mSrc );
}
inline Void SIMD::Store256( Double * outDst, __m256d mSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_maskstore_pd( outDst, mSigns, mSrc );
}

inline Void SIMD::Store128( Int32 * outDst, __m128i mSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    _mm_maskstore_epi32( outDst, mSigns, mSrc );
}
inline Void SIMD::Store256( Int32 * outDst, __m256i mSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    _mm256_maskstore_epi32( outDst, mSigns, mSrc );
}

inline Void SIMD::Store128( Int64 * outDst, __m128i mSrc, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    _mm_maskstore_epi64( outDst, mSigns, mSrc );
}
inline Void SIMD::Store256( Int64 * outDst, __m256i mSrc, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    _mm256_maskstore_epi64( outDst, mSigns, mSrc );
}

inline Void SIMD::Store128( __m128i * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeu_si128( outDst, mSrc );
}
inline Void SIMD::Store256( __m256i * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_si256( outDst, mSrc );
}

inline Void SIMD::Store128AlignedR( Float * outDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_storer_ps( outDst, mSrc );
}
inline Void SIMD::Store128AlignedR( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storer_pd( outDst, mSrc );
}

inline Void SIMD::StoreOneDoubleL( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storel_pd( outDst, mSrc );
}
inline Void SIMD::StoreOneDoubleH( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeh_pd( outDst, mSrc );
}

inline Void SIMD::StoreOneInt64L( __m128i * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storel_epi64( outDst, mSrc );
}

inline Void SIMD::Spread128( Float * outDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_store1_ps( outDst, mSrc );
}
inline Void SIMD::Spread128( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store1_pd( outDst, mSrc );
}

inline Void SIMD::StoreNTLower( Float * outDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    _mm_stream_ss( outDst, mSrc );
}
inline Void SIMD::StoreNTLower( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    _mm_stream_sd( outDst, mSrc );
}

inline Void SIMD::StoreNT128Aligned( Float * outDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_stream_ps( outDst, mSrc );
}
inline Void SIMD::StoreNT256Aligned( Float * outDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_stream_ps( outDst, mSrc );
}

inline Void SIMD::StoreNT128Aligned( Double * outDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_stream_pd( outDst, mSrc );
}
inline Void SIMD::StoreNT256Aligned( Double * outDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_stream_pd( outDst, mSrc );
}

inline Void SIMD::StoreNT128Aligned( __m128i * outDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_stream_si128( outDst, mSrc );
}
inline Void SIMD::StoreNT256Aligned( __m256i * outDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_stream_si256( outDst, mSrc );
}

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

inline __m128d SIMD::MoveOneDoubleLL( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_move_sd( mDst, mSrc );
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

//inline __m128 SIMD::Extract128F( __m256 mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_extractf128_ps( mSrc, iIndex );
//}
//inline __m128d SIMD::Extract128D( __m256d mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_extractf128_pd( mSrc, iIndex );
//}
inline __m128i SIMD::Extract128I( __m256i mSrc, Int32 iIndex ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_extracti128_si256( mSrc, iIndex );
}

//inline __m256 SIMD::Insert128F( __m256 mDst, __m128 mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_insertf128_ps( mDst, mSrc, iIndex );
//}
//inline __m256d SIMD::Insert128D( __m256d mDst, __m128d mSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_insertf128_pd( mDst, mSrc, iIndex );
//}
inline __m256i SIMD::Insert128I( __m256i mDst, __m128i mSrc, Int32 iIndex ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_inserti128_si256( mDst, mSrc, iIndex );
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

////////////////////////////////////////////////////////////// Cast (Free, 0 instruction generated)
inline __m128 SIMD::CastToFloat( __m128d mDouble ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castpd_ps( mDouble );
}
inline __m128 SIMD::CastToFloat( __m128i mInteger ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castsi128_ps( mInteger );
}
inline __m256 SIMD::CastToFloat( __m256d mDouble ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castpd_ps( mDouble );
}
inline __m256 SIMD::CastToFloat( __m256i mInteger ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castsi256_ps( mInteger );
}

inline __m128d SIMD::CastToDouble( __m128 mFloat ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castps_pd( mFloat );
}
inline __m128d SIMD::CastToDouble( __m128i mInteger ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castsi128_pd( mInteger );
}
inline __m256d SIMD::CastToDouble( __m256 mFloat ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castps_pd( mFloat );
}
inline __m256d SIMD::CastToDouble( __m256i mInteger ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castsi256_pd( mInteger );
}

inline __m128i SIMD::CastToInteger( __m128 mFloat ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castps_si128( mFloat );
}
inline __m128i SIMD::CastToInteger( __m128d mDouble ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castpd_si128( mDouble );
}
inline __m256i SIMD::CastToInteger( __m256 mFloat ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castps_si256( mFloat );
}
inline __m256i SIMD::CastToInteger( __m256d mDouble ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castpd_si256( mDouble );
}

inline __m128 SIMD::CastDown( __m256 mFloat ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castps256_ps128( mFloat );
}
inline __m128d SIMD::CastDown( __m256d mDouble ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castpd256_pd128( mDouble );
}
inline __m128i SIMD::CastDown( __m256i mInteger ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castsi256_si128( mInteger );
}

inline __m256 SIMD::CastUp( __m128 mFloat ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castps128_ps256( mFloat );
}
inline __m256d SIMD::CastUp( __m128d mDouble ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castpd128_pd256( mDouble );
}
inline __m256i SIMD::CastUp( __m128i mInteger ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castsi128_si256( mInteger );
}

////////////////////////////////////////////////////////////// Convert & Copy
inline __m128 SIMD::ConvertLower( __m128 mDst, Int32 iSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtsi32_ss( mDst, iSrc );
}
inline __m128 SIMD::ConvertLower( __m128 mDst, Int64 iSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtsi64_ss( mDst, iSrc );
}

inline __m128d SIMD::ConvertLower( __m128d mDst, Int32 iSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi32_sd( mDst, iSrc );
}
inline __m128d SIMD::ConvertLower( __m128d mDst, Int64 iSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi64_sd( mDst, iSrc );
}

inline __m128 SIMD::ConvertLower( __m128 mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsd_ss( mDst, mSrc );
}
inline __m128d SIMD::ConvertLower( __m128d mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtss_sd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Convert
inline Int32 SIMD::ConvertLowerToInt32( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtss_si32( mSrc );
}
inline Int32 SIMD::ConvertLowerToInt32( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsd_si32( mSrc );
}

inline Int64 SIMD::ConvertLowerToInt64( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtss_si64( mSrc );
}
inline Int64 SIMD::ConvertLowerToInt64( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsd_si64( mSrc );
}

inline __m128 SIMD::Convert128ToFloat( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtpd_ps( mSrc );
}
inline __m128 SIMD::Convert128ToFloat( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtpd_ps( mSrc );
}

inline __m128 SIMD::Convert128ToFloat( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtepi32_ps( mSrc );
}
inline __m256 SIMD::Convert256ToFloat( __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtepi32_ps( mSrc );
}

inline __m128d SIMD::Convert128ToDouble( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtps_pd( mSrc );
}
inline __m256d SIMD::Convert256ToDouble( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtps_pd( mSrc );
}

inline __m128d SIMD::Convert128ToDouble( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtepi32_pd( mSrc );
}
inline __m256d SIMD::Convert256ToDouble( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtepi32_pd( mSrc );
}

inline __m128i SIMD::Convert128ToInt32( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtps_epi32( mSrc );
}
inline __m256i SIMD::Convert256ToInt32( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtps_epi32( mSrc );
}

inline __m128i SIMD::Convert128ToInt32( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtpd_epi32( mSrc );
}
inline __m128i SIMD::Convert128ToInt32( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtpd_epi32( mSrc );
}

////////////////////////////////////////////////////////////// Truncate
inline Int32 SIMD::TruncateLowerToInt32( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvttss_si32( mSrc );
}
inline Int32 SIMD::TruncateLowerToInt32( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvttsd_si32( mSrc );
}

inline Int64 SIMD::TruncateLowerToInt64( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvttss_si64( mSrc );
}
inline Int64 SIMD::TruncateLowerToInt64( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvttsd_si64( mSrc );
}

inline __m128i SIMD::TruncateToInt32( __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvttps_epi32( mSrc );
}
inline __m256i SIMD::TruncateToInt32( __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvttps_epi32( mSrc );
}

inline __m128i SIMD::TruncateToInt32( __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvttpd_epi32( mSrc );
}
inline __m128i SIMD::TruncateToInt32( __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvttpd_epi32( mSrc );
}

////////////////////////////////////////////////////////////// Sign-Extend
inline __m128i SIMD::SignExtend128Int8To16( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi8_epi16( mSrc );
}
inline __m128i SIMD::SignExtend128Int8To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi8_epi32( mSrc );
}
inline __m128i SIMD::SignExtend128Int8To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi8_epi64( mSrc );
}
inline __m128i SIMD::SignExtend128Int16To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi16_epi32( mSrc );
}
inline __m128i SIMD::SignExtend128Int16To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi16_epi64( mSrc );
}
inline __m128i SIMD::SignExtend128Int32To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi32_epi64( mSrc );
}

inline __m256i SIMD::SignExtend256Int8To16( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi8_epi16( mSrc );
}
inline __m256i SIMD::SignExtend256Int8To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi8_epi32( mSrc );
}
inline __m256i SIMD::SignExtend256Int8To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi8_epi64( mSrc );
}
inline __m256i SIMD::SignExtend256Int16To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi16_epi32( mSrc );
}
inline __m256i SIMD::SignExtend256Int16To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi16_epi64( mSrc );
}
inline __m256i SIMD::SignExtend256Int32To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi32_epi64( mSrc );
}

////////////////////////////////////////////////////////////// Zero-Extend
inline __m128i SIMD::ZeroExtend128Int8To16( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu8_epi16( mSrc );
}
inline __m128i SIMD::ZeroExtend128Int8To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu8_epi32( mSrc );
}
inline __m128i SIMD::ZeroExtend128Int8To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu8_epi64( mSrc );
}
inline __m128i SIMD::ZeroExtend128Int16To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu16_epi32( mSrc );
}
inline __m128i SIMD::ZeroExtend128Int16To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu16_epi64( mSrc );
}
inline __m128i SIMD::ZeroExtend128Int32To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu32_epi64( mSrc );
}

inline __m256i SIMD::ZeroExtend256Int8To16( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu8_epi16( mSrc );
}
inline __m256i SIMD::ZeroExtend256Int8To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu8_epi32( mSrc );
}
inline __m256i SIMD::ZeroExtend256Int8To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu8_epi64( mSrc );
}
inline __m256i SIMD::ZeroExtend256Int16To32( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu16_epi32( mSrc );
}
inline __m256i SIMD::ZeroExtend256Int16To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu16_epi64( mSrc );
}
inline __m256i SIMD::ZeroExtend256Int32To64( __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu32_epi64( mSrc );
}

////////////////////////////////////////////////////////////// Absolute Value
inline __m128i SIMD::Abs8( __m128i mValue ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_abs_epi8( mValue );
}
inline __m256i SIMD::Abs8( __m256i mValue ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_abs_epi8( mValue );
}

inline __m128i SIMD::Abs16( __m128i mValue ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_abs_epi16( mValue );
}
inline __m256i SIMD::Abs16( __m256i mValue ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_abs_epi16( mValue );
}

inline __m128i SIMD::Abs32( __m128i mValue ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_abs_epi32( mValue );
}
inline __m256i SIMD::Abs32( __m256i mValue ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_abs_epi32( mValue );
}

inline __m128i SIMD::Abs64( __m128i mValue ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_abs_epi64( mValue );
}
inline __m256i SIMD::Abs64( __m256i mValue ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_abs_epi64( mValue );
}

////////////////////////////////////////////////////////////// Sign Change
inline __m128i SIMD::Negate8( __m128i mValue, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_sign_epi8( mValue, mSigns );
}
inline __m256i SIMD::Negate8( __m256i mValue, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sign_epi8( mValue, mSigns );
}

inline __m128i SIMD::Negate16( __m128i mValue, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_sign_epi16( mValue, mSigns );
}
inline __m256i SIMD::Negate16( __m256i mValue, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sign_epi16( mValue, mSigns );
}

inline __m128i SIMD::Negate32( __m128i mValue, __m128i mSigns ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_sign_epi32( mValue, mSigns );
}
inline __m256i SIMD::Negate32( __m256i mValue, __m256i mSigns ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sign_epi32( mValue, mSigns );
}

////////////////////////////////////////////////////////////// Rounding
inline __m128 SIMD::FloorLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_floor_ss( mDst, mSrc );
}
inline __m128d SIMD::FloorLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_floor_sd( mDst, mSrc );
}

inline __m128 SIMD::Floor( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_floor_ps( mValue );
}
inline __m256 SIMD::Floor( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_floor_ps( mValue );
}

inline __m128d SIMD::Floor( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_floor_pd( mValue );
}
inline __m256d SIMD::Floor( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_floor_pd( mValue );
}

inline __m128 SIMD::CeilLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_ceil_ss( mDst, mSrc );
}
inline __m128d SIMD::CeilLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_ceil_sd( mDst, mSrc );
}

inline __m128 SIMD::Ceil( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_ceil_ps( mValue );
}
inline __m256 SIMD::Ceil( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_ceil_ps( mValue );
}

inline __m128d SIMD::Ceil( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_ceil_pd( mValue );
}
inline __m256d SIMD::Ceil( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_ceil_pd( mValue );
}

inline __m128 SIMD::RoundLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_round_ss( mDst, mSrc, _MM_FROUND_NINT );
}
inline __m128d SIMD::RoundLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_round_sd( mDst, mSrc, _MM_FROUND_NINT );
}

inline __m128 SIMD::Round( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_round_ps( mValue, _MM_FROUND_NINT );
}
inline __m256 SIMD::Round( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_round_ps( mValue, _MM_FROUND_NINT );
}

inline __m128d SIMD::Round( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_round_pd( mValue, _MM_FROUND_NINT );
}
inline __m256d SIMD::Round( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_round_pd( mValue, _MM_FROUND_NINT );
}

////////////////////////////////////////////////////////////// Addition
inline __m128 SIMD::AddLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_add_ss( mDst, mSrc );
}
inline __m128d SIMD::AddLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_sd( mDst, mSrc );
}

inline __m128 SIMD::Add( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_add_ps( mDst, mSrc );
}
inline __m256 SIMD::Add( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_add_ps( mDst, mSrc );
}

inline __m128d SIMD::Add( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_pd( mDst, mSrc );
}
inline __m256d SIMD::Add( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_add_pd( mDst, mSrc );
}

inline __m128i SIMD::Add8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_epi8( mDst, mSrc );
}
inline __m256i SIMD::Add8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_add_epi8( mDst, mSrc );
}

inline __m128i SIMD::Add16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_epi16( mDst, mSrc );
}
inline __m256i SIMD::Add16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_add_epi16( mDst, mSrc );
}

inline __m128i SIMD::Add32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_epi32( mDst, mSrc );
}
inline __m256i SIMD::Add32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_add_epi32( mDst, mSrc );
}

inline __m128i SIMD::Add64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_epi64( mDst, mSrc );
}
inline __m256i SIMD::Add64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_add_epi64( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Addition with Saturation
inline __m128i SIMD::AddSigned8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_adds_epi8( mDst, mSrc );
}
inline __m256i SIMD::AddSigned8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_adds_epi8( mDst, mSrc );
}

inline __m128i SIMD::AddSigned16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_adds_epi16( mDst, mSrc );
}
inline __m256i SIMD::AddSigned16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_adds_epi16( mDst, mSrc );
}

inline __m128i SIMD::AddUnsigned8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_adds_epu8( mDst, mSrc );
}
inline __m256i SIMD::AddUnsigned8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_adds_epu8( mDst, mSrc );
}

inline __m128i SIMD::AddUnsigned16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_adds_epu16( mDst, mSrc );
}
inline __m256i SIMD::AddUnsigned16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_adds_epu16( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Horizontal Addition
inline __m128 SIMD::HAdd( __m128 mSrc1, __m128 mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_hadd_ps( mSrc1, mSrc2 );
}
inline __m256 SIMD::HAdd( __m256 mSrc1, __m256 mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hadd_ps( mSrc1, mSrc2 );
}

inline __m128d SIMD::HAdd( __m128d mSrc1, __m128d mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_hadd_pd( mSrc1, mSrc2 );
}
inline __m256d SIMD::HAdd( __m256d mSrc1, __m256d mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hadd_pd( mSrc1, mSrc2 );
}

inline __m128i SIMD::HAdd16( __m128i mSrc1, __m128i mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hadd_epi16( mSrc1, mSrc2 );
}
inline __m256i SIMD::HAdd16( __m256i mSrc1, __m256i mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hadd_epi16( mSrc1, mSrc2 );
}

inline __m128i SIMD::HAdd32( __m128i mSrc1, __m128i mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hadd_epi32( mSrc1, mSrc2 );
}
inline __m256i SIMD::HAdd32( __m256i mSrc1, __m256i mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hadd_epi32( mSrc1, mSrc2 );
}

////////////////////////////////////////////////////////////// Horizontal Addition with Saturation
inline __m128i SIMD::HAddSigned16( __m128i mSrc1, __m128i mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hadds_epi16( mSrc1, mSrc2 );
}
inline __m256i SIMD::HAddSigned16( __m256i mSrc1, __m256i mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hadds_epi16( mSrc1, mSrc2 );
}

////////////////////////////////////////////////////////////// Substraction
inline __m128 SIMD::SubLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sub_ss( mDst, mSrc );
}
inline __m128d SIMD::SubLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_sd( mDst, mSrc );
}

inline __m128 SIMD::Sub( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sub_ps( mDst, mSrc );
}
inline __m256 SIMD::Sub( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sub_ps( mDst, mSrc );
}

inline __m128d SIMD::Sub( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_pd( mDst, mSrc );
}
inline __m256d SIMD::Sub( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sub_pd( mDst, mSrc );
}

inline __m128i SIMD::Sub8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_epi8( mDst, mSrc );
}
inline __m256i SIMD::Sub8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sub_epi8( mDst, mSrc );
}

inline __m128i SIMD::Sub16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_epi16( mDst, mSrc );
}
inline __m256i SIMD::Sub16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sub_epi16( mDst, mSrc );
}

inline __m128i SIMD::Sub32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_epi32( mDst, mSrc );
}
inline __m256i SIMD::Sub32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sub_epi32( mDst, mSrc );
}

inline __m128i SIMD::Sub64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_epi64( mDst, mSrc );
}
inline __m256i SIMD::Sub64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sub_epi64( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Substraction with Saturation
inline __m128i SIMD::SubSigned8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_subs_epi8( mDst, mSrc );
}
inline __m256i SIMD::SubSigned8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_subs_epi8( mDst, mSrc );
}

inline __m128i SIMD::SubSigned16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_subs_epi16( mDst, mSrc );
}
inline __m256i SIMD::SubSigned16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_subs_epi16( mDst, mSrc );
}

inline __m128i SIMD::SubUnsigned8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_subs_epu8( mDst, mSrc );
}
inline __m256i SIMD::SubUnsigned8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_subs_epu8( mDst, mSrc );
}

inline __m128i SIMD::SubUnsigned16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_subs_epu16( mDst, mSrc );
}
inline __m256i SIMD::SubUnsigned16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_subs_epu16( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Horizontal Substraction
inline __m128 SIMD::HSub( __m128 mSrc1, __m128 mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_hsub_ps( mSrc1, mSrc2 );
}
inline __m256 SIMD::HSub( __m256 mSrc1, __m256 mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hsub_ps( mSrc1, mSrc2 );
}

inline __m128d SIMD::HSub( __m128d mSrc1, __m128d mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_hsub_pd( mSrc1, mSrc2 );
}
inline __m256d SIMD::HSub( __m256d mSrc1, __m256d mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hsub_pd( mSrc1, mSrc2 );
}

inline __m128i SIMD::HSub16( __m128i mSrc1, __m128i mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hsub_epi16( mSrc1, mSrc2 );
}
inline __m256i SIMD::HSub16( __m256i mSrc1, __m256i mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hsub_epi16( mSrc1, mSrc2 );
}

inline __m128i SIMD::HSub32( __m128i mSrc1, __m128i mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hsub_epi32( mSrc1, mSrc2 );
}
inline __m256i SIMD::HSub32( __m256i mSrc1, __m256i mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hsub_epi32( mSrc1, mSrc2 );
}

////////////////////////////////////////////////////////////// Horizontal Substraction with Saturation
inline __m128i SIMD::HSubSigned16( __m128i mSrc1, __m128i mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hsubs_epi16( mSrc1, mSrc2 );
}
inline __m256i SIMD::HSubSigned16( __m256i mSrc1, __m256i mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hsubs_epi16( mSrc1, mSrc2 );
}

////////////////////////////////////////////////////////////// Interleaved Add & Sub
inline __m128 SIMD::AddSub( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_addsub_ps( mDst, mSrc );
}
inline __m256 SIMD::AddSub( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_addsub_ps( mDst, mSrc );
}

inline __m128d SIMD::AddSub( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_addsub_pd( mDst, mSrc );
}
inline __m256d SIMD::AddSub( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_addsub_pd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// SAD (Sum Absolute Differences)
inline __m128i SIMD::SAD( __m128i mSrc1, __m128i mSrc2 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sad_epu8( mSrc1, mSrc2 );
}
inline __m256i SIMD::SAD( __m256i mSrc1, __m256i mSrc2 ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sad_epu8( mSrc1, mSrc2 );
}

//inline __m128i SIMD::SAD( __m128i mSrc1, __m128i mSrc2, Int iMask ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_mpsadbw_epu8( mSrc1, mSrc2, iMask );
//}
//inline __m256i SIMD::SAD( __m256i mSrc1, __m256i mSrc2, Int iMask ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_mpsadbw_epu8( mSrc1, mSrc2, iMask );
//}

////////////////////////////////////////////////////////////// Multiplication
inline __m128 SIMD::MulLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_mul_ss( mDst, mSrc );
}
inline __m128d SIMD::MulLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mul_sd( mDst, mSrc );
}

inline __m128 SIMD::Mul( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_mul_ps( mDst, mSrc );
}
inline __m256 SIMD::Mul( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_mul_ps( mDst, mSrc );
}

inline __m128d SIMD::Mul( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mul_pd( mDst, mSrc );
}
inline __m256d SIMD::Mul( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_mul_pd( mDst, mSrc );
}

inline __m128i SIMD::MulSigned16L( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mullo_epi16( mDst, mSrc );
}
inline __m256i SIMD::MulSigned16L( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mullo_epi16( mDst, mSrc );
}

inline __m128i SIMD::MulSigned16H( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mulhi_epi16( mDst, mSrc );
}
inline __m256i SIMD::MulSigned16H( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mulhi_epi16( mDst, mSrc );
}

inline __m128i SIMD::MulSigned32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_mul_epi32( mDst, mSrc );
}
inline __m256i SIMD::MulSigned32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mul_epi32( mDst, mSrc );
}

inline __m128i SIMD::MulSigned32L( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_mullo_epi32( mDst, mSrc );
}
inline __m256i SIMD::MulSigned32L( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mullo_epi32( mDst, mSrc );
}

inline __m128i SIMD::MulSigned64L( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_mullo_epi64( mDst, mSrc );
}
inline __m256i SIMD::MulSigned64L( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mullo_epi64( mDst, mSrc );
}

inline __m128i SIMD::MulUnsigned16H( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mulhi_epu16( mDst, mSrc );
}
inline __m256i SIMD::MulUnsigned16H( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mulhi_epu16( mDst, mSrc );
}

inline __m128i SIMD::MulUnsigned32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mul_epu32( mDst, mSrc );
}
inline __m256i SIMD::MulUnsigned32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mul_epu32( mDst, mSrc );
}

////////////////////////////////////////////////////////////// MADD (Multiply and Add)
inline __m128i SIMD::MAdd( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_madd_epi16( mDst, mSrc );
}
inline __m256i SIMD::MAdd( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_madd_epi16( mDst, mSrc );
}

inline __m128i SIMD::MAddUS( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_maddubs_epi16( mDst, mSrc );
}
inline __m256i SIMD::MAddUS( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_maddubs_epi16( mDst, mSrc );
}

////////////////////////////////////////////////////////////// DP (Dot Product)
//inline __m128 SIMD::DotP( __m128 mDst, __m128 mSrc, Int iMask4 ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_dp_ps( mDst, mSrc, iMask4 );
//}
//inline __m256 SIMD::DotP( __m256 mDst, __m256 mSrc, Int iMask4 ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_dp_ps( mDst, mSrc, iMask4 );
//}

//inline __m128d SIMD::DotP( __m128d mDst, __m128d mSrc, Int iMask2 ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_dp_pd( mDst, mSrc, iMask2 );
//}

inline __m128 SIMD::Dot2( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_dp_ps( mDst, mSrc, SIMD_DOTP_MASK_4(1,1,0,0,1,0,0,0) );
}
inline __m128 SIMD::Dot3( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_dp_ps( mDst, mSrc, SIMD_DOTP_MASK_4(1,1,1,0,1,0,0,0) );
}
inline __m128 SIMD::Dot4( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_dp_ps( mDst, mSrc, SIMD_DOTP_MASK_4(1,1,1,1,1,0,0,0) );
}

inline __m256 SIMD::Dot2( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_dp_ps( mDst, mSrc, SIMD_DOTP_MASK_4(1,1,0,0,1,0,0,0) );
}
inline __m256 SIMD::Dot3( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_dp_ps( mDst, mSrc, SIMD_DOTP_MASK_4(1,1,1,0,1,0,0,0) );
}
inline __m256 SIMD::Dot4( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_dp_ps( mDst, mSrc, SIMD_DOTP_MASK_4(1,1,1,1,1,0,0,0) );
}

inline __m128d SIMD::Dot2( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_dp_pd( mDst, mSrc, SIMD_DOTP_MASK_2(1,1,1,0) );
}

////////////////////////////////////////////////////////////// Division
inline __m128 SIMD::DivLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_div_ss( mDst, mSrc );
}
inline __m128d SIMD::DivLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_sd( mDst, mSrc );
}

inline __m128 SIMD::Div( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_div_ps( mDst, mSrc );
}
inline __m256 SIMD::Div( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_ps( mDst, mSrc );
}

inline __m128d SIMD::Div( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_pd( mDst, mSrc );
}
inline __m256d SIMD::Div( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_pd( mDst, mSrc );
}

inline __m128i SIMD::DivSigned8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epi8( mDst, mSrc );
}
inline __m256i SIMD::DivSigned8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epi8( mDst, mSrc );
}

inline __m128i SIMD::DivSigned16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epi16( mDst, mSrc );
}
inline __m256i SIMD::DivSigned16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epi16( mDst, mSrc );
}

inline __m128i SIMD::DivSigned32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epi32( mDst, mSrc );
}
inline __m256i SIMD::DivSigned32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epi32( mDst, mSrc );
}

inline __m128i SIMD::DivSigned64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epi64( mDst, mSrc );
}
inline __m256i SIMD::DivSigned64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epi64( mDst, mSrc );
}

inline __m128i SIMD::DivUnsigned8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epu8( mDst, mSrc );
}
inline __m256i SIMD::DivUnsigned8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epu8( mDst, mSrc );
}

inline __m128i SIMD::DivUnsigned16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epu16( mDst, mSrc );
}
inline __m256i SIMD::DivUnsigned16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epu16( mDst, mSrc );
}

inline __m128i SIMD::DivUnsigned32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epu32( mDst, mSrc );
}
inline __m256i SIMD::DivUnsigned32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epu32( mDst, mSrc );
}

inline __m128i SIMD::DivUnsigned64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_epu64( mDst, mSrc );
}
inline __m256i SIMD::DivUnsigned64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_epu64( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Modulo
inline __m128i SIMD::ModSigned8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epi8( mDst, mSrc );
}
inline __m256i SIMD::ModSigned8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epi8( mDst, mSrc );
}

inline __m128i SIMD::ModSigned16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epi16( mDst, mSrc );
}
inline __m256i SIMD::ModSigned16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epi16( mDst, mSrc );
}

inline __m128i SIMD::ModSigned32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epi32( mDst, mSrc );
}
inline __m256i SIMD::ModSigned32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epi32( mDst, mSrc );
}

inline __m128i SIMD::ModSigned64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epi64( mDst, mSrc );
}
inline __m256i SIMD::ModSigned64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epi64( mDst, mSrc );
}

inline __m128i SIMD::ModUnsigned8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epu8( mDst, mSrc );
}
inline __m256i SIMD::ModUnsigned8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epu8( mDst, mSrc );
}

inline __m128i SIMD::ModUnsigned16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epu16( mDst, mSrc );
}
inline __m256i SIMD::ModUnsigned16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epu16( mDst, mSrc );
}

inline __m128i SIMD::ModUnsigned32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epu32( mDst, mSrc );
}
inline __m256i SIMD::ModUnsigned32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epu32( mDst, mSrc );
}

inline __m128i SIMD::ModUnsigned64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_rem_epu64( mDst, mSrc );
}
inline __m256i SIMD::ModUnsigned64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rem_epu64( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Division and Modulo
inline __m128i SIMD::DivModSigned32( __m128i * outMod, __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_divrem_epi32( outMod, mDst, mSrc );
}
inline __m256i SIMD::DivModSigned32( __m256i * outMod, __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_divrem_epi32( outMod, mDst, mSrc );
}

inline __m128i SIMD::DivModUnsigned32( __m128i * outMod, __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_divrem_epu32( outMod, mDst, mSrc );
}
inline __m256i SIMD::DivModUnsigned32( __m256i * outMod, __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_divrem_epu32( outMod, mDst, mSrc );
}

////////////////////////////////////////////////////////////// Average (always unsigned)
inline __m128i SIMD::Avg8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_avg_epu8( mDst, mSrc );
}
inline __m256i SIMD::Avg8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_avg_epu8( mDst, mSrc );
}

inline __m128i SIMD::Avg16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_avg_epu16( mDst, mSrc );
}
inline __m256i SIMD::Avg16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_avg_epu16( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Equal)
inline __m128 SIMD::CmpEQLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpeq_ss( mDst, mSrc );
}
inline __m128d SIMD::CmpEQLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_sd( mDst, mSrc );
}

inline __m128 SIMD::CmpEQ( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpeq_ps( mDst, mSrc );
}
inline __m256 SIMD::CmpEQ( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, SIMD_CMP_EQ_OS );
}

inline __m128d SIMD::CmpEQ( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_pd( mDst, mSrc );
}
inline __m256d SIMD::CmpEQ( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, SIMD_CMP_EQ_OS );
}

inline __m128i SIMD::CmpEQ8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_epi8( mDst, mSrc );
}
inline __m256i SIMD::CmpEQ8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpeq_epi8( mDst, mSrc );
}

inline __m128i SIMD::CmpEQ16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_epi16( mDst, mSrc );
}
inline __m256i SIMD::CmpEQ16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpeq_epi16( mDst, mSrc );
}

inline __m128i SIMD::CmpEQ32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_epi32( mDst, mSrc );
}
inline __m256i SIMD::CmpEQ32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpeq_epi32( mDst, mSrc );
}

inline __m128i SIMD::CmpEQ64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cmpeq_epi64( mDst, mSrc );
}
inline __m256i SIMD::CmpEQ64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpeq_epi64( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Not Equal)
inline __m128 SIMD::CmpNEQLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpneq_ss( mDst, mSrc );
}
inline __m128d SIMD::CmpNEQLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpneq_sd( mDst, mSrc );
}

inline __m128 SIMD::CmpNEQ( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpneq_ps( mDst, mSrc );
}
inline __m256 SIMD::CmpNEQ( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, SIMD_CMP_NEQ_OS );
}

inline __m128d SIMD::CmpNEQ( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpneq_pd( mDst, mSrc );
}
inline __m256d SIMD::CmpNEQ( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, SIMD_CMP_NEQ_OS );
}

////////////////////////////////////////////////////////////// Compare (Lesser-Than)
inline __m128 SIMD::CmpLTLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmplt_ss( mDst, mSrc );
}
inline __m128d SIMD::CmpLTLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_sd( mDst, mSrc );
}

inline __m128 SIMD::CmpLT( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmplt_ps( mDst, mSrc );
}
inline __m256 SIMD::CmpLT( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, SIMD_CMP_LT_OS );
}

inline __m128d SIMD::CmpLT( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_pd( mDst, mSrc );
}
inline __m256d SIMD::CmpLT( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, SIMD_CMP_LT_OS );
}

inline __m128i SIMD::CmpLT8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_epi8( mDst, mSrc );
}

inline __m128i SIMD::CmpLT16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_epi16( mDst, mSrc );
}

inline __m128i SIMD::CmpLT32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_epi32( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Not Lesser-Than)
inline __m128 SIMD::CmpNLTLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnlt_ss( mDst, mSrc );
}
inline __m128d SIMD::CmpNLTLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnlt_sd( mDst, mSrc );
}

inline __m128 SIMD::CmpNLT( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnlt_ps( mDst, mSrc );
}
inline __m256 SIMD::CmpNLT( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, SIMD_CMP_NLT_US );
}

inline __m128d SIMD::CmpNLT( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnlt_pd( mDst, mSrc );
}
inline __m256d SIMD::CmpNLT( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, SIMD_CMP_NLT_US );
}

////////////////////////////////////////////////////////////// Compare (Lesser-or-Equal)
inline __m128 SIMD::CmpLELower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmple_ss( mDst, mSrc );
}
inline __m128d SIMD::CmpLELower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmple_sd( mDst, mSrc );
}

inline __m128 SIMD::CmpLE( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmple_ps( mDst, mSrc );
}
inline __m256 SIMD::CmpLE( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, SIMD_CMP_LE_OS );
}

inline __m128d SIMD::CmpLE( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmple_pd( mDst, mSrc );
}
inline __m256d SIMD::CmpLE( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, SIMD_CMP_LE_OS );
}

////////////////////////////////////////////////////////////// Compare (Not Lesser-or-Equal)
inline __m128 SIMD::CmpNLELower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnle_ss( mDst, mSrc );
}
inline __m128d SIMD::CmpNLELower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnle_sd( mDst, mSrc );
}

inline __m128 SIMD::CmpNLE( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnle_ps( mDst, mSrc );
}
inline __m256 SIMD::CmpNLE( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, SIMD_CMP_NLE_US );
}

inline __m128d SIMD::CmpNLE( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnle_pd( mDst, mSrc );
}
inline __m256d SIMD::CmpNLE( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, SIMD_CMP_NLE_US );
}

////////////////////////////////////////////////////////////// Compare (Greater-Than)
inline __m128 SIMD::CmpGTLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpgt_ss( mDst, mSrc );
}
inline __m128d SIMD::CmpGTLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_sd( mDst, mSrc );
}

inline __m128 SIMD::CmpGT( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpgt_ps( mDst, mSrc );
}
inline __m256 SIMD::CmpGT( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, SIMD_CMP_GT_OS );
}

inline __m128d SIMD::CmpGT( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_pd( mDst, mSrc );
}
inline __m256d SIMD::CmpGT( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, SIMD_CMP_GT_OS );
}

inline __m128i SIMD::CmpGT8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_epi8( mDst, mSrc );
}
inline __m256i SIMD::CmpGT8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpgt_epi8( mDst, mSrc );
}

inline __m128i SIMD::CmpGT16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_epi16( mDst, mSrc );
}
inline __m256i SIMD::CmpGT16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpgt_epi16( mDst, mSrc );
}

inline __m128i SIMD::CmpGT32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_epi32( mDst, mSrc );
}
inline __m256i SIMD::CmpGT32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpgt_epi32( mDst, mSrc );
}

inline __m128i SIMD::CmpGT64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cmpgt_epi64( mDst, mSrc );
}
inline __m256i SIMD::CmpGT64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpgt_epi64( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Not Greater-Than)
inline __m128 SIMD::CmpNGTLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpngt_ss( mDst, mSrc );
}
inline __m128d SIMD::CmpNGTLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpngt_sd( mDst, mSrc );
}

inline __m128 SIMD::CmpNGT( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpngt_ps( mDst, mSrc );
}
inline __m256 SIMD::CmpNGT( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, SIMD_CMP_NGT_US );
}

inline __m128d SIMD::CmpNGT( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpngt_pd( mDst, mSrc );
}
inline __m256d SIMD::CmpNGT( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, SIMD_CMP_NGT_US );
}

////////////////////////////////////////////////////////////// Compare (Greater-or-Equal)
inline __m128 SIMD::CmpGELower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpge_ss( mDst, mSrc );
}
inline __m128d SIMD::CmpGELower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpge_sd( mDst, mSrc );
}

inline __m128 SIMD::CmpGE( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpge_ps( mDst, mSrc );
}
inline __m256 SIMD::CmpGE( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, SIMD_CMP_GE_OS );
}

inline __m128d SIMD::CmpGE( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpge_pd( mDst, mSrc );
}
inline __m256d SIMD::CmpGE( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, SIMD_CMP_GE_OS );
}

////////////////////////////////////////////////////////////// Compare (Not Greater-or-Equal)
inline __m128 SIMD::CmpNGELower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnge_ss( mDst, mSrc );
}
inline __m128d SIMD::CmpNGELower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnge_sd( mDst, mSrc );
}

inline __m128 SIMD::CmpNGE( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnge_ps( mDst, mSrc );
}
inline __m256 SIMD::CmpNGE( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, SIMD_CMP_NGE_US );
}

inline __m128d SIMD::CmpNGE( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnge_pd( mDst, mSrc );
}
inline __m256d SIMD::CmpNGE( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, SIMD_CMP_NGE_US );
}

////////////////////////////////////////////////////////////// Compare (Ordered)
inline __m128 SIMD::CmpORDLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpord_ss( mDst, mSrc );
}
inline __m128d SIMD::CmpORDLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpord_sd( mDst, mSrc );
}

inline __m128 SIMD::CmpORD( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpord_ps( mDst, mSrc );
}
inline __m256 SIMD::CmpORD( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, SIMD_CMP_ORD_S );
}

inline __m128d SIMD::CmpORD( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpord_pd( mDst, mSrc );
}
inline __m256d SIMD::CmpORD( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, SIMD_CMP_ORD_S );
}

////////////////////////////////////////////////////////////// Compare (Unordered)
inline __m128 SIMD::CmpUNORDLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpunord_ss( mDst, mSrc );
}
inline __m128d SIMD::CmpUNORDLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpunord_sd( mDst, mSrc );
}

inline __m128 SIMD::CmpUNORD( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpunord_ps( mDst, mSrc );
}
inline __m256 SIMD::CmpUNORD( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, SIMD_CMP_UNORD_S );
}

inline __m128d SIMD::CmpUNORD( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpunord_pd( mDst, mSrc );
}
inline __m256d SIMD::CmpUNORD( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, SIMD_CMP_UNORD_S );
}

////////////////////////////////////////////////////////////// Compare (Bool results, always on lower element, _Q = non signaling versions)
inline Int SIMD::IsEQ( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comieq_ss( mDst, mSrc );
}
inline Int SIMD::IsEQ_Q( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomieq_ss( mDst, mSrc );
}
inline Int SIMD::IsEQ( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comieq_sd( mDst, mSrc );
}
inline Int SIMD::IsEQ_Q( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomieq_sd( mDst, mSrc );
}

inline Int SIMD::IsNEQ( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comineq_ss( mDst, mSrc );
}
inline Int SIMD::IsNEQ_Q( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomineq_ss( mDst, mSrc );
}
inline Int SIMD::IsNEQ( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comineq_sd( mDst, mSrc );
}
inline Int SIMD::IsNEQ_Q( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomineq_sd( mDst, mSrc );
}

inline Int SIMD::IsLT( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comilt_ss( mDst, mSrc );
}
inline Int SIMD::IsLT_Q( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomilt_ss( mDst, mSrc );
}
inline Int SIMD::IsLT( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comilt_sd( mDst, mSrc );
}
inline Int SIMD::IsLT_Q( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomilt_sd( mDst, mSrc );
}

inline Int SIMD::IsLE( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comile_ss( mDst, mSrc );
}
inline Int SIMD::IsLE_Q( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomile_ss( mDst, mSrc );
}
inline Int SIMD::IsLE( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comile_sd( mDst, mSrc );
}
inline Int SIMD::IsLE_Q( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomile_sd( mDst, mSrc );
}

inline Int SIMD::IsGT( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comigt_ss( mDst, mSrc );
}
inline Int SIMD::IsGT_Q( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomigt_ss( mDst, mSrc );
}
inline Int SIMD::IsGT( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comigt_sd( mDst, mSrc );
}
inline Int SIMD::IsGT_Q( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomigt_sd( mDst, mSrc );
}

inline Int SIMD::IsGE( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comige_ss( mDst, mSrc );
}
inline Int SIMD::IsGE_Q( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomige_ss( mDst, mSrc );
}
inline Int SIMD::IsGE( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comige_sd( mDst, mSrc );
}
inline Int SIMD::IsGE_Q( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomige_sd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Minimum Value
inline __m128 SIMD::MinLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_min_ss( mDst, mSrc );
}
inline __m128d SIMD::MinLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_min_sd( mDst, mSrc );
}

inline __m128 SIMD::Min( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_min_ps( mDst, mSrc );
}
inline __m256 SIMD::Min( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_min_ps( mDst, mSrc );
}

inline __m128d SIMD::Min( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_min_pd( mDst, mSrc );
}
inline __m256d SIMD::Min( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_min_pd( mDst, mSrc );
}

inline __m128i SIMD::MinSigned8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epi8( mDst, mSrc );
}
inline __m256i SIMD::MinSigned8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epi8( mDst, mSrc );
}

inline __m128i SIMD::MinSigned16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_min_epi16( mDst, mSrc );
}
inline __m256i SIMD::MinSigned16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epi16( mDst, mSrc );
}

inline __m128i SIMD::MinSigned32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epi32( mDst, mSrc );
}
inline __m256i SIMD::MinSigned32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epi32( mDst, mSrc );
}

inline __m128i SIMD::MinSigned64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epi64( mDst, mSrc );
}
inline __m256i SIMD::MinSigned64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epi64( mDst, mSrc );
}

inline __m128i SIMD::MinUnsigned8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_min_epu8( mDst, mSrc );
}
inline __m256i SIMD::MinUnsigned8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epu8( mDst, mSrc );
}

inline __m128i SIMD::MinUnsigned16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epu16( mDst, mSrc );
}
inline __m256i SIMD::MinUnsigned16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epu16( mDst, mSrc );
}

inline __m128i SIMD::MinUnsigned32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epu32( mDst, mSrc );
}
inline __m256i SIMD::MinUnsigned32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epu32( mDst, mSrc );
}

inline __m128i SIMD::MinUnsigned64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epu64( mDst, mSrc );
}
inline __m256i SIMD::MinUnsigned64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epu64( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Maximum Value
inline __m128 SIMD::MaxLower( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_max_ss( mDst, mSrc );
}
inline __m128d SIMD::MaxLower( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_max_sd( mDst, mSrc );
}

inline __m128 SIMD::Max( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_max_ps( mDst, mSrc );
}
inline __m256 SIMD::Max( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_max_ps( mDst, mSrc );
}

inline __m128d SIMD::Max( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_max_pd( mDst, mSrc );
}
inline __m256d SIMD::Max( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_max_pd( mDst, mSrc );
}

inline __m128i SIMD::MaxSigned8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epi8( mDst, mSrc );
}
inline __m256i SIMD::MaxSigned8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epi8( mDst, mSrc );
}

inline __m128i SIMD::MaxSigned16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_max_epi16( mDst, mSrc );
}
inline __m256i SIMD::MaxSigned16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epi16( mDst, mSrc );
}

inline __m128i SIMD::MaxSigned32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epi32( mDst, mSrc );
}
inline __m256i SIMD::MaxSigned32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epi32( mDst, mSrc );
}

inline __m128i SIMD::MaxSigned64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epi64( mDst, mSrc );
}
inline __m256i SIMD::MaxSigned64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epi64( mDst, mSrc );
}

inline __m128i SIMD::MaxUnsigned8( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_max_epu8( mDst, mSrc );
}
inline __m256i SIMD::MaxUnsigned8( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epu8( mDst, mSrc );
}

inline __m128i SIMD::MaxUnsigned16( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epu16( mDst, mSrc );
}
inline __m256i SIMD::MaxUnsigned16( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epu16( mDst, mSrc );
}

inline __m128i SIMD::MaxUnsigned32( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epu32( mDst, mSrc );
}
inline __m256i SIMD::MaxUnsigned32( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epu32( mDst, mSrc );
}

inline __m128i SIMD::MaxUnsigned64( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epu64( mDst, mSrc );
}
inline __m256i SIMD::MaxUnsigned64( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epu64( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Bitwise : And
inline __m128 SIMD::And( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_and_ps( mDst, mSrc );
}
inline __m256 SIMD::And( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_and_ps( mDst, mSrc );
}

inline __m128d SIMD::And( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_and_pd( mDst, mSrc );
}
inline __m256d SIMD::And( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_and_pd( mDst, mSrc );
}

inline __m128i SIMD::And( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_and_si128( mDst, mSrc );
}
inline __m256i SIMD::And( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_and_si256( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Bitwise : AndNot
inline __m128 SIMD::AndNot( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_andnot_ps( mDst, mSrc );
}
inline __m256 SIMD::AndNot( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_andnot_ps( mDst, mSrc );
}

inline __m128d SIMD::AndNot( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_andnot_pd( mDst, mSrc );
}
inline __m256d SIMD::AndNot( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_andnot_pd( mDst, mSrc );
}

inline __m128i SIMD::AndNot( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_andnot_si128( mDst, mSrc );
}
inline __m256i SIMD::AndNot( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_andnot_si256( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Bitwise : Or
inline __m128 SIMD::Or( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_or_ps( mDst, mSrc );
}
inline __m256 SIMD::Or( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_or_ps( mDst, mSrc );
}

inline __m128d SIMD::Or( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_or_pd( mDst, mSrc );
}
inline __m256d SIMD::Or( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_or_pd( mDst, mSrc );
}

inline __m128i SIMD::Or( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_or_si128( mDst, mSrc );
}
inline __m256i SIMD::Or( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_or_si256( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Bitwise : Xor
inline __m128 SIMD::Xor( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_xor_ps( mDst, mSrc );
}
inline __m256 SIMD::Xor( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_xor_ps( mDst, mSrc );
}

inline __m128d SIMD::Xor( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_xor_pd( mDst, mSrc );
}
inline __m256d SIMD::Xor( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_xor_pd( mDst, mSrc );
}

inline __m128i SIMD::Xor( __m128i mDst, __m128i mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_xor_si128( mDst, mSrc );
}
inline __m256i SIMD::Xor( __m256i mDst, __m256i mSrc ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_xor_si256( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Bitwise : Shift Left, Zero Extend
inline __m128i SIMD::Shift16L( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_slli_epi16( mDst, iCount );
}
inline __m128i SIMD::Shift16L( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sll_epi16( mDst, mCount );
}
inline __m256i SIMD::Shift16L( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_slli_epi16( mDst, iCount );
}
inline __m256i SIMD::Shift16L( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sll_epi16( mDst, mCount );
}

inline __m128i SIMD::Shift32L( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_slli_epi32( mDst, iCount );
}
inline __m128i SIMD::Shift32L( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sll_epi32( mDst, mCount );
}
inline __m256i SIMD::Shift32L( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_slli_epi32( mDst, iCount );
}
inline __m256i SIMD::Shift32L( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sll_epi32( mDst, mCount );
}

inline __m128i SIMD::Shift64L( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_slli_epi64( mDst, iCount );
}
inline __m128i SIMD::Shift64L( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sll_epi64( mDst, mCount );
}
inline __m256i SIMD::Shift64L( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_slli_epi64( mDst, iCount );
}
inline __m256i SIMD::Shift64L( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sll_epi64( mDst, mCount );
}

//inline __m128i SIMD::Shift128L( __m128i mDst, Int iCount ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_slli_si128( mDst, iCount );
//}
//inline __m256i SIMD::Shift256L( __m256i mDst, Int iCount ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_slli_si256( mDst, iCount );
//}

inline __m128i SIMD::ShiftV32L( __m128i mDst, __m128i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_sllv_epi32( mDst, mCounts );
}
inline __m256i SIMD::ShiftV32L( __m256i mDst, __m256i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sllv_epi32( mDst, mCounts );
}

inline __m128i SIMD::ShiftV64L( __m128i mDst, __m128i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_sllv_epi64( mDst, mCounts );
}
inline __m256i SIMD::ShiftV64L( __m256i mDst, __m256i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sllv_epi64( mDst, mCounts );
}

////////////////////////////////////////////////////////////// Bitwise : Shift Right, Zero Extend
inline __m128i SIMD::Shift16R( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srli_epi16( mDst, iCount );
}
inline __m128i SIMD::Shift16R( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srl_epi16( mDst, mCount );
}
inline __m256i SIMD::Shift16R( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srli_epi16( mDst, iCount );
}
inline __m256i SIMD::Shift16R( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srl_epi16( mDst, mCount );
}

inline __m128i SIMD::Shift32R( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srli_epi32( mDst, iCount );
}
inline __m128i SIMD::Shift32R( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srl_epi32( mDst, mCount );
}
inline __m256i SIMD::Shift32R( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srli_epi32( mDst, iCount );
}
inline __m256i SIMD::Shift32R( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srl_epi32( mDst, mCount );
}

inline __m128i SIMD::Shift64R( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srli_epi64( mDst, iCount );
}
inline __m128i SIMD::Shift64R( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srl_epi64( mDst, mCount );
}
inline __m256i SIMD::Shift64R( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srli_epi64( mDst, iCount );
}
inline __m256i SIMD::Shift64R( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srl_epi64( mDst, mCount );
}

//inline __m128i SIMD::Shift128R( __m128i mDst, Int iCount ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_srli_si128( mDst, iCount );
//}
//inline __m256i SIMD::Shift256R( __m256i mDst, Int iCount ) {
//    DebugAssert( CPUIDFn->HasAVX2() );
//    return _mm256_srli_si256( mDst, iCount );
//}

inline __m128i SIMD::ShiftV32R( __m128i mDst, __m128i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_srlv_epi32( mDst, mCounts );
}
inline __m256i SIMD::ShiftV32R( __m256i mDst, __m256i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srlv_epi32( mDst, mCounts );
}

inline __m128i SIMD::ShiftV64R( __m128i mDst, __m128i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_srlv_epi64( mDst, mCounts );
}
inline __m256i SIMD::ShiftV64R( __m256i mDst, __m256i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srlv_epi64( mDst, mCounts );
}

////////////////////////////////////////////////////////////// Bitwise : Shift Right, Sign Extend
inline __m128i SIMD::Shift16RSE( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srai_epi16( mDst, iCount );
}
inline __m128i SIMD::Shift16RSE( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sra_epi16( mDst, mCount );
}
inline __m256i SIMD::Shift16RSE( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srai_epi16( mDst, iCount );
}
inline __m256i SIMD::Shift16RSE( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sra_epi16( mDst, mCount );
}

inline __m128i SIMD::Shift32RSE( __m128i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srai_epi32( mDst, iCount );
}
inline __m128i SIMD::Shift32RSE( __m128i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sra_epi32( mDst, mCount );
}
inline __m256i SIMD::Shift32RSE( __m256i mDst, Int iCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srai_epi32( mDst, iCount );
}
inline __m256i SIMD::Shift32RSE( __m256i mDst, __m128i mCount ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sra_epi32( mDst, mCount );
}

inline __m128i SIMD::ShiftV32RSE( __m128i mDst, __m128i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_srav_epi32( mDst, mCounts );
}
inline __m256i SIMD::ShiftV32RSE( __m256i mDst, __m256i mCounts ) {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srav_epi32( mDst, mCounts );
}

////////////////////////////////////////////////////////////// CRC32
inline UInt32 SIMD::CRC32( UInt32 iCRC, UInt8 iValue ) {
    DebugAssert( CPUIDFn->HasSSE42() );
    return _mm_crc32_u8( iCRC, iValue );
}
inline UInt32 SIMD::CRC32( UInt32 iCRC, UInt16 iValue ) {
    DebugAssert( CPUIDFn->HasSSE42() );
    return _mm_crc32_u16( iCRC, iValue );
}
inline UInt32 SIMD::CRC32( UInt32 iCRC, UInt32 iValue ) {
    DebugAssert( CPUIDFn->HasSSE42() );
    return _mm_crc32_u32( iCRC, iValue );
}
inline UInt64 SIMD::CRC32( UInt64 iCRC, UInt64 iValue ) {
    DebugAssert( CPUIDFn->HasSSE42() );
    return _mm_crc32_u64( iCRC, iValue );
}

////////////////////////////////////////////////////////////// Invert
inline __m128 SIMD::InvertLower( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rcp_ss( mValue );
}

inline __m128 SIMD::Invert( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rcp_ps( mValue );
}
inline __m256 SIMD::Invert( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rcp_ps( mValue );
}

////////////////////////////////////////////////////////////// SquareRoot
inline __m128 SIMD::SqrtLower( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sqrt_ss( mValue );
}
inline __m128d SIMD::SqrtLower( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sqrt_sd( mValue, mValue );
}

inline __m128 SIMD::Sqrt( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sqrt_ps( mValue );
}
inline __m256 SIMD::Sqrt( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sqrt_ps( mValue );
}

inline __m128d SIMD::Sqrt( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sqrt_pd( mValue );
}
inline __m256d SIMD::Sqrt( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sqrt_pd( mValue );
}

////////////////////////////////////////////////////////////// Inverted SquareRoot
inline __m128 SIMD::InvSqrtLower( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rsqrt_ss( mValue );
}

inline __m128 SIMD::InvSqrt( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rsqrt_ps( mValue );
}
inline __m256 SIMD::InvSqrt( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rsqrt_ps( mValue );
}

inline __m128d SIMD::InvSqrt( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_invsqrt_pd( mValue );
}
inline __m256d SIMD::InvSqrt( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_invsqrt_pd( mValue );
}

////////////////////////////////////////////////////////////// Cube Root
inline __m128 SIMD::Cbrt( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cbrt_ps( mValue );
}
inline __m256 SIMD::Cbrt( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cbrt_ps( mValue );
}

inline __m128d SIMD::Cbrt( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cbrt_pd( mValue );
}
inline __m256d SIMD::Cbrt( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cbrt_pd( mValue );
}

////////////////////////////////////////////////////////////// Inverted Cube Root
inline __m128 SIMD::InvCbrt( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_invcbrt_ps( mValue );
}
inline __m256 SIMD::InvCbrt( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_invcbrt_ps( mValue );
}

inline __m128d SIMD::InvCbrt( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_invcbrt_pd( mValue );
}
inline __m256d SIMD::InvCbrt( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_invcbrt_pd( mValue );
}

////////////////////////////////////////////////////////////// Hypothenus (Square Root of summed products)
inline __m128 SIMD::Hypot( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_hypot_ps( mDst, mSrc );
}
inline __m256 SIMD::Hypot( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hypot_ps( mDst, mSrc );
}

inline __m128d SIMD::Hypot( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_hypot_pd( mDst, mSrc );
}
inline __m256d SIMD::Hypot( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hypot_pd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Natural Logarithm
inline __m128 SIMD::Ln( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_log_ps( mValue );
}
inline __m256 SIMD::Ln( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log_ps( mValue );
}

inline __m128d SIMD::Ln( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_log_pd( mValue );
}
inline __m256d SIMD::Ln( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log_pd( mValue );
}

////////////////////////////////////////////////////////////// Natural Logarithm of (1 + x)
inline __m128 SIMD::Ln1P( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_log1p_ps( mValue );
}
inline __m256 SIMD::Ln1P( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log1p_ps( mValue );
}

inline __m128d SIMD::Ln1P( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_log1p_pd( mValue );
}
inline __m256d SIMD::Ln1P( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log1p_pd( mValue );
}

////////////////////////////////////////////////////////////// Logarithm Base 2
inline __m128 SIMD::Log2( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_log2_ps( mValue );
}
inline __m256 SIMD::Log2( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log2_ps( mValue );
}

inline __m128d SIMD::Log2( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_log2_pd( mValue );
}
inline __m256d SIMD::Log2( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log2_pd( mValue );
}

////////////////////////////////////////////////////////////// Logarithm Base 10
inline __m128 SIMD::Log10( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_log10_ps( mValue );
}
inline __m256 SIMD::Log10( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log10_ps( mValue );
}

inline __m128d SIMD::Log10( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_log10_pd( mValue );
}
inline __m256d SIMD::Log10( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log10_pd( mValue );
}

////////////////////////////////////////////////////////////// Natural Exponential
inline __m128 SIMD::Exp( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_exp_ps( mValue );
}
inline __m256 SIMD::Exp( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_exp_ps( mValue );
}

inline __m128d SIMD::Exp( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_exp_pd( mValue );
}
inline __m256d SIMD::Exp( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_exp_pd( mValue );
}

////////////////////////////////////////////////////////////// Natural Exponential - 1
inline __m128 SIMD::ExpM1( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_expm1_ps( mValue );
}
inline __m256 SIMD::ExpM1( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_expm1_ps( mValue );
}

inline __m128d SIMD::ExpM1( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_expm1_pd( mValue );
}
inline __m256d SIMD::ExpM1( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_expm1_pd( mValue );
}

////////////////////////////////////////////////////////////// Exponential Base 2
inline __m128 SIMD::Exp2( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_exp2_ps( mValue );
}
inline __m256 SIMD::Exp2( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_exp2_ps( mValue );
}

inline __m128d SIMD::Exp2( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_exp2_pd( mValue );
}
inline __m256d SIMD::Exp2( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_exp2_pd( mValue );
}

////////////////////////////////////////////////////////////// Exponential Base 10
inline __m128 SIMD::Exp10( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_exp10_ps( mValue );
}
inline __m256 SIMD::Exp10( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_exp10_ps( mValue );
}

inline __m128d SIMD::Exp10( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_exp10_pd( mValue );
}
inline __m256d SIMD::Exp10( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_exp10_pd( mValue );
}

////////////////////////////////////////////////////////////// Power
inline __m128 SIMD::Pow( __m128 mBase, __m128 mExponent ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_pow_ps( mBase, mExponent );
}
inline __m256 SIMD::Pow( __m256 mBase, __m256 mExponent ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_pow_ps( mBase, mExponent );
}

inline __m128d SIMD::Pow( __m128d mBase, __m128d mExponent ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_pow_pd( mBase, mExponent );
}
inline __m256d SIMD::Pow( __m256d mBase, __m256d mExponent ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_pow_pd( mBase, mExponent );
}

////////////////////////////////////////////////////////////// Sine
inline __m128 SIMD::Sin( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sin_ps( mValue );
}
inline __m256 SIMD::Sin( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sin_ps( mValue );
}

inline __m128d SIMD::Sin( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sin_pd( mValue );
}
inline __m256d SIMD::Sin( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sin_pd( mValue );
}

////////////////////////////////////////////////////////////// Cosine
inline __m128 SIMD::Cos( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cos_ps( mValue );
}
inline __m256 SIMD::Cos( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cos_ps( mValue );
}

inline __m128d SIMD::Cos( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cos_pd( mValue );
}
inline __m256d SIMD::Cos( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cos_pd( mValue );
}

////////////////////////////////////////////////////////////// Sine and Cosine
inline __m128 SIMD::SinCos( __m128 * outCos, __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sincos_ps( outCos, mValue );
}
inline __m256 SIMD::SinCos( __m256 * outCos, __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sincos_ps( outCos, mValue );
}

inline __m128d SIMD::SinCos( __m128d * outCos, __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sincos_pd( outCos, mValue );
}
inline __m256d SIMD::SinCos( __m256d * outCos, __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sincos_pd( outCos, mValue );
}

////////////////////////////////////////////////////////////// Tangent
inline __m128 SIMD::Tan( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_tan_ps( mValue );
}
inline __m256 SIMD::Tan( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_tan_ps( mValue );
}

inline __m128d SIMD::Tan( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_tan_pd( mValue );
}
inline __m256d SIMD::Tan( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_tan_pd( mValue );
}

////////////////////////////////////////////////////////////// ArcSine
inline __m128 SIMD::ArcSin( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_asin_ps( mValue );
}
inline __m256 SIMD::ArcSin( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_asin_ps( mValue );
}

inline __m128d SIMD::ArcSin( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_asin_pd( mValue );
}
inline __m256d SIMD::ArcSin( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_asin_pd( mValue );
}

////////////////////////////////////////////////////////////// ArcCosine
inline __m128 SIMD::ArcCos( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_acos_ps( mValue );
}
inline __m256 SIMD::ArcCos( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_acos_ps( mValue );
}

inline __m128d SIMD::ArcCos( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_acos_pd( mValue );
}
inline __m256d SIMD::ArcCos( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_acos_pd( mValue );
}

////////////////////////////////////////////////////////////// ArcTangent
inline __m128 SIMD::ArcTan( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_atan_ps( mValue );
}
inline __m256 SIMD::ArcTan( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_atan_ps( mValue );
}

inline __m128d SIMD::ArcTan( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_atan_pd( mValue );
}
inline __m256d SIMD::ArcTan( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_atan_pd( mValue );
}

////////////////////////////////////////////////////////////// ArcTangent2
inline __m128 SIMD::ArcTan2( __m128 mNum, __m128 mDenom ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_atan2_ps( mNum, mDenom );
}
inline __m256 SIMD::ArcTan2( __m256 mNum, __m256 mDenom ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_atan2_ps( mNum, mDenom );
}

inline __m128d SIMD::ArcTan2( __m128d mNum, __m128d mDenom ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_atan2_pd( mNum, mDenom );
}
inline __m256d SIMD::ArcTan2( __m256d mNum, __m256d mDenom ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_atan2_pd( mNum, mDenom );
}

////////////////////////////////////////////////////////////// Hyperbolic Sine
inline __m128 SIMD::SinH( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sinh_ps( mValue );
}
inline __m256 SIMD::SinH( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sinh_ps( mValue );
}

inline __m128d SIMD::SinH( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sinh_pd( mValue );
}
inline __m256d SIMD::SinH( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sinh_pd( mValue );
}

////////////////////////////////////////////////////////////// Hyperbolic Cosine
inline __m128 SIMD::CosH( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cosh_ps( mValue );
}
inline __m256 SIMD::CosH( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cosh_ps( mValue );
}

inline __m128d SIMD::CosH( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cosh_pd( mValue );
}
inline __m256d SIMD::CosH( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cosh_pd( mValue );
}

////////////////////////////////////////////////////////////// Hyperbolic Tangent
inline __m128 SIMD::TanH( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_tanh_ps( mValue );
}
inline __m256 SIMD::TanH( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_tanh_ps( mValue );
}

inline __m128d SIMD::TanH( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_tanh_pd( mValue );
}
inline __m256d SIMD::TanH( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_tanh_pd( mValue );
}

////////////////////////////////////////////////////////////// Hyperbolic ArcSine
inline __m128 SIMD::ArgSinH( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_asinh_ps( mValue );
}
inline __m256 SIMD::ArgSinH( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_asinh_ps( mValue );
}

inline __m128d SIMD::ArgSinH( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_asinh_pd( mValue );
}
inline __m256d SIMD::ArgSinH( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_asinh_pd( mValue );
}

////////////////////////////////////////////////////////////// Hyperbolic ArcCosine
inline __m128 SIMD::ArgCosH( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_acosh_ps( mValue );
}
inline __m256 SIMD::ArgCosH( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_acosh_ps( mValue );
}

inline __m128d SIMD::ArgCosH( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_acosh_pd( mValue );
}
inline __m256d SIMD::ArgCosH( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_acosh_pd( mValue );
}

////////////////////////////////////////////////////////////// Hyperbolic ArcTangent
inline __m128 SIMD::ArgTanH( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_atanh_ps( mValue );
}
inline __m256 SIMD::ArgTanH( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_atanh_ps( mValue );
}

inline __m128d SIMD::ArgTanH( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_atanh_pd( mValue );
}
inline __m256d SIMD::ArgTanH( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_atanh_pd( mValue );
}

////////////////////////////////////////////////////////////// Gauss Error Function
inline __m128 SIMD::Erf( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_erf_ps( mValue );
}
inline __m256 SIMD::Erf( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erf_ps( mValue );
}

inline __m128d SIMD::Erf( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_erf_pd( mValue );
}
inline __m256d SIMD::Erf( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erf_pd( mValue );
}

////////////////////////////////////////////////////////////// Inverted Gauss Error Function
inline __m128 SIMD::InvErf( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_erfinv_ps( mValue );
}
inline __m256 SIMD::InvErf( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erfinv_ps( mValue );
}

inline __m128d SIMD::InvErf( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_erfinv_pd( mValue );
}
inline __m256d SIMD::InvErf( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erfinv_pd( mValue );
}

////////////////////////////////////////////////////////////// Complementary Gauss Error Function
inline __m128 SIMD::ErfC( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_erfc_ps( mValue );
}
inline __m256 SIMD::ErfC( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erfc_ps( mValue );
}

inline __m128d SIMD::ErfC( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_erfc_pd( mValue );
}
inline __m256d SIMD::ErfC( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erfc_pd( mValue );
}

////////////////////////////////////////////////////////////// Inverted Complementary Gauss Error Function
inline __m128 SIMD::InvErfC( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_erfcinv_ps( mValue );
}
inline __m256 SIMD::InvErfC( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erfcinv_ps( mValue );
}

inline __m128d SIMD::InvErfC( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_erfcinv_pd( mValue );
}
inline __m256d SIMD::InvErfC( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erfcinv_pd( mValue );
}

////////////////////////////////////////////////////////////// Normal Cumulative Distribution Function
inline __m128 SIMD::CDFNorm( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cdfnorm_ps( mValue );
}
inline __m256 SIMD::CDFNorm( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cdfnorm_ps( mValue );
}

inline __m128d SIMD::CDFNorm( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cdfnorm_pd( mValue );
}
inline __m256d SIMD::CDFNorm( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cdfnorm_pd( mValue );
}

////////////////////////////////////////////////////////////// Inverted Normal Cumulative Distribution Function
inline __m128 SIMD::InvCDFNorm( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cdfnorminv_ps( mValue );
}
inline __m256 SIMD::InvCDFNorm( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cdfnorminv_ps( mValue );
}

inline __m128d SIMD::InvCDFNorm( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cdfnorminv_pd( mValue );
}
inline __m256d SIMD::InvCDFNorm( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cdfnorminv_pd( mValue );
}

////////////////////////////////////////////////////////////// Complex Square Root
inline __m128 SIMD::CSqrt( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_csqrt_ps( mValue );
}
inline __m256 SIMD::CSqrt( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_csqrt_ps( mValue );
}

////////////////////////////////////////////////////////////// Complex Logarithm
inline __m128 SIMD::CLog( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_clog_ps( mValue );
}
inline __m256 SIMD::CLog( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_clog_ps( mValue );
}

////////////////////////////////////////////////////////////// Complex Exponential
inline __m128 SIMD::CExp( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cexp_ps( mValue );
}
inline __m256 SIMD::CExp( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cexp_ps( mValue );
}


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
