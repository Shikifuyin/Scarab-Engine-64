/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SSEWrappers.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SSE low level abstraction layer
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// SIMDMath implementation
inline SIMDMath * SIMDMath::GetInstance() {
    static SIMDMath s_Instance;
    return &s_Instance;
}

////////////////////////////////////////////////////////////// Serializing instruction (makes sure everything is flushed)
inline Void SIMDMath::SerializeMemoryStore() const {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_sfence();
}
inline Void SIMDMath::SerializeMemoryLoad() const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_lfence();
}
inline Void SIMDMath::SerializeMemory() const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_mfence();
}

////////////////////////////////////////////////////////////// Register Initialization
inline __m128 SIMDMath::Zero128F() const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_setzero_ps();
}
inline __m256 SIMDMath::Zero256F() const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_setzero_ps();
}

inline __m128d SIMDMath::Zero128D() const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_setzero_pd();
}
inline __m256d SIMDMath::Zero256D() const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_setzero_pd();
}

inline __m128i SIMDMath::Zero128I() const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_setzero_si128();
}
inline __m256i SIMDMath::Zero256I() const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_setzero_si256();
}

////////////////////////////////////////////////////////////// Values -> Registers
inline __m128 SIMDMath::SetLower( Float f0 ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_set_ss( f0 );
}
inline __m128d SIMDMath::SetLower( Double f0 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_sd( f0 );
}

inline __m128 SIMDMath::Set128( Float f0, Float f1, Float f2, Float f3 ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_set_ps( f0, f1, f2, f3 );
}
inline __m256 SIMDMath::Set256( Float f0, Float f1, Float f2, Float f3, Float f4, Float f5, Float f6, Float f7 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_ps( f7, f6, f5, f4, f3, f2, f1, f0 );
}

inline __m128d SIMDMath::Set128( Double f0, Double f1 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_pd( f0, f1 );
}
inline __m256d SIMDMath::Set256( Double f0, Double f1, Double f2, Double f3 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_pd( f3, f2, f1, f0 );
}

inline __m128i SIMDMath::Set128( Int8 i0, Int8 i1, Int8 i2, Int8 i3, Int8 i4, Int8 i5, Int8 i6, Int8 i7,
                                 Int8 i8, Int8 i9, Int8 i10, Int8 i11, Int8 i12, Int8 i13, Int8 i14, Int8 i15 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_epi8( i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0 );
}
inline __m256i SIMDMath::Set256( Int8 i0, Int8 i1, Int8 i2, Int8 i3, Int8 i4, Int8 i5, Int8 i6, Int8 i7,
                                 Int8 i8, Int8 i9, Int8 i10, Int8 i11, Int8 i12, Int8 i13, Int8 i14, Int8 i15,
                                 Int8 i16, Int8 i17, Int8 i18, Int8 i19, Int8 i20, Int8 i21, Int8 i22, Int8 i23,
                                 Int8 i24, Int8 i25, Int8 i26, Int8 i27, Int8 i28, Int8 i29, Int8 i30, Int8 i31 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_epi8( i31, i30, i29, i28, i27, i26, i25, i24, i23, i22, i21, i20, i19, i18, i17, i16,
                            i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0 );
}

inline __m128i SIMDMath::Set128( Int16 i0, Int16 i1, Int16 i2, Int16 i3, Int16 i4, Int16 i5, Int16 i6, Int16 i7 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_epi16( i7, i6, i5, i4, i3, i2, i1, i0 );
}
inline __m256i SIMDMath::Set256( Int16 i0, Int16 i1, Int16 i2, Int16 i3, Int16 i4, Int16 i5, Int16 i6, Int16 i7,
                                 Int16 i8, Int16 i9, Int16 i10, Int16 i11, Int16 i12, Int16 i13, Int16 i14, Int16 i15 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_epi16( i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0 );
}

inline __m128i SIMDMath::Set128( Int32 i0, Int32 i1, Int32 i2, Int32 i3 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_epi32( i3, i2, i1, i0 );
}
inline __m256i SIMDMath::Set256( Int32 i0, Int32 i1, Int32 i2, Int32 i3, Int32 i4, Int32 i5, Int32 i6, Int32 i7 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_epi32( i7, i6, i5, i4, i3, i2, i1, i0 );
}

inline __m128i SIMDMath::Set128( Int64 i0, Int64 i1 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_epi64x( i1, i0 );
}
inline __m256i SIMDMath::Set256( Int64 i0, Int64 i1, Int64 i2, Int64 i3 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_epi64x( i3, i2, i1, i0 );
}

inline __m128 SIMDMath::Set128( Float f ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_set_ps1( f );
}
inline __m256 SIMDMath::Set256( Float f ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_ps( f );
}

inline __m128d SIMDMath::Set128( Double f ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_pd( f );
}
inline __m256d SIMDMath::Set256( Double f ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_pd( f );
}

inline __m128i SIMDMath::Set128( Int8 i ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_epi8( i );
}
inline __m256i SIMDMath::Set256( Int8 i ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_epi8( i );
}

inline __m128i SIMDMath::Set128( Int16 i ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_epi16( i );
}
inline __m256i SIMDMath::Set256( Int16 i ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_epi16( i );
}

inline __m128i SIMDMath::Set128( Int32 i ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_epi32( i );
}
inline __m256i SIMDMath::Set256( Int32 i ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_epi32( i );
}

inline __m128i SIMDMath::Set128( Int64 i ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_epi64x( i );
}
inline __m256i SIMDMath::Set256( Int64 i ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_epi64x( i );
}

inline __m128i SIMDMath::Set8I( __m128i mDst, Int8 iSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_insert_epi8( mDst, iSrc, iIndex );
}
inline __m256i SIMDMath::Set8I( __m256i mDst, Int8 iSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insert_epi8( mDst, iSrc, iIndex );
}

inline __m128i SIMDMath::Set16I( __m128i mDst, Int16 iSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_insert_epi16( mDst, iSrc, iIndex );
}
inline __m256i SIMDMath::Set16I( __m256i mDst, Int16 iSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insert_epi16( mDst, iSrc, iIndex );
}

inline __m128 SIMDMath::Set32F( __m128 mDst, Float fSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    dfhbd;
}
inline __m256 SIMDMath::Set32F( __m256 mDst, Float fSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    dfhbd;
}

inline __m128i SIMDMath::Set32I( __m128i mDst, Int32 iSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_insert_epi32( mDst, iSrc, iIndex );
}
inline __m256i SIMDMath::Set32I( __m256i mDst, Int32 iSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insert_epi32( mDst, iSrc, iIndex );
}

inline __m128d SIMDMath::Set64D( __m128d mDst, Double fSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    dfhbd;
}
inline __m256d SIMDMath::Set64D( __m256d mDst, Double fSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    dfhbd;
}

inline __m128i SIMDMath::Set64I( __m128i mDst, Int64 iSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_insert_epi64( mDst, iSrc, iIndex );
}
inline __m256i SIMDMath::Set64I( __m256i mDst, Int64 iSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insert_epi64( mDst, iSrc, iIndex );
}

////////////////////////////////////////////////////////////// Registers -> Values
inline Int32 SIMDMath::Get8I( __m128i mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_extract_epi8( mSrc, iIndex );
}
inline Int32 SIMDMath::Get8I( __m256i mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_extract_epi8( mSrc, iIndex );
}

inline Int32 SIMDMath::Get16I( __m128i mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_extract_epi16( mSrc, iIndex );
}
inline Int32 SIMDMath::Get16I( __m256i mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_extract_epi16( mSrc, iIndex );
}

inline Float SIMDMath::Get32F( __m128 mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    FloatConverter hConv;
    hConv.i = _mm_extract_ps( mSrc, iIndex );
    return hConv.f;
}
inline Float SIMDMath::Get32F( __m256 mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    FloatConverter hConv;
    hConv.i = _mm256_extract_ps( mSrc, iIndex );
    return hConv.f;
}

inline Int32 SIMDMath::Get32I( __m128i mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_extract_epi32( mSrc, iIndex );
}
inline Int32 SIMDMath::Get32I( __m256i mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extract_epi32( mSrc, iIndex );
}

inline Double SIMDMath::Get64D( __m128d mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    DoubleConverter hConv;
    hConv.i = _mm_extract_pd( mSrc, iIndex );
    return hConv.f;
}
inline Double SIMDMath::Get64D( __m256d mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    DoubleConverter hConv;
    hConv.i = _mm256_extract_pd( mSrc, iIndex );
    return hConv.f;
}

inline Int64 SIMDMath::Get64I( __m128i mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_extract_epi64( mSrc, iIndex );
}
inline Int64 SIMDMath::Get64I( __m256i mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extract_epi64( mSrc, iIndex );
}

////////////////////////////////////////////////////////////// Memory -> Registers
inline __m128 SIMDMath::LoadLower( const Float * arrF ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_load_ss( arrF );
}
inline __m128d SIMDMath::LoadLower( const Double * arrF ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_sd( arrF );
}

inline __m128 SIMDMath::Load128Aligned( const Float * arrF ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_load_ps( arrF );
}
inline __m256 SIMDMath::Load256Aligned( const Float * arrF ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_ps( arrF );
}

inline __m128d SIMDMath::Load128Aligned( const Double * arrF ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_pd( arrF );
}
inline __m256d SIMDMath::Load256Aligned( const Double * arrF ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_pd( arrF );
}

inline __m128i SIMDMath::Load128Aligned( const __m128i * arrSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_load_si128( arrSrc );
}
inline __m256i SIMDMath::Load256Aligned( const __m256i * arrSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_load_si256( arrSrc );
}

inline __m128 SIMDMath::Load128( const Float * arrF ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_loadu_ps( arrF );
}
inline __m256 SIMDMath::Load256( const Float * arrF ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_loadu_ps( arrF );
}

inline __m128d SIMDMath::Load128( const Double * arrF ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadu_pd( arrF );
}
inline __m256d SIMDMath::Load256( const Double * arrF ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_loadu_pd( arrF );
}

inline __m128i SIMDMath::Load128( const __m128i * arrSrc ) const {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_lddqu_si128( arrSrc );
}
inline __m256i SIMDMath::Load256( const __m256i * arrSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_lddqu_si256( arrSrc );
}

inline __m128 SIMDMath::Load128AlignedR( const Float * arrF ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_loadr_ps( arrF );
}
inline __m128d SIMDMath::Load128AlignedR( const Double * arrF ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadr_pd( arrF );
}

inline __m128 SIMDMath::Load128Dupe( const Float * pF ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_load_ps1( pF );
}
inline __m128d SIMDMath::Load128Dupe( const Double * pF ) const {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_loaddup_pd( pF );
}

inline __m128 SIMDMath::LoadTwoFloatL( __m128 mDst, const __m64 * arrSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_loadl_pi( mDst, arrSrc );
}
inline __m128 SIMDMath::LoadTwoFloatH( __m128 mDst, const __m64 * arrSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_loadh_pi( mDst, arrSrc );
}

inline __m128d SIMDMath::LoadOneDoubleL( __m128d mDst, const Double * arrF ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadl_pd( mDst, arrF );
}
inline __m128d SIMDMath::LoadOneDoubleH( __m128d mDst, const Double * arrF ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadh_pd( mDst, arrF );
}

inline __m128i SIMDMath::LoadOneInt64L( const __m128i * arrSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadl_epi64( arrSrc );
}

inline __m128 SIMDMath::Dupe128Float( const Float * pF ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_broadcast_ss( pF );
}
inline __m256 SIMDMath::Dupe256Float( const Float * pF ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_broadcast_ss( pF );
}

inline __m256d SIMDMath::Dupe256Double( const Double * pF ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_broadcast_sd( pF );
}

inline __m256 SIMDMath::Dupe256FourFloat( const __m128 * pSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_broadcast_ps( pSrc );
}
inline __m256d SIMDMath::Dupe256TwoDouble( const __m128d * pSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_broadcast_pd( pSrc );
}

inline __m128 SIMDMath::Load32FourFloat( const Float * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_ps( pSrc, mIndices, iStride );
}
inline __m256 SIMDMath::Load32EightFloat( const Float * pSrc, __m256i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_ps( pSrc, mIndices, iStride );
}

inline __m128d SIMDMath::Load32TwoDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_pd( pSrc, mIndices, iStride );
}
inline __m256d SIMDMath::Load32FourDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_pd( pSrc, mIndices, iStride );
}

inline __m128i SIMDMath::Load32FourInt32( const Int32 * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_epi32( pSrc, mIndices, iStride );
}
inline __m256i SIMDMath::Load32EightInt32( const Int32 * pSrc, __m256i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_epi32( pSrc, mIndices, iStride );
}

inline __m128i SIMDMath::Load32TwoInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_epi64( pSrc, mIndices, iStride );
}
inline __m256i SIMDMath::Load32FourInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_epi64( pSrc, mIndices, iStride );
}

inline __m128 SIMDMath::Load64TwoFloat( const Float * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_ps( pSrc, mIndices, iStride );
}
inline __m128 SIMDMath::Load64FourFloat( const Float * pSrc, __m256i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_ps( pSrc, mIndices, iStride );
}

inline __m128d SIMDMath::Load64TwoDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_pd( pSrc, mIndices, iStride );
}
inline __m256d SIMDMath::Load64FourDouble( const Double * pSrc, __m256i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_pd( pSrc, mIndices, iStride );
}

inline __m128i SIMDMath::Load64TwoInt32( const Int32 * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_epi32( pSrc, mIndices, iStride );
}
inline __m128i SIMDMath::Load64FourInt32( const Int32 * pSrc, __m256i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_epi32( pSrc, mIndices, iStride );
}

inline __m128i SIMDMath::Load64TwoInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_epi64( pSrc, mIndices, iStride );
}
inline __m256i SIMDMath::Load64FourInt64( const Int64 * pSrc, __m256i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_epi64( pSrc, mIndices, iStride );
}

inline __m128i SIMDMath::LoadNT128Aligned( const __m128i * arrSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_stream_load_si128( arrSrc );
}
inline __m256i SIMDMath::LoadNT256Aligned( const __m256i * arrSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_stream_load_si256( arrSrc );
}

////////////////////////////////////////////////////////////// Registers -> Memory
inline Void SIMDMath::StoreLower( Float * outDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_store_ss( outDst, mSrc );
}
inline Void SIMDMath::StoreLower( Double * outDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_sd( outDst, mSrc );
}

inline Void SIMDMath::Store128Aligned( Float * outDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_store_ps( outDst, mSrc );
}
inline Void SIMDMath::Store256Aligned( Float * outDst, __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_ps( outDst, mSrc );
}

inline Void SIMDMath::Store128Aligned( Double * outDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_pd( outDst, mSrc );
}
inline Void SIMDMath::Store256Aligned( Double * outDst, __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_pd( outDst, mSrc );
}

inline Void SIMDMath::Store128Aligned( __m128i * outDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store_si128( outDst, mSrc );
}
inline Void SIMDMath::Store256Aligned( __m256i * outDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_store_si256( outDst, mSrc );
}

inline Void SIMDMath::Store128( Float * outDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_storeu_ps( outDst, mSrc );
}
inline Void SIMDMath::Store256( Float * outDst, __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_ps( outDst, mSrc );
}

inline Void SIMDMath::Store128( Double * outDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeu_pd( outDst, mSrc );
}
inline Void SIMDMath::Store256( Double * outDst, __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_pd( outDst, mSrc );
}

inline Void SIMDMath::Store128( __m128i * outDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeu_si128( outDst, mSrc );
}
inline Void SIMDMath::Store256( __m256i * outDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_storeu_si256( outDst, mSrc );
}

inline Void SIMDMath::Store128AlignedR( Float * outDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_storer_ps( outDst, mSrc );
}
inline Void SIMDMath::Store128AlignedR( Double * outDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storer_pd( outDst, mSrc );
}

inline Void SIMDMath::Store128Dupe( Float * outDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_store_ps1( outDst, mSrc );
}
inline Void SIMDMath::Store128Dupe( Double * outDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_store1_pd( outDst, mSrc );
}

inline Void SIMDMath::StoreTwoFloatL( __m64 * outDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_storel_pi( outDst, mSrc );
}
inline Void SIMDMath::StoreTwoFloatH( __m64 * outDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_storeh_pi( outDst, mSrc );
}

inline Void SIMDMath::StoreOneDoubleL( Double * outDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storel_pd( outDst, mSrc );
}
inline Void SIMDMath::StoreOneDoubleH( Double * outDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storeh_pd( outDst, mSrc );
}

inline Void SIMDMath::StoreOneInt64L( __m128i * outDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_storel_epi64( outDst, mSrc );
}

inline Void SIMDMath::StoreNTLower( Float * outDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    _mm_stream_ss( outDst, mSrc );
}
inline Void SIMDMath::StoreNTLower( Double * outDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    _mm_stream_sd( outDst, mSrc );
}

inline Void SIMDMath::StoreNT128Aligned( Float * outDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_stream_ps( outDst, mSrc );
}
inline Void SIMDMath::StoreNT256Aligned( Float * outDst, __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_stream_ps( outDst, mSrc );
}

inline Void SIMDMath::StoreNT128Aligned( Double * outDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_stream_pd( outDst, mSrc );
}
inline Void SIMDMath::StoreNT256Aligned( Double * outDst, __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    __mm256_stream_pd( outDst, mSrc );
}

inline Void SIMDMath::StoreNT128Aligned( __m128i * outDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_stream_si128( outDst, mSrc );
}
inline Void SIMDMath::StoreNT256Aligned( __m256i * outDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    __mm256_stream_si256( outDst, mSrc );
}

////////////////////////////////////////////////////////////// Registers <-> Registers
inline __m128 SIMDMath::MoveOneFloatLL( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_move_ss( mDst, mSrc );
}
inline __m128 SIMDMath::MoveTwoFloatLH( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_movehl_ps( mDst, mSrc );
}
inline __m128 SIMDMath::MoveTwoFloatHL( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_movelh_ps( mDst, mSrc );
}

inline __m128d SIMDMath::MoveOneDoubleLL( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_move_sd( mDst, mSrc );
}

inline __m128i SIMDMath::MoveOneInt64LL( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_move_epi64( mSrc );
}

inline __m128 SIMDMath::DupeTwoFloatEven( __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_moveldup_ps( mSrc );
}
inline __m128 SIMDMath::DupeTwoFloatOdd( __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_movehdup_ps( mSrc );
}
inline __m128d SIMDMath::DupeOneDoubleL( __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_movedup_pd( mSrc );
}

inline __m256 SIMDMath::DupeFourFloatEven( __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_moveldup_ps( mSrc );
}
inline __m256 SIMDMath::DupeFourFloatOdd( __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_movehdup_ps( mSrc );
}
inline __m256d SIMDMath::DupeTwoDoubleEven( __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_movedup_pd( mSrc );
}

inline __m128 SIMDMath::Dupe128Float( __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastss_ps( mSrc );
}
inline __m256 SIMDMath::Dupe256Float( __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastss_ps( mSrc );
}

inline __m128d SIMDMath::Dupe128Double( __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastsd_pd( mSrc );
}
inline __m256d SIMDMath::Dupe256Double( __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastsd_pd( mSrc );
}

inline __m128i SIMDMath::Dupe128Int8( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastb_epi8( mSrc );
}
inline __m256i SIMDMath::Dupe256Int8( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastb_epi8( mSrc );
}

inline __m128i SIMDMath::Dupe128Int16( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastw_epi16( mSrc );
}
inline __m256i SIMDMath::Dupe256Int16( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastw_epi16( mSrc );
}

inline __m128i SIMDMath::Dupe128Int32( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastd_epi32( mSrc );
}
inline __m256i SIMDMath::Dupe256Int32( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastd_epi32( mSrc );
}

inline __m128i SIMDMath::Dupe128Int64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_broadcastq_epi64( mSrc );
}
inline __m256i SIMDMath::Dupe256Int64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastq_epi64( mSrc );
}

inline __m256i SIMDMath::Dupe256Int128( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_broadcastsi128_si256( mSrc );
}

inline __m128 SIMDMath::Extract128F( __m256 mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extractf128_ps( mSrc, iIndex );
}
inline __m128d SIMDMath::Extract128D( __m256d mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_extractf128_pd( mSrc, iIndex );
}
inline __m128i SIMDMath::Extract128I( __m256i mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_extracti128_si256( mSrc, iIndex );
}

inline __m256 SIMDMath::Insert128F( __m256 mDst, __m128 mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insertf128_ps( mDst, mSrc, iIndex );
}
inline __m256d SIMDMath::Insert128D( __m256d mDst, __m128d mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_insertf128_pd( mDst, mSrc, iIndex );
}
inline __m256i SIMDMath::Insert128I( __m256i mDst, __m128i mSrc, Int32 iIndex ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_inserti128_si256( mDst, mSrc, iIndex );
}

////////////////////////////////////////////////////////////// Pack / Unpack
inline __m128i SIMDMath::PackSigned16To8( __m128i mSrcLow, __m128i mSrcHigh ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_packs_epi16( mSrcLow, mSrcHigh );
}
inline __m256i SIMDMath::PackSigned16To8( __m256i mSrcLow, __m256i mSrcHigh ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packs_epi16( mSrcLow, mSrcHigh );
}

inline __m128i SIMDMath::PackSigned32To16( __m128i mSrcLow, __m128i mSrcHigh ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_packs_epi32( mSrcLow, mSrcHigh );
}
inline __m256i SIMDMath::PackSigned32To16( __m256i mSrcLow, __m256i mSrcHigh ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packs_epi32( mSrcLow, mSrcHigh );
}

inline __m128i SIMDMath::PackUnsigned16To8( __m128i mSrcLow, __m128i mSrcHigh ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_packus_epi16( mSrcLow, mSrcHigh );
}
inline __m256i SIMDMath::PackUnsigned16To8( __m256i mSrcLow, __m256i mSrcHigh ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packus_epi16( mSrcLow, mSrcHigh );
}

inline __m128i SIMDMath::PackUnsigned32To16( __m128i mSrcLow, __m128i mSrcHigh ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_packus_epi32( mSrcLow, mSrcHigh );
}
inline __m256i SIMDMath::PackUnsigned32To16( __m256i mSrcLow, __m256i mSrcHigh ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_packus_epi32( mSrcLow, mSrcHigh );
}

inline __m128 SIMDMath::UnpackFloatL( __m128 mSrcEven, __m128 mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_unpacklo_ps( mSrcEven, mSrcOdd );
}
inline __m256 SIMDMath::UnpackFloatL( __m256 mSrcEven, __m256 mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpacklo_ps( mSrcEven, mSrcOdd );
}

inline __m128 SIMDMath::UnpackFloatH( __m128 mSrcEven, __m128 mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_unpackhi_ps( mSrcEven, mSrcOdd );
}
inline __m256 SIMDMath::UnpackFloatH( __m256 mSrcEven, __m256 mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpackhi_ps( mSrcEven, mSrcOdd );
}

inline __m128d SIMDMath::UnpackDoubleL( __m128d mSrcEven, __m128d mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_pd( mSrcEven, mSrcOdd );
}
inline __m256d SIMDMath::UnpackDoubleL( __m256d mSrcEven, __m256d mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpacklo_pd( mSrcEven, mSrcOdd );
}

inline __m128d SIMDMath::UnpackDoubleH( __m128d mSrcEven, __m128d mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_pd( mSrcEven, mSrcOdd );
}
inline __m256d SIMDMath::UnpackDoubleH( __m256d mSrcEven, __m256d mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_unpackhi_pd( mSrcEven, mSrcOdd );
}

inline __m128i SIMDMath::UnpackInt8L( __m128i mSrcEven, __m128i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi8( mSrcEven, mSrcOdd );
}
inline __m256i SIMDMath::UnpackInt8L( __m256i mSrcEven, __m256i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi8( mSrcEven, mSrcOdd );
}

inline __m128i SIMDMath::UnpackInt8H( __m128i mSrcEven, __m128i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi8( mSrcEven, mSrcOdd );
}
inline __m256i SIMDMath::UnpackInt8H( __m256i mSrcEven, __m256i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi8( mSrcEven, mSrcOdd );
}

inline __m128i SIMDMath::UnpackInt16L( __m128i mSrcEven, __m128i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi16( mSrcEven, mSrcOdd );
}
inline __m256i SIMDMath::UnpackInt16L( __m256i mSrcEven, __m256i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi16( mSrcEven, mSrcOdd );
}

inline __m128i SIMDMath::UnpackInt16H( __m128i mSrcEven, __m128i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi16( mSrcEven, mSrcOdd );
}
inline __m256i SIMDMath::UnpackInt16H( __m256i mSrcEven, __m256i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi16( mSrcEven, mSrcOdd );
}

inline __m128i SIMDMath::UnpackInt32L( __m128i mSrcEven, __m128i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi32( mSrcEven, mSrcOdd );
}
inline __m256i SIMDMath::UnpackInt32L( __m256i mSrcEven, __m256i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi32( mSrcEven, mSrcOdd );
}

inline __m128i SIMDMath::UnpackInt32H( __m128i mSrcEven, __m128i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi32( mSrcEven, mSrcOdd );
}
inline __m256i SIMDMath::UnpackInt32H( __m256i mSrcEven, __m256i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi32( mSrcEven, mSrcOdd );
}

inline __m128i SIMDMath::UnpackInt64L( __m128i mSrcEven, __m128i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpacklo_epi64( mSrcEven, mSrcOdd );
}
inline __m256i SIMDMath::UnpackInt64L( __m256i mSrcEven, __m256i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpacklo_epi64( mSrcEven, mSrcOdd );
}

inline __m128i SIMDMath::UnpackInt64H( __m128i mSrcEven, __m128i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_unpackhi_epi64( mSrcEven, mSrcOdd );
}
inline __m256i SIMDMath::UnpackInt64H( __m256i mSrcEven, __m256i mSrcOdd ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_unpackhi_epi64( mSrcEven, mSrcOdd );
}

////////////////////////////////////////////////////////////// Shuffle
inline __m128 SIMDMath::Shuffle128Float( __m128 mSrcLow, __m128 mSrcHigh, Int iMask4x4 ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_shuffle_ps( mSrcLow, mSrcHigh, (unsigned)iMask4x4 );
}
inline __m128 SIMDMath::Shuffle128Float( __m128 mSrc, Int iMask4x4 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_ps( mSrc, iMask4x4 );
}
inline __m128 SIMDMath::Shuffle128Float( __m128 mSrc, __m128i mMask1x4 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permutevar_ps( mSrc, mMask1x4 );
}

inline __m256 SIMDMath::Shuffle128Float( __m256 mSrcLow, __m256 mSrcHigh, Int iMask4x4 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_shuffle_ps( mSrcLow, mSrcHigh, iMask4x4 );
}
inline __m256 SIMDMath::Shuffle128Float( __m256 mSrc, Int iMask4x4 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_ps( mSrc, iMask4x4 );
}
inline __m256 SIMDMath::Shuffle128Float( __m256 mSrc, __m256i mMask1x4 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permutevar_ps( mSrc, mMask1x4 );
}

inline __m256 SIMDMath::Shuffle256Float( __m256 mSrc, __m256i mMask1x8 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permutevar8x32_ps( mSrc, mMask1x8 );
}

inline __m256 SIMDMath::Shuffle512FourFloat( __m256 mSrc1, __m256 mSrc2, Int iMask2x4_Z ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_ps( mSrc1, mSrc2, iMask2x4_Z );
}

inline __m128d SIMDMath::Shuffle128Double( __m128d mSrcLow, __m128d mSrcHigh, Int iMask2x2 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shuffle_pd( mSrcLow, mSrcHigh, iMask2x2 );
}
inline __m128d SIMDMath::Shuffle128Double( __m128d mSrc, Int iMask2x2 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permute_pd( mSrc, iMask2x2 );
}
inline __m128d SIMDMath::Shuffle128Double( __m128d mSrc, __m128i mMask1x2 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_permutevar_pd( mSrc, mMask1x2 );
}

inline __m256d SIMDMath::Shuffle128Double( __m256d mSrcLow, __m256d mSrcHigh, Int iMask4x2 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_shuffle_pd( mSrcLow, mSrcHigh, iMask4x2 );
}
inline __m256d SIMDMath::Shuffle128Double( __m256d mSrc, Int iMask4x2 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute_pd( mSrc, iMask4x2 );
}
inline __m256d SIMDMath::Shuffle128Double( __m256d mSrc, __m256i mMask1x2 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permutevar_pd( mSrc, mMask1x2 );
}

inline __m256d SIMDMath::Shuffle256Double( __m256d mSrc, Int iMask4x4 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_pd( mSrc, iMask4x4 );
}

inline __m256d SIMDMath::Shuffle512TwoDouble( __m256d mSrc1, __m256d mSrc2, Int iMask2x4_Z ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_permute2f128_pd( mSrc1, mSrc2, iMask2x4_Z );
}

inline __m128i SIMDMath::Shuffle128Int8( __m128i mSrc, __m128i mMask1x16_Z ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_shuffle_epi8( mSrc, mMask1x16_Z );
}
inline __m256i SIMDMath::Shuffle128Int8( __m256i mSrc, __m256i mMask1x16_Z ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi8( mSrc, mMask1x16_Z );
}

inline __m128i SIMDMath::Shuffle64Int16L( __m128i mSrc, Int iMask4x4 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflelo_epi16( mSrc, iMask4x4 );
}
inline __m256i SIMDMath::Shuffle64Int16L( __m256i mSrc, Int iMask4x4 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shufflelo_epi16( mSrc, iMask4x4 );
}

inline __m128i SIMDMath::Shuffle64Int16H( __m128i mSrc, Int iMask4x4 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shufflehi_epi16( mSrc, iMask4x4 );
}
inline __m256i SIMDMath::Shuffle64Int16H( __m256i mSrc, Int iMask4x4 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shufflehi_epi16( mSrc, iMask4x4 );
}

inline __m128i SIMDMath::Shuffle128Int32( __m128i mSrc, Int iMask4x4 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_shuffle_epi32( mSrc, iMask4x4 );
}
inline __m256i SIMDMath::Shuffle128Int32( __m256i mSrc, Int iMask4x4 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_shuffle_epi32( mSrc, iMask4x4 );
}

inline __m256i SIMDMath::Shuffle256Int32( __m256i mSrc, __m256i mMask1x8 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permutevar8x32_epi32( mSrc, mMask1x8 );
}

inline __m256i SIMDMath::Shuffle512FourInt32( __m256i mSrc1, __m256i mSrc2, Int iMask2x4_Z ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute2x128_si256( mSrc1, mSrc2, iMask2x4_Z );
}

inline __m256i SIMDMath::Shuffle256Int64( __m256i mSrc, Int iMask4x4 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_permute4x64_epi64( mSrc, iMask4x4 );
}

////////////////////////////////////////////////////////////// Fast Invert & SquareRoot Functions
inline __m128 SIMDMath::InvertLower( __m128 mFloat4 ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rcp_ss( mFloat4 );
}

inline __m128 SIMDMath::Invert( __m128 mFloat4 ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rcp_ps( mFloat4 );
}
inline __m256 SIMDMath::Invert( __m256 mFloat8 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rcp_ps( mFloat8 );
}

inline __m128 SIMDMath::SqrtLower( __m128 mFloat4 ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sqrt_ss( mFloat4 );
}
inline __m128d SIMDMath::SqrtLower( __m128d mDouble2 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sqrt_sd( mDouble2, mDouble2 );
}

inline __m128 SIMDMath::Sqrt( __m128 mFloat4 ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sqrt_ps( mFloat4 );
}
inline __m256 SIMDMath::Sqrt( __m256 mFloat8 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sqrt_ps( mFloat8 );
}

inline __m128d SIMDMath::Sqrt( __m128d mDouble2 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sqrt_pd( mDouble2 );
}
inline __m256d SIMDMath::Sqrt( __m256d mDouble4 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sqrt_pd( mDouble4 );
}

inline __m128 SIMDMath::InvSqrtLower( __m128 mFloat4 ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rsqrt_ss( mFloat4 );
}

inline __m128 SIMDMath::InvSqrt( __m128 mFloat4 ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rsqrt_ps( mFloat4 );
}
inline __m256 SIMDMath::InvSqrt( __m256 mFloat8 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rsqrt_ps( mFloat8 );
}



//
//inline Void SSEStack::Push( Float fValue0 ) {
//    m_pAlignedFloatScratch[0] = fValue0;
//    _IPush( _SSE_SS, (QWord*)m_pAlignedFloatScratch );
//}
//inline Void SSEStack::Push( Float fValue0, Float fValue1 ) {
//    m_pAlignedFloatScratch[0] = fValue0;
//    m_pAlignedFloatScratch[1] = fValue1;
//    m_pAlignedFloatScratch[2] = 1.0f;
//    m_pAlignedFloatScratch[3] = 1.0f;
//    _IPush( _SSE_PS, (QWord*)m_pAlignedFloatScratch );
//}
//inline Void SSEStack::Push( Float fValue0, Float fValue1, Float fValue2 ) {
//    m_pAlignedFloatScratch[0] = fValue0;
//    m_pAlignedFloatScratch[1] = fValue1;
//    m_pAlignedFloatScratch[2] = fValue2;
//    m_pAlignedFloatScratch[3] = 1.0f;
//    _IPush( _SSE_PS, (QWord*)m_pAlignedFloatScratch );
//}
//inline Void SSEStack::Push( Float fValue0, Float fValue1, Float fValue2, Float fValue3 ) {
//    m_pAlignedFloatScratch[0] = fValue0;
//    m_pAlignedFloatScratch[1] = fValue1;
//    m_pAlignedFloatScratch[2] = fValue2;
//    m_pAlignedFloatScratch[3] = fValue3;
//    _IPush( _SSE_PS, (QWord*)m_pAlignedFloatScratch );
//}
//inline Void SSEStack::Push( Float * fValues ) {
//    m_pAlignedFloatScratch[0] = fValues[0];
//    m_pAlignedFloatScratch[1] = fValues[1];
//    m_pAlignedFloatScratch[2] = fValues[2];
//    m_pAlignedFloatScratch[3] = fValues[3];
//    _IPush( _SSE_PS, (QWord*)m_pAlignedFloatScratch );
//}
//
//inline Void SSEStack::Push( Double fValue0 ) {
//    m_pAlignedDoubleScratch[0] = fValue0;
//    _IPush( _SSE_SD, (QWord*)m_pAlignedDoubleScratch );
//}
//inline Void SSEStack::Push( Double fValue0, Double fValue1 ) {
//    m_pAlignedDoubleScratch[0] = fValue0;
//    m_pAlignedDoubleScratch[1] = fValue1;
//    _IPush( _SSE_PD, (QWord*)m_pAlignedDoubleScratch );
//}
//inline Void SSEStack::Push( Double * fValues ) {
//    m_pAlignedDoubleScratch[0] = fValues[0];
//    m_pAlignedDoubleScratch[1] = fValues[1];
//    _IPush( _SSE_PD, (QWord*)m_pAlignedDoubleScratch );
//}
//
//inline Void SSEStack::PopF() {
//    _IPop( _SSE_PS, (QWord*)m_pAlignedFloatScratch );
//}
//inline Void SSEStack::Pop( Float & outValue0 ) {
//    _IPop( _SSE_SS, (QWord*)m_pAlignedFloatScratch );
//    outValue0 = m_pAlignedFloatScratch[0];
//}
//inline Void SSEStack::Pop( Float & outValue0, Float & outValue1 ) {
//    _IPop( _SSE_PS, (QWord*)m_pAlignedFloatScratch );
//    outValue0 = m_pAlignedFloatScratch[0];
//    outValue1 = m_pAlignedFloatScratch[1];
//}
//inline Void SSEStack::Pop( Float & outValue0, Float & outValue1, Float & outValue2 ) {
//    _IPop( _SSE_PS, (QWord*)m_pAlignedFloatScratch );
//    outValue0 = m_pAlignedFloatScratch[0];
//    outValue1 = m_pAlignedFloatScratch[1];
//    outValue2 = m_pAlignedFloatScratch[2];
//}
//inline Void SSEStack::Pop( Float & outValue0, Float & outValue1, Float & outValue2, Float & outValue3 ) {
//    _IPop( _SSE_PS, (QWord*)m_pAlignedFloatScratch );
//    outValue0 = m_pAlignedFloatScratch[0];
//    outValue1 = m_pAlignedFloatScratch[1];
//    outValue2 = m_pAlignedFloatScratch[2];
//    outValue3 = m_pAlignedFloatScratch[3];
//}
//inline Void SSEStack::Pop( Float * outValues ) {
//    _IPop( _SSE_PS, (QWord*)m_pAlignedFloatScratch );
//    outValues[0] = m_pAlignedFloatScratch[0];
//    outValues[1] = m_pAlignedFloatScratch[1];
//    outValues[2] = m_pAlignedFloatScratch[2];
//    outValues[3] = m_pAlignedFloatScratch[3];
//}
//
//inline Void SSEStack::PopD() {
//    _IPop( _SSE_PD, (QWord*)m_pAlignedDoubleScratch );
//}
//inline Void SSEStack::Pop( Double & outValue0 ) {
//    _IPop( _SSE_SD, (QWord*)m_pAlignedDoubleScratch );
//    outValue0 = m_pAlignedDoubleScratch[0];
//}
//inline Void SSEStack::Pop( Double & outValue0, Double & outValue1 ) {
//    _IPop( _SSE_PD, (QWord*)m_pAlignedDoubleScratch );
//    outValue0 = m_pAlignedDoubleScratch[0];
//    outValue1 = m_pAlignedDoubleScratch[1];
//}
//inline Void SSEStack::Pop( Double * outValues ) {
//    _IPop( _SSE_PD, (QWord*)m_pAlignedDoubleScratch );
//    outValues[0] = m_pAlignedDoubleScratch[0];
//    outValues[1] = m_pAlignedDoubleScratch[1];
//}
//
//inline Void SSEStack::Set( UInt iStackIndex, Float fValue0 ) {
//    m_pAlignedFloatScratch[0] = fValue0;
//    _SSE::_ISet( _SSE_SS, _SSE_ST(iStackIndex), (QWord*)m_pAlignedFloatScratch );
//}
//inline Void SSEStack::Set( UInt iStackIndex, Float fValue0, Float fValue1 ) {
//    m_pAlignedFloatScratch[0] = fValue0;
//    m_pAlignedFloatScratch[1] = fValue1;
//    m_pAlignedFloatScratch[2] = 1.0f;
//    m_pAlignedFloatScratch[3] = 1.0f;
//    _SSE::_ISet( _SSE_PS, _SSE_ST(iStackIndex), (QWord*)m_pAlignedFloatScratch );
//}
//inline Void SSEStack::Set( UInt iStackIndex, Float fValue0, Float fValue1, Float fValue2 ) {
//    m_pAlignedFloatScratch[0] = fValue0;
//    m_pAlignedFloatScratch[1] = fValue1;
//    m_pAlignedFloatScratch[2] = fValue2;
//    m_pAlignedFloatScratch[3] = 1.0f;
//    _SSE::_ISet( _SSE_PS, _SSE_ST(iStackIndex), (QWord*)m_pAlignedFloatScratch );
//}
//inline Void SSEStack::Set( UInt iStackIndex, Float fValue0, Float fValue1, Float fValue2, Float fValue3 ) {
//    m_pAlignedFloatScratch[0] = fValue0;
//    m_pAlignedFloatScratch[1] = fValue1;
//    m_pAlignedFloatScratch[2] = fValue2;
//    m_pAlignedFloatScratch[3] = fValue3;
//    _SSE::_ISet( _SSE_PS, _SSE_ST(iStackIndex), (QWord*)m_pAlignedFloatScratch );
//}
//inline Void SSEStack::Set( UInt iStackIndex, Float * fValues ) {
//    m_pAlignedFloatScratch[0] = fValues[0];
//    m_pAlignedFloatScratch[1] = fValues[1];
//    m_pAlignedFloatScratch[2] = fValues[2];
//    m_pAlignedFloatScratch[3] = fValues[3];
//    _SSE::_ISet( _SSE_PS, _SSE_ST(iStackIndex), (QWord*)m_pAlignedFloatScratch );
//}
//
//inline Void SSEStack::Set( UInt iStackIndex, Double fValue0 ) {
//    m_pAlignedDoubleScratch[0] = fValue0;
//    _SSE::_ISet( _SSE_SD, _SSE_ST(iStackIndex), (QWord*)m_pAlignedDoubleScratch );
//}
//inline Void SSEStack::Set( UInt iStackIndex, Double fValue0, Double fValue1 ) {
//    m_pAlignedDoubleScratch[0] = fValue0;
//    m_pAlignedDoubleScratch[1] = fValue1;
//    _SSE::_ISet( _SSE_PD, _SSE_ST(iStackIndex), (QWord*)m_pAlignedDoubleScratch );
//}
//inline Void SSEStack::Set( UInt iStackIndex, Double * fValues ) {
//    m_pAlignedDoubleScratch[0] = fValues[0];
//    m_pAlignedDoubleScratch[1] = fValues[1];
//    _SSE::_ISet( _SSE_PD, _SSE_ST(iStackIndex), (QWord*)m_pAlignedDoubleScratch );
//}
//
//inline Void SSEStack::Get( UInt iStackIndex, Float & outValue0 ) {
//    _SSE::_IGet( _SSE_SS, (QWord*)m_pAlignedFloatScratch, _SSE_ST(iStackIndex) );
//    outValue0 = m_pAlignedFloatScratch[0];
//}
//inline Void SSEStack::Get( UInt iStackIndex, Float & outValue0, Float & outValue1 ) {
//    _SSE::_IGet( _SSE_PS, (QWord*)m_pAlignedFloatScratch, _SSE_ST(iStackIndex) );
//    outValue0 = m_pAlignedFloatScratch[0];
//    outValue1 = m_pAlignedFloatScratch[1];
//}
//inline Void SSEStack::Get( UInt iStackIndex, Float & outValue0, Float & outValue1, Float & outValue2 ) {
//    _SSE::_IGet( _SSE_PS, (QWord*)m_pAlignedFloatScratch, _SSE_ST(iStackIndex) );
//    outValue0 = m_pAlignedFloatScratch[0];
//    outValue1 = m_pAlignedFloatScratch[1];
//    outValue2 = m_pAlignedFloatScratch[2];
//}
//inline Void SSEStack::Get( UInt iStackIndex, Float & outValue0, Float & outValue1, Float & outValue2, Float & outValue3 ) {
//    _SSE::_IGet( _SSE_PS, (QWord*)m_pAlignedFloatScratch, _SSE_ST(iStackIndex) );
//    outValue0 = m_pAlignedFloatScratch[0];
//    outValue1 = m_pAlignedFloatScratch[1];
//    outValue2 = m_pAlignedFloatScratch[2];
//    outValue3 = m_pAlignedFloatScratch[3];
//}
//inline Void SSEStack::Get( UInt iStackIndex, Float * outValues ) {
//    _SSE::_IGet( _SSE_PS, (QWord*)m_pAlignedFloatScratch, _SSE_ST(iStackIndex) );
//    outValues[0] = m_pAlignedFloatScratch[0];
//    outValues[1] = m_pAlignedFloatScratch[1];
//    outValues[2] = m_pAlignedFloatScratch[2];
//    outValues[3] = m_pAlignedFloatScratch[3];
//}
//
//inline Void SSEStack::Get( UInt iStackIndex, Double & outValue0 ) {
//    _SSE::_IGet( _SSE_SD, (QWord*)m_pAlignedDoubleScratch, _SSE_ST(iStackIndex) );
//    outValue0 = m_pAlignedDoubleScratch[0];
//}
//inline Void SSEStack::Get( UInt iStackIndex, Double & outValue0, Double & outValue1 ) {
//    _SSE::_IGet( _SSE_PD, (QWord*)m_pAlignedDoubleScratch, _SSE_ST(iStackIndex) );
//    outValue0 = m_pAlignedDoubleScratch[0];
//    outValue1 = m_pAlignedDoubleScratch[1];
//}
//inline Void SSEStack::Get( UInt iStackIndex, Double * outValues ) {
//    _SSE::_IGet( _SSE_PD, (QWord*)m_pAlignedDoubleScratch, _SSE_ST(iStackIndex) );
//    outValues[0] = m_pAlignedDoubleScratch[0];
//    outValues[1] = m_pAlignedDoubleScratch[1];
//}
//
//inline Void SSEStack::PushF( UInt iSrcIndex ) {
//    _SSE::_IMov( _SSE_PS, _SSE_ST(-1), _SSE_ST(iSrcIndex) );
//    ++m_iTop;
//}
//inline Void SSEStack::PushD( UInt iSrcIndex ) {
//    _SSE::_IMov( _SSE_PD, _SSE_ST(-1), _SSE_ST(iSrcIndex) );
//    ++m_iTop;
//}
//inline Void SSEStack::PopF( UInt iDestIndex ) {
//    _SSE::_IMov( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) );
//    --m_iTop;
//}
//inline Void SSEStack::PopD( UInt iDestIndex ) {
//    _SSE::_IMov( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) );
//    --m_iTop;
//}
//inline Void SSEStack::MovF( UInt iDestIndex, UInt iSrcIndex ) {
//    _SSE::_IMov( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) );
//}
//inline Void SSEStack::MovD( UInt iDestIndex, UInt iSrcIndex ) {
//    _SSE::_IMov( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) );
//}
//
//inline Void SSEStack::AddF() { _SSE::_IAdd( _SSE_PS, _SSE_ST(1), _SSE_ST(0) ); }
//inline Void SSEStack::AddD() { _SSE::_IAdd( _SSE_PD, _SSE_ST(1), _SSE_ST(0) ); }
//inline Void SSEStack::SubF() { _SSE::_ISub( _SSE_PS, _SSE_ST(1), _SSE_ST(0) ); }
//inline Void SSEStack::SubD() { _SSE::_ISub( _SSE_PD, _SSE_ST(1), _SSE_ST(0) ); }
//inline Void SSEStack::MulF() { _SSE::_IMul( _SSE_PS, _SSE_ST(1), _SSE_ST(0) ); }
//inline Void SSEStack::MulD() { _SSE::_IMul( _SSE_PD, _SSE_ST(1), _SSE_ST(0) ); }
//inline Void SSEStack::DivF() { _SSE::_IDiv( _SSE_PS, _SSE_ST(1), _SSE_ST(0) ); }
//inline Void SSEStack::DivD() { _SSE::_IDiv( _SSE_PD, _SSE_ST(1), _SSE_ST(0) ); }
//
//inline Void SSEStack::AddF( UInt iDestIndex ) { _SSE::_IAdd( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) ); }
//inline Void SSEStack::AddD( UInt iDestIndex ) { _SSE::_IAdd( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) ); }
//inline Void SSEStack::SubF( UInt iDestIndex ) { _SSE::_ISub( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) ); }
//inline Void SSEStack::SubD( UInt iDestIndex ) { _SSE::_ISub( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) ); }
//inline Void SSEStack::MulF( UInt iDestIndex ) { _SSE::_IMul( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) ); }
//inline Void SSEStack::MulD( UInt iDestIndex ) { _SSE::_IMul( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) ); }
//inline Void SSEStack::DivF( UInt iDestIndex ) { _SSE::_IDiv( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) ); }
//inline Void SSEStack::DivD( UInt iDestIndex ) { _SSE::_IDiv( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) ); }
//
//inline Void SSEStack::AddF( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_IAdd( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//inline Void SSEStack::AddD( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_IAdd( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//inline Void SSEStack::SubF( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_ISub( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//inline Void SSEStack::SubD( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_ISub( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//inline Void SSEStack::MulF( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_IMul( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//inline Void SSEStack::MulD( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_IMul( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//inline Void SSEStack::DivF( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_IDiv( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//inline Void SSEStack::DivD( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_IDiv( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//
//inline Void SSEStack::SubRF() {
//    _SSE::_IMov( _SSE_PS, _SSE_ST(-1), _SSE_ST(0) );
//    _SSE::_ISub( _SSE_PS, _SSE_ST(0), _SSE_ST(1) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(1), _SSE_ST(0) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(0), _SSE_ST(-1) );
//}
//inline Void SSEStack::SubRD() {
//    _SSE::_IMov( _SSE_PD, _SSE_ST(-1), _SSE_ST(0) );
//    _SSE::_ISub( _SSE_PD, _SSE_ST(0), _SSE_ST(1) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(1), _SSE_ST(0) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(0), _SSE_ST(-1) );
//}
//inline Void SSEStack::DivRF() {
//    _SSE::_IMov( _SSE_PS, _SSE_ST(-1), _SSE_ST(0) );
//    _SSE::_IDiv( _SSE_PS, _SSE_ST(0), _SSE_ST(1) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(1), _SSE_ST(0) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(0), _SSE_ST(-1) );
//}
//inline Void SSEStack::DivRD() {
//    _SSE::_IMov( _SSE_PD, _SSE_ST(-1), _SSE_ST(0) );
//    _SSE::_IDiv( _SSE_PD, _SSE_ST(0), _SSE_ST(1) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(1), _SSE_ST(0) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(0), _SSE_ST(-1) );
//}
//
//inline Void SSEStack::SubRF( UInt iDestIndex ) {
//    _SSE::_IMov( _SSE_PS, _SSE_ST(-1), _SSE_ST(0) );
//    _SSE::_ISub( _SSE_PS, _SSE_ST(0), _SSE_ST(iDestIndex) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(0), _SSE_ST(-1) );
//}
//inline Void SSEStack::SubRD( UInt iDestIndex ) {
//    _SSE::_IMov( _SSE_PD, _SSE_ST(-1), _SSE_ST(0) );
//    _SSE::_ISub( _SSE_PD, _SSE_ST(0), _SSE_ST(iDestIndex) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(0), _SSE_ST(-1) );
//}
//inline Void SSEStack::DivRF( UInt iDestIndex ) {
//    _SSE::_IMov( _SSE_PS, _SSE_ST(-1), _SSE_ST(0) );
//    _SSE::_IDiv( _SSE_PS, _SSE_ST(0), _SSE_ST(iDestIndex) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(0), _SSE_ST(-1) );
//}
//inline Void SSEStack::DivRD( UInt iDestIndex ) {
//    _SSE::_IMov( _SSE_PD, _SSE_ST(-1), _SSE_ST(0) );
//    _SSE::_IDiv( _SSE_PD, _SSE_ST(0), _SSE_ST(iDestIndex) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(0), _SSE_ST(-1) );
//}
//
//inline Void SSEStack::SubRF( UInt iDestIndex, UInt iSrcIndex ) {
//    _SSE::_IMov( _SSE_PS, _SSE_ST(-1), _SSE_ST(iSrcIndex) );
//    _SSE::_ISub( _SSE_PS, _SSE_ST(iSrcIndex), _SSE_ST(iDestIndex) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(iSrcIndex), _SSE_ST(-1) );
//}
//inline Void SSEStack::SubRD( UInt iDestIndex, UInt iSrcIndex ) {
//    _SSE::_IMov( _SSE_PD, _SSE_ST(-1), _SSE_ST(iSrcIndex) );
//    _SSE::_ISub( _SSE_PD, _SSE_ST(iSrcIndex), _SSE_ST(iDestIndex) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(iSrcIndex), _SSE_ST(-1) );
//}
//inline Void SSEStack::DivRF( UInt iDestIndex, UInt iSrcIndex ) {
//    _SSE::_IMov( _SSE_PS, _SSE_ST(-1), _SSE_ST(iSrcIndex) );
//    _SSE::_IDiv( _SSE_PS, _SSE_ST(iSrcIndex), _SSE_ST(iDestIndex) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(iSrcIndex), _SSE_ST(-1) );
//}
//inline Void SSEStack::DivRD( UInt iDestIndex, UInt iSrcIndex ) {
//    _SSE::_IMov( _SSE_PD, _SSE_ST(-1), _SSE_ST(iSrcIndex) );
//    _SSE::_IDiv( _SSE_PD, _SSE_ST(iSrcIndex), _SSE_ST(iDestIndex) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(iSrcIndex), _SSE_ST(-1) );
//}
//
//inline Void SSEStack::AddPF() { _SSE::_IAdd( _SSE_PS, _SSE_ST(1), _SSE_ST(0) ); --m_iTop; }
//inline Void SSEStack::AddPD() { _SSE::_IAdd( _SSE_PD, _SSE_ST(1), _SSE_ST(0) ); --m_iTop; }
//inline Void SSEStack::SubPF() { _SSE::_ISub( _SSE_PS, _SSE_ST(1), _SSE_ST(0) ); --m_iTop; }
//inline Void SSEStack::SubPD() { _SSE::_ISub( _SSE_PD, _SSE_ST(1), _SSE_ST(0) ); --m_iTop; }
//inline Void SSEStack::MulPF() { _SSE::_IMul( _SSE_PS, _SSE_ST(1), _SSE_ST(0) ); --m_iTop; }
//inline Void SSEStack::MulPD() { _SSE::_IMul( _SSE_PD, _SSE_ST(1), _SSE_ST(0) ); --m_iTop; }
//inline Void SSEStack::DivPF() { _SSE::_IDiv( _SSE_PS, _SSE_ST(1), _SSE_ST(0) ); --m_iTop; }
//inline Void SSEStack::DivPD() { _SSE::_IDiv( _SSE_PD, _SSE_ST(1), _SSE_ST(0) ); --m_iTop; }
//
//inline Void SSEStack::AddPF( UInt iDestIndex ) { _SSE::_IAdd( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) ); --m_iTop; }
//inline Void SSEStack::AddPD( UInt iDestIndex ) { _SSE::_IAdd( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) ); --m_iTop; }
//inline Void SSEStack::SubPF( UInt iDestIndex ) { _SSE::_ISub( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) ); --m_iTop; }
//inline Void SSEStack::SubPD( UInt iDestIndex ) { _SSE::_ISub( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) ); --m_iTop; }
//inline Void SSEStack::MulPF( UInt iDestIndex ) { _SSE::_IMul( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) ); --m_iTop; }
//inline Void SSEStack::MulPD( UInt iDestIndex ) { _SSE::_IMul( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) ); --m_iTop; }
//inline Void SSEStack::DivPF( UInt iDestIndex ) { _SSE::_IDiv( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) ); --m_iTop; }
//inline Void SSEStack::DivPD( UInt iDestIndex ) { _SSE::_IDiv( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) ); --m_iTop; }
//
//inline Void SSEStack::SubRPF() {
//    _SSE::_ISub( _SSE_PS, _SSE_ST(0), _SSE_ST(1) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(1), _SSE_ST(0) );
//    --m_iTop;
//}
//inline Void SSEStack::SubRPD() {
//    _SSE::_ISub( _SSE_PD, _SSE_ST(0), _SSE_ST(1) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(1), _SSE_ST(0) );
//    --m_iTop;
//}
//inline Void SSEStack::DivRPF() {
//    _SSE::_IDiv( _SSE_PS, _SSE_ST(0), _SSE_ST(1) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(1), _SSE_ST(0) );
//    --m_iTop;
//}
//inline Void SSEStack::DivRPD() {
//    _SSE::_IDiv( _SSE_PD, _SSE_ST(0), _SSE_ST(1) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(1), _SSE_ST(0) );
//    --m_iTop;
//}
//
//inline Void SSEStack::SubRPF( UInt iDestIndex ) {
//    _SSE::_ISub( _SSE_PS, _SSE_ST(0), _SSE_ST(iDestIndex) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) );
//    --m_iTop;
//}
//inline Void SSEStack::SubRPD( UInt iDestIndex ) {
//    _SSE::_ISub( _SSE_PD, _SSE_ST(0), _SSE_ST(iDestIndex) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) );
//    --m_iTop;
//}
//inline Void SSEStack::DivRPF( UInt iDestIndex ) {
//    _SSE::_IDiv( _SSE_PS, _SSE_ST(0), _SSE_ST(iDestIndex) );
//    _SSE::_IMov( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) );
//    --m_iTop;
//}
//inline Void SSEStack::DivRPD( UInt iDestIndex ) {
//    _SSE::_IDiv( _SSE_PD, _SSE_ST(0), _SSE_ST(iDestIndex) );
//    _SSE::_IMov( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) );
//    --m_iTop;
//}
//
//inline Void SSEStack::HAddF() { _SSE::_IHAdd( _SSE_PS, _SSE_ST(1), _SSE_ST(0) ); }
//inline Void SSEStack::HAddD() { _SSE::_IHAdd( _SSE_PD, _SSE_ST(1), _SSE_ST(0) ); }
//inline Void SSEStack::HSubF() { _SSE::_IHSub( _SSE_PS, _SSE_ST(1), _SSE_ST(0) ); }
//inline Void SSEStack::HSubD() { _SSE::_IHSub( _SSE_PD, _SSE_ST(1), _SSE_ST(0) ); }
//
//inline Void SSEStack::HAddF( UInt iDestIndex ) { _SSE::_IHAdd( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) ); }
//inline Void SSEStack::HAddD( UInt iDestIndex ) { _SSE::_IHAdd( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) ); }
//inline Void SSEStack::HSubF( UInt iDestIndex ) { _SSE::_IHSub( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(0) ); }
//inline Void SSEStack::HSubD( UInt iDestIndex ) { _SSE::_IHSub( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(0) ); }
//
//inline Void SSEStack::HAddF( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_IHAdd( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//inline Void SSEStack::HAddD( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_IHAdd( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//inline Void SSEStack::HSubF( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_IHSub( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//inline Void SSEStack::HSubD( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_IHSub( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//
//inline Void SSEStack::SqrtF() { _SSE::_ISqrt( _SSE_PS, _SSE_ST(0), _SSE_ST(0) ); }
//inline Void SSEStack::SqrtD() { _SSE::_ISqrt( _SSE_PD, _SSE_ST(0), _SSE_ST(0) ); }
//inline Void SSEStack::InvF()  { _SSE::_IInv( _SSE_PS, _SSE_ST(0), _SSE_ST(0) ); }
//inline Void SSEStack::InvSqrtF() { _SSE::_IInvSqrt( _SSE_PS, _SSE_ST(0), _SSE_ST(0) ); }
//
//inline Void SSEStack::SqrtF( UInt iStackIndex ) { _SSE::_ISqrt( _SSE_PS, _SSE_ST(iStackIndex), _SSE_ST(iStackIndex) ); }
//inline Void SSEStack::SqrtD( UInt iStackIndex ) { _SSE::_ISqrt( _SSE_PD, _SSE_ST(iStackIndex), _SSE_ST(iStackIndex) ); }
//inline Void SSEStack::InvF( UInt iStackIndex )  { _SSE::_IInv( _SSE_PS, _SSE_ST(iStackIndex), _SSE_ST(iStackIndex) ); }
//inline Void SSEStack::InvSqrtF( UInt iStackIndex ) { _SSE::_IInvSqrt( _SSE_PS, _SSE_ST(iStackIndex), _SSE_ST(iStackIndex) ); }
//
//inline Void SSEStack::SqrtF( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_ISqrt( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//inline Void SSEStack::SqrtD( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_ISqrt( _SSE_PD, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//inline Void SSEStack::InvF( UInt iDestIndex, UInt iSrcIndex )  { _SSE::_IInv( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//inline Void SSEStack::InvSqrtF( UInt iDestIndex, UInt iSrcIndex ) { _SSE::_IInvSqrt( _SSE_PS, _SSE_ST(iDestIndex), _SSE_ST(iSrcIndex) ); }
//
///////////////////////////////////////////////////////////////////////////////////
//
//inline Void SSEStack::_IPush( UInt iVariant, QWord inMem128[2] ) {
//    _SSE::_ISet( iVariant, m_iTop, inMem128 );
//    ++m_iTop;
//}
//inline Void SSEStack::_IPop( UInt iVariant, QWord outMem128[2] ) {
//    --m_iTop;
//    _SSE::_IGet( iVariant, outMem128, m_iTop );
//}
//
