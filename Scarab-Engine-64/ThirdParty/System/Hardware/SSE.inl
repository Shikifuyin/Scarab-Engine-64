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

////////////////////////////////////////////////////////////// Control instructions

inline UInt32 SIMDMath::GetCSR() const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_getcsr();
}
inline Void SIMDMath::SetCSR( UInt32 iValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_setcsr( iValue );
}

inline Void SIMDMath::ClearAndFlushCacheLine( Void * pAddress ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_clflush( pAddress );
}

inline Void SIMDMath::Pause() const {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_pause();
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

inline Void SIMDMath::ZeroUpper128() const {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_zeroupper();
}
inline Void SIMDMath::Zero256() const {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_zeroall();
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

inline __m128i SIMDMath::SetLower( Int32 i0 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi32_si128( i0 );
}
inline __m128i SIMDMath::SetLower( Int64 i0 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi64_si128( i0 );
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
inline Float SIMDMath::GetLowerFloat( __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtss_f32( mSrc );
}
inline Float SIMDMath::GetLowerFloat( __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtss_f32( mSrc );
}

inline Double SIMDMath::GetLowerDouble( __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsd_f64( mSrc );
}
inline Double SIMDMath::GetLowerDouble( __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtsd_f64( mSrc );
}

inline Int32 SIMDMath::GetLowerInt32( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi128_si32( mSrc );
}
inline Int32 SIMDMath::GetLowerInt32( __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtsi256_si32( mSrc );
}

inline Int64 SIMDMath::GetLowerInt64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi128_si64( mSrc );
}
inline Int64 SIMDMath::GetLowerInt64( __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtsi256_si64( mSrc );
}

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
inline __m128 SIMDMath::Load128( const Float * arrF, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_maskload_ps( arrF, mSigns );
}
inline __m256 SIMDMath::Load256( const Float * arrF ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_loadu_ps( arrF );
}
inline __m256 SIMDMath::Load256( const Float * arrF, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_maskload_ps( arrF, mSigns );
}

inline __m128d SIMDMath::Load128( const Double * arrF ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_loadu_pd( arrF );
}
inline __m128d SIMDMath::Load128( const Double * arrF, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_maskload_pd( arrF, mSigns );
}
inline __m256d SIMDMath::Load256( const Double * arrF ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_loadu_pd( arrF );
}
inline __m256d SIMDMath::Load256( const Double * arrF, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_maskload_pd( arrF, mSigns );
}

inline __m128i SIMDMath::Load128( const Int32 * arrI, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_maskload_epi32( arrI, mSigns );
}
inline __m256i SIMDMath::Load256( const Int32 * arrI, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_maskload_epi32( arrI, mSigns );
}

inline __m128i SIMDMath::Load128( const Int64 * arrI, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_maskload_epi64( arrI, mSigns );
}
inline __m256i SIMDMath::Load256( const Int64 * arrI, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_maskload_epi64( arrI, mSigns );
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
inline __m128 SIMDMath::Load32FourFloat( __m128 mDst, const Float * pSrc, __m128i mIndices, Int32 iStride, __m128 mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_ps( mDst, pSrc, mIndices, mSigns, iStride );
}
inline __m256 SIMDMath::Load32EightFloat( const Float * pSrc, __m256i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_ps( pSrc, mIndices, iStride );
}
inline __m256 SIMDMath::Load32EightFloat( __m256 mDst, const Float * pSrc, __m256i mIndices, Int32 iStride, __m256 mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_ps( mDst, pSrc, mIndices, mSigns, iStride );
}

inline __m128d SIMDMath::Load32TwoDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_pd( pSrc, mIndices, iStride );
}
inline __m128d SIMDMath::Load32TwoDouble( __m128d mDst, const Double * pSrc, __m128i mIndices, Int32 iStride, __m128d mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_pd( mDst, pSrc, mIndices, mSigns, iStride );
}
inline __m256d SIMDMath::Load32FourDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_pd( pSrc, mIndices, iStride );
}
inline __m256d SIMDMath::Load32FourDouble( __m256d mDst, const Double * pSrc, __m128i mIndices, Int32 iStride, __m256d mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_pd( mDst, pSrc, mIndices, mSigns, iStride );
}

inline __m128i SIMDMath::Load32FourInt32( const Int32 * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_epi32( pSrc, mIndices, iStride );
}
inline __m128i SIMDMath::Load32FourInt32( __m128i mDst, const Int32 * pSrc, __m128i mIndices, Int32 iStride, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_epi32( mDst, pSrc, mIndices, mSigns, iStride );
}
inline __m256i SIMDMath::Load32EightInt32( const Int32 * pSrc, __m256i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_epi32( pSrc, mIndices, iStride );
}
inline __m256i SIMDMath::Load32EightInt32( __m256i mDst, const Int32 * pSrc, __m256i mIndices, Int32 iStride, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_epi32( mDst, pSrc, mIndices, mSigns, iStride );
}

inline __m128i SIMDMath::Load32TwoInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i32gather_epi64( pSrc, mIndices, iStride );
}
inline __m128i SIMDMath::Load32TwoInt64( __m128i mDst, const Int64 * pSrc, __m128i mIndices, Int32 iStride, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i32gather_epi64( mDst, pSrc, mIndices, mSigns, iStride );
}
inline __m256i SIMDMath::Load32FourInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i32gather_epi64( pSrc, mIndices, iStride );
}
inline __m256i SIMDMath::Load32FourInt64( __m256i mDst, const Int64 * pSrc, __m128i mIndices, Int32 iStride, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i32gather_epi64( mDst, pSrc, mIndices, mSigns, iStride );
}

inline __m128 SIMDMath::Load64TwoFloat( const Float * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_ps( pSrc, mIndices, iStride );
}
inline __m128 SIMDMath::Load64TwoFloat( __m128 mDst, const Float * pSrc, __m128i mIndices, Int32 iStride, __m128 mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_ps( mDst, pSrc, mIndices, mSigns, iStride );
}
inline __m128 SIMDMath::Load64FourFloat( const Float * pSrc, __m256i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_ps( pSrc, mIndices, iStride );
}
inline __m128 SIMDMath::Load64FourFloat( __m128 mDst, const Float * pSrc, __m256i mIndices, Int32 iStride, __m128 mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_ps( mDst, pSrc, mIndices, mSigns, iStride );
}

inline __m128d SIMDMath::Load64TwoDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_pd( pSrc, mIndices, iStride );
}
inline __m128d SIMDMath::Load64TwoDouble( __m128d mDst, const Double * pSrc, __m128i mIndices, Int32 iStride, __m128d mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_pd( mDst, pSrc, mIndices, mSigns, iStride );
}
inline __m256d SIMDMath::Load64FourDouble( const Double * pSrc, __m256i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_pd( pSrc, mIndices, iStride );
}
inline __m256d SIMDMath::Load64FourDouble( __m256d mDst, const Double * pSrc, __m256i mIndices, Int32 iStride, __m256d mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_pd( mDst, pSrc, mIndices, mSigns, iStride );
}

inline __m128i SIMDMath::Load64TwoInt32( const Int32 * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_epi32( pSrc, mIndices, iStride );
}
inline __m128i SIMDMath::Load64TwoInt32( __m128i mDst, const Int32 * pSrc, __m128i mIndices, Int32 iStride, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_epi32( mDst, pSrc, mIndices, mSigns, iStride );
}
inline __m128i SIMDMath::Load64FourInt32( const Int32 * pSrc, __m256i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_epi32( pSrc, mIndices, iStride );
}
inline __m128i SIMDMath::Load64FourInt32( __m128i mDst, const Int32 * pSrc, __m256i mIndices, Int32 iStride, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_epi32( mDst, pSrc, mIndices, mSigns, iStride );
}

inline __m128i SIMDMath::Load64TwoInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_i64gather_epi64( pSrc, mIndices, iStride );
}
inline __m128i SIMDMath::Load64TwoInt64( __m128i mDst, const Int64 * pSrc, __m128i mIndices, Int32 iStride, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_mask_i64gather_epi64( mDst, pSrc, mIndices, mSigns, iStride );
}
inline __m256i SIMDMath::Load64FourInt64( const Int64 * pSrc, __m256i mIndices, Int32 iStride ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_i64gather_epi64( pSrc, mIndices, iStride );
}
inline __m256i SIMDMath::Load64FourInt64( __m256i mDst, const Int64 * pSrc, __m256i mIndices, Int32 iStride, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mask_i64gather_epi64( mDst, pSrc, mIndices, mSigns, iStride );
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

inline Void SIMDMath::Store128( Float * outDst, __m128 mSrc, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm_maskstore_ps( outDst, mSigns, mSrc );
}
inline Void SIMDMath::Store256( Float * outDst, __m256 mSrc, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_maskstore_ps( outDst, mSigns, mSrc );
}

inline Void SIMDMath::Store128( Double * outDst, __m128d mSrc, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm_maskstore_pd( outDst, mSigns, mSrc );
}
inline Void SIMDMath::Store256( Double * outDst, __m256d mSrc, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_maskstore_pd( outDst, mSigns, mSrc );
}

inline Void SIMDMath::Store128( Int32 * outDst, __m128i mSrc, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    _mm_maskstore_epi32( outDst, mSigns, mSrc );
}
inline Void SIMDMath::Store256( Int32 * outDst, __m256i mSrc, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    _mm256_maskstore_epi32( outDst, mSigns, mSrc );
}

inline Void SIMDMath::Store128( Int64 * outDst, __m128i mSrc, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    _mm_maskstore_epi64( outDst, mSigns, mSrc );
}
inline Void SIMDMath::Store256( Int64 * outDst, __m256i mSrc, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    _mm256_maskstore_epi64( outDst, mSigns, mSrc );
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

////////////////////////////////////////////////////////////// Blend
inline __m128 SIMDMath::BlendFloat( __m128 mDst, __m128 mSrc, Int iMask4 ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_blend_ps( mDst, mSrc, iMask4 );
}
inline __m128 SIMDMath::BlendFloat( __m128 mDst, __m128 mSrc, __m128 mSigns ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_blendv_ps( mDst, mSrc, mSigns );
}
inline __m256 SIMDMath::BlendFloat( __m256 mDst, __m256 mSrc, Int iMask8 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_blend_ps( mDst, mSrc, iMask8 );
}
inline __m256 SIMDMath::BlendFloat( __m256 mDst, __m256 mSrc, __m256 mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_blendv_ps( mDst, mSrc, mSigns );
}

inline __m128d SIMDMath::BlendDouble( __m128d mDst, __m128d mSrc, Int iMask2 ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_blend_pd( mDst, mSrc, iMask2 );
}
inline __m128d SIMDMath::BlendDouble( __m128d mDst, __m128d mSrc, __m128d mSigns ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_blendv_pd( mDst, mSrc, mSigns );
}
inline __m256d SIMDMath::BlendDouble( __m256d mDst, __m256d mSrc, Int iMask4 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_blend_pd( mDst, mSrc, iMask4 );
}
inline __m256d SIMDMath::BlendDouble( __m256d mDst, __m256d mSrc, __m256d mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_blendv_pd( mDst, mSrc, mSigns );
}

inline __m128i SIMDMath::BlendInt8( __m128i mDst, __m128i mSrc, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_blendv_epi8( mDst, mSrc, mSigns );
}
inline __m256i SIMDMath::BlendInt8( __m256i mDst, __m256i mSrc, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_blendv_epi8( mDst, mSrc, mSigns );
}

inline __m128i SIMDMath::BlendInt16( __m128i mDst, __m128i mSrc, Int iMask8 ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_blend_epi16( mDst, mSrc, iMask8 );
}
inline __m256i SIMDMath::BlendInt16( __m256i mDst, __m256i mSrc, Int iMask8 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_blend_epi16( mDst, mSrc, iMask8 );
}

inline __m128i SIMDMath::BlendInt32( __m128i mDst, __m128i mSrc, Int iMask4 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_blend_epi32( mDst, mSrc, iMask4 );
}
inline __m256i SIMDMath::BlendInt32( __m256i mDst, __m256i mSrc, Int iMask8 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_blend_epi32( mDst, mSrc, iMask8 );
}

////////////////////////////////////////////////////////////// Cast (Free, 0 instruction generated)
inline __m128 SIMDMath::CastToFloat( __m128d mDouble ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castpd_ps( mDouble );
}
inline __m128 SIMDMath::CastToFloat( __m128i mInteger ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castsi128_ps( mInteger );
}
inline __m256 SIMDMath::CastToFloat( __m256d mDouble ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castpd_ps( mDouble );
}
inline __m256 SIMDMath::CastToFloat( __m256i mInteger ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castsi256_ps( mInteger );
}

inline __m128d SIMDMath::CastToDouble( __m128 mFloat ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castps_pd( mFloat );
}
inline __m128d SIMDMath::CastToDouble( __m128i mInteger ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castsi128_pd( mInteger );
}
inline __m256d SIMDMath::CastToDouble( __m256 mFloat ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castps_pd( mFloat );
}
inline __m256d SIMDMath::CastToDouble( __m256i mInteger ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castsi256_pd( mInteger );
}

inline __m128i SIMDMath::CastToInteger( __m128 mFloat ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castps_si128( mFloat );
}
inline __m128i SIMDMath::CastToInteger( __m128d mDouble ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_castpd_si128( mDouble );
}
inline __m256i SIMDMath::CastToInteger( __m256 mFloat ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castps_si256( mFloat );
}
inline __m256i SIMDMath::CastToInteger( __m256d mDouble ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castpd_si256( mDouble );
}

inline __m128 SIMDMath::CastDown( __m256 mFloat ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castps256_ps128( mFloat );
}
inline __m128d SIMDMath::CastDown( __m256d mDouble ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castpd256_pd128( mDouble );
}
inline __m128i SIMDMath::CastDown( __m256i mInteger ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castsi256_si128( mInteger );
}

inline __m256 SIMDMath::CastUp( __m128 mFloat ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castps128_ps256( mFloat );
}
inline __m256d SIMDMath::CastUp( __m128d mDouble ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castpd128_pd256( mDouble );
}
inline __m256i SIMDMath::CastUp( __m128i mInteger ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_castsi128_si256( mInteger );
}

////////////////////////////////////////////////////////////// Convert & Copy
inline __m128 SIMDMath::ConvertLower( __m128 mDst, Int32 iSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtsi32_ss( mDst, iSrc );
}
inline __m128 SIMDMath::ConvertLower( __m128 mDst, Int64 iSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtsi64_ss( mDst, iSrc );
}

inline __m128d SIMDMath::ConvertLower( __m128d mDst, Int32 iSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi32_sd( mDst, iSrc );
}
inline __m128d SIMDMath::ConvertLower( __m128d mDst, Int64 iSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi64_sd( mDst, iSrc );
}

inline __m128 SIMDMath::ConvertLower( __m128 mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsd_ss( mDst, mSrc );
}
inline __m128d SIMDMath::ConvertLower( __m128d mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtss_sd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Convert
inline Int32 SIMDMath::ConvertLowerToInt32( __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtss_si32( mSrc );
}
inline Int32 SIMDMath::ConvertLowerToInt32( __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsd_si32( mSrc );
}

inline Int64 SIMDMath::ConvertLowerToInt64( __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvtss_si64( mSrc );
}
inline Int64 SIMDMath::ConvertLowerToInt64( __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsd_si64( mSrc );
}

inline __m128 SIMDMath::Convert128ToFloat( __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtpd_ps( mSrc );
}
inline __m128 SIMDMath::Convert128ToFloat( __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtpd_ps( mSrc );
}

inline __m128 SIMDMath::Convert128ToFloat( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtepi32_ps( mSrc );
}
inline __m256 SIMDMath::Convert256ToFloat( __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtepi32_ps( mSrc );
}

inline __m128d SIMDMath::Convert128ToDouble( __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtps_pd( mSrc );
}
inline __m256d SIMDMath::Convert256ToDouble( __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtps_pd( mSrc );
}

inline __m128d SIMDMath::Convert128ToDouble( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtepi32_pd( mSrc );
}
inline __m256d SIMDMath::Convert256ToDouble( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtepi32_pd( mSrc );
}

inline __m128i SIMDMath::Convert128ToInt32( __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtps_epi32( mSrc );
}
inline __m256i SIMDMath::Convert256ToInt32( __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtps_epi32( mSrc );
}

inline __m128i SIMDMath::Convert128ToInt32( __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtpd_epi32( mSrc );
}
inline __m128i SIMDMath::Convert128ToInt32( __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvtpd_epi32( mSrc );
}

////////////////////////////////////////////////////////////// Truncate
inline Int32 SIMDMath::TruncateLowerToInt32( __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvttss_si32( mSrc );
}
inline Int32 SIMDMath::TruncateLowerToInt32( __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvttsd_si32( mSrc );
}

inline Int64 SIMDMath::TruncateLowerToInt64( __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cvttss_si64( mSrc );
}
inline Int64 SIMDMath::TruncateLowerToInt64( __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvttsd_si64( mSrc );
}

inline __m128i SIMDMath::TruncateToInt32( __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvttps_epi32( mSrc );
}
inline __m256i SIMDMath::TruncateToInt32( __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvttps_epi32( mSrc );
}

inline __m128i SIMDMath::TruncateToInt32( __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvttpd_epi32( mSrc );
}
inline __m128i SIMDMath::TruncateToInt32( __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cvttpd_epi32( mSrc );
}

////////////////////////////////////////////////////////////// Sign-Extend
inline __m128i SIMDMath::SignExtend128Int8To16( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi8_epi16( mSrc );
}
inline __m128i SIMDMath::SignExtend128Int8To32( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi8_epi32( mSrc );
}
inline __m128i SIMDMath::SignExtend128Int8To64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi8_epi64( mSrc );
}
inline __m128i SIMDMath::SignExtend128Int16To32( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi16_epi32( mSrc );
}
inline __m128i SIMDMath::SignExtend128Int16To64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi16_epi64( mSrc );
}
inline __m128i SIMDMath::SignExtend128Int32To64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepi32_epi64( mSrc );
}

inline __m256i SIMDMath::SignExtend256Int8To16( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi8_epi16( mSrc );
}
inline __m256i SIMDMath::SignExtend256Int8To32( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi8_epi32( mSrc );
}
inline __m256i SIMDMath::SignExtend256Int8To64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi8_epi64( mSrc );
}
inline __m256i SIMDMath::SignExtend256Int16To32( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi16_epi32( mSrc );
}
inline __m256i SIMDMath::SignExtend256Int16To64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi16_epi64( mSrc );
}
inline __m256i SIMDMath::SignExtend256Int32To64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepi32_epi64( mSrc );
}

////////////////////////////////////////////////////////////// Zero-Extend
inline __m128i SIMDMath::ZeroExtend128Int8To16( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu8_epi16( mSrc );
}
inline __m128i SIMDMath::ZeroExtend128Int8To32( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu8_epi32( mSrc );
}
inline __m128i SIMDMath::ZeroExtend128Int8To64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu8_epi64( mSrc );
}
inline __m128i SIMDMath::ZeroExtend128Int16To32( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu16_epi32( mSrc );
}
inline __m128i SIMDMath::ZeroExtend128Int16To64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu16_epi64( mSrc );
}
inline __m128i SIMDMath::ZeroExtend128Int32To64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cvtepu32_epi64( mSrc );
}

inline __m256i SIMDMath::ZeroExtend256Int8To16( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu8_epi16( mSrc );
}
inline __m256i SIMDMath::ZeroExtend256Int8To32( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu8_epi32( mSrc );
}
inline __m256i SIMDMath::ZeroExtend256Int8To64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu8_epi64( mSrc );
}
inline __m256i SIMDMath::ZeroExtend256Int16To32( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu16_epi32( mSrc );
}
inline __m256i SIMDMath::ZeroExtend256Int16To64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu16_epi64( mSrc );
}
inline __m256i SIMDMath::ZeroExtend256Int32To64( __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cvtepu32_epi64( mSrc );
}

////////////////////////////////////////////////////////////// Absolute Value
inline __m128i SIMDMath::Abs8( __m128i mValue ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_abs_epi8( mValue );
}
inline __m256i SIMDMath::Abs8( __m256i mValue ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_abs_epi8( mValue );
}

inline __m128i SIMDMath::Abs16( __m128i mValue ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_abs_epi16( mValue );
}
inline __m256i SIMDMath::Abs16( __m256i mValue ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_abs_epi16( mValue );
}

inline __m128i SIMDMath::Abs32( __m128i mValue ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_abs_epi32( mValue );
}
inline __m256i SIMDMath::Abs32( __m256i mValue ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_abs_epi32( mValue );
}

inline __m128i SIMDMath::Abs64( __m128i mValue ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_abs_epi64( mValue );
}
inline __m256i SIMDMath::Abs64( __m256i mValue ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_abs_epi64( mValue );
}

////////////////////////////////////////////////////////////// Sign Change
inline __m128i SIMDMath::Negate8( __m128i mValue, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_sign_epi8( mValue, mSigns );
}
inline __m256i SIMDMath::Negate8( __m256i mValue, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sign_epi8( mValue, mSigns );
}

inline __m128i SIMDMath::Negate16( __m128i mValue, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_sign_epi16( mValue, mSigns );
}
inline __m256i SIMDMath::Negate16( __m256i mValue, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sign_epi16( mValue, mSigns );
}

inline __m128i SIMDMath::Negate32( __m128i mValue, __m128i mSigns ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_sign_epi32( mValue, mSigns );
}
inline __m256i SIMDMath::Negate32( __m256i mValue, __m256i mSigns ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sign_epi32( mValue, mSigns );
}

////////////////////////////////////////////////////////////// Rounding
inline __m128 SIMDMath::FloorLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_floor_ss( mDst, mSrc );
}
inline __m128d SIMDMath::FloorLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_floor_sd( mDst, mSrc );
}

inline __m128 SIMDMath::Floor( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_floor_ps( mValue );
}
inline __m256 SIMDMath::Floor( __m256 mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_floor_ps( mValue );
}

inline __m128d SIMDMath::Floor( __m128d mValue ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_floor_pd( mValue );
}
inline __m256d SIMDMath::Floor( __m256d mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_floor_pd( mValue );
}

inline __m128 SIMDMath::CeilLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_ceil_ss( mDst, mSrc );
}
inline __m128d SIMDMath::CeilLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_ceil_sd( mDst, mSrc );
}

inline __m128 SIMDMath::Ceil( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_ceil_ps( mValue );
}
inline __m256 SIMDMath::Ceil( __m256 mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_ceil_ps( mValue );
}

inline __m128d SIMDMath::Ceil( __m128d mValue ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_ceil_pd( mValue );
}
inline __m256d SIMDMath::Ceil( __m256d mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_ceil_pd( mValue );
}

inline __m128 SIMDMath::RoundLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_round_ss( mDst, mSrc, _MM_FROUND_NINT );
}
inline __m128d SIMDMath::RoundLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_round_sd( mDst, mSrc, _MM_FROUND_NINT );
}

inline __m128 SIMDMath::Round( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_round_ps( mValue, _MM_FROUND_NINT );
}
inline __m256 SIMDMath::Round( __m256 mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_round_ps( mValue, _MM_FROUND_NINT );
}

inline __m128d SIMDMath::Round( __m128d mValue ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_round_pd( mValue, _MM_FROUND_NINT );
}
inline __m256d SIMDMath::Round( __m256d mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_round_pd( mValue, _MM_FROUND_NINT );
}

////////////////////////////////////////////////////////////// Addition
inline __m128 SIMDMath::AddLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_add_ss( mDst, mSrc );
}
inline __m128d SIMDMath::AddLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_sd( mDst, mSrc );
}

inline __m128 SIMDMath::Add( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_add_ps( mDst, mSrc );
}
inline __m256 SIMDMath::Add( __m256 mDst, __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_add_ps( mDst, mSrc );
}

inline __m128d SIMDMath::Add( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_pd( mDst, mSrc );
}
inline __m256d SIMDMath::Add( __m256d mDst, __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_add_pd( mDst, mSrc );
}

inline __m128i SIMDMath::Add8( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_epi8( mDst, mSrc );
}
inline __m256i SIMDMath::Add8( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_add_epi8( mDst, mSrc );
}

inline __m128i SIMDMath::Add16( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_epi16( mDst, mSrc );
}
inline __m256i SIMDMath::Add16( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_add_epi16( mDst, mSrc );
}

inline __m128i SIMDMath::Add32( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_epi32( mDst, mSrc );
}
inline __m256i SIMDMath::Add32( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_add_epi32( mDst, mSrc );
}

inline __m128i SIMDMath::Add64( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_add_epi64( mDst, mSrc );
}
inline __m256i SIMDMath::Add64( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_add_epi64( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Addition with Saturation
inline __m128i SIMDMath::AddSigned8( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_adds_epi8( mDst, mSrc );
}
inline __m256i SIMDMath::AddSigned8( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_adds_epi8( mDst, mSrc );
}

inline __m128i SIMDMath::AddSigned16( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_adds_epi16( mDst, mSrc );
}
inline __m256i SIMDMath::AddSigned16( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_adds_epi16( mDst, mSrc );
}

inline __m128i SIMDMath::AddUnsigned8( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_adds_epu8( mDst, mSrc );
}
inline __m256i SIMDMath::AddUnsigned8( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_adds_epu8( mDst, mSrc );
}

inline __m128i SIMDMath::AddUnsigned16( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_adds_epu16( mDst, mSrc );
}
inline __m256i SIMDMath::AddUnsigned16( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_adds_epu16( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Horizontal Addition
inline __m128 SIMDMath::HAdd( __m128 mSrc1, __m128 mSrc2 ) const {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_hadd_ps( mSrc1, mSrc2 );
}
inline __m256 SIMDMath::HAdd( __m256 mSrc1, __m256 mSrc2 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hadd_ps( mSrc1, mSrc2 );
}

inline __m128d SIMDMath::HAdd( __m128d mSrc1, __m128d mSrc2 ) const {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_hadd_pd( mSrc1, mSrc2 );
}
inline __m256d SIMDMath::HAdd( __m256d mSrc1, __m256d mSrc2 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hadd_pd( mSrc1, mSrc2 );
}

inline __m128i SIMDMath::HAdd16( __m128i mSrc1, __m128i mSrc2 ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hadd_epi16( mSrc1, mSrc2 );
}
inline __m256i SIMDMath::HAdd16( __m256i mSrc1, __m256i mSrc2 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hadd_epi16( mSrc1, mSrc2 );
}

inline __m128i SIMDMath::HAdd32( __m128i mSrc1, __m128i mSrc2 ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hadd_epi32( mSrc1, mSrc2 );
}
inline __m256i SIMDMath::HAdd32( __m256i mSrc1, __m256i mSrc2 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hadd_epi32( mSrc1, mSrc2 );
}

////////////////////////////////////////////////////////////// Horizontal Addition with Saturation
inline __m128i SIMDMath::HAddSigned16( __m128i mSrc1, __m128i mSrc2 ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hadds_epi16( mSrc1, mSrc2 );
}
inline __m256i SIMDMath::HAddSigned16( __m256i mSrc1, __m256i mSrc2 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hadds_epi16( mSrc1, mSrc2 );
}

////////////////////////////////////////////////////////////// Substraction
inline __m128 SIMDMath::SubLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sub_ss( mDst, mSrc );
}
inline __m128d SIMDMath::SubLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_sd( mDst, mSrc );
}

inline __m128 SIMDMath::Sub( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sub_ps( mDst, mSrc );
}
inline __m256 SIMDMath::Sub( __m256 mDst, __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sub_ps( mDst, mSrc );
}

inline __m128d SIMDMath::Sub( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_pd( mDst, mSrc );
}
inline __m256d SIMDMath::Sub( __m256d mDst, __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sub_pd( mDst, mSrc );
}

inline __m128i SIMDMath::Sub8( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_epi8( mDst, mSrc );
}
inline __m256i SIMDMath::Sub8( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sub_epi8( mDst, mSrc );
}

inline __m128i SIMDMath::Sub16( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_epi16( mDst, mSrc );
}
inline __m256i SIMDMath::Sub16( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sub_epi16( mDst, mSrc );
}

inline __m128i SIMDMath::Sub32( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_epi32( mDst, mSrc );
}
inline __m256i SIMDMath::Sub32( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sub_epi32( mDst, mSrc );
}

inline __m128i SIMDMath::Sub64( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sub_epi64( mDst, mSrc );
}
inline __m256i SIMDMath::Sub64( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sub_epi64( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Substraction with Saturation
inline __m128i SIMDMath::SubSigned8( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_subs_epi8( mDst, mSrc );
}
inline __m256i SIMDMath::SubSigned8( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_subs_epi8( mDst, mSrc );
}

inline __m128i SIMDMath::SubSigned16( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_subs_epi16( mDst, mSrc );
}
inline __m256i SIMDMath::SubSigned16( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_subs_epi16( mDst, mSrc );
}

inline __m128i SIMDMath::SubUnsigned8( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_subs_epu8( mDst, mSrc );
}
inline __m256i SIMDMath::SubUnsigned8( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_subs_epu8( mDst, mSrc );
}

inline __m128i SIMDMath::SubUnsigned16( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_subs_epu16( mDst, mSrc );
}
inline __m256i SIMDMath::SubUnsigned16( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_subs_epu16( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Horizontal Substraction
inline __m128 SIMDMath::HSub( __m128 mSrc1, __m128 mSrc2 ) const {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_hsub_ps( mSrc1, mSrc2 );
}
inline __m256 SIMDMath::HSub( __m256 mSrc1, __m256 mSrc2 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hsub_ps( mSrc1, mSrc2 );
}

inline __m128d SIMDMath::HSub( __m128d mSrc1, __m128d mSrc2 ) const {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_hsub_pd( mSrc1, mSrc2 );
}
inline __m256d SIMDMath::HSub( __m256d mSrc1, __m256d mSrc2 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hsub_pd( mSrc1, mSrc2 );
}

inline __m128i SIMDMath::HSub16( __m128i mSrc1, __m128i mSrc2 ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hsub_epi16( mSrc1, mSrc2 );
}
inline __m256i SIMDMath::HSub16( __m256i mSrc1, __m256i mSrc2 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hsub_epi16( mSrc1, mSrc2 );
}

inline __m128i SIMDMath::HSub32( __m128i mSrc1, __m128i mSrc2 ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hsub_epi32( mSrc1, mSrc2 );
}
inline __m256i SIMDMath::HSub32( __m256i mSrc1, __m256i mSrc2 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hsub_epi32( mSrc1, mSrc2 );
}

////////////////////////////////////////////////////////////// Horizontal Substraction with Saturation
inline __m128i SIMDMath::HSubSigned16( __m128i mSrc1, __m128i mSrc2 ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_hsubs_epi16( mSrc1, mSrc2 );
}
inline __m256i SIMDMath::HSubSigned16( __m256i mSrc1, __m256i mSrc2 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_hsubs_epi16( mSrc1, mSrc2 );
}

////////////////////////////////////////////////////////////// Interleaved Add & Sub
inline __m128 SIMDMath::AddSub( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_addsub_ps( mDst, mSrc );
}
inline __m256 SIMDMath::AddSub( __m256 mDst, __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_addsub_ps( mDst, mSrc );
}

inline __m128d SIMDMath::AddSub( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE3() );
    return _mm_addsub_pd( mDst, mSrc );
}
inline __m256d SIMDMath::AddSub( __m256d mDst, __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_addsub_pd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// SAD (Sum Absolute Differences)
inline __m128i SIMDMath::SAD( __m128i mSrc1, __m128i mSrc2 ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sad_epu8( mSrc1, mSrc2 );
}
inline __m256i SIMDMath::SAD( __m256i mSrc1, __m256i mSrc2 ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sad_epu8( mSrc1, mSrc2 );
}

inline __m128i SIMDMath::SAD( __m128i mSrc1, __m128i mSrc2, Int iMask ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_mpsadbw_epu8( mSrc1, mSrc2, iMask );
}
inline __m256i SIMDMath::SAD( __m256i mSrc1, __m256i mSrc2, Int iMask ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mpsadbw_epu8( mSrc1, mSrc2, iMask );
}

////////////////////////////////////////////////////////////// Multiplication
inline __m128 SIMDMath::MulLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_mul_ss( mDst, mSrc );
}
inline __m128d SIMDMath::MulLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mul_sd( mDst, mSrc );
}

inline __m128 SIMDMath::Mul( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_mul_ps( mDst, mSrc );
}
inline __m256 SIMDMath::Mul( __m256 mDst, __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_mul_ps( mDst, mSrc );
}

inline __m128d SIMDMath::Mul( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mul_pd( mDst, mSrc );
}
inline __m256d SIMDMath::Mul( __m256d mDst, __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_mul_pd( mDst, mSrc );
}

inline __m128i SIMDMath::MulSigned16L( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mullo_epi16( mDst, mSrc );
}
inline __m256i SIMDMath::MulSigned16L( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mullo_epi16( mDst, mSrc );
}

inline __m128i SIMDMath::MulSigned16H( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mulhi_epi16( mDst, mSrc );
}
inline __m256i SIMDMath::MulSigned16H( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mulhi_epi16( mDst, mSrc );
}

inline __m128i SIMDMath::MulSigned32( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_mul_epi32( mDst, mSrc );
}
inline __m256i SIMDMath::MulSigned32( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mul_epi32( mDst, mSrc );
}

inline __m128i SIMDMath::MulSigned32L( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_mullo_epi32( mDst, mSrc );
}
inline __m256i SIMDMath::MulSigned32L( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mullo_epi32( mDst, mSrc );
}

inline __m128i SIMDMath::MulSigned64L( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_mullo_epi64( mDst, mSrc );
}
inline __m256i SIMDMath::MulSigned64L( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mullo_epi64( mDst, mSrc );
}

inline __m128i SIMDMath::MulUnsigned16H( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mulhi_epu16( mDst, mSrc );
}
inline __m256i SIMDMath::MulUnsigned16H( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mulhi_epu16( mDst, mSrc );
}

inline __m128i SIMDMath::MulUnsigned32( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_mul_epu32( mDst, mSrc );
}
inline __m256i SIMDMath::MulUnsigned32( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_mul_epu32( mDst, mSrc );
}

////////////////////////////////////////////////////////////// MADD (Multiply and Add)
inline __m128i SIMDMath::MAdd( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_madd_epi16( mDst, mSrc );
}
inline __m256i SIMDMath::MAdd( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_madd_epi16( mDst, mSrc );
}

inline __m128i SIMDMath::MAddUS( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSSE3() );
    return _mm_maddubs_epi16( mDst, mSrc );
}
inline __m256i SIMDMath::MAddUS( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_maddubs_epi16( mDst, mSrc );
}

////////////////////////////////////////////////////////////// DP (Dot Product)
inline __m128 SIMDMath::DotP( __m128 mDst, __m128 mSrc, Int iMask4 ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_dp_ps( mDst, mSrc, iMask4 );
}
inline __m256 SIMDMath::DotP( __m256 mDst, __m256 mSrc, Int iMask4 ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_dp_ps( mDst, mSrc, iMask4 );
}

inline __m128d SIMDMath::DotP( __m128d mDst, __m128d mSrc, Int iMask2 ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_dp_pd( mDst, mSrc, iMask2 );
}

////////////////////////////////////////////////////////////// Division
inline __m128 SIMDMath::DivLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_div_ss( mDst, mSrc );
}
inline __m128d SIMDMath::DivLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_sd( mDst, mSrc );
}

inline __m128 SIMDMath::Div( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_div_ps( mDst, mSrc );
}
inline __m256 SIMDMath::Div( __m256 mDst, __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_ps( mDst, mSrc );
}

inline __m128d SIMDMath::Div( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_div_pd( mDst, mSrc );
}
inline __m256d SIMDMath::Div( __m256d mDst, __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_div_pd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Average (always unsigned)
inline __m128i SIMDMath::Avg8( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_avg_epu8( mDst, mSrc );
}
inline __m256i SIMDMath::Avg8( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_avg_epu8( mDst, mSrc );
}

inline __m128i SIMDMath::Avg16( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_avg_epu16( mDst, mSrc );
}
inline __m256i SIMDMath::Avg16( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_avg_epu16( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Generic)
inline __m128 SIMDMath::CmpLower( __m128 mDst, __m128 mSrc, Int iCmpMode ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_cmp_ss( mDst, mSrc, iCmpMode );
}
inline __m128d SIMDMath::CmpLower( __m128d mDst, __m128d mSrc, Int iCmpMode ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_cmp_sd( mDst, mSrc, iCmpMode );
}

inline __m128 SIMDMath::Cmp( __m128 mDst, __m128 mSrc, Int iCmpMode ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_cmp_ps( mDst, mSrc, iCmpMode );
}
inline __m256 SIMDMath::Cmp( __m256 mDst, __m256 mSrc, Int iCmpMode ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_ps( mDst, mSrc, iCmpMode );
}

inline __m128d SIMDMath::Cmp( __m128d mDst, __m128d mSrc, Int iCmpMode ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm_cmp_pd( mDst, mSrc, iCmpMode );
}
inline __m256d SIMDMath::Cmp( __m256d mDst, __m256d mSrc, Int iCmpMode ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cmp_pd( mDst, mSrc, iCmpMode );
}

////////////////////////////////////////////////////////////// Compare (Equal)
inline __m128 SIMDMath::CmpEQLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpeq_ss( mDst, mSrc );
}
inline __m128d SIMDMath::CmpEQLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_sd( mDst, mSrc );
}

inline __m128 SIMDMath::CmpEQ( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpeq_ps( mDst, mSrc );
}

inline __m128d SIMDMath::CmpEQ( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_pd( mDst, mSrc );
}

inline __m128i SIMDMath::CmpEQ8( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_epi8( mDst, mSrc );
}
inline __m256i SIMDMath::CmpEQ8( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpeq_epi8( mDst, mSrc );
}

inline __m128i SIMDMath::CmpEQ16( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_epi16( mDst, mSrc );
}
inline __m256i SIMDMath::CmpEQ16( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpeq_epi16( mDst, mSrc );
}

inline __m128i SIMDMath::CmpEQ32( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpeq_epi32( mDst, mSrc );
}
inline __m256i SIMDMath::CmpEQ32( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpeq_epi32( mDst, mSrc );
}

inline __m128i SIMDMath::CmpEQ64( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cmpeq_epi64( mDst, mSrc );
}
inline __m256i SIMDMath::CmpEQ64( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpeq_epi64( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Not Equal)
inline __m128 SIMDMath::CmpNEQLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpneq_ss( mDst, mSrc );
}
inline __m128d SIMDMath::CmpNEQLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpneq_sd( mDst, mSrc );
}

inline __m128 SIMDMath::CmpNEQ( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpneq_ps( mDst, mSrc );
}

inline __m128d SIMDMath::CmpNEQ( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpneq_pd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Lesser-Than)
inline __m128 SIMDMath::CmpLTLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmplt_ss( mDst, mSrc );
}
inline __m128d SIMDMath::CmpLTLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_sd( mDst, mSrc );
}

inline __m128 SIMDMath::CmpLT( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmplt_ps( mDst, mSrc );
}

inline __m128d SIMDMath::CmpLT( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_pd( mDst, mSrc );
}

inline __m128i SIMDMath::CmpLT8( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_epi8( mDst, mSrc );
}

inline __m128i SIMDMath::CmpLT16( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_epi16( mDst, mSrc );
}

inline __m128i SIMDMath::CmpLT32( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmplt_epi32( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Not Lesser-Than)
inline __m128 SIMDMath::CmpNLTLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnlt_ss( mDst, mSrc );
}
inline __m128d SIMDMath::CmpNLTLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnlt_sd( mDst, mSrc );
}

inline __m128 SIMDMath::CmpNLT( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnlt_ps( mDst, mSrc );
}

inline __m128d SIMDMath::CmpNLT( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnlt_pd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Lesser-or-Equal)
inline __m128 SIMDMath::CmpLELower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmple_ss( mDst, mSrc );
}
inline __m128d SIMDMath::CmpLELower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmple_sd( mDst, mSrc );
}

inline __m128 SIMDMath::CmpLE( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmple_ps( mDst, mSrc );
}

inline __m128d SIMDMath::CmpLE( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmple_pd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Not Lesser-or-Equal)
inline __m128 SIMDMath::CmpNLELower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnle_ss( mDst, mSrc );
}
inline __m128d SIMDMath::CmpNLELower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnle_sd( mDst, mSrc );
}

inline __m128 SIMDMath::CmpNLE( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnle_ps( mDst, mSrc );
}

inline __m128d SIMDMath::CmpNLE( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnle_pd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Greater-Than)
inline __m128 SIMDMath::CmpGTLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpgt_ss( mDst, mSrc );
}
inline __m128d SIMDMath::CmpGTLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_sd( mDst, mSrc );
}

inline __m128 SIMDMath::CmpGT( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpgt_ps( mDst, mSrc );
}

inline __m128d SIMDMath::CmpGT( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_pd( mDst, mSrc );
}

inline __m128i SIMDMath::CmpGT8( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_epi8( mDst, mSrc );
}
inline __m256i SIMDMath::CmpGT8( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpgt_epi8( mDst, mSrc );
}

inline __m128i SIMDMath::CmpGT16( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_epi16( mDst, mSrc );
}
inline __m256i SIMDMath::CmpGT16( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpgt_epi16( mDst, mSrc );
}

inline __m128i SIMDMath::CmpGT32( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpgt_epi32( mDst, mSrc );
}
inline __m256i SIMDMath::CmpGT32( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpgt_epi32( mDst, mSrc );
}

inline __m128i SIMDMath::CmpGT64( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_cmpgt_epi64( mDst, mSrc );
}
inline __m256i SIMDMath::CmpGT64( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_cmpgt_epi64( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Not Greater-Than)
inline __m128 SIMDMath::CmpNGTLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpngt_ss( mDst, mSrc );
}
inline __m128d SIMDMath::CmpNGTLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpngt_sd( mDst, mSrc );
}

inline __m128 SIMDMath::CmpNGT( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpngt_ps( mDst, mSrc );
}

inline __m128d SIMDMath::CmpNGT( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpngt_pd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Greater-or-Equal)
inline __m128 SIMDMath::CmpGELower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpge_ss( mDst, mSrc );
}
inline __m128d SIMDMath::CmpGELower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpge_sd( mDst, mSrc );
}

inline __m128 SIMDMath::CmpGE( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpge_ps( mDst, mSrc );
}

inline __m128d SIMDMath::CmpGE( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpge_pd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Not Greater-or-Equal)
inline __m128 SIMDMath::CmpNGELower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnge_ss( mDst, mSrc );
}
inline __m128d SIMDMath::CmpNGELower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnge_sd( mDst, mSrc );
}

inline __m128 SIMDMath::CmpNGE( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpnge_ps( mDst, mSrc );
}

inline __m128d SIMDMath::CmpNGE( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpnge_pd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Ordered)
inline __m128 SIMDMath::CmpORDLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpord_ss( mDst, mSrc );
}
inline __m128d SIMDMath::CmpORDLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpord_sd( mDst, mSrc );
}

inline __m128 SIMDMath::CmpORD( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpord_ps( mDst, mSrc );
}

inline __m128d SIMDMath::CmpORD( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpord_pd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Unordered)
inline __m128 SIMDMath::CmpUNORDLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpunord_ss( mDst, mSrc );
}
inline __m128d SIMDMath::CmpUNORDLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpunord_sd( mDst, mSrc );
}

inline __m128 SIMDMath::CmpUNORD( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cmpunord_ps( mDst, mSrc );
}

inline __m128d SIMDMath::CmpUNORD( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cmpunord_pd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Compare (Bool results, always on lower element, _Q = non signaling versions)
inline Int SIMDMath::IsEQ( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comieq_ss( mDst, mSrc );
}
inline Int SIMDMath::IsEQ_Q( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomieq_ss( mDst, mSrc );
}
inline Int SIMDMath::IsEQ( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comieq_sd( mDst, mSrc );
}
inline Int SIMDMath::IsEQ_Q( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomieq_sd( mDst, mSrc );
}

inline Int SIMDMath::IsNEQ( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comineq_ss( mDst, mSrc );
}
inline Int SIMDMath::IsNEQ_Q( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomineq_ss( mDst, mSrc );
}
inline Int SIMDMath::IsNEQ( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comineq_sd( mDst, mSrc );
}
inline Int SIMDMath::IsNEQ_Q( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomineq_sd( mDst, mSrc );
}

inline Int SIMDMath::IsLT( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comilt_ss( mDst, mSrc );
}
inline Int SIMDMath::IsLT_Q( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomilt_ss( mDst, mSrc );
}
inline Int SIMDMath::IsLT( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comilt_sd( mDst, mSrc );
}
inline Int SIMDMath::IsLT_Q( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomilt_sd( mDst, mSrc );
}

inline Int SIMDMath::IsLE( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comile_ss( mDst, mSrc );
}
inline Int SIMDMath::IsLE_Q( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomile_ss( mDst, mSrc );
}
inline Int SIMDMath::IsLE( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comile_sd( mDst, mSrc );
}
inline Int SIMDMath::IsLE_Q( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomile_sd( mDst, mSrc );
}

inline Int SIMDMath::IsGT( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comigt_ss( mDst, mSrc );
}
inline Int SIMDMath::IsGT_Q( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomigt_ss( mDst, mSrc );
}
inline Int SIMDMath::IsGT( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comigt_sd( mDst, mSrc );
}
inline Int SIMDMath::IsGT_Q( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomigt_sd( mDst, mSrc );
}

inline Int SIMDMath::IsGE( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_comige_ss( mDst, mSrc );
}
inline Int SIMDMath::IsGE_Q( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_ucomige_ss( mDst, mSrc );
}
inline Int SIMDMath::IsGE( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_comige_sd( mDst, mSrc );
}
inline Int SIMDMath::IsGE_Q( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_ucomige_sd( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Minimum Value
inline __m128 SIMDMath::MinLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_min_ss( mDst, mSrc );
}
inline __m128d SIMDMath::MinLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_min_sd( mDst, mSrc );
}

inline __m128 SIMDMath::Min( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_min_ps( mDst, mSrc );
}
inline __m256 SIMDMath::Min( __m256 mDst, __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_min_ps( mDst, mSrc );
}

inline __m128d SIMDMath::Min( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_min_pd( mDst, mSrc );
}
inline __m256d SIMDMath::Min( __m256d mDst, __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_min_pd( mDst, mSrc );
}

inline __m128i SIMDMath::MinSigned8( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epi8( mDst, mSrc );
}
inline __m256i SIMDMath::MinSigned8( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epi8( mDst, mSrc );
}

inline __m128i SIMDMath::MinSigned16( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_min_epi16( mDst, mSrc );
}
inline __m256i SIMDMath::MinSigned16( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epi16( mDst, mSrc );
}

inline __m128i SIMDMath::MinSigned32( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epi32( mDst, mSrc );
}
inline __m256i SIMDMath::MinSigned32( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epi32( mDst, mSrc );
}

inline __m128i SIMDMath::MinSigned64( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epi64( mDst, mSrc );
}
inline __m256i SIMDMath::MinSigned64( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epi64( mDst, mSrc );
}

inline __m128i SIMDMath::MinUnsigned8( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_min_epu8( mDst, mSrc );
}
inline __m256i SIMDMath::MinUnsigned8( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epu8( mDst, mSrc );
}

inline __m128i SIMDMath::MinUnsigned16( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epu16( mDst, mSrc );
}
inline __m256i SIMDMath::MinUnsigned16( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epu16( mDst, mSrc );
}

inline __m128i SIMDMath::MinUnsigned32( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epu32( mDst, mSrc );
}
inline __m256i SIMDMath::MinUnsigned32( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epu32( mDst, mSrc );
}

inline __m128i SIMDMath::MinUnsigned64( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_min_epu64( mDst, mSrc );
}
inline __m256i SIMDMath::MinUnsigned64( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_min_epu64( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Maximum Value
inline __m128 SIMDMath::MaxLower( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_max_ss( mDst, mSrc );
}
inline __m128d SIMDMath::MaxLower( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_max_sd( mDst, mSrc );
}

inline __m128 SIMDMath::Max( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_max_ps( mDst, mSrc );
}
inline __m256 SIMDMath::Max( __m256 mDst, __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_max_ps( mDst, mSrc );
}

inline __m128d SIMDMath::Max( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_max_pd( mDst, mSrc );
}
inline __m256d SIMDMath::Max( __m256d mDst, __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_max_pd( mDst, mSrc );
}

inline __m128i SIMDMath::MaxSigned8( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epi8( mDst, mSrc );
}
inline __m256i SIMDMath::MaxSigned8( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epi8( mDst, mSrc );
}

inline __m128i SIMDMath::MaxSigned16( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_max_epi16( mDst, mSrc );
}
inline __m256i SIMDMath::MaxSigned16( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epi16( mDst, mSrc );
}

inline __m128i SIMDMath::MaxSigned32( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epi32( mDst, mSrc );
}
inline __m256i SIMDMath::MaxSigned32( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epi32( mDst, mSrc );
}

inline __m128i SIMDMath::MaxSigned64( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epi64( mDst, mSrc );
}
inline __m256i SIMDMath::MaxSigned64( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epi64( mDst, mSrc );
}

inline __m128i SIMDMath::MaxUnsigned8( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_max_epu8( mDst, mSrc );
}
inline __m256i SIMDMath::MaxUnsigned8( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epu8( mDst, mSrc );
}

inline __m128i SIMDMath::MaxUnsigned16( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epu16( mDst, mSrc );
}
inline __m256i SIMDMath::MaxUnsigned16( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epu16( mDst, mSrc );
}

inline __m128i SIMDMath::MaxUnsigned32( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epu32( mDst, mSrc );
}
inline __m256i SIMDMath::MaxUnsigned32( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epu32( mDst, mSrc );
}

inline __m128i SIMDMath::MaxUnsigned64( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE41() );
    return _mm_max_epu64( mDst, mSrc );
}
inline __m256i SIMDMath::MaxUnsigned64( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_max_epu64( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Bitwise : And
inline __m128 SIMDMath::And( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_and_ps( mDst, mSrc );
}
inline __m256 SIMDMath::And( __m256 mDst, __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_and_ps( mDst, mSrc );
}

inline __m128d SIMDMath::And( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_and_pd( mDst, mSrc );
}
inline __m256d SIMDMath::And( __m256d mDst, __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_and_pd( mDst, mSrc );
}

inline __m128i SIMDMath::And( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_and_si128( mDst, mSrc );
}
inline __m256i SIMDMath::And( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_and_si256( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Bitwise : AndNot
inline __m128 SIMDMath::AndNot( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_andnot_ps( mDst, mSrc );
}
inline __m256 SIMDMath::AndNot( __m256 mDst, __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_andnot_ps( mDst, mSrc );
}

inline __m128d SIMDMath::AndNot( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_andnot_pd( mDst, mSrc );
}
inline __m256d SIMDMath::AndNot( __m256d mDst, __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_andnot_pd( mDst, mSrc );
}

inline __m128i SIMDMath::AndNot( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_andnot_si128( mDst, mSrc );
}
inline __m256i SIMDMath::AndNot( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_andnot_si256( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Bitwise : Or
inline __m128 SIMDMath::Or( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_or_ps( mDst, mSrc );
}
inline __m256 SIMDMath::Or( __m256 mDst, __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_or_ps( mDst, mSrc );
}

inline __m128d SIMDMath::Or( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_or_pd( mDst, mSrc );
}
inline __m256d SIMDMath::Or( __m256d mDst, __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_or_pd( mDst, mSrc );
}

inline __m128i SIMDMath::Or( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_or_si128( mDst, mSrc );
}
inline __m256i SIMDMath::Or( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_or_si256( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Bitwise : Xor
inline __m128 SIMDMath::Xor( __m128 mDst, __m128 mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_xor_ps( mDst, mSrc );
}
inline __m256 SIMDMath::Xor( __m256 mDst, __m256 mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_xor_ps( mDst, mSrc );
}

inline __m128d SIMDMath::Xor( __m128d mDst, __m128d mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_xor_pd( mDst, mSrc );
}
inline __m256d SIMDMath::Xor( __m256d mDst, __m256d mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_xor_pd( mDst, mSrc );
}

inline __m128i SIMDMath::Xor( __m128i mDst, __m128i mSrc ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_xor_si128( mDst, mSrc );
}
inline __m256i SIMDMath::Xor( __m256i mDst, __m256i mSrc ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_xor_si256( mDst, mSrc );
}

////////////////////////////////////////////////////////////// Bitwise : Shift Left, Zero Extend
inline __m128i SIMDMath::Shift16L( __m128i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_slli_epi16( mDst, iCount );
}
inline __m128i SIMDMath::Shift16L( __m128i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sll_epi16( mDst, mCount );
}
inline __m256i SIMDMath::Shift16L( __m256i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_slli_epi16( mDst, iCount );
}
inline __m256i SIMDMath::Shift16L( __m256i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sll_epi16( mDst, mCount );
}

inline __m128i SIMDMath::Shift32L( __m128i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_slli_epi32( mDst, iCount );
}
inline __m128i SIMDMath::Shift32L( __m128i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sll_epi32( mDst, mCount );
}
inline __m256i SIMDMath::Shift32L( __m256i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_slli_epi32( mDst, iCount );
}
inline __m256i SIMDMath::Shift32L( __m256i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sll_epi32( mDst, mCount );
}

inline __m128i SIMDMath::Shift64L( __m128i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_slli_epi64( mDst, iCount );
}
inline __m128i SIMDMath::Shift64L( __m128i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sll_epi64( mDst, mCount );
}
inline __m256i SIMDMath::Shift64L( __m256i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_slli_epi64( mDst, iCount );
}
inline __m256i SIMDMath::Shift64L( __m256i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sll_epi64( mDst, mCount );
}

inline __m128i SIMDMath::Shift128L( __m128i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_slli_si128( mDst, iCount );
}
inline __m256i SIMDMath::Shift256L( __m256i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_slli_si256( mDst, iCount );
}

inline __m128i SIMDMath::ShiftV32L( __m128i mDst, __m128i mCounts ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_sllv_epi32( mDst, mCounts );
}
inline __m256i SIMDMath::ShiftV32L( __m256i mDst, __m256i mCounts ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sllv_epi32( mDst, mCounts );
}

inline __m128i SIMDMath::ShiftV64L( __m128i mDst, __m128i mCounts ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_sllv_epi64( mDst, mCounts );
}
inline __m256i SIMDMath::ShiftV64L( __m256i mDst, __m256i mCounts ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sllv_epi64( mDst, mCounts );
}

////////////////////////////////////////////////////////////// Bitwise : Shift Right, Zero Extend
inline __m128i SIMDMath::Shift16R( __m128i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srli_epi16( mDst, iCount );
}
inline __m128i SIMDMath::Shift16R( __m128i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srl_epi16( mDst, mCount );
}
inline __m256i SIMDMath::Shift16R( __m256i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srli_epi16( mDst, iCount );
}
inline __m256i SIMDMath::Shift16R( __m256i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srl_epi16( mDst, mCount );
}

inline __m128i SIMDMath::Shift32R( __m128i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srli_epi32( mDst, iCount );
}
inline __m128i SIMDMath::Shift32R( __m128i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srl_epi32( mDst, mCount );
}
inline __m256i SIMDMath::Shift32R( __m256i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srli_epi32( mDst, iCount );
}
inline __m256i SIMDMath::Shift32R( __m256i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srl_epi32( mDst, mCount );
}

inline __m128i SIMDMath::Shift64R( __m128i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srli_epi64( mDst, iCount );
}
inline __m128i SIMDMath::Shift64R( __m128i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srl_epi64( mDst, mCount );
}
inline __m256i SIMDMath::Shift64R( __m256i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srli_epi64( mDst, iCount );
}
inline __m256i SIMDMath::Shift64R( __m256i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srl_epi64( mDst, mCount );
}

inline __m128i SIMDMath::Shift128R( __m128i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srli_si128( mDst, iCount );
}
inline __m256i SIMDMath::Shift256R( __m256i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srli_si256( mDst, iCount );
}

inline __m128i SIMDMath::ShiftV32R( __m128i mDst, __m128i mCounts ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_srlv_epi32( mDst, mCounts );
}
inline __m256i SIMDMath::ShiftV32R( __m256i mDst, __m256i mCounts ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srlv_epi32( mDst, mCounts );
}

inline __m128i SIMDMath::ShiftV64R( __m128i mDst, __m128i mCounts ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_srlv_epi64( mDst, mCounts );
}
inline __m256i SIMDMath::ShiftV64R( __m256i mDst, __m256i mCounts ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srlv_epi64( mDst, mCounts );
}

////////////////////////////////////////////////////////////// Bitwise : Shift Right, Sign Extend
inline __m128i SIMDMath::Shift16RSE( __m128i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srai_epi16( mDst, iCount );
}
inline __m128i SIMDMath::Shift16RSE( __m128i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sra_epi16( mDst, mCount );
}
inline __m256i SIMDMath::Shift16RSE( __m256i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srai_epi16( mDst, iCount );
}
inline __m256i SIMDMath::Shift16RSE( __m256i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sra_epi16( mDst, mCount );
}

inline __m128i SIMDMath::Shift32RSE( __m128i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_srai_epi32( mDst, iCount );
}
inline __m128i SIMDMath::Shift32RSE( __m128i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sra_epi32( mDst, mCount );
}
inline __m256i SIMDMath::Shift32RSE( __m256i mDst, Int iCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srai_epi32( mDst, iCount );
}
inline __m256i SIMDMath::Shift32RSE( __m256i mDst, __m128i mCount ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_sra_epi32( mDst, mCount );
}

inline __m128i SIMDMath::ShiftV32RSE( __m128i mDst, __m128i mCounts ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm_srav_epi32( mDst, mCounts );
}
inline __m256i SIMDMath::ShiftV32RSE( __m256i mDst, __m256i mCounts ) const {
    DebugAssert( CPUIDFn->HasAVX2() );
    return _mm256_srav_epi32( mDst, mCounts );
}

////////////////////////////////////////////////////////////// CRC32
inline UInt32 SIMDMath::CRC32( UInt32 iCRC, UInt8 iValue ) const {
    DebugAssert( CPUIDFn->HasSSE42() );
    return _mm_crc32_u8( iCRC, iValue );
}
inline UInt32 SIMDMath::CRC32( UInt32 iCRC, UInt16 iValue ) const {
    DebugAssert( CPUIDFn->HasSSE42() );
    return _mm_crc32_u16( iCRC, iValue );
}
inline UInt32 SIMDMath::CRC32( UInt32 iCRC, UInt32 iValue ) const {
    DebugAssert( CPUIDFn->HasSSE42() );
    return _mm_crc32_u32( iCRC, iValue );
}
inline UInt64 SIMDMath::CRC32( UInt64 iCRC, UInt64 iValue ) const {
    DebugAssert( CPUIDFn->HasSSE42() );
    return _mm_crc32_u64( iCRC, iValue );
}

////////////////////////////////////////////////////////////// Invert
inline __m128 SIMDMath::InvertLower( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rcp_ss( mValue );
}

inline __m128 SIMDMath::Invert( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rcp_ps( mValue );
}
inline __m256 SIMDMath::Invert( __m256 mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rcp_ps( mValue );
}

////////////////////////////////////////////////////////////// SquareRoot
inline __m128 SIMDMath::SqrtLower( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sqrt_ss( mValue );
}
inline __m128d SIMDMath::SqrtLower( __m128d mValue ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sqrt_sd( mValue, mValue );
}

inline __m128 SIMDMath::Sqrt( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sqrt_ps( mValue );
}
inline __m256 SIMDMath::Sqrt( __m256 mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sqrt_ps( mValue );
}

inline __m128d SIMDMath::Sqrt( __m128d mValue ) const {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sqrt_pd( mValue );
}
inline __m256d SIMDMath::Sqrt( __m256d mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sqrt_pd( mValue );
}

////////////////////////////////////////////////////////////// Inverted SquareRoot
inline __m128 SIMDMath::InvSqrtLower( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rsqrt_ss( mValue );
}

inline __m128 SIMDMath::InvSqrt( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rsqrt_ps( mValue );
}
inline __m256 SIMDMath::InvSqrt( __m256 mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rsqrt_ps( mValue );
}

inline __m128d SIMDMath::InvSqrt( __m128d mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_invsqrt_pd( mValue );
}
inline __m256d SIMDMath::InvSqrt( __m256d mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_invsqrt_pd( mValue );
}

////////////////////////////////////////////////////////////// Cube Root
inline __m128 SIMDMath::Cbrt( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cbrt_ps( mValue );
}
inline __m256 SIMDMath::Cbrt( __m256 mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cbrt_ps( mValue );
}

inline __m128d SIMDMath::Cbrt( __m128d mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cbrt_pd( mValue );
}
inline __m256d SIMDMath::Cbrt( __m256d mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cbrt_pd( mValue );
}

////////////////////////////////////////////////////////////// Inverted Cube Root
inline __m128 SIMDMath::InvCbrt( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_invcbrt_ps( mValue );
}
inline __m256 SIMDMath::InvCbrt( __m256 mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_invcbrt_ps( mValue );
}

inline __m128d SIMDMath::InvCbrt( __m128d mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_invcbrt_pd( mValue );
}
inline __m256d SIMDMath::InvCbrt( __m256d mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_invcbrt_pd( mValue );
}

////////////////////////////////////////////////////////////// Hypothenus (Square Root of summed products)
inline __m128 SIMDMath::Hypot( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_hypot_ps( mValue );
}
inline __m256 SIMDMath::Hypot( __m256 mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hypot_ps( mValue );
}

inline __m128d SIMDMath::Hypot( __m128d mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_hypot_pd( mValue );
}
inline __m256d SIMDMath::Hypot( __m256d mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hypot_pd( mValue );
}

////////////////////////////////////////////////////////////// Natural Logarithm
inline __m128 SIMDMath::Ln( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_log_ps( mValue );
}
inline __m256 SIMDMath::Ln( __m256 mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log_ps( mValue );
}

inline __m128d SIMDMath::Ln( __m128d mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_log_pd( mValue );
}
inline __m256d SIMDMath::Ln( __m256d mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log_pd( mValue );
}

////////////////////////////////////////////////////////////// Logarithm Base 2
inline __m128 SIMDMath::Log2( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_log2_ps( mValue );
}
inline __m256 SIMDMath::Log2( __m256 mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log2_ps( mValue );
}

inline __m128d SIMDMath::Log2( __m128d mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_log2_pd( mValue );
}
inline __m256d SIMDMath::Log2( __m256d mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log2_pd( mValue );
}

////////////////////////////////////////////////////////////// Logarithm Base 10
inline __m128 SIMDMath::Log10( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_log10_ps( mValue );
}
inline __m256 SIMDMath::Log10( __m256 mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log10_ps( mValue );
}

inline __m128d SIMDMath::Log10( __m128d mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_log10_pd( mValue );
}
inline __m256d SIMDMath::Log10( __m256d mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log10_pd( mValue );
}

////////////////////////////////////////////////////////////// Natural Exponential
inline __m128 SIMDMath::Exp( __m128 mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_exp_ps( mValue );
}
inline __m256 SIMDMath::Exp( __m256 mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_exp_ps( mValue );
}

inline __m128d SIMDMath::Exp( __m128d mValue ) const {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_exp_pd( mValue );
}
inline __m256d SIMDMath::Exp( __m256d mValue ) const {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_exp_pd( mValue );
}
