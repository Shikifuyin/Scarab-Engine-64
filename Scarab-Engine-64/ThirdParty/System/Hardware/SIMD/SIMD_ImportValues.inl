/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_ImportValues.inl
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
// SIMD::Import::Values implementation
__forceinline __m128 SIMD::Import::Values::Zero128F() {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_setzero_ps();
}
__forceinline __m128d SIMD::Import::Values::Zero128D() {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_setzero_pd();
}
__forceinline __m128i SIMD::Import::Values::Zero128I() {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_setzero_si128();
}

__forceinline __m256 SIMD::Import::Values::Zero256F() {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_setzero_ps();
}
__forceinline __m256d SIMD::Import::Values::Zero256D() {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_setzero_pd();
}
__forceinline __m256i SIMD::Import::Values::Zero256I() {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_setzero_si256();
}

__forceinline __m128 SIMD::Import::Values::SetOne( Float fValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_set_ss( fValue );
}
__forceinline __m128d SIMD::Import::Values::SetOne( Double fValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_sd( fValue );
}
__forceinline __m128i SIMD::Import::Values::SetOne( Int32 iValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi32_si128( iValue );
}
__forceinline __m128i SIMD::Import::Values::SetOne( Int64 iValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cvtsi64_si128( iValue );
}

__forceinline __m128 SIMD::Import::Values::Set( Float f0, Float f1, Float f2, Float f3 ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_set_ps( f0, f1, f2, f3 );
}

__forceinline __m128d SIMD::Import::Values::Set( Double f0, Double f1 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_pd( f0, f1 );
}

__forceinline __m128i SIMD::Import::Values::Set( Int8 i0, Int8 i1, Int8 i2, Int8 i3, Int8 i4, Int8 i5, Int8 i6, Int8 i7,
                                          Int8 i8, Int8 i9, Int8 i10, Int8 i11, Int8 i12, Int8 i13, Int8 i14, Int8 i15 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_epi8( i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0 );
}
__forceinline __m128i SIMD::Import::Values::Set( Int16 i0, Int16 i1, Int16 i2, Int16 i3, Int16 i4, Int16 i5, Int16 i6, Int16 i7 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_epi16( i7, i6, i5, i4, i3, i2, i1, i0 );
}
__forceinline __m128i SIMD::Import::Values::Set( Int32 i0, Int32 i1, Int32 i2, Int32 i3 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_epi32( i3, i2, i1, i0 );
}
__forceinline __m128i SIMD::Import::Values::Set( Int64 i0, Int64 i1 ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set_epi64x( i1, i0 );
}

__forceinline __m256 SIMD::Import::Values::Set( Float f0, Float f1, Float f2, Float f3, Float f4, Float f5, Float f6, Float f7 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_ps( f7, f6, f5, f4, f3, f2, f1, f0 );
}


__forceinline __m256d SIMD::Import::Values::Set( Double f0, Double f1, Double f2, Double f3 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_pd( f3, f2, f1, f0 );
}

__forceinline __m256i SIMD::Import::Values::Set( Int8 i0, Int8 i1, Int8 i2, Int8 i3, Int8 i4, Int8 i5, Int8 i6, Int8 i7,
                                          Int8 i8, Int8 i9, Int8 i10, Int8 i11, Int8 i12, Int8 i13, Int8 i14, Int8 i15,
                                          Int8 i16, Int8 i17, Int8 i18, Int8 i19, Int8 i20, Int8 i21, Int8 i22, Int8 i23,
                                          Int8 i24, Int8 i25, Int8 i26, Int8 i27, Int8 i28, Int8 i29, Int8 i30, Int8 i31 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_epi8( i31, i30, i29, i28, i27, i26, i25, i24, i23, i22, i21, i20, i19, i18, i17, i16,
                            i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0 );
}
__forceinline __m256i SIMD::Import::Values::Set( Int16 i0, Int16 i1, Int16 i2, Int16 i3, Int16 i4, Int16 i5, Int16 i6, Int16 i7,
                                          Int16 i8, Int16 i9, Int16 i10, Int16 i11, Int16 i12, Int16 i13, Int16 i14, Int16 i15 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_epi16( i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0 );
}
__forceinline __m256i SIMD::Import::Values::Set( Int32 i0, Int32 i1, Int32 i2, Int32 i3, Int32 i4, Int32 i5, Int32 i6, Int32 i7 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_epi32( i7, i6, i5, i4, i3, i2, i1, i0 );
}
__forceinline __m256i SIMD::Import::Values::Set( Int64 i0, Int64 i1, Int64 i2, Int64 i3 ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set_epi64x( i3, i2, i1, i0 );
}

//__forceinline __m128 SIMD::Import::Values::Set( __m128 mDst, Float fSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_insert_ps( mDst, fSrc, iIndex );
//}

//__forceinline __m128d SIMD::Import::Values::Set( __m128d mDst, Double fSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_insert_pd( mDst, fSrc, iIndex );
//}

//__forceinline __m128i SIMD::Import::Values::Set( __m128i mDst, Int8 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_insert_epi8( mDst, iSrc, iIndex );
//}
//__forceinline __m128i SIMD::Import::Values::Set( __m128i mDst, Int16 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE2() );
//    return _mm_insert_epi16( mDst, iSrc, iIndex );
//}
//__forceinline __m128i SIMD::Import::Values::Set( __m128i mDst, Int32 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_insert_epi32( mDst, iSrc, iIndex );
//}
//__forceinline __m128i SIMD::Import::Values::Set( __m128i mDst, Int64 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasSSE41() );
//    return _mm_insert_epi64( mDst, iSrc, iIndex );
//}

//__forceinline __m256 SIMD::Import::Values::Set( __m256 mDst, Float fSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_insert_ps( mDst, fSrc, iIndex );
//}

//__forceinline __m256d SIMD::Import::Values::Set( __m256d mDst, Double fSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_insert_pd( mDst, fSrc, iIndex );
//}

//__forceinline __m256i SIMD::Import::Values::Set( __m256i mDst, Int8 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_insert_epi8( mDst, iSrc, iIndex );
//}
//__forceinline __m256i SIMD::Import::Values::Set( __m256i mDst, Int16 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_insert_epi16( mDst, iSrc, iIndex );
//}
//__forceinline __m256i SIMD::Import::Values::Set( __m256i mDst, Int32 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_insert_epi32( mDst, iSrc, iIndex );
//}
//__forceinline __m256i SIMD::Import::Values::Set( __m256i mDst, Int64 iSrc, Int32 iIndex ) {
//    DebugAssert( CPUIDFn->HasAVX() );
//    return _mm256_insert_epi64( mDst, iSrc, iIndex );
//}

__forceinline __m128 SIMD::Import::Values::Spread128( Float fValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_set1_ps( fValue );
}

__forceinline __m128d SIMD::Import::Values::Spread128( Double fValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_pd( fValue );
}

__forceinline __m128i SIMD::Import::Values::Spread128( Int8 iValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_epi8( iValue );
}
__forceinline __m128i SIMD::Import::Values::Spread128( Int16 iValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_epi16( iValue );
}
__forceinline __m128i SIMD::Import::Values::Spread128( Int32 iValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_epi32( iValue );
}
__forceinline __m128i SIMD::Import::Values::Spread128( Int64 iValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_set1_epi64x( iValue );
}

__forceinline __m256 SIMD::Import::Values::Spread256( Float fValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_ps( fValue );
}

__forceinline __m256d SIMD::Import::Values::Spread256( Double fValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_pd( fValue );
}

__forceinline __m256i SIMD::Import::Values::Spread256( Int8 iValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_epi8( iValue );
}
__forceinline __m256i SIMD::Import::Values::Spread256( Int16 iValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_epi16( iValue );
}
__forceinline __m256i SIMD::Import::Values::Spread256( Int32 iValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_epi32( iValue );
}
__forceinline __m256i SIMD::Import::Values::Spread256( Int64 iValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_set1_epi64x( iValue );
}

