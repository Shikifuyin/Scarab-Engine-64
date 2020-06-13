/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Function.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Function operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Function implementation
inline __m128 SIMD::Function::InvertOne( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rcp_ss( mValue );
}

inline __m128 SIMD::Function::Invert( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rcp_ps( mValue );
}

inline __m256 SIMD::Function::Invert( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rcp_ps( mValue );
}

inline __m128 SIMD::Function::SqrtOne( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sqrt_ss( mValue );
}
inline __m128d SIMD::Function::SqrtOne( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sqrt_sd( mValue, mValue );
}

inline __m128 SIMD::Function::Sqrt( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sqrt_ps( mValue );
}
inline __m128d SIMD::Function::Sqrt( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sqrt_pd( mValue );
}

inline __m256 SIMD::Function::Sqrt( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sqrt_ps( mValue );
}
inline __m256d SIMD::Function::Sqrt( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sqrt_pd( mValue );
}

inline __m128 SIMD::Function::InvSqrtOne( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rsqrt_ss( mValue );
}

inline __m128 SIMD::Function::InvSqrt( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_rsqrt_ps( mValue );
}
inline __m128d SIMD::Function::InvSqrt( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_invsqrt_pd( mValue );
}

inline __m256 SIMD::Function::InvSqrt( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_rsqrt_ps( mValue );
}
inline __m256d SIMD::Function::InvSqrt( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_invsqrt_pd( mValue );
}

inline __m128 SIMD::Function::Cbrt( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cbrt_ps( mValue );
}
inline __m128d SIMD::Function::Cbrt( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cbrt_pd( mValue );
}

inline __m256 SIMD::Function::Cbrt( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cbrt_ps( mValue );
}
inline __m256d SIMD::Function::Cbrt( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cbrt_pd( mValue );
}

inline __m128 SIMD::Function::InvCbrt( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_invcbrt_ps( mValue );
}
inline __m128d SIMD::Function::InvCbrt( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_invcbrt_pd( mValue );
}

inline __m256 SIMD::Function::InvCbrt( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_invcbrt_ps( mValue );
}
inline __m256d SIMD::Function::InvCbrt( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_invcbrt_pd( mValue );
}

inline __m128 SIMD::Function::Hypot( __m128 mDst, __m128 mSrc ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_hypot_ps( mDst, mSrc );
}
inline __m128d SIMD::Function::Hypot( __m128d mDst, __m128d mSrc ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_hypot_pd( mDst, mSrc );
}

inline __m256 SIMD::Function::Hypot( __m256 mDst, __m256 mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hypot_ps( mDst, mSrc );
}
inline __m256d SIMD::Function::Hypot( __m256d mDst, __m256d mSrc ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_hypot_pd( mDst, mSrc );
}

inline __m128 SIMD::Function::Ln( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_log_ps( mValue );
}
inline __m128d SIMD::Function::Ln( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_log_pd( mValue );
}

inline __m256 SIMD::Function::Ln( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log_ps( mValue );
}
inline __m256d SIMD::Function::Ln( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log_pd( mValue );
}

inline __m128 SIMD::Function::Ln1P( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_log1p_ps( mValue );
}
inline __m128d SIMD::Function::Ln1P( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_log1p_pd( mValue );
}

inline __m256 SIMD::Function::Ln1P( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log1p_ps( mValue );
}
inline __m256d SIMD::Function::Ln1P( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log1p_pd( mValue );
}

inline __m128 SIMD::Function::Log2( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_log2_ps( mValue );
}
inline __m128d SIMD::Function::Log2( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_log2_pd( mValue );
}

inline __m256 SIMD::Function::Log2( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log2_ps( mValue );
}
inline __m256d SIMD::Function::Log2( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log2_pd( mValue );
}

inline __m128 SIMD::Function::Log10( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_log10_ps( mValue );
}
inline __m128d SIMD::Function::Log10( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_log10_pd( mValue );
}

inline __m256 SIMD::Function::Log10( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log10_ps( mValue );
}
inline __m256d SIMD::Function::Log10( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_log10_pd( mValue );
}

inline __m128 SIMD::Function::Exp( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_exp_ps( mValue );
}
inline __m128d SIMD::Function::Exp( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_exp_pd( mValue );
}

inline __m256 SIMD::Function::Exp( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_exp_ps( mValue );
}
inline __m256d SIMD::Function::Exp( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_exp_pd( mValue );
}

inline __m128 SIMD::Function::ExpM1( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_expm1_ps( mValue );
}
inline __m128d SIMD::Function::ExpM1( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_expm1_pd( mValue );
}

inline __m256 SIMD::Function::ExpM1( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_expm1_ps( mValue );
}
inline __m256d SIMD::Function::ExpM1( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_expm1_pd( mValue );
}

inline __m128 SIMD::Function::Exp2( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_exp2_ps( mValue );
}
inline __m128d SIMD::Function::Exp2( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_exp2_pd( mValue );
}

inline __m256 SIMD::Function::Exp2( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_exp2_ps( mValue );
}
inline __m256d SIMD::Function::Exp2( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_exp2_pd( mValue );
}

inline __m128 SIMD::Function::Exp10( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_exp10_ps( mValue );
}
inline __m128d SIMD::Function::Exp10( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_exp10_pd( mValue );
}

inline __m256 SIMD::Function::Exp10( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_exp10_ps( mValue );
}
inline __m256d SIMD::Function::Exp10( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_exp10_pd( mValue );
}

inline __m128 SIMD::Function::Pow( __m128 mBase, __m128 mExponent ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_pow_ps( mBase, mExponent );
}
inline __m128d SIMD::Function::Pow( __m128d mBase, __m128d mExponent ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_pow_pd( mBase, mExponent );
}

inline __m256 SIMD::Function::Pow( __m256 mBase, __m256 mExponent ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_pow_ps( mBase, mExponent );
}
inline __m256d SIMD::Function::Pow( __m256d mBase, __m256d mExponent ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_pow_pd( mBase, mExponent );
}

inline __m128 SIMD::Function::Sin( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sin_ps( mValue );
}
inline __m128d SIMD::Function::Sin( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sin_pd( mValue );
}

inline __m256 SIMD::Function::Sin( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sin_ps( mValue );
}
inline __m256d SIMD::Function::Sin( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sin_pd( mValue );
}

inline __m128 SIMD::Function::Cos( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cos_ps( mValue );
}
inline __m128d SIMD::Function::Cos( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cos_pd( mValue );
}

inline __m256 SIMD::Function::Cos( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cos_ps( mValue );
}
inline __m256d SIMD::Function::Cos( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cos_pd( mValue );
}

inline __m128 SIMD::Function::SinCos( __m128 * outCos, __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sincos_ps( outCos, mValue );
}
inline __m128d SIMD::Function::SinCos( __m128d * outCos, __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sincos_pd( outCos, mValue );
}

inline __m256 SIMD::Function::SinCos( __m256 * outCos, __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sincos_ps( outCos, mValue );
}
inline __m256d SIMD::Function::SinCos( __m256d * outCos, __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sincos_pd( outCos, mValue );
}

inline __m128 SIMD::Function::Tan( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_tan_ps( mValue );
}
inline __m128d SIMD::Function::Tan( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_tan_pd( mValue );
}

inline __m256 SIMD::Function::Tan( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_tan_ps( mValue );
}
inline __m256d SIMD::Function::Tan( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_tan_pd( mValue );
}

inline __m128 SIMD::Function::ArcSin( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_asin_ps( mValue );
}
inline __m128d SIMD::Function::ArcSin( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_asin_pd( mValue );
}

inline __m256 SIMD::Function::ArcSin( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_asin_ps( mValue );
}
inline __m256d SIMD::Function::ArcSin( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_asin_pd( mValue );
}

inline __m128 SIMD::Function::ArcCos( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_acos_ps( mValue );
}
inline __m128d SIMD::Function::ArcCos( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_acos_pd( mValue );
}

inline __m256 SIMD::Function::ArcCos( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_acos_ps( mValue );
}
inline __m256d SIMD::Function::ArcCos( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_acos_pd( mValue );
}

inline __m128 SIMD::Function::ArcTan( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_atan_ps( mValue );
}
inline __m128d SIMD::Function::ArcTan( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_atan_pd( mValue );
}

inline __m256 SIMD::Function::ArcTan( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_atan_ps( mValue );
}
inline __m256d SIMD::Function::ArcTan( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_atan_pd( mValue );
}

inline __m128 SIMD::Function::ArcTan2( __m128 mNum, __m128 mDenom ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_atan2_ps( mNum, mDenom );
}
inline __m128d SIMD::Function::ArcTan2( __m128d mNum, __m128d mDenom ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_atan2_pd( mNum, mDenom );
}

inline __m256 SIMD::Function::ArcTan2( __m256 mNum, __m256 mDenom ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_atan2_ps( mNum, mDenom );
}
inline __m256d SIMD::Function::ArcTan2( __m256d mNum, __m256d mDenom ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_atan2_pd( mNum, mDenom );
}

inline __m128 SIMD::Function::SinH( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_sinh_ps( mValue );
}
inline __m128d SIMD::Function::SinH( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_sinh_pd( mValue );
}

inline __m256 SIMD::Function::SinH( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sinh_ps( mValue );
}
inline __m256d SIMD::Function::SinH( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_sinh_pd( mValue );
}

inline __m128 SIMD::Function::CosH( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cosh_ps( mValue );
}
inline __m128d SIMD::Function::CosH( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cosh_pd( mValue );
}

inline __m256 SIMD::Function::CosH( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cosh_ps( mValue );
}
inline __m256d SIMD::Function::CosH( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cosh_pd( mValue );
}

inline __m128 SIMD::Function::TanH( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_tanh_ps( mValue );
}
inline __m128d SIMD::Function::TanH( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_tanh_pd( mValue );
}

inline __m256 SIMD::Function::TanH( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_tanh_ps( mValue );
}
inline __m256d SIMD::Function::TanH( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_tanh_pd( mValue );
}

inline __m128 SIMD::Function::ArgSinH( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_asinh_ps( mValue );
}
inline __m128d SIMD::Function::ArgSinH( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_asinh_pd( mValue );
}

inline __m256 SIMD::Function::ArgSinH( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_asinh_ps( mValue );
}
inline __m256d SIMD::Function::ArgSinH( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_asinh_pd( mValue );
}

inline __m128 SIMD::Function::ArgCosH( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_acosh_ps( mValue );
}
inline __m128d SIMD::Function::ArgCosH( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_acosh_pd( mValue );
}

inline __m256 SIMD::Function::ArgCosH( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_acosh_ps( mValue );
}
inline __m256d SIMD::Function::ArgCosH( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_acosh_pd( mValue );
}

inline __m128 SIMD::Function::ArgTanH( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_atanh_ps( mValue );
}
inline __m128d SIMD::Function::ArgTanH( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_atanh_pd( mValue );
}

inline __m256 SIMD::Function::ArgTanH( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_atanh_ps( mValue );
}
inline __m256d SIMD::Function::ArgTanH( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_atanh_pd( mValue );
}

inline __m128 SIMD::Function::Erf( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_erf_ps( mValue );
}
inline __m128d SIMD::Function::Erf( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_erf_pd( mValue );
}

inline __m256 SIMD::Function::Erf( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erf_ps( mValue );
}
inline __m256d SIMD::Function::Erf( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erf_pd( mValue );
}

inline __m128 SIMD::Function::InvErf( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_erfinv_ps( mValue );
}
inline __m128d SIMD::Function::InvErf( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_erfinv_pd( mValue );
}

inline __m256 SIMD::Function::InvErf( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erfinv_ps( mValue );
}
inline __m256d SIMD::Function::InvErf( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erfinv_pd( mValue );
}

inline __m128 SIMD::Function::ErfC( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_erfc_ps( mValue );
}
inline __m128d SIMD::Function::ErfC( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_erfc_pd( mValue );
}

inline __m256 SIMD::Function::ErfC( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erfc_ps( mValue );
}
inline __m256d SIMD::Function::ErfC( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erfc_pd( mValue );
}

inline __m128 SIMD::Function::InvErfC( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_erfcinv_ps( mValue );
}
inline __m128d SIMD::Function::InvErfC( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_erfcinv_pd( mValue );
}

inline __m256 SIMD::Function::InvErfC( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erfcinv_ps( mValue );
}
inline __m256d SIMD::Function::InvErfC( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_erfcinv_pd( mValue );
}

inline __m128 SIMD::Function::CDFNorm( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cdfnorm_ps( mValue );
}
inline __m128d SIMD::Function::CDFNorm( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cdfnorm_pd( mValue );
}

inline __m256 SIMD::Function::CDFNorm( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cdfnorm_ps( mValue );
}
inline __m256d SIMD::Function::CDFNorm( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cdfnorm_pd( mValue );
}

inline __m128 SIMD::Function::InvCDFNorm( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cdfnorminv_ps( mValue );
}
inline __m128d SIMD::Function::InvCDFNorm( __m128d mValue ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    return _mm_cdfnorminv_pd( mValue );
}

inline __m256 SIMD::Function::InvCDFNorm( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cdfnorminv_ps( mValue );
}
inline __m256d SIMD::Function::InvCDFNorm( __m256d mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cdfnorminv_pd( mValue );
}

inline __m128 SIMD::Function::CSqrt( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_csqrt_ps( mValue );
}

inline __m256 SIMD::Function::CSqrt( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_csqrt_ps( mValue );
}

inline __m128 SIMD::Function::CLog( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_clog_ps( mValue );
}

inline __m256 SIMD::Function::CLog( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_clog_ps( mValue );
}

inline __m128 SIMD::Function::CExp( __m128 mValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_cexp_ps( mValue );
}

inline __m256 SIMD::Function::CExp( __m256 mValue ) {
    DebugAssert( CPUIDFn->HasAVX() );
    return _mm256_cexp_ps( mValue );
}

