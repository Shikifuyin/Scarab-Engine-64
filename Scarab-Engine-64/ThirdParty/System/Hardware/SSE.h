/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SSE.h
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
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// MISC
// void _m_empty(void)      - MMX
// void _mm_empty (void)    - MMX*
// __m64 _m_from_int(int)   - MMX
// int _m_to_int(__m64)     - MMX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// FENCE (serializing instructions)
// void _mm_sfence(void)    - SSE  (store to memory)
// void _mm_lfence(void)    - SSE2 (load to memory
// void _mm_mfence(void)    - SSE2 (both)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SETZERO
// __m64 _mm_setzero_si64(void)         - MMX
// __m128 _mm_setzero_ps(void)          - SSE
// __m128d _mm_setzero_pd(void)         - SSE2
// __m128i _mm_setzero_si128(void)      - SSE2
// __m256d _mm256_setzero_pd(void)      - AVX
// __m256 _mm256_setzero_ps(void)       - AVX
// __m256i _mm256_setzero_si256(void)   - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SET
// __m64 _mm_set_pi16(short, short, short, short)                                       - MMX
// __m64 _mm_set_pi32(int, int)                                                         - MMX
// __m64 _mm_set_pi8(char, char, char, char, char, char, char, char)                    - MMX
// __m128 _mm_set_ps(float, float, float, float)                                        - SSE
// __m128 _mm_set_ss(float)                                                             - SSE
// __m128i _mm_set_epi16(short, short, short, short, short, short, short, short)        - SSE2
// __m128i _mm_set_epi32(int, int, int, int)                                            - SSE2
// __m128i _mm_set_epi64(__m64, __m64)                                                  - SSE2
// __m128i _mm_set_epi8(char, char, char, char, char, char, char, char,
//                      char, char, char, char, char, char, char, char)                 - SSE2
// __m128d _mm_set_pd(double, double)                                                   - SSE2
// __m128d _mm_set_sd(double)                                                           - SSE2
// __m256i _mm256_set_epi16(short, short, short, short, short, short, short, short,
//                          short, short, short, short, short, short, short, short )    - AVX
// __m256i _mm256_set_epi32(int, int, int, int, int, int, int, int)                     - AVX
// __m256i _mm256_set_epi8(char, char, char, char, char, char, char, char,
//                         char, char, char, char, char, char, char, char,
//                         char, char, char, char, char, char, char, char,
//                         char, char, char, char, char, char, char, char)              - AVX
// __m256d _mm256_set_pd(double, double, double, double)                                - AVX
// __m256 _mm256_set_ps(float, float, float, float, float, float, float, float)         - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SET1
// __m64 _mm_set1_pi16(short)           - MMX
// __m64 _mm_set1_pi32(int)             - MMX
// __m64 _mm_set1_pi8(char)             - MMX
// __m128 _mm_set_ps1(float)            - SSE
// __m128i _mm_set1_epi16(short)        - SSE2
// __m128i _mm_set1_epi32(int)          - SSE2
// __m128i _mm_set1_epi64(__m64)        - SSE2
// __m128i _mm_set1_epi8(char)          - SSE2
// __m128d _mm_set1_pd(double)          - SSE2
// __m256i _mm256_set1_epi16(short)     - AVX
// __m256i _mm256_set1_epi32(int)       - AVX
// __m256i _mm256_set1_epi8(char)       - AVX
// __m256d _mm256_set1_pd(double)       - AVX
// __m256 _mm256_set1_ps(float)         - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SETR
// __m64 _mm_setr_pi16(short, short, short, short)                                      - MMX
// __m64 _mm_setr_pi32(int, int)                                                        - MMX
// __m64 _mm_setr_pi8(char, char, char, char, char, char, char, char)                   - MMX
// __m128 _mm_setr_ps(float, float, float, float)                                       - SSE
// __m128i _mm_setr_epi16(short, short, short, short, short, short, short, short)       - SSE2
// __m128i _mm_setr_epi32(int, int, int, int)                                           - SSE2
// __m128i _mm_setr_epi64(__m64, __m64)                                                 - SSE2
// __m128i _mm_setr_epi8(char, char, char, char, char, char, char, char,
//                       char, char, char, char, char, char, char, char)                - SSE2
// __m128d _mm_setr_pd(double, double)                                                  - SSE2
// __m256i _mm256_setr_epi16(short, short, short, short, short, short, short, short,
//                           short, short, short, short, short, short, short, short )   - AVX
// __m256i _mm256_setr_epi32(int, int, int, int, int, int, int, int)                    - AVX
// __m256i _mm256_setr_epi8(char, char, char, char, char, char, char, char,
//                          char, char, char, char, char, char, char, char,
//                          char, char, char, char, char, char, char, char,
//                          char, char, char, char, char, char, char, char)             - AVX
// __m256d _mm256_setr_pd(double, double, double, double)                               - AVX
// __m256 _mm256_setr_ps(float, float, float, float, float, float, float, float)        - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SETL (deprecated ?)
// __m128i _mm_setl_epi64(__m128i)      - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// MOVE
// __m128 _mm_move_ss(__m128, __m128)           - SSE
// __m128 _mm_movehl_ps(__m128, __m128)         - SSE
// __m128 _mm_movelh_ps(__m128, __m128)         - SSE
// __m128i _mm_move_epi64(__m128i)              - SSE2
// __m128d _mm_move_sd(__m128d, __m128d)        - SSE2
// __m64 _mm_movepi64_pi64(__m128i)             - SSE2
// __m128i _mm_movpi64_epi64(__m64)             - SSE2
// __m128d _mm_movedup_pd(__m128d)              - SSE3
// __m128 _mm_movehdup_ps(__m128)               - SSE3
// __m128 _mm_moveldup_ps(__m128)               - SSE3
// __m256d _mm256_movedup_pd(__m256d)           - AVX
// __m256 _mm256_movehdup_ps(__m256)            - AVX
// __m256 _mm256_moveldup_ps(__m256)            - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// LOAD
// __m128 _mm_load_ps(float*)                   - SSE
// __m128 _mm_load_ss(float*)                   - SSE
// __m128 _mm_loadh_pi(__m128, __m64*)          - SSE
// __m128 _mm_loadl_pi(__m128, __m64*)          - SSE
// __m128 _mm_loadr_ps(float*)                  - SSE
// __m128 _mm_loadu_ps(float*)                  - SSE
// __m128d _mm_load_pd(double*)                 - SSE2
// __m128d _mm_load_sd(double*)                 - SSE2
// __m128i _mm_load_si128(__m128i*)             - SSE2
// __m128d _mm_loadh_pd(__m128d, double*)       - SSE2
// __m128i _mm_loadl_epi64(__m128i*)            - SSE2
// __m128d _mm_loadl_pd(__m128d, double*)       - SSE2
// __m128d _mm_loadr_pd(double*)                - SSE2
// __m128d _mm_loadu_pd(double*)                - SSE2
// __m128i _mm_loadu_si128(__m128i*)            - SSE2
// __m128d _mm_loaddup_pd(double const*)        - SSE3
// __m256d _mm256_load_pd(double const *)       - AVX
// __m256 _mm256_load_ps(float const *)         - AVX
// __m256i _mm256_load_si256(__m256i *)         - AVX
// __m256d _mm256_loadu_pd(double const *)      - AVX
// __m256 _mm256_loadu_ps(float const *)        - AVX
// __m256i _mm256_loadu_si256(__m256i *)        - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// LOAD1
// __m128 _mm_load_ps1(float*)                  - SSE
// __m128d _mm_load1_pd(double*)                - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// LDD (LOAD from unaligned memory)
// __m128i _mm_lddqu_si128(__m128i const*)      - SSE3
// __m256i _mm256_lddqu_si256(__m256i *)        - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// STORE
// void _mm_store_ps(float*, __m128)            - SSE
// void _mm_store_ss(float*, __m128)            - SSE
// void _mm_storeh_pi(__m64*, __m128)           - SSE
// void _mm_storel_pi(__m64*, __m128)           - SSE
// void _mm_storer_ps(float*, __m128)           - SSE
// void _mm_storeu_ps(float*, __m128)           - SSE
// void _mm_store_pd(double*, __m128d)          - SSE2
// void _mm_store_sd(double*, __m128d)          - SSE2
// void _mm_store_si128(__m128i*, __m128i)      - SSE2
// void _mm_storeh_pd(double*, __m128d)         - SSE2
// void _mm_storel_epi64(__m128i*, __m128i)     - SSE2
// void _mm_storel_pd(double*, __m128d)         - SSE2
// void _mm_storer_pd(double*, __m128d)         - SSE2
// void _mm_storeu_pd(double*, __m128d)         - SSE2
// void _mm_storeu_si128(__m128i*, __m128i)     - SSE2
// void _mm256_store_pd(double *, __m256d)      - AVX
// void _mm256_store_ps(float *, __m256)        - AVX
// void _mm256_store_si256(__m256i *, __m256i)  - AVX
// void _mm256_storeu_pd(double *, __m256d)     - AVX
// void _mm256_storeu_ps(float *, __m256)       - AVX
// void _mm256_storeu_si256(__m256i *, __m256i) - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// STORE1
// void _mm_store_ps1(float*, __m128)           - SSE
// void _mm_store1_pd(double*, __m128d)         - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// STREAM
// void _mm_stream_pi(__m64*, __m64)                - SSE
// void _mm_stream_ps(float*, __m128)               - SSE
// void _mm_stream_pd(double*, __m128d)             - SSE2
// void _mm_stream_si128(__m128i*, __m128i)         - SSE2
// void _mm_stream_si32(int*, int)                  - SSE2
// void _mm_stream_sd(double*, __m128d)             - SSE4a
// void _mm_stream_ss(float*, __m128)               - SSE4a
// void __mm256_stream_pd(double *, __m256d)        - AVX
// void _mm256_stream_ps(float *, __m256)           - AVX
// void __mm256_stream_si256(__m256i *, __m256i)    - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// STREAM LOAD
// __m128i _mm_stream_load_si128(__m128i*)              - SSE41
// __m256i _mm256_stream_load_si256(__m256i const *)    - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// EXTRACT
// int _m_pextrw(__m64, int)                            - SSE
// int _mm_extract_epi16(__m128i, int)                  - SSE2
// __m128i _mm_extract_si64(__m128i, __m128i)           - SSE4a
// __m128i _mm_extracti_si64(__m128i, int, int)         - SSE4a
// int _mm_extract_epi32(__m128i, const int)            - SSE41
// int _mm_extract_epi8 (__m128i, const int)            - SSE41
// int _mm_extract_ps(__m128, const int)                - SSE41
// __m128d _mm256_extractf128_pd(__m256d, const int)    - AVX
// __m128 _mm256_extractf128_ps(__m256, const int)      - AVX
// __m128i _mm256_extractf128_si256(__m256i, const int) - AVX
// __m128i _mm256_extracti128_si256(__m256i, int)       - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// INSERT
// __m64 _m_pinsrw(__m64, int, int)                         - SSE
// __m128i _mm_insert_epi16(__m128i, int, int)              - SSE2
// __m128i _mm_insert_si64(__m128i, __m128i)                - SSE4a
// __m128i _mm_inserti_si64(__m128i, __m128i, int, int)     - SSE4a
// __m128i _mm_insert_epi32(__m128i, int, const int)        - SSE41
// __m128i _mm_insert_epi8 (__m128i, int, const int)        - SSE41
// __m128 _mm_insert_ps(__m128, __m128, const int)          - SSE41
// __m256d _mm256_insertf128_pd(__m256d, __m128d, int)      - AVX
// __m256 _mm256_insertf128_ps(__m256, __m128, int)         - AVX
// __m256i _mm256_insertf128_si256(__m256i, __m128i, int)   - AVX
// __m256i _mm256_inserti128_si256(__m256i, __m128i, int)   - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// GATHER32
// __m128i _mm_i32gather_epi32(int const *, __m128i, const int)                                 - AVX2
// __m128i _mm_i32gather_epi64(__int64 const *, __m128i, const int)                             - AVX2
// __m128d _mm_i32gather_pd(double const *, __m128i, const int)                                 - AVX2
// __m128 _mm_i32gather_ps(float const *, __m128i, const int)                                   - AVX2
// __m128i _mm_mask_i32gather_epi32(__m128i, int const *, __m128i, __m128i, const int)          - AVX2
// __m128i _mm_mask_i32gather_epi64(__m128i, __int64 const *, __m128i, __m128i, const int)      - AVX2
// __m128d _mm_mask_i32gather_pd(__m128d, double const *, __m128i, __m128d, const int)          - AVX2
// __m128 _mm_mask_i32gather_ps(__m128, float const *, __m128i, __m128, const int)              - AVX2
// __m256i _mm256_i32gather_epi32(int const *, __m256i, const int)                              - AVX2
// __m256i _mm256_i32gather_epi64(__int64 const *, __m128i, const int)                          - AVX2
// __m256d _mm256_i32gather_pd(double const *, __m128i, const int)                              - AVX2
// __m256 _mm256_i32gather_ps(float const *, __m256i, const int)                                - AVX2
// __m256i _mm256_mask_i32gather_epi32(__m256i, int const *, __m256i, __m256i, const int)       - AVX2
// __m256i _mm256_mask_i32gather_epi64(__m256i, __int64 const *, __m128i, __m256i, const int)   - AVX2
// __m256d _mm256_mask_i32gather_pd(__m256d, double const *, __m128i, __m256d, const int)       - AVX2
// __m256 _mm256_mask_i32gather_ps(__m256, float const *, __m256i, __m256, const int)           - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// GATHER64
// __m128i _mm_i64gather_epi32(int const *, __m128i, const int)                                 - AVX2
// __m128i _mm_i64gather_epi64(__int64 const *, __m128i, const int)                             - AVX2
// __m128d _mm_i64gather_pd(double const *, __m128i, const int)                                 - AVX2
// __m128 _mm_i64gather_ps(float const *, __m128i, const int)                                   - AVX2
// __m128i _mm_mask_i64gather_epi32(__m128i, int const *, __m128i, __m128i, const int)          - AVX2
// __m128i _mm_mask_i64gather_epi64(__m128i, __int64 const *, __m128i, __m128i, const int)      - AVX2
// __m128d _mm_mask_i64gather_pd(__m128d, double const *, __m128i, __m128d, const int)          - AVX2
// __m128 _mm_mask_i64gather_ps(__m128, float const *, __m128i, __m128, const int)              - AVX2
// __m256i _mm256_i64gather_epi32(int const *, __m256i, const int)                              - AVX2
// __m256i _mm256_i64gather_epi64(__int64 const *, __m256i, const int)                          - AVX2
// __m256d _mm256_i64gather_pd(double const *, __m256i, const int)                              - AVX2
// __m128 _mm256_i64gather_ps(float const *, __m256i, const int)                                - AVX2
// __m128i _mm256_mask_i64gather_epi32(__m128i, int const *, __m256i, __m128i, const int)       - AVX2
// __m256i _mm256_mask_i64gather_epi64(__m256i, __int64 const *, __m256i, __m256i, const int)   - AVX2
// __m256d _mm256_mask_i64gather_pd(__m256d, double const *, __m256i, __m256d, const int)       - AVX2
// __m128 _mm256_mask_i64gather_ps(__m128, float const *, __m256i, __m128, const int)           - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// BROADCAST
// __m128 _mm_broadcast_ss(float const *)       - AVX
// __m256d _mm256_broadcast_pd(__m128d const *) - AVX
// __m256 _mm256_broadcast_ps(__m128 const *)   - AVX
// __m256d _mm256_broadcast_sd(double const *)  - AVX
// __m256 _mm256_broadcast_ss(float const *)    - AVX
// __m128i _mm_broadcastb_epi8(__m128i)         - AVX2
// __m128i _mm_broadcastd_epi32(__m128i)        - AVX2
// __m128i _mm_broadcastq_epi64(__m128i)        - AVX2
// __m128d _mm_broadcastsd_pd(__m128d)          - AVX2
// __m128 _mm_broadcastss_ps(__m128)            - AVX2
// __m128i _mm_broadcastw_epi16(__m128i)        - AVX2
// __m256i _mm256_broadcastb_epi8 (__m128i)     - AVX2
// __m256i _mm256_broadcastd_epi32(__m128i)     - AVX2
// __m256i _mm256_broadcastq_epi64(__m128i)     - AVX2
// __m256d _mm256_broadcastsd_pd(__m128d)       - AVX2
// __m256i _mm256_broadcastsi128_si256(__m128i) - AVX2
// __m256 _mm256_broadcastss_ps(__m128)         - AVX2
// __m256i _mm256_broadcastw_epi16(__m128i)     - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// MASKMOVE
// void _m_maskmovq(__m64, __m64, char*)                - SSE
// void _mm_maskmoveu_si128(__m128i, __m128i, char*)    - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// MASKLOAD
// __m128d _mm_maskload_pd(double const *, __m128i)         - AVX
//  __m128 _mm_maskload_ps(float const *, __m128i)          - AVX
// __m256d _mm256_maskload_pd(double const *, __m256i)      - AVX
// __m256 _mm256_maskload_ps(float const *, __m256i)        - AVX
// __m128i _mm_maskload_epi32(int const *, __m128i)         - AVX2
// __m128i _mm_maskload_epi64(__int64 const *, __m128i)     - AVX2
// __m256i _mm256_maskload_epi32(int const *, __m256i)      - AVX2
// __m256i _mm256_maskload_epi64(__int64 const *, __m256i)  - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// MASKSTORE
// void _mm_maskstore_pd(double *, __m128i, __m128d)        - AVX
// void _mm_maskstore_ps(float *, __m128i, __m128)          - AVX
// void _mm256_maskstore_pd(double *, __m256i, __m256d)     - AVX
// void _mm256_maskstore_ps(float *, __m256i, __m256)       - AVX
// void _mm_maskstore_epi32(int *, __m128i, __m128i)        - AVX2
// void _mm_maskstore_epi64(__int64 *, __m128i, __m128i)    - AVX2
// void _mm256_maskstore_epi32(int *, __m256i, __m256i)     - AVX2
// void _mm256_maskstore_epi64(__int64 *, __m256i, __m256i) - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// MOVEMASK
// int _m_pmovmskb(__m64)               - SSE
// int _mm_movemask_ps(__m128)          - SSE
// int _mm_movemask_epi8(__m128i)       - SSE2
// int _mm_movemask_pd(__m128d)         - SSE2
// int _mm256_movemask_pd(__m256d)      - AVX
// int _mm256_movemask_ps(__m256)       - AVX
// int _mm256_movemask_epi8(__m256i)    - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// PACK
// __m64 _m_packssdw(__m64, __m64)                  - MMX
// __m64 _m_packsswb(__m64, __m64)                  - MMX
// __m64 _m_packuswb(__m64, __m64)                  - MMX
// __m64 _mm_packs_pi16 (__m64, __m64)              - MMX*
// __m64 _mm_packs_pi32 (__m64, __m64)              - MMX*
// __m64 _mm_packs_pu16 (__m64, __m64)              - MMX*
// __m128i _mm_packs_epi16(__m128i, __m128i)        - SSE2
// __m128i _mm_packs_epi32(__m128i, __m128i)        - SSE2
// __m128i _mm_packus_epi16(__m128i, __m128i)       - SSE2
// __m128i _mm_packus_epi32(__m128i, __m128i)       - SSE41
// __m256i _mm256_packs_epi16(__m256i, __m256i)     - AVX2
// __m256i _mm256_packs_epi32(__m256i, __m256i)     - AVX2
// __m256i _mm256_packus_epi16(__m256i, __m256i)    - AVX2
// __m256i _mm256_packus_epi32(__m256i, __m256i)    - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// UNPACK
// __m64 _m_punpckhbw(__m64, __m64)                 - MMX
// __m64 _m_punpckhdq(__m64, __m64)                 - MMX
// __m64 _m_punpckhwd(__m64, __m64)                 - MMX
// __m64 _m_punpcklbw(__m64, __m64)                 - MMX
// __m64 _m_punpckldq(__m64, __m64)                 - MMX
// __m64 _m_punpcklwd(__m64, __m64)                 - MMX
// __m64 _mm_unpackhi_pi8 (__m64, __m64)            - MMX*
// __m64 _mm_unpackhi_pi16 (__m64, __m64)           - MMX*
// __m64 _mm_unpackhi_pi32 (__m64, __m64)           - MMX*
// __m64 _mm_unpacklo_pi8 (__m64, __m64)            - MMX*
// __m64 _mm_unpacklo_pi16 (__m64, __m64)           - MMX*
// __m64 _mm_unpacklo_pi32 (__m64, __m64)           - MMX*
// __m128 _mm_unpackhi_ps(__m128, __m128)           - SSE
// __m128 _mm_unpacklo_ps(__m128, __m128)           - SSE
// __m128i _mm_unpackhi_epi16(__m128i, __m128i)     - SSE2
// __m128i _mm_unpackhi_epi32(__m128i, __m128i)     - SSE2
// __m128i _mm_unpackhi_epi64(__m128i, __m128i)     - SSE2
// __m128i _mm_unpackhi_epi8(__m128i, __m128i)      - SSE2
// __m128d _mm_unpackhi_pd(__m128d, __m128d)        - SSE2
// __m128i _mm_unpacklo_epi16(__m128i, __m128i)     - SSE2
// __m128i _mm_unpacklo_epi32(__m128i, __m128i)     - SSE2
// __m128i _mm_unpacklo_epi64(__m128i, __m128i)     - SSE2
// __m128i _mm_unpacklo_epi8(__m128i, __m128i)      - SSE2
// __m128d _mm_unpacklo_pd(__m128d, __m128d)        - SSE2
// __m256d _mm256_unpackhi_pd(__m256d, __m256d)     - AVX
// __m256 _mm256_unpackhi_ps(__m256, __m256)        - AVX
// __m256d _mm256_unpacklo_pd(__m256d, __m256d)     - AVX
// __m256 _mm256_unpacklo_ps(__m256, __m256)        - AVX
// __m256i _mm256_unpackhi_epi16(__m256i, __m256i)  - AVX2
// __m256i _mm256_unpackhi_epi32(__m256i, __m256i)  - AVX2
// __m256i _mm256_unpackhi_epi64(__m256i, __m256i)  - AVX2
// __m256i _mm256_unpackhi_epi8(__m256i, __m256i)   - AVX2
// __m256i _mm256_unpacklo_epi16(__m256i, __m256i)  - AVX2
// __m256i _mm256_unpacklo_epi32(__m256i, __m256i)  - AVX2
// __m256i _mm256_unpacklo_epi64(__m256i, __m256i)  - AVX2
// __m256i _mm256_unpacklo_epi8(__m256i, __m256i)   - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SHUFFLE
// __m64 _m_pshufw(__m64, int)                              - SSE
// __m128 _mm_shuffle_ps(__m128, __m128, unsigned int)      - SSE
// __m128i _mm_shuffle_epi32(__m128i, int)                  - SSE2
// __m128d _mm_shuffle_pd(__m128d, __m128d, int)            - SSE2
// __m128i _mm_shufflehi_epi16(__m128i, int)                - SSE2
// __m128i _mm_shufflelo_epi16(__m128i, int)                - SSE2
// __m128i _mm_shuffle_epi8(__m128i, __m128i)               - SSSE3
// __m64 _mm_shuffle_pi8(__m64, __m64)                      - SSSE3
// __m256d _mm256_shuffle_pd(__m256d, __m256d, const int)   - AVX
// __m256 _mm256_shuffle_ps(__m256, __m256, const int)      - AVX
// __m256i _mm256_shuffle_epi32(__m256i, const int)         - AVX2
// __m256i _mm256_shuffle_epi8(__m256i, __m256i)            - AVX2
// __m256i _mm256_shufflehi_epi16(__m256i, const int)       - AVX2
// __m256i _mm256_shufflelo_epi16(__m256i, const int)       - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// PERMUTE
// __m128d _mm_permute_pd(__m128d, int)      - AVX
// __m128 _mm_permute_ps(__m128, int)        - AVX
// __m256d _mm256_permute_pd(__m256d, int)   - AVX
// __m256 _mm256_permute_ps(__m256, int)     - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// PERMUTE2
// __m256d _mm256_permute2f128_pd(__m256d, __m256d, int)            - AVX
// __m256 _mm256_permute2f128_ps(__m256, __m256, int)               - AVX
// __m256i _mm256_permute2f128_si256(__m256i, __m256i, int)         - AVX
// __m256i _mm256_permute2x128_si256(__m256i, __m256i, const int)   - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// PERMUTE4
// __m256i _mm256_permute4x64_epi64 (__m256i, const int)    - AVX2
// __m256d _mm256_permute4x64_pd(__m256d, const int)        - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// PERMUTEVAR
// __m128d _mm_permutevar_pd(__m128d, __m128i)              - AVX
// __m128 _mm_permutevar_ps(__m128, __m128i)                - AVX
// __m256d _mm256_permutevar_pd(__m256d, __m256i)           - AVX
// __m256 _mm256_permutevar_ps(__m256, __m256i)             - AVX
// __m256i _mm256_permutevar8x32_epi32(__m256i, __m256i)    - AVX2
// __m256 _mm256_permutevar8x32_ps (__m256, __m256i)        - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CONVERT
// __m64 _mm_cvtsi32_si64(int)              - MMX*
// int _mm_cvtsi64_si32 (__m64)             - MMX*
// __m128 _mm_cvt_pi2ps(__m128, __m64)      - SSE
// __m64 _mm_cvt_ps2pi(__m128)              - SSE
// __m128 _mm_cvt_si2ss(__m128, int)        - SSE
// int _mm_cvt_ss2si(__m128)                - SSE
// __m64 _mm_cvtt_ps2pi(__m128)             - SSE
// int _mm_cvtt_ss2si(__m128)               - SSE
// __m128d _mm_cvtepi32_pd(__m128i)         - SSE2
// __m128 _mm_cvtepi32_ps(__m128i)          - SSE2
// __m128i _mm_cvtpd_epi32(__m128d)         - SSE2
// __m64 _mm_cvtpd_pi32(__m128d)            - SSE2
// __m128 _mm_cvtpd_ps(__m128d)             - SSE2
// __m128d _mm_cvtpi32_pd(__m64)            - SSE2
// __m128i _mm_cvtps_epi32(__m128)          - SSE2
// __m128d _mm_cvtps_pd(__m128)             - SSE2
// int _mm_cvtsd_si32(__m128d)              - SSE2
// __m128 _mm_cvtsd_ss(__m128, __m128d)     - SSE2
// int _mm_cvtsi128_si32(__m128i)           - SSE2
// __m128d _mm_cvtsi32_sd(__m128d, int)     - SSE2
// __m128i _mm_cvtsi32_si128(int)           - SSE2
// __m128d _mm_cvtss_sd(__m128d, __m128)    - SSE2
// __m128i _mm_cvttpd_epi32(__m128d)        - SSE2
// __m64 _mm_cvttpd_pi32(__m128d)           - SSE2
// __m128i _mm_cvttps_epi32(__m128)         - SSE2
// int _mm_cvttsd_si32(__m128d)             - SSE2
// double _mm_cvtsd_f64(__m128d)            - SSSE3
// float _mm_cvtss_f32(__m128)              - SSSE3
// __m128i _mm_cvtepi16_epi32(__m128i)      - SSE41
// __m128i _mm_cvtepi16_epi64(__m128i)      - SSE41
// __m128i _mm_cvtepi32_epi64(__m128i)      - SSE41
// __m128i _mm_cvtepi8_epi16 (__m128i)      - SSE41
// __m128i _mm_cvtepi8_epi32 (__m128i)      - SSE41
// __m128i _mm_cvtepi8_epi64 (__m128i)      - SSE41
// __m128i _mm_cvtepu16_epi32(__m128i)      - SSE41
// __m128i _mm_cvtepu16_epi64(__m128i)      - SSE41
// __m128i _mm_cvtepu32_epi64(__m128i)      - SSE41
// __m128i _mm_cvtepu8_epi16 (__m128i)      - SSE41
// __m128i _mm_cvtepu8_epi32 (__m128i)      - SSE41
// __m128i _mm_cvtepu8_epi64 (__m128i)      - SSE41
// __m256d _mm256_cvtepi32_pd(__m128i)      - AVX
// __m256 _mm256_cvtepi32_ps(__m256i)       - AVX
// __m128i _mm256_cvtpd_epi32(__m256d)      - AVX
// __m128 _mm256_cvtpd_ps(__m256d)          - AVX
// __m256i _mm256_cvtps_epi32(__m256)       - AVX
// __m256d _mm256_cvtps_pd(__m128)          - AVX
// __m128i _mm256_cvttpd_epi32(__m256d)     - AVX
// __m256i _mm256_cvttps_epi32(__m256)      - AVX
// __m256i _mm256_cvtepi16_epi32(__m128i)   - AVX2
// __m256i _mm256_cvtepi16_epi64(__m128i)   - AVX2
// __m256i _mm256_cvtepi32_epi64(__m128i)   - AVX2
// __m256i _mm256_cvtepi8_epi16(__m128i)    - AVX2
// __m256i _mm256_cvtepi8_epi32(__m128i)    - AVX2
// __m256i _mm256_cvtepi8_epi64(__m128i)    - AVX2
// __m256i _mm256_cvtepu16_epi32(__m128i)   - AVX2
// __m256i _mm256_cvtepu16_epi64(__m128i)   - AVX2
// __m256i _mm256_cvtepu32_epi64(__m128i)   - AVX2
// __m256i _mm256_cvtepu8_epi16(__m128i)    - AVX2
// __m256i _mm256_cvtepu8_epi32(__m128i)    - AVX2
// __m256i _mm256_cvtepu8_epi64(__m128i)    - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CAST
// __m128 _mm_castpd_ps(__m128d)            - SSSE3
// __m128i _mm_castpd_si128(__m128d)        - SSSE3
// __m128d _mm_castps_pd(__m128)            - SSSE3
// __m128i _mm_castps_si128(__m128)         - SSSE3
// __m128d _mm_castsi128_pd(__m128i)        - SSSE3
// __m128 _mm_castsi128_ps(__m128i)         - SSSE3
// __m256 _mm256_castpd_ps(__m256d)         - AVX
// __m256i _mm256_castpd_si256(__m256d)     - AVX
// __m256d _mm256_castpd128_pd256(__m128d)  - AVX
// __m128d _mm256_castpd256_pd128(__m256d)  - AVX
// __m256d _mm256_castps_pd(__m256)         - AVX
// __m256i _mm256_castps_si256(__m256)      - AVX
// __m256 _mm256_castps128_ps256(__m128)    - AVX
// __m128 _mm256_castps256_ps128(__m256)    - AVX
// __m256i _mm256_castsi128_si256(__m128i)  - AVX
// __m256d _mm256_castsi256_pd(__m256i)     - AVX
// __m256 _mm256_castsi256_ps(__m256i)      - AVX
// __m128i _mm256_castsi256_si128(__m256i)  - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ABS
// __m128i _mm_abs_epi16(__m128i)       - SSSE3
// __m128i _mm_abs_epi32(__m128i)       - SSSE3
// __m128i _mm_abs_epi8(__m128i)        - SSSE3
// __m64 _mm_abs_pi16(__m64)            - SSSE3
// __m64 _mm_abs_pi32(__m64)            - SSSE3
// __m64 _mm_abs_pi8(__m64)             - SSSE3
// __m256i _mm256_abs_epi16(__m256i)    - AVX2
// __m256i _mm256_abs_epi32(__m256i)    - AVX2
// __m256i _mm256_abs_epi8(__m256i)     - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SIGN
// __m128i _mm_sign_epi16(__m128i, __m128i)     - SSSE3
// __m128i _mm_sign_epi32(__m128i, __m128i)     - SSSE3
// __m128i _mm_sign_epi8(__m128i, __m128i)      - SSSE3
// __m64 _mm_sign_pi16(__m64, __m64)            - SSSE3
// __m64 _mm_sign_pi32(__m64, __m64)            - SSSE3
// __m64 _mm_sign_pi8(__m64, __m64)             - SSSE3
// __m256i _mm256_sign_epi16(__m256i, __m256i)  - AVX2
// __m256i _mm256_sign_epi32(__m256i, __m256i)  - AVX2
// __m256i _mm256_sign_epi8(__m256i, __m256i)   - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ROUND
// __m128d _mm_round_pd(__m128d, const int)             - SSE41
// __m128 _mm_round_ps(__m128, const int)               - SSE41
// __m128d _mm_round_sd(__m128d, __m128d, const int)    - SSE41
// __m128 _mm_round_ss(__m128, __m128, const int)       - SSE41
// __m256d _mm256_round_pd(__m256d, int)                - AVX
// __m256 _mm256_round_ps(__m256, int)                  - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ADD
// __m64 _m_paddb(__m64, __m64)                 - MMX
// __m64 _m_paddd(__m64, __m64)                 - MMX
// __m64 _m_paddsb(__m64, __m64)                - MMX
// __m64 _m_paddsw(__m64, __m64)                - MMX
// __m64 _m_paddusb(__m64, __m64)               - MMX
// __m64 _m_paddusw(__m64, __m64)               - MMX
// __m64 _m_paddw(__m64, __m64)                 - MMX
// __m64 _mm_add_pi8(__m64, __m64)              - MMX*
// __m64 _mm_add_pi16(__m64, __m64)             - MMX*
// __m64 _mm_add_pi32(__m64, __m64)             - MMX*
// __m64 _mm_adds_pi8(__m64, __m64)             - MMX*
// __m64 _mm_adds_pi16(__m64, __m64)            - MMX*
// __m64 _mm_adds_pu8(__m64, __m64)             - MMX*
// __m64 _mm_adds_pu16(__m64, __m64)            - MMX*
// __m128 _mm_add_ps(__m128, __m128)            - SSE
// __m128 _mm_add_ss(__m128, __m128)            - SSE
// __m128i _mm_add_epi16(__m128i, __m128i)      - SSE2
// __m128i _mm_add_epi32(__m128i, __m128i)      - SSE2
// __m128i _mm_add_epi64(__m128i, __m128i)      - SSE2
// __m128i _mm_add_epi8(__m128i, __m128i)       - SSE2
// __m128d _mm_add_pd(__m128d, __m128d)         - SSE2
// __m128d _mm_add_sd(__m128d, __m128d)         - SSE2
// __m64 _mm_add_si64(__m64, __m64)             - SSE2
// __m128i _mm_adds_epi16(__m128i, __m128i)     - SSE2
// __m128i _mm_adds_epi8(__m128i, __m128i)      - SSE2
// __m128i _mm_adds_epu16(__m128i, __m128i)     - SSE2
// __m128i _mm_adds_epu8(__m128i, __m128i)      - SSE2
// __m256d _mm256_add_pd(__m256d, __m256d)      - AVX
// __m256 _mm256_add_ps(__m256, __m256)         - AVX
// __m256i _mm256_add_epi16(__m256i, __m256i)   - AVX2
// __m256i _mm256_add_epi32(__m256i, __m256i)   - AVX2
// __m256i _mm256_add_epi64(__m256i, __m256i)   - AVX2
// __m256i _mm256_add_epi8(__m256i, __m256i)    - AVX2
// __m256i _mm256_adds_epi16(__m256i, __m256i)  - AVX2
// __m256i _mm256_adds_epi8(__m256i, __m256i)   - AVX2
// __m256i _mm256_adds_epu16(__m256i, __m256i)  - AVX2
// __m256i _mm256_adds_epu8(__m256i, __m256i)   - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// HADD
// __m128d _mm_hadd_pd(__m128d, __m128d)        - SSE3
// __m128 _mm_hadd_ps(__m128, __m128)           - SSE3
// __m128i _mm_hadd_epi16(__m128i, __m128i)     - SSSE3
// __m128i _mm_hadd_epi32(__m128i, __m128i)     - SSSE3
// __m64 _mm_hadd_pi16(__m64, __m64)            - SSSE3
// __m64 _mm_hadd_pi32(__m64, __m64)            - SSSE3
// __m128i _mm_hadds_epi16(__m128i, __m128i)    - SSSE3
// __m64 _mm_hadds_pi16(__m64, __m64)           - SSSE3
// __m256d _mm256_hadd_pd(__m256d, __m256d)     - AVX
// __m256 _mm256_hadd_ps(__m256, __m256)        - AVX
// __m256i _mm256_hadd_epi16(__m256i, __m256i)  - AVX2
// __m256i _mm256_hadd_epi32(__m256i, __m256i)  - AVX2
// __m256i _mm256_hadds_epi16(__m256i, __m256i) - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SUB
// __m64 _m_psubb(__m64, __m64)                 - MMX
// __m64 _m_psubd(__m64, __m64)                 - MMX
// __m64 _m_psubsb(__m64, __m64)                - MMX
// __m64 _m_psubsw(__m64, __m64)                - MMX
// __m64 _m_psubusb(__m64, __m64)               - MMX
// __m64 _m_psubusw(__m64, __m64)               - MMX
// __m64 _m_psubw(__m64, __m64)                 - MMX
// __m64 _mm_sub_pi8(__m64, __m64)              - MMX*
// __m64 _mm_sub_pi16(__m64, __m64)             - MMX*
// __m64 _mm_sub_pi32(__m64, __m64)             - MMX*
// __m64 _mm_subs_pi8(__m64, __m64)             - MMX*
// __m64 _mm_subs_pi16(__m64, __m64)            - MMX*
// __m64 _mm_subs_pu8(__m64, __m64)             - MMX*
// __m64 _mm_subs_pu16(__m64, __m64)            - MMX*
// __m128 _mm_sub_ps(__m128, __m128)            - SSE
// __m128 _mm_sub_ss(__m128, __m128)            - SSE
// __m128i _mm_sub_epi16(__m128i, __m128i)      - SSE2
// __m128i _mm_sub_epi32(__m128i, __m128i)      - SSE2
// __m128i _mm_sub_epi64(__m128i, __m128i)      - SSE2
// __m128i _mm_sub_epi8(__m128i, __m128i)       - SSE2
// __m128d _mm_sub_pd(__m128d, __m128d)         - SSE2
// __m128d _mm_sub_sd(__m128d, __m128d)         - SSE2
// __m64 _mm_sub_si64(__m64, __m64)             - SSE2
// __m128i _mm_subs_epi16(__m128i, __m128i)     - SSE2
// __m128i _mm_subs_epi8(__m128i, __m128i)      - SSE2
// __m128i _mm_subs_epu16(__m128i, __m128i)     - SSE2
// __m128i _mm_subs_epu8(__m128i, __m128i)      - SSE2
// __m256d _mm256_sub_pd(__m256d, __m256d)      - AVX
// __m256 _mm256_sub_ps(__m256, __m256)         - AVX
// __m256i _mm256_sub_epi16(__m256i, __m256i)   - AVX2
// __m256i _mm256_sub_epi32(__m256i, __m256i)   - AVX2
// __m256i _mm256_sub_epi64(__m256i, __m256i)   - AVX2
// __m256i _mm256_sub_epi8(__m256i, __m256i)    - AVX2
// __m256i _mm256_subs_epi16(__m256i, __m256i)  - AVX2
// __m256i _mm256_subs_epi8(__m256i, __m256i)   - AVX2
// __m256i _mm256_subs_epu16(__m256i, __m256i)  - AVX2
// __m256i _mm256_subs_epu8(__m256i, __m256i)   - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// HSUB
// __m128d _mm_hsub_pd(__m128d, __m128d)        - SSE3
// __m128 _mm_hsub_ps(__m128, __m128)           - SSE3
// __m128i _mm_hsub_epi16(__m128i, __m128i)     - SSSE3
// __m128i _mm_hsub_epi32(__m128i, __m128i)     - SSSE3
// __m64 _mm_hsub_pi16(__m64, __m64)            - SSSE3
// __m64 _mm_hsub_pi32(__m64, __m64)            - SSSE3
// __m128i _mm_hsubs_epi16(__m128i, __m128i)    - SSSE3
// __m64 _mm_hsubs_pi16(__m64, __m64)           - SSSE3
// __m256d _mm256_hsub_pd(__m256d, __m256d)     - AVX
// __m256 _mm256_hsub_ps(__m256, __m256)        - AVX
// __m256i _mm256_hsub_epi16(__m256i, __m256i)  - AVX2
// __m256i _mm256_hsub_epi32(__m256i, __m256i)  - AVX2
// __m256i _mm256_hsubs_epi16(__m256i, __m256i) - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ADDSUB
// __m128d _mm_addsub_pd(__m128d, __m128d)      - SSE3
// __m128 _mm_addsub_ps(__m128, __m128)         - SSE3
// __m256d _mm256_addsub_pd(__m256d, __m256d)   - AVX
// __m256 _mm256_addsub_ps(__m256, __m256)      - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SAD
// __m64 _m_psadbw(__m64, __m64)                            - SSE
// __m128i _mm_sad_epu8(__m128i, __m128i)                   - SSE2
// __m128i _mm_mpsadbw_epu8(__m128i, __m128i, const int)    - SSE41
// __m256i _mm256_sad_epu8(__m256i, __m256i)                - AVX2
// __m256i _mm256_mpsadbw_epu8(__m256i, __m256i, const int) - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// MUL
// __m64 _m_pmulhw(__m64, __m64)                    - MMX
// __m64 _m_pmullw(__m64, __m64)                    - MMX
// __m64 _mm_mulhi_pi16(__m64, __m64)               - MMX*
// __m64 _mm_mullo_pi16(__m64, __m64)               - MMX*
// __m64 _m_pmulhuw(__m64, __m64)                   - SSE
// __m128 _mm_mul_ps(__m128, __m128)                - SSE
// __m128 _mm_mul_ss(__m128, __m128)                - SSE
// __m128i _mm_mul_epu32(__m128i, __m128i)          - SSE2
// __m128d _mm_mul_pd(__m128d, __m128d)             - SSE2
// __m128d _mm_mul_sd(__m128d, __m128d)             - SSE2
// __m64 _mm_mul_su32(__m64, __m64)                 - SSE2
// __m128i _mm_mulhi_epi16(__m128i, __m128i)        - SSE2
// __m128i _mm_mulhi_epu16(__m128i, __m128i)        - SSE2
// __m128i _mm_mullo_epi16(__m128i, __m128i)        - SSE2
// __m128i _mm_mulhrs_epi16(__m128i, __m128i)       - SSSE3
// __m64 _mm_mulhrs_pi16(__m64, __m64)              - SSSE3
// __m128i _mm_mul_epi32(__m128i, __m128i)          - SSE41
// __m128i _mm_mullo_epi32(__m128i, __m128i)        - SSE41
// __m256d _mm256_mul_pd(__m256d, __m256d)          - AVX
// __m256 _mm256_mul_ps(__m256, __m256)             - AVX
// __m256i _mm256_mul_epi32(__m256i, __m256i)       - AVX2
// __m256i _mm256_mul_epu32(__m256i, __m256i)       - AVX2
// __m256i _mm256_mulhi_epi16(__m256i, __m256i)     - AVX2
// __m256i _mm256_mulhi_epu16(__m256i, __m256i)     - AVX2
// __m256i _mm256_mulhrs_epi16(__m256i, __m256i)    - AVX2
// __m256i _mm256_mullo_epi16(__m256i, __m256i)     - AVX2
// __m256i _mm256_mullo_epi32(__m256i, __m256i)     - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// MADD
// __m64 _m_pmaddwd(__m64, __m64)                   - MMX
// __m64 _mm_madd_pi16(__m64, __m64)                - MMX*
// __m128i _mm_madd_epi16(__m128i, __m128i)         - SSE2
// __m128i _mm_maddubs_epi16(__m128i, __m128i)      - SSSE3
// __m64 _mm_maddubs_pi16(__m64, __m64)             - SSSE3
// __m256i _mm256_madd_epi16(__m256i, __m256i)      - AVX2
// __m256i _mm256_maddubs_epi16(__m256i, __m256i)   - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// BLEND
// __m128i _mm_blend_epi16 (__m128i, __m128i, const int)    - SSE41
// __m128d _mm_blend_pd (__m128d, __m128d, const int)       - SSE41
// __m128 _mm_blend_ps (__m128, __m128, const int)          - SSE41
// __m128i _mm_blendv_epi8 (__m128i, __m128i, __m128i)      - SSE41
// __m128d _mm_blendv_pd(__m128d, __m128d, __m128d)         - SSE41
// __m128 _mm_blendv_ps(__m128, __m128, __m128)             - SSE41
// __m256d _mm256_blend_pd(__m256d, __m256d, const int)     - AVX
// __m256 _mm256_blend_ps(__m256, __m256, const int)        - AVX
// __m256d _mm256_blendv_pd(__m256d, __m256d, __m256d)      - AVX
// __m256 _mm256_blendv_ps(__m256, __m256, __m256)          - AVX
// __m128i _mm_blend_epi32(__m128i, __m128i, const int)     - AVX2
// __m256i _mm256_blend_epi16(__m256i, __m256i, const int)  - AVX2
// __m256i _mm256_blend_epi32(__m256i, __m256i, const int)  - AVX2
// __m256i _mm256_blendv_epi8(__m256i, __m256i, __m256i)    - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// DP
// __m128d _mm_dp_pd(__m128d, __m128d, const int)   - SSE41
// __m128 _mm_dp_ps(__m128, __m128, const int)      - SSE41
// __m256 _mm256_dp_ps(__m256, __m256, const int)   - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// DIV
// __m128 _mm_div_ps(__m128, __m128)        - SSE
// __m128 _mm_div_ss(__m128, __m128)        - SSE
// __m128d _mm_div_pd(__m128d, __m128d)     - SSE2
// __m128d _mm_div_sd(__m128d, __m128d)     - SSE2
// __m256d _mm256_div_pd(__m256d, __m256d)  - AVX
// __m256 _mm256_div_ps(__m256, __m256)     - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// AVG
// __m64 _m_pavgb(__m64, __m64)                 - SSE
// __m64 _m_pavgw(__m64, __m64)                 - SSE
// __m128i _mm_avg_epu16(__m128i, __m128i)      - SSE2
// __m128i _mm_avg_epu8(__m128i, __m128i)       - SSE2
// __m256i _mm256_avg_epu16(__m256i, __m256i)   - AVX2
// __m256i _mm256_avg_epu8(__m256i, __m256i)    - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// RCP
// __m128 _mm_rcp_ps(__m128)    - SSE
// __m128 _mm_rcp_ss(__m128)    - SSE
// __m256 _mm256_rcp_ps(__m256) - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SQRT
// __m128 _mm_sqrt_ps(__m128)               - SSE
// __m128 _mm_sqrt_ss(__m128)               - SSE
// __m128d _mm_sqrt_pd(__m128d)             - SSE2
// __m128d _mm_sqrt_sd(__m128d, __m128d)    - SSE2
// __m256d _mm256_sqrt_pd(__m256d)          - AVX
// __m256 _mm256_sqrt_ps(__m256)            - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// RSQRT
// __m128 _mm_rsqrt_ps(__m128)      - SSE
// __m128 _mm_rsqrt_ss(__m128)      - SSE
// __m256 _mm256_rsqrt_ps(__m256)   - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMP
// __m64 _m_pcmpeqb(__m64, __m64)                               - MMX
// __m64 _m_pcmpeqd(__m64, __m64)                               - MMX
// __m64 _m_pcmpeqw(__m64, __m64)                               - MMX
// __m64 _m_pcmpgtb(__m64, __m64)                               - MMX
// __m64 _m_pcmpgtd(__m64, __m64)                               - MMX
// __m64 _m_pcmpgtw(__m64, __m64)                               - MMX
// __m64 _mm_cmpeq_pi8(__m64, __m64)                            - MMX*
// __m64 _mm_cmpeq_pi16(__m64, __m64)                           - MMX*
// __m64 _mm_cmpeq_pi32(__m64, __m64)                           - MMX*
// __m64 _mm_cmpgt_pi8(__m64, __m64)                            - MMX*
// __m64 _mm_cmpgt_pi16(__m64, __m64)                           - MMX*
// __m64 _mm_cmpgt_pi32(__m64, __m64)                           - MMX*
// __m128 _mm_cmpeq_ps(__m128, __m128)                          - SSE
// __m128 _mm_cmpeq_ss(__m128, __m128)                          - SSE
// __m128 _mm_cmpge_ps(__m128, __m128)                          - SSE
// __m128 _mm_cmpge_ss(__m128, __m128)                          - SSE
// __m128 _mm_cmpgt_ps(__m128, __m128)                          - SSE
// __m128 _mm_cmpgt_ss(__m128, __m128)                          - SSE
// __m128 _mm_cmple_ps(__m128, __m128)                          - SSE
// __m128 _mm_cmple_ss(__m128, __m128)                          - SSE
// __m128 _mm_cmplt_ps(__m128, __m128)                          - SSE
// __m128 _mm_cmplt_ss(__m128, __m128)                          - SSE
// __m128 _mm_cmpneq_ps(__m128, __m128)                         - SSE
// __m128 _mm_cmpneq_ss(__m128, __m128)                         - SSE
// __m128 _mm_cmpnge_ps(__m128, __m128)                         - SSE
// __m128 _mm_cmpnge_ss(__m128, __m128)                         - SSE
// __m128 _mm_cmpngt_ps(__m128, __m128)                         - SSE
// __m128 _mm_cmpngt_ss(__m128, __m128)                         - SSE
// __m128 _mm_cmpnle_ps(__m128, __m128)                         - SSE
// __m128 _mm_cmpnle_ss(__m128, __m128)                         - SSE
// __m128 _mm_cmpnlt_ps(__m128, __m128)                         - SSE
// __m128 _mm_cmpnlt_ss(__m128, __m128)                         - SSE
// __m128 _mm_cmpord_ps(__m128, __m128)                         - SSE
// __m128 _mm_cmpord_ss(__m128, __m128)                         - SSE
// __m128 _mm_cmpunord_ps(__m128, __m128)                       - SSE
// __m128 _mm_cmpunord_ss(__m128, __m128)                       - SSE
// __m128i _mm_cmpeq_epi16(__m128i, __m128i)                    - SSE2
// __m128i _mm_cmpeq_epi32(__m128i, __m128i)                    - SSE2
// __m128i _mm_cmpeq_epi8(__m128i, __m128i)                     - SSE2
// __m128d _mm_cmpeq_pd(__m128d, __m128d)                       - SSE2
// __m128d _mm_cmpeq_sd(__m128d, __m128d)                       - SSE2
// __m128d _mm_cmpge_pd(__m128d, __m128d)                       - SSE2
// __m128d _mm_cmpge_sd(__m128d, __m128d)                       - SSE2
// __m128i _mm_cmpgt_epi16(__m128i, __m128i)                    - SSE2
// __m128i _mm_cmpgt_epi32(__m128i, __m128i)                    - SSE2
// __m128i _mm_cmpgt_epi8(__m128i, __m128i)                     - SSE2
// __m128d _mm_cmpgt_pd(__m128d, __m128d)                       - SSE2
// __m128d _mm_cmpgt_sd(__m128d, __m128d)                       - SSE2
// __m128d _mm_cmple_pd(__m128d, __m128d)                       - SSE2
// __m128d _mm_cmple_sd(__m128d, __m128d)                       - SSE2
// __m128i _mm_cmplt_epi16(__m128i, __m128i)                    - SSE2
// __m128i _mm_cmplt_epi32(__m128i, __m128i)                    - SSE2
// __m128i _mm_cmplt_epi8(__m128i, __m128i)                     - SSE2
// __m128d _mm_cmplt_pd(__m128d, __m128d)                       - SSE2
// __m128d _mm_cmplt_sd(__m128d, __m128d)                       - SSE2
// __m128d _mm_cmpneq_pd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpneq_sd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpnge_pd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpnge_sd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpngt_pd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpngt_sd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpnle_pd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpnle_sd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpnlt_pd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpnlt_sd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpord_pd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpord_sd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpunord_pd(__m128d, __m128d)                    - SSE2
// __m128d _mm_cmpunord_sd(__m128d, __m128d)                    - SSE2
// __m128i _mm_cmpeq_epi64(__m128i, __m128i)                    - SSE41
// int _mm_cmpestra(__m128i, int, __m128i, int, const int)      - SSE42
// int _mm_cmpestrc(__m128i, int, __m128i, int, const int)      - SSE42
// int _mm_cmpestri(__m128i, int, __m128i, int, const int)      - SSE42
// __m128i _mm_cmpestrm(__m128i, int, __m128i, int, const int)  - SSE42
// int _mm_cmpestro(__m128i, int, __m128i, int, const int)      - SSE42
// int _mm_cmpestrs(__m128i, int, __m128i, int, const int)      - SSE42
// int _mm_cmpestrz(__m128i, int, __m128i, int, const int)      - SSE42
// __m128i _mm_cmpgt_epi64(__m128i, __m128i)                    - SSE42
// int _mm_cmpistra(__m128i, __m128i, const int)                - SSE42
// int _mm_cmpistrc(__m128i, __m128i, const int)                - SSE42
// int _mm_cmpistri(__m128i, __m128i, const int)                - SSE42
// __m128i _mm_cmpistrm(__m128i, __m128i, const int)            - SSE42
// int _mm_cmpistro(__m128i, __m128i, const int)                - SSE42
// int _mm_cmpistrs(__m128i, __m128i, const int)                - SSE42
// int _mm_cmpistrz(__m128i, __m128i, const int)                - SSE42
// __m128d _mm_cmp_pd(__m128d, __m128d, const int)              - AVX
// __m128 _mm_cmp_ps(__m128, __m128, const int)                 - AVX
// __m128d _mm_cmp_sd(__m128d, __m128d, const int)              - AVX
// __m128 _mm_cmp_ss(__m128, __m128, const int)                 - AVX
// __m256d _mm256_cmp_pd(__m256d, __m256d, const int)           - AVX
// __m256 _mm256_cmp_ps(__m256, __m256, const int)              - AVX
// __m256i _mm256_cmpeq_epi16(__m256i, __m256i)                 - AVX2
// __m256i _mm256_cmpeq_epi32(__m256i, __m256i)                 - AVX2
// __m256i _mm256_cmpeq_epi64(__m256i, __m256i)                 - AVX2
// __m256i _mm256_cmpeq_epi8(__m256i, __m256i)                  - AVX2
// __m256i _mm256_cmpgt_epi16(__m256i, __m256i)                 - AVX2
// __m256i _mm256_cmpgt_epi32(__m256i, __m256i)                 - AVX2
// __m256i _mm256_cmpgt_epi64(__m256i, __m256i)                 - AVX2
// __m256i _mm256_cmpgt_epi8(__m256i, __m256i)                  - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// COMI
// int _mm_comieq_ss(__m128, __m128)        - SSE
// int _mm_comige_ss(__m128, __m128)        - SSE
// int _mm_comigt_ss(__m128, __m128)        - SSE
// int _mm_comile_ss(__m128, __m128)        - SSE
// int _mm_comilt_ss(__m128, __m128)        - SSE
// int _mm_comineq_ss(__m128, __m128)       - SSE
// int _mm_comieq_sd(__m128d, __m128d)      - SSE2
// int _mm_comige_sd(__m128d, __m128d)      - SSE2
// int _mm_comigt_sd(__m128d, __m128d)      - SSE2
// int _mm_comile_sd(__m128d, __m128d)      - SSE2
// int _mm_comilt_sd(__m128d, __m128d)      - SSE2
// int _mm_comineq_sd(__m128d, __m128d)     - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// UCOMI
// int _mm_ucomieq_ss(__m128, __m128)       - SSE
// int _mm_ucomige_ss(__m128, __m128)       - SSE
// int _mm_ucomigt_ss(__m128, __m128)       - SSE
// int _mm_ucomile_ss(__m128, __m128)       - SSE
// int _mm_ucomilt_ss(__m128, __m128)       - SSE
// int _mm_ucomineq_ss(__m128, __m128)      - SSE
// int _mm_ucomieq_sd(__m128d, __m128d)     - SSE2
// int _mm_ucomige_sd(__m128d, __m128d)     - SSE2
// int _mm_ucomigt_sd(__m128d, __m128d)     - SSE2
// int _mm_ucomile_sd(__m128d, __m128d)     - SSE2
// int _mm_ucomilt_sd(__m128d, __m128d)     - SSE2
// int _mm_ucomineq_sd(__m128d, __m128d)    - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// TEST
// int _mm_testc_si128(__m128i, __m128i)        - SSE41
// int _mm_testnzc_si128(__m128i, __m128i)      - SSE41
// int _mm_testz_si128(__m128i, __m128i)        - SSE41
// int _mm_testc_pd(__m128d, __m128d)           - AVX
// int _mm_testc_ps(__m128, __m128)             - AVX
// int _mm_testnzc_pd(__m128d, __m128d)         - AVX
// int _mm_testnzc_ps(__m128, __m128)           - AVX
// int _mm_testz_pd(__m128d, __m128d)           - AVX
// int _mm_testz_ps(__m128, __m128)             - AVX
// int _mm256_testc_pd(__m256d, __m256d)        - AVX
// int _mm256_testc_ps(__m256, __m256)          - AVX
// int _mm256_testc_si256(__m256i, __m256i)     - AVX
// int _mm256_testnzc_pd(__m256d, __m256d)      - AVX
// int _mm256_testnzc_ps(__m256, __m256)        - AVX
// int _mm256_testnzc_si256(__m256i, __m256i)   - AVX
// int _mm256_testz_pd(__m256d, __m256d)        - AVX
// int _mm256_testz_ps(__m256, __m256)          - AVX
// int _mm256_testz_si256(__m256i, __m256i)     - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// MIN
// __m64 _m_pminsw(__m64, __m64)                - SSE
// __m64 _m_pminub(__m64, __m64)                - SSE
// __m128 _mm_min_ps(__m128, __m128)            - SSE
// __m128 _mm_min_ss(__m128, __m128)            - SSE
// __m128i _mm_min_epi16(__m128i, __m128i)      - SSE2
// __m128i _mm_min_epu8(__m128i, __m128i)       - SSE2
// __m128d _mm_min_pd(__m128d, __m128d)         - SSE2
// __m128d _mm_min_sd(__m128d, __m128d)         - SSE2
// __m128i _mm_min_epi32(__m128i, __m128i)      - SSE41
// __m128i _mm_min_epi8 (__m128i, __m128i)      - SSE41
// __m128i _mm_min_epu16(__m128i, __m128i)      - SSE41
// __m128i _mm_min_epu32(__m128i, __m128i)      - SSE41
// __m256d _mm256_min_pd(__m256d, __m256d)      - AVX
// __m256 _mm256_min_ps(__m256, __m256)         - AVX
// __m256i _mm256_min_epi16(__m256i, __m256i)   - AVX2
// __m256i _mm256_min_epi32(__m256i, __m256i)   - AVX2
// __m256i _mm256_min_epi8(__m256i, __m256i)    - AVX2
// __m256i _mm256_min_epu16(__m256i, __m256i)   - AVX2
// __m256i _mm256_min_epu32(__m256i, __m256i)   - AVX2
// __m256i _mm256_min_epu8(__m256i, __m256i)    - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// MAX
// __m64 _m_pmaxsw(__m64, __m64)                - SSE
// __m64 _m_pmaxub(__m64, __m64)                - SSE
// __m128 _mm_max_ps(__m128, __m128)            - SSE
// __m128 _mm_max_ss(__m128, __m128)            - SSE
// __m128i _mm_max_epi16(__m128i, __m128i)      - SSE2
// __m128i _mm_max_epu8(__m128i, __m128i)       - SSE2
// __m128d _mm_max_pd(__m128d, __m128d)         - SSE2
// __m128d _mm_max_sd(__m128d, __m128d)         - SSE2
// __m128i _mm_max_epi32(__m128i, __m128i)      - SSE41
// __m128i _mm_max_epi8 (__m128i, __m128i)      - SSE41
// __m128i _mm_max_epu16(__m128i, __m128i)      - SSE41
// __m128i _mm_max_epu32(__m128i, __m128i)      - SSE41
// __m256d _mm256_max_pd(__m256d, __m256d)      - AVX
// __m256 _mm256_max_ps(__m256, __m256)         - AVX
// __m256i _mm256_max_epi16(__m256i, __m256i)   - AVX2
// __m256i _mm256_max_epi32(__m256i, __m256i)   - AVX2
// __m256i _mm256_max_epi8(__m256i, __m256i)    - AVX2
// __m256i _mm256_max_epu16(__m256i, __m256i)   - AVX2
// __m256i _mm256_max_epu32(__m256i, __m256i)   - AVX2
// __m256i _mm256_max_epu8(__m256i, __m256i)    - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// AND
// __m64 _m_pand(__m64, __m64)                  - MMX
// __m64 _mm_and_si64(__m64, __m64)             - MMX*
// __m128 _mm_and_ps(__m128, __m128)            - SSE
// __m128d _mm_and_pd(__m128d, __m128d)         - SSE2
// __m128i _mm_and_si128(__m128i, __m128i)      - SSE2
// __m256d _mm256_and_pd(__m256d, __m256d)      - AVX
// __m256 _mm256_and_ps(__m256, __m256)         - AVX
// __m256i _mm256_and_si256(__m256i, __m256i)   - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ANDN
// __m64 _m_pandn(__m64, __m64)                     - MMX
// __m64 _mm_andnot_si64(__m64, __m64)              - MMX*
// __m128 _mm_andnot_ps(__m128, __m128)             - SSE
// __m128d _mm_andnot_pd(__m128d, __m128d)          - SSE2
// __m128i _mm_andnot_si128(__m128i, __m128i)       - SSE2
// __m256d _mm256_andnot_pd(__m256d, __m256d)       - AVX
// __m256 _mm256_andnot_ps(__m256, __m256)          - AVX
// __m256i _mm256_andnot_si256(__m256i, __m256i)    - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// OR
// __m64 _m_por(__m64, __m64)                   - MMX
// __m64 _mm_or_si64(__m64, __m64)              - MMX*
// __m128 _mm_or_ps(__m128, __m128)             - SSE
// __m128d _mm_or_pd(__m128d, __m128d)          - SSE2
// __m128i _mm_or_si128(__m128i, __m128i)       - SSE2
// __m256d _mm256_or_pd(__m256d, __m256d)       - AVX
// __m256 _mm256_or_ps(__m256, __m256)          - AVX
// __m256i _mm256_or_si256(__m256i, __m256i)    - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// XOR
// __m64 _m_pxor(__m64, __m64)                  - MMX
// __m64 _mm_xor_si64(__m64, __m64)             - MMX*
// __m128 _mm_xor_ps(__m128, __m128)            - SSE
// __m128d _mm_xor_pd(__m128d, __m128d)         - SSE2
// __m128i _mm_xor_si128(__m128i, __m128i)      - SSE2
// __m256d _mm256_xor_pd(__m256d, __m256d)      - AVX
// __m256 _mm256_xor_ps(__m256, __m256)         - AVX
// __m256i _mm256_xor_si256(__m256i, __m256i)   - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SLL
// __m64 _m_pslld(__m64, __m64)                 - MMX
// __m64 _m_pslldi(__m64, int)                  - MMX
// __m64 _m_psllq(__m64, __m64)                 - MMX
// __m64 _m_psllqi(__m64, int)                  - MMX
// __m64 _m_psllw(__m64, __m64)                 - MMX
// __m64 _m_psllwi(__m64, int)                  - MMX
// __m64 _mm_sll_pi16(__m64, __m64)             - MMX*
// __m64 _mm_sll_pi32(__m64, __m64)             - MMX*
// __m64 _mm_sll_si64(__m64, __m64)             - MMX*
// __m64 _mm_slli_pi16(__m64, int)              - MMX*
// __m64 _mm_slli_pi32(__m64, int)              - MMX*
// __m64 _mm_slli_si64(__m64, int)              - MMX*
// __m128i _mm_sll_epi16(__m128i, __m128i)      - SSE2
// __m128i _mm_sll_epi32(__m128i, __m128i)      - SSE2
// __m128i _mm_sll_epi64(__m128i, __m128i)      - SSE2
// __m128i _mm_slli_epi16(__m128i, int)         - SSE2
// __m128i _mm_slli_epi32(__m128i, int)         - SSE2
// __m128i _mm_slli_epi64(__m128i, int)         - SSE2
// __m128i _mm_slli_si128(__m128i, int)         - SSE2
// __m128i _mm_sllv_epi32(__m128i, __m128i)     - AVX2
// __m128i _mm_sllv_epi64(__m128i, __m128i)     - AVX2
// __m256i _mm256_sll_epi16(__m256i, __m128i)   - AVX2
// __m256i _mm256_sll_epi32(__m256i, __m128i)   - AVX2
// __m256i _mm256_sll_epi64(__m256i, __m128i)   - AVX2
// __m256i _mm256_slli_epi16(__m256i, int)      - AVX2
// __m256i _mm256_slli_epi32(__m256i, int)      - AVX2
// __m256i _mm256_slli_epi64(__m256i, int)      - AVX2
// __m256i _mm256_slli_si256(__m256i, int)      - AVX2
// __m256i _mm256_sllv_epi32(__m256i, __m256i)  - AVX2
// __m256i _mm256_sllv_epi64(__m256i, __m256i)  - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SRA
// __m64 _m_psrad(__m64, __m64)                 - MMX
// __m64 _m_psradi(__m64, int)                  - MMX
// __m64 _m_psraw(__m64, __m64)                 - MMX
// __m64 _m_psrawi(__m64, int)                  - MMX
// __m64 _mm_sra_pi16(__m64, __m64)             - MMX*
// __m64 _mm_sra_pi32(__m64, __m64)             - MMX*
// __m64 _mm_srai_pi16(__m64, int)              - MMX*
// __m64 _mm_srai_pi32(__m64, int)              - MMX*
// __m128i _mm_sra_epi16(__m128i, __m128i)      - SSE2
// __m128i _mm_sra_epi32(__m128i, __m128i)      - SSE2
// __m128i _mm_srai_epi16(__m128i, int)         - SSE2
// __m128i _mm_srai_epi32(__m128i, int)         - SSE2
// __m128i _mm_srav_epi32(__m128i, __m128i)     - AVX2
// __m256i _mm256_sra_epi16(__m256i, __m128i)   - AVX2
// __m256i _mm256_sra_epi32(__m256i, __m128i)   - AVX2
// __m256i _mm256_srai_epi16(__m256i, int)      - AVX2
// __m256i _mm256_srai_epi32(__m256i, int)      - AVX2
// __m256i _mm256_srav_epi32(__m256i, __m256i)  - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SRL
// __m64 _m_psrld(__m64, __m64)                 - MMX
// __m64 _m_psrldi(__m64, int)                  - MMX
// __m64 _m_psrlq(__m64, __m64)                 - MMX
// __m64 _m_psrlqi(__m64, int)                  - MMX
// __m64 _m_psrlw(__m64, __m64)                 - MMX
// __m64 _m_psrlwi(__m64, int)                  - MMX
// __m64 _mm_srl_pi16(__m64, __m64)             - MMX*
// __m64 _mm_srl_pi32(__m64, __m64)             - MMX*
// __m64 _mm_srl_si64(__m64, __m64)             - MMX*
// __m64 _mm_srli_pi16(__m64, int)              - MMX*
// __m64 _mm_srli_pi32(__m64, int)              - MMX*
// __m64 _mm_srli_si64(__m64, int)              - MMX*
// __m128i _mm_srl_epi16(__m128i, __m128i)      - SSE2
// __m128i _mm_srl_epi32(__m128i, __m128i)      - SSE2
// __m128i _mm_srl_epi64(__m128i, __m128i)      - SSE2
// __m128i _mm_srli_epi16(__m128i, int)         - SSE2
// __m128i _mm_srli_epi32(__m128i, int)         - SSE2
// __m128i _mm_srli_epi64(__m128i, int)         - SSE2
// __m128i _mm_srli_si128(__m128i, int)         - SSE2
// __m128i _mm_srlv_epi32(__m128i, __m128i)     - AVX2
// __m128i _mm_srlv_epi64(__m128i, __m128i)     - AVX2
// __m256i _mm256_srl_epi16(__m256i, __m128i)   - AVX2
// __m256i _mm256_srl_epi32(__m256i, __m128i)   - AVX2
// __m256i _mm256_srl_epi64(__m256i, __m128i)   - AVX2
// __m256i _mm256_srli_epi16(__m256i, int)      - AVX2
// __m256i _mm256_srli_epi32(__m256i, int)      - AVX2
// __m256i _mm256_srli_epi64(__m256i, int)      - AVX2
// __m256i _mm256_srli_si256(__m256i, int)      - AVX2
// __m256i _mm256_srlv_epi32(__m256i, __m256i)  - AVX2
// __m256i _mm256_srlv_epi64(__m256i, __m256i)  - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ALIGNR
//// __m128i _mm_alignr_epi8(__m128i, __m128i, int)         - SSSE3
// __m64 _mm_alignr_pi8(__m64, __m64, int)                  - SSSE3
// __m256i _mm256_alignr_epi8(__m256i, __m256i, const int)  - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CRC32
// unsigned int _mm_crc32_u16(unsigned int, unsigned short) - SSE42
// unsigned int _mm_crc32_u32(unsigned int, unsigned int)   - SSE42
// unsigned int _mm_crc32_u8(unsigned int, unsigned char)   - SSE42
//
//
// 
// 
// 
// 
// 
// 
// 
/////////////////////////////////////////////////////////// SSE
// unsigned int _mm_getcsr(void)
// void _mm_prefetch(char*, int)
// void _mm_setcsr(unsigned int)
// 
// 
/////////////////////////////////////////////////////////// SSE2
// void _mm_clflush(void const *)
// void _mm_pause(void)
// 
// 
/////////////////////////////////////////////////////////// SSE3
// void _mm_monitor(void const*, unsigned int, unsigned int)
// void _mm_mwait(unsigned int, unsigned int)
// 
// 
/////////////////////////////////////////////////////////// SSE41
// __m128i _mm_minpos_epu16(__m128i)
// 
// 
/////////////////////////////////////////////////////////// AVX
// void _mm256_zeroall(void)
// void _mm256_zeroupper(void)
// 
// 
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SSE_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SSE_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../Platform.h"

// General define for SSE use in a lot of math code
// comment this line if you want to switch to non-SSE math code
//#define MATH_USE_SSE

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define SSEFn SSEStack::GetInstancePtr()

// Streaming SIMD Types
typedef union __declspec(intrin_type) __declspec(align(8)) _Mixed64 {
    Int8    _i8[8];
    Int16   _i16[4];
    Int32   _i32[2];
    Int64   _i64;
    UInt8   _u8[8];
    UInt16  _u16[4];
    UInt32  _u32[2];
    UInt64  _u64;
    Float   _f32[2];
    Double  _f64;
} Mixed64;
typedef union __declspec(intrin_type) __declspec(align(16)) _Mixed128 {
    Int8    _i8[16];
    Int16   _i16[8];
    Int32   _i32[4];
    Int64   _i64[2];
    UInt8   _u8[16];
    UInt16  _u16[8];
    UInt32  _u32[4];
    UInt64  _u64[2];
    Float   _f32[4];
    Double  _f64[2];
} Mixed128;
typedef union __declspec(intrin_type) __declspec(align(32)) _Mixed256 {
    Int8    _i8[32];
    Int16   _i16[16];
    Int32   _i32[8];
    Int64   _i64[4];
    UInt8   _u8[32];
    UInt16  _u16[16];
    UInt32  _u32[8];
    UInt64  _u64[4];
    Float   _f32[8];
    Double  _f64[4];
} Mixed256;

// Instruction variants
#define _SIMD_PS 0 // Packed format, Single precision
#define _SIMD_SS 1 // Simple format, Single precision
#define _SIMD_PD 2 // Packed format, Double precision
#define _SIMD_SD 3 // Simple format, Double precision

// Stack Indices, st(i) grows with i from top (0) to bottom (7)
#define _SIMD_ST( _i ) ( (m_iTop-1) - (_i) )

/////////////////////////////////////////////////////////////////////////////////
// Intrinsic-wrapping Macros

// _prefix = '_m_' or '_mm_'
#define _SIMD_INTRINSIC_MMX( _prefix, _operation, _format ) \
    _prefix ## _operation ## _format

// _prefix = '_m_' or '_mm_'
#define _SIMD_INTRINSIC_SSE( _prefix, _operation, _format ) \
    _prefix ## _operation ## _format

// _prefix = '_mm_' or '_mm256_'
#define _SIMD_INTRINSIC_AVX( _prefix, _operation, _format ) \
    _prefix ## _operation ## _format
















    // Mask builders for _IShuf & _IBlend
#define _SSE_ISHUF_MASK_PS( _seldest_0, _seldest_1, _selsrc_2, _selsrc_3 ) \
    ( ((_seldest_0) & 0x03) | (((_seldest_1) & 0x03) << 2) | (((_selsrc_2) & 0x03) << 4) | (((_selsrc_3) & 0x03) << 6) )
#define _SSE_ISHUF_MASK_PD( _seldest_0, _selsrc_1 ) \
    ( ((_seldest_0) & 0x01) | (((_selsrc_1) & 0x01) << 1) )

#define _SSE_IBLEND_MASK_PS( _pick_0, _pick_1, _pick_2, _pick_3 ) \
    ( ((_pick_0) & 0x01) | (((_pick_1) & 0x01) << 1) | (((_pick_2) & 0x01) << 2) | (((_pick_3) & 0x01) << 3) )
#define _SSE_IBLEND_MASK_PD( _pick_0, _pick_1 ) \
    ( ((_pick_0) & 0x01) | (((_pick_1) & 0x01) << 1) )

    // Simple helper for stack indices, st(i) grows with i from top (0) to bottom (7).

/////////////////////////////////////////////////////////////////////////////////
// Instruction-wrapping declaration macros
    // 8 elements declarations
#define _SSE_DECLARE_INSTRUCTION_XMM_OPERAND( _funcname, _mnemonic, _operand, ... ) \
    inline Void _fastcall _funcname##_Xmm0( __VA_ARGS__ ) { __asm _mnemonic xmm0, _operand } \
    inline Void _fastcall _funcname##_Xmm1( __VA_ARGS__ ) { __asm _mnemonic xmm1, _operand } \
    inline Void _fastcall _funcname##_Xmm2( __VA_ARGS__ ) { __asm _mnemonic xmm2, _operand } \
    inline Void _fastcall _funcname##_Xmm3( __VA_ARGS__ ) { __asm _mnemonic xmm3, _operand } \
    inline Void _fastcall _funcname##_Xmm4( __VA_ARGS__ ) { __asm _mnemonic xmm4, _operand } \
    inline Void _fastcall _funcname##_Xmm5( __VA_ARGS__ ) { __asm _mnemonic xmm5, _operand } \
    inline Void _fastcall _funcname##_Xmm6( __VA_ARGS__ ) { __asm _mnemonic xmm6, _operand } \
    inline Void _fastcall _funcname##_Xmm7( __VA_ARGS__ ) { __asm _mnemonic xmm7, _operand } \
    typedef Void (_fastcall * _SSE_Functor_##_funcname)( __VA_ARGS__ ); \
    extern _SSE_Functor_##_funcname _SSE_FTable_##_funcname[8]; \
    inline Void _fastcall _funcname( UInt outRegister, __VA_ARGS__ ) { _SSE_FTable_##_funcname[outRegister]( _operand ); }

#define _SSE_DECLARE_INSTRUCTION_OPERAND_XMM( _funcname, _mnemonic, _operand, ... ) \
    inline Void _fastcall _funcname##_Xmm0( __VA_ARGS__ ) { __asm _mnemonic _operand, xmm0 } \
    inline Void _fastcall _funcname##_Xmm1( __VA_ARGS__ ) { __asm _mnemonic _operand, xmm1 } \
    inline Void _fastcall _funcname##_Xmm2( __VA_ARGS__ ) { __asm _mnemonic _operand, xmm2 } \
    inline Void _fastcall _funcname##_Xmm3( __VA_ARGS__ ) { __asm _mnemonic _operand, xmm3 } \
    inline Void _fastcall _funcname##_Xmm4( __VA_ARGS__ ) { __asm _mnemonic _operand, xmm4 } \
    inline Void _fastcall _funcname##_Xmm5( __VA_ARGS__ ) { __asm _mnemonic _operand, xmm5 } \
    inline Void _fastcall _funcname##_Xmm6( __VA_ARGS__ ) { __asm _mnemonic _operand, xmm6 } \
    inline Void _fastcall _funcname##_Xmm7( __VA_ARGS__ ) { __asm _mnemonic _operand, xmm7 } \
    typedef Void (_fastcall * _SSE_Functor_##_funcname)( __VA_ARGS__ ); \
    extern _SSE_Functor_##_funcname _SSE_FTable_##_funcname[8]; \
    inline Void _fastcall _funcname( __VA_ARGS__, UInt inRegister ) { _SSE_FTable_##_funcname[inRegister]( _operand ); }

#define _SSE_DECLARE_INSTRUCTION_XMM_REG32_OPERAND( _funcname, _mnemonic, _register, _operand, ... ) \
    inline Void _fastcall _funcname##_Xmm0( __VA_ARGS__ ) { __asm mov _register, _operand __asm _mnemonic xmm0, _register } \
    inline Void _fastcall _funcname##_Xmm1( __VA_ARGS__ ) { __asm mov _register, _operand __asm _mnemonic xmm1, _register } \
    inline Void _fastcall _funcname##_Xmm2( __VA_ARGS__ ) { __asm mov _register, _operand __asm _mnemonic xmm2, _register } \
    inline Void _fastcall _funcname##_Xmm3( __VA_ARGS__ ) { __asm mov _register, _operand __asm _mnemonic xmm3, _register } \
    inline Void _fastcall _funcname##_Xmm4( __VA_ARGS__ ) { __asm mov _register, _operand __asm _mnemonic xmm4, _register } \
    inline Void _fastcall _funcname##_Xmm5( __VA_ARGS__ ) { __asm mov _register, _operand __asm _mnemonic xmm5, _register } \
    inline Void _fastcall _funcname##_Xmm6( __VA_ARGS__ ) { __asm mov _register, _operand __asm _mnemonic xmm6, _register } \
    inline Void _fastcall _funcname##_Xmm7( __VA_ARGS__ ) { __asm mov _register, _operand __asm _mnemonic xmm7, _register } \
    typedef Void (_fastcall * _SSE_Functor_##_funcname)( __VA_ARGS__ ); \
    extern _SSE_Functor_##_funcname _SSE_FTable_##_funcname[8]; \
    inline Void _fastcall _funcname( UInt outRegister, __VA_ARGS__ ) { _SSE_FTable_##_funcname[outRegister]( _operand ); }

#define _SSE_DECLARE_INSTRUCTION_OPERAND_REG32_XMM( _funcname, _mnemonic, _register, _operand, ... ) \
    inline Void _fastcall _funcname##_Xmm0( __VA_ARGS__ ) { __asm _mnemonic _register, xmm0 __asm mov _operand, _register } \
    inline Void _fastcall _funcname##_Xmm1( __VA_ARGS__ ) { __asm _mnemonic _register, xmm1 __asm mov _operand, _register } \
    inline Void _fastcall _funcname##_Xmm2( __VA_ARGS__ ) { __asm _mnemonic _register, xmm2 __asm mov _operand, _register } \
    inline Void _fastcall _funcname##_Xmm3( __VA_ARGS__ ) { __asm _mnemonic _register, xmm3 __asm mov _operand, _register } \
    inline Void _fastcall _funcname##_Xmm4( __VA_ARGS__ ) { __asm _mnemonic _register, xmm4 __asm mov _operand, _register } \
    inline Void _fastcall _funcname##_Xmm5( __VA_ARGS__ ) { __asm _mnemonic _register, xmm5 __asm mov _operand, _register } \
    inline Void _fastcall _funcname##_Xmm6( __VA_ARGS__ ) { __asm _mnemonic _register, xmm6 __asm mov _operand, _register } \
    inline Void _fastcall _funcname##_Xmm7( __VA_ARGS__ ) { __asm _mnemonic _register, xmm7 __asm mov _operand, _register } \
    typedef Void (_fastcall * _SSE_Functor_##_funcname)(__VA_ARGS__); \
    extern _SSE_Functor_##_funcname _SSE_FTable_##_funcname[8]; \
    inline Void _fastcall _funcname( __VA_ARGS__, UInt inRegister ) { _SSE_FTable_##_funcname[inRegister]( _operand ); }

    // 64 elements declarations
#define _SSE_DECLARE_INSTRUCTION_XMM_XMM_HELPER( _funcname, _mnemonic, _i ) \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm0() { __asm _mnemonic xmm##_i, xmm0 } \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm1() { __asm _mnemonic xmm##_i, xmm1 } \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm2() { __asm _mnemonic xmm##_i, xmm2 } \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm3() { __asm _mnemonic xmm##_i, xmm3 } \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm4() { __asm _mnemonic xmm##_i, xmm4 } \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm5() { __asm _mnemonic xmm##_i, xmm5 } \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm6() { __asm _mnemonic xmm##_i, xmm6 } \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm7() { __asm _mnemonic xmm##_i, xmm7 }

#define _SSE_DECLARE_INSTRUCTION_XMM_XMM( _funcname, _mnemonic ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_HELPER( _funcname, _mnemonic, 0 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_HELPER( _funcname, _mnemonic, 1 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_HELPER( _funcname, _mnemonic, 2 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_HELPER( _funcname, _mnemonic, 3 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_HELPER( _funcname, _mnemonic, 4 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_HELPER( _funcname, _mnemonic, 5 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_HELPER( _funcname, _mnemonic, 6 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_HELPER( _funcname, _mnemonic, 7 ) \
    typedef Void (_fastcall * _SSE_Functor_##_funcname)(); \
    extern _SSE_Functor_##_funcname _SSE_FTable_##_funcname[64]; \
    inline Void _fastcall _funcname( UInt outRegister, UInt inRegister ) { _SSE_FTable_##_funcname[(outRegister << 3) + inRegister](); }

#define _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, _i, _immediate ) \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm0_##_immediate() { __asm _mnemonic xmm##_i, xmm0, _immediate } \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm1_##_immediate() { __asm _mnemonic xmm##_i, xmm1, _immediate } \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm2_##_immediate() { __asm _mnemonic xmm##_i, xmm2, _immediate } \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm3_##_immediate() { __asm _mnemonic xmm##_i, xmm3, _immediate } \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm4_##_immediate() { __asm _mnemonic xmm##_i, xmm4, _immediate } \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm5_##_immediate() { __asm _mnemonic xmm##_i, xmm5, _immediate } \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm6_##_immediate() { __asm _mnemonic xmm##_i, xmm6, _immediate } \
    inline Void _fastcall _funcname##_Xmm##_i##_Xmm7_##_immediate() { __asm _mnemonic xmm##_i, xmm7, _immediate }

#define _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( _funcname, _mnemonic, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 0, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 1, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 2, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 3, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 4, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 5, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 6, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 7, _immediate ) \
    typedef Void (_fastcall * _SSE_Functor_##_funcname)(); \
    extern _SSE_Functor_##_funcname _SSE_FTable_##_funcname[64]; \
    inline Void _fastcall _funcname( UInt outRegister, UInt inRegister ) { _SSE_FTable_##_funcname[(outRegister << 3) + inRegister](); }

    // 16384 elements declarations
#define _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 0, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 1, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 2, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 3, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 4, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 5, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 6, _immediate ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _mnemonic, 7, _immediate )

#define _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, _i ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##0 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##1 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##2 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##3 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##4 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##5 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##6 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##7 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##8 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##9 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##a ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##b ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##c ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##d ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##e ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _mnemonic, 0x##_i##f )

#define _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX( _funcname, _mnemonic ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, 0 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, 1 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, 2 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, 3 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, 4 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, 5 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, 6 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, 7 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, 8 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, 9 ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, a ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, b ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, c ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, d ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, e ) \
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _mnemonic, f ) \
    typedef Void (_fastcall * _SSE_Functor_##_funcname)(); \
    extern _SSE_Functor_##_funcname _SSE_FTable_##_funcname[16384]; \
    inline Void _fastcall _funcname( UInt outRegister, UInt inRegister, UInt inImm8 ) \
        { _SSE_FTable_##_funcname[(inImm8 << 6) + (outRegister << 3) + inRegister](); }

/////////////////////////////////////////////////////////////////////////////////
// Instruction-wrapping initialization macros
    // 8 elements initializations
#define _SSE_INITIALIZE_INSTRUCTION_XMM( _funcname ) \
    _SSE_Functor_##_funcname _SSE_FTable_##_funcname[8] = { \
        _funcname##_Xmm0, _funcname##_Xmm1, _funcname##_Xmm2, _funcname##_Xmm3, \
        _funcname##_Xmm4, _funcname##_Xmm5, _funcname##_Xmm6, _funcname##_Xmm7 \
    }

    // 64 elements initializations
#define _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_HELPER( _funcname, _i ) \
    _funcname##_Xmm##_i##_Xmm0, _funcname##_Xmm##_i##_Xmm1, \
    _funcname##_Xmm##_i##_Xmm2, _funcname##_Xmm##_i##_Xmm3, \
    _funcname##_Xmm##_i##_Xmm4, _funcname##_Xmm##_i##_Xmm5, \
    _funcname##_Xmm##_i##_Xmm6, _funcname##_Xmm##_i##_Xmm7

#define _SSE_INITIALIZE_INSTRUCTION_XMM_XMM( _funcname ) \
    _SSE_Functor_##_funcname _SSE_FTable_##_funcname[64] = { \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_HELPER( _funcname, 0 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_HELPER( _funcname, 1 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_HELPER( _funcname, 2 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_HELPER( _funcname, 3 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_HELPER( _funcname, 4 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_HELPER( _funcname, 5 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_HELPER( _funcname, 6 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_HELPER( _funcname, 7 ) \
    }

#define _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, _i, _immediate ) \
    _funcname##_Xmm##_i##_Xmm0_##_immediate, _funcname##_Xmm##_i##_Xmm1_##_immediate, \
    _funcname##_Xmm##_i##_Xmm2_##_immediate, _funcname##_Xmm##_i##_Xmm3_##_immediate, \
    _funcname##_Xmm##_i##_Xmm4_##_immediate, _funcname##_Xmm##_i##_Xmm5_##_immediate, \
    _funcname##_Xmm##_i##_Xmm6_##_immediate, _funcname##_Xmm##_i##_Xmm7_##_immediate

#define _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8( _funcname, _immediate ) \
    _SSE_Functor_##_funcname _SSE_FTable_##_funcname[64] = { \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 0, _immediate ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 1, _immediate ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 2, _immediate ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 3, _immediate ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 4, _immediate ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 5, _immediate ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 6, _immediate ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 7, _immediate ) \
    }

    // 16384 elements initializations
#define _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, _immediate ) \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 0, _immediate ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 1, _immediate ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 2, _immediate ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 3, _immediate ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 4, _immediate ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 5, _immediate ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 6, _immediate ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8_HELPER( _funcname, 7, _immediate )

#define _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, _i ) \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##0 ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##1 ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##2 ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##3 ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##4 ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##5 ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##6 ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##7 ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##8 ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##8 ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##a ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##b ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##c ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##d ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##e ), \
    _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L1( _funcname, 0x##_i##f )

#define _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX( _funcname ) \
    _SSE_Functor_##_funcname _SSE_FTable_##_funcname[16384] = { \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, 0 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, 1 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, 2 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, 3 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, 4 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, 5 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, 6 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, 7 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, 8 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, 9 ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, a ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, b ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, c ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, d ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, e ), \
        _SSE_INITIALIZE_INSTRUCTION_XMM_XMM_IMM8EX_HELPER_L2( _funcname, f ) \
    }

/////////////////////////////////////////////////////////////////////////////////
// Instruction-wrapping packing macros
#define _SSE_DECLARE_PACK_0( _packsize, _interfacename ) \
    typedef Void (_fastcall * _SSE_Functor_##_interfacename)(); \
    extern _SSE_Functor_##_interfacename _SSE_FTable_##_interfacename[_packsize]; \
    inline Void _fastcall _interfacename( UInt inVariant ) { _SSE_FTable_##_interfacename[inVariant](); }

#define _SSE_DECLARE_PACK_1( _packsize, _interfacename, _operand1, ... ) \
    typedef Void (_fastcall * _SSE_Functor_##_interfacename)( __VA_ARGS__ ); \
    extern _SSE_Functor_##_interfacename _SSE_FTable_##_interfacename[_packsize]; \
    inline Void _fastcall _interfacename( UInt inVariant, __VA_ARGS__ ) { _SSE_FTable_##_interfacename[inVariant](_operand1); }

#define _SSE_DECLARE_PACK_2( _packsize, _interfacename, _operand1, _operand2, ... ) \
    typedef Void (_fastcall * _SSE_Functor_##_interfacename)( __VA_ARGS__ ); \
    extern _SSE_Functor_##_interfacename _SSE_FTable_##_interfacename[_packsize]; \
    inline Void _fastcall _interfacename( UInt inVariant, __VA_ARGS__ ) { _SSE_FTable_##_interfacename[inVariant](_operand1, _operand2); }

#define _SSE_DECLARE_PACK_3( _packsize, _interfacename, _operand1, _operand2, _operand3, ... ) \
    typedef Void (_fastcall * _SSE_Functor_##_interfacename)( __VA_ARGS__ ); \
    extern _SSE_Functor_##_interfacename _SSE_FTable_##_interfacename[_packsize]; \
    inline Void _fastcall _interfacename( UInt inVariant, __VA_ARGS__ ) { _SSE_FTable_##_interfacename[inVariant](_operand1, _operand2, _operand3); }

#define _SSE_INITIALIZE_PACK( _packsize, _interfacename, ... ) \
    _SSE_Functor_##_interfacename _SSE_FTable_##_interfacename[_packsize] = { __VA_ARGS__ }

/////////////////////////////////////////////////////////////////////////////////
// The _SSE instruction set (a class can't accept such a huge load of declarations)
namespace _SSE
{
    // Use only _SSE_I<instr> macros / _I<instr> functions, these are the low level interface
    // Variants are PS, SS, PD, SD for macro versions

        // IGet
#define __SSE_Get_PS( _outMem128, _inRegister )         { __asm movaps _outMem128, xmm##_inRegister }
#define __SSE_Get_SS( _outMem128, _inRegister )         { __asm movss  _outMem128, xmm##_inRegister }
#define __SSE_Get_PD( _outMem128, _inRegister )         { __asm movapd _outMem128, xmm##_inRegister }
#define __SSE_Get_SD( _outMem128, _inRegister )         { __asm movsd  _outMem128, xmm##_inRegister }
#define _SSE_IGet( _variant, _outMem128, _inRegister )  __SSE_Get_##_variant( _outMem128, _inRegister )

    _SSE_DECLARE_INSTRUCTION_OPERAND_XMM( __GetPS, movaps, outMem128, QWord outMem128[2] );
    _SSE_DECLARE_INSTRUCTION_OPERAND_XMM( __GetSS, movss,  outMem128, QWord outMem128[2] );
    _SSE_DECLARE_INSTRUCTION_OPERAND_XMM( __GetPD, movapd, outMem128, QWord outMem128[2] );
    _SSE_DECLARE_INSTRUCTION_OPERAND_XMM( __GetSD, movsd,  outMem128, QWord outMem128[2] );
    _SSE_DECLARE_PACK_2( 4, _IGet, outMem128, inRegister, QWord outMem128[2], UInt inRegister );

        // IGetNT (non-temporal, bypasses all caching procedures)
#define __SSE_GetNT_PS( _outMem128, _inRegister )           { __asm movntps _outMem128, xmm##_inRegister }
#define __SSE_GetNT_PD( _outMem128, _inRegister )           { __asm movntpd _outMem128, xmm##_inRegister }
#define _SSE_IGetNT( _variant, _outMem128, _inRegister )    __SSE_GetNT_##_variant( _outMem128, _inRegister )

    _SSE_DECLARE_INSTRUCTION_OPERAND_XMM( __GetNTPS, movntps, outMem128, QWord outMem128[2] );
    _SSE_DECLARE_INSTRUCTION_OPERAND_XMM( __GetNTPD, movntpd, outMem128, QWord outMem128[2] );
    _SSE_DECLARE_PACK_2( 4, _IGetNT, outMem128, inRegister, QWord outMem128[2], UInt inRegister );

        // ISet
#define __SSE_Set_PS( _outRegister, _inMem128 )         { __asm movaps xmm##_outRegister, _inMem128 }
#define __SSE_Set_SS( _outRegister, _inMem128 )         { __asm movss  xmm##_outRegister, _inMem128 }
#define __SSE_Set_PD( _outRegister, _inMem128 )         { __asm movapd xmm##_outRegister, _inMem128 }
#define __SSE_Set_SD( _outRegister, _inMem128 )         { __asm movsd  xmm##_outRegister, _inMem128 }
#define _SSE_ISet( _variant, _outRegister, _inMem128 )  __SSE_Set_##_variant( _outRegister, _inMem128 )

    _SSE_DECLARE_INSTRUCTION_XMM_OPERAND( __SetPS, movaps, inMem128, QWord inMem128[2] );
    _SSE_DECLARE_INSTRUCTION_XMM_OPERAND( __SetSS, movss, inMem128, QWord inMem128[2] );
    _SSE_DECLARE_INSTRUCTION_XMM_OPERAND( __SetPD, movapd, inMem128, QWord inMem128[2] );
    _SSE_DECLARE_INSTRUCTION_XMM_OPERAND( __SetSD, movsd, inMem128, QWord inMem128[2] );
    _SSE_DECLARE_PACK_2( 4, _ISet, outRegister, inMem128, UInt outRegister, QWord inMem128[2] );

        // ISetNT (non-temporal, bypasses all caching procedures)
#define _SSE_ISetNT( _outRegister, _inMem128 )  { __asm movntdqa xmm##_outRegister, _inMem128 }

    _SSE_DECLARE_INSTRUCTION_XMM_OPERAND( _ISetNT, movntdqa, inMem128, QWord inMem128[2] );

        // IMov
#define __SSE_Mov_PS( _outRegister, _inRegister )           { __asm movaps xmm##_outRegister, xmm##_inRegister }
#define __SSE_Mov_SS( _outRegister, _inRegister )           { __asm movss  xmm##_outRegister, xmm##_inRegister }
#define __SSE_Mov_PD( _outRegister, _inRegister )           { __asm movapd xmm##_outRegister, xmm##_inRegister }
#define __SSE_Mov_SD( _outRegister, _inRegister )           { __asm movsd  xmm##_outRegister, xmm##_inRegister }
#define _SSE_IMov( _variant, _outRegister, _inRegister )    __SSE_Mov_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MovPS, movaps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MovSS, movss );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MovPD, movapd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MovSD, movsd );
    _SSE_DECLARE_PACK_2( 4, _IMov, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IMovHL, IMovLH
#define _SSE_IMovHL( _outRegister, _inRegister )    { __asm movhlps xmm##_outRegister, xmm##_inRegister }
#define _SSE_IMovLH( _outRegister, _inRegister )    { __asm movlhps xmm##_outRegister, xmm##_inRegister }

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( _IMovHL, movhlps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( _IMovLH, movlhps );

        // IMovDupH, IMovDupL
#define __SSE_MovDupH_PS( _outRegister, _inRegister )           { __asm movshdup xmm##_outRegister, xmm##_inRegister }
#define __SSE_MovDupH_SS( _outRegister, _inRegister )           { __asm movshdup xmm##_outRegister, xmm##_inRegister }
#define __SSE_MovDupH_PD( _outRegister, _inRegister )           { __asm movddup  xmm##_outRegister, xmm##_inRegister }
#define __SSE_MovDupH_SD( _outRegister, _inRegister )           { __asm movddup  xmm##_outRegister, xmm##_inRegister }
#define _SSE_IMovDupH( _variant, _outRegister, _inRegister )    __SSE_MovDupH_##_variant( _outRegister, _inRegister )

#define __SSE_MovDupL_PS( _outRegister, _inRegister )           { __asm movsldup xmm##_outRegister, xmm##_inRegister }
#define __SSE_MovDupL_SS( _outRegister, _inRegister )           { __asm movsldup xmm##_outRegister, xmm##_inRegister }
#define __SSE_MovDupL_PD( _outRegister, _inRegister )           { __asm movddup  xmm##_outRegister, xmm##_inRegister }
#define __SSE_MovDupL_SD( _outRegister, _inRegister )           { __asm movddup  xmm##_outRegister, xmm##_inRegister }
#define _SSE_IMovDupL( _variant, _outRegister, _inRegister )    __SSE_MovDupL_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MovDupHS, movshdup );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MovDupLS, movsldup );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MovDupD, movddup );
    _SSE_DECLARE_PACK_2( 4, _IMovDupH, outRegister, inRegister, UInt outRegister, UInt inRegister );
    _SSE_DECLARE_PACK_2( 4, _IMovDupL, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IUnpckH
#define __SSE_UnpckH_PS( _outRegister, _inRegister )        { __asm unpckhps xmm##_outRegister, xmm##_inRegister }
#define __SSE_UnpckH_PD( _outRegister, _inRegister )        { __asm unpckhpd xmm##_outRegister, xmm##_inRegister }
#define _SSE_IUnpckH( _variant, _outRegister, _inRegister ) __SSE_UnpckH_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __UnpckHPS, unpckhps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __UnpckHPD, unpckhpd );
    _SSE_DECLARE_PACK_2( 4, _IUnpckH, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IUnpckL
#define __SSE_UnpckL_PS( _outRegister, _inRegister )        { __asm unpcklps xmm##_outRegister, xmm##_inRegister }
#define __SSE_UnpckL_PD( _outRegister, _inRegister )        { __asm unpcklpd xmm##_outRegister, xmm##_inRegister }
#define _SSE_IUnpckL( _variant, _outRegister, _inRegister ) __SSE_UnpckL_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __UnpckLPS, unpcklps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __UnpckLPD, unpcklpd );
    _SSE_DECLARE_PACK_2( 4, _IUnpckL, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IShuf
#define __SSE_Shuf_PS( _outRegister, _inRegister, _inMask8 )        { __asm shufps xmm##_outRegister, xmm##_inRegister, _inMask8 }
#define __SSE_Shuf_PD( _outRegister, _inRegister, _inMask8 )        { __asm shufpd xmm##_outRegister, xmm##_inRegister, _inMask8 }
#define _SSE_IShuf( _variant, _outRegister, _inRegister, _inMask8 ) __SSE_Shuf_##_variant( _outRegister, _inRegister, _inMask8 )

    // Uncomment this if you dare ...
    //_SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX( __ShufPS, shufps );
    //_SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX( __ShufPD, shufpd );
    //_SSE_DECLARE_PACK_3( 4, _IShuf, outRegister, inRegister, inMask8, UInt outRegister, UInt inRegister, Byte inMask8 );

    //_SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX( _IShufPS, shufps );

    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __ShufPD00, shufpd, 0x00 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __ShufPD01, shufpd, 0x01 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __ShufPD10, shufpd, 0x02 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __ShufPD11, shufpd, 0x03 );
    _SSE_DECLARE_PACK_2( 4, _IShufPD, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IBlend
#define __SSE_Blend_PS( _outRegister, _inRegister, _inMask8 )           { __asm blendps xmm##_outRegister, xmm##_inRegister, _inMask8 }
#define __SSE_Blend_PD( _outRegister, _inRegister, _inMask8 )           { __asm blendpd xmm##_outRegister, xmm##_inRegister, _inMask8 }
#define _SSE_IBlend( _variant, _outRegister, _inRegister, _inMask8 )    __SSE_Blend_##_variant( _outRegister, _inRegister, _inMask8 )

    // Uncomment this if you dare ...
    //_SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX( __BlendPS, blendps );
    //_SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8EX( __BlendPD, blendpd );
    //_SSE_DECLARE_PACK_3( 4, _IBlend, outRegister, inRegister, inMask8, UInt outRegister, UInt inRegister, Byte inMask8 );

    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS0000, blendps, 0x00 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS0001, blendps, 0x01 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS0010, blendps, 0x02 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS0011, blendps, 0x03 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS0100, blendps, 0x04 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS0101, blendps, 0x05 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS0110, blendps, 0x06 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS0111, blendps, 0x07 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS1000, blendps, 0x08 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS1001, blendps, 0x09 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS1010, blendps, 0x0a );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS1011, blendps, 0x0b );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS1100, blendps, 0x0c );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS1101, blendps, 0x0d );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS1110, blendps, 0x0e );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPS1111, blendps, 0x0f );
    _SSE_DECLARE_PACK_2( 16, _IBlendPS, outRegister, inRegister, UInt outRegister, UInt inRegister );

    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPD00, blendpd, 0x00 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPD01, blendpd, 0x01 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPD10, blendpd, 0x02 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __BlendPD11, blendpd, 0x03 );
    _SSE_DECLARE_PACK_2( 4, _IBlendPD, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // ICmpE
#define __SSE_CmpE_PS( _outRegister, _inRegister )          { __asm cmpeqps xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpE_SS( _outRegister, _inRegister )          { __asm cmpeqss xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpE_PD( _outRegister, _inRegister )          { __asm cmpeqpd xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpE_SD( _outRegister, _inRegister )          { __asm cmpeqsd xmm##_outRegister, xmm##_inRegister }
#define _SSE_ICmpE( _variant, _outRegister, _inRegister )   __SSE_CmpE_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpEPS, cmpeqps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpESS, cmpeqss );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpEPD, cmpeqpd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpESD, cmpeqsd );
    _SSE_DECLARE_PACK_2( 4, _ICmpE, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // ICmpNE
#define __SSE_CmpNE_PS( _outRegister, _inRegister )         { __asm cmpneqps xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpNE_SS( _outRegister, _inRegister )         { __asm cmpneqss xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpNE_PD( _outRegister, _inRegister )         { __asm cmpneqpd xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpNE_SD( _outRegister, _inRegister )         { __asm cmpneqsd xmm##_outRegister, xmm##_inRegister }
#define _SSE_ICmpNE( _variant, _outRegister, _inRegister )  __SSE_CmpNE_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpNEPS, cmpneqps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpNESS, cmpneqss );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpNEPD, cmpneqpd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpNESD, cmpneqsd );
    _SSE_DECLARE_PACK_2( 4, _ICmpNE, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // ICmpL
#define __SSE_CmpL_PS( _outRegister, _inRegister )          { __asm cmpltps xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpL_SS( _outRegister, _inRegister )          { __asm cmpltss xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpL_PD( _outRegister, _inRegister )          { __asm cmpltpd xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpL_SD( _outRegister, _inRegister )          { __asm cmpltsd xmm##_outRegister, xmm##_inRegister }
#define _SSE_ICmpL( _variant, _outRegister, _inRegister )   __SSE_CmpL_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpLPS, cmpltps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpLSS, cmpltss );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpLPD, cmpltpd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpLSD, cmpltsd );
    _SSE_DECLARE_PACK_2( 4, _ICmpL, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // ICmpLE
#define __SSE_CmpLE_PS( _outRegister, _inRegister )         { __asm cmpleps xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpLE_SS( _outRegister, _inRegister )         { __asm cmpless xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpLE_PD( _outRegister, _inRegister )         { __asm cmplepd xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpLE_SD( _outRegister, _inRegister )         { __asm cmplesd xmm##_outRegister, xmm##_inRegister }
#define _SSE_ICmpLE( _variant, _outRegister, _inRegister )  __SSE_CmpLE_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpLEPS, cmpleps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpLESS, cmpless );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpLEPD, cmplepd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpLESD, cmplesd );
    _SSE_DECLARE_PACK_2( 4, _ICmpLE, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // ICmpNL
#define __SSE_CmpNL_PS( _outRegister, _inRegister )         { __asm cmpnltps xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpNL_SS( _outRegister, _inRegister )         { __asm cmpnltss xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpNL_PD( _outRegister, _inRegister )         { __asm cmpnltpd xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpNL_SD( _outRegister, _inRegister )         { __asm cmpnltsd xmm##_outRegister, xmm##_inRegister }
#define _SSE_ICmpNL( _variant, _outRegister, _inRegister )  __SSE_CmpNL_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpNLPS, cmpnltps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpNLSS, cmpnltss );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpNLPD, cmpnltpd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpNLSD, cmpnltsd );
    _SSE_DECLARE_PACK_2( 4, _ICmpNL, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // ICmpNLE
#define __SSE_CmpNLE_PS( _outRegister, _inRegister )        { __asm cmpnleps xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpNLE_SS( _outRegister, _inRegister )        { __asm cmpnless xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpNLE_PD( _outRegister, _inRegister )        { __asm cmpnlepd xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpNLE_SD( _outRegister, _inRegister )        { __asm cmpnlesd xmm##_outRegister, xmm##_inRegister }
#define _SSE_ICmpNLE( _variant, _outRegister, _inRegister ) __SSE_CmpNLE_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpNLEPS, cmpnleps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpNLESS, cmpnless );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpNLEPD, cmpnlepd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpNLESD, cmpnlesd );
    _SSE_DECLARE_PACK_2( 4, _ICmpNLE, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // ICmpO
#define __SSE_CmpO_PS( _outRegister, _inRegister )          { __asm cmpordps xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpO_SS( _outRegister, _inRegister )          { __asm cmpordss xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpO_PD( _outRegister, _inRegister )          { __asm cmpordpd xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpO_SD( _outRegister, _inRegister )          { __asm cmpordsd xmm##_outRegister, xmm##_inRegister }
#define _SSE_ICmpO( _variant, _outRegister, _inRegister )   __SSE_CmpO_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpOPS, cmpordps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpOSS, cmpordss );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpOPD, cmpordpd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpOSD, cmpordsd );
    _SSE_DECLARE_PACK_2( 4, _ICmpO, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // ICmpU
#define __SSE_CmpU_PS( _outRegister, _inRegister )          { __asm cmpunordps xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpU_SS( _outRegister, _inRegister )          { __asm cmpunordss xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpU_PD( _outRegister, _inRegister )          { __asm cmpunordpd xmm##_outRegister, xmm##_inRegister }
#define __SSE_CmpU_SD( _outRegister, _inRegister )          { __asm cmpunordsd xmm##_outRegister, xmm##_inRegister }
#define _SSE_ICmpU( _variant, _outRegister, _inRegister )   __SSE_CmpU_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpUPS, cmpunordps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpUSS, cmpunordss );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpUPD, cmpunordpd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __CmpUSD, cmpunordsd );
    _SSE_DECLARE_PACK_2( 4, _ICmpU, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IMin
#define __SSE_Min_PS( _outRegister, _inRegister )           { __asm minps xmm##_outRegister, xmm##_inRegister }
#define __SSE_Min_SS( _outRegister, _inRegister )           { __asm minss xmm##_outRegister, xmm##_inRegister }
#define __SSE_Min_PD( _outRegister, _inRegister )           { __asm minpd xmm##_outRegister, xmm##_inRegister }
#define __SSE_Min_SD( _outRegister, _inRegister )           { __asm minsd xmm##_outRegister, xmm##_inRegister }
#define _SSE_IMin( _variant, _outRegister, _inRegister )    __SSE_Min_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MinPS, minps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MinSS, minss );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MinPD, minpd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MinSD, minsd );
    _SSE_DECLARE_PACK_2( 4, _IMin, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IMax
#define __SSE_Max_PS( _outRegister, _inRegister )           { __asm maxps xmm##_outRegister, xmm##_inRegister }
#define __SSE_Max_SS( _outRegister, _inRegister )           { __asm maxss xmm##_outRegister, xmm##_inRegister }
#define __SSE_Max_PD( _outRegister, _inRegister )           { __asm maxpd xmm##_outRegister, xmm##_inRegister }
#define __SSE_Max_SD( _outRegister, _inRegister )           { __asm maxsd xmm##_outRegister, xmm##_inRegister }
#define _SSE_IMax( _variant, _outRegister, _inRegister )    __SSE_Max_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MaxPS, maxps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MaxSS, maxss );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MaxPD, maxpd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MaxSD, maxsd );
    _SSE_DECLARE_PACK_2( 4, _IMax, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IMask
#define __SSE_Mask_PS( _outMask32, _inRegister )        { __asm movmskps eax, xmm##_inRegister __asm mov _outMask32, eax }
#define __SSE_Mask_PD( _outMask32, _inRegister )        { __asm movmskpd eax, xmm##_inRegister __asm mov _outMask32, eax }
#define _SSE_IMask( _variant, _outMask32, _inRegister ) __SSE_Mask_##_variant( _outMask32, _inRegister )

    _SSE_DECLARE_INSTRUCTION_OPERAND_REG32_XMM( __MaskPS, movmskps, eax, outMask32, DWord & outMask32 );
    _SSE_DECLARE_INSTRUCTION_OPERAND_REG32_XMM( __MaskPD, movmskpd, eax, outMask32, DWord & outMask32 );
    _SSE_DECLARE_PACK_2( 4, _IMask, outMask32, inRegister, DWord & outMask32, UInt inRegister );

        // IRound
#define __SSE_Round_PS( _outRegister, _inRegister )         { __asm roundps xmm##_outRegister, xmm##_inRegister, 0x00 }
#define __SSE_Round_SS( _outRegister, _inRegister )         { __asm roundss xmm##_outRegister, xmm##_inRegister, 0x00 }
#define __SSE_Round_PD( _outRegister, _inRegister )         { __asm roundpd xmm##_outRegister, xmm##_inRegister, 0x00 }
#define __SSE_Round_SD( _outRegister, _inRegister )         { __asm roundsd xmm##_outRegister, xmm##_inRegister, 0x00 }
#define _SSE_IRound( _variant, _outRegister, _inRegister )  __SSE_Round_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __RoundPS, roundps, 0x00 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __RoundSS, roundss, 0x00 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __RoundPD, roundpd, 0x00 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __RoundSD, roundsd, 0x00 );
    _SSE_DECLARE_PACK_2( 4, _IRound, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IFloor
#define __SSE_Floor_PS( _outRegister, _inRegister )         { __asm roundps xmm##_outRegister, xmm##_inRegister, 0x01 }
#define __SSE_Floor_SS( _outRegister, _inRegister )         { __asm roundss xmm##_outRegister, xmm##_inRegister, 0x01 }
#define __SSE_Floor_PD( _outRegister, _inRegister )         { __asm roundpd xmm##_outRegister, xmm##_inRegister, 0x01 }
#define __SSE_Floor_SD( _outRegister, _inRegister )         { __asm roundsd xmm##_outRegister, xmm##_inRegister, 0x01 }
#define _SSE_IFloor( _variant, _outRegister, _inRegister )  __SSE_Floor_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __FloorPS, roundps, 0x01 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __FloorSS, roundss, 0x01 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __FloorPD, roundpd, 0x01 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __FloorSD, roundsd, 0x01 );
    _SSE_DECLARE_PACK_2( 4, _IFloor, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // ICeil
#define __SSE_Ceil_PS( _outRegister, _inRegister )          { __asm roundps xmm##_outRegister, xmm##_inRegister, 0x02 }
#define __SSE_Ceil_SS( _outRegister, _inRegister )          { __asm roundss xmm##_outRegister, xmm##_inRegister, 0x02 }
#define __SSE_Ceil_PD( _outRegister, _inRegister )          { __asm roundpd xmm##_outRegister, xmm##_inRegister, 0x02 }
#define __SSE_Ceil_SD( _outRegister, _inRegister )          { __asm roundsd xmm##_outRegister, xmm##_inRegister, 0x02 }
#define _SSE_ICeil( _variant, _outRegister, _inRegister )   __SSE_Ceil_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __CeilPS, roundps, 0x02 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __CeilSS, roundss, 0x02 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __CeilPD, roundpd, 0x02 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __CeilSD, roundsd, 0x02 );
    _SSE_DECLARE_PACK_2( 4, _ICeil, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // ITruncate
#define __SSE_Truncate_PS( _outRegister, _inRegister )          { __asm roundps xmm##_outRegister, xmm##_inRegister, 0x03 }
#define __SSE_Truncate_SS( _outRegister, _inRegister )          { __asm roundss xmm##_outRegister, xmm##_inRegister, 0x03 }
#define __SSE_Truncate_PD( _outRegister, _inRegister )          { __asm roundpd xmm##_outRegister, xmm##_inRegister, 0x03 }
#define __SSE_Truncate_SD( _outRegister, _inRegister )          { __asm roundsd xmm##_outRegister, xmm##_inRegister, 0x03 }
#define _SSE_ITruncate( _variant, _outRegister, _inRegister )   __SSE_Truncate_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __TruncatePS, roundps, 0x03 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __TruncateSS, roundss, 0x03 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __TruncatePD, roundpd, 0x03 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __TruncateSD, roundsd, 0x03 );
    _SSE_DECLARE_PACK_2( 4, _ITruncate, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IAdd
#define __SSE_Add_PS( _outRegister, _inRegister )           { __asm addps xmm##_outRegister, xmm##_inRegister }
#define __SSE_Add_SS( _outRegister, _inRegister )           { __asm addss xmm##_outRegister, xmm##_inRegister }
#define __SSE_Add_PD( _outRegister, _inRegister )           { __asm addpd xmm##_outRegister, xmm##_inRegister }
#define __SSE_Add_SD( _outRegister, _inRegister )           { __asm addsd xmm##_outRegister, xmm##_inRegister }
#define _SSE_IAdd( _variant, _outRegister, _inRegister )    __SSE_Add_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __AddPS, addps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __AddSS, addss );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __AddPD, addpd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __AddSD, addsd );
    _SSE_DECLARE_PACK_2( 4, _IAdd, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // ISub
#define __SSE_Sub_PS( _outRegister, _inRegister )           { __asm subps xmm##_outRegister, xmm##_inRegister }
#define __SSE_Sub_SS( _outRegister, _inRegister )           { __asm subss xmm##_outRegister, xmm##_inRegister }
#define __SSE_Sub_PD( _outRegister, _inRegister )           { __asm subpd xmm##_outRegister, xmm##_inRegister }
#define __SSE_Sub_SD( _outRegister, _inRegister )           { __asm subsd xmm##_outRegister, xmm##_inRegister }
#define _SSE_ISub( _variant, _outRegister, _inRegister )    __SSE_Sub_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __SubPS, subps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __SubSS, subss );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __SubPD, subpd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __SubSD, subsd );
    _SSE_DECLARE_PACK_2( 4, _ISub, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IMul
#define __SSE_Mul_PS( _outRegister, _inRegister )           { __asm mulps xmm##_outRegister, xmm##_inRegister }
#define __SSE_Mul_SS( _outRegister, _inRegister )           { __asm mulss xmm##_outRegister, xmm##_inRegister }
#define __SSE_Mul_PD( _outRegister, _inRegister )           { __asm mulpd xmm##_outRegister, xmm##_inRegister }
#define __SSE_Mul_SD( _outRegister, _inRegister )           { __asm mulsd xmm##_outRegister, xmm##_inRegister }
#define _SSE_IMul( _variant, _outRegister, _inRegister )    __SSE_Mul_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MulPS, mulps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MulSS, mulss );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MulPD, mulpd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __MulSD, mulsd );
    _SSE_DECLARE_PACK_2( 4, _IMul, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IDiv
#define __SSE_Div_PS( _outRegister, _inRegister )           { __asm divps xmm##_outRegister, xmm##_inRegister }
#define __SSE_Div_SS( _outRegister, _inRegister )           { __asm divss xmm##_outRegister, xmm##_inRegister }
#define __SSE_Div_PD( _outRegister, _inRegister )           { __asm divpd xmm##_outRegister, xmm##_inRegister }
#define __SSE_Div_SD( _outRegister, _inRegister )           { __asm divsd xmm##_outRegister, xmm##_inRegister }
#define _SSE_IDiv( _variant, _outRegister, _inRegister )    __SSE_Div_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __DivPS, divps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __DivSS, divss );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __DivPD, divpd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __DivSD, divsd );
    _SSE_DECLARE_PACK_2( 4, _IDiv, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IAddSub
#define __SSE_AddSub_PS( _outRegister, _inRegister )        { __asm addsubps xmm##_outRegister, xmm##_inRegister }
#define __SSE_AddSub_PD( _outRegister, _inRegister )        { __asm addsubpd xmm##_outRegister, xmm##_inRegister }
#define _SSE_IAddSub( _variant, _outRegister, _inRegister ) __SSE_AddSub_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __AddSubPS, addsubps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __AddSubPD, addsubpd );
    _SSE_DECLARE_PACK_2( 4, _IAddSub, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IHAdd
#define __SSE_HAdd_PS( _outRegister, _inRegister )          { __asm haddps xmm##_outRegister, xmm##_inRegister }
#define __SSE_HAdd_PD( _outRegister, _inRegister )          { __asm haddpd xmm##_outRegister, xmm##_inRegister }
#define _SSE_IHAdd( _variant, _outRegister, _inRegister )   __SSE_HAdd_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __HAddPS, haddps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __HAddPD, haddpd );
    _SSE_DECLARE_PACK_2( 4, _IHAdd, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IHSub
#define __SSE_HSub_PS( _outRegister, _inRegister )          { __asm hsubps xmm##_outRegister, xmm##_inRegister }
#define __SSE_HSub_PD( _outRegister, _inRegister )          { __asm hsubpd xmm##_outRegister, xmm##_inRegister }
#define _SSE_IHSub( _variant, _outRegister, _inRegister )   __SSE_HSub_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __HSubPS, hsubps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __HSubPD, hsubpd );
    _SSE_DECLARE_PACK_2( 4, _IHSub, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IDot2
#define __SSE_Dot2_PS( _outRegister, _inRegister )          { __asm dpps xmm##_outRegister, xmm##_inRegister, 0x3f }
#define __SSE_Dot2_SS( _outRegister, _inRegister )          { __asm dpps xmm##_outRegister, xmm##_inRegister, 0x31 }
#define __SSE_Dot2_PD( _outRegister, _inRegister )          { __asm dppd xmm##_outRegister, xmm##_inRegister, 0x33 }
#define __SSE_Dot2_SD( _outRegister, _inRegister )          { __asm dppd xmm##_outRegister, xmm##_inRegister, 0x31 }
#define _SSE_IDot2( _variant, _outRegister, _inRegister )   __SSE_Dot2_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __Dot2PS, dpps, 0x3f );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __Dot2SS, dpps, 0x31 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __Dot2PD, dppd, 0x33 );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __Dot2SD, dppd, 0x31 );
    _SSE_DECLARE_PACK_2( 4, _IDot2, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IDot3
#define __SSE_Dot3_PS( _outRegister, _inRegister )          { __asm dpps xmm##_outRegister, xmm##_inRegister, 0x7f }
#define __SSE_Dot3_SS( _outRegister, _inRegister )          { __asm dpps xmm##_outRegister, xmm##_inRegister, 0x71 }
#define _SSE_IDot3( _variant, _outRegister, _inRegister )   __SSE_Dot3_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __Dot3PS, dpps, 0x7f );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __Dot3SS, dpps, 0x71 );
    _SSE_DECLARE_PACK_2( 4, _IDot3, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IDot4
#define __SSE_Dot4_PS( _outRegister, _inRegister )          { __asm dpps xmm##_outRegister, xmm##_inRegister, 0xff }
#define __SSE_Dot4_SS( _outRegister, _inRegister )          { __asm dpps xmm##_outRegister, xmm##_inRegister, 0xf1 }
#define _SSE_IDot4( _variant, _outRegister, _inRegister )   __SSE_Dot4_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __Dot4PS, dpps, 0xff );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM_IMM8( __Dot4SS, dpps, 0xf1 );
    _SSE_DECLARE_PACK_2( 4, _IDot4, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IInv
#define __SSE_Inv_PS( _outRegister, _inRegister )           { __asm rcpps xmm##_outRegister, xmm##_inRegister }
#define __SSE_Inv_SS( _outRegister, _inRegister )           { __asm rcpss xmm##_outRegister, xmm##_inRegister }
#define _SSE_IInv( _variant, _outRegister, _inRegister )    __SSE_Inv_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __InvPS, rcpps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __InvSS, rcpss );
    _SSE_DECLARE_PACK_2( 4, _IInv, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // ISqrt
#define __SSE_Sqrt_PS( _outRegister, _inRegister )          { __asm sqrtps xmm##_outRegister, xmm##_inRegister }
#define __SSE_Sqrt_SS( _outRegister, _inRegister )          { __asm sqrtss xmm##_outRegister, xmm##_inRegister }
#define __SSE_Sqrt_PD( _outRegister, _inRegister )          { __asm sqrtpd xmm##_outRegister, xmm##_inRegister }
#define __SSE_Sqrt_SD( _outRegister, _inRegister )          { __asm sqrtsd xmm##_outRegister, xmm##_inRegister }
#define _SSE_ISqrt( _variant, _outRegister, _inRegister )   __SSE_Sqrt_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __SqrtPS, sqrtps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __SqrtSS, sqrtss );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __SqrtPD, sqrtpd );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __SqrtSD, sqrtsd );
    _SSE_DECLARE_PACK_2( 4, _ISqrt, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // IInvSqrt
#define __SSE_InvSqrt_PS( _outRegister, _inRegister )           { __asm rsqrtps xmm##_outRegister, xmm##_inRegister }
#define __SSE_InvSqrt_SS( _outRegister, _inRegister )           { __asm rsqrtss xmm##_outRegister, xmm##_inRegister }
#define _SSE_IInvSqrt( _variant, _outRegister, _inRegister )    __SSE_InvSqrt_##_variant( _outRegister, _inRegister )

    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __InvSqrtPS, rsqrtps );
    _SSE_DECLARE_INSTRUCTION_XMM_XMM( __InvSqrtSS, rsqrtss );
    _SSE_DECLARE_PACK_2( 4, _IInvSqrt, outRegister, inRegister, UInt outRegister, UInt inRegister );

        // string operations
    /////////////////////////////////////////////////////////////////////
}

/////////////////////////////////////////////////////////////////////////////////
// The SSEStack class
class SSEStack
{
    // Discrete singleton interface
public:
    inline static SSEStack * GetInstancePtr();

private:
    SSEStack();
    ~SSEStack();

public:
    inline Void Push( Float fValue0 );
    inline Void Push( Float fValue0, Float fValue1 );
    inline Void Push( Float fValue0, Float fValue1, Float fValue2 );
    inline Void Push( Float fValue0, Float fValue1, Float fValue2, Float fValue3 );
    inline Void Push( Float * fValues );

    inline Void Push( Double fValue0 );
    inline Void Push( Double fValue0, Double fValue1 );
    inline Void Push( Double * fValues );

    inline Void PopF();
    inline Void Pop( Float & outValue0 );
    inline Void Pop( Float & outValue0, Float & outValue1 );
    inline Void Pop( Float & outValue0, Float & outValue1, Float & outValue2 );
    inline Void Pop( Float & outValue0, Float & outValue1, Float & outValue2, Float & outValue3 );
    inline Void Pop( Float * outValues );

    inline Void PopD();
    inline Void Pop( Double & outValue0 );
    inline Void Pop( Double & outValue0, Double & outValue1 );
    inline Void Pop( Double * outValues );

    inline Void Set( UInt iStackIndex, Float fValue0 );
    inline Void Set( UInt iStackIndex, Float fValue0, Float fValue1 );
    inline Void Set( UInt iStackIndex, Float fValue0, Float fValue1, Float fValue2 );
    inline Void Set( UInt iStackIndex, Float fValue0, Float fValue1, Float fValue2, Float fValue3 );
    inline Void Set( UInt iStackIndex, Float * fValues );

    inline Void Set( UInt iStackIndex, Double fValue0 );
    inline Void Set( UInt iStackIndex, Double fValue0, Double fValue1 );
    inline Void Set( UInt iStackIndex, Double * fValues );

    inline Void Get( UInt iStackIndex, Float & outValue0 );
    inline Void Get( UInt iStackIndex, Float & outValue0, Float & outValue1 );
    inline Void Get( UInt iStackIndex, Float & outValue0, Float & outValue1, Float & outValue2 );
    inline Void Get( UInt iStackIndex, Float & outValue0, Float & outValue1, Float & outValue2, Float & outValue3 );
    inline Void Get( UInt iStackIndex, Float * outValues );

    inline Void Get( UInt iStackIndex, Double & outValue0 );
    inline Void Get( UInt iStackIndex, Double & outValue0, Double & outValue1 );
    inline Void Get( UInt iStackIndex, Double * outValues );

    inline Void PushF( UInt iSrcIndex ); // push, then st(0) = st(i)
    inline Void PushD( UInt iSrcIndex );
    inline Void PopF( UInt iDestIndex ); // st(i) = st(0), and pop
    inline Void PopD( UInt iDestIndex );
    inline Void MovF( UInt iDestIndex, UInt iSrcIndex ); // st(dst) = st(src)
    inline Void MovD( UInt iDestIndex, UInt iSrcIndex );

    inline Void AddF(); // st(1) = st(1) + st(0)
    inline Void AddD();
    inline Void SubF(); // st(1) = st(1) - st(0)
    inline Void SubD();
    inline Void MulF(); // st(1) = st(1) * st(0)
    inline Void MulD();
    inline Void DivF(); // st(1) = st(1) / st(0)
    inline Void DivD();

    inline Void AddF( UInt iDestIndex ); // st(i) = st(i) + st(0)
    inline Void AddD( UInt iDestIndex );
    inline Void SubF( UInt iDestIndex ); // st(i) = st(i) - st(0)
    inline Void SubD( UInt iDestIndex );
    inline Void MulF( UInt iDestIndex ); // st(i) = st(i) * st(0)
    inline Void MulD( UInt iDestIndex );
    inline Void DivF( UInt iDestIndex ); // st(i) = st(i) / st(0)
    inline Void DivD( UInt iDestIndex );

    inline Void AddF( UInt iDestIndex, UInt iSrcIndex ); // st(dst) = st(dst) + st(src)
    inline Void AddD( UInt iDestIndex, UInt iSrcIndex );
    inline Void SubF( UInt iDestIndex, UInt iSrcIndex ); // st(dst) = st(dst) - st(src)
    inline Void SubD( UInt iDestIndex, UInt iSrcIndex );
    inline Void MulF( UInt iDestIndex, UInt iSrcIndex ); // st(dst) = st(dst) * st(src)
    inline Void MulD( UInt iDestIndex, UInt iSrcIndex );
    inline Void DivF( UInt iDestIndex, UInt iSrcIndex ); // st(dst) = st(dst) / st(src)
    inline Void DivD( UInt iDestIndex, UInt iSrcIndex );

    // NOTE: SubR*/DivR* need st(0)/st(src) to be saved and will use st(-1) for this purpose,
    // if stack is full that's a crash ! Be aware of this fact that should never be much
    // of a problem ...
    inline Void SubRF(); // st(1) = st(0) - st(1)
    inline Void SubRD();
    inline Void DivRF(); // st(1) = st(0) / st(1)
    inline Void DivRD();

    inline Void SubRF( UInt iDestIndex ); // st(i) = st(0) - st(i)
    inline Void SubRD( UInt iDestIndex );
    inline Void DivRF( UInt iDestIndex ); // st(i) = st(0) / st(i)
    inline Void DivRD( UInt iDestIndex );

    inline Void SubRF( UInt iDestIndex, UInt iSrcIndex ); // st(dst) = st(src) - st(dst)
    inline Void SubRD( UInt iDestIndex, UInt iSrcIndex );
    inline Void DivRF( UInt iDestIndex, UInt iSrcIndex ); // st(dst) = st(src) / st(dst)
    inline Void DivRD( UInt iDestIndex, UInt iSrcIndex );

    inline Void AddPF(); // st(1) = st(1) + st(0), and pop
    inline Void AddPD();
    inline Void SubPF(); // st(1) = st(1) - st(0), and pop
    inline Void SubPD();
    inline Void MulPF(); // st(1) = st(1) * st(0), and pop
    inline Void MulPD();
    inline Void DivPF(); // st(1) = st(1) / st(0), and pop
    inline Void DivPD();

    inline Void AddPF( UInt iDestIndex ); // st(i) = st(i) + st(0), and pop
    inline Void AddPD( UInt iDestIndex );
    inline Void SubPF( UInt iDestIndex ); // st(i) = st(i) - st(0), and pop
    inline Void SubPD( UInt iDestIndex );
    inline Void MulPF( UInt iDestIndex ); // st(i) = st(i) * st(0), and pop
    inline Void MulPD( UInt iDestIndex );
    inline Void DivPF( UInt iDestIndex ); // st(i) = st(i) / st(0), and pop
    inline Void DivPD( UInt iDestIndex );

    inline Void SubRPF(); // st(1) = st(0) - st(1), and pop
    inline Void SubRPD();
    inline Void DivRPF(); // st(1) = st(0) / st(1), and pop
    inline Void DivRPD();

    inline Void SubRPF( UInt iDestIndex ); // st(i) = st(0) - st(i), and pop
    inline Void SubRPD( UInt iDestIndex );
    inline Void DivRPF( UInt iDestIndex ); // st(i) = st(0) / st(i), and pop
    inline Void DivRPD( UInt iDestIndex );

    inline Void HAddF();
    inline Void HAddD();
    inline Void HSubF();
    inline Void HSubD();

    inline Void HAddF( UInt iDestIndex );
    inline Void HAddD( UInt iDestIndex );
    inline Void HSubF( UInt iDestIndex );
    inline Void HSubD( UInt iDestIndex );

    inline Void HAddF( UInt iDestIndex, UInt iSrcIndex );
    inline Void HAddD( UInt iDestIndex, UInt iSrcIndex );
    inline Void HSubF( UInt iDestIndex, UInt iSrcIndex );
    inline Void HSubD( UInt iDestIndex, UInt iSrcIndex );

    inline Void SqrtF(); // st(0) = func( st(0) )
    inline Void SqrtD();
    inline Void InvF();
    inline Void InvSqrtF();

    inline Void SqrtF( UInt iStackIndex ); // st(i) = func( st(i) )
    inline Void SqrtD( UInt iStackIndex );
    inline Void InvF( UInt iStackIndex );
    inline Void InvSqrtF( UInt iStackIndex );

    inline Void SqrtF( UInt iDestIndex, UInt iSrcIndex ); // st(dst) = func( st(src) )
    inline Void SqrtD( UInt iDestIndex, UInt iSrcIndex );
    inline Void InvF( UInt iDestIndex, UInt iSrcIndex );
    inline Void InvSqrtF( UInt iDestIndex, UInt iSrcIndex );

    // More instructions coming soon ...

private:
    // add us 2 virtual instructions for push & pop
    inline Void _IPush( UInt iVariant, QWord inMem128[2] );
    inline Void _IPop( UInt iVariant, QWord outMem128[2] );

    // 128-bits Scratch area to allow aligned transfers between registers & memory
    Byte m_arrUnalignedScratch[32]; // (2 doubles/4 floats) = 16 bytes, max padding = 15+1 bytes => 32 bytes
    Float * m_pAlignedFloatScratch;
    Double * m_pAlignedDoubleScratch; // TODO : add NT instructions support to take advantage of nocache / cache hints ...

    // Stack pointer
    UInt m_iTop;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SSE.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SSE_H
