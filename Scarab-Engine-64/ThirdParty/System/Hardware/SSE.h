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
// Known Bugs : Unfortunately, there is no way to completely hide third-party
//              headers this time ... we have to make everything inline ...
//              Also compiler will require direct usage of intrinsic types.
//
// You should prefer AVX/AVX2 instructions when available !
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SSE_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SSE_H

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <intrin.h>
#include <mmintrin.h>
#include <emmintrin.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CPUID.h"

// General define for SIMD use in a lot of math code, comment those lines accordingly
//#define MATH_USE_SIMD_SSE // Assumes SSE42
#define MATH_USE_SIMD_AVX // Assumes AVX2 and SSE42

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define SIMDFn SIMDMath::GetInstance()

// Shuffle Control Mask
// Naming Convention : SIMD_SHUFFLE_MASK_NxP(_Z)
// N = Number of elements being picked.
// P = Number of elements to choose from.
// _Z = Has a Set to Zero flag.

// TODO : Something odd in Intel documentation, check if this one really needs to be shifted by 1
#define SIMD_SHUFFLE_MASK_1x2( _element_0 ) \
    ( (_element_0) & 0x01 )

#define SIMD_SHUFFLE_MASK_1x4( _element_0 ) \
    ( (_element_0) & 0x03 )

#define SIMD_SHUFFLE_MASK_1x8( _element_0 ) \
    ( (_element_0) & 0x07 )

#define SIMD_SHUFFLE_MASK_1x16_Z( _element_0, _zeroflag_0 ) \
    ( ((_element_0) & 0x0f) | (((_zeroflag_0) & 0x01) << 7) )

#define SIMD_SHUFFLE_MASK_2x2( _element_0, _element_1 ) \
    ( ((_element_0) & 0x01) | (((_element_1) & 0x01) << 1) )

#define SIMD_SHUFFLE_MASK_2x4_Z( _element_0, _zeroflag_0, _element_1, _zeroflag_1 ) \
    ( ((_element_0) & 0x03) | (((_zeroflag_0) & 0x01) << 3) | (((_element_1) & 0x03) << 4) | (((_zeroflag_1) & 0x01) << 7) )

#define SIMD_SHUFFLE_MASK_4x2( _element_0, _element_1, _element_2, _element_3 ) \
    ( ((_element_0) & 0x01) | (((_element_1) & 0x01) << 2) | (((_element_2) & 0x01) << 4) | (((_element_3) & 0x01) << 6) )

#define SIMD_SHUFFLE_MASK_4x4( _element_0, _element_1, _element_2, _element_3 ) \
    ( ((_element_0) & 0x03) | (((_element_1) & 0x03) << 2) | (((_element_2) & 0x03) << 4) | (((_element_3) & 0x03) << 6) )

// Blend Control Mask
#define SIMD_BLEND_MASK_2( _element_0, _element_1 ) \
    ( ((_element_0) & 0x01) | (((_element_1) & 0x01) << 1) )

#define SIMD_BLEND_MASK_4( _element_0, _element_1, _element_2, _element_3 ) \
    ( ((_element_0) & 0x01) | (((_element_1) & 0x01) << 1) | (((_element_2) & 0x01) << 2) | (((_element_3) & 0x01) << 3) )

#define SIMD_BLEND_MASK_8( _element_0, _element_1, _element_2, _element_3, _element_4, _element_5, _element_6, _element_7 ) \
    (        ((_element_0) & 0x01) | (((_element_1) & 0x01) << 1) | (((_element_2) & 0x01) << 2) | (((_element_3) & 0x01) << 3) | \
      (((_element_4) & 0x01) << 4) | (((_element_5) & 0x01) << 5) | (((_element_6) & 0x01) << 6) | (((_element_7) & 0x01) << 7) )

// SAD Control Mask
#define SIMD_SAD_MASK( _sel_src1, _sel_src2 ) \
    ( (((_sel_src1) & 0x01) << 2) | ((_sel_src2) & 0x03) )

// DotP Control Mask
#define SIMD_DOTP_MASK_2( _element_0, _element_1, _dest_0, _dest_1 ) \
    ( (((_element_0) & 0x01) << 4) | (((_element_1) & 0x01) << 5) | \
      ((_dest_0) & 0x01) | (((_dest_1) & 0x01) << 1) )

#define SIMD_DOTP_MASK_4( _element_0, _element_1, _element_2, _element_3, _dest_0, _dest_1, _dest_2, _dest_3 ) \
    ( (((_element_0) & 0x01) << 4) | (((_element_1) & 0x01) << 5) | (((_element_2) & 0x01) << 6) | (((_element_3) & 0x01) << 7) | \
      ((_dest_0) & 0x01) | (((_dest_1) & 0x01) << 1) | (((_dest_2) & 0x01) << 2) | (((_dest_3) & 0x01) << 3) )

// Comparison Modes
#define SIMD_CMP_TRUE_UQ    _CMP_TRUE_UQ    // True (unordered, quiet)
#define SIMD_CMP_TRUE_US    _CMP_TRUE_US    // True (unordered, signaling)

#define SIMD_CMP_FALSE_OQ   _CMP_FALSE_OQ   // False (ordered, quiet)
#define SIMD_CMP_FALSE_OS   _CMP_FALSE_OS   // False (ordered, signaling)

#define SIMD_CMP_ORD_Q      _CMP_ORD_Q      // Ordered (quiet)
#define SIMD_CMP_ORD_S      _CMP_ORD_S      // Ordered (signaling)

#define SIMD_CMP_UNORD_Q    _CMP_UNORD_Q    // Unordered (quiet)
#define SIMD_CMP_UNORD_S    _CMP_UNORD_S    // Unordered (signaling)

#define SIMD_CMP_EQ_UQ      _CMP_EQ_UQ      // Equal (unordered, quiet)
#define SIMD_CMP_EQ_US      _CMP_EQ_US      // Equal (unordered, signaling)
#define SIMD_CMP_EQ_OQ      _CMP_EQ_OQ      // Equal (ordered, quiet)
#define SIMD_CMP_EQ_OS      _CMP_EQ_OS      // Equal (ordered, signaling)

#define SIMD_CMP_NEQ_UQ     _CMP_NEQ_UQ     // Not Equal (unordered, quiet)
#define SIMD_CMP_NEQ_US     _CMP_NEQ_US     // Not Equal (unordered, signaling)
#define SIMD_CMP_NEQ_OQ     _CMP_NEQ_OQ     // Not Equal (ordered, quiet)
#define SIMD_CMP_NEQ_OS     _CMP_NEQ_OS     // Not Equal (ordered, signaling)

#define SIMD_CMP_LT_OQ      _CMP_LT_OQ      // Lesser-Than (ordered, quiet)
#define SIMD_CMP_LT_OS      _CMP_LT_OS      // Lesser-Than (ordered, signaling)

#define SIMD_CMP_NLT_UQ     _CMP_NLT_UQ     // Not Lesser-Than (unordered, quiet)
#define SIMD_CMP_NLT_US     _CMP_NLT_US     // Not Lesser-Than (unordered, signaling)

#define SIMD_CMP_LE_OQ      _CMP_LE_OQ      // Lesser-or-Equal (ordered, quiet)
#define SIMD_CMP_LE_OS      _CMP_LE_OS      // Lesser-or-Equal (ordered, signaling)

#define SIMD_CMP_NLE_UQ     _CMP_NLE_UQ     // Not Lesser-or-Equal (unordered, quiet)
#define SIMD_CMP_NLE_US     _CMP_NLE_US     // Not Lesser-or-Equal (unordered, signaling)

#define SIMD_CMP_GT_OQ      _CMP_GT_OQ      // Greater-Than (ordered, quiet)
#define SIMD_CMP_GT_OS      _CMP_GT_OS      // Greater-Than (ordered, signaling)

#define SIMD_CMP_NGT_UQ     _CMP_NGT_UQ     // Not Greater-Than (unordered, quiet)
#define SIMD_CMP_NGT_US     _CMP_NGT_US     // Not Greater-Than (unordered, signaling)

#define SIMD_CMP_GE_OQ      _CMP_GE_OQ      // Greater-or-Equal (ordered, quiet)
#define SIMD_CMP_GE_OS      _CMP_GE_OS      // Greater-or-Equal (ordered, signaling)

#define SIMD_CMP_NGE_UQ     _CMP_NGE_UQ     // Not Greater-or-Equal (unordererd, quiet)
#define SIMD_CMP_NGE_US     _CMP_NGE_US     // Not Greater-or-Equal (unordererd, signaling)

/////////////////////////////////////////////////////////////////////////////////
// The SIMDMath class
class SIMDMath
{
    // Discrete singleton interface
public:
    inline static SIMDMath * GetInstance();

private:
    SIMDMath();
    ~SIMDMath();

public:

    ////////////////////////////////////////////////////////////// Control instructions
        // Control & Status Register
    inline UInt32 GetCSR() const;               // SSE
    inline Void SetCSR( UInt32 iValue ) const;  // SSE

        // Clear and flush cache-line containing given address from all cache hierarchy levels
    inline Void ClearAndFlushCacheLine( Void * pAddress ) const; // SSE2

        // Spin-Wait Loop Hint for the Processor
    inline Void Pause() const; // SSE2

    ////////////////////////////////////////////////////////////// Serializing instructions (makes sure everything is flushed)
    inline Void SerializeMemoryStore() const;   // SSE
    inline Void SerializeMemoryLoad() const;    // SSE2
    inline Void SerializeMemory() const;        // SSE2

    ////////////////////////////////////////////////////////////// Register Initialization
    inline __m128 Zero128F() const;     // SSE
    inline __m256 Zero256F() const;     // AVX

    inline __m128d Zero128D() const;    // SSE2
    inline __m256d Zero256D() const;    // AVX

    inline __m128i Zero128I() const;    // SSE2
    inline __m256i Zero256I() const;    // AVX

        // Init all XMM/YMM registers
    inline Void ZeroUpper128() const;   // AVX
    inline Void Zero256() const;        // AVX

    ////////////////////////////////////////////////////////////// Values -> Registers
    inline __m128 SetLower( Float f0 ) const;   // SSE
    inline __m128d SetLower( Double f0 ) const; // SSE2

    inline __m128i SetLower( Int32 i0 ) const; // SSE2
    inline __m128i SetLower( Int64 i0 ) const; // SSE2

    inline __m128 Set128( Float f0, Float f1, Float f2, Float f3 ) const;                                           // SSE
    inline __m256 Set256( Float f0, Float f1, Float f2, Float f3, Float f4, Float f5, Float f6, Float f7 ) const;   // AVX

    inline __m128d Set128( Double f0, Double f1 ) const;                        // SSE2
    inline __m256d Set256( Double f0, Double f1, Double f2, Double f3 ) const;  // AVX

    inline __m128i Set128( Int8 i0, Int8 i1, Int8 i2, Int8 i3, Int8 i4, Int8 i5, Int8 i6, Int8 i7,
                           Int8 i8, Int8 i9, Int8 i10, Int8 i11, Int8 i12, Int8 i13, Int8 i14, Int8 i15 ) const;    // SSE2
    inline __m256i Set256( Int8 i0, Int8 i1, Int8 i2, Int8 i3, Int8 i4, Int8 i5, Int8 i6, Int8 i7,
                           Int8 i8, Int8 i9, Int8 i10, Int8 i11, Int8 i12, Int8 i13, Int8 i14, Int8 i15,
                           Int8 i16, Int8 i17, Int8 i18, Int8 i19, Int8 i20, Int8 i21, Int8 i22, Int8 i23,
                           Int8 i24, Int8 i25, Int8 i26, Int8 i27, Int8 i28, Int8 i29, Int8 i30, Int8 i31 ) const;  // AVX

    inline __m128i Set128( Int16 i0, Int16 i1, Int16 i2, Int16 i3, Int16 i4, Int16 i5, Int16 i6, Int16 i7 ) const;          // SSE2
    inline __m256i Set256( Int16 i0, Int16 i1, Int16 i2, Int16 i3, Int16 i4, Int16 i5, Int16 i6, Int16 i7,
                           Int16 i8, Int16 i9, Int16 i10, Int16 i11, Int16 i12, Int16 i13, Int16 i14, Int16 i15 ) const;    // AVX

    inline __m128i Set128( Int32 i0, Int32 i1, Int32 i2, Int32 i3 ) const;                                          // SSE2
    inline __m256i Set256( Int32 i0, Int32 i1, Int32 i2, Int32 i3, Int32 i4, Int32 i5, Int32 i6, Int32 i7 ) const;  // AVX

    inline __m128i Set128( Int64 i0, Int64 i1 ) const;                      // SSE2
    inline __m256i Set256( Int64 i0, Int64 i1, Int64 i2, Int64 i3 ) const;  // AVX

    inline __m128 Set128( Float f ) const;   // SSE
    inline __m256 Set256( Float f ) const;   // AVX

    inline __m128d Set128( Double f ) const; // SSE2
    inline __m256d Set256( Double f ) const; // AVX
    
    inline __m128i Set128( Int8 i ) const;   // SSE2
    inline __m256i Set256( Int8 i ) const;   // AVX

    inline __m128i Set128( Int16 i ) const;  // SSE2
    inline __m256i Set256( Int16 i ) const;  // AVX

    inline __m128i Set128( Int32 i ) const;  // SSE2
    inline __m256i Set256( Int32 i ) const;  // AVX

    inline __m128i Set128( Int64 i ) const;  // SSE2
    inline __m256i Set256( Int64 i ) const;  // AVX

    inline __m128i Set8I( __m128i mDst, Int8 iSrc, Int32 iIndex ) const; // SSE41
    inline __m256i Set8I( __m256i mDst, Int8 iSrc, Int32 iIndex ) const; // AVX

    inline __m128i Set16I( __m128i mDst, Int16 iSrc, Int32 iIndex ) const; // SSE2
    inline __m256i Set16I( __m256i mDst, Int16 iSrc, Int32 iIndex ) const; // AVX

    inline __m128 Set32F( __m128 mDst, Float fSrc, Int32 iIndex ) const; // SSE41
    inline __m256 Set32F( __m256 mDst, Float fSrc, Int32 iIndex ) const; // AVX

    inline __m128i Set32I( __m128i mDst, Int32 iSrc, Int32 iIndex ) const; // SSE41
    inline __m256i Set32I( __m256i mDst, Int32 iSrc, Int32 iIndex ) const; // AVX

    inline __m128d Set64D( __m128d mDst, Double fSrc, Int32 iIndex ) const; // SSE41
    inline __m256d Set64D( __m256d mDst, Double fSrc, Int32 iIndex ) const; // AVX

    inline __m128i Set64I( __m128i mDst, Int64 iSrc, Int32 iIndex ) const; // SSE41
    inline __m256i Set64I( __m256i mDst, Int64 iSrc, Int32 iIndex ) const; // AVX

    ////////////////////////////////////////////////////////////// Registers -> Values
    inline Float GetLowerFloat( __m128 mSrc ) const; // SSE
    inline Float GetLowerFloat( __m256 mSrc ) const; // AVX

    inline Double GetLowerDouble( __m128d mSrc ) const; // SSE2
    inline Double GetLowerDouble( __m256d mSrc ) const; // AVX

    inline Int32 GetLowerInt32( __m128i mSrc ) const; // SSE2
    inline Int32 GetLowerInt32( __m256i mSrc ) const; // AVX

    inline Int64 GetLowerInt64( __m128i mSrc ) const; // SSE2
    inline Int64 GetLowerInt64( __m256i mSrc ) const; // AVX

    inline Int32 Get8I( __m128i mSrc, Int32 iIndex ) const; // SSE41
    inline Int32 Get8I( __m256i mSrc, Int32 iIndex ) const; // AVX2

    inline Int32 Get16I( __m128i mSrc, Int32 iIndex ) const; // SSE2
    inline Int32 Get16I( __m256i mSrc, Int32 iIndex ) const; // AVX2

    inline Float Get32F( __m128 mSrc, Int32 iIndex ) const; // SSE41
    inline Float Get32F( __m256 mSrc, Int32 iIndex ) const; // AVX

    inline Int32 Get32I( __m128i mSrc, Int32 iIndex ) const; // SSE41
    inline Int32 Get32I( __m256i mSrc, Int32 iIndex ) const; // AVX

    inline Double Get64D( __m128d mSrc, Int32 iIndex ) const; // SSE41
    inline Double Get64D( __m256d mSrc, Int32 iIndex ) const; // AVX

    inline Int64 Get64I( __m128i mSrc, Int32 iIndex ) const; // SSE41
    inline Int64 Get64I( __m256i mSrc, Int32 iIndex ) const; // AVX

    ////////////////////////////////////////////////////////////// Memory -> Registers
        // Contiguous memory
    inline __m128 LoadLower( const Float * arrF ) const;    // SSE
    inline __m128d LoadLower( const Double * arrF ) const;  // SSE2

    inline __m128 Load128Aligned( const Float * arrF ) const; // SSE
    inline __m256 Load256Aligned( const Float * arrF ) const; // AVX

    inline __m128d Load128Aligned( const Double * arrF ) const; // SSE2
    inline __m256d Load256Aligned( const Double * arrF ) const; // AVX

    inline __m128i Load128Aligned( const __m128i * arrSrc ) const; // SSE2
    inline __m256i Load256Aligned( const __m256i * arrSrc ) const; // AVX

    inline __m128 Load128( const Float * arrF ) const; // SSE
    inline __m128 Load128( const Float * arrF, __m128i mSigns ) const; // AVX
    inline __m256 Load256( const Float * arrF ) const; // AVX
    inline __m256 Load256( const Float * arrF, __m256i mSigns ) const; // AVX

    inline __m128d Load128( const Double * arrF ) const; // SSE2
    inline __m128d Load128( const Double * arrF, __m128i mSigns ) const; // AVX
    inline __m256d Load256( const Double * arrF ) const; // AVX
    inline __m256d Load256( const Double * arrF, __m256i mSigns ) const; // AVX

    inline __m128i Load128( const Int32 * arrI, __m128i mSigns ) const; // AVX2
    inline __m256i Load256( const Int32 * arrI, __m256i mSigns ) const; // AVX2

    inline __m128i Load128( const Int64 * arrI, __m128i mSigns ) const; // AVX2
    inline __m256i Load256( const Int64 * arrI, __m256i mSigns ) const; // AVX2

    inline __m128i Load128( const __m128i * arrSrc ) const; // SSE3
    inline __m256i Load256( const __m256i * arrSrc ) const; // AVX

    inline __m128 Load128AlignedR( const Float * arrF ) const;      // SSE
    inline __m128d Load128AlignedR( const Double * arrF ) const;    // SSE2

    inline __m128 Load128Dupe( const Float * pF ) const;      // SSE
    inline __m128d Load128Dupe( const Double * pF ) const;    // SSE3

    inline __m128 LoadTwoFloatL( __m128 mDst, const __m64 * arrSrc ) const; // SSE
    inline __m128 LoadTwoFloatH( __m128 mDst, const __m64 * arrSrc ) const; // SSE

    inline __m128d LoadOneDoubleL( __m128d mDst, const Double * arrF ) const; // SSE2
    inline __m128d LoadOneDoubleH( __m128d mDst, const Double * arrF ) const; // SSE2

    inline __m128i LoadOneInt64L( const __m128i * arrSrc ) const;   // SSE2

    inline __m128 Dupe128Float( const Float * pF ) const; // AVX
    inline __m256 Dupe256Float( const Float * pF ) const; // AVX

    inline __m256d Dupe256Double( const Double * pF ) const; // AVX

    inline __m256 Dupe256FourFloat( const __m128 * pSrc ) const; // AVX
    inline __m256d Dupe256TwoDouble( const __m128d * pSrc ) const; // AVX

        // Sparse memory, 32-bit indices
    inline __m128 Load32FourFloat( const Float * pSrc, __m128i mIndices, Int32 iStride ) const;                                 // AVX2
    inline __m128 Load32FourFloat( __m128 mDst, const Float * pSrc, __m128i mIndices, Int32 iStride, __m128 mSigns ) const;     // AVX2
    inline __m256 Load32EightFloat( const Float * pSrc, __m256i mIndices, Int32 iStride ) const;                                // AVX2
    inline __m256 Load32EightFloat( __m256 mDst, const Float * pSrc, __m256i mIndices, Int32 iStride, __m256 mSigns ) const;    // AVX2

    inline __m128d Load32TwoDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) const;                                   // AVX2
    inline __m128d Load32TwoDouble( __m128d mDst, const Double * pSrc, __m128i mIndices, Int32 iStride, __m128d mSigns ) const;     // AVX2
    inline __m256d Load32FourDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) const;                                  // AVX2
    inline __m256d Load32FourDouble( __m256d mDst, const Double * pSrc, __m128i mIndices, Int32 iStride, __m256d mSigns ) const;    // AVX2

    inline __m128i Load32FourInt32( const Int32 * pSrc, __m128i mIndices, Int32 iStride ) const;                                // AVX2
    inline __m128i Load32FourInt32( __m128i mDst, const Int32 * pSrc, __m128i mIndices, Int32 iStride, __m128i mSigns ) const;  // AVX2
    inline __m256i Load32EightInt32( const Int32 * pSrc, __m256i mIndices, Int32 iStride ) const;                               // AVX2
    inline __m256i Load32EightInt32( __m256i mDst, const Int32 * pSrc, __m256i mIndices, Int32 iStride, __m256i mSigns ) const; // AVX2

    inline __m128i Load32TwoInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) const;  // AVX2
    inline __m128i Load32TwoInt64( __m128i mDst, const Int64 * pSrc, __m128i mIndices, Int32 iStride, __m128i mSigns ) const;  // AVX2
    inline __m256i Load32FourInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) const; // AVX2
    inline __m256i Load32FourInt64( __m256i mDst, const Int64 * pSrc, __m128i mIndices, Int32 iStride, __m256i mSigns ) const; // AVX2

        // Sparse memory, 64-bit indices
    inline __m128 Load64TwoFloat( const Float * pSrc, __m128i mIndices, Int32 iStride ) const;                              // AVX2
    inline __m128 Load64TwoFloat( __m128 mDst, const Float * pSrc, __m128i mIndices, Int32 iStride, __m128 mSigns ) const;  // AVX2
    inline __m128 Load64FourFloat( const Float * pSrc, __m256i mIndices, Int32 iStride ) const;                             // AVX2
    inline __m128 Load64FourFloat( __m128 mDst, const Float * pSrc, __m256i mIndices, Int32 iStride, __m128 mSigns ) const; // AVX2

    inline __m128d Load64TwoDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) const;                                   // AVX2
    inline __m128d Load64TwoDouble( __m128d mDst, const Double * pSrc, __m128i mIndices, Int32 iStride, __m128d mSigns ) const;     // AVX2
    inline __m256d Load64FourDouble( const Double * pSrc, __m256i mIndices, Int32 iStride ) const;                                  // AVX2
    inline __m256d Load64FourDouble( __m256d mDst, const Double * pSrc, __m256i mIndices, Int32 iStride, __m256d mSigns ) const;    // AVX2

    inline __m128i Load64TwoInt32( const Int32 * pSrc, __m128i mIndices, Int32 iStride ) const;                                 // AVX2
    inline __m128i Load64TwoInt32( __m128i mDst, const Int32 * pSrc, __m128i mIndices, Int32 iStride, __m128i mSigns ) const;   // AVX2
    inline __m128i Load64FourInt32( const Int32 * pSrc, __m256i mIndices, Int32 iStride ) const;                                // AVX2
    inline __m128i Load64FourInt32( __m128i mDst, const Int32 * pSrc, __m256i mIndices, Int32 iStride, __m128i mSigns ) const;  // AVX2

    inline __m128i Load64TwoInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) const;                                 // AVX2
    inline __m128i Load64TwoInt64( __m128i mDst, const Int64 * pSrc, __m128i mIndices, Int32 iStride, __m128i mSigns ) const;   // AVX2
    inline __m256i Load64FourInt64( const Int64 * pSrc, __m256i mIndices, Int32 iStride ) const;                                // AVX2
    inline __m256i Load64FourInt64( __m256i mDst, const Int64 * pSrc, __m256i mIndices, Int32 iStride, __m256i mSigns ) const;  // AVX2

        // Non-Temporal variants (stream_load instructions)
    inline __m128i LoadNT128Aligned( const __m128i * arrSrc ) const; // SSE41
    inline __m256i LoadNT256Aligned( const __m256i * arrSrc ) const; // AVX2

    ////////////////////////////////////////////////////////////// Registers -> Memory
    inline Void StoreLower( Float * outDst, __m128 mSrc ) const;  // SSE
    inline Void StoreLower( Double * outDst, __m128d mSrc ) const; // SSE2

    inline Void Store128Aligned( Float * outDst, __m128 mSrc ) const; // SSE
    inline Void Store256Aligned( Float * outDst, __m256 mSrc ) const; // AVX

    inline Void Store128Aligned( Double * outDst, __m128d mSrc ) const; // SSE2
    inline Void Store256Aligned( Double * outDst, __m256d mSrc ) const; // AVX

    inline Void Store128Aligned( __m128i * outDst, __m128i mSrc ) const; // SSE2
    inline Void Store256Aligned( __m256i * outDst, __m256i mSrc ) const; // AVX

    inline Void Store128( Float * outDst, __m128 mSrc ) const; // SSE
    inline Void Store256( Float * outDst, __m256 mSrc ) const; // AVX

    inline Void Store128( Double * outDst, __m128d mSrc ) const; // SSE2
    inline Void Store256( Double * outDst, __m256d mSrc ) const; // AVX

    inline Void Store128( __m128i * outDst, __m128i mSrc ) const; // SSE2
    inline Void Store256( __m256i * outDst, __m256i mSrc ) const; // AVX

    inline Void Store128AlignedR( Float * outDst, __m128 mSrc ) const;      // SSE
    inline Void Store128AlignedR( Double * outDst, __m128d mSrc ) const;    // SSE2

    inline Void Store128Dupe( Float * outDst, __m128 mSrc ) const;      // SSE
    inline Void Store128Dupe( Double * outDst, __m128d mSrc ) const;    // SSE2

    inline Void StoreTwoFloatL( __m64 * outDst, __m128 mSrc ) const; // SSE
    inline Void StoreTwoFloatH( __m64 * outDst, __m128 mSrc ) const; // SSE

    inline Void StoreOneDoubleL( Double * outDst, __m128d mSrc ) const; // SSE2
    inline Void StoreOneDoubleH( Double * outDst, __m128d mSrc ) const; // SSE2

    inline Void StoreOneInt64L( __m128i * outDst, __m128i mSrc ) const;   // SSE2

        // Non-Temporal variants (stream instructions)
    inline Void StoreNTLower( Float * outDst, __m128 mSrc ) const;  // SSE4a
    inline Void StoreNTLower( Double * outDst, __m128d mSrc ) const; // SSE4a

    inline Void StoreNT128Aligned( Float * outDst, __m128 mSrc ) const; // SSE
    inline Void StoreNT256Aligned( Float * outDst, __m256 mSrc ) const; // AVX

    inline Void StoreNT128Aligned( Double * outDst, __m128d mSrc ) const; // SSE2
    inline Void StoreNT256Aligned( Double * outDst, __m256d mSrc ) const; // AVX

    inline Void StoreNT128Aligned( __m128i * outDst, __m128i mSrc ) const; // SSE2
    inline Void StoreNT256Aligned( __m256i * outDst, __m256i mSrc ) const; // AVX

        // Masks
    inline Void Store128( Float * outDst, __m128 mSrc, __m128i mSigns ) const; // AVX
    inline Void Store256( Float * outDst, __m256 mSrc, __m256i mSigns ) const; // AVX

    inline Void Store128( Double * outDst, __m128d mSrc, __m128i mSigns ) const; // AVX
    inline Void Store256( Double * outDst, __m256d mSrc, __m256i mSigns ) const; // AVX

    inline Void Store128( Int32 * outDst, __m128i mSrc, __m128i mSigns ) const; // AVX2
    inline Void Store256( Int32 * outDst, __m256i mSrc, __m256i mSigns ) const; // AVX2

    inline Void Store128( Int64 * outDst, __m128i mSrc, __m128i mSigns ) const; // AVX2
    inline Void Store256( Int64 * outDst, __m256i mSrc, __m256i mSigns ) const; // AVX2

    ////////////////////////////////////////////////////////////// Registers <-> Registers
        // Move
            // Dst argument : Unaffected elements are copied
    inline __m128 MoveOneFloatLL( __m128 mDst, __m128 mSrc ) const;   // SSE
    inline __m128 MoveTwoFloatHL( __m128 mDst, __m128 mSrc ) const;   // SSE
    inline __m128 MoveTwoFloatLH( __m128 mDst, __m128 mSrc ) const;   // SSE

    inline __m128d MoveOneDoubleLL( __m128d mDst, __m128d mSrc ) const;   // SSE2

            // No Dst argument : Unaffected elements are zeroed
    inline __m128i MoveOneInt64LL( __m128i mSrc ) const; // SSE2

        // Duplicate
    inline __m128 DupeTwoFloatEven( __m128 mSrc ) const;    // SSE3
    inline __m128 DupeTwoFloatOdd( __m128 mSrc ) const;     // SSE3
    inline __m128d DupeOneDoubleL( __m128d mSrc ) const;    // SSE3

    inline __m256 DupeFourFloatEven( __m256 mSrc ) const;    // AVX
    inline __m256 DupeFourFloatOdd( __m256 mSrc ) const;     // AVX
    inline __m256d DupeTwoDoubleEven( __m256d mSrc ) const;    // AVX

    inline __m128 Dupe128Float( __m128 mSrc ) const; // AVX2
    inline __m256 Dupe256Float( __m128 mSrc ) const; // AVX2

    inline __m128d Dupe128Double( __m128d mSrc ) const; // AVX2
    inline __m256d Dupe256Double( __m128d mSrc ) const; // AVX2

    inline __m128i Dupe128Int8( __m128i mSrc ) const; // AVX2
    inline __m256i Dupe256Int8( __m128i mSrc ) const; // AVX2

    inline __m128i Dupe128Int16( __m128i mSrc ) const; // AVX2
    inline __m256i Dupe256Int16( __m128i mSrc ) const; // AVX2

    inline __m128i Dupe128Int32( __m128i mSrc ) const; // AVX2
    inline __m256i Dupe256Int32( __m128i mSrc ) const; // AVX2

    inline __m128i Dupe128Int64( __m128i mSrc ) const; // AVX2
    inline __m256i Dupe256Int64( __m128i mSrc ) const; // AVX2

    inline __m256i Dupe256Int128( __m128i mSrc ) const; // AVX2

        // Extract
    inline __m128 Extract128F( __m256 mSrc, Int32 iIndex ) const; // AVX
    inline __m128d Extract128D( __m256d mSrc, Int32 iIndex ) const; // AVX
    inline __m128i Extract128I( __m256i mSrc, Int32 iIndex ) const; // AVX2

        // Insert
    inline __m256 Insert128F( __m256 mDst, __m128 mSrc, Int32 iIndex ) const; // AVX
    inline __m256d Insert128D( __m256d mDst, __m128d mSrc, Int32 iIndex ) const; // AVX
    inline __m256i Insert128I( __m256i mDst, __m128i mSrc, Int32 iIndex ) const; // AVX2

    ////////////////////////////////////////////////////////////// Pack / Unpack
    inline __m128i PackSigned16To8( __m128i mSrcLow, __m128i mSrcHigh ) const; // SSE2
    inline __m256i PackSigned16To8( __m256i mSrcLow, __m256i mSrcHigh ) const; // AVX2

    inline __m128i PackSigned32To16( __m128i mSrcLow, __m128i mSrcHigh ) const; // SSE2
    inline __m256i PackSigned32To16( __m256i mSrcLow, __m256i mSrcHigh ) const; // AVX2

    inline __m128i PackUnsigned16To8( __m128i mSrcLow, __m128i mSrcHigh ) const; // SSE2
    inline __m256i PackUnsigned16To8( __m256i mSrcLow, __m256i mSrcHigh ) const; // AVX2

    inline __m128i PackUnsigned32To16( __m128i mSrcLow, __m128i mSrcHigh ) const; // SSE41
    inline __m256i PackUnsigned32To16( __m256i mSrcLow, __m256i mSrcHigh ) const; // AVX2

    inline __m128 UnpackFloatL( __m128 mSrcEven, __m128 mSrcOdd ) const; // SSE
    inline __m256 UnpackFloatL( __m256 mSrcEven, __m256 mSrcOdd ) const; // AVX

    inline __m128 UnpackFloatH( __m128 mSrcEven, __m128 mSrcOdd ) const; // SSE
    inline __m256 UnpackFloatH( __m256 mSrcEven, __m256 mSrcOdd ) const; // AVX

    inline __m128d UnpackDoubleL( __m128d mSrcEven, __m128d mSrcOdd ) const; // SSE2
    inline __m256d UnpackDoubleL( __m256d mSrcEven, __m256d mSrcOdd ) const; // AVX

    inline __m128d UnpackDoubleH( __m128d mSrcEven, __m128d mSrcOdd ) const; // SSE2
    inline __m256d UnpackDoubleH( __m256d mSrcEven, __m256d mSrcOdd ) const; // AVX

    inline __m128i UnpackInt8L( __m128i mSrcEven, __m128i mSrcOdd ) const; // SSE2
    inline __m256i UnpackInt8L( __m256i mSrcEven, __m256i mSrcOdd ) const; // AVX2

    inline __m128i UnpackInt8H( __m128i mSrcEven, __m128i mSrcOdd ) const; // SSE2
    inline __m256i UnpackInt8H( __m256i mSrcEven, __m256i mSrcOdd ) const; // AVX2

    inline __m128i UnpackInt16L( __m128i mSrcEven, __m128i mSrcOdd ) const; // SSE2
    inline __m256i UnpackInt16L( __m256i mSrcEven, __m256i mSrcOdd ) const; // AVX2

    inline __m128i UnpackInt16H( __m128i mSrcEven, __m128i mSrcOdd ) const; // SSE2
    inline __m256i UnpackInt16H( __m256i mSrcEven, __m256i mSrcOdd ) const; // AVX2

    inline __m128i UnpackInt32L( __m128i mSrcEven, __m128i mSrcOdd ) const; // SSE2
    inline __m256i UnpackInt32L( __m256i mSrcEven, __m256i mSrcOdd ) const; // AVX2

    inline __m128i UnpackInt32H( __m128i mSrcEven, __m128i mSrcOdd ) const; // SSE2
    inline __m256i UnpackInt32H( __m256i mSrcEven, __m256i mSrcOdd ) const; // AVX2

    inline __m128i UnpackInt64L( __m128i mSrcEven, __m128i mSrcOdd ) const; // SSE2
    inline __m256i UnpackInt64L( __m256i mSrcEven, __m256i mSrcOdd ) const; // AVX2

    inline __m128i UnpackInt64H( __m128i mSrcEven, __m128i mSrcOdd ) const; // SSE2
    inline __m256i UnpackInt64H( __m256i mSrcEven, __m256i mSrcOdd ) const; // AVX2

    ////////////////////////////////////////////////////////////// Shuffle
    inline __m128 Shuffle128Float( __m128 mSrcLow, __m128 mSrcHigh, Int iMask4x4 ) const;   // SSE
    inline __m128 Shuffle128Float( __m128 mSrc, Int iMask4x4 ) const;                       // AVX
    inline __m128 Shuffle128Float( __m128 mSrc, __m128i mMask1x4 ) const;                   // AVX

    inline __m256 Shuffle128Float( __m256 mSrcLow, __m256 mSrcHigh, Int iMask4x4 ) const;   // AVX
    inline __m256 Shuffle128Float( __m256 mSrc, Int iMask4x4 ) const;                       // AVX
    inline __m256 Shuffle128Float( __m256 mSrc, __m256i mMask1x4 ) const;                   // AVX

    inline __m256 Shuffle256Float( __m256 mSrc, __m256i mMask1x8 ) const;   // AVX2

    inline __m256 Shuffle512FourFloat( __m256 mSrc1, __m256 mSrc2, Int iMask2x4_Z ) const;  // AVX

    inline __m128d Shuffle128Double( __m128d mSrcLow, __m128d mSrcHigh, Int iMask2x2 ) const;   // SSE2
    inline __m128d Shuffle128Double( __m128d mSrc, Int iMask2x2 ) const;                        // AVX
    inline __m128d Shuffle128Double( __m128d mSrc, __m128i mMask1x2 ) const;                    // AVX

    inline __m256d Shuffle128Double( __m256d mSrcLow, __m256d mSrcHigh, Int iMask4x2 ) const;   // AVX
    inline __m256d Shuffle128Double( __m256d mSrc, Int iMask4x2 ) const;                        // AVX
    inline __m256d Shuffle128Double( __m256d mSrc, __m256i mMask1x2 ) const;                    // AVX

    inline __m256d Shuffle256Double( __m256d mSrc, Int iMask4x4 ) const;    // AVX2

    inline __m256d Shuffle512TwoDouble( __m256d mSrc1, __m256d mSrc2, Int iMask2x4_Z ) const;   // AVX

    inline __m128i Shuffle128Int8( __m128i mSrc, __m128i mMask1x16_Z ) const;   // SSSE3
    inline __m256i Shuffle128Int8( __m256i mSrc, __m256i mMask1x16_Z ) const;   // AVX2

    inline __m128i Shuffle64Int16L( __m128i mSrc, Int iMask4x4 ) const; // SSE2
    inline __m256i Shuffle64Int16L( __m256i mSrc, Int iMask4x4 ) const; // AVX2

    inline __m128i Shuffle64Int16H( __m128i mSrc, Int iMask4x4 ) const; // SSE2
    inline __m256i Shuffle64Int16H( __m256i mSrc, Int iMask4x4 ) const; // AVX2

    inline __m128i Shuffle128Int32( __m128i mSrc, Int iMask4x4 ) const; // SSE2
    inline __m256i Shuffle128Int32( __m256i mSrc, Int iMask4x4 ) const; // AVX2

    inline __m256i Shuffle256Int32( __m256i mSrc, __m256i mMask1x8 ) const; // AVX2

    inline __m256i Shuffle512FourInt32( __m256i mSrc1, __m256i mSrc2, Int iMask2x4_Z ) const;   // AVX2

    inline __m256i Shuffle256Int64( __m256i mSrc, Int iMask4x4 ) const; // AVX2

    ////////////////////////////////////////////////////////////// Blend
    inline __m128 BlendFloat( __m128 mDst, __m128 mSrc, Int iMask4 ) const;     // SSE41
    inline __m128 BlendFloat( __m128 mDst, __m128 mSrc, __m128 mSigns ) const;  // SSE41
    inline __m256 BlendFloat( __m256 mDst, __m256 mSrc, Int iMask8 ) const;     // AVX
    inline __m256 BlendFloat( __m256 mDst, __m256 mSrc, __m256 mSigns ) const;  // AVX

    inline __m128d BlendDouble( __m128d mDst, __m128d mSrc, Int iMask2 ) const;     // SSE41
    inline __m128d BlendDouble( __m128d mDst, __m128d mSrc, __m128d mSigns ) const; // SSE41
    inline __m256d BlendDouble( __m256d mDst, __m256d mSrc, Int iMask4 ) const;     // AVX
    inline __m256d BlendDouble( __m256d mDst, __m256d mSrc, __m256d mSigns ) const; // AVX

    inline __m128i BlendInt8( __m128i mDst, __m128i mSrc, __m128i mSigns ) const;   // SSE41
    inline __m256i BlendInt8( __m256i mDst, __m256i mSrc, __m256i mSigns ) const;   // AVX2

    inline __m128i BlendInt16( __m128i mDst, __m128i mSrc, Int iMask8 ) const;  // SSE41
    inline __m256i BlendInt16( __m256i mDst, __m256i mSrc, Int iMask8 ) const;  // AVX2

    inline __m128i BlendInt32( __m128i mDst, __m128i mSrc, Int iMask4 ) const;  // AVX2
    inline __m256i BlendInt32( __m256i mDst, __m256i mSrc, Int iMask8 ) const;  // AVX2

    ////////////////////////////////////////////////////////////// Cast (Free, 0 instruction generated)
    inline __m128 CastToFloat( __m128d mDouble ) const;     // SSE2
    inline __m128 CastToFloat( __m128i mInteger ) const;    // SSE2
    inline __m256 CastToFloat( __m256d mDouble ) const;     // AVX
    inline __m256 CastToFloat( __m256i mInteger ) const;    // AVX

    inline __m128d CastToDouble( __m128 mFloat ) const;     // SSE2
    inline __m128d CastToDouble( __m128i mInteger ) const;  // SSE2
    inline __m256d CastToDouble( __m256 mFloat ) const;     // AVX
    inline __m256d CastToDouble( __m256i mInteger ) const;  // AVX

    inline __m128i CastToInteger( __m128 mFloat ) const;    // SSE2
    inline __m128i CastToInteger( __m128d mDouble ) const;  // SSE2
    inline __m256i CastToInteger( __m256 mFloat ) const;    // AVX
    inline __m256i CastToInteger( __m256d mDouble ) const;  // AVX

    inline __m128 CastDown( __m256 mFloat ) const;      // AVX
    inline __m128d CastDown( __m256d mDouble ) const;   // AVX
    inline __m128i CastDown( __m256i mInteger ) const;  // AVX

    inline __m256 CastUp( __m128 mFloat ) const;        // AVX
    inline __m256d CastUp( __m128d mDouble ) const;     // AVX
    inline __m256i CastUp( __m128i mInteger ) const;    // AVX

    ////////////////////////////////////////////////////////////// Convert & Copy
    inline __m128 ConvertLower( __m128 mDst, Int32 iSrc ) const;    // SSE
    inline __m128 ConvertLower( __m128 mDst, Int64 iSrc ) const;    // SSE

    inline __m128d ConvertLower( __m128d mDst, Int32 iSrc ) const;  // SSE2
    inline __m128d ConvertLower( __m128d mDst, Int64 iSrc ) const;  // SSE2

    inline __m128 ConvertLower( __m128 mDst, __m128d mSrc ) const;  // SSE2
    inline __m128d ConvertLower( __m128d mDst, __m128 mSrc ) const; // SSE2

    ////////////////////////////////////////////////////////////// Convert
    inline Int32 ConvertLowerToInt32( __m128 mSrc ) const;      // SSE
    inline Int32 ConvertLowerToInt32( __m128d mSrc ) const;     // SSE2

    inline Int64 ConvertLowerToInt64( __m128 mSrc ) const;      // SSE
    inline Int64 ConvertLowerToInt64( __m128d mSrc ) const;     // SSE2

    inline __m128 Convert128ToFloat( __m128d mSrc ) const;      // SSE2
    inline __m128 Convert128ToFloat( __m256d mSrc ) const;      // AVX

    inline __m128 Convert128ToFloat( __m128i mSrc ) const;      // SSE2
    inline __m256 Convert256ToFloat( __m256i mSrc ) const;      // AVX

    inline __m128d Convert128ToDouble( __m128 mSrc ) const;     // SSE2
    inline __m256d Convert256ToDouble( __m128 mSrc ) const;     // AVX

    inline __m128d Convert128ToDouble( __m128i mSrc ) const;    // SSE2
    inline __m256d Convert256ToDouble( __m128i mSrc ) const;    // AVX

    inline __m128i Convert128ToInt32( __m128 mSrc ) const;      // SSE2
    inline __m256i Convert256ToInt32( __m256 mSrc ) const;      // AVX

    inline __m128i Convert128ToInt32( __m128d mSrc ) const;     // SSE2
    inline __m128i Convert128ToInt32( __m256d mSrc ) const;     // AVX

    ////////////////////////////////////////////////////////////// Truncate
    inline Int32 TruncateLowerToInt32( __m128 mSrc ) const;     // SSE
    inline Int32 TruncateLowerToInt32( __m128d mSrc ) const;    // SSE2

    inline Int64 TruncateLowerToInt64( __m128 mSrc ) const;     // SSE
    inline Int64 TruncateLowerToInt64( __m128d mSrc ) const;    // SSE2

    inline __m128i TruncateToInt32( __m128 mSrc ) const;    // SSE2
    inline __m256i TruncateToInt32( __m256 mSrc ) const;    // AVX

    inline __m128i TruncateToInt32( __m128d mSrc ) const;   // SSE2
    inline __m128i TruncateToInt32( __m256d mSrc ) const;   // AVX

    ////////////////////////////////////////////////////////////// Sign-Extend
    inline __m128i SignExtend128Int8To16( __m128i mSrc ) const;     // SSE41
    inline __m128i SignExtend128Int8To32( __m128i mSrc ) const;     // SSE41
    inline __m128i SignExtend128Int8To64( __m128i mSrc ) const;     // SSE41
    inline __m128i SignExtend128Int16To32( __m128i mSrc ) const;    // SSE41
    inline __m128i SignExtend128Int16To64( __m128i mSrc ) const;    // SSE41
    inline __m128i SignExtend128Int32To64( __m128i mSrc ) const;    // SSE41

    inline __m256i SignExtend256Int8To16( __m128i mSrc ) const;     // AVX2
    inline __m256i SignExtend256Int8To32( __m128i mSrc ) const;     // AVX2
    inline __m256i SignExtend256Int8To64( __m128i mSrc ) const;     // AVX2
    inline __m256i SignExtend256Int16To32( __m128i mSrc ) const;    // AVX2
    inline __m256i SignExtend256Int16To64( __m128i mSrc ) const;    // AVX2
    inline __m256i SignExtend256Int32To64( __m128i mSrc ) const;    // AVX2

    ////////////////////////////////////////////////////////////// Zero-Extend
    inline __m128i ZeroExtend128Int8To16( __m128i mSrc ) const;     // SSE41
    inline __m128i ZeroExtend128Int8To32( __m128i mSrc ) const;     // SSE41
    inline __m128i ZeroExtend128Int8To64( __m128i mSrc ) const;     // SSE41
    inline __m128i ZeroExtend128Int16To32( __m128i mSrc ) const;    // SSE41
    inline __m128i ZeroExtend128Int16To64( __m128i mSrc ) const;    // SSE41
    inline __m128i ZeroExtend128Int32To64( __m128i mSrc ) const;    // SSE41

    inline __m256i ZeroExtend256Int8To16( __m128i mSrc ) const;     // AVX2
    inline __m256i ZeroExtend256Int8To32( __m128i mSrc ) const;     // AVX2
    inline __m256i ZeroExtend256Int8To64( __m128i mSrc ) const;     // AVX2
    inline __m256i ZeroExtend256Int16To32( __m128i mSrc ) const;    // AVX2
    inline __m256i ZeroExtend256Int16To64( __m128i mSrc ) const;    // AVX2
    inline __m256i ZeroExtend256Int32To64( __m128i mSrc ) const;    // AVX2

    ////////////////////////////////////////////////////////////// Absolute Value
    inline __m128i Abs8( __m128i mValue ) const; // SSSE3
    inline __m256i Abs8( __m256i mValue ) const; // AVX2

    inline __m128i Abs16( __m128i mValue ) const; // SSSE3
    inline __m256i Abs16( __m256i mValue ) const; // AVX2

    inline __m128i Abs32( __m128i mValue ) const; // SSSE3
    inline __m256i Abs32( __m256i mValue ) const; // AVX2

    inline __m128i Abs64( __m128i mValue ) const; // SSSE3
    inline __m256i Abs64( __m256i mValue ) const; // AVX2

    ////////////////////////////////////////////////////////////// Sign Change
    inline __m128i Negate8( __m128i mValue, __m128i mSigns ) const; // SSSE3
    inline __m256i Negate8( __m256i mValue, __m256i mSigns ) const; // AVX2

    inline __m128i Negate16( __m128i mValue, __m128i mSigns ) const; // SSSE3
    inline __m256i Negate16( __m256i mValue, __m256i mSigns ) const; // AVX2

    inline __m128i Negate32( __m128i mValue, __m128i mSigns ) const; // SSSE3
    inline __m256i Negate32( __m256i mValue, __m256i mSigns ) const; // AVX2

    ////////////////////////////////////////////////////////////// Rounding
    inline __m128 FloorLower( __m128 mDst, __m128 mSrc ) const;     // SSE41
    inline __m128d FloorLower( __m128d mDst, __m128d mSrc ) const;  // SSE41

    inline __m128 Floor( __m128 mValue ) const; // SSE41
    inline __m256 Floor( __m256 mValue ) const; // AVX

    inline __m128d Floor( __m128d mValue ) const; // SSE41
    inline __m256d Floor( __m256d mValue ) const; // AVX

    inline __m128 CeilLower( __m128 mDst, __m128 mSrc ) const;      // SSE41
    inline __m128d CeilLower( __m128d mDst, __m128d mSrc ) const;   // SSE41

    inline __m128 Ceil( __m128 mValue ) const; // SSE41
    inline __m256 Ceil( __m256 mValue ) const; // AVX

    inline __m128d Ceil( __m128d mValue ) const; // SSE41
    inline __m256d Ceil( __m256d mValue ) const; // AVX

    inline __m128 RoundLower( __m128 mDst, __m128 mSrc ) const;     // SSE41
    inline __m128d RoundLower( __m128d mDst, __m128d mSrc ) const;  // SSE41

    inline __m128 Round( __m128 mValue ) const; // SSE41
    inline __m256 Round( __m256 mValue ) const; // AVX

    inline __m128d Round( __m128d mValue ) const; // SSE41
    inline __m256d Round( __m256d mValue ) const; // AVX

    ////////////////////////////////////////////////////////////// Addition
    inline __m128 AddLower( __m128 mDst, __m128 mSrc ) const;       // SSE
    inline __m128d AddLower( __m128d mDst, __m128d mSrc ) const;    // SSE2

    inline __m128 Add( __m128 mDst, __m128 mSrc ) const; // SSE
    inline __m256 Add( __m256 mDst, __m256 mSrc ) const; // AVX

    inline __m128d Add( __m128d mDst, __m128d mSrc ) const; // SSE2
    inline __m256d Add( __m256d mDst, __m256d mSrc ) const; // AVX

    inline __m128i Add8( __m128i mDst, __m128i mSrc ) const;    // SSE2
    inline __m256i Add8( __m256i mDst, __m256i mSrc ) const;    // AVX2

    inline __m128i Add16( __m128i mDst, __m128i mSrc ) const;   // SSE2
    inline __m256i Add16( __m256i mDst, __m256i mSrc ) const;   // AVX2

    inline __m128i Add32( __m128i mDst, __m128i mSrc ) const;   // SSE2
    inline __m256i Add32( __m256i mDst, __m256i mSrc ) const;   // AVX2

    inline __m128i Add64( __m128i mDst, __m128i mSrc ) const;   // SSE2
    inline __m256i Add64( __m256i mDst, __m256i mSrc ) const;   // AVX2

    ////////////////////////////////////////////////////////////// Addition with Saturation
    inline __m128i AddSigned8( __m128i mDst, __m128i mSrc ) const;  // SSE2
    inline __m256i AddSigned8( __m256i mDst, __m256i mSrc ) const;  // AVX2

    inline __m128i AddSigned16( __m128i mDst, __m128i mSrc ) const; // SSE2
    inline __m256i AddSigned16( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i AddUnsigned8( __m128i mDst, __m128i mSrc ) const;    // SSE2
    inline __m256i AddUnsigned8( __m256i mDst, __m256i mSrc ) const;    // AVX2

    inline __m128i AddUnsigned16( __m128i mDst, __m128i mSrc ) const;   // SSE2
    inline __m256i AddUnsigned16( __m256i mDst, __m256i mSrc ) const;   // AVX2

    ////////////////////////////////////////////////////////////// Horizontal Addition
    inline __m128 HAdd( __m128 mSrc1, __m128 mSrc2 ) const; // SSE3
    inline __m256 HAdd( __m256 mSrc1, __m256 mSrc2 ) const; // AVX

    inline __m128d HAdd( __m128d mSrc1, __m128d mSrc2 ) const; // SSE3
    inline __m256d HAdd( __m256d mSrc1, __m256d mSrc2 ) const; // AVX

    inline __m128i HAdd16( __m128i mSrc1, __m128i mSrc2 ) const; // SSSE3
    inline __m256i HAdd16( __m256i mSrc1, __m256i mSrc2 ) const; // AVX2

    inline __m128i HAdd32( __m128i mSrc1, __m128i mSrc2 ) const; // SSSE3
    inline __m256i HAdd32( __m256i mSrc1, __m256i mSrc2 ) const; // AVX2

    ////////////////////////////////////////////////////////////// Horizontal Addition with Saturation
    inline __m128i HAddSigned16( __m128i mSrc1, __m128i mSrc2 ) const; // SSSE3
    inline __m256i HAddSigned16( __m256i mSrc1, __m256i mSrc2 ) const; // AVX2

    ////////////////////////////////////////////////////////////// Substraction
    inline __m128 SubLower( __m128 mDst, __m128 mSrc ) const;       // SSE
    inline __m128d SubLower( __m128d mDst, __m128d mSrc ) const;    // SSE2

    inline __m128 Sub( __m128 mDst, __m128 mSrc ) const; // SSE
    inline __m256 Sub( __m256 mDst, __m256 mSrc ) const; // AVX

    inline __m128d Sub( __m128d mDst, __m128d mSrc ) const; // SSE2
    inline __m256d Sub( __m256d mDst, __m256d mSrc ) const; // AVX

    inline __m128i Sub8( __m128i mDst, __m128i mSrc ) const;    // SSE2
    inline __m256i Sub8( __m256i mDst, __m256i mSrc ) const;    // AVX2

    inline __m128i Sub16( __m128i mDst, __m128i mSrc ) const;   // SSE2
    inline __m256i Sub16( __m256i mDst, __m256i mSrc ) const;   // AVX2

    inline __m128i Sub32( __m128i mDst, __m128i mSrc ) const;   // SSE2
    inline __m256i Sub32( __m256i mDst, __m256i mSrc ) const;   // AVX2

    inline __m128i Sub64( __m128i mDst, __m128i mSrc ) const;   // SSE2
    inline __m256i Sub64( __m256i mDst, __m256i mSrc ) const;   // AVX2

    ////////////////////////////////////////////////////////////// Substraction with Saturation
    inline __m128i SubSigned8( __m128i mDst, __m128i mSrc ) const;  // SSE2
    inline __m256i SubSigned8( __m256i mDst, __m256i mSrc ) const;  // AVX2

    inline __m128i SubSigned16( __m128i mDst, __m128i mSrc ) const; // SSE2
    inline __m256i SubSigned16( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i SubUnsigned8( __m128i mDst, __m128i mSrc ) const;    // SSE2
    inline __m256i SubUnsigned8( __m256i mDst, __m256i mSrc ) const;    // AVX2

    inline __m128i SubUnsigned16( __m128i mDst, __m128i mSrc ) const;   // SSE2
    inline __m256i SubUnsigned16( __m256i mDst, __m256i mSrc ) const;   // AVX2

    ////////////////////////////////////////////////////////////// Horizontal Substraction
    inline __m128 HSub( __m128 mSrc1, __m128 mSrc2 ) const; // SSE3
    inline __m256 HSub( __m256 mSrc1, __m256 mSrc2 ) const; // AVX

    inline __m128d HSub( __m128d mSrc1, __m128d mSrc2 ) const; // SSE3
    inline __m256d HSub( __m256d mSrc1, __m256d mSrc2 ) const; // AVX

    inline __m128i HSub16( __m128i mSrc1, __m128i mSrc2 ) const; // SSSE3
    inline __m256i HSub16( __m256i mSrc1, __m256i mSrc2 ) const; // AVX2

    inline __m128i HSub32( __m128i mSrc1, __m128i mSrc2 ) const; // SSSE3
    inline __m256i HSub32( __m256i mSrc1, __m256i mSrc2 ) const; // AVX2

    ////////////////////////////////////////////////////////////// Horizontal Substraction with Saturation
    inline __m128i HSubSigned16( __m128i mSrc1, __m128i mSrc2 ) const; // SSSE3
    inline __m256i HSubSigned16( __m256i mSrc1, __m256i mSrc2 ) const; // AVX2

    ////////////////////////////////////////////////////////////// Interleaved Add & Sub (Sub Even / Add Odd)
    inline __m128 AddSub( __m128 mDst, __m128 mSrc ) const; // SSE3
    inline __m256 AddSub( __m256 mDst, __m256 mSrc ) const; // AVX

    inline __m128d AddSub( __m128d mDst, __m128d mSrc ) const; // SSE3
    inline __m256d AddSub( __m256d mDst, __m256d mSrc ) const; // AVX

    ////////////////////////////////////////////////////////////// SAD (Sum Absolute Differences)
    inline __m128i SAD( __m128i mSrc1, __m128i mSrc2 ) const; // SSE2
    inline __m256i SAD( __m256i mSrc1, __m256i mSrc2 ) const; // AVX2

    inline __m128i SAD( __m128i mSrc1, __m128i mSrc2, Int iMask ) const; // SSE41
    inline __m256i SAD( __m256i mSrc1, __m256i mSrc2, Int iMask ) const; // AVX2

    ////////////////////////////////////////////////////////////// Multiplication
    inline __m128 MulLower( __m128 mDst, __m128 mSrc ) const;       // SSE
    inline __m128d MulLower( __m128d mDst, __m128d mSrc ) const;    // SSE2

    inline __m128 Mul( __m128 mDst, __m128 mSrc ) const; // SSE
    inline __m256 Mul( __m256 mDst, __m256 mSrc ) const; // AVX

    inline __m128d Mul( __m128d mDst, __m128d mSrc ) const; // SSE2
    inline __m256d Mul( __m256d mDst, __m256d mSrc ) const; // AVX

    inline __m128i MulSigned16L( __m128i mDst, __m128i mSrc ) const;   // SSE2
    inline __m256i MulSigned16L( __m256i mDst, __m256i mSrc ) const;   // AVX2

    inline __m128i MulSigned16H( __m128i mDst, __m128i mSrc ) const;   // SSE2
    inline __m256i MulSigned16H( __m256i mDst, __m256i mSrc ) const;   // AVX2

    inline __m128i MulSigned32( __m128i mDst, __m128i mSrc ) const;   // SSE41
    inline __m256i MulSigned32( __m256i mDst, __m256i mSrc ) const;   // AVX2

    inline __m128i MulSigned32L( __m128i mDst, __m128i mSrc ) const;   // SSE41
    inline __m256i MulSigned32L( __m256i mDst, __m256i mSrc ) const;   // AVX2

    inline __m128i MulSigned64L( __m128i mDst, __m128i mSrc ) const;   // SSE41
    inline __m256i MulSigned64L( __m256i mDst, __m256i mSrc ) const;   // AVX2

    inline __m128i MulUnsigned16H( __m128i mDst, __m128i mSrc ) const;   // SSE2
    inline __m256i MulUnsigned16H( __m256i mDst, __m256i mSrc ) const;   // AVX2

    inline __m128i MulUnsigned32( __m128i mDst, __m128i mSrc ) const;   // SSE2
    inline __m256i MulUnsigned32( __m256i mDst, __m256i mSrc ) const;   // AVX2

    ////////////////////////////////////////////////////////////// MADD (Multiply and Add)
        // Signed 16-bits -> Signed 32-bits
    inline __m128i MAdd( __m128i mDst, __m128i mSrc ) const;   // SSE2
    inline __m256i MAdd( __m256i mDst, __m256i mSrc ) const;   // AVX2

        // Unsigned-Signed 8-bits -> Signed 16-bits
    inline __m128i MAddUS( __m128i mDst, __m128i mSrc ) const;   // SSSE3
    inline __m256i MAddUS( __m256i mDst, __m256i mSrc ) const;   // AVX2

    ////////////////////////////////////////////////////////////// DP (Dot Product)
    inline __m128 DotP( __m128 mDst, __m128 mSrc, Int iMask4 ) const; // SSE41
    inline __m256 DotP( __m256 mDst, __m256 mSrc, Int iMask4 ) const; // AVX

    inline __m128d DotP( __m128d mDst, __m128d mSrc, Int iMask2 ) const; // SSE41

    ////////////////////////////////////////////////////////////// Division
    inline __m128 DivLower( __m128 mDst, __m128 mSrc ) const;       // SSE
    inline __m128d DivLower( __m128d mDst, __m128d mSrc ) const;    // SSE2

    inline __m128 Div( __m128 mDst, __m128 mSrc ) const; // SSE
    inline __m256 Div( __m256 mDst, __m256 mSrc ) const; // AVX

    inline __m128d Div( __m128d mDst, __m128d mSrc ) const; // SSE2
    inline __m256d Div( __m256d mDst, __m256d mSrc ) const; // AVX

    ////////////////////////////////////////////////////////////// Modulo




    ////////////////////////////////////////////////////////////// Average (always unsigned)
    inline __m128i Avg8( __m128i mDst, __m128i mSrc ) const;    // SSE2
    inline __m256i Avg8( __m256i mDst, __m256i mSrc ) const;    // AVX2

    inline __m128i Avg16( __m128i mDst, __m128i mSrc ) const;   // SSE2
    inline __m256i Avg16( __m256i mDst, __m256i mSrc ) const;   // AVX2

    ////////////////////////////////////////////////////////////// Compare (Generic)
    inline __m128 CmpLower( __m128 mDst, __m128 mSrc, Int iCmpMode ) const;     // AVX
    inline __m128d CmpLower( __m128d mDst, __m128d mSrc, Int iCmpMode ) const;  // AVX

    inline __m128 Cmp( __m128 mDst, __m128 mSrc, Int iCmpMode ) const; // AVX
    inline __m256 Cmp( __m256 mDst, __m256 mSrc, Int iCmpMode ) const; // AVX

    inline __m128d Cmp( __m128d mDst, __m128d mSrc, Int iCmpMode ) const; // AVX
    inline __m256d Cmp( __m256d mDst, __m256d mSrc, Int iCmpMode ) const; // AVX

    ////////////////////////////////////////////////////////////// Compare (Equal)
    inline __m128 CmpEQLower( __m128 mDst, __m128 mSrc ) const;     // SSE
    inline __m128d CmpEQLower( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline __m128 CmpEQ( __m128 mDst, __m128 mSrc ) const;  // SSE

    inline __m128d CmpEQ( __m128d mDst, __m128d mSrc ) const;   // SSE2

    inline __m128i CmpEQ8( __m128i mDst, __m128i mSrc ) const;  // SSE2
    inline __m256i CmpEQ8( __m256i mDst, __m256i mSrc ) const;  // AVX2

    inline __m128i CmpEQ16( __m128i mDst, __m128i mSrc ) const; // SSE2
    inline __m256i CmpEQ16( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i CmpEQ32( __m128i mDst, __m128i mSrc ) const; // SSE2
    inline __m256i CmpEQ32( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i CmpEQ64( __m128i mDst, __m128i mSrc ) const; // SSE41
    inline __m256i CmpEQ64( __m256i mDst, __m256i mSrc ) const; // AVX2

    ////////////////////////////////////////////////////////////// Compare (Not Equal)
    inline __m128 CmpNEQLower( __m128 mDst, __m128 mSrc ) const;     // SSE
    inline __m128d CmpNEQLower( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline __m128 CmpNEQ( __m128 mDst, __m128 mSrc ) const;  // SSE

    inline __m128d CmpNEQ( __m128d mDst, __m128d mSrc ) const;   // SSE2

    ////////////////////////////////////////////////////////////// Compare (Lesser-Than)
    inline __m128 CmpLTLower( __m128 mDst, __m128 mSrc ) const;     // SSE
    inline __m128d CmpLTLower( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline __m128 CmpLT( __m128 mDst, __m128 mSrc ) const;  // SSE

    inline __m128d CmpLT( __m128d mDst, __m128d mSrc ) const;   // SSE2

    inline __m128i CmpLT8( __m128i mDst, __m128i mSrc ) const;  // SSE2

    inline __m128i CmpLT16( __m128i mDst, __m128i mSrc ) const; // SSE2

    inline __m128i CmpLT32( __m128i mDst, __m128i mSrc ) const; // SSE2

    ////////////////////////////////////////////////////////////// Compare (Not Lesser-Than)
    inline __m128 CmpNLTLower( __m128 mDst, __m128 mSrc ) const;     // SSE
    inline __m128d CmpNLTLower( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline __m128 CmpNLT( __m128 mDst, __m128 mSrc ) const;  // SSE

    inline __m128d CmpNLT( __m128d mDst, __m128d mSrc ) const;   // SSE2

    ////////////////////////////////////////////////////////////// Compare (Lesser-or-Equal)
    inline __m128 CmpLELower( __m128 mDst, __m128 mSrc ) const;     // SSE
    inline __m128d CmpLELower( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline __m128 CmpLE( __m128 mDst, __m128 mSrc ) const;  // SSE

    inline __m128d CmpLE( __m128d mDst, __m128d mSrc ) const;   // SSE2

    ////////////////////////////////////////////////////////////// Compare (Not Lesser-or-Equal)
    inline __m128 CmpNLELower( __m128 mDst, __m128 mSrc ) const;     // SSE
    inline __m128d CmpNLELower( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline __m128 CmpNLE( __m128 mDst, __m128 mSrc ) const;  // SSE

    inline __m128d CmpNLE( __m128d mDst, __m128d mSrc ) const;   // SSE2

    ////////////////////////////////////////////////////////////// Compare (Greater-Than)
    inline __m128 CmpGTLower( __m128 mDst, __m128 mSrc ) const;     // SSE
    inline __m128d CmpGTLower( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline __m128 CmpGT( __m128 mDst, __m128 mSrc ) const;  // SSE

    inline __m128d CmpGT( __m128d mDst, __m128d mSrc ) const;   // SSE2

    inline __m128i CmpGT8( __m128i mDst, __m128i mSrc ) const;  // SSE2
    inline __m256i CmpGT8( __m256i mDst, __m256i mSrc ) const;  // AVX2

    inline __m128i CmpGT16( __m128i mDst, __m128i mSrc ) const; // SSE2
    inline __m256i CmpGT16( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i CmpGT32( __m128i mDst, __m128i mSrc ) const; // SSE2
    inline __m256i CmpGT32( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i CmpGT64( __m128i mDst, __m128i mSrc ) const; // SSE42
    inline __m256i CmpGT64( __m256i mDst, __m256i mSrc ) const; // AVX2

    ////////////////////////////////////////////////////////////// Compare (Not Greater-Than)
    inline __m128 CmpNGTLower( __m128 mDst, __m128 mSrc ) const;     // SSE
    inline __m128d CmpNGTLower( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline __m128 CmpNGT( __m128 mDst, __m128 mSrc ) const;  // SSE

    inline __m128d CmpNGT( __m128d mDst, __m128d mSrc ) const;   // SSE2

    ////////////////////////////////////////////////////////////// Compare (Greater-or-Equal)
    inline __m128 CmpGELower( __m128 mDst, __m128 mSrc ) const;     // SSE
    inline __m128d CmpGELower( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline __m128 CmpGE( __m128 mDst, __m128 mSrc ) const;  // SSE

    inline __m128d CmpGE( __m128d mDst, __m128d mSrc ) const;   // SSE2

    ////////////////////////////////////////////////////////////// Compare (Not Greater-or-Equal)
    inline __m128 CmpNGELower( __m128 mDst, __m128 mSrc ) const;     // SSE
    inline __m128d CmpNGELower( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline __m128 CmpNGE( __m128 mDst, __m128 mSrc ) const;  // SSE

    inline __m128d CmpNGE( __m128d mDst, __m128d mSrc ) const;   // SSE2

    ////////////////////////////////////////////////////////////// Compare (Ordered)
    inline __m128 CmpORDLower( __m128 mDst, __m128 mSrc ) const;     // SSE
    inline __m128d CmpORDLower( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline __m128 CmpORD( __m128 mDst, __m128 mSrc ) const;  // SSE

    inline __m128d CmpORD( __m128d mDst, __m128d mSrc ) const;   // SSE2

    ////////////////////////////////////////////////////////////// Compare (Unordered)
    inline __m128 CmpUNORDLower( __m128 mDst, __m128 mSrc ) const;     // SSE
    inline __m128d CmpUNORDLower( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline __m128 CmpUNORD( __m128 mDst, __m128 mSrc ) const;  // SSE

    inline __m128d CmpUNORD( __m128d mDst, __m128d mSrc ) const;   // SSE2

    ////////////////////////////////////////////////////////////// Compare (Bool results, always on lower element, _Q = non signaling versions)
    inline Int IsEQ( __m128 mDst, __m128 mSrc ) const;      // SSE
    inline Int IsEQ_Q( __m128 mDst, __m128 mSrc ) const;    // SSE
    inline Int IsEQ( __m128d mDst, __m128d mSrc ) const;    // SSE2
    inline Int IsEQ_Q( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline Int IsNEQ( __m128 mDst, __m128 mSrc ) const;     // SSE
    inline Int IsNEQ_Q( __m128 mDst, __m128 mSrc ) const;   // SSE
    inline Int IsNEQ( __m128d mDst, __m128d mSrc ) const;   // SSE2
    inline Int IsNEQ_Q( __m128d mDst, __m128d mSrc ) const; // SSE2

    inline Int IsLT( __m128 mDst, __m128 mSrc ) const;      // SSE
    inline Int IsLT_Q( __m128 mDst, __m128 mSrc ) const;    // SSE
    inline Int IsLT( __m128d mDst, __m128d mSrc ) const;    // SSE2
    inline Int IsLT_Q( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline Int IsLE( __m128 mDst, __m128 mSrc ) const;      // SSE
    inline Int IsLE_Q( __m128 mDst, __m128 mSrc ) const;    // SSE
    inline Int IsLE( __m128d mDst, __m128d mSrc ) const;    // SSE2
    inline Int IsLE_Q( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline Int IsGT( __m128 mDst, __m128 mSrc ) const;      // SSE
    inline Int IsGT_Q( __m128 mDst, __m128 mSrc ) const;    // SSE
    inline Int IsGT( __m128d mDst, __m128d mSrc ) const;    // SSE2
    inline Int IsGT_Q( __m128d mDst, __m128d mSrc ) const;  // SSE2

    inline Int IsGE( __m128 mDst, __m128 mSrc ) const;      // SSE
    inline Int IsGE_Q( __m128 mDst, __m128 mSrc ) const;    // SSE
    inline Int IsGE( __m128d mDst, __m128d mSrc ) const;    // SSE2
    inline Int IsGE_Q( __m128d mDst, __m128d mSrc ) const;  // SSE2

    ////////////////////////////////////////////////////////////// Compare (Strings)

    ////////////////////////////////////////////////////////////// Minimum Value
    inline __m128 MinLower( __m128 mDst, __m128 mSrc ) const;       // SSE
    inline __m128d MinLower( __m128d mDst, __m128d mSrc ) const;    // SSE2

    inline __m128 Min( __m128 mDst, __m128 mSrc ) const; // SSE
    inline __m256 Min( __m256 mDst, __m256 mSrc ) const; // AVX

    inline __m128d Min( __m128d mDst, __m128d mSrc ) const; // SSE2
    inline __m256d Min( __m256d mDst, __m256d mSrc ) const; // AVX

    inline __m128i MinSigned8( __m128i mDst, __m128i mSrc ) const; // SSE41
    inline __m256i MinSigned8( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i MinSigned16( __m128i mDst, __m128i mSrc ) const; // SSE2
    inline __m256i MinSigned16( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i MinSigned32( __m128i mDst, __m128i mSrc ) const; // SSE41
    inline __m256i MinSigned32( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i MinSigned64( __m128i mDst, __m128i mSrc ) const; // SSE41
    inline __m256i MinSigned64( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i MinUnsigned8( __m128i mDst, __m128i mSrc ) const; // SSE2
    inline __m256i MinUnsigned8( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i MinUnsigned16( __m128i mDst, __m128i mSrc ) const; // SSE41
    inline __m256i MinUnsigned16( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i MinUnsigned32( __m128i mDst, __m128i mSrc ) const; // SSE41
    inline __m256i MinUnsigned32( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i MinUnsigned64( __m128i mDst, __m128i mSrc ) const; // SSE41
    inline __m256i MinUnsigned64( __m256i mDst, __m256i mSrc ) const; // AVX2

    ////////////////////////////////////////////////////////////// Maximum Value
    inline __m128 MaxLower( __m128 mDst, __m128 mSrc ) const;       // SSE
    inline __m128d MaxLower( __m128d mDst, __m128d mSrc ) const;    // SSE2

    inline __m128 Max( __m128 mDst, __m128 mSrc ) const; // SSE
    inline __m256 Max( __m256 mDst, __m256 mSrc ) const; // AVX

    inline __m128d Max( __m128d mDst, __m128d mSrc ) const; // SSE2
    inline __m256d Max( __m256d mDst, __m256d mSrc ) const; // AVX

    inline __m128i MaxSigned8( __m128i mDst, __m128i mSrc ) const; // SSE41
    inline __m256i MaxSigned8( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i MaxSigned16( __m128i mDst, __m128i mSrc ) const; // SSE2
    inline __m256i MaxSigned16( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i MaxSigned32( __m128i mDst, __m128i mSrc ) const; // SSE41
    inline __m256i MaxSigned32( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i MaxSigned64( __m128i mDst, __m128i mSrc ) const; // SSE41
    inline __m256i MaxSigned64( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i MaxUnsigned8( __m128i mDst, __m128i mSrc ) const; // SSE2
    inline __m256i MaxUnsigned8( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i MaxUnsigned16( __m128i mDst, __m128i mSrc ) const; // SSE41
    inline __m256i MaxUnsigned16( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i MaxUnsigned32( __m128i mDst, __m128i mSrc ) const; // SSE41
    inline __m256i MaxUnsigned32( __m256i mDst, __m256i mSrc ) const; // AVX2

    inline __m128i MaxUnsigned64( __m128i mDst, __m128i mSrc ) const; // SSE41
    inline __m256i MaxUnsigned64( __m256i mDst, __m256i mSrc ) const; // AVX2

    ////////////////////////////////////////////////////////////// Bitwise : And
    inline __m128 And( __m128 mDst, __m128 mSrc ) const; // SSE
    inline __m256 And( __m256 mDst, __m256 mSrc ) const; // AVX

    inline __m128d And( __m128d mDst, __m128d mSrc ) const; // SSE2
    inline __m256d And( __m256d mDst, __m256d mSrc ) const; // AVX

    inline __m128i And( __m128i mDst, __m128i mSrc ) const; // SSE2
    inline __m256i And( __m256i mDst, __m256i mSrc ) const; // AVX2

    ////////////////////////////////////////////////////////////// Bitwise : AndNot
    inline __m128 AndNot( __m128 mDst, __m128 mSrc ) const; // SSE
    inline __m256 AndNot( __m256 mDst, __m256 mSrc ) const; // AVX

    inline __m128d AndNot( __m128d mDst, __m128d mSrc ) const; // SSE2
    inline __m256d AndNot( __m256d mDst, __m256d mSrc ) const; // AVX

    inline __m128i AndNot( __m128i mDst, __m128i mSrc ) const; // SSE2
    inline __m256i AndNot( __m256i mDst, __m256i mSrc ) const; // AVX2

    ////////////////////////////////////////////////////////////// Bitwise : Or
    inline __m128 Or( __m128 mDst, __m128 mSrc ) const; // SSE
    inline __m256 Or( __m256 mDst, __m256 mSrc ) const; // AVX

    inline __m128d Or( __m128d mDst, __m128d mSrc ) const; // SSE2
    inline __m256d Or( __m256d mDst, __m256d mSrc ) const; // AVX

    inline __m128i Or( __m128i mDst, __m128i mSrc ) const; // SSE2
    inline __m256i Or( __m256i mDst, __m256i mSrc ) const; // AVX2

    ////////////////////////////////////////////////////////////// Bitwise : Xor
    inline __m128 Xor( __m128 mDst, __m128 mSrc ) const; // SSE
    inline __m256 Xor( __m256 mDst, __m256 mSrc ) const; // AVX

    inline __m128d Xor( __m128d mDst, __m128d mSrc ) const; // SSE2
    inline __m256d Xor( __m256d mDst, __m256d mSrc ) const; // AVX

    inline __m128i Xor( __m128i mDst, __m128i mSrc ) const; // SSE2
    inline __m256i Xor( __m256i mDst, __m256i mSrc ) const; // AVX2

    ////////////////////////////////////////////////////////////// Bitwise : Shift Left, Zero Extend
    inline __m128i Shift16L( __m128i mDst, Int iCount ) const;      // SSE2
    inline __m128i Shift16L( __m128i mDst, __m128i mCount ) const;  // SSE2
    inline __m256i Shift16L( __m256i mDst, Int iCount ) const;      // AVX2
    inline __m256i Shift16L( __m256i mDst, __m128i mCount ) const;  // AVX2

    inline __m128i Shift32L( __m128i mDst, Int iCount ) const;      // SSE2
    inline __m128i Shift32L( __m128i mDst, __m128i mCount ) const;  // SSE2
    inline __m256i Shift32L( __m256i mDst, Int iCount ) const;      // AVX2
    inline __m256i Shift32L( __m256i mDst, __m128i mCount ) const;  // AVX2

    inline __m128i Shift64L( __m128i mDst, Int iCount ) const;      // SSE2
    inline __m128i Shift64L( __m128i mDst, __m128i mCount ) const;  // SSE2
    inline __m256i Shift64L( __m256i mDst, Int iCount ) const;      // AVX2
    inline __m256i Shift64L( __m256i mDst, __m128i mCount ) const;  // AVX2

    inline __m128i Shift128L( __m128i mDst, Int iCount ) const; // SSE2
    inline __m256i Shift256L( __m256i mDst, Int iCount ) const; // AVX2

    inline __m128i ShiftV32L( __m128i mDst, __m128i mCounts ) const; // AVX2
    inline __m256i ShiftV32L( __m256i mDst, __m256i mCounts ) const; // AVX2

    inline __m128i ShiftV64L( __m128i mDst, __m128i mCounts ) const; // AVX2
    inline __m256i ShiftV64L( __m256i mDst, __m256i mCounts ) const; // AVX2

    ////////////////////////////////////////////////////////////// Bitwise : Shift Right, Zero Extend
    inline __m128i Shift16R( __m128i mDst, Int iCount ) const;      // SSE2
    inline __m128i Shift16R( __m128i mDst, __m128i mCount ) const;  // SSE2
    inline __m256i Shift16R( __m256i mDst, Int iCount ) const;      // AVX2
    inline __m256i Shift16R( __m256i mDst, __m128i mCount ) const;  // AVX2

    inline __m128i Shift32R( __m128i mDst, Int iCount ) const;      // SSE2
    inline __m128i Shift32R( __m128i mDst, __m128i mCount ) const;  // SSE2
    inline __m256i Shift32R( __m256i mDst, Int iCount ) const;      // AVX2
    inline __m256i Shift32R( __m256i mDst, __m128i mCount ) const;  // AVX2

    inline __m128i Shift64R( __m128i mDst, Int iCount ) const;      // SSE2
    inline __m128i Shift64R( __m128i mDst, __m128i mCount ) const;  // SSE2
    inline __m256i Shift64R( __m256i mDst, Int iCount ) const;      // AVX2
    inline __m256i Shift64R( __m256i mDst, __m128i mCount ) const;  // AVX2

    inline __m128i Shift128R( __m128i mDst, Int iCount ) const; // SSE2
    inline __m256i Shift256R( __m256i mDst, Int iCount ) const; // AVX2

    inline __m128i ShiftV32R( __m128i mDst, __m128i mCounts ) const; // AVX2
    inline __m256i ShiftV32R( __m256i mDst, __m256i mCounts ) const; // AVX2

    inline __m128i ShiftV64R( __m128i mDst, __m128i mCounts ) const; // AVX2
    inline __m256i ShiftV64R( __m256i mDst, __m256i mCounts ) const; // AVX2

    ////////////////////////////////////////////////////////////// Bitwise : Shift Right, Sign Extend
    inline __m128i Shift16RSE( __m128i mDst, Int iCount ) const;      // SSE2
    inline __m128i Shift16RSE( __m128i mDst, __m128i mCount ) const;  // SSE2
    inline __m256i Shift16RSE( __m256i mDst, Int iCount ) const;      // AVX2
    inline __m256i Shift16RSE( __m256i mDst, __m128i mCount ) const;  // AVX2

    inline __m128i Shift32RSE( __m128i mDst, Int iCount ) const;      // SSE2
    inline __m128i Shift32RSE( __m128i mDst, __m128i mCount ) const;  // SSE2
    inline __m256i Shift32RSE( __m256i mDst, Int iCount ) const;      // AVX2
    inline __m256i Shift32RSE( __m256i mDst, __m128i mCount ) const;  // AVX2

    inline __m128i ShiftV32RSE( __m128i mDst, __m128i mCounts ) const; // AVX2
    inline __m256i ShiftV32RSE( __m256i mDst, __m256i mCounts ) const; // AVX2

    ////////////////////////////////////////////////////////////// CRC32
    inline UInt32 CRC32( UInt32 iCRC, UInt8 iValue ) const;     // SSE42
    inline UInt32 CRC32( UInt32 iCRC, UInt16 iValue ) const;    // SSE42
    inline UInt32 CRC32( UInt32 iCRC, UInt32 iValue ) const;    // SSE42
    inline UInt64 CRC32( UInt64 iCRC, UInt64 iValue ) const;    // SSE42

    ////////////////////////////////////////////////////////////// Invert
    inline __m128 InvertLower( __m128 mValue ) const;  // SSE

    inline __m128 Invert( __m128 mValue ) const;    // SSE
    inline __m256 Invert( __m256 mValue ) const;    // AVX

    ////////////////////////////////////////////////////////////// Square Root
    inline __m128 SqrtLower( __m128 mValue ) const;    // SSE
    inline __m128d SqrtLower( __m128d mValue ) const; // SSE2

    inline __m128 Sqrt( __m128 mValue ) const;      // SSE
    inline __m256 Sqrt( __m256 mValue ) const;      // AVX

    inline __m128d Sqrt( __m128d mValue ) const;   // SSE2
    inline __m256d Sqrt( __m256d mValue ) const;   // AVX

    ////////////////////////////////////////////////////////////// Inverted SquareRoot
    inline __m128 InvSqrtLower( __m128 mValue ) const; // SSE

    inline __m128 InvSqrt( __m128 mValue ) const;   // SSE
    inline __m256 InvSqrt( __m256 mValue ) const;   // AVX

    inline __m128d InvSqrt( __m128d mValue ) const; // SSE
    inline __m256d InvSqrt( __m256d mValue ) const; // AVX

    ////////////////////////////////////////////////////////////// Cube Root
    inline __m128 Cbrt( __m128 mValue ) const;  // SSE
    inline __m256 Cbrt( __m256 mValue ) const;  // AVX

    inline __m128d Cbrt( __m128d mValue ) const;    // SSE
    inline __m256d Cbrt( __m256d mValue ) const;    // AVX

    ////////////////////////////////////////////////////////////// Inverted Cube Root
    inline __m128 InvCbrt( __m128 mValue ) const;   // SSE
    inline __m256 InvCbrt( __m256 mValue ) const;   // AVX

    inline __m128d InvCbrt( __m128d mValue ) const; // SSE
    inline __m256d InvCbrt( __m256d mValue ) const; // AVX

    ////////////////////////////////////////////////////////////// Hypothenus (Square Root of summed products)
    inline __m128 Hypot( __m128 mValue ) const;   // SSE
    inline __m256 Hypot( __m256 mValue ) const;   // AVX

    inline __m128d Hypot( __m128d mValue ) const; // SSE
    inline __m256d Hypot( __m256d mValue ) const; // AVX

    ////////////////////////////////////////////////////////////// Natural Logarithm
    inline __m128 Ln( __m128 mValue ) const;   // SSE
    inline __m256 Ln( __m256 mValue ) const;   // AVX

    inline __m128d Ln( __m128d mValue ) const; // SSE
    inline __m256d Ln( __m256d mValue ) const; // AVX

    ////////////////////////////////////////////////////////////// Logarithm Base 2
    inline __m128 Log2( __m128 mValue ) const;   // SSE
    inline __m256 Log2( __m256 mValue ) const;   // AVX

    inline __m128d Log2( __m128d mValue ) const; // SSE
    inline __m256d Log2( __m256d mValue ) const; // AVX

    ////////////////////////////////////////////////////////////// Logarithm Base 10
    inline __m128 Log10( __m128 mValue ) const;   // SSE
    inline __m256 Log10( __m256 mValue ) const;   // AVX

    inline __m128d Log10( __m128d mValue ) const; // SSE
    inline __m256d Log10( __m256d mValue ) const; // AVX

    ////////////////////////////////////////////////////////////// Natural Exponential
    inline __m128 Exp( __m128 mValue ) const;   // SSE
    inline __m256 Exp( __m256 mValue ) const;   // AVX

    inline __m128d Exp( __m128d mValue ) const; // SSE
    inline __m256d Exp( __m256d mValue ) const; // AVX

    ////////////////////////////////////////////////////////////// Exponential Base 2
    ////////////////////////////////////////////////////////////// Exponential Base 10
    ////////////////////////////////////////////////////////////// Exponential Any Base

    ////////////////////////////////////////////////////////////// Power

    ////////////////////////////////////////////////////////////// Sin
    ////////////////////////////////////////////////////////////// Cos
    ////////////////////////////////////////////////////////////// Tan
    ////////////////////////////////////////////////////////////// ArcSin
    ////////////////////////////////////////////////////////////// ArcCos
    ////////////////////////////////////////////////////////////// ArcTan
    ////////////////////////////////////////////////////////////// ArcTan2

    ////////////////////////////////////////////////////////////// SinH
    ////////////////////////////////////////////////////////////// CosH
    ////////////////////////////////////////////////////////////// TanH
    ////////////////////////////////////////////////////////////// ArcSinH
    ////////////////////////////////////////////////////////////// ArcCosH
    ////////////////////////////////////////////////////////////// ArcTanH

    ////////////////////////////////////////////////////////////// Gauss Error Function
    ////////////////////////////////////////////////////////////// Inverted Gauss Error Function






};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SSE.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SSE_H

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Design Choice : DO NOT USE SETR ! It's just confusing ... useless at best, bug mayhem in most cases.

// __m128 _mm_setr_ps(float, float, float, float)                                       - SSE
// __m256 _mm256_setr_ps(float, float, float, float, float, float, float, float)        - AVX
// __m128d _mm_setr_pd(double, double)                                                  - SSE2
// __m256d _mm256_setr_pd(double, double, double, double)                               - AVX
// __m128i _mm_setr_epi8(char, char, char, char, char, char, char, char,
//                       char, char, char, char, char, char, char, char)                - SSE2
// __m128i _mm_setr_epi16(short, short, short, short, short, short, short, short)       - SSE2
// __m128i _mm_setr_epi32(int, int, int, int)                                           - SSE2
// __m128i _mm_setr_epi64(int64, int64)                                                 - SSE2
// __m256i _mm256_setr_epi8(char, char, char, char, char, char, char, char,
//                          char, char, char, char, char, char, char, char,
//                          char, char, char, char, char, char, char, char,
//                          char, char, char, char, char, char, char, char)             - AVX
// __m256i _mm256_setr_epi16(short, short, short, short, short, short, short, short,
//                           short, short, short, short, short, short, short, short )   - AVX
// __m256i _mm256_setr_epi32(int, int, int, int, int, int, int, int)                    - AVX
// __m256i _mm256_setr_epi64(int64, int64, int64, int64)                                - AVX

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Deprecated stuff

// __m128i _mm_setl_epi64(__m128i)      - SSE2

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Leftover misc/weird instructions ... most likely introduced for very specific optimizations
// Will add them if ever needed ...

// void _mm_stream_si32(int*, int)                  - SSE2

// __m128i _mm_insert_si64(__m128i, __m128i)                - SSE4a
// __m128i _mm_inserti_si64(__m128i, __m128i, int, int)     - SSE4a
// __m128 _mm_insert_ps(__m128, __m128, const int)          - SSE41

// void _mm_maskmoveu_si128(__m128i, __m128i, char*)    - SSE2

// int _mm_movemask_ps(__m128)          - SSE
// int _mm_movemask_epi8(__m128i)       - SSE2
// int _mm_movemask_pd(__m128d)         - SSE2
// int _mm256_movemask_pd(__m256d)      - AVX
// int _mm256_movemask_ps(__m256)       - AVX
// int _mm256_movemask_epi8(__m256i)    - AVX2

// __m128i _mm_mulhrs_epi16(__m128i, __m128i)       - SSSE3
// __m256i _mm256_mulhrs_epi16(__m256i, __m256i)    - AVX2

// __m128i _mm_minpos_epu16(__m128i)    - SSE41

// __m128i _mm_alignr_epi8(__m128i, __m128i, int)         - SSSE3
// __m256i _mm256_alignr_epi8(__m256i, __m256i, const int)  - AVX2

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Unimplemented stuff ... Most likely will never use those ...

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPESTR
// int _mm_cmpestra(__m128i, int, __m128i, int, const int)      - SSE42
// int _mm_cmpestrc(__m128i, int, __m128i, int, const int)      - SSE42
// int _mm_cmpestri(__m128i, int, __m128i, int, const int)      - SSE42
// __m128i _mm_cmpestrm(__m128i, int, __m128i, int, const int)  - SSE42
// int _mm_cmpestro(__m128i, int, __m128i, int, const int)      - SSE42
// int _mm_cmpestrs(__m128i, int, __m128i, int, const int)      - SSE42
// int _mm_cmpestrz(__m128i, int, __m128i, int, const int)      - SSE42
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPISTR
// int _mm_cmpistra(__m128i, __m128i, const int)                - SSE42
// int _mm_cmpistrc(__m128i, __m128i, const int)                - SSE42
// int _mm_cmpistri(__m128i, __m128i, const int)                - SSE42
// __m128i _mm_cmpistrm(__m128i, __m128i, const int)            - SSE42
// int _mm_cmpistro(__m128i, __m128i, const int)                - SSE42
// int _mm_cmpistrs(__m128i, __m128i, const int)                - SSE42
// int _mm_cmpistrz(__m128i, __m128i, const int)                - SSE42

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// TESTC
// int _mm_testc_si128(__m128i, __m128i)        - SSE41
// int _mm_testc_pd(__m128d, __m128d)           - AVX
// int _mm_testc_ps(__m128, __m128)             - AVX
// int _mm256_testc_pd(__m256d, __m256d)        - AVX
// int _mm256_testc_ps(__m256, __m256)          - AVX
// int _mm256_testc_si256(__m256i, __m256i)     - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// TESTZ
// int _mm_testz_si128(__m128i, __m128i)        - SSE41
// int _mm_testz_pd(__m128d, __m128d)           - AVX
// int _mm_testz_ps(__m128, __m128)             - AVX
// int _mm256_testz_pd(__m256d, __m256d)        - AVX
// int _mm256_testz_ps(__m256, __m256)          - AVX
// int _mm256_testz_si256(__m256i, __m256i)     - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// TESTNZC
// int _mm_testnzc_si128(__m128i, __m128i)      - SSE41
// int _mm_testnzc_pd(__m128d, __m128d)         - AVX
// int _mm_testnzc_ps(__m128, __m128)           - AVX
// int _mm256_testnzc_pd(__m256d, __m256d)      - AVX
// int _mm256_testnzc_ps(__m256, __m256)        - AVX
// int _mm256_testnzc_si256(__m256i, __m256i)   - AVX

