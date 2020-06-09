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
#define SIMD_SHUFFLE_MASK_4( _selsrc_dst0, _selsrc_dst1, _selsrc_dst2, _selsrc_dst3 ) \
    ( ((_selsrc_dst0) & 0x03) | (((_selsrc_dst1) & 0x03) << 2) | (((_selsrc_dst2) & 0x03) << 4) | (((_selsrc_dst3) & 0x03) << 6) )

#define SIMD_SHUFFLE_MASK_2( _selsrc_dst0, _selsrc_dst1 ) \
    ( ((_selsrc_dst0) & 0x01) | (((_selsrc_dst1) & 0x01) << 1) )

#define SIMD_SHUFFLE_MASK_EMBED( _set_zero, _selsrc ) \
    ( ((_set_zero) ? 0x80 : 0) | ((_selsrc) & 0x0f) )

// Blend Control Mask



#define _SSE_IBLEND_MASK_PS( _pick_0, _pick_1, _pick_2, _pick_3 ) \
    ( ((_pick_0) & 0x01) | (((_pick_1) & 0x01) << 1) | (((_pick_2) & 0x01) << 2) | (((_pick_3) & 0x01) << 3) )
#define _SSE_IBLEND_MASK_PD( _pick_0, _pick_1 ) \
    ( ((_pick_0) & 0x01) | (((_pick_1) & 0x01) << 1) )

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
    ////////////////////////////////////////////////////////////// Serializing instruction (makes sure everything is flushed)
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

    ////////////////////////////////////////////////////////////// Values -> Registers
    inline __m128 SetLower( Float f0 ) const;   // SSE
    inline __m128d SetLower( Double f0 ) const; // SSE2

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
    inline __m256 Load256( const Float * arrF ) const; // AVX

    inline __m128d Load128( const Double * arrF ) const; // SSE2
    inline __m256d Load256( const Double * arrF ) const; // AVX

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
    inline __m128 Load32FourFloat( const Float * pSrc, __m128i mIndices, Int32 iStride ) const;  // AVX2
    inline __m256 Load32EightFloat( const Float * pSrc, __m256i mIndices, Int32 iStride ) const; // AVX2

    inline __m128d Load32TwoDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) const;  // AVX2
    inline __m256d Load32FourDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) const; // AVX2

    inline __m128i Load32FourInt32( const Int32 * pSrc, __m128i mIndices, Int32 iStride ) const;  // AVX2
    inline __m256i Load32EightInt32( const Int32 * pSrc, __m256i mIndices, Int32 iStride ) const; // AVX2

    inline __m128i Load32TwoInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) const;  // AVX2
    inline __m256i Load32FourInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) const; // AVX2

        // Sparse memory, 64-bit indices
    inline __m128 Load64TwoFloat( const Float * pSrc, __m128i mIndices, Int32 iStride ) const;  // AVX2
    inline __m128 Load64FourFloat( const Float * pSrc, __m256i mIndices, Int32 iStride ) const; // AVX2

    inline __m128d Load64TwoDouble( const Double * pSrc, __m128i mIndices, Int32 iStride ) const;  // AVX2
    inline __m256d Load64FourDouble( const Double * pSrc, __m256i mIndices, Int32 iStride ) const; // AVX2

    inline __m128i Load64TwoInt32( const Int32 * pSrc, __m128i mIndices, Int32 iStride ) const;  // AVX2
    inline __m128i Load64FourInt32( const Int32 * pSrc, __m256i mIndices, Int32 iStride ) const; // AVX2

    inline __m128i Load64TwoInt64( const Int64 * pSrc, __m128i mIndices, Int32 iStride ) const;  // AVX2
    inline __m256i Load64FourInt64( const Int64 * pSrc, __m256i mIndices, Int32 iStride ) const; // AVX2

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
    inline __m128 ShuffleFloat( __m128 mSrcLow, __m128 mSrcHigh, Int iMask4 ) const; // SSE
    inline __m256 ShuffleFloat( __m256 mSrcLow, __m256 mSrcHigh, Int iMask4 ) const; // AVX

    inline __m128d ShuffleDouble( __m128d mSrcLow, __m128d mSrcHigh, Int iMask2 ) const; // SSE2
    inline __m256d ShuffleDouble( __m256d mSrcEven, __m256d mSrcOdd, Int iMask4 ) const; // AVX

    inline __m128i ShuffleInt8( __m128i mSrc, __m128i mMaskEmbed ) const; // SSSE3
    inline __m256i ShuffleInt8( __m256i mSrc, __m256i mMaskEmbed ) const; // AVX2

    inline __m128i ShuffleInt16L( __m128i mSrc, Int iMask4 ) const; // SSE2
    inline __m256i ShuffleInt16L( __m256i mSrc, Int iMask4 ) const; // AVX2

    inline __m128i ShuffleInt16H( __m128i mSrc, Int iMask4 ) const; // SSE2
    inline __m256i ShuffleInt16H( __m256i mSrc, Int iMask4 ) const; // AVX2

    inline __m128i ShuffleInt32( __m128i mSrc, Int iMask4 ) const; // SSE2
    inline __m256i ShuffleInt32( __m256i mSrc, Int iMask4 ) const; // AVX2

    ////////////////////////////////////////////////////////////// Permutation



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


    // Convert operations

    // Cast operations (Free, 0 instruction generated)

    // Logical operations

    // Comparison operations

    // Rounding operations

    // Arithmetic operations

    ////////////////////////////////////////////////////////////// Invert & SquareRoot
    inline __m128 InvertLower( __m128 mFloat4 ) const;  // SSE

    inline __m128 Invert( __m128 mFloat4 ) const;    // SSE
    inline __m256 Invert( __m256 mFloat8 ) const;    // AVX

    inline __m128 SqrtLower( __m128 mFloat4 ) const;    // SSE
    inline __m128d SqrtLower( __m128d mDouble2 ) const; // SSE2

    inline __m128 Sqrt( __m128 mFloat4 ) const;      // SSE
    inline __m256 Sqrt( __m256 mFloat8 ) const;      // AVX

    inline __m128d Sqrt( __m128d mDouble2 ) const;   // SSE2
    inline __m256d Sqrt( __m256d mDouble4 ) const;   // AVX

    inline __m128 InvSqrtLower( __m128 mFloat4 ) const; // SSE

    inline __m128 InvSqrt( __m128 mFloat4 ) const;   // SSE
    inline __m256 InvSqrt( __m256 mFloat8 ) const;   // AVX

    

private:

};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SSE.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SSE_H

//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// MISC
// void _m_empty(void)      - MMX
// void _mm_empty (void)    - MMX*
//
// __m64 _m_from_int(int)   - MMX
// int _m_to_int(__m64)     - MMX
//
// void _mm_prefetch(char*, int)    - SSE
// unsigned int _mm_getcsr(void)    - SSE
// void _mm_setcsr(unsigned int)    - SSE
//
// void _mm_clflush(void const *)   - SSE2
// void _mm_pause(void)             - SSE2
//
// void _mm_monitor(void const*, unsigned int, unsigned int)    - SSE3
// void _mm_mwait(unsigned int, unsigned int)                   - SSE3
//
// __m128i _mm_minpos_epu16(__m128i)    - SSE41
//
// void _mm256_zeroall(void)    - AVX
// void _mm256_zeroupper(void)  - AVX
//
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
// __m64 _mm_movepi64_pi64(__m128i)             - SSE2
// __m128i _mm_movpi64_epi64(__m64)             - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// STREAM
// void _mm_stream_pi(__m64*, __m64)                - SSE
// void _mm_stream_si32(int*, int)                  - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// INSERT
// __m128i _mm_insert_si64(__m128i, __m128i)                - SSE4a
// __m128i _mm_inserti_si64(__m128i, __m128i, int, int)     - SSE4a
// __m128 _mm_insert_ps(__m128, __m128, const int)          - SSE41
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// GATHER MASK
// __m128i _mm_mask_i32gather_epi32(__m128i, int const *, __m128i, __m128i, const int)          - AVX2
// __m128i _mm_mask_i32gather_epi64(__m128i, __int64 const *, __m128i, __m128i, const int)      - AVX2
// __m128d _mm_mask_i32gather_pd(__m128d, double const *, __m128i, __m128d, const int)          - AVX2
// __m128 _mm_mask_i32gather_ps(__m128, float const *, __m128i, __m128, const int)              - AVX2
// __m256i _mm256_mask_i32gather_epi32(__m256i, int const *, __m256i, __m256i, const int)       - AVX2
// __m256i _mm256_mask_i32gather_epi64(__m256i, __int64 const *, __m128i, __m256i, const int)   - AVX2
// __m256d _mm256_mask_i32gather_pd(__m256d, double const *, __m128i, __m256d, const int)       - AVX2
// __m256 _mm256_mask_i32gather_ps(__m256, float const *, __m256i, __m256, const int)           - AVX2
// __m128i _mm_mask_i64gather_epi32(__m128i, int const *, __m128i, __m128i, const int)          - AVX2
// __m128i _mm_mask_i64gather_epi64(__m128i, __int64 const *, __m128i, __m128i, const int)      - AVX2
// __m128d _mm_mask_i64gather_pd(__m128d, double const *, __m128i, __m128d, const int)          - AVX2
// __m128 _mm_mask_i64gather_ps(__m128, float const *, __m128i, __m128, const int)              - AVX2
// __m128i _mm256_mask_i64gather_epi32(__m128i, int const *, __m256i, __m128i, const int)       - AVX2
// __m256i _mm256_mask_i64gather_epi64(__m256i, __int64 const *, __m256i, __m256i, const int)   - AVX2
// __m256d _mm256_mask_i64gather_pd(__m256d, double const *, __m256i, __m256d, const int)       - AVX2
// __m128 _mm256_mask_i64gather_ps(__m128, float const *, __m256i, __m128, const int)           - AVX2
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
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMP
// __m128d _mm_cmp_pd(__m128d, __m128d, const int)              - AVX
// __m128 _mm_cmp_ps(__m128, __m128, const int)                 - AVX
// __m128d _mm_cmp_sd(__m128d, __m128d, const int)              - AVX
// __m128 _mm_cmp_ss(__m128, __m128, const int)                 - AVX
// __m256d _mm256_cmp_pd(__m256d, __m256d, const int)           - AVX
// __m256 _mm256_cmp_ps(__m256, __m256, const int)              - AVX
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPEQ
// __m64 _m_pcmpeqb(__m64, __m64)                               - MMX
// __m64 _m_pcmpeqd(__m64, __m64)                               - MMX
// __m64 _m_pcmpeqw(__m64, __m64)                               - MMX
// __m64 _mm_cmpeq_pi8(__m64, __m64)                            - MMX*
// __m64 _mm_cmpeq_pi16(__m64, __m64)                           - MMX*
// __m64 _mm_cmpeq_pi32(__m64, __m64)                           - MMX*
// __m128 _mm_cmpeq_ps(__m128, __m128)                          - SSE
// __m128 _mm_cmpeq_ss(__m128, __m128)                          - SSE
// __m128i _mm_cmpeq_epi16(__m128i, __m128i)                    - SSE2
// __m128i _mm_cmpeq_epi32(__m128i, __m128i)                    - SSE2
// __m128i _mm_cmpeq_epi8(__m128i, __m128i)                     - SSE2
// __m128d _mm_cmpeq_pd(__m128d, __m128d)                       - SSE2
// __m128d _mm_cmpeq_sd(__m128d, __m128d)                       - SSE2
// __m128i _mm_cmpeq_epi64(__m128i, __m128i)                    - SSE41
// __m256i _mm256_cmpeq_epi16(__m256i, __m256i)                 - AVX2
// __m256i _mm256_cmpeq_epi32(__m256i, __m256i)                 - AVX2
// __m256i _mm256_cmpeq_epi64(__m256i, __m256i)                 - AVX2
// __m256i _mm256_cmpeq_epi8(__m256i, __m256i)                  - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPNEQ
// __m128 _mm_cmpneq_ps(__m128, __m128)                         - SSE
// __m128 _mm_cmpneq_ss(__m128, __m128)                         - SSE
// __m128d _mm_cmpneq_pd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpneq_sd(__m128d, __m128d)                      - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPGT
// __m64 _m_pcmpgtb(__m64, __m64)                               - MMX
// __m64 _m_pcmpgtd(__m64, __m64)                               - MMX
// __m64 _m_pcmpgtw(__m64, __m64)                               - MMX
// __m64 _mm_cmpgt_pi8(__m64, __m64)                            - MMX*
// __m64 _mm_cmpgt_pi16(__m64, __m64)                           - MMX*
// __m64 _mm_cmpgt_pi32(__m64, __m64)                           - MMX*
// __m128 _mm_cmpgt_ps(__m128, __m128)                          - SSE
// __m128 _mm_cmpgt_ss(__m128, __m128)                          - SSE
// __m128i _mm_cmpgt_epi16(__m128i, __m128i)                    - SSE2
// __m128i _mm_cmpgt_epi32(__m128i, __m128i)                    - SSE2
// __m128i _mm_cmpgt_epi8(__m128i, __m128i)                     - SSE2
// __m128d _mm_cmpgt_pd(__m128d, __m128d)                       - SSE2
// __m128d _mm_cmpgt_sd(__m128d, __m128d)                       - SSE2
// __m128i _mm_cmpgt_epi64(__m128i, __m128i)                    - SSE42
// __m256i _mm256_cmpgt_epi16(__m256i, __m256i)                 - AVX2
// __m256i _mm256_cmpgt_epi32(__m256i, __m256i)                 - AVX2
// __m256i _mm256_cmpgt_epi64(__m256i, __m256i)                 - AVX2
// __m256i _mm256_cmpgt_epi8(__m256i, __m256i)                  - AVX2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPNGT
// __m128 _mm_cmpngt_ps(__m128, __m128)                         - SSE
// __m128 _mm_cmpngt_ss(__m128, __m128)                         - SSE
// __m128d _mm_cmpngt_pd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpngt_sd(__m128d, __m128d)                      - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPGE
// __m128 _mm_cmpge_ps(__m128, __m128)                          - SSE
// __m128 _mm_cmpge_ss(__m128, __m128)                          - SSE
// __m128d _mm_cmpge_pd(__m128d, __m128d)                       - SSE2
// __m128d _mm_cmpge_sd(__m128d, __m128d)                       - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPNGE
// __m128 _mm_cmpnge_ps(__m128, __m128)                         - SSE
// __m128 _mm_cmpnge_ss(__m128, __m128)                         - SSE
// __m128d _mm_cmpnge_pd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpnge_sd(__m128d, __m128d)                      - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPLT
// __m128 _mm_cmplt_ps(__m128, __m128)                          - SSE
// __m128 _mm_cmplt_ss(__m128, __m128)                          - SSE
// __m128i _mm_cmplt_epi16(__m128i, __m128i)                    - SSE2
// __m128i _mm_cmplt_epi32(__m128i, __m128i)                    - SSE2
// __m128i _mm_cmplt_epi8(__m128i, __m128i)                     - SSE2
// __m128d _mm_cmplt_pd(__m128d, __m128d)                       - SSE2
// __m128d _mm_cmplt_sd(__m128d, __m128d)                       - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPNLT
// __m128 _mm_cmpnlt_ps(__m128, __m128)                         - SSE
// __m128 _mm_cmpnlt_ss(__m128, __m128)                         - SSE
// __m128d _mm_cmpnlt_pd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpnlt_sd(__m128d, __m128d)                      - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPLE
// __m128 _mm_cmple_ps(__m128, __m128)                          - SSE
// __m128 _mm_cmple_ss(__m128, __m128)                          - SSE
// __m128d _mm_cmple_pd(__m128d, __m128d)                       - SSE2
// __m128d _mm_cmple_sd(__m128d, __m128d)                       - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPNLE
// __m128 _mm_cmpnle_ps(__m128, __m128)                         - SSE
// __m128 _mm_cmpnle_ss(__m128, __m128)                         - SSE
// __m128d _mm_cmpnle_pd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpnle_sd(__m128d, __m128d)                      - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPORD
// __m128 _mm_cmpord_ps(__m128, __m128)                         - SSE
// __m128 _mm_cmpord_ss(__m128, __m128)                         - SSE
// __m128d _mm_cmpord_pd(__m128d, __m128d)                      - SSE2
// __m128d _mm_cmpord_sd(__m128d, __m128d)                      - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CMPUNORD
// __m128 _mm_cmpunord_ps(__m128, __m128)                       - SSE
// __m128 _mm_cmpunord_ss(__m128, __m128)                       - SSE
// __m128d _mm_cmpunord_pd(__m128d, __m128d)                    - SSE2
// __m128d _mm_cmpunord_sd(__m128d, __m128d)                    - SSE2
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
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// COMIEQ
// int _mm_comieq_ss(__m128, __m128)        - SSE
// int _mm_comieq_sd(__m128d, __m128d)      - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// COMINEQ
// int _mm_comineq_ss(__m128, __m128)       - SSE
// int _mm_comineq_sd(__m128d, __m128d)     - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// COMIGT
// int _mm_comigt_ss(__m128, __m128)        - SSE
// int _mm_comigt_sd(__m128d, __m128d)      - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// COMIGE
// int _mm_comige_ss(__m128, __m128)        - SSE
// int _mm_comige_sd(__m128d, __m128d)      - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// COMILT
// int _mm_comilt_ss(__m128, __m128)        - SSE
// int _mm_comilt_sd(__m128d, __m128d)      - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// COMILE
// int _mm_comile_ss(__m128, __m128)        - SSE
// int _mm_comile_sd(__m128d, __m128d)      - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// UCOMIEQ
// int _mm_ucomieq_ss(__m128, __m128)       - SSE
// int _mm_ucomieq_sd(__m128d, __m128d)     - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// UCOMINEQ
// int _mm_ucomineq_ss(__m128, __m128)      - SSE
// int _mm_ucomineq_sd(__m128d, __m128d)    - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// UCOMIGT
// int _mm_ucomigt_ss(__m128, __m128)       - SSE
// int _mm_ucomigt_sd(__m128d, __m128d)     - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// UCOMIGE
// int _mm_ucomige_ss(__m128, __m128)       - SSE
// int _mm_ucomige_sd(__m128d, __m128d)     - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// UCOMILT
// int _mm_ucomilt_ss(__m128, __m128)       - SSE
// int _mm_ucomilt_sd(__m128d, __m128d)     - SSE2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// UCOMILE
// int _mm_ucomile_ss(__m128, __m128)       - SSE
// int _mm_ucomile_sd(__m128d, __m128d)     - SSE2
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
// 
// 