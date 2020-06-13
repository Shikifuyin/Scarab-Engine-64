/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_ExportMemory.h
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
// Header prelude
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDEXPORTMEMORY_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDEXPORTMEMORY_H

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <intrin.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../CPUID.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The SIMD::Export::Memory namespace
namespace SIMD { namespace Export { namespace Memory {

	// Lower Element
	inline Void SaveOne( Float * outDst, __m128 mSrc );   // SSE
    inline Void SaveOne( Double * outDst, __m128d mSrc ); // SSE2

	// Specific (mostly historic ...)
    inline Void SaveOneDoubleL( Double * outDst, __m128d mSrc ); // SSE2
    inline Void SaveOneDoubleH( Double * outDst, __m128d mSrc ); // SSE2

    inline Void SaveOneInt64L( __m128i * outDst, __m128i mSrc ); // SSE2

	// Contiguous Memory
	inline Void Save128( Float * outDst, __m128 mSrc ); // SSE

    inline Void Save128( Double * outDst, __m128d mSrc ); // SSE2

    inline Void Save128( Int8 * outDst, __m128i mSrc );  // SSE2
    inline Void Save128( Int16 * outDst, __m128i mSrc ); // SSE2
    inline Void Save128( Int32 * outDst, __m128i mSrc ); // SSE2
    inline Void Save128( Int64 * outDst, __m128i mSrc ); // SSE2

    inline Void Save128( UInt8 * outDst, __m128i mSrc );  // SSE2
    inline Void Save128( UInt16 * outDst, __m128i mSrc ); // SSE2
    inline Void Save128( UInt32 * outDst, __m128i mSrc ); // SSE2
    inline Void Save128( UInt64 * outDst, __m128i mSrc ); // SSE2

    inline Void Save256( Float * outDst, __m256 mSrc ); // AVX

    inline Void Save256( Double * outDst, __m256d mSrc ); // AVX

    inline Void Save256( Int8 * outDst, __m256i mSrc );  // AVX
    inline Void Save256( Int16 * outDst, __m256i mSrc ); // AVX
    inline Void Save256( Int32 * outDst, __m256i mSrc ); // AVX
    inline Void Save256( Int64 * outDst, __m256i mSrc ); // AVX

    inline Void Save256( UInt8 * outDst, __m256i mSrc );  // AVX
    inline Void Save256( UInt16 * outDst, __m256i mSrc ); // AVX
    inline Void Save256( UInt32 * outDst, __m256i mSrc ); // AVX
    inline Void Save256( UInt64 * outDst, __m256i mSrc ); // AVX

	// Mask variants
    inline Void Save128( Float * outDst, __m128 mSrc, __m128i mSigns ); // AVX

    inline Void Save128( Double * outDst, __m128d mSrc, __m128i mSigns ); // AVX

    inline Void Save128( Int32 * outDst, __m128i mSrc, __m128i mSigns ); // AVX2
    inline Void Save128( Int64 * outDst, __m128i mSrc, __m128i mSigns ); // AVX2

    inline Void Save256( Float * outDst, __m256 mSrc, __m256i mSigns ); // AVX

    inline Void Save256( Double * outDst, __m256d mSrc, __m256i mSigns ); // AVX

    inline Void Save256( Int32 * outDst, __m256i mSrc, __m256i mSigns ); // AVX2
    inline Void Save256( Int64 * outDst, __m256i mSrc, __m256i mSigns ); // AVX2

    // Aligned Memory
    namespace Aligned {

		inline Void Save128( Float * outDst, __m128 mSrc ); // SSE

		inline Void Save128( Double * outDst, __m128d mSrc ); // SSE2

        inline Void Save128( Int8 * outDst, __m128i mSrc );  // SSE2
		inline Void Save128( Int16 * outDst, __m128i mSrc ); // SSE2
		inline Void Save128( Int32 * outDst, __m128i mSrc ); // SSE2
		inline Void Save128( Int64 * outDst, __m128i mSrc ); // SSE2

		inline Void Save128( UInt8 * outDst, __m128i mSrc );  // SSE2
		inline Void Save128( UInt16 * outDst, __m128i mSrc ); // SSE2
		inline Void Save128( UInt32 * outDst, __m128i mSrc ); // SSE2
		inline Void Save128( UInt64 * outDst, __m128i mSrc ); // SSE2

		inline Void Save256( Float * outDst, __m256 mSrc ); // AVX

		inline Void Save256( Double * outDst, __m256d mSrc ); // AVX

        inline Void Save256( Int8 * outDst, __m256i mSrc );  // AVX
		inline Void Save256( Int16 * outDst, __m256i mSrc ); // AVX
		inline Void Save256( Int32 * outDst, __m256i mSrc ); // AVX
		inline Void Save256( Int64 * outDst, __m256i mSrc ); // AVX

		inline Void Save256( UInt8 * outDst, __m256i mSrc );  // AVX
		inline Void Save256( UInt16 * outDst, __m256i mSrc ); // AVX
		inline Void Save256( UInt32 * outDst, __m256i mSrc ); // AVX
		inline Void Save256( UInt64 * outDst, __m256i mSrc ); // AVX

		// Reversed variants
		inline Void Save128R( Float * outDst, __m128 mSrc );   // SSE
		inline Void Save128R( Double * outDst, __m128d mSrc ); // SSE2

		// Non-Temporal variants (stream instructions)
        inline Void SaveOneNT( Float * outDst, __m128 mSrc );   // SSE4a
        inline Void SaveOneNT( Double * outDst, __m128d mSrc ); // SSE4a

        inline Void Save128NT( Float * outDst, __m128 mSrc ); // SSE

        inline Void Save128NT( Double * outDst, __m128d mSrc ); // SSE2

        inline Void Save128NT( Int8 * outDst, __m128i mSrc );  // SSE2
        inline Void Save128NT( Int16 * outDst, __m128i mSrc ); // SSE2
        inline Void Save128NT( Int32 * outDst, __m128i mSrc ); // SSE2
        inline Void Save128NT( Int64 * outDst, __m128i mSrc ); // SSE2

        inline Void Save128NT( UInt8 * outDst, __m128i mSrc );  // SSE2
        inline Void Save128NT( UInt16 * outDst, __m128i mSrc ); // SSE2
        inline Void Save128NT( UInt32 * outDst, __m128i mSrc ); // SSE2
        inline Void Save128NT( UInt64 * outDst, __m128i mSrc ); // SSE2

        inline Void Save256NT( Float * outDst, __m256 mSrc ); // AVX

        inline Void Save256NT( Double * outDst, __m256d mSrc ); // AVX

        inline Void Save256NT( Int8 * outDst, __m256i mSrc );  // AVX
        inline Void Save256NT( Int16 * outDst, __m256i mSrc ); // AVX
        inline Void Save256NT( Int32 * outDst, __m256i mSrc ); // AVX
        inline Void Save256NT( Int64 * outDst, __m256i mSrc ); // AVX

        inline Void Save256NT( UInt8 * outDst, __m256i mSrc );  // AVX
        inline Void Save256NT( UInt16 * outDst, __m256i mSrc ); // AVX
        inline Void Save256NT( UInt32 * outDst, __m256i mSrc ); // AVX
        inline Void Save256NT( UInt64 * outDst, __m256i mSrc ); // AVX

        // Spread Values
	    inline Void Spread128( Float * outDst, __m128 mSrc );   // SSE
        inline Void Spread128( Double * outDst, __m128d mSrc ); // SSE2

    };

}; }; };

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SIMD_ExportMemory.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDEXPORTMEMORY_H
