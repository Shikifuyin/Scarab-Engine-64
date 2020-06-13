/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_ImportMemory.h
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
// Header prelude
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDIMPORTMEMORY_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDIMPORTMEMORY_H

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <intrin.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../CPUID.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The SIMD::Import::Memory namespace
namespace SIMD { namespace Import { namespace Memory {

	// Lower Element
	inline __m128 LoadOne( const Float * pSrc );   // SSE
    inline __m128d LoadOne( const Double * pSrc ); // SSE2

	// Specific (mostly historic ...)
	inline __m128d LoadOneDoubleL( __m128d mDst, const Double * pSrc ); // SSE2
    inline __m128d LoadOneDoubleH( __m128d mDst, const Double * pSrc ); // SSE2

    inline __m128i LoadOneInt64L( const __m128i * pSrc ); // SSE2

	// Contiguous Memory
	inline __m128 Load128( const Float * pSrc ); // SSE

    inline __m128d Load128( const Double * pSrc ); // SSE2

    inline __m128i Load128( const Int8 * pSrc );  // SSE3
    inline __m128i Load128( const Int16 * pSrc ); // SSE3
    inline __m128i Load128( const Int32 * pSrc ); // SSE3
    inline __m128i Load128( const Int64 * pSrc ); // SSE3

    inline __m128i Load128( const UInt8 * pSrc );  // SSE3
    inline __m128i Load128( const UInt16 * pSrc ); // SSE3
    inline __m128i Load128( const UInt32 * pSrc ); // SSE3
    inline __m128i Load128( const UInt64 * pSrc ); // SSE3

    inline __m256 Load256( const Float * pSrc ); // AVX

    inline __m256d Load256( const Double * pSrc ); // AVX

    inline __m256i Load256( const Int8 * pSrc );  // AVX
    inline __m256i Load256( const Int16 * pSrc ); // AVX
    inline __m256i Load256( const Int32 * pSrc ); // AVX
    inline __m256i Load256( const Int64 * pSrc ); // AVX

    inline __m256i Load256( const UInt8 * pSrc );  // AVX
    inline __m256i Load256( const UInt16 * pSrc ); // AVX
    inline __m256i Load256( const UInt32 * pSrc ); // AVX
    inline __m256i Load256( const UInt64 * pSrc ); // AVX

	// Mask variants
    inline __m128 Load128( const Float * pSrc, __m128i mSigns ); // AVX

    inline __m128d Load128( const Double * pSrc, __m128i mSigns ); // AVX

    inline __m128i Load128( const Int32 * pSrc, __m128i mSigns ); // AVX2
    inline __m128i Load128( const Int64 * pSrc, __m128i mSigns ); // AVX2

    inline __m256 Load256( const Float * pSrc, __m256i mSigns ); // AVX

    inline __m256d Load256( const Double * pSrc, __m256i mSigns ); // AVX

    inline __m256i Load256( const Int32 * pSrc, __m256i mSigns ); // AVX2
    inline __m256i Load256( const Int64 * pSrc, __m256i mSigns ); // AVX2

	// Spread Values
	inline __m128 Spread128( const Float * pSrc ); // SSE or AVX

    inline __m128d Spread128( const Double * pSrc ); // SSE3

    inline __m256 Spread256( const Float * pSrc );  // AVX
    inline __m256 Spread256( const __m128 * pSrc ); // AVX

    inline __m256d Spread256( const Double * pSrc );  // AVX
    inline __m256d Spread256( const __m128d * pSrc ); // AVX

	// Aligned Memory
	namespace Aligned {

		inline __m128 Load128( const Float * pSrc ); // SSE

		inline __m128d Load128( const Double * pSrc ); // SSE2

        inline __m128i Load128( const Int8 * pSrc );  // SSE2
		inline __m128i Load128( const Int16 * pSrc ); // SSE2
		inline __m128i Load128( const Int32 * pSrc ); // SSE2
		inline __m128i Load128( const Int64 * pSrc ); // SSE2

		inline __m128i Load128( const UInt8 * pSrc );  // SSE2
		inline __m128i Load128( const UInt16 * pSrc ); // SSE2
		inline __m128i Load128( const UInt32 * pSrc ); // SSE2
		inline __m128i Load128( const UInt64 * pSrc ); // SSE2

		inline __m256 Load256( const Float * pSrc ); // AVX

		inline __m256d Load256( const Double * pSrc ); // AVX

        inline __m256i Load256( const Int8 * pSrc );  // AVX
		inline __m256i Load256( const Int16 * pSrc ); // AVX
		inline __m256i Load256( const Int32 * pSrc ); // AVX
		inline __m256i Load256( const Int64 * pSrc ); // AVX

		inline __m256i Load256( const UInt8 * pSrc );  // AVX
		inline __m256i Load256( const UInt16 * pSrc ); // AVX
		inline __m256i Load256( const UInt32 * pSrc ); // AVX
		inline __m256i Load256( const UInt64 * pSrc ); // AVX

		// Reversed variants
		inline __m128 Load128R( const Float * pSrc );   // SSE
		inline __m128d Load128R( const Double * pSrc ); // SSE2

		// Non-Temporal variants (stream_load instructions)
		inline __m128 Load128NT( const Float * pSrc ); // SSE41

		inline __m128d Load128NT( const Double * pSrc ); // SSE41

        inline __m128i Load128NT( const Int8 * pSrc );  // SSE41
		inline __m128i Load128NT( const Int16 * pSrc ); // SSE41
		inline __m128i Load128NT( const Int32 * pSrc ); // SSE41
		inline __m128i Load128NT( const Int64 * pSrc ); // SSE41

		inline __m128i Load128NT( const UInt8 * pSrc );  // SSE41
		inline __m128i Load128NT( const UInt16 * pSrc ); // SSE41
		inline __m128i Load128NT( const UInt32 * pSrc ); // SSE41
		inline __m128i Load128NT( const UInt64 * pSrc ); // SSE41

		inline __m256 Load256NT( const Float * pSrc ); // AVX2

		inline __m256d Load256NT( const Double * pSrc ); // AVX2

        inline __m256i Load256NT( const Int8 * pSrc );  // AVX2
		inline __m256i Load256NT( const Int16 * pSrc ); // AVX2
		inline __m256i Load256NT( const Int32 * pSrc ); // AVX2
		inline __m256i Load256NT( const Int64 * pSrc ); // AVX2

		inline __m256i Load256NT( const UInt8 * pSrc );  // AVX2
		inline __m256i Load256NT( const UInt16 * pSrc ); // AVX2
		inline __m256i Load256NT( const UInt32 * pSrc ); // AVX2
		inline __m256i Load256NT( const UInt64 * pSrc ); // AVX2

	};

	// Sparse Memory, 32-bits indices
	namespace Sparse32 {
		namespace Stride1 {

			inline __m128 Load128( const Float * pSrc, __m128i mIndices ); // AVX2

            inline __m128d Load128( const Double * pSrc, __m128i mIndices ); // AVX2

            inline __m128i Load128( const Int32 * pSrc, __m128i mIndices ); // AVX2
            inline __m128i Load128( const Int64 * pSrc, __m128i mIndices ); // AVX2

            inline __m256 Load256( const Float * pSrc, __m256i mIndices ); // AVX2

            inline __m256d Load256( const Double * pSrc, __m128i mIndices ); // AVX2

            inline __m256i Load256( const Int32 * pSrc, __m256i mIndices ); // AVX2
            inline __m256i Load256( const Int64 * pSrc, __m128i mIndices ); // AVX2

	        // Mask variants
            inline __m128 Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ); // AVX2

            inline __m128d Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ); // AVX2

            inline __m128i Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2
            inline __m128i Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2

            inline __m256 Load256( __m256 mDst, const Float * pSrc, __m256i mIndices, __m256 mSigns ); // AVX2

            inline __m256d Load256( __m256d mDst, const Double * pSrc, __m128i mIndices, __m256d mSigns ); // AVX2

            inline __m256i Load256( __m256i mDst, const Int32 * pSrc, __m256i mIndices, __m256i mSigns ); // AVX2
            inline __m256i Load256( __m256i mDst, const Int64 * pSrc, __m128i mIndices, __m256i mSigns ); // AVX2

		};
		namespace Stride2 {

			inline __m128 Load128( const Float * pSrc, __m128i mIndices ); // AVX2

            inline __m128d Load128( const Double * pSrc, __m128i mIndices ); // AVX2

            inline __m128i Load128( const Int32 * pSrc, __m128i mIndices ); // AVX2
            inline __m128i Load128( const Int64 * pSrc, __m128i mIndices ); // AVX2

            inline __m256 Load256( const Float * pSrc, __m256i mIndices ); // AVX2

            inline __m256d Load256( const Double * pSrc, __m128i mIndices ); // AVX2

            inline __m256i Load256( const Int32 * pSrc, __m256i mIndices ); // AVX2
            inline __m256i Load256( const Int64 * pSrc, __m128i mIndices ); // AVX2

            // Mask variants
            inline __m128 Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ); // AVX2

            inline __m128d Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ); // AVX2

            inline __m128i Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2
            inline __m128i Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2

            inline __m256 Load256( __m256 mDst, const Float * pSrc, __m256i mIndices, __m256 mSigns ); // AVX2

            inline __m256d Load256( __m256d mDst, const Double * pSrc, __m128i mIndices, __m256d mSigns ); // AVX2

            inline __m256i Load256( __m256i mDst, const Int32 * pSrc, __m256i mIndices, __m256i mSigns ); // AVX2
            inline __m256i Load256( __m256i mDst, const Int64 * pSrc, __m128i mIndices, __m256i mSigns ); // AVX2

		};
		namespace Stride4 {

			inline __m128 Load128( const Float * pSrc, __m128i mIndices ); // AVX2

            inline __m128d Load128( const Double * pSrc, __m128i mIndices ); // AVX2

            inline __m128i Load128( const Int32 * pSrc, __m128i mIndices ); // AVX2
            inline __m128i Load128( const Int64 * pSrc, __m128i mIndices ); // AVX2

            inline __m256 Load256( const Float * pSrc, __m256i mIndices ); // AVX2

            inline __m256d Load256( const Double * pSrc, __m128i mIndices ); // AVX2

            inline __m256i Load256( const Int32 * pSrc, __m256i mIndices ); // AVX2
            inline __m256i Load256( const Int64 * pSrc, __m128i mIndices ); // AVX2

            // Mask variants
            inline __m128 Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ); // AVX2

            inline __m128d Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ); // AVX2

            inline __m128i Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2
            inline __m128i Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2

            inline __m256 Load256( __m256 mDst, const Float * pSrc, __m256i mIndices, __m256 mSigns ); // AVX2

            inline __m256d Load256( __m256d mDst, const Double * pSrc, __m128i mIndices, __m256d mSigns ); // AVX2

            inline __m256i Load256( __m256i mDst, const Int32 * pSrc, __m256i mIndices, __m256i mSigns ); // AVX2
            inline __m256i Load256( __m256i mDst, const Int64 * pSrc, __m128i mIndices, __m256i mSigns ); // AVX2

		};
		namespace Stride8 {

            inline __m128 Load128( const Float * pSrc, __m128i mIndices ); // AVX2

            inline __m128d Load128( const Double * pSrc, __m128i mIndices ); // AVX2

            inline __m128i Load128( const Int32 * pSrc, __m128i mIndices ); // AVX2
            inline __m128i Load128( const Int64 * pSrc, __m128i mIndices ); // AVX2

            inline __m256 Load256( const Float * pSrc, __m256i mIndices ); // AVX2

            inline __m256d Load256( const Double * pSrc, __m128i mIndices ); // AVX2

            inline __m256i Load256( const Int32 * pSrc, __m256i mIndices ); // AVX2
            inline __m256i Load256( const Int64 * pSrc, __m128i mIndices ); // AVX2

            // Mask variants
            inline __m128 Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ); // AVX2

            inline __m128d Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ); // AVX2

            inline __m128i Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2
            inline __m128i Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2

            inline __m256 Load256( __m256 mDst, const Float * pSrc, __m256i mIndices, __m256 mSigns ); // AVX2

            inline __m256d Load256( __m256d mDst, const Double * pSrc, __m128i mIndices, __m256d mSigns ); // AVX2

            inline __m256i Load256( __m256i mDst, const Int32 * pSrc, __m256i mIndices, __m256i mSigns ); // AVX2
            inline __m256i Load256( __m256i mDst, const Int64 * pSrc, __m128i mIndices, __m256i mSigns ); // AVX2

		};
	};

	// Sparse Memory, 64-bits indices
	namespace Sparse64 {
		namespace Stride1 {

            inline __m128 Load128( const Float * pSrc, __m128i mIndices ); // AVX2

            inline __m128d Load128( const Double * pSrc, __m128i mIndices ); // AVX2

            inline __m128i Load128( const Int32 * pSrc, __m128i mIndices ); // AVX2
            inline __m128i Load128( const Int64 * pSrc, __m128i mIndices ); // AVX2

            inline __m128 Load256( const Float * pSrc, __m256i mIndices ); // AVX2

            inline __m256d Load256( const Double * pSrc, __m256i mIndices ); // AVX2

            inline __m128i Load256( const Int32 * pSrc, __m256i mIndices ); // AVX2
            inline __m256i Load256( const Int64 * pSrc, __m256i mIndices ); // AVX2

            // Mask variants
            inline __m128 Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ); // AVX2

            inline __m128d Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ); // AVX2

            inline __m128i Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2
            inline __m128i Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2

            inline __m128 Load256( __m128 mDst, const Float * pSrc, __m256i mIndices, __m128 mSigns ); // AVX2

            inline __m256d Load256( __m256d mDst, const Double * pSrc, __m256i mIndices, __m256d mSigns ); // AVX2

            inline __m128i Load256( __m128i mDst, const Int32 * pSrc, __m256i mIndices, __m128i mSigns ); // AVX2
            inline __m256i Load256( __m256i mDst, const Int64 * pSrc, __m256i mIndices, __m256i mSigns ); // AVX2

		};
		namespace Stride2 {

            inline __m128 Load128( const Float * pSrc, __m128i mIndices ); // AVX2

            inline __m128d Load128( const Double * pSrc, __m128i mIndices ); // AVX2

            inline __m128i Load128( const Int32 * pSrc, __m128i mIndices ); // AVX2
            inline __m128i Load128( const Int64 * pSrc, __m128i mIndices ); // AVX2

            inline __m128 Load256( const Float * pSrc, __m256i mIndices ); // AVX2

            inline __m256d Load256( const Double * pSrc, __m256i mIndices ); // AVX2

            inline __m128i Load256( const Int32 * pSrc, __m256i mIndices ); // AVX2
            inline __m256i Load256( const Int64 * pSrc, __m256i mIndices ); // AVX2

            // Mask variants
            inline __m128 Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ); // AVX2

            inline __m128d Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ); // AVX2

            inline __m128i Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2
            inline __m128i Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2

            inline __m128 Load256( __m128 mDst, const Float * pSrc, __m256i mIndices, __m128 mSigns ); // AVX2

            inline __m256d Load256( __m256d mDst, const Double * pSrc, __m256i mIndices, __m256d mSigns ); // AVX2

            inline __m128i Load256( __m128i mDst, const Int32 * pSrc, __m256i mIndices, __m128i mSigns ); // AVX2
            inline __m256i Load256( __m256i mDst, const Int64 * pSrc, __m256i mIndices, __m256i mSigns ); // AVX2

		};
		namespace Stride4 {

            inline __m128 Load128( const Float * pSrc, __m128i mIndices ); // AVX2

            inline __m128d Load128( const Double * pSrc, __m128i mIndices ); // AVX2

            inline __m128i Load128( const Int32 * pSrc, __m128i mIndices ); // AVX2
            inline __m128i Load128( const Int64 * pSrc, __m128i mIndices ); // AVX2

            inline __m128 Load256( const Float * pSrc, __m256i mIndices ); // AVX2

            inline __m256d Load256( const Double * pSrc, __m256i mIndices ); // AVX2

            inline __m128i Load256( const Int32 * pSrc, __m256i mIndices ); // AVX2
            inline __m256i Load256( const Int64 * pSrc, __m256i mIndices ); // AVX2

            // Mask variants
            inline __m128 Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ); // AVX2

            inline __m128d Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ); // AVX2

            inline __m128i Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2
            inline __m128i Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2

            inline __m128 Load256( __m128 mDst, const Float * pSrc, __m256i mIndices, __m128 mSigns ); // AVX2

            inline __m256d Load256( __m256d mDst, const Double * pSrc, __m256i mIndices, __m256d mSigns ); // AVX2

            inline __m128i Load256( __m128i mDst, const Int32 * pSrc, __m256i mIndices, __m128i mSigns ); // AVX2
            inline __m256i Load256( __m256i mDst, const Int64 * pSrc, __m256i mIndices, __m256i mSigns ); // AVX2

		};
		namespace Stride8 {

            inline __m128 Load128( const Float * pSrc, __m128i mIndices ); // AVX2

            inline __m128d Load128( const Double * pSrc, __m128i mIndices ); // AVX2

            inline __m128i Load128( const Int32 * pSrc, __m128i mIndices ); // AVX2
            inline __m128i Load128( const Int64 * pSrc, __m128i mIndices ); // AVX2

            inline __m128 Load256( const Float * pSrc, __m256i mIndices ); // AVX2

            inline __m256d Load256( const Double * pSrc, __m256i mIndices ); // AVX2

            inline __m128i Load256( const Int32 * pSrc, __m256i mIndices ); // AVX2
            inline __m256i Load256( const Int64 * pSrc, __m256i mIndices ); // AVX2

            // Mask variants
            inline __m128 Load128( __m128 mDst, const Float * pSrc, __m128i mIndices, __m128 mSigns ); // AVX2

            inline __m128d Load128( __m128d mDst, const Double * pSrc, __m128i mIndices, __m128d mSigns ); // AVX2

            inline __m128i Load128( __m128i mDst, const Int32 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2
            inline __m128i Load128( __m128i mDst, const Int64 * pSrc, __m128i mIndices, __m128i mSigns ); // AVX2

            inline __m128 Load256( __m128 mDst, const Float * pSrc, __m256i mIndices, __m128 mSigns ); // AVX2

            inline __m256d Load256( __m256d mDst, const Double * pSrc, __m256i mIndices, __m256d mSigns ); // AVX2

            inline __m128i Load256( __m128i mDst, const Int32 * pSrc, __m256i mIndices, __m128i mSigns ); // AVX2
            inline __m256i Load256( __m256i mDst, const Int64 * pSrc, __m256i mIndices, __m256i mSigns ); // AVX2

		};
	};

}; }; };

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "SIMD_ImportMemory.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_SIMD_SIMDIMPORTMEMORY_H

