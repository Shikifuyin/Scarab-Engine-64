/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Hash/Hashing.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Simple, fast & specialized hashing algorithms ...
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : TODO : Recode PaulHsieh for string ... (boring)
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_LIB_MATH_HASH_HASHING_H
#define SCARAB_LIB_MATH_HASH_HASHING_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../../../ThirdParty/System/String.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define HashFn Hashing::GetInstance()

/////////////////////////////////////////////////////////////////////////////////
// The Hashing class
class Hashing
{
    // Discrete singleton interface
public:
    inline static Hashing * GetInstance();

private:
    Hashing();
    ~Hashing();

public:
    // Main Hashing Method : Default is BobJenkinsRot
    inline DWord Hash( const Byte * pData, SizeT iLength ) const;
    inline DWord Hash( const GChar * strData ) const;

    // Okay, we list algorithm ordered by method, complexity & efficiency
    // Here is a little comparative :
    // Min-Size = Minimum reasonable table size greater than 1000.
    // Complexity = Overall complexity in instructions
    // Streamed-Complexity = Complexity when streamed, gaining prelude/conclude code
    // Funnel-15 = Largest set of input-bits affecting the smallest set of internal-bits when mapping 15-byte keys into a 1-byte result.
    // Funnel-100 = Largest set of input-bits affecting the smallest set of internal-bits when mapping 100-byte keys into a 4-byte result.
    // Collide-32 = Number of collisions found when a dictionary of 38,470 English words was hashed into a 32-bit result.
    // Collide-1000 = Chi^2 measure of how well the hash did at mapping the dictionary
    //    ( Chi^2 measure means : "better than random < -3 < random fluctuations < +3 < worse than random" )
    //               Min-Size   Complexity 	Streamed-Complexity 	Funnel-15 	Funnel-100 	Collide-32 	Collide-1000
    // Bernstein         1024 	      7n+3 	               3n+2 	   3 => 2 	    3 => 2 	         4 	       +1.69
    // CRC32 	         1024         9n+3 	               5n+2 	   2 => 1 	  11 => 10 	         0 	       +0.07
    // CRCGeneric	     1024         9n+3 	               5n+2 	     none 	      none 	         0 	       -1.83
    // One-at-a-Time     1024 	      9n+9 	               5n+8 	     none 	      none 	         0 	       -0.05
    // BobJenkins        1024 	     6n+35 	                N/A 	     none 	      none 	         0 	       +0.33
    // Paul Hsieh        1024 	     5n+17 	                N/A 	   3 => 2 	    3 => 2 	         1 	       +1.12
    // BobJenkinsRot     1024 	     5n+20 	                N/A 	     none 	      none 	         0 	       -0.08
    //
    // IMPORTANT : None of those function can do better than 2^32 collision probability,
    // which is NOT suitable for cryptographic-level security, use higher-order hashers
    // like MD5 or SHA1 if you need cryptographic-level hashing ...
        // 32-bits hashers
    DWord MixedBits( DWord dwValue ) const;
    DWord ThomasWang( DWord dwValue ) const;
        // Simple hashers
    DWord BKDR( const Byte * pData, SizeT iLength ) const;
    DWord BKDR( const GChar * strData ) const;
    DWord Bernstein( const Byte * pData, SizeT iLength ) const;
    DWord Bernstein( const GChar * strData ) const;
    DWord AlphaNum( const Byte * pData, SizeT iLength ) const;
    DWord AlphaNum( const GChar * strData ) const;
    DWord SDBM( const Byte * pData, SizeT iLength ) const;
    DWord SDBM( const GChar * strData ) const;
    DWord RS( const Byte * pData, SizeT iLength ) const;
    DWord RS( const GChar * strData ) const;
    DWord ELF( const Byte * pData, SizeT iLength ) const;
    DWord ELF( const GChar * strData ) const;
    DWord FNV( const Byte * pData, SizeT iLength ) const;
    DWord FNV( const GChar * strData ) const;
    DWord JS( const Byte * pData, SizeT iLength ) const;
    DWord JS( const GChar * strData ) const;
    DWord AP( const Byte * pData, SizeT iLength ) const;
    DWord AP( const GChar * strData ) const;
        // Cyclic Redundancy code hashers
    DWord CRC32( const Byte * pData, SizeT iLength ) const;
    DWord CRC32( const GChar * strData ) const;
    DWord CRCGeneric( const Byte * pData, SizeT iLength ) const;
    DWord CRCGeneric( const GChar * strData ) const;
        // Advanced hashers
    DWord OneAtATime( const Byte * pData, SizeT iLength ) const;
    DWord OneAtATime( const GChar * strData ) const;
    DWord BobJenkins( const Byte * pData, SizeT iLength ) const;
    DWord BobJenkins( const GChar * strData ) const;
    DWord PaulHsieh( const Byte * pData, SizeT iLength ) const;
    DWord PaulHsieh( const GChar * strData ) const;
    DWord BobJenkinsRot( const Byte * pData, SizeT iLength ) const; // <= The best one so far ...
    DWord BobJenkinsRot( const GChar * strData ) const;             //

private:
    // internal helpers
    inline Word _Get16Bits( const Byte * pData ) const;
    inline Word _Get16Bits( const AChar * strData ) const;
    inline Word _Get16Bits( const WChar * strData ) const;
    inline DWord _Get32Bits( const Byte * pData ) const;
    inline DWord _Get32Bits( const AChar * strData ) const;
    inline DWord _Get32Bits( const WChar * strData ) const;

    Void _CRC32_GenerateTable( DWord dwPolynomial );
    DWord m_arrTableCRC32[256];

    Void _CRCGeneric_GenerateTable();
    DWord m_arrTableCRCGeneric[256];

    inline Void _BobJenkins_Mix( DWord & a, DWord & b, DWord & c ) const;
    inline Void _BobJenkins_MixRot( DWord & a, DWord & b, DWord & c ) const;
    inline Void _BobJenkins_Final( DWord & a, DWord & b, DWord & c ) const;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "Hashing.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_MATH_HASH_HASHING_H
