/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Hash/Hashing.cpp
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
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "Hashing.h"

/////////////////////////////////////////////////////////////////////////////////
// Hashing implementation
Hashing::Hashing()
{
    _CRC32_GenerateTable(0xedb88320); // Polynomial
    _CRCGeneric_GenerateTable();
}
Hashing::~Hashing()
{
    // nothing to do
}

DWord Hashing::MixedBits( DWord dwValue ) const
{
    dwValue += ~(dwValue << 15);
    dwValue ^=  (dwValue >> 10);
    dwValue +=  (dwValue << 3);
    dwValue ^=  (dwValue >> 6);
    dwValue += ~(dwValue << 11);
    dwValue ^=  (dwValue >> 16);
    return dwValue;
}
DWord Hashing::ThomasWang( DWord dwValue ) const
{
    dwValue = ~dwValue + (dwValue << 15);
    dwValue ^= (dwValue >> 12);
    dwValue += (dwValue << 2);
    dwValue ^= (dwValue >> 4);
    dwValue *= 2057;
    dwValue ^= (dwValue >> 16);
    return dwValue;
}

/////////////////////////////////////////////////////////////////////////////////

DWord Hashing::BKDR( const Byte * pData, SizeT iLength ) const
{
    DWord dwHash = 0, dwSeed = 131; // 31 131 1313 13131 etc ...
    const Byte * pEnd = pData + iLength;
    while( pData < pEnd ) {
        dwHash = (dwHash * dwSeed) + *pData;
        ++pData;
    }
    return dwHash;
}
DWord Hashing::BKDR( const GChar * strData ) const
{
    DWord dwHash = 0, dwSeed = 131; // 31 131 1313 13131 etc ...
    while( *strData != NULLBYTE ) {
        dwHash = (dwHash * dwSeed) + (*strData & 0xff);
        ++strData;
    }
    return dwHash;
}
DWord Hashing::Bernstein( const Byte * pData, SizeT iLength ) const
{
    DWord dwHash = 5381;
    const Byte * pEnd = pData + iLength;
    while( pData < pEnd ) {
        dwHash = ((dwHash << 5) + dwHash) + *pData; // *33
        ++pData;
    }
    return dwHash;
}
DWord Hashing::Bernstein( const GChar * strData ) const
{
    DWord dwHash = 5381;
    while( *strData != NULLBYTE ) {
        dwHash = ((dwHash << 5) + dwHash) + (*strData & 0xff); // *33
        ++strData;
    }
    return dwHash;
}
DWord Hashing::AlphaNum( const Byte * pData, SizeT iLength ) const
{
    DWord dwHash = 0;
    const Byte * pEnd = pData + iLength;
    while( pData < pEnd ) {
        dwHash = ( (dwHash << 5) + (dwHash * 5) + *pData );
        ++pData;
    }
    return dwHash;
}
DWord Hashing::AlphaNum( const GChar * strData ) const
{
    DWord dwHash = 0;
    while( *strData != NULLBYTE ) {
        dwHash = ( (dwHash << 5) + (dwHash * 5) + (*strData & 0xff) );
        ++strData;
    }
    return dwHash;
}
DWord Hashing::SDBM( const Byte * pData, SizeT iLength ) const
{
    DWord dwHash = 0;
    const Byte * pEnd = pData + iLength;
    while( pData < pEnd ) {
        dwHash = ( (dwHash << 6) + (dwHash << 16) - dwHash + *pData );
        ++pData;
    }
    return dwHash;
}
DWord Hashing::SDBM( const GChar * strData ) const
{
    DWord dwHash = 0;
    while( *strData != NULLBYTE ) {
        dwHash = ( (dwHash << 6) + (dwHash << 16) - dwHash + (*strData & 0xff) );
        ++strData;
    }
    return dwHash;
}
DWord Hashing::RS( const Byte * pData, SizeT iLength ) const
{
    UInt iA = 63689, iB = 378551;
    DWord dwHash = 0;
    const Byte * pEnd = pData + iLength;
    while( pData < pEnd ) {
        dwHash = dwHash * iA + *pData;
        iA *= iB;
        ++pData;
    }
    return dwHash;
}
DWord Hashing::RS( const GChar * strData ) const
{
    UInt iA = 63689, iB = 378551;
    DWord dwHash = 0;
    while( *strData != NULLBYTE ) {
        dwHash = dwHash * iA + (*strData & 0xff);
        iA *= iB;
        ++strData;
    }
    return dwHash;
}
DWord Hashing::ELF( const Byte * pData, SizeT iLength ) const
{
    DWord dwHash = 0, X;
    const Byte * pEnd = pData + iLength;
    while( pData < pEnd ) {
        dwHash = (dwHash << 4) + *pData;
        X = dwHash & 0xf0000000;
        if (X != 0) {
            dwHash ^= (X >> 24);
            dwHash &= (~X);
        }
        ++pData;
    }
    return dwHash;
}
DWord Hashing::ELF( const GChar * strData ) const
{
    DWord dwHash = 0, X;
    while( *strData != NULLBYTE ) {
        dwHash = (dwHash << 4) + (*strData & 0xff);
        X = dwHash & 0xf0000000;
        if (X != 0) {
            dwHash ^= (X >> 24);
            dwHash &= (~X);
        }
        ++strData;
    }
    return dwHash;
}
DWord Hashing::FNV( const Byte * pData, SizeT iLength ) const
{
    DWord dwHash = 2166136261;
    const Byte * pEnd = pData + iLength;
    while( pData < pEnd ) {
        dwHash = (dwHash * 16777619) ^ *pData;
        ++pData;
    }
    return dwHash;
}
DWord Hashing::FNV( const GChar * strData ) const
{
    DWord dwHash = 2166136261;
    while( *strData != NULLBYTE ) {
        dwHash = (dwHash * 16777619) ^ (*strData & 0xff);
        ++strData;
    }
    return dwHash;
}
DWord Hashing::JS( const Byte * pData, SizeT iLength ) const
{
    DWord dwHash = 1315423911;
    const Byte * pEnd = pData + iLength;
    while( pData < pEnd ) {
        dwHash ^= ( (dwHash << 5) + (dwHash >> 2) + *pData );
        ++pData;
    }
    return dwHash;
}
DWord Hashing::JS( const GChar * strData ) const
{
    DWord dwHash = 1315423911;
    while( *strData != NULLBYTE ) {
        dwHash ^= ( (dwHash << 5) + (dwHash >> 2) + (*strData & 0xff) );
        ++strData;
    }
    return dwHash;
}
DWord Hashing::AP( const Byte * pData, SizeT iLength ) const
{
    DWord dwHash = 0;
    Bool bBias = false;
    const Byte * pEnd = pData + iLength;
    while( pData < pEnd ) {
        if (bBias)
            dwHash ^= ~( (dwHash << 11) ^ (dwHash >> 5) ^ *pData );
        else
            dwHash ^=  ( (dwHash << 7) ^ (dwHash >> 3) ^ *pData );
        bBias = !bBias;
        ++pData;
    }
    return dwHash;
}
DWord Hashing::AP( const GChar * strData ) const
{
    DWord dwHash = 0;
    Bool bBias = false;
    while( *strData != NULLBYTE ) {
        if (bBias)
            dwHash ^= ~( (dwHash << 11) ^ (dwHash >> 5) ^ (*strData & 0xff) );
        else
            dwHash ^=  ( (dwHash << 7) ^ (dwHash >> 3) ^ (*strData & 0xff) );
        bBias = !bBias;
        ++strData;
    }
    return dwHash;
}

/////////////////////////////////////////////////////////////////////////////////

DWord Hashing::CRC32( const Byte * pData, SizeT iLength ) const
{
    DWord dwCRC = iLength; // Init value
    const Byte * pEnd = pData + iLength;
    while( pData < pEnd ) {
        dwCRC = (dwCRC >> 8) ^ m_arrTableCRC32[ (dwCRC & 0xff) ^ *pData ];
        ++pData;
    }
    return dwCRC;
}
DWord Hashing::CRC32( const GChar * strData ) const
{
    DWord dwCRC = StringFn->Length( strData ); // Init value
    while( *strData != NULLBYTE ) {
        dwCRC = (dwCRC >> 8) ^ m_arrTableCRC32[ (dwCRC & 0xff) ^ (*strData & 0xff) ];
        ++strData;
    }
    return dwCRC;
}
DWord Hashing::CRCGeneric( const Byte * pData, SizeT iLength ) const
{
    DWord dwCRC = iLength; // Init value
    const Byte * pEnd = pData + iLength;
    while( pData < pEnd ) {
        dwCRC = (dwCRC >> 8) ^ m_arrTableCRCGeneric[ (dwCRC & 0xff) ^ *pData ];
        ++pData;
    }
    return dwCRC;
}
DWord Hashing::CRCGeneric( const GChar * strData ) const
{
    DWord dwCRC = StringFn->Length( strData ); // Init value
    while( *strData != NULLBYTE ) {
        dwCRC = (dwCRC >> 8) ^ m_arrTableCRCGeneric[ (dwCRC & 0xff) ^ (*strData & 0xff) ];
        ++strData;
    }
    return dwCRC;
}

/////////////////////////////////////////////////////////////////////////////////

DWord Hashing::OneAtATime( const Byte * pData, SizeT iLength ) const
{
    DWord dwHash = 0;
    const Byte * pEnd = pData + iLength;
    while( pData < pEnd ) {
        dwHash += *pData;
        dwHash += (dwHash << 10);
        dwHash ^= (dwHash >>  6);
        ++pData;
    }
    dwHash += (dwHash <<  3);
    dwHash ^= (dwHash >> 11);
    dwHash += (dwHash << 15);
    return dwHash;
}
DWord Hashing::OneAtATime( const GChar * strData ) const
{
    DWord dwHash = 0;
    while( *strData != NULLBYTE ) {
        dwHash += (*strData & 0xff);
        dwHash += (dwHash << 10);
        dwHash ^= (dwHash >>  6);
        ++strData;
    }
    dwHash += (dwHash <<  3);
    dwHash ^= (dwHash >> 11);
    dwHash += (dwHash << 15);
    return dwHash;
}
DWord Hashing::BobJenkins( const Byte * pData, SizeT iLength ) const
{
    // Internal state
    DWord dwA = 0x9e3779b9; // The magical
    DWord dwB = 0x9e3779b9; // golden ratio
    DWord dwC = 0; // Init value (0 to begin)

    // Main Loop
    while( iLength >= 12 ) {
        dwA += _Get32Bits(pData); pData += 4;
        dwB += _Get32Bits(pData); pData += 4;
        dwC += _Get32Bits(pData); pData += 4;
        _BobJenkins_Mix(dwA, dwB, dwC);
        iLength -= 12;
    }

    // End Cases
    dwC += iLength;
    switch( iLength ) {
        case 11: dwC += ( (DWord)(pData[10]) ) << 24;
        case 10: dwC += ( (DWord)(pData[9]) ) << 16;
        case  9: dwC += ( (DWord)(pData[8]) ) << 8;
        // first byte of C contains length
        case  8: dwB += ( (DWord)(pData[7]) ) << 24;
        case  7: dwB += ( (DWord)(pData[6]) ) << 16;
        case  6: dwB += ( (DWord)(pData[5]) ) << 8;
        case  5: dwB += ( (DWord)(pData[4]) );
        case  4: dwA += ( (DWord)(pData[3]) ) << 24;
        case  3: dwA += ( (DWord)(pData[2]) ) << 16;
        case  2: dwA += ( (DWord)(pData[1]) ) << 8;
        case  1: dwA += ( (DWord)(pData[0]) );
        case  0: ;// nothing to do
    }
    _BobJenkins_Mix(dwA, dwB, dwC);

    return dwC;
}
DWord Hashing::BobJenkins( const GChar * strData ) const
{
    // Internal state
    DWord dwA = 0x9e3779b9; // The magical
    DWord dwB = 0x9e3779b9; // golden ratio
    DWord dwC = 0; // Init value (0 to begin)

    UInt iLength = 0;
    const GChar * pEnd = strData;
    while( *pEnd != NULLBYTE && iLength < 12 ) {
        ++pEnd; ++iLength;
    }

    // Main Loop
    while( iLength >= 12 ) {
        dwA += _Get32Bits(strData); strData += 4;
        dwB += _Get32Bits(strData); strData += 4;
        dwC += _Get32Bits(strData); strData += 4;
        _BobJenkins_Mix(dwA, dwB, dwC);
        iLength -= 12;
        while( *pEnd != NULLBYTE && iLength < 12 ) {
            ++pEnd; ++iLength;
        }
    }

    // End Cases
    dwC += iLength;
    switch( iLength ) {
        case 11: dwC += ( (DWord)(strData[10] & 0xff) ) << 24;
        case 10: dwC += ( (DWord)(strData[9]  & 0xff) ) << 16;
        case  9: dwC += ( (DWord)(strData[8]  & 0xff) ) << 8;
        // first byte of C contains length
        case  8: dwB += ( (DWord)(strData[7]  & 0xff) ) << 24;
        case  7: dwB += ( (DWord)(strData[6]  & 0xff) ) << 16;
        case  6: dwB += ( (DWord)(strData[5]  & 0xff) ) << 8;
        case  5: dwB += ( (DWord)(strData[4]  & 0xff) );
        case  4: dwA += ( (DWord)(strData[3]  & 0xff) ) << 24;
        case  3: dwA += ( (DWord)(strData[2]  & 0xff) ) << 16;
        case  2: dwA += ( (DWord)(strData[1]  & 0xff) ) << 8;
        case  1: dwA += ( (DWord)(strData[0]  & 0xff) );
        case  0: ;// nothing to do
    }
    _BobJenkins_Mix(dwA, dwB, dwC);

    return dwC;
}
DWord Hashing::PaulHsieh( const Byte * pData, SizeT iLength ) const
{
    DWord dwHash = iLength, dwTemp;
    Int iRem = (iLength & 0x03);
    iLength >>= 2;

    // Main Loop
    while( iLength > 0 ) {
        dwHash += _Get16Bits(pData);                  pData += 2;
        dwTemp  = (_Get16Bits(pData) << 11) ^ dwHash; pData += 2;
        dwHash  = (dwHash << 16) ^ dwTemp;
        dwHash += (dwHash >> 11);
        --iLength;
    }

    // End Cases
    if (iRem == 3) {
        dwHash += _Get16Bits(pData); pData += 2;
        dwHash ^= (dwHash << 16);
        dwHash ^= (  *pData << 18);
        dwHash += (dwHash >> 11);
    } else if (iRem == 2) {
        dwHash += _Get16Bits(pData);
        dwHash ^= (dwHash << 11);
        dwHash += (dwHash >> 17);
    } else if (iRem == 1) {
        dwHash += *pData;
        dwHash ^= (dwHash << 10);
        dwHash += (dwHash >> 1);
    }

    // Force avalanching for last 127 bits
    dwHash ^= dwHash << 3;
    dwHash += dwHash >> 5;
    dwHash ^= dwHash << 4;
    dwHash += dwHash >> 17;
    dwHash ^= dwHash << 25;
    dwHash += dwHash >> 6;

    return dwHash;
}
DWord Hashing::PaulHsieh( const GChar * strData ) const
{
    UInt iLength = StringFn->Length( strData );

    DWord dwHash = iLength, dwTemp;
    Int iRem = (iLength & 0x03);
    iLength >>= 2;

    // Main Loop
    while( iLength > 0 ) {
        dwHash += _Get16Bits(strData);                  strData += 2;
        dwTemp  = (_Get16Bits(strData) << 11) ^ dwHash; strData += 2;
        dwHash  = (dwHash << 16) ^ dwTemp;
        dwHash += (dwHash >> 11);
        --iLength;
    }

    // End Cases
    if (iRem == 3) {
        dwHash += _Get16Bits(strData); strData += 2;
        dwHash ^= (dwHash << 16);
        dwHash ^= ( ( (DWord)(*strData & 0xff) ) << 18 );
        dwHash += (dwHash >> 11);
    } else if (iRem == 2) {
        dwHash += _Get16Bits(strData);
        dwHash ^= (dwHash << 11);
        dwHash += (dwHash >> 17);
    } else if (iRem == 1) {
        dwHash += (*strData & 0xff);
        dwHash ^= (dwHash << 10);
        dwHash += (dwHash >> 1);
    }

    // Force avalanching for last 127 bits
    dwHash ^= dwHash << 3;
    dwHash += dwHash >> 5;
    dwHash ^= dwHash << 4;
    dwHash += dwHash >> 17;
    dwHash ^= dwHash << 25;
    dwHash += dwHash >> 6;

    return dwHash;
}
DWord Hashing::BobJenkinsRot( const Byte * pData, SizeT iLength ) const
{
    // Internal state
    DWord dwA = 0xdeadbeef + iLength + 0; // Init value (0 to begin)
    DWord dwB = dwA;
    DWord dwC = dwA;

    // Main Loop
    while( iLength > 12 ) {
        dwA += _Get32Bits(pData); pData += 4;
        dwB += _Get32Bits(pData); pData += 4;
        dwC += _Get32Bits(pData); pData += 4;
        _BobJenkins_MixRot(dwA, dwB, dwC);
        iLength -= 12;
    }

    // End Cases
    switch( iLength ) {
        case 12: dwC += ( (DWord)(pData[11]) ) << 24;
        case 11: dwC += ( (DWord)(pData[10]) ) << 16;
        case 10: dwC += ( (DWord)(pData[9]) ) << 8;
        case  9: dwC += ( (DWord)(pData[8]) );
        case  8: dwB += ( (DWord)(pData[7]) ) << 24;
        case  7: dwB += ( (DWord)(pData[6]) ) << 16;
        case  6: dwB += ( (DWord)(pData[5]) ) << 8;
        case  5: dwB += ( (DWord)(pData[4]) );
        case  4: dwA += ( (DWord)(pData[3]) ) << 24;
        case  3: dwA += ( (DWord)(pData[2]) ) << 16;
        case  2: dwA += ( (DWord)(pData[1]) ) << 8;
        case  1: dwA += ( (DWord)(pData[0]) ); break;
        case  0: return dwC;
    }
    _BobJenkins_Final(dwA, dwB, dwC);

    return dwC;
}
DWord Hashing::BobJenkinsRot( const GChar * strData ) const
{
    // Internal state
    DWord dwA = 0xdeadbeef + (*strData & 0xff) + 0; // Init value (0 to begin)
    DWord dwB = dwA;
    DWord dwC = dwA;

    UInt iLength = 0;
    const GChar * pEnd = strData;
    while( *pEnd != NULLBYTE && iLength <= 12 ) {
        ++pEnd; ++iLength;
    }

    // Main Loop
    while( iLength > 12 ) {
        dwA += _Get32Bits(strData); strData += 4;
        dwB += _Get32Bits(strData); strData += 4;
        dwC += _Get32Bits(strData); strData += 4;
        _BobJenkins_MixRot(dwA, dwB, dwC);
        iLength -= 12;
        while( *pEnd != NULLBYTE && iLength <= 12 ) {
            ++pEnd; ++iLength;
        }
    }

    // End Cases
    switch( iLength ) {
        case 12: dwC += ( (DWord)(strData[11] & 0xff) ) << 24;
        case 11: dwC += ( (DWord)(strData[10] & 0xff) ) << 16;
        case 10: dwC += ( (DWord)(strData[9]  & 0xff) ) << 8;
        case  9: dwC += ( (DWord)(strData[8]  & 0xff) );
        case  8: dwB += ( (DWord)(strData[7]  & 0xff) ) << 24;
        case  7: dwB += ( (DWord)(strData[6]  & 0xff) ) << 16;
        case  6: dwB += ( (DWord)(strData[5]  & 0xff) ) << 8;
        case  5: dwB += ( (DWord)(strData[4]  & 0xff) );
        case  4: dwA += ( (DWord)(strData[3]  & 0xff) ) << 24;
        case  3: dwA += ( (DWord)(strData[2]  & 0xff) ) << 16;
        case  2: dwA += ( (DWord)(strData[1]  & 0xff) ) << 8;
        case  1: dwA += ( (DWord)(strData[0]  & 0xff) ); break;
        case  0: return dwC;
    }
    _BobJenkins_Final(dwA, dwB, dwC);

    return dwC;
}

/////////////////////////////////////////////////////////////////////////////////

Void Hashing::_CRC32_GenerateTable( DWord dwPolynomial )
{
    DWord dwCRC;
    for( UInt i = 0; i < 256; ++i ) {
        dwCRC = i;
        dwCRC = (dwCRC >> 1) ^ ( (dwCRC & 0x01) ? dwPolynomial : 0 );
        dwCRC = (dwCRC >> 1) ^ ( (dwCRC & 0x01) ? dwPolynomial : 0 );
        dwCRC = (dwCRC >> 1) ^ ( (dwCRC & 0x01) ? dwPolynomial : 0 );
        dwCRC = (dwCRC >> 1) ^ ( (dwCRC & 0x01) ? dwPolynomial : 0 );
        dwCRC = (dwCRC >> 1) ^ ( (dwCRC & 0x01) ? dwPolynomial : 0 );
        dwCRC = (dwCRC >> 1) ^ ( (dwCRC & 0x01) ? dwPolynomial : 0 );
        dwCRC = (dwCRC >> 1) ^ ( (dwCRC & 0x01) ? dwPolynomial : 0 );
        dwCRC = (dwCRC >> 1) ^ ( (dwCRC & 0x01) ? dwPolynomial : 0 );
        m_arrTableCRC32[i] = dwCRC;
    }
}
Void Hashing::_CRCGeneric_GenerateTable()
{
    Byte X;
    UInt i,j;
    for( i = 0; i < 256; ++i ) {
        X = (Byte)i;
        for( j = 0; j < 5; ++j ) {
            X += 1; X += (X << 1); X ^= (X >> 1);
        }
        m_arrTableCRCGeneric[i] = X;
        for( j = 0; j < 5; ++j ) {
            X += 2; X += (X << 1); X ^= (X >> 1);
        }
        m_arrTableCRCGeneric[i] ^= ( ((DWord)X) << 8 );
        for( j = 0; j < 5; ++j ) {
            X += 3; X += (X << 1); X ^= (X >> 1);
        }
        m_arrTableCRCGeneric[i] ^= ( ((DWord)X) << 16 );
        for( j = 0; j < 5; ++j ) {
            X += 4; X += (X << 1); X ^= (X >> 1);
        }
        m_arrTableCRCGeneric[i] ^= ( ((DWord)X) << 24 );
    }
}
