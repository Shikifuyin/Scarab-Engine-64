/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Hash/Hashing.inl
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
// Hashing implementation
inline Hashing * Hashing::GetInstance() {
    static Hashing s_Instance;
    return &s_Instance;
}

inline DWord Hashing::Hash( const Byte * pData, SizeT iLength ) const {
    return BobJenkinsRot( pData, iLength );
}
inline DWord Hashing::Hash( const GChar * strData ) const {
    return BobJenkinsRot( strData );
}

/////////////////////////////////////////////////////////////////////////////////

inline Word Hashing::_Get16Bits( const Byte * pData ) const {
    return *( (const Word *)pData );
}
inline Word Hashing::_Get16Bits( const AChar * strData ) const {
    return *( (const Word *)strData );
}
inline Word Hashing::_Get16Bits( const WChar * strData ) const {
    static Word s_wValue;
    static Byte * s_pValue = (Byte*)(&s_wValue);
    Byte * pVal = s_pValue;
    *pVal = (Byte)(*strData & 0xff); ++pVal; ++strData;
    *pVal = (Byte)(*strData & 0xff);
    return s_wValue;
}
inline DWord Hashing::_Get32Bits( const Byte * pData ) const {
    return *( (const DWord *)pData );
}
inline DWord Hashing::_Get32Bits( const AChar * strData ) const {
    return *( (const DWord *)strData );
}
inline DWord Hashing::_Get32Bits( const WChar * strData ) const {
    static DWord s_dwValue;
    static Byte * s_pValue = (Byte*)(&s_dwValue);
    Byte * pVal = s_pValue;
    *pVal = (Byte)(*strData & 0xff); ++pVal; ++strData;
    *pVal = (Byte)(*strData & 0xff); ++pVal; ++strData;
    *pVal = (Byte)(*strData & 0xff); ++pVal; ++strData;
    *pVal = (Byte)(*strData & 0xff);
    return s_dwValue;
}

inline Void Hashing::_BobJenkins_Mix( DWord & a, DWord & b, DWord & c ) const {
    a -= b; a -= c; a ^= (c >> 13);
    b -= c; b -= a; b ^= (a <<  8);
    c -= a; c -= b; c ^= (b >> 13);
    a -= b; a -= c; a ^= (c >> 12);
    b -= c; b -= a; b ^= (a << 16);
    c -= a; c -= b; c ^= (b >>  5);
    a -= b; a -= c; a ^= (c >>  3);
    b -= c; b -= a; b ^= (a << 10);
    c -= a; c -= b; c ^= (b >> 15);
}
inline Void Hashing::_BobJenkins_MixRot( DWord & a, DWord & b, DWord & c ) const {
    a -= c; a ^= BitRotL(c, 4); c += b;
    b -= a; b ^= BitRotL(a, 6); a += c;
    c -= b; c ^= BitRotL(b, 8); b += a;
    a -= c; a ^= BitRotL(c,16); c += b;
    b -= a; b ^= BitRotL(a,19); a += c;
    c -= b; c ^= BitRotL(b, 4); b += a;
}
inline Void Hashing::_BobJenkins_Final( DWord & a, DWord & b, DWord & c ) const {
    c ^= b; c -= BitRotL(b,14);
    a ^= c; a -= BitRotL(c,11);
    b ^= a; b -= BitRotL(a,25);
    c ^= b; c -= BitRotL(b,16);
    a ^= c; a -= BitRotL(c, 4);
    b ^= a; b -= BitRotL(a,14);
    c ^= b; c -= BitRotL(b,24);
}

