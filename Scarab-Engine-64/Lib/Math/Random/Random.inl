/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Math/Random/Random.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Random generator implementation.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// RandomGenerator implementation
inline RandomGenerator * RandomGenerator::GetInstance() {
    static RandomGenerator s_Instance;
    return &s_Instance;
}

inline DWord RandomGenerator::RandMax() const {
    return XorShiftMax();
}
inline Void RandomGenerator::RandSeed( DWord dwSeed, UInt iWarmUp ) {
    XorShiftWeylSeed( dwSeed, iWarmUp );
}
inline DWord RandomGenerator::Rand() {
    return XorShiftWeyl();
}
inline Float RandomGenerator::RandF() {
    static Float s_fInvMax = ( 1.0f / (Float)XorShiftMax() );
    return ( s_fInvMax * (Float)XorShiftWeyl() );
}

inline DWord RandomGenerator::LecuyerMax() const {
    return (unsigned)(sm_iLecuyerM1 - 2);
}

inline DWord RandomGenerator::KnuthMax() const {
    return (unsigned)(sm_iKnuthBig - 1);
}

inline UInt64 RandomGenerator::RanRotMax() const {
    return UINT64_MAX;
}

inline DWord RandomGenerator::MersenneTwisterMax() const {
    return UINT_MAX;
}

inline DWord RandomGenerator::MotherOfAllMax() const {
    return UINT_MAX;
}

inline DWord RandomGenerator::CompMWCMax() const {
    return UINT_MAX;
}

inline DWord RandomGenerator::XorShiftMax() const {
    return UINT_MAX;
}

inline Void RandomGenerator::SeedAll( DWord dwSeed, UInt iWarmUp ) {
    LecuyerSeed(dwSeed, iWarmUp);
    KnuthSeed(dwSeed, iWarmUp);
    RanRotSeed(dwSeed, iWarmUp);
    MersenneTwisterSeed(dwSeed, iWarmUp);
    MotherOfAll32Seed(dwSeed, iWarmUp);
    MotherOfAll16Seed(dwSeed, iWarmUp);
    CompMWCSeed(dwSeed, iWarmUp);
    XorShift128Seed(dwSeed, iWarmUp);
    XorShiftWeylSeed(dwSeed, iWarmUp);
}

/////////////////////////////////////////////////////////////////////////////////

inline DWord RandomGenerator::_MT_MultiplyMatrix( DWord iX, DWord iY ) const {
    static const DWord MatrixA[2] = { 0x00000000, 0x9908b0df };
    return ( iX ^ (iY >> 1) ^ MatrixA[iY & 1] );
}
inline DWord RandomGenerator::_MT_CombineBits( DWord iX, DWord iY ) const {
    return ( (iX & 0x80000000ul) | (iY & 0x7ffffffful) );
}

