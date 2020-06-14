/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/SIMD/SIMD_Control.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : SIMD, Control operations
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// SIMD::Control implementation
__forceinline UInt32 SIMD::Control::GetCSR() {
    DebugAssert( CPUIDFn->HasSSE() );
    return _mm_getcsr();
}
__forceinline Void SIMD::Control::SetCSR( UInt32 iValue ) {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_setcsr( iValue );
}

__forceinline Void SIMD::Control::ClearAndFlushCacheLine( Void * pAddress ) {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_clflush( pAddress );
}

__forceinline Void SIMD::Control::Pause() {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_pause();
}

__forceinline Void SIMD::Control::SerializeMemoryStore() {
    DebugAssert( CPUIDFn->HasSSE() );
    _mm_sfence();
}
__forceinline Void SIMD::Control::SerializeMemoryLoad() {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_lfence();
}
__forceinline Void SIMD::Control::SerializeMemory() {
    DebugAssert( CPUIDFn->HasSSE2() );
    _mm_mfence();
}

__forceinline Void SIMD::Control::ZeroUpper128() {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_zeroupper();
}
__forceinline Void SIMD::Control::Zero256() {
    DebugAssert( CPUIDFn->HasAVX() );
    _mm256_zeroall();
}

