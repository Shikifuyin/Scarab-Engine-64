/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/CPUID.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CPUID abstraction layer
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// CPUID implementation
inline CPUID * CPUID::GetInstance() {
    static CPUID s_Instance;
    return &s_Instance;
}

inline const Char * CPUID::GetVendorString() const {
    return m_strVendorString;
}
inline Bool CPUID::IsIntel() const {
    return m_bIsIntel;
}
inline Bool CPUID::IsAMD() const {
    return m_bIsAMD;
}

inline const CpuDescriptor * CPUID::GetDescriptor() const {
    return &m_hCPUDesc;
}

inline Bool CPUID::HasMMX() const {
    if ( m_bIsIntel ) {
        return m_hCPUDesc.Intel.FeaturesBasic.bHasMMX;
    } else if ( m_bIsAMD ) {
        return false;
    } else {
        // Unsupported
        DebugAssert( false );
        return false;
    }
}

inline Bool CPUID::HasSSE() const {
    if ( m_bIsIntel ) {
        return m_hCPUDesc.Intel.FeaturesBasic.bHasSSE;
    } else if ( m_bIsAMD ) {
        /////////////////////////////////////////
        return false;
    } else {
        // Unsupported
        DebugAssert( false );
        return false;
    }
}
inline Bool CPUID::HasSSE2() const {
    if ( m_bIsIntel ) {
        return m_hCPUDesc.Intel.FeaturesBasic.bHasSSE2;
    } else if ( m_bIsAMD ) {
        /////////////////////////////////////////
        return false;
    } else {
        // Unsupported
        DebugAssert( false );
        return false;
    }
}
inline Bool CPUID::HasSSE3() const {
    if ( m_bIsIntel ) {
        return m_hCPUDesc.Intel.FeaturesExtended1.bHasSSE3;
    } else if ( m_bIsAMD ) {
        /////////////////////////////////////////
        return false;
    } else {
        // Unsupported
        DebugAssert( false );
        return false;
    }
}
inline Bool CPUID::HasSSSE3() const {
    if ( m_bIsIntel ) {
        return m_hCPUDesc.Intel.FeaturesExtended1.bHasSSSE3;
    } else if ( m_bIsAMD ) {
        /////////////////////////////////////////
        return false;
    } else {
        // Unsupported
        DebugAssert( false );
        return false;
    }
}
inline Bool CPUID::HasSSE41() const {
    if ( m_bIsIntel ) {
        return m_hCPUDesc.Intel.FeaturesExtended1.bHasSSE41;
    } else if ( m_bIsAMD ) {
        /////////////////////////////////////////
        return false;
    } else {
        // Unsupported
        DebugAssert( false );
        return false;
    }
}
inline Bool CPUID::HasSSE42() const {
    if ( m_bIsIntel ) {
        return m_hCPUDesc.Intel.FeaturesExtended1.bHasSSE42;
    } else if ( m_bIsAMD ) {
        /////////////////////////////////////////
        return false;
    } else {
        // Unsupported
        DebugAssert( false );
        return false;
    }
}

inline Bool CPUID::HasAVX() const {
    if ( m_bIsIntel ) {
        return m_hCPUDesc.Intel.FeaturesExtended1.bHasAVX;
    } else if ( m_bIsAMD ) {
        /////////////////////////////////////////
        return false;
    } else {
        // Unsupported
        DebugAssert( false );
        return false;
    }
}
inline Bool CPUID::HasAVX2() const {
    if ( m_bIsIntel ) {
        return m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX2;
    } else if ( m_bIsAMD ) {
        /////////////////////////////////////////
        return false;
    } else {
        // Unsupported
        DebugAssert( false );
        return false;
    }
}
