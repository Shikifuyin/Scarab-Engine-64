/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/CPUID.cpp
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
// Third-Party Includes
#include <intrin.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CPUID.h"

/////////////////////////////////////////////////////////////////////////////////
// Hardware implementation
CPUID::CPUID()
{
    MemZero( &m_hCPUDesc, sizeof(m_hCPUDesc) );

    // Get Vendor and CPUID Caps
    _Initialize();

    // Vendor-Dependant
    if ( m_bIsIntel ) {
        _Intel_VersionAndFeatures();
        _Intel_MoreExtendedFeatures();
        _Intel_ProcessorFrequency();
        _Intel_SOCVendorData();
        _Intel_Architecture();
        _Intel_ProcessorBrandString();
        _Intel_AdressableSpace();
    } else if ( m_bIsAMD ) {
        /////////////////////////////////////////
    } else {
        // Unsupported
        DebugAssert( false );
    }
}
CPUID::~CPUID()
{
    // nothing to do
}

/////////////////////////////////////////////////////////////////////////////////

Void CPUID::_Initialize()
{
    Int arrData[4]; // EAX, EBX, ECX, EDX
    __cpuid( arrData, 0 );

    // Get valid function IDs
    m_iHighestFunctionID = arrData[0];

    // Get Vendor String
    MemZero( m_strVendorString, 32 );
    *( (Int*)m_strVendorString ) = arrData[1];
    *( (Int*)(m_strVendorString + 4) ) = arrData[3];
    *( (Int*)(m_strVendorString + 8) ) = arrData[2];

    // Check for Intel & AMD
    m_bIsIntel = false;
    m_bIsAMD = false;
    if ( StringFn->CmpA( (const AChar *)m_strVendorString, CPU_VENDOR_INTEL ) == 0 )
        m_bIsIntel = true;
    else if ( StringFn->CmpA( (const AChar *)m_strVendorString, CPU_VENDOR_AMD ) == 0 )
        m_bIsAMD = true;

    // Get valid extended IDs
    __cpuid( arrData, 0x80000000 );
    m_iHighestExtendedID = arrData[0];
}

// CPU : Intel /////////////////////////////////////////////////////////

Void CPUID::_Intel_VersionAndFeatures()
{
    if ( m_iHighestFunctionID < 0x01 )
        return;
    Int arrData[4]; // EAX, EBX, ECX, EDX
    __cpuid( arrData, 0x01 );

    // Version data
    Word wFamilyID     = (Word)( (arrData[0] & 0x00000f00) >> 8 );
    Word wExtFamilyID  = (Word)( (arrData[0] & 0x0ff00000) >> 20 );
    if ( wFamilyID == 0x0f )
        m_hCPUDesc.Intel.FamilyID = wFamilyID + wExtFamilyID;
    else
        m_hCPUDesc.Intel.FamilyID = wFamilyID;

    Word wModelID       = (Word)( (arrData[0] & 0x000000f0) >> 4 );
    Word wExtModelID    = (Word)( (arrData[0] & 0x000f0000) >> 16 );
    if ( wFamilyID == 0x06 || wFamilyID == 0x0f )
        m_hCPUDesc.Intel.ModelID = wModelID + (wExtModelID << 4);
    else
        m_hCPUDesc.Intel.ModelID = wModelID;

    m_hCPUDesc.Intel.SteppingID     = (Byte)( (arrData[0] & 0x0000000f) );
    m_hCPUDesc.Intel.ProcessorType  = (Byte)( (arrData[0] & 0x00003000) >> 12 );

    m_hCPUDesc.Intel.BrandIndex             = (Byte)( (arrData[1] & 0x000000ff) );
    m_hCPUDesc.Intel.CacheLineSize          = (Word)( (arrData[1] & 0x0000ff00) >> 5 );
    m_hCPUDesc.Intel.MaxLogicalProcessors   = (Byte)( (arrData[1] & 0x00ff0000) >> 16 );
    m_hCPUDesc.Intel.InitialAPICID          = (Byte)( (arrData[1] & 0xff000000) >> 24 );

    // Features data
    m_hCPUDesc.Intel.FeaturesBasic.bHasFPU       = ( (arrData[3] & 0x00000001) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasVME       = ( (arrData[3] & 0x00000002) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasDE        = ( (arrData[3] & 0x00000004) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasPSE       = ( (arrData[3] & 0x00000008) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasTSC       = ( (arrData[3] & 0x00000010) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasMSR       = ( (arrData[3] & 0x00000020) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasPAE       = ( (arrData[3] & 0x00000040) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasMCE       = ( (arrData[3] & 0x00000080) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasCMPXCHG8B = ( (arrData[3] & 0x00000100) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasAPIC      = ( (arrData[3] & 0x00000200) != 0 );
    //m_hCPUDesc.Intel.FeaturesBasic.b__         = ( (arrData[3] & 0x00000400) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasSEP       = ( (arrData[3] & 0x00000800) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasMTRR      = ( (arrData[3] & 0x00001000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasPGE       = ( (arrData[3] & 0x00002000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasMCA       = ( (arrData[3] & 0x00004000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasCMOV      = ( (arrData[3] & 0x00008000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasPAT       = ( (arrData[3] & 0x00010000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasPSE36     = ( (arrData[3] & 0x00020000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasPSN       = ( (arrData[3] & 0x00040000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasCLFLUSH   = ( (arrData[3] & 0x00080000) != 0 );
    //m_hCPUDesc.Intel.FeaturesBasic.b__         = ( (arrData[3] & 0x00100000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasDS        = ( (arrData[3] & 0x00200000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasACPI      = ( (arrData[3] & 0x00400000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasMMX       = ( (arrData[3] & 0x00800000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasFXSR      = ( (arrData[3] & 0x01000000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasSSE       = ( (arrData[3] & 0x02000000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasSSE2      = ( (arrData[3] & 0x04000000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasSS        = ( (arrData[3] & 0x08000000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasHTT       = ( (arrData[3] & 0x10000000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasTM        = ( (arrData[3] & 0x20000000) != 0 );
    //m_hCPUDesc.Intel.FeaturesBasic.b__         = ( (arrData[3] & 0x40000000) != 0 );
    m_hCPUDesc.Intel.FeaturesBasic.bHasPBE       = ( (arrData[3] & 0x80000000) != 0 );

    // Extended1 Features data
    m_hCPUDesc.Intel.FeaturesExtended1.bHasSSE3          = ( (arrData[2] & 0x00000001) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasPCLMULQDQ     = ( (arrData[2] & 0x00000002) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasDTES64        = ( (arrData[2] & 0x00000004) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasMONITOR       = ( (arrData[2] & 0x00000008) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasDSCPL         = ( (arrData[2] & 0x00000010) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasVMX           = ( (arrData[2] & 0x00000020) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasSMX           = ( (arrData[2] & 0x00000040) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasEIST          = ( (arrData[2] & 0x00000080) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasTM2           = ( (arrData[2] & 0x00000100) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasSSSE3         = ( (arrData[2] & 0x00000200) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasCNTXID        = ( (arrData[2] & 0x00000400) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasSDBG          = ( (arrData[2] & 0x00000800) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasFMA           = ( (arrData[2] & 0x00001000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasCMPXCHG16B    = ( (arrData[2] & 0x00002000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasXTPR          = ( (arrData[2] & 0x00004000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasPDCM          = ( (arrData[2] & 0x00008000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended1.bHas_           = ( (arrData[2] & 0x00010000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasPCID          = ( (arrData[2] & 0x00020000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasDCA           = ( (arrData[2] & 0x00040000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasSSE41         = ( (arrData[2] & 0x00080000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasSSE42         = ( (arrData[2] & 0x00100000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasX2APIC        = ( (arrData[2] & 0x00200000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasMOVBE         = ( (arrData[2] & 0x00400000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasPOPCNT        = ( (arrData[2] & 0x00800000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasTSCD          = ( (arrData[2] & 0x01000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasAES           = ( (arrData[2] & 0x02000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasXSAVE         = ( (arrData[2] & 0x04000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasOSXSAVE       = ( (arrData[2] & 0x08000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasAVX           = ( (arrData[2] & 0x10000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasF16C          = ( (arrData[2] & 0x20000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended1.bHasRDRAND        = ( (arrData[2] & 0x40000000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended1.bHas_           = ( (arrData[2] & 0x80000000) != 0 );
}
Void CPUID::_Intel_MoreExtendedFeatures()
{
    if ( m_iHighestFunctionID < 0x07 )
        return;
    Int arrData[4]; // EAX, EBX, ECX, EDX
    __cpuidex( arrData, 0x07, 0 );

    DWord dwMaxSubFunctionID = (DWord)( arrData[0] );

    // Extended2 Features data
    m_hCPUDesc.Intel.FeaturesExtended2.bHasFSGSBASE             = ( (arrData[1] & 0x00000001) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasTSCAdjust            = ( (arrData[1] & 0x00000002) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasSGX                  = ( (arrData[1] & 0x00000004) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasBMI1                 = ( (arrData[1] & 0x00000008) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasHLE                  = ( (arrData[1] & 0x00000010) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX2                 = ( (arrData[1] & 0x00000020) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasFDPExcptOnly         = ( (arrData[1] & 0x00000040) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasSMEP                 = ( (arrData[1] & 0x00000080) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasBMI2                 = ( (arrData[1] & 0x00000100) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasEREP                 = ( (arrData[1] & 0x00000200) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasINVPCID              = ( (arrData[1] & 0x00000400) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasRTM                  = ( (arrData[1] & 0x00000800) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasRDTM                 = ( (arrData[1] & 0x00001000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bIsFPUCSDSDeprecated     = ( (arrData[1] & 0x00002000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasMPX                  = ( (arrData[1] & 0x00004000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasRDTA                 = ( (arrData[1] & 0x00008000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_F             = ( (arrData[1] & 0x00010000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_DQ            = ( (arrData[1] & 0x00020000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasRDSEED               = ( (arrData[1] & 0x00040000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasADX                  = ( (arrData[1] & 0x00080000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasSMAP                 = ( (arrData[1] & 0x00100000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_IFMA          = ( (arrData[1] & 0x00200000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[1] & 0x00400000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasCLFLUSHOPT           = ( (arrData[1] & 0x00800000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasCLWB                 = ( (arrData[1] & 0x01000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasProcTrace            = ( (arrData[1] & 0x02000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_PF            = ( (arrData[1] & 0x04000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_ER            = ( (arrData[1] & 0x08000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_CD            = ( (arrData[1] & 0x10000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasSHA                  = ( (arrData[1] & 0x20000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_BW            = ( (arrData[1] & 0x40000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_VL            = ( (arrData[1] & 0x80000000) != 0 );

    m_hCPUDesc.Intel.FeaturesExtended2.bHasPreFetchWT1          = ( (arrData[2] & 0x00000001) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_VBMI          = ( (arrData[2] & 0x00000002) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasUMIP                 = ( (arrData[2] & 0x00000004) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasPKU                  = ( (arrData[2] & 0x00000008) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasOSPKE                = ( (arrData[2] & 0x00000010) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasWAITPKG              = ( (arrData[2] & 0x00000020) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_VBMI2         = ( (arrData[2] & 0x00000040) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasCET_SS               = ( (arrData[2] & 0x00000080) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasGFNI                 = ( (arrData[2] & 0x00000100) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasVAES                 = ( (arrData[2] & 0x00000200) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasVPCLMULQDQ           = ( (arrData[2] & 0x00000400) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_VNNI          = ( (arrData[2] & 0x00000800) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_BITALG        = ( (arrData[2] & 0x00001000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[2] & 0x00002000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_VPOPCNTDQ     = ( (arrData[2] & 0x00004000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[2] & 0x00008000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[2] & 0x00010000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.iMAWAU             = (Byte)( (arrData[2] & 0x003e0000) >> 17 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasRDPID                = ( (arrData[2] & 0x00400000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[2] & 0x00800000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[2] & 0x01000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasCLDEMOTE             = ( (arrData[2] & 0x02000000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[2] & 0x04000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasMOVDIRI              = ( (arrData[2] & 0x08000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasMOVDIR64B            = ( (arrData[2] & 0x10000000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[2] & 0x20000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasSGX_LC               = ( (arrData[2] & 0x40000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasPKS                  = ( (arrData[2] & 0x80000000) != 0 );

    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00000001) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00000002) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_4VNNIW        = ( (arrData[3] & 0x00000004) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasAVX512_4FMAPS        = ( (arrData[3] & 0x00000008) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasFSREPMOV             = ( (arrData[3] & 0x00000010) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00000020) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00000040) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00000080) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00000100) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00000200) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasMDCLEAR              = ( (arrData[3] & 0x00000400) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00000800) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00001000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00002000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00004000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bIsHybrid                = ( (arrData[3] & 0x00008000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00010000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00020000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00040000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00080000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasCET_IBT              = ( (arrData[3] & 0x00100000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00200000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00400000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x00800000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x01000000) != 0 );
    //m_hCPUDesc.Intel.FeaturesExtended2.bHas_                  = ( (arrData[3] & 0x02000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasIBRSIBPB             = ( (arrData[3] & 0x04000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasSTIBP                = ( (arrData[3] & 0x08000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasL1DFLUSH             = ( (arrData[3] & 0x10000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasArchCaps             = ( (arrData[3] & 0x20000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasCoreCaps             = ( (arrData[3] & 0x40000000) != 0 );
    m_hCPUDesc.Intel.FeaturesExtended2.bHasSSBD                 = ( (arrData[3] & 0x80000000) != 0 );
}
Void CPUID::_Intel_ProcessorFrequency()
{
    if ( m_iHighestFunctionID < 0x16 )
        return;
    Int arrData[4]; // EAX, EBX, ECX, EDX
    __cpuidex( arrData, 0x16, 0 );

    m_hCPUDesc.Intel.BaseFrequency = (Word)( arrData[0] & 0x0000ffff );
    m_hCPUDesc.Intel.MaxFrequency = (Word)( arrData[1] & 0x0000ffff );
    m_hCPUDesc.Intel.BusFrequency = (Word)( arrData[2] & 0x0000ffff );
}
Void CPUID::_Intel_SOCVendorData()
{
    if ( m_iHighestFunctionID < 0x17 )
        return;
    Int arrData[4]; // EAX, EBX, ECX, EDX
    __cpuidex( arrData, 0x17, 0 );

    int iMaxSubFunctionID = arrData[0];
    if ( iMaxSubFunctionID < 3 )
        return;

    m_hCPUDesc.Intel.SOCVendorID            = (Word)( arrData[1] & 0x0000ffff );
    m_hCPUDesc.Intel.SOCIsStandardVendorID  = ( (arrData[1] & 0x00010000) != 0 );
    m_hCPUDesc.Intel.SOCProjectID           = (DWord)( arrData[2] & 0x0000ffff );
    m_hCPUDesc.Intel.SOCSteppingID          = (DWord)( arrData[3] & 0x0000ffff );

    MBChar arrSOCBrandString[64];
    Byte * pCurChar = (Byte*)arrSOCBrandString;
    for ( int i = 1; i <= 3; ++i ) {
        __cpuidex( arrData, 0x17, i );
        *(pCurChar++) = ( arrData[0] & 0x000000ff );
        *(pCurChar++) = ( arrData[0] & 0x0000ff00 ) >> 8;
        *(pCurChar++) = ( arrData[0] & 0x00ff0000 ) >> 16;
        *(pCurChar++) = ( arrData[0] & 0xff000000 ) >> 24;
        *(pCurChar++) = ( arrData[1] & 0x000000ff );
        *(pCurChar++) = ( arrData[1] & 0x0000ff00 ) >> 8;
        *(pCurChar++) = ( arrData[1] & 0x00ff0000 ) >> 16;
        *(pCurChar++) = ( arrData[1] & 0xff000000 ) >> 24;
        *(pCurChar++) = ( arrData[2] & 0x000000ff );
        *(pCurChar++) = ( arrData[2] & 0x0000ff00 ) >> 8;
        *(pCurChar++) = ( arrData[2] & 0x00ff0000 ) >> 16;
        *(pCurChar++) = ( arrData[2] & 0xff000000 ) >> 24;
        *(pCurChar++) = ( arrData[3] & 0x000000ff );
        *(pCurChar++) = ( arrData[3] & 0x0000ff00 ) >> 8;
        *(pCurChar++) = ( arrData[3] & 0x00ff0000 ) >> 16;
        *(pCurChar++) = ( arrData[3] & 0xff000000 ) >> 24;
    }

    StringFn->MultiByteToWideChar( m_hCPUDesc.Intel.SOCVendorBrand, arrSOCBrandString, 64 );
}
Void CPUID::_Intel_Architecture()
{
    if ( m_iHighestExtendedID < 0x80000001 )
        return;
    Int arrData[4]; // EAX, EBX, ECX, EDX
    __cpuidex( arrData, 0x80000001, 0 );

    m_hCPUDesc.Intel.ExtendedSignature = (DWord)( arrData[0] );

    m_hCPUDesc.Intel.FeaturesArchitecture.bHasLAHFSAHF64    = ( (arrData[2] & 0x00000001) != 0 );
    m_hCPUDesc.Intel.FeaturesArchitecture.bHasLZCNT         = ( (arrData[2] & 0x00000020) != 0 );
    m_hCPUDesc.Intel.FeaturesArchitecture.bHasPreFetchW     = ( (arrData[2] & 0x00000100) != 0 );

    m_hCPUDesc.Intel.FeaturesArchitecture.bHasSysCallSysRet = ( (arrData[3] & 0x00000800) != 0 );
    m_hCPUDesc.Intel.FeaturesArchitecture.bHasEDB           = ( (arrData[3] & 0x00100000) != 0 );
    m_hCPUDesc.Intel.FeaturesArchitecture.bHas1GBPages      = ( (arrData[3] & 0x04000000) != 0 );
    m_hCPUDesc.Intel.FeaturesArchitecture.bHasRDTSCP        = ( (arrData[3] & 0x08000000) != 0 );
    m_hCPUDesc.Intel.FeaturesArchitecture.bHasIA64          = ( (arrData[3] & 0x20000000) != 0 );
}
Void CPUID::_Intel_ProcessorBrandString()
{
    if ( m_iHighestExtendedID < 0x80000004 )
        return;
    Int arrData[4]; // EAX, EBX, ECX, EDX

    AChar arrProcessorBrandString[64];
    Byte * pCurChar = (Byte*)arrProcessorBrandString;
    for ( int i = 2; i <= 4; ++i ) {
        __cpuidex( arrData, 0x80000000 + i, 0 );
        *(pCurChar++) = ( arrData[0] & 0x000000ff );
        *(pCurChar++) = ( arrData[0] & 0x0000ff00 ) >> 8;
        *(pCurChar++) = ( arrData[0] & 0x00ff0000 ) >> 16;
        *(pCurChar++) = ( arrData[0] & 0xff000000 ) >> 24;
        *(pCurChar++) = ( arrData[1] & 0x000000ff );
        *(pCurChar++) = ( arrData[1] & 0x0000ff00 ) >> 8;
        *(pCurChar++) = ( arrData[1] & 0x00ff0000 ) >> 16;
        *(pCurChar++) = ( arrData[1] & 0xff000000 ) >> 24;
        *(pCurChar++) = ( arrData[2] & 0x000000ff );
        *(pCurChar++) = ( arrData[2] & 0x0000ff00 ) >> 8;
        *(pCurChar++) = ( arrData[2] & 0x00ff0000 ) >> 16;
        *(pCurChar++) = ( arrData[2] & 0xff000000 ) >> 24;
        *(pCurChar++) = ( arrData[3] & 0x000000ff );
        *(pCurChar++) = ( arrData[3] & 0x0000ff00 ) >> 8;
        *(pCurChar++) = ( arrData[3] & 0x00ff0000 ) >> 16;
        *(pCurChar++) = ( arrData[3] & 0xff000000 ) >> 24;
    }

    StringFn->AsciiToWideChar( m_hCPUDesc.Intel.ProcessorBrandString, arrProcessorBrandString, 64 );
}
Void CPUID::_Intel_AdressableSpace()
{
    if ( m_iHighestExtendedID < 0x80000008 )
        return;
    Int arrData[4]; // EAX, EBX, ECX, EDX
    __cpuidex( arrData, 0x80000008, 0 );

    m_hCPUDesc.Intel.PhysicalAddressBits    = (Byte)( arrData[0] & 0x000000ff );
    m_hCPUDesc.Intel.LinearAddressBits      = (Byte)( (arrData[0] & 0x0000ff00) >> 8 );
}

