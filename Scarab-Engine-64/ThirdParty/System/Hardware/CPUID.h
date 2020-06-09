/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/Hardware/CPUID.h
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
// Header prelude
#ifndef SCARAB_THIRDPARTY_SYSTEM_HARDWARE_CPUID_H
#define SCARAB_THIRDPARTY_SYSTEM_HARDWARE_CPUID_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../System.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define CPUIDFn CPUID::GetInstance()

    // Supported Vendors
#define CPU_VENDOR_INTEL      "GenuineIntel"
#define CPU_VENDOR_AMD        "AuthenticAMD"

// Unsupported exotic / historical stuff
//#define CPU_VENDOR_CENTAUR    "CentaurHauls"
//#define CPU_VENDOR_CYRIX      "CyrixInstead"
//#define CPU_VENDOR_TRANSMETA1 "GenuineTMx86"
//#define CPU_VENDOR_TRANSMETA2 "TransmetaCPU"
//#define CPU_VENDOR_NSC        "Geode by NSC"
//#define CPU_VENDOR_NEXGEN     "NexGenDriven"
//#define CPU_VENDOR_RISE       "RiseRiseRise"
//#define CPU_VENDOR_SIS        "SiS SiS SiS "
//#define CPU_VENDOR_UMC        "UMC UMC UMC "
//#define CPU_VENDOR_VIA        "VIA VIA VIA "

    // Intel
#define CPU_INTEL_PROCESSOR_TYPE_ORIGINAL_OEM 0x00
#define CPU_INTEL_PROCESSOR_TYPE_OVERDRIVE    0x01
#define CPU_INTEL_PROCESSOR_TYPE_DUAL         0x02
#define CPU_INTEL_PROCESSOR_TYPE_RESERVED     0x03

typedef struct _cpu_descriptor_intel
{
    // Version information
    Word FamilyID;
    Word ModelID;
    Byte SteppingID;
    Byte ProcessorType;
    Byte BrandIndex;
    Word CacheLineSize;        // Instruction cache line size (clflush)
    Byte MaxLogicalProcessors; // Max addressable logic processor IDs
    Byte InitialAPICID;        // Local APIC ID on power-up, gets replaced by 32-bits x2APIC ID

    // Frequency information (NOT ACCURATE, JUST SPECIFICATION DATA)
    Word BaseFrequency; // in MHz
    Word MaxFrequency;  // in MHz
    Word BusFrequency;  // in MHz

    // System-On-Chip (SOC) Vendor information
    Word SOCVendorID;
    Bool SOCIsStandardVendorID;
    DWord SOCProjectID;
    DWord SOCSteppingID;
    WChar SOCVendorBrand[64];

    // Extended Architecture information
    DWord ExtendedSignature;
    struct _features_architecture {
        Bool bHasLAHFSAHF64;    // LAHF/SAHF instructions available in 64-bits mode
        Bool bHasLZCNT;         // LZCNT instruction
        Bool bHasPreFetchW;     // PREFETCHW instruction
        Bool bHasSysCallSysRet; // 64-bits mode only, SYSCALL/SYSRET instructions
        Bool bHasEDB;           // Execute Disable Bit
        Bool bHas1GBPages;      // 1Gb pages support
        Bool bHasRDTSCP;        // RDTSCP instruction and IA32_TSC_AUX MSR
        Bool bHasIA64;          // Intel 64-bits Architecture (IA64)
    } FeaturesArchitecture;

    // Processor Brand String
    WChar ProcessorBrandString[64];

    // Addressable space
    Byte PhysicalAddressBits;
    Byte LinearAddressBits;

    // Basic Features
    struct _features_basic {
        Bool bHasFPU;        // integrated x87 FPU
        Bool bHasVME;        // virtual 8086 mode enhancements (CR4.VME, CR4.PVI, EFLAGS.VIF, EFLAGS.VIP)
        Bool bHasDE;         // debugging extensions (CR4.DE, DR4, DR5)
        Bool bHasPSE;        // page size extension (CR4.PSE, PDEs, PTEs, CR3)
        Bool bHasTSC;        // time stamp counter, RDTSC instruction (CR4.TSD)
        Bool bHasMSR;        // model specific registers, RDMSR / WRMSR instructions
        Bool bHasPAE;        // physical address extension for pointers greater than 32 bits
        Bool bHasMCE;        // machine check exception, exception 18 is defined (CR4.MCE, model dependant)
        Bool bHasCMPXCHG8B;  // compare and exchange 8 bytes (64 bits) instruction
        Bool bHasAPIC;       // integrated advanced programmable interrupt controller (memory mapped commands in physical 0xfffe0000 - 0xfffe0fff)
        Bool bHasSEP;        // SYSENTER / SYSEXIT instructions and associated MSRs
        Bool bHasMTRR;       // memory type range registers are supported (MTRRcap MSR contains feature bits)
        Bool bHasPGE;        // page global bit support for page mapping, indicating TLB common to diff processes don't need flushing (CR4.PGE)
        Bool bHasMCA;        // machine check architecture for error reporting (MCG_CAP MSR contains feature bits)
        Bool bHasCMOV;       // conditional MOV instructions : CMOV, FCMOV and also FCOMI, FCOMIP, FUCOMI, FUCOMIP
        Bool bHasPAT;        // page attribute table
        Bool bHasPSE36;      // 36-bits page size extension (physical adresses up to 40 bits)
        Bool bHasPSN;        // processor serial number (96 bits identification number)
        Bool bHasCLFLUSH;    // CLFLUSH instruction
        Bool bHasDS;         // debug store, memory resident buffer (BTS and PEBS facilities)
        Bool bHasACPI;       // thermal monitor and software controlled clock facilities (via MSRs)
        Bool bHasMMX;        // intel MMX technology supported
        Bool bHasFXSR;       // FXSAVE / FXRSTOR instructions for fast floating point context switches (CR4.OSFXSR)
        Bool bHasSSE;        // streaming single-instruction multiple-data extensions
        Bool bHasSSE2;       // streaming SIMD extensions 2
        Bool bHasSS;         // self snoop for conflicting memory types management
        Bool bHasHTT;        // multi-threading with more than one logical processor
        Bool bHasTM;         // thermal monitor, automatic thermal control circuitry (TCC)
        Bool bHasPBE;        // pending break enable
    } FeaturesBasic;

    // Extended Features
    struct _features_extended1 {
        Bool bHasSSE3;       // streaming SIMD extensions 3
        Bool bHasPCLMULQDQ;  // carryless multiplication instruction
        Bool bHasDTES64;     // DS area using 64-bit layout
        Bool bHasMONITOR;    // MONITOR/MWAIT available
        Bool bHasDSCPL;      // CPL qualified debug store (for branch messages)
        Bool bHasVMX;        // virtual machine extensions
        Bool bHasSMX;        // safer mode extensions
        Bool bHasEIST;       // enhanced intel speedstep technology
        Bool bHasTM2;        // thermal monitor 2
        Bool bHasSSSE3;      // supplemental SSE3
        Bool bHasCNTXID;     // L1 data cache can be set to either adaptive or shared mode (IA32_MISC_ENABLE MSR bit 24)
        Bool bHasSDBG;       // Silicon debug MSR (IA32_DEBUG_INTERFACE)
        Bool bHasFMA;        // fused multiply add
        Bool bHasCMPXCHG16B; // compare and exchange 16 bytes instruction
        Bool bHasXTPR;       // xTPR update control (IA32_MISC_ENABLE MSR bit 23)
        Bool bHasPDCM;       // performance & debug capability MSR (IA32_PERF_CAPABILITIES MSR)
        Bool bHasPCID;       // Process-Context identifiers
        Bool bHasDCA;        // direct cache access (prefetch data from a memory mapped device)
        Bool bHasSSE41;      // streaming SIMD extensions 4.1
        Bool bHasSSE42;      // streaming SIMD extensions 4.2
        Bool bHasX2APIC;     // x2APIC supported
        Bool bHasMOVBE;      // move data after swapping bytes instruction
        Bool bHasPOPCNT;     // return number of bits set to 1 instruction
        Bool bHasTSCD;       // local APIC timer supports one-shot operation using a TSC deadline value
        Bool bHasAES;        // AESNI instruction extensions (AES 128 192 and 256 bits, symmetric encrypt/decrypt, all RJINDAEL cyphers)
        Bool bHasXSAVE;      // XSAVE / XRSTOR extended states, XSETBV / XGETBV instructions and XCR0 register (XFEATURE_ENABLED_MASK)
        Bool bHasOSXSAVE;    // OS has enabled XSETBV / XGETBV to access XCR0 and extended state management using XSAVE / XRSTOR
        Bool bHasAVX;        // AVX instructions extension
        Bool bHasF16C;       // 16-bits floating point conversion
        Bool bHasRDRAND;     // RDRAND instruction
    } FeaturesExtended1;

    // More Extended Features
    struct _features_extended2 {
        Bool bHasFSGSBASE;          // RDFSBASE, RDGSBASE, WRDFSBASE, WRDGSBASE instructions
        Bool bHasTSCAdjust;         // TSC adjust (IA32_TSC_ADJUST MSR)
        Bool bHasSGX;               // Software Guard Extensions
        Bool bHasBMI1;              // Bit Manipulation instruction set 1
        Bool bHasHLE;               // Hardware Lock Elision
        Bool bHasAVX2;              // AVX2 instructions extension
        Bool bHasFDPExcptOnly;      // only update FPU data pointer on FPU exception
        Bool bHasSMEP;              // Supervisor-Mode Execution Prevention
        Bool bHasBMI2;              // Bit Manipulation instruction set 2
        Bool bHasEREP;              // Enhanced REP MOVSB / STOSB instructions
        Bool bHasINVPCID;           // INVPCID instruction for Process-Context identifiers
        Bool bHasRTM;               // ???
        Bool bHasRDTM;              // Resource Director Technology Monitoring
        Bool bIsFPUCSDSDeprecated;  // FPU CS and DS values are deprecated
        Bool bHasMPX;               // Memory Protection Extensions
        Bool bHasRDTA;              // Resource Director Technology Allocation
        Bool bHasAVX512_F;          // AVX512F instructions extension
        Bool bHasAVX512_DQ;         // AVX512DQ instructions extension
        Bool bHasRDSEED;            // RDSEED instruction
        Bool bHasADX;               // arbitrary precision arithmetics (ADCX and ADOX instructions)
        Bool bHasSMAP;              // Supervisor-Mode Access Prevention (and CLAC/STAC instructions)
        Bool bHasAVX512_IFMA;       // AVX512 IFMA instructions extension
        Bool bHasCLFLUSHOPT;        // CLFLUSHOPT instruction
        Bool bHasCLWB;              // CLWB instruction
        Bool bHasProcTrace;         // Processor Trace
        Bool bHasAVX512_PF;         // AVX512PF instructions extension (Xeon Phi only)
        Bool bHasAVX512_ER;         // AVX512ER instructions extension (Xeon Phi only)
        Bool bHasAVX512_CD;         // AVX512CD instructions extension
        Bool bHasSHA;               // Secure Hash Algorithm extensions (SHA)
        Bool bHasAVX512_BW;         // AVX512BW instructions extension
        Bool bHasAVX512_VL;         // AVX512VL instructions extension

        Bool bHasPreFetchWT1;       // ??? (Xeon Phi only)
        Bool bHasAVX512_VBMI;       // AVX512 VBMI instructions extension
        Bool bHasUMIP;              // User-Mode Instruction Prevention
        Bool bHasPKU;               // Protection Keys for User-Mode pages
        Bool bHasOSPKE;             // Protection Keys enabled by OS (and RDPKRU, WRPKRU instructions)
        Bool bHasWAITPKG;           // ???
        Bool bHasAVX512_VBMI2;      // AVX512 VBMI2 instructions extension
        Bool bHasCET_SS;            // CET Shadow Stack features
        Bool bHasGFNI;              // ???
        Bool bHasVAES;              // ???
        Bool bHasVPCLMULQDQ;        // VPCLMULQDQ instruction
        Bool bHasAVX512_VNNI;       // AVX512 VNNI instructions extension
        Bool bHasAVX512_BITALG;     // AVX512 BITALG instructions extension
        Bool bHasAVX512_VPOPCNTDQ;  // AVX512 VPOPCNTDQ instruction (Xeon Phi only)
        Byte iMAWAU;                // Value used by BNDLDX and BNDSTX instructions in 64-bits mode
        Bool bHasRDPID;             // RDPID and IA32_TSC_AUX are available
        Bool bHasCLDEMOTE;          // cache-line demote instruction
        Bool bHasMOVDIRI;           // MOVDIRI instruction
        Bool bHasMOVDIR64B;         // MOVDIR64B instruction
        Bool bHasSGX_LC;            // SGX Launch Configuration
        Bool bHasPKS;               // Protection Keys for Supervisor-Mode pages

        Bool bHasAVX512_4VNNIW;     // AVX512 4VNNIW instructions extension (Xeon Phi only)
        Bool bHasAVX512_4FMAPS;     // AVX512 4FMAPS instructions extension (Xeon Phi only)
        Bool bHasFSREPMOV;          // fast short REP MOV instruction
        Bool bHasMDCLEAR;           // MD_CLEAR instruction
        Bool bIsHybrid;             // Processor is a Hybrid part
        Bool bHasCET_IBT;           // CET Indirect Branch Tracking features
        Bool bHasIBRSIBPB;          // Indirect Branch Restricted Speculation and Indirect Branch Predictor Barrier (IA32_SPEC_CTRL and IA32_PRED_CMD MSRs)
        Bool bHasSTIBP;             // Single Thread Indirect Branch Predictor (IA32_SPEC_CTRL MSR)
        Bool bHasL1DFLUSH;          // IA32_FLUSH_CMD MSR
        Bool bHasArchCaps;          // IA32_ARCH_CAPABILITIES MSR
        Bool bHasCoreCaps;          // IA32_CORE_CAPABILITIES MSR
        Bool bHasSSBD;              // Speculative Store Bypass Disable (IA32_SPEC_CTRL MSR)
    } FeaturesExtended2;
} CpuDescriptorIntel;

    // AMD
typedef struct _cpu_descriptor_amd
{
    ////////////////////////////////////
} CpuDescriptorAMD;

    // Generic CPU Descriptor
typedef union _cpu_descriptor
{
    CpuDescriptorIntel Intel;
    CpuDescriptorAMD AMD;
} CpuDescriptor;

/////////////////////////////////////////////////////////////////////////////////
// The CPUID class
class CPUID
{
    // Discrete singleton interface
public:
    inline static CPUID * GetInstance();

private:
    CPUID();
    ~CPUID();

public:
    inline const Char * GetVendorString() const;
    inline Bool IsIntel() const;
    inline Bool IsAMD() const;

    inline const CpuDescriptor * GetDescriptor() const;

    // MMX support
    inline Bool HasMMX() const;

    // SSE support
    inline Bool HasSSE() const;
    inline Bool HasSSE2() const;
    inline Bool HasSSE3() const;
    inline Bool HasSSSE3() const;
    inline Bool HasSSE41() const;
    inline Bool HasSSE42() const;

    // AVX support
    inline Bool HasAVX() const;
    inline Bool HasAVX2() const;
    // AVX512 not available on development machine ! Can't implement yet ...

private:
    Void _Initialize();

    Int m_iHighestFunctionID;
    Int m_iHighestExtendedID;

    Char m_strVendorString[32];
    Bool m_bIsIntel;
    Bool m_bIsAMD;

    CpuDescriptor m_hCPUDesc;

    // CPU : Intel /////////////////////////////////////////////////////////
    Void _Intel_VersionAndFeatures();
    Void _Intel_MoreExtendedFeatures();
    Void _Intel_ProcessorFrequency();
    Void _Intel_SOCVendorData();
    Void _Intel_Architecture();
    Void _Intel_ProcessorBrandString();
    Void _Intel_AdressableSpace();

    // CPU : AMD ///////////////////////////////////////////////////////////

};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CPUID.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_HARDWARE_CPUID_H
