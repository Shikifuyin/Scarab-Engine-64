/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/System/System.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : System dependant APIs.
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
#ifndef SCARAB_THIRDPARTY_SYSTEM_SYSTEM_H
#define SCARAB_THIRDPARTY_SYSTEM_SYSTEM_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "Platform.h"

#include "Hardware/FPU.h"
//#include "Hardware/Hardware.h"

#include "String.h"
#include "File.h"
#include "Threading.h"
#include "Networking.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define SystemFn System::GetInstance()

// Memory Allocation ////////////////////////////////////////////////////////////
//Void * operator new( SizeT iSize, Void * pAddress );
//Void operator delete( Void * pAddress, Void * );   // Required if constructor
//Void operator delete[]( Void * pAddress, Void * ); // throws an exception
// Can't do placement delete, call destructor explicitely !

// Time /////////////////////////////////////////////////////////////////////////
enum TimeUnit
{
    TIMEUNIT_NANOSECONDS = 0,
    TIMEUNIT_MICROSECONDS,
    TIMEUNIT_MILLISECONDS,
    TIMEUNIT_SECONDS
};

enum TimeMonth
{
    MONTH_JANUARY = 1,
    MONTH_FEBRUARY,
    MONTH_MARCH,
    MONTH_APRIL,
    MONTH_MAY,
    MONTH_JUNE,
    MONTH_JULY,
    MONTH_AUGUST,
    MONTH_SEPTEMBER,
    MONTH_OCTOBER,
    MONTH_NOVEMBER,
    MONTH_DECEMBER
};
enum TimeDay
{
    DAY_SUNDAY = 1,
    DAY_MONDAY,
    DAY_TUESDAY,
    DAY_WEDNESDAY,
    DAY_THURSDAY,
    DAY_FRIDAY,
    DAY_SATURDAY
};

typedef UInt64 TickValue;
typedef Double TimeMeasure;

typedef UInt64 TimeValue;

typedef struct _time_date
{
    UShort iYear;         // 1601 - 30827 (ScarabEngine will persist in 28k years !!!)
    TimeMonth iMonth;     // 1 - 12
    UShort iDay;          // 1 - 31
    TimeDay iDayOfWeek;   // 1 - 7
    UShort iHours;        // 0 - 23
    UShort iMinutes;      // 0 - 59
    UShort iSeconds;      // 0 - 59
    UShort iMilliseconds; // 0 - 999
} TimeDate;

typedef struct _time_zone {
    Long iBaseBias; // UTC time = local time + bias

    GChar strStandardName[32];
    Long iStandardBias;     // RealBias = iBaseBias + iStandardBias, typically 0
    TimeDate hStandardTime; // Daylight => Standard switch

    GChar strDaylightName[32];
    Long iDaylightBias;     // RealBias = iBaseBias + iDaylightBias, typically -60
    TimeDate hDaylightTime; // Standard => Daylight switch
} TimeZone;

// Threading ////////////////////////////////////////////////////////////////////
#define WAIT_MAX_OBJECTS 1024

enum WaitResult
{
    WAIT_OK = 0,
    WAIT_EXPIRED,      // Wait has timed-out
    WAIT_ORPHANED,     // Object was abandonned by it's owning thread
    WAIT_IO_INTERRUPT, // Sleeping thread awaken by IO callback
    WAIT_ERROR
};

// Networking ///////////////////////////////////////////////////////////////////
typedef DWord AddressResolveFlags;
#define ADDRRESOLVE_FLAG_NONE            0x00
#define ADDRRESOLVE_FLAG_SERVER_USAGE    0x01 // Address will be used for bind rather than connect
#define ADDRRESOLVE_FLAG_NUMERIC_SERVICE 0x02 // Return port name as numeric
#define ADDRRESOLVE_FLAG_NUMERIC_HOST    0x04 // Return host name as numeric / HostName supplied is numeric
#define ADDRRESOLVE_FLAG_CANONICAL_NAME  0x08 // Give canonical name
#define ADDRRESOLVE_FLAG_RELATIVE_NAME   0x10 // Give RDN (Relative Distinguished Name) instead of FQDN (Fully Qualified Domain Name)
#define ADDRRESOLVE_FLAG_REQUIRED_NAME   0x20 // Name resolution is required, error on fail
#define ADDRRESOLVE_FLAG_DATAGRAM        0x40 // Required for service with different port values accross protocols

/////////////////////////////////////////////////////////////////////////////////
// The System class
class System
{
    // Discrete singleton interface
public:
    inline static System * GetInstance();

private:
    System();
    ~System();

public:
    // Debug helpers ///////////////////////////////////////////////
    Void DebugMessage( const GChar * strMessage ) const;
    Void DebugBreak() const;

    // Memory allocation ///////////////////////////////////////////
    Void * MemAlloc( SizeT iSize ) const;
    Void MemFree( Void * pMemory ) const;    

    // Time ////////////////////////////////////////////////////////
    TickValue TicksAbsolute() const;
    TickValue TicksRelative( TickValue iOrigin ) const;

    TimeMeasure TimeAbsolute( TimeUnit iResolution = TIMEUNIT_NANOSECONDS ) const;
    TimeMeasure TimeRelative( TimeMeasure fStart, TimeUnit iResolution = TIMEUNIT_NANOSECONDS, TimeMeasure * outAbsolute = NULL ) const;

    TimeMeasure ConvertTicks( TickValue iTickValue, TimeUnit iResolution = TIMEUNIT_NANOSECONDS ) const;
    TickValue ConvertTimeMeasure( TimeMeasure fTimeValue, TimeUnit iResolution = TIMEUNIT_NANOSECONDS ) const;

    Bool GetTimeZone( TimeZone * outTimeZone ) const; // returns whether daylight saving is currently active

    Void GetUTCTime( TimeDate * outUTCTime ) const;
    Void GetLocalTime( TimeDate * outLocalTime, const TimeZone * pTimeZone = NULL ) const;

    Void UTCToLocalTime( TimeDate * outLocalTime, const TimeDate & hUTCTime, const TimeZone * pTimeZone = NULL ) const;
    Void LocalToUTCTime( TimeDate * outUTCTime, const TimeDate & hLocalTime, const TimeZone * pTimeZone = NULL ) const;

    TimeValue GetUTCTime( TimeUnit iResolution = TIMEUNIT_MILLISECONDS ) const;
    TimeValue GetLocalTime( TimeUnit iResolution, const TimeZone * pTimeZone = NULL ) const;

    TimeValue UTCToLocalTime( TimeValue iUTCTime, TimeUnit iResolution, const TimeZone * pTimeZone = NULL ) const;
    TimeValue LocalToUTCTime( TimeValue iLocalTime, TimeUnit iResolution, const TimeZone * pTimeZone = NULL ) const;

    TimeValue ConvertTimeDate( const TimeDate & hTimeDate, TimeUnit iResolution = TIMEUNIT_MILLISECONDS ) const;
    Void ConvertTimeValue( TimeDate * outTimeDate, TimeValue iTimeValue, TimeUnit iResolution = TIMEUNIT_MILLISECONDS ) const;

    const GChar * MonthToString( TimeMonth iMonth ) const;
    const GChar * DayToString( TimeDay iDay ) const;

    GChar * FormatTime( GChar * strOutput, const TimeDate & hTimeDate ) const;
    GChar * FormatDate( GChar * strOutput, const TimeDate & hTimeDate ) const;
    GChar * FormatTimeDate( GChar * strOutput, const TimeDate & hTimeDate ) const;
    
    // Console IO //////////////////////////////////////////////////
    UInt ConsoleRead( GChar * outBuffer, UInt iMaxLength ) const;
    Void ConsoleWrite( const GChar * strText, UInt iLength = INVALID_OFFSET ) const;

    // Add more stuff (cursor manipulation, scrolling, clearing, ...)

    // File IO /////////////////////////////////////////////////////
    Bool CreateDirectory( const GChar * strPathName ) const;
    Bool DestroyDirectory( const GChar * strPathName );

    Bool ListDirectoryFirst( const GChar * strPathName, Bool * outIsSubDirectory, GChar * outFileName, UInt iMaxLength );
    Bool ListDirectoryNext( Bool * outIsSubDirectory, GChar * outFileName, UInt iMaxLength );

    Bool GetCurrentDirectory( GChar * outPathName, UInt iMaxLength ) const;
    Bool SetCurrentDirectory( const GChar * strPathName ) const;

    inline HFile CreateFile( const GChar * strPathName, FileMode iMode, FileFlags iFlags = 0 );
	inline HFile OpenFile( const GChar * strPathName, FileMode iMode, FileFlags iFlags = 0 );

    Bool MoveFile( const GChar * strDestPathName, const GChar * strSrcPathName, Bool bOverwrite = false ) const;
    Bool CopyFile( const GChar * strDestPathName, const GChar * strSrcPathName, Bool bOverwrite = false ) const;
    Bool DestroyFile( const GChar * strPathName ) const;

    Bool SearchFile( GChar * outPathName, UInt iMaxLength, const GChar * strSearchPath, const GChar * strSearchName ) const;

    FileAttributes GetFileAttributes( const GChar * strPathName ) const;
    Void SetFileAttributes( const GChar * strPathName, FileAttributes iAttributes ) const;

    Bool CreateHardLink( const GChar * strPathName, const GChar * strTargetPathName ) const;
    Bool CreateSymbolicLink( const GChar * strPathName, const GChar * strTargetPathName, Bool bIsDirectory = false ) const;

    // Processes & Threads /////////////////////////////////////////
    const GChar * GetCommandLine() const;
    const GChar * GetEnvironmentString() const;
    Void GetEnvironmentVar( const GChar * strName, GChar * outValue, UInt iMaxSize ) const;
    Void SetEnvironmentVar( const GChar * strName, const GChar * strValue ) const;

    HProcess GetCurrentProcess() const;
    ProcessID GetCurrentProcessID() const;

    HThread GetCurrentThread() const;
    ThreadID GetCurrentThreadID() const;

    inline HProcess CreateProcess( ProcessCreationFlags iFlags, GChar * strCmdLine, Void * pEnvironment = NULL,
                                   HThread * outMainThread = NULL, ProcessID * outProcessID = NULL, ThreadID * outMainThreadID = NULL ) const;
    inline HProcess OpenProcess( ProcessID iProcessID, ThreadingAccess iAccessFlags ) const;

    inline HThread CreateThread( HThreadRoutine pfEntryPoint, Void * pParameters, Bool bRun = false, ThreadID * outID = NULL ) const;
    inline HThread CreateThread( const HProcess & hHostProcess, HThreadRoutine pfEntryPoint, Void * pParameters, Bool bRun = false,
                                 ThreadID * outID = NULL ) const;
    inline HThread OpenThread( ThreadID iThreadID, ThreadingAccess iAccessFlags ) const;

    inline HThreadEvent CreateThreadEvent( Bool bSignaled, Bool bManualReset = false ) const;
    inline HThreadEvent CreateThreadEvent( const GChar * strName, Bool bSignaled, Bool bManualReset = false ) const;
    inline HThreadEvent OpenThreadEvent( const GChar * strName, ThreadingAccess iAccessFlags ) const;

    inline HMutex CreateMutex( Bool bAcquire = false ) const;
    inline HMutex CreateMutex( const GChar * strName, Bool bAcquire = false ) const;
    inline HMutex OpenMutex( const GChar * strName, ThreadingAccess iAccessFlags ) const;

    inline HSemaphore CreateSemaphore( UInt iInitCount, UInt iMaxCount ) const;
    inline HSemaphore CreateSemaphore( const GChar * strName, UInt iInitCount, UInt iMaxCount ) const;
    inline HSemaphore OpenSemaphore( const GChar * strName, ThreadingAccess iAccessFlags ) const;

    inline HCriticalSection CreateCriticalSection( UInt iSpinCount = 0 ) const;

    WaitResult WaitOne( HThreadingObject * pObject, UInt iMilliseconds = INVALID_SIZE ) const;
    WaitResult WaitAll( HThreadingObject ** arrObjects, UInt iCount, UInt iMilliseconds = INVALID_SIZE ) const;
    WaitResult WaitAny( HThreadingObject ** arrObjects, UInt iCount, UInt iMilliseconds = INVALID_SIZE,
                        HThreadingObject ** outObject = NULL ) const;

    WaitResult WaitInputIdle( const HProcess & hProcess, UInt iMilliseconds = INVALID_SIZE ) const;
    Bool WaitInput() const;

    WaitResult Sleep( UInt iMilliseconds = INVALID_SIZE, Bool bAwakable = false ) const;
    WaitResult YieldTime() const;

    // Networking //////////////////////////////////////////////////
    Void NetworkInitialize() const;
    Void NetworkCleanup() const;

    inline HSocket CreateSocket( SocketProtocol iProtocol ) const;

    Void GetMACAddress( Byte outMACAddress[8] ) const;
    Void GetHostName( GChar outHostName[256] ) const;

    Void StringToNetAddress( NetAddress * outAddress, const GChar * strDottedFormat, NetAddressType iType = NETADDRESS_IPv4 ) const;
    Void NetAddressToString( GChar * outString, const NetAddress * pAddress ) const;

    NetworkError ResolveNetAddress( GChar * outServiceName, GChar * outHostName, const NetAddress * pAddress,
                                    AddressResolveFlags iFlags = ADDRRESOLVE_FLAG_NONE ) const;

    NetworkError ResolveHostName( const GChar * strServiceName, const GChar * strHostName,
                                  SocketProtocol protocolHint = SOCKET_TCPIPv4, AddressResolveFlags iFlags = ADDRRESOLVE_FLAG_NONE );
    Bool EnumResolvedNetAddresses( NetAddress * outAddress, SocketProtocol * outProtocol );

    Word NetByteOrderW( Word iValue ) const;
    DWord NetByteOrderDW( DWord iValue ) const;
    QWord NetByteOrderQW( QWord iValue ) const;
    Word HostByteOrderW( Word iValue ) const;
    DWord HostByteOrderDW( DWord iValue ) const;
    QWord HostByteOrderQW( QWord iValue ) const;

    inline Void NetByteOrder( AddrIPv4 * pAddr ) const;
    //Void NetByteOrder( AddrIPv6 * pAddr ) const;
    inline Void HostByteOrder( AddrIPv4 * pAddr ) const;
    //Void HostByteOrder( AddrIPv6 * pAddr ) const;

private:
    // Memory allocation ///////////////////////////////////////////
    Void * _ScratchAlloc( UInt iSize ) const;
    Void _ScratchFree( Void * pMemory ) const;

    Void * m_hDefaultHeap; // only used as low-level scratch
    Void * m_hMainHeap;

    // Time ////////////////////////////////////////////////////////
    Double m_fPerformanceFrequency; // ticks / second
    Double m_fInvPerformanceFrequency;
    TickValue m_iTicksOverhead;
    Double m_fNanoSecondsOverhead;

    // File IO /////////////////////////////////////////////////////
    Void * m_hDirectoryListing;

    // Networking //////////////////////////////////////////////////
    Void * m_pCurAddrInfo;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "System.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_SYSTEM_SYSTEM_H
