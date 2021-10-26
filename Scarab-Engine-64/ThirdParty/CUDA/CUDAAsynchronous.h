/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDAAsynchronous.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA Streams for asynchronous operations
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
#ifndef SCARAB_THIRDPARTY_CUDA_CUDAASYNCHRONOUS_H
#define SCARAB_THIRDPARTY_CUDA_CUDAASYNCHRONOUS_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUDAMappings.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Prototypes
class CUDAEvent;
class CUDAStream;

class CUDAManagedMemory;
class CUDAGraph;
class CUDAGraphInstance;
class CUDANodeEventRecord;
class CUDANodeEventRequire;

// Host Functions
typedef Void (__stdcall * CUDAHostFunction)( Void * pUserData );

/////////////////////////////////////////////////////////////////////////////////
// The CUDAEvent class
class CUDAEvent
{
public:
	CUDAEvent();
	~CUDAEvent();
	
	// Deferred Creation/Destruction
		// Use CUDA_EVENT_FLAG_NOTIMING for optimal performance when timing is not required
		// Destroying an event is Asynchronous and will happen after event completion
	inline Bool IsCreated() const;
	Void Create( UInt iEventFlags = CUDA_EVENT_FLAG_DEFAULT );
	Void Destroy();
	
	// Synchronization
		// Check if an event has been completed
	Bool IsCompleted() const;
		// Wait for an event to complete (ie. Synchronize)
	Void WaitCompletion() const;
	
		// Timing (Milliseconds, resolution ~ 0.5 microseconds)
		// Both events need to have completed !
	Float GetElapsedTime( CUDAEvent * pPreviousEvent ) const;
	
private:
	friend class CUDAStream;
	friend class CUDAGraph;
	friend class CUDANodeEventRecord;
	friend class CUDANodeEventRequire;
	
	Void * m_hEvent;
};

/////////////////////////////////////////////////////////////////////////////////
// The CUDAStream class
class CUDAStream
{
public:
	CUDAStream();
	~CUDAStream();
	
	// Default CUDA streams
		// Legacy stream synchronizes with all other non-concurrent streams on the same CUDA device :
		// -> Add work to legacy stream (kernel launch or using RequireStep for instance)
		// -> Legacy stream waits for all non-concurrent streams to complete
		// -> Work is queued in legacy stream
		// -> All non-concurrent streams wait for legacy stream to complete
	static CUDAStream * GetDefaultLegacyStream();
		// PerThread streams are local to both a CUDA device and the current thread, they do NOT
		// synchronize with other streams, like explicitely created user streams.
		// PerThread streams are non-concurrent and will synchronize with the Legacy stream as stated above.
	static CUDAStream * GetDefaultPerThreadStream();
	
	// Deferred Creation/Destruction
	inline Bool IsCreated() const;
		// Concurrent streams are non-blocking and will not implicitely synchronize with the Legacy stream.
		// Lower priority index => Higher priority
		// Lowest/Highest meaningful priority indices can be obtained from CUDAContext::GetCurrentDeviceStreamPriorityRange.
	Void Create( Bool bConcurrent = false, Int iPriority = 0 );
		// Destroying a stream is Asynchronous and will happen after stream completion
	Void Destroy();
	
	Bool IsConcurrent() const;
	Int GetPriority() const;
	
	// Cache Hints & Synchronization Policy
	// Low-level cache hints & synchronization behaviour
	// Users should never have to mess with those unless they REALLY know what they're doing
	// Flagged as unsafe, disabled ! Let the driver API layer do the job ...
	
	// Void GetAccessPolicy() const;
	// Void SetAccessPolicy() const;
	
	// CUDAStreamSyncPolicy GetSynchronizationPolicy() const;
	// Void SetSynchronizationPolicy( CUDAStreamSyncPolicy iSynchronizationPolicy ) const;
	
	// Void CopyAttributes( CUDAStream * pSrcStream ) const; // Must be on same CUDA device !
	
	// Attach Memory (Asynchronous, stream-ordered)
	Void AttachMemory( Void * pSystemMemory, SizeT iSize, UInt iAttachMemoryFlags = CUDA_STREAM_ATTACH_MEMORY_FLAG_SINGLE ) const;
	Void AttachMemory( CUDAManagedMemory * pManagedMemory, UInt iAttachMemoryFlags = CUDA_STREAM_ATTACH_MEMORY_FLAG_SINGLE ) const;
	
	// Synchronization
		// Check if a stream has been completed
	Bool IsCompleted() const;
		// Wait for a stream to complete (ie. Synchronize)
	Void WaitCompletion() const;
	
		// Record an event, storing all work in the stream up to this point
	Void RecordEvent( CUDAEvent * pEvent, Bool bExternalEvent = false ) const;
	
		// Make all further work submitted to the stream wait for an event to complete
		// The required event may belong to another stream, allowing multiple streams synchronization
	Void RequireEvent( CUDAEvent * pEvent, Bool bExternalEvent = false ) const;
	
	// Capture (deferred execution)
	Void CaptureBegin( CUDAStreamCaptureMode iMode ) const;
	Void CaptureEnd( CUDAGraph * outGraph ) const;
	
	Void GetCaptureInfo( CUDAStreamCaptureStatus * outStatus, UInt64 * outCaptureID ) const;
	inline Bool IsCapturing() const;
	
		// Use in a push-pop fashion :
		// CUDAStreamCaptureMode iMode = iDesiredMode;
		// CUDAStream::SwapThreadCaptureMode( &iMode ); // Enter new mode
		// ...
		// CUDAStream::SwapThreadCaptureMode( &iMode ); // Restore previous mode
	static Void SwapThreadCaptureMode( CUDAStreamCaptureMode * pMode );

	// Host Execution support
		// Host function is enqueued after currently queued work and will block further enqueued work.
		// Host function CANNOT make any CUDA API call !
		// Host function CANNOT perform synchronization that may depend on outstanding CUDA work.
		// Host function in independent streams execute in undefined order and may be serialized.
		// Stream is idle during execution, thus using stream-attached memory is always fine.
		// Function execution start is a synchronizing operation.
		// Adding device work does NOT make the stream active if there are pending host functions.
		// Function execution end does NOT make the stream active unless there is further device work.
		// Stream stays idle across consecutive host functions unless device work is in-between.
	Void PushHostFunction( CUDAHostFunction pfHostFunction, Void * pUserData ) const;
	
private:
	friend class CUDAKernel;
	friend class CUDAGraphInstance;
	friend class CUBLASContext;
	friend class CUSolverContextDense;
	
	Void * m_hStream;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUDAAsynchronous.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_CUDAASYNCHRONOUS_H
