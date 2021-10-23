/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDAAsynchronous.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA Events & Streams for asynchronous operations
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
#include <cuda_runtime.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUDAAsynchronous.h"

#include "CUDAMemory.h"
#include "CUDAGraph.h"

/////////////////////////////////////////////////////////////////////////////////
// CUDAEvent implementation
CUDAEvent::CUDAEvent()
{
	m_hEvent = NULL;
}
CUDAEvent::~CUDAEvent()
{
	if ( m_hEvent != NULL )
		Destroy();
}

Void CUDAEvent::Create( UInt iEventFlags )
{
	DebugAssert( m_hEvent == NULL );
	
	cudaEvent_t hCUDAEvent = NULL;
	
	cudaError_t iError = cudaEventCreateWithFlags( &hCUDAEvent, iEventFlags );
	DebugAssert( iError == cudaSuccess && hCUDAEvent != NULL );
	
	m_hEvent = hCUDAEvent;
}
Void CUDAEvent::Destroy()
{
	DebugAssert( m_hEvent != NULL );
	
	cudaEvent_t hCUDAEvent = (cudaEvent_t)m_hEvent;
	
	cudaError_t iError = cudaEventDestroy( hCUDAEvent );
	DebugAssert( iError == cudaSuccess );
	
	m_hEvent = NULL;
}

Bool CUDAEvent::IsCompleted() const
{
	DebugAssert( m_hEvent != NULL );
	
	cudaEvent_t hCUDAEvent = (cudaEvent_t)m_hEvent;
	
	cudaError_t iError = cudaEventQuery( hCUDAEvent );
	
	if ( iError == cudaSuccess )
		return true;
	
	DebugAssert( iError == cudaErrorNotReady );
	return false;
}
Void CUDAEvent::WaitCompletion() const
{
	DebugAssert( m_hEvent != NULL );
	
	cudaEvent_t hCUDAEvent = (cudaEvent_t)m_hEvent;
	
	cudaError_t iError = cudaEventSynchronize( hCUDAEvent );
	DebugAssert( iError == cudaSuccess );
}

Float CUDAEvent::GetElapsedTime( CUDAEvent * pPreviousEvent ) const
{
	DebugAssert( m_hEvent != NULL );
	DebugAssert( pPreviousEvent != NULL && pPreviousEvent->m_hEvent != NULL );

	cudaEvent_t hCUDAEventStart = (cudaEvent_t)( pPreviousEvent->m_hEvent );
	cudaEvent_t hCUDAEventEnd = (cudaEvent_t)m_hEvent;
	Float fElapsedTime = 0.0f;
	
	cudaError_t iError = cudaEventElapsedTime( &fElapsedTime, hCUDAEventStart, hCUDAEventEnd );
	DebugAssert( iError == cudaSuccess );
	
	return fElapsedTime;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDAStream implementation
CUDAStream::CUDAStream()
{
	m_hStream = NULL;
}
CUDAStream::~CUDAStream()
{
	if ( IsCreated() && m_hStream != cudaStreamLegacy && m_hStream != cudaStreamPerThread )
		Destroy();
}

CUDAStream * CUDAStream::GetDefaultLegacyStream()
{
	static CUDAStream s_hDefaultStream;
	if ( s_hDefaultStream.m_hStream == NULL )
		s_hDefaultStream.m_hStream = cudaStreamLegacy;
	return &s_hDefaultStream;
}
CUDAStream * CUDAStream::GetDefaultPerThreadStream()
{
	static CUDAStream s_hDefaultStream;
	if ( s_hDefaultStream.m_hStream == NULL )
		s_hDefaultStream.m_hStream = cudaStreamPerThread;
	return &s_hDefaultStream;
}

Void CUDAStream::Create( Bool bConcurrent, Int iPriority )
{
	DebugAssert( m_hStream == NULL );
	
	cudaStream_t hCUDAStream = NULL;
	UInt iCUDAStreamFlags = cudaStreamDefault;
	if ( bConcurrent )
		iCUDAStreamFlags = cudaStreamNonBlocking;
	
	cudaError_t iError = cudaStreamCreateWithPriority( &hCUDAStream, iCUDAStreamFlags, iPriority );
	DebugAssert( iError == cudaSuccess && hCUDAStream != NULL );
	
	m_hStream = (Void*)hCUDAStream;
}
Void CUDAStream::Destroy()
{
	DebugAssert( m_hStream != NULL && m_hStream != cudaStreamLegacy && m_hStream != cudaStreamPerThread );
	
	cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	
	cudaError_t iError = cudaStreamDestroy( hCUDAStream );
	DebugAssert( iError == cudaSuccess );
	
	m_hStream = NULL;
}

Bool CUDAStream::IsConcurrent() const
{
	DebugAssert( m_hStream != NULL );
	
	cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	UInt iCUDAStreamFlags = 0;
	
	cudaError_t iError = cudaStreamGetFlags( hCUDAStream, &iCUDAStreamFlags );
	DebugAssert( iError == cudaSuccess );
	
	return ( (iCUDAStreamFlags & cudaStreamNonBlocking) != 0 );
}
Int CUDAStream::GetPriority() const
{
	DebugAssert( m_hStream != NULL );
	
	cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	Int iCUDAStreamPriority = 0;
	
	cudaError_t iError = cudaStreamGetPriority( hCUDAStream, &iCUDAStreamPriority );
	DebugAssert( iError == cudaSuccess );
	
	return iCUDAStreamPriority;
}

// Void CUDAStream::GetAccessPolicy() const
// {
	// DebugAssert( m_hStream != NULL );
	
	// cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	// cudaStreamAttrValue hAttributeValue;
	
	// cudaError_t iError = cudaStreamGetAttribute( hCUDAStream, cudaStreamAttributeAccessPolicyWindow, &hAttributeValue );
	// DebugAssert( iError == cudaSuccess );
	
	// hAttributeValue.accessPolicyWindow // cudaAccessPolicyWindow struct
// }
// Void CUDAStream::SetAccessPolicy() const
// {
	// DebugAssert( m_hStream != NULL );
	
	// cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	// cudaStreamAttrValue hAttributeValue;
	
	// hAttributeValue.accessPolicyWindow // cudaAccessPolicyWindow struct
	
	// cudaError_t iError = cudaStreamSetAttribute( hCUDAStream, cudaStreamAttributeAccessPolicyWindow, &hAttributeValue );
	// DebugAssert( iError == cudaSuccess );
// }

// CUDAStreamSyncPolicy CUDAStream::GetSynchronizationPolicy() const
// {
	// DebugAssert( m_hStream != NULL );
	
	// cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	// cudaStreamAttrValue hAttributeValue;
	
	// cudaError_t iError = cudaStreamGetAttribute( hCUDAStream, cudaStreamAttributeSynchronizationPolicy, &hAttributeValue );
	// DebugAssert( iError == cudaSuccess );
	
	// return CUDAStreamSyncPolicyFromCUDA[hAttributeValue.syncPolicy];
// }
// Void CUDAStream::SetSynchronizationPolicy( CUDAStreamSyncPolicy iSynchronizationPolicy ) const
// {
	// DebugAssert( m_hStream != NULL );
	
	// cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	// cudaStreamAttrValue hAttributeValue;
	// hAttributeValue.syncPolicy = (cudaSynchronizationPolicy)( CUDAStreamSyncPolicyToCUDA[iSynchronizationPolicy] );
	
	// cudaError_t iError = cudaStreamSetAttribute( hCUDAStream, cudaStreamAttributeSynchronizationPolicy, &hAttributeValue );
	// DebugAssert( iError == cudaSuccess );
// }

// Void CUDAStream::CopyAttributes( CUDAStream * pSrcStream ) const
// {
	// DebugAssert( m_hStream != NULL );
	// DebugAssert( pSrcStream->m_hStream != NULL );
	
	// cudaStream_t hCUDAStreamDest = (cudaStream_t)m_hStream;
	// cudaStream_t hCUDAStreamSrc = (cudaStream_t)( pSrcStream->m_hStream );
	
	// cudaError_t iError = cudaStreamCopyAttributes( hCUDAStreamDest, hCUDAStreamSrc );
	// DebugAssert( iError == cudaSuccess );
// }

Void CUDAStream::AttachMemory( Void * pSystemMemory, SizeT iSize, UInt iAttachMemoryFlags ) const
{
	DebugAssert( m_hStream != NULL );
	
	cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	
	cudaError_t iError = cudaStreamAttachMemAsync( hCUDAStream, pSystemMemory, iSize, iAttachMemoryFlags );
	DebugAssert( iError == cudaSuccess );
}
Void CUDAStream::AttachMemory( CUDAManagedMemory * pManagedMemory, UInt iAttachMemoryFlags ) const
{
	DebugAssert( m_hStream != NULL );
	DebugAssert( pManagedMemory->IsAllocated() );
	
	cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	
	cudaError_t iError = cudaStreamAttachMemAsync( hCUDAStream, pManagedMemory->GetPointer(), pManagedMemory->GetSize(), iAttachMemoryFlags );
	DebugAssert( iError == cudaSuccess );
}

Bool CUDAStream::IsCompleted() const
{
	DebugAssert( m_hStream != NULL );

	cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;

	cudaError_t iError = cudaStreamQuery( hCUDAStream );

	if ( iError == cudaSuccess )
		return true;
	
	DebugAssert( iError == cudaErrorNotReady );
	return false;
}
Void CUDAStream::WaitCompletion() const
{
	DebugAssert( m_hStream != NULL );
	
	cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	
	cudaError_t iError = cudaStreamSynchronize( hCUDAStream );
	DebugAssert( iError == cudaSuccess );
}

Void CUDAStream::RecordEvent( CUDAEvent * pEvent, Bool bExternalEvent ) const
{
	DebugAssert( m_hStream != NULL );
	DebugAssert( pEvent != NULL && pEvent->m_hEvent != NULL );
	
	cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	cudaEvent_t hCUDAEvent = (cudaEvent_t)( pEvent->m_hEvent );
	
	UInt iCUDAEventRecordFlags = cudaEventRecordDefault;
	if ( bExternalEvent )
		iCUDAEventRecordFlags = cudaEventRecordExternal;
	
	cudaError_t iError = cudaEventRecordWithFlags( hCUDAEvent, hCUDAStream, iCUDAEventRecordFlags );
	DebugAssert( iError == cudaSuccess );
}

Void CUDAStream::RequireEvent( CUDAEvent * pEvent, Bool bExternalEvent ) const
{
	DebugAssert( m_hStream != NULL );
	DebugAssert( pEvent != NULL && pEvent->m_hEvent != NULL );
	
	cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	cudaEvent_t hCUDAEvent = (cudaEvent_t)( pEvent->m_hEvent );
	
	UInt iCUDAEventWaitFlags = cudaEventWaitDefault;
	if ( bExternalEvent )
		iCUDAEventWaitFlags = cudaEventWaitExternal;
	
	cudaError_t iError = cudaStreamWaitEvent( hCUDAStream, hCUDAEvent, iCUDAEventWaitFlags );
	DebugAssert( iError == cudaSuccess );
}

Void CUDAStream::CaptureBegin( CUDAStreamCaptureMode iMode ) const
{
	DebugAssert( m_hStream != NULL );
	
	cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	cudaStreamCaptureMode iCUDAMode = (cudaStreamCaptureMode)( CUDAStreamCaptureModeToCUDA[iMode] );
	
	cudaError_t iError = cudaStreamBeginCapture( hCUDAStream, iCUDAMode );
	DebugAssert( iError == cudaSuccess );
}
Void CUDAStream::CaptureEnd( CUDAGraph * outGraph ) const
{
	DebugAssert( m_hStream != NULL );
	DebugAssert( !(outGraph->IsCreated()) );
	
	cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	cudaGraph_t hCUDAGraph = NULL;
	
	cudaError_t iError = cudaStreamEndCapture( hCUDAStream, &hCUDAGraph );
	DebugAssert( iError == cudaSuccess && hCUDAGraph != NULL );
	
	outGraph->m_hGraph = hCUDAGraph;
}

Void CUDAStream::GetCaptureInfo( CUDAStreamCaptureStatus * outStatus, UInt64 * outCaptureID ) const
{
	DebugAssert( m_hStream != NULL );
	
	cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	cudaStreamCaptureStatus iCUDAStatus;
	
	cudaError_t iError = cudaStreamGetCaptureInfo( hCUDAStream, &iCUDAStatus, outCaptureID );
	DebugAssert( iError == cudaSuccess );
	
	*outStatus = CUDAStreamCaptureStatusFromCUDA[iCUDAStatus];
}

Void CUDAStream::SwapThreadCaptureMode( CUDAStreamCaptureMode * pMode )
{
	cudaStreamCaptureMode iCUDAStreamCaptureMode = (cudaStreamCaptureMode)( CUDAStreamCaptureModeToCUDA[*pMode] );
	
	cudaError_t iError = cudaThreadExchangeStreamCaptureMode( &iCUDAStreamCaptureMode );
	DebugAssert( iError == cudaSuccess );
	
	*pMode = CUDAStreamCaptureModeFromCUDA[iCUDAStreamCaptureMode];
}

Void CUDAStream::PushHostFunction( CUDAHostFunction pfHostFunction, Void * pUserData ) const
{
	DebugAssert( m_hStream != NULL );
	
	cudaStream_t hCUDAStream = (cudaStream_t)m_hStream;
	
	cudaError_t iError = cudaLaunchHostFunc( hCUDAStream, pfHostFunction, pUserData );
	DebugAssert( iError == cudaSuccess );
}


