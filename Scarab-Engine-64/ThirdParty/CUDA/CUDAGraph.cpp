/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDAGraph.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : CUDA Graphs management (deferred execution)
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
#include "CUDAGraph.h"

/////////////////////////////////////////////////////////////////////////////////
// CUDANode implementation
CUDANode::CUDANode()
{
	m_pOwnerGraph = NULL;
	m_hNode = NULL;
}
CUDANode::~CUDANode()
{
	if ( IsCreated() )
		Destroy();
}

Void CUDANode::Destroy()
{
	DebugAssert( m_hNode != NULL );
	
	cudaGraphNode_t hCUDAGraphNode = (cudaGraphNode_t)m_hNode;
	
	cudaError_t iError = cudaGraphDestroyNode( hCUDAGraphNode );
	DebugAssert( iError == cudaSuccess );
	
	m_pOwnerGraph = NULL;
	m_hNode = NULL;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeEmpty implementation
CUDANodeEmpty::CUDANodeEmpty():
	CUDANode()
{
	// nothing to do
}
CUDANodeEmpty::~CUDANodeEmpty()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeMemSet implementation
CUDANodeMemSet::CUDANodeMemSet():
	CUDANode()
{
	m_pDest = NULL;
	m_hDestPos.iX = 0;
	m_hDestPos.iY = 0;
	m_hDestPos.iZ = 0;
	m_hSetRegion.iWidth = 0;
	m_hSetRegion.iHeight = 0;
	m_hSetRegion.iDepth = 0;
	m_iValue = 0;
}
CUDANodeMemSet::~CUDANodeMemSet()
{
	// nothing to do
}

Void CUDANodeMemSet::SetMemSet( CUDAMemory * pDest, const CUDAMemoryPosition & hDestPos, const CUDAMemoryRegion & hSetRegion, UInt iValue,
								CUDAGraphInstance * pInstance )
{
	DebugAssert( m_hNode != NULL );
	DebugAssert( pDest->IsAllocated() );
	DebugAssert( pDest->IsValidRegion(hDestPos, hSetRegion) );
	
	cudaGraphNode_t hCUDAGraphNode = (cudaGraphNode_t)m_hNode;
	
	cudaMemsetParams hCUDAParams;
	hCUDAParams.dst = pDest->GetPointer( hDestPos );
	hCUDAParams.elementSize = 1;
	if ( pDest->m_iShape >= CUDA_MEMORY_SHAPE_1D )
		hCUDAParams.elementSize = pDest->m_iStride;
	hCUDAParams.width = hSetRegion.iWidth;
	hCUDAParams.height = hSetRegion.iHeight;
	hCUDAParams.pitch = hCUDAParams.elementSize * hCUDAParams.width;
	if ( pDest->m_iShape >= CUDA_MEMORY_SHAPE_2D )
		hCUDAParams.pitch = pDest->m_iPitch;
	hCUDAParams.value = iValue;
	
	if ( pInstance != NULL ) {
		DebugAssert( pInstance->m_hGraphInstance != NULL );
		
		cudaGraphExec_t hCUDAGraphInstance = (cudaGraphExec_t)( pInstance->m_hGraphInstance );
		
		cudaError_t iError = cudaGraphExecMemsetNodeSetParams( hCUDAGraphInstance, hCUDAGraphNode, &hCUDAParams );
		DebugAssert( iError == cudaSuccess );
	} else {
		cudaError_t iError = cudaGraphMemsetNodeSetParams( hCUDAGraphNode, &hCUDAParams );
		DebugAssert( iError == cudaSuccess );
		
		m_pDest = pDest;
		m_hDestPos = hDestPos;
		m_hSetRegion = hSetRegion;
		m_iValue = iValue;
	}
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeMemCopy implementation
CUDANodeMemCopy::CUDANodeMemCopy():
	CUDANode()
{
	m_pDest = NULL;
	m_pSrc = NULL;
	m_bIsFlatCopy = true;
	m_hCopyParams.hFlat.iSize = 0;
}
CUDANodeMemCopy::~CUDANodeMemCopy()
{
	// nothing to do
}

Void CUDANodeMemCopy::SetMemCopy( CUDAMemory * pDest, const CUDAMemory * pSrc, SizeT iSize,
								  CUDAGraphInstance * pInstance )
{
	DebugAssert( m_hNode != NULL );
	DebugAssert( pDest->IsAllocated() && pSrc->IsAllocated() );
	DebugAssert( iSize <= pDest->GetSize() );
	DebugAssert( iSize <= pSrc->GetSize() );
	
	cudaGraphNode_t hCUDAGraphNode = (cudaGraphNode_t)m_hNode;
	
	// Check iSize is valid
	DebugAssert( iSize <= pDest->m_iSize && iSize <= pSrc->m_iSize );
	
	// Get Transfer Kind
	cudaMemcpyKind iKind = (cudaMemcpyKind)( pDest->_GetMemCopyKind(pSrc) );
	
	// Setup Copy
	if ( pInstance != NULL ) {
		DebugAssert( pInstance->m_hGraphInstance != NULL );
		
		cudaGraphExec_t hCUDAGraphInstance = (cudaGraphExec_t)( pInstance->m_hGraphInstance );
		
		cudaError_t iError = cudaGraphExecMemcpyNodeSetParams1D( hCUDAGraphInstance, hCUDAGraphNode, pDest->m_pMemory, pSrc->m_pMemory, iSize, iKind );
		DebugAssert( iError == cudaSuccess );
	} else {
		cudaError_t iError = cudaGraphMemcpyNodeSetParams1D( hCUDAGraphNode, pDest->m_pMemory, pSrc->m_pMemory, iSize, iKind );
		DebugAssert( iError == cudaSuccess );
		
		m_pDest = pDest;
		m_pSrc = pSrc;
		m_bIsFlatCopy = true;
		m_hCopyParams.hFlat.iSize = iSize;
	}
}
Void CUDANodeMemCopy::SetMemCopy( CUDAMemory * pDest, const CUDAMemoryPosition & hDestPos,
								  const CUDAMemory * pSrc, const CUDAMemoryPosition & hSrcPos,
								  const CUDAMemoryRegion & hCopyRegion,
								  CUDAGraphInstance * pInstance )
{
	DebugAssert( m_hNode != NULL );
	DebugAssert( pDest->IsAllocated() && pSrc->IsAllocated() );
	DebugAssert( pDest->IsValidRegion(hDestPos, hCopyRegion) );
	DebugAssert( pSrc->IsValidRegion(hSrcPos, hCopyRegion) );
	
	cudaGraphNode_t hCUDAGraphNode = (cudaGraphNode_t)m_hNode;
	
	// Get Copy parameters
	cudaMemcpy3DParms hCUDAParams;
	pDest->_ConvertCopyParams( &hCUDAParams, hDestPos, pSrc, hSrcPos, hCopyRegion );
	
	// Setup Copy
	if ( pInstance != NULL ) {
		DebugAssert( pInstance->m_hGraphInstance != NULL );
		
		cudaGraphExec_t hCUDAGraphInstance = (cudaGraphExec_t)( pInstance->m_hGraphInstance );
		
		cudaError_t iError = cudaGraphExecMemcpyNodeSetParams( hCUDAGraphInstance, hCUDAGraphNode, &hCUDAParams );
		DebugAssert( iError == cudaSuccess );
	} else {
		cudaError_t iError = cudaGraphMemcpyNodeSetParams( hCUDAGraphNode, &hCUDAParams );
		DebugAssert( iError == cudaSuccess );
		
		m_pDest = pDest;
		m_pSrc = pSrc;
		m_bIsFlatCopy = false;
		m_hCopyParams.hShaped.hDestPos = hDestPos;
		m_hCopyParams.hShaped.hSrcPos = hSrcPos;
		m_hCopyParams.hShaped.hCopyRegion = hCopyRegion;
	}
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeKernel implementation
CUDANodeKernel::CUDANodeKernel():
	CUDANode(), m_hGridSize(), m_hBlockSize()
{
	m_pKernel = NULL;
	m_bCooperative = false;
}
CUDANodeKernel::~CUDANodeKernel()
{
	// nothing to do
}

Void CUDANodeKernel::SetKernel( CUDAKernel * pKernel, const CUDAKernelDimension & hGridSize, const CUDAKernelDimension & hBlockSize, Bool bCooperative, CUDAGraphInstance * pInstance )
{
	DebugAssert( m_hNode != NULL );
	
	cudaGraphNode_t hCUDAGraphNode = (cudaGraphNode_t)m_hNode;
	
	cudaKernelNodeParams hCUDAParams;
	hCUDAParams.func = pKernel->m_pKernelFunction;
	hCUDAParams.gridDim.x = hGridSize.iX;
	hCUDAParams.gridDim.y = hGridSize.iY;
	hCUDAParams.gridDim.z = hGridSize.iZ;
	hCUDAParams.blockDim.x = hBlockSize.iX;
	hCUDAParams.blockDim.y = hBlockSize.iY;
	hCUDAParams.blockDim.z = hBlockSize.iZ;
	hCUDAParams.sharedMemBytes = pKernel->m_iSharedMemorySize;
	hCUDAParams.kernelParams = pKernel->m_arrKernelParameters;
	hCUDAParams.extra = NULL;
	
	if ( pInstance != NULL ) {
		DebugAssert( pInstance->m_hGraphInstance != NULL );
		
		cudaGraphExec_t hCUDAGraphInstance = (cudaGraphExec_t)( pInstance->m_hGraphInstance );
		
		cudaError_t iError = cudaGraphExecKernelNodeSetParams( hCUDAGraphInstance, hCUDAGraphNode, &hCUDAParams );
		DebugAssert( iError == cudaSuccess );
	} else {
		cudaError_t iError = cudaGraphKernelNodeSetParams( hCUDAGraphNode, &hCUDAParams );
		DebugAssert( iError == cudaSuccess );
		
		cudaKernelNodeAttrValue hValue;
		hValue.cooperative = bCooperative ? 1 : 0;
		
		iError = cudaGraphKernelNodeSetAttribute( hCUDAGraphNode, cudaKernelNodeAttributeCooperative, &hValue );
		DebugAssert( iError == cudaSuccess );
		
		m_pKernel = pKernel;
		m_hGridSize = hGridSize;
		m_hBlockSize = hBlockSize;
		m_bCooperative = bCooperative;
	}
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeHostFunction implementation
CUDANodeHostFunction::CUDANodeHostFunction():
	CUDANode()
{
	m_pfHostFunction = NULL;
	m_pUserData = NULL;
}
CUDANodeHostFunction::~CUDANodeHostFunction()
{
	// nothing to do
}

Void CUDANodeHostFunction::SetHostFunction( CUDAHostFunction pfHostFunction, Void * pUserData, CUDAGraphInstance * pInstance )
{
	DebugAssert( m_hNode != NULL );
	
	cudaGraphNode_t hCUDAGraphNode = (cudaGraphNode_t)m_hNode;
	
	cudaHostNodeParams hCUDAParams;
	hCUDAParams.fn = pfHostFunction;
	hCUDAParams.userData = pUserData;
	
	if ( pInstance != NULL ) {
		DebugAssert( pInstance->m_hGraphInstance != NULL );
		
		cudaGraphExec_t hCUDAGraphInstance = (cudaGraphExec_t)( pInstance->m_hGraphInstance );
		
		cudaError_t iError = cudaGraphExecHostNodeSetParams( hCUDAGraphInstance, hCUDAGraphNode, &hCUDAParams );
		DebugAssert( iError == cudaSuccess );
	} else {
		cudaError_t iError = cudaGraphHostNodeSetParams( hCUDAGraphNode, &hCUDAParams );
		DebugAssert( iError == cudaSuccess );
		
		m_pfHostFunction = pfHostFunction;
		m_pUserData = pUserData;
	}
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeEventRecord implementation
CUDANodeEventRecord::CUDANodeEventRecord():
	CUDANode()
{
	m_pEvent = NULL;
}
CUDANodeEventRecord::~CUDANodeEventRecord()
{
	// nothing to do
}

Void CUDANodeEventRecord::SetEvent( CUDAEvent * pEvent, CUDAGraphInstance * pInstance )
{
	DebugAssert( m_hNode != NULL );
	DebugAssert( pEvent->IsCreated() );
	
	cudaGraphNode_t hCUDAGraphNode = (cudaGraphNode_t)m_hNode;
	cudaEvent_t hCUDAEvent = (cudaEvent_t)( pEvent->m_hEvent );
	
	if ( pInstance != NULL ) {
		DebugAssert( pInstance->m_hGraphInstance != NULL );
		
		cudaGraphExec_t hCUDAGraphInstance = (cudaGraphExec_t)( pInstance->m_hGraphInstance );
		
		cudaError_t iError = cudaGraphExecEventRecordNodeSetEvent( hCUDAGraphInstance, hCUDAGraphNode, hCUDAEvent );
		DebugAssert( iError == cudaSuccess );
	} else {
		cudaError_t iError = cudaGraphEventRecordNodeSetEvent( hCUDAGraphNode, hCUDAEvent );
		DebugAssert( iError == cudaSuccess );
		
		m_pEvent = pEvent;
	}
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeEventRequire implementation
CUDANodeEventRequire::CUDANodeEventRequire():
	CUDANode()
{
	m_pEvent = NULL;
}
CUDANodeEventRequire::~CUDANodeEventRequire()
{
	// nothing to do
}

Void CUDANodeEventRequire::SetEvent( CUDAEvent * pEvent, CUDAGraphInstance * pInstance )
{
	DebugAssert( m_hNode != NULL );
	DebugAssert( pEvent->IsCreated() );
	
	cudaGraphNode_t hCUDAGraphNode = (cudaGraphNode_t)m_hNode;
	cudaEvent_t hCUDAEvent = (cudaEvent_t)( pEvent->m_hEvent );
	
	if ( pInstance != NULL ) {
		DebugAssert( pInstance->m_hGraphInstance != NULL );
		
		cudaGraphExec_t hCUDAGraphInstance = (cudaGraphExec_t)( pInstance->m_hGraphInstance );
		
		cudaError_t iError = cudaGraphExecEventWaitNodeSetEvent( hCUDAGraphInstance, hCUDAGraphNode, hCUDAEvent );
		DebugAssert( iError == cudaSuccess );
	} else {
		cudaError_t iError = cudaGraphEventWaitNodeSetEvent( hCUDAGraphNode, hCUDAEvent );
		DebugAssert( iError == cudaSuccess );
		
		m_pEvent = pEvent;
	}
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeChildGraph implementation
CUDANodeChildGraph::CUDANodeChildGraph():
	CUDANode()
{
	m_pChildGraph = NULL;
}
CUDANodeChildGraph::~CUDANodeChildGraph()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// CUDAGraph implementation
CUDAGraph::CUDAGraph()
{
	m_hGraph = NULL;
}
CUDAGraph::~CUDAGraph()
{
	if ( IsCreated() )
		Destroy();
}

Void CUDAGraph::Create()
{
	DebugAssert( m_hGraph == NULL );
	
	cudaGraph_t hCUDAGraph = NULL;
	
	cudaError_t iError = cudaGraphCreate( &hCUDAGraph, 0 );
	DebugAssert( iError == cudaSuccess && hCUDAGraph != NULL );
	
	m_hGraph = hCUDAGraph;
}
Void CUDAGraph::Destroy()
{
	DebugAssert( m_hGraph != NULL );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;
	
	cudaError_t iError = cudaGraphDestroy( hCUDAGraph );
	DebugAssert( iError == cudaSuccess );
	
	m_hGraph = NULL;
}

Void CUDAGraph::Clone( CUDAGraph * outGraph ) const
{
	DebugAssert( m_hGraph != NULL );
	DebugAssert( outGraph->m_hGraph == NULL );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;
	
	cudaGraph_t hCUDAClonedGraph = NULL;
	
	cudaError_t iError = cudaGraphClone( &hCUDAClonedGraph, hCUDAGraph );
	DebugAssert( iError == cudaSuccess && hCUDAClonedGraph != NULL );
	
	outGraph->m_hGraph = hCUDAClonedGraph;
}

SizeT CUDAGraph::GetNodeCount() const
{
	DebugAssert( m_hGraph != NULL );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;
	
	SizeT iNodeCount = 0;
	
	cudaError_t iError = cudaGraphGetNodes( hCUDAGraph, NULL, &iNodeCount );
	DebugAssert( iError == cudaSuccess );
	
	return iNodeCount;
}

Void CUDAGraph::CreateNodeEmpty( CUDANodeEmpty * outNode )
{
	DebugAssert( m_hGraph != NULL );
	DebugAssert( outNode->m_hNode == NULL );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;

	cudaGraphNode_t hCUDAGraphNode = NULL;
	
	cudaError_t iError = cudaGraphAddEmptyNode( &hCUDAGraphNode, hCUDAGraph, NULL, 0 );
	DebugAssert( iError == cudaSuccess && hCUDAGraphNode != NULL );
	
	outNode->m_pOwnerGraph = this;
	outNode->m_hNode = hCUDAGraphNode;
}

Void CUDAGraph::CreateNodeMemSet( CUDANodeMemSet * outNode,
								  CUDAMemory * pDest, const CUDAMemoryPosition & hDestPos, const CUDAMemoryRegion & hSetRegion, UInt iValue )
{
	DebugAssert( m_hGraph != NULL );
	DebugAssert( outNode->m_hNode == NULL );
	DebugAssert( pDest->IsAllocated() );
	DebugAssert( pDest->IsValidRegion(hDestPos, hSetRegion) );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;

	cudaMemsetParams hCUDAParams;
	hCUDAParams.dst = pDest->GetPointer( hDestPos );
	hCUDAParams.elementSize = 1;
	if ( pDest->m_iShape >= CUDA_MEMORY_SHAPE_1D )
		hCUDAParams.elementSize = pDest->m_iStride;
	hCUDAParams.width = hSetRegion.iWidth;
	hCUDAParams.height = hSetRegion.iHeight;
	hCUDAParams.pitch = hCUDAParams.elementSize * hCUDAParams.width;
	if ( pDest->m_iShape >= CUDA_MEMORY_SHAPE_2D )
		hCUDAParams.pitch = pDest->m_iPitch;
	hCUDAParams.value = iValue;
	
	cudaGraphNode_t hCUDAGraphNode = NULL;
	
	cudaError_t iError = cudaGraphAddMemsetNode( &hCUDAGraphNode, hCUDAGraph, NULL, 0, &hCUDAParams );
	DebugAssert( iError == cudaSuccess && hCUDAGraphNode != NULL );
	
	outNode->m_pOwnerGraph = this;
	outNode->m_hNode = hCUDAGraphNode;
	outNode->m_pDest = pDest;
	outNode->m_hDestPos = hDestPos;
	outNode->m_hSetRegion = hSetRegion;
	outNode->m_iValue = iValue;
}
Void CUDAGraph::CreateNodeMemCopy( CUDANodeMemCopy * outNode,
								   CUDAMemory * pDest, const CUDAMemory * pSrc, SizeT iSize )
{
	DebugAssert( m_hGraph != NULL );
	DebugAssert( outNode->m_hNode == NULL );
	DebugAssert( pDest->IsAllocated() && pSrc->IsAllocated() );
	DebugAssert( iSize <= pDest->GetSize() );
	DebugAssert( iSize <= pSrc->GetSize() );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;

	// Check iSize is valid
	DebugAssert( iSize <= pDest->m_iSize && iSize <= pSrc->m_iSize );
	
	// Get Transfer Kind
	cudaMemcpyKind iKind = (cudaMemcpyKind)( pDest->_GetMemCopyKind(pSrc) );
	
	// Setup Copy
	cudaGraphNode_t hCUDAGraphNode = NULL;
	
	cudaError_t iError = cudaGraphAddMemcpyNode1D( &hCUDAGraphNode, hCUDAGraph, NULL, 0, pDest->m_pMemory, pSrc->m_pMemory, iSize, iKind );
	DebugAssert( iError == cudaSuccess && hCUDAGraphNode != NULL );
	
	outNode->m_pOwnerGraph = this;
	outNode->m_hNode = hCUDAGraphNode;
	outNode->m_pDest = pDest;
	outNode->m_pSrc = pSrc;
	outNode->m_bIsFlatCopy = true;
	outNode->m_hCopyParams.hFlat.iSize = iSize;
}
Void CUDAGraph::CreateNodeMemCopy( CUDANodeMemCopy * outNode,
								   CUDAMemory * pDest, const CUDAMemoryPosition & hDestPos,
								   const CUDAMemory * pSrc, const CUDAMemoryPosition & hSrcPos,
								   const CUDAMemoryRegion & hCopyRegion )
{
	DebugAssert( m_hGraph != NULL );
	DebugAssert( outNode->m_hNode == NULL );
	DebugAssert( pDest->IsAllocated() && pSrc->IsAllocated() );
	DebugAssert( pDest->IsValidRegion(hDestPos, hCopyRegion) );
	DebugAssert( pSrc->IsValidRegion(hSrcPos, hCopyRegion) );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;

	// Get Copy parameters
	cudaMemcpy3DParms hCUDAParams;
	pDest->_ConvertCopyParams( &hCUDAParams, hDestPos, pSrc, hSrcPos, hCopyRegion );
	
	// Setup Copy
	cudaGraphNode_t hCUDAGraphNode = NULL;
	
	cudaError_t iError = cudaGraphAddMemcpyNode( &hCUDAGraphNode, hCUDAGraph, NULL, 0, &hCUDAParams );
	DebugAssert( iError == cudaSuccess && hCUDAGraphNode != NULL );
	
	outNode->m_pOwnerGraph = this;
	outNode->m_hNode = hCUDAGraphNode;
	outNode->m_pDest = pDest;
	outNode->m_pSrc = pSrc;
	outNode->m_bIsFlatCopy = false;
	outNode->m_hCopyParams.hShaped.hDestPos = hDestPos;
	outNode->m_hCopyParams.hShaped.hSrcPos = hSrcPos;
	outNode->m_hCopyParams.hShaped.hCopyRegion = hCopyRegion;
}

Void CUDAGraph::CreateNodeKernel( CUDANodeKernel * outNode, CUDAKernel * pKernel, const CUDAKernelDimension & hGridSize, const CUDAKernelDimension & hBlockSize, Bool bCooperative )
{
	DebugAssert( m_hGraph != NULL );
	DebugAssert( outNode->m_hNode == NULL );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;

	cudaKernelNodeParams hCUDAParams;
	hCUDAParams.func = pKernel->m_pKernelFunction;
	hCUDAParams.gridDim.x = hGridSize.iX;
	hCUDAParams.gridDim.y = hGridSize.iY;
	hCUDAParams.gridDim.z = hGridSize.iZ;
	hCUDAParams.blockDim.x = hBlockSize.iX;
	hCUDAParams.blockDim.y = hBlockSize.iY;
	hCUDAParams.blockDim.z = hBlockSize.iZ;
	hCUDAParams.sharedMemBytes = pKernel->m_iSharedMemorySize;
	hCUDAParams.kernelParams = pKernel->m_arrKernelParameters;
	hCUDAParams.extra = NULL;
	
	cudaGraphNode_t hCUDAGraphNode = NULL;
	
	cudaError_t iError = cudaGraphAddKernelNode( &hCUDAGraphNode, hCUDAGraph, NULL, 0, &hCUDAParams );
	DebugAssert( iError == cudaSuccess && hCUDAGraphNode != NULL );
	
	cudaKernelNodeAttrValue hValue;
	hValue.cooperative = bCooperative ? 1 : 0;
	
	iError = cudaGraphKernelNodeSetAttribute( hCUDAGraphNode, cudaKernelNodeAttributeCooperative, &hValue );
	DebugAssert( iError == cudaSuccess );
	
	outNode->m_pOwnerGraph = this;
	outNode->m_hNode = hCUDAGraphNode;
	outNode->m_pKernel = pKernel;
	outNode->m_hGridSize = hGridSize;
	outNode->m_hBlockSize = hBlockSize;
	outNode->m_bCooperative = bCooperative;
}
Void CUDAGraph::CreateNodeHostFunction( CUDANodeHostFunction * outNode, CUDAHostFunction pfHostFunction, Void * pUserData )
{
	DebugAssert( m_hGraph != NULL );
	DebugAssert( outNode->m_hNode == NULL );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;
	
	cudaHostNodeParams hCUDAParams;
	hCUDAParams.fn = pfHostFunction;
	hCUDAParams.userData = pUserData;
	
	cudaGraphNode_t hCUDAGraphNode = NULL;
	
	cudaError_t iError = cudaGraphAddHostNode( &hCUDAGraphNode, hCUDAGraph, NULL, 0, &hCUDAParams );
	DebugAssert( iError == cudaSuccess && hCUDAGraphNode != NULL );
	
	outNode->m_pOwnerGraph = this;
	outNode->m_hNode = hCUDAGraphNode;
	outNode->m_pfHostFunction = pfHostFunction;
	outNode->m_pUserData = pUserData;
}

Void CUDAGraph::CreateNodeEventRecord( CUDANodeEventRecord * outNode, CUDAEvent * pEvent )
{
	DebugAssert( m_hGraph != NULL );
	DebugAssert( outNode->m_hNode == NULL );
	DebugAssert( pEvent->IsCreated() );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;
	cudaEvent_t hCUDAEvent = (cudaEvent_t)( pEvent->m_hEvent );

	cudaGraphNode_t hCUDAGraphNode = NULL;
	
	cudaError_t iError = cudaGraphAddEventRecordNode( &hCUDAGraphNode, hCUDAGraph, NULL, 0, hCUDAEvent );
	DebugAssert( iError == cudaSuccess && hCUDAGraphNode != NULL );
	
	outNode->m_pOwnerGraph = this;
	outNode->m_hNode = hCUDAGraphNode;
	outNode->m_pEvent = pEvent;
}
Void CUDAGraph::CreateNodeEventRequire( CUDANodeEventRequire * outNode, CUDAEvent * pEvent )
{
	DebugAssert( m_hGraph != NULL );
	DebugAssert( outNode->m_hNode == NULL );
	DebugAssert( pEvent->IsCreated() );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;
	cudaEvent_t hCUDAEvent = (cudaEvent_t)( pEvent->m_hEvent );
	
	cudaGraphNode_t hCUDAGraphNode = NULL;
	
	cudaError_t iError = cudaGraphAddEventWaitNode( &hCUDAGraphNode, hCUDAGraph, NULL, 0, hCUDAEvent );
	DebugAssert( iError == cudaSuccess && hCUDAGraphNode != NULL );
	
	outNode->m_pOwnerGraph = this;
	outNode->m_hNode = hCUDAGraphNode;
	outNode->m_pEvent = pEvent;
}

Void CUDAGraph::CreateNodeChildGraph( CUDANodeChildGraph * outNode, CUDAGraph * pChildGraph )
{
	DebugAssert( m_hGraph != NULL );
	DebugAssert( outNode->m_hNode == NULL );
	DebugAssert( pChildGraph->m_hGraph != NULL );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;
	cudaGraph_t hCUDAChildGraph = (cudaGraph_t)( pChildGraph->m_hGraph );

	cudaGraphNode_t hCUDAGraphNode = NULL;
	
	cudaError_t iError = cudaGraphAddChildGraphNode( &hCUDAGraphNode, hCUDAGraph, NULL, 0, hCUDAChildGraph );
	DebugAssert( iError == cudaSuccess && hCUDAGraphNode != NULL );
	
	outNode->m_pOwnerGraph = this;
	outNode->m_hNode = hCUDAGraphNode;
	outNode->m_pChildGraph = pChildGraph;
}

SizeT CUDAGraph::GetDependencyCount() const
{
	DebugAssert( m_hGraph != NULL );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;
	
	SizeT iDependencyCount = 0;
	
	cudaError_t iError = cudaGraphGetEdges( hCUDAGraph, NULL, NULL, &iDependencyCount );
	DebugAssert( iError == cudaSuccess );
	
	return iDependencyCount;
}

Void CUDAGraph::AddDependency( const CUDANode * pRequiredNode, const CUDANode * pDependentNode ) const
{
	DebugAssert( m_hGraph != NULL );
	DebugAssert( pRequiredNode->m_hNode != NULL );
	DebugAssert( pDependentNode->m_hNode != NULL );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;
	const cudaGraphNode_t * arrFrom = (const cudaGraphNode_t *)( &(pRequiredNode->m_hNode) );
	const cudaGraphNode_t * arrTo = (const cudaGraphNode_t *)( &(pDependentNode->m_hNode) );
	
	cudaError_t iError = cudaGraphAddDependencies( hCUDAGraph, arrFrom, arrTo, 1 );
	DebugAssert( iError == cudaSuccess );
}
Void CUDAGraph::RemoveDependency( const CUDANode * pRequiredNode, const CUDANode * pDependentNode ) const
{
	DebugAssert( m_hGraph != NULL );
	DebugAssert( pRequiredNode->m_hNode != NULL );
	DebugAssert( pDependentNode->m_hNode != NULL );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;
	const cudaGraphNode_t * arrFrom = (const cudaGraphNode_t *)( &(pRequiredNode->m_hNode) );
	const cudaGraphNode_t * arrTo = (const cudaGraphNode_t *)( &(pDependentNode->m_hNode) );
	
	cudaError_t iError = cudaGraphRemoveDependencies( hCUDAGraph, arrFrom, arrTo, 1 );
	DebugAssert( iError == cudaSuccess );
}

Void CUDAGraph::CreateInstance( CUDAGraphInstance * outGraphInstance ) const
{
	DebugAssert( m_hGraph != NULL );
	DebugAssert( outGraphInstance->m_hGraphInstance == NULL );
	
	cudaGraph_t hCUDAGraph = (cudaGraph_t)m_hGraph;
	
	cudaGraphExec_t hCUDAGraphInstance = NULL;
	cudaGraphNode_t hErrorNode;
	Char arrLogBuffer[1024];
	
	cudaError_t iError = cudaGraphInstantiate( &hCUDAGraphInstance, hCUDAGraph, &hErrorNode, (char*)arrLogBuffer, 1024 );
	DebugAssert( iError == cudaSuccess && hCUDAGraphInstance != NULL );
	
	outGraphInstance->m_hGraphInstance = hCUDAGraphInstance;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDAGraphInstance implementation
CUDAGraphInstance::CUDAGraphInstance()
{
	m_hGraphInstance = NULL;
}
CUDAGraphInstance::~CUDAGraphInstance()
{
	if ( IsCreated() )
		Destroy();
}

Void CUDAGraphInstance::Destroy()
{
	DebugAssert( m_hGraphInstance != NULL );
	
	cudaGraphExec_t hCUDAGraphInstance = (cudaGraphExec_t)m_hGraphInstance;
	
	cudaError_t iError = cudaGraphExecDestroy( hCUDAGraphInstance );
	DebugAssert( iError == cudaSuccess );
	
	m_hGraphInstance = NULL;
}

Bool CUDAGraphInstance::Update( CUDAGraph * pGraph ) const
{
	DebugAssert( m_hGraphInstance != NULL );
	DebugAssert( pGraph->m_hGraph != NULL );
	
	cudaGraphExec_t hCUDAGraphInstance = (cudaGraphExec_t)m_hGraphInstance;
	cudaGraph_t hCUDAGraph = (cudaGraph_t)( pGraph->m_hGraph );
	
	cudaGraphNode_t hErrorNode;
	cudaGraphExecUpdateResult iUpdateResult;
	
	cudaError_t iError = cudaGraphExecUpdate( hCUDAGraphInstance, hCUDAGraph, &hErrorNode, &iUpdateResult );
	DebugAssert( iError == cudaSuccess || iError == cudaErrorGraphExecUpdateFailure );
	
	return ( iError == cudaSuccess && iUpdateResult == cudaGraphExecUpdateSuccess );
}
Void CUDAGraphInstance::Update( CUDANodeChildGraph * pNodeChildGraph, CUDAGraph * pChildGraph ) const
{
	DebugAssert( m_hGraphInstance != NULL );
	DebugAssert( pNodeChildGraph->m_hNode != NULL );
	DebugAssert( pChildGraph->m_hGraph != NULL );
	
	cudaGraphExec_t hCUDAGraphInstance = (cudaGraphExec_t)m_hGraphInstance;
	cudaGraphNode_t hCUDAGraphNode = (cudaGraphNode_t)( pNodeChildGraph->m_hNode );
	cudaGraph_t hCUDAGraph = (cudaGraph_t)( pChildGraph->m_hGraph );
	
	cudaError_t iError = cudaGraphExecChildGraphNodeSetParams( hCUDAGraphInstance, hCUDAGraphNode, hCUDAGraph );
	DebugAssert( iError == cudaSuccess );
}

Void CUDAGraphInstance::Upload( CUDAStream * pStream ) const
{
	DebugAssert( m_hGraphInstance != NULL );
	DebugAssert( pStream->IsCreated() );
	
	cudaGraphExec_t hCUDAGraphInstance = (cudaGraphExec_t)m_hGraphInstance;
	cudaStream_t hCUDAStream = (cudaStream_t)( pStream->m_hStream );
	
	cudaError_t iError = cudaGraphUpload( hCUDAGraphInstance, hCUDAStream );
	DebugAssert( iError == cudaSuccess );
}
Void CUDAGraphInstance::Execute( CUDAStream * pStream ) const
{
	DebugAssert( m_hGraphInstance != NULL );
	DebugAssert( pStream->IsCreated() );
	
	cudaGraphExec_t hCUDAGraphInstance = (cudaGraphExec_t)m_hGraphInstance;
	cudaStream_t hCUDAStream = (cudaStream_t)( pStream->m_hStream );
	
	cudaError_t iError = cudaGraphLaunch( hCUDAGraphInstance, hCUDAStream );
	DebugAssert( iError == cudaSuccess );
}


