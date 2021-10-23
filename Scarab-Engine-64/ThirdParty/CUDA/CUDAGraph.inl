/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDAGraph.inl
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
// CUDANode implementation
inline CUDAGraph * CUDANode::GetOwnerGraph() const {
	return m_pOwnerGraph;
}

inline Bool CUDANode::IsCreated() const {
	return ( m_hNode != NULL );
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeEmpty implementation
inline CUDANodeType CUDANodeEmpty::GetType() const {
	return CUDA_NODE_EMPTY;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeMemSet implementation
inline CUDANodeType CUDANodeMemSet::GetType() const {
	return CUDA_NODE_MEMSET;
}

inline CUDAMemory * CUDANodeMemSet::GetDest() const {
	return m_pDest;
}
inline const CUDAMemoryPosition & CUDANodeMemSet::GetDestPos() const {
	return m_hDestPos;
}
inline const CUDAMemoryRegion & CUDANodeMemSet::GetSetRegion() const {
	return m_hSetRegion;
}
inline UInt CUDANodeMemSet::GetValue() const {
	return m_iValue;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeMemCopy implementation
inline CUDANodeType CUDANodeMemCopy::GetType() const {
	return CUDA_NODE_MEMCOPY;
}

inline CUDAMemory * CUDANodeMemCopy::GetDest() const {
	return m_pDest;
}
inline const CUDAMemory * CUDANodeMemCopy::GetSrc() const {
	return m_pSrc;
}

inline Bool CUDANodeMemCopy::IsFlatCopy() const {
	return m_bIsFlatCopy;
}
inline SizeT CUDANodeMemCopy::GetFlatCopySize() const {
	DebugAssert( m_bIsFlatCopy );
	return m_hCopyParams.hFlat.iSize;
}

inline const CUDAMemoryPosition & CUDANodeMemCopy::GetDestPos() const {
	DebugAssert( !m_bIsFlatCopy );
	return m_hCopyParams.hShaped.hDestPos;
}
inline const CUDAMemoryPosition & CUDANodeMemCopy::GetSrcPos() const {
	DebugAssert( !m_bIsFlatCopy );
	return m_hCopyParams.hShaped.hSrcPos;
}
inline const CUDAMemoryRegion & CUDANodeMemCopy::GetCopyRegion() const {
	DebugAssert( !m_bIsFlatCopy );
	return m_hCopyParams.hShaped.hCopyRegion;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeKernel implementation
inline CUDANodeType CUDANodeKernel::GetType() const {
	return CUDA_NODE_KERNEL;
}

inline CUDAKernel * CUDANodeKernel::GetKernel() const {
	return m_pKernel;
}
inline const CUDAKernelDimension & CUDANodeKernel::GetGridSize() const {
	return m_hGridSize;
}
inline const CUDAKernelDimension & CUDANodeKernel::GetBlockSize() const {
	return m_hBlockSize;
}
inline Bool CUDANodeKernel::IsCooperative() const {
	return m_bCooperative;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeHostFunction implementation
inline CUDANodeType CUDANodeHostFunction::GetType() const {
	return CUDA_NODE_HOSTFUNCTION;
}

inline CUDAHostFunction CUDANodeHostFunction::GetHostFunction() const {
	return m_pfHostFunction;
}
inline Void * CUDANodeHostFunction::GetUserData() const {
	return m_pUserData;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeEventRecord implementation
inline CUDANodeType CUDANodeEventRecord::GetType() const {
	return CUDA_NODE_EVENT_RECORD;
}

inline CUDAEvent * CUDANodeEventRecord::GetEvent() const {
	return m_pEvent;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeEventRequire implementation
inline CUDANodeType CUDANodeEventRequire::GetType() const {
	return CUDA_NODE_EVENT_REQUIRE;
}

inline CUDAEvent * CUDANodeEventRequire::GetEvent() const {
	return m_pEvent;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDANodeChildGraph implementation
inline CUDANodeType CUDANodeChildGraph::GetType() const {
	return CUDA_NODE_CHILDGRAPH;
}

inline CUDAGraph * CUDANodeChildGraph::GetChildGraph() const {
	return m_pChildGraph;
}

/////////////////////////////////////////////////////////////////////////////////
// CUDAGraph implementation
inline Bool CUDAGraph::IsCreated() const {
	return ( m_hGraph != NULL );
}

/////////////////////////////////////////////////////////////////////////////////
// CUDAGraphInstance implementation
inline Bool CUDAGraphInstance::IsCreated() const {
	return ( m_hGraphInstance != NULL );
}



