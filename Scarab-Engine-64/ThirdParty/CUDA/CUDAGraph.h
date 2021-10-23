/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/CUDA/CUDAGraph.h
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
// Header prelude
#ifndef SCARAB_THIRDPARTY_CUDA_CUDAGRAPH_H
#define SCARAB_THIRDPARTY_CUDA_CUDAGRAPH_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "CUDAMemory.h"
#include "CUDAAsynchronous.h"
#include "CUDAKernel.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Graph Node Types
enum CUDANodeType {
	CUDA_NODE_EMPTY = 0,
	CUDA_NODE_MEMSET,
	CUDA_NODE_MEMCOPY,
	CUDA_NODE_KERNEL,
	CUDA_NODE_HOSTFUNCTION,
	CUDA_NODE_EVENT_RECORD,
	CUDA_NODE_EVENT_REQUIRE,
	CUDA_NODE_CHILDGRAPH
};

// Prototypes
class CUDAGraph;
class CUDAGraphInstance;

/////////////////////////////////////////////////////////////////////////////////
// The CUDANode class
class CUDANode
{
public:
	CUDANode();
	virtual ~CUDANode();
	
	virtual CUDANodeType GetType() const = 0;
	
	inline CUDAGraph * GetOwnerGraph() const;
	
	// Deferred Creation / Destruction
	inline Bool IsCreated() const;
	Void Destroy(); // NOT Recursive, but remove dependencies both ways !
	
	//cudaGraphNodeGetDependencies, Requires reverse-lookup tables ... not sure we want this ! Best to stay in "pure" mode !
	//cudaGraphNodeGetDependentNodes, Requires reverse-lookup tables ... not sure we want this ! Best to stay in "pure" mode !
	
protected:
	friend class CUDAGraph;
	friend class CUDAGraphInstance;

	CUDAGraph * m_pOwnerGraph;
	Void * m_hNode;
};

/////////////////////////////////////////////////////////////////////////////////
// The CUDANodeEmpty class
class CUDANodeEmpty : public CUDANode
{
public:
	CUDANodeEmpty();
	virtual ~CUDANodeEmpty();
	
	inline virtual CUDANodeType GetType() const;
	
};

/////////////////////////////////////////////////////////////////////////////////
// The CUDANodeMemSet class
class CUDANodeMemSet : public CUDANode
{
public:
	CUDANodeMemSet();
	virtual ~CUDANodeMemSet();
	
	inline virtual CUDANodeType GetType() const;

	inline CUDAMemory * GetDest() const;
	inline const CUDAMemoryPosition & GetDestPos() const;
	inline const CUDAMemoryRegion & GetSetRegion() const;
	inline UInt GetValue() const;
	
	Void SetMemSet( CUDAMemory * pDest, const CUDAMemoryPosition & hDestPos, const CUDAMemoryRegion & hSetRegion, UInt iValue,
					CUDAGraphInstance * pInstance = NULL );

protected:
	friend class CUDAGraph;

	CUDAMemory * m_pDest;
	CUDAMemoryPosition m_hDestPos;
	CUDAMemoryRegion m_hSetRegion;
	UInt m_iValue;
};

/////////////////////////////////////////////////////////////////////////////////
// The CUDANodeMemCopy class
class CUDANodeMemCopy : public CUDANode
{
public:
	CUDANodeMemCopy();
	virtual ~CUDANodeMemCopy();
	
	inline virtual CUDANodeType GetType() const;
	
	inline CUDAMemory * GetDest() const;
	inline const CUDAMemory * GetSrc() const;
	
	inline Bool IsFlatCopy() const;
	inline SizeT GetFlatCopySize() const;
	
	inline const CUDAMemoryPosition & GetDestPos() const;
	inline const CUDAMemoryPosition & GetSrcPos() const;
	inline const CUDAMemoryRegion & GetCopyRegion() const;

	Void SetMemCopy( CUDAMemory * pDest, const CUDAMemory * pSrc, SizeT iSize,
					 CUDAGraphInstance * pInstance = NULL );
	Void SetMemCopy( CUDAMemory * pDest, const CUDAMemoryPosition & hDestPos,
					 const CUDAMemory * pSrc, const CUDAMemoryPosition & hSrcPos,
					 const CUDAMemoryRegion & hCopyRegion,
					 CUDAGraphInstance * pInstance = NULL );

protected:
	friend class CUDAGraph;

	CUDAMemory * m_pDest;
	const CUDAMemory * m_pSrc;
	
	Bool m_bIsFlatCopy;
	union _copy_params {
		struct _copy_params_flat {
			SizeT iSize;
		} hFlat;
		struct _copy_params_multi {
			CUDAMemoryPosition hDestPos;
			CUDAMemoryPosition hSrcPos;
			CUDAMemoryRegion hCopyRegion;
		} hShaped;
	} m_hCopyParams;
};

/////////////////////////////////////////////////////////////////////////////////
// The CUDANodeKernel class
class CUDANodeKernel : public CUDANode
{
public:
	CUDANodeKernel();
	virtual ~CUDANodeKernel();
	
	inline virtual CUDANodeType GetType() const;
	
	inline CUDAKernel * GetKernel() const;
	inline const CUDAKernelDimension & GetGridSize() const;
	inline const CUDAKernelDimension & GetBlockSize() const;
	inline Bool IsCooperative() const;
	
	Void SetKernel( CUDAKernel * pKernel, const CUDAKernelDimension & hGridSize, const CUDAKernelDimension & hBlockSize, Bool bCooperative = false, CUDAGraphInstance * pInstance = NULL );
	
protected:
	friend class CUDAGraph;

	CUDAKernel * m_pKernel;
	CUDAKernelDimension m_hGridSize;
	CUDAKernelDimension m_hBlockSize;
	Bool m_bCooperative;
};

/////////////////////////////////////////////////////////////////////////////////
// The CUDANodeHostFunction class
class CUDANodeHostFunction : public CUDANode
{
public:
	CUDANodeHostFunction();
	virtual ~CUDANodeHostFunction();
	
	inline virtual CUDANodeType GetType() const;
	
	inline CUDAHostFunction GetHostFunction() const;
	inline Void * GetUserData() const;
	
	Void SetHostFunction( CUDAHostFunction pfHostFunction, Void * pUserData, CUDAGraphInstance * pInstance = NULL );
	
protected:
	friend class CUDAGraph;

	CUDAHostFunction m_pfHostFunction;
	Void * m_pUserData;
};

/////////////////////////////////////////////////////////////////////////////////
// The CUDANodeEventRecord class
class CUDANodeEventRecord : public CUDANode
{
public:
	CUDANodeEventRecord();
	virtual ~CUDANodeEventRecord();
	
	inline virtual CUDANodeType GetType() const;
	
	inline CUDAEvent * GetEvent() const;
	
	Void SetEvent( CUDAEvent * pEvent, CUDAGraphInstance * pInstance = NULL );
	
protected:
	friend class CUDAGraph;

	CUDAEvent * m_pEvent;
};

/////////////////////////////////////////////////////////////////////////////////
// The CUDANodeEventRequire class
class CUDANodeEventRequire : public CUDANode
{
public:
	CUDANodeEventRequire();
	virtual ~CUDANodeEventRequire();
	
	inline virtual CUDANodeType GetType() const;
	
	inline CUDAEvent * GetEvent() const;
	
	Void SetEvent( CUDAEvent * pEvent, CUDAGraphInstance * pInstance = NULL );
	
protected:
	friend class CUDAGraph;

	CUDAEvent * m_pEvent;
};

/////////////////////////////////////////////////////////////////////////////////
// The CUDANodeChildGraph class
class CUDANodeChildGraph : public CUDANode
{
public:
	CUDANodeChildGraph();
	virtual ~CUDANodeChildGraph();
	
	inline virtual CUDANodeType GetType() const;
	
	inline CUDAGraph * GetChildGraph() const;
	
protected:
	friend class CUDAGraph;

	CUDAGraph * m_pChildGraph;
};

/////////////////////////////////////////////////////////////////////////////////
// The CUDAGraph class
class CUDAGraph
{
public:
	CUDAGraph();
	~CUDAGraph();
	
	// Deferred Creation/Destruction
	inline Bool IsCreated() const;
	
	Void Create();
	Void Destroy();
	
	Void Clone( CUDAGraph * outGraph ) const;
	// TODO : cudaGraphNodeFindInClone helper ?
	
	// Nodes management
	SizeT GetNodeCount() const;
	//cudaGraphGetNodes, Requires reverse-lookup tables ... not sure we want this ! Best to stay in "pure" mode !
	//cudaGraphGetRootNodes, Requires reverse-lookup tables ... not sure we want this ! Best to stay in "pure" mode !
	
		// Empty Nodes
	Void CreateNodeEmpty( CUDANodeEmpty * outNode );
	
		// Memory Nodes
	Void CreateNodeMemSet( CUDANodeMemSet * outNode,
						   CUDAMemory * pDest, const CUDAMemoryPosition & hDestPos, const CUDAMemoryRegion & hSetRegion, UInt iValue );
	Void CreateNodeMemCopy( CUDANodeMemCopy * outNode,
							CUDAMemory * pDest, const CUDAMemory * pSrc, SizeT iSize );
	Void CreateNodeMemCopy( CUDANodeMemCopy * outNode,
							CUDAMemory * pDest, const CUDAMemoryPosition & hDestPos,
							const CUDAMemory * pSrc, const CUDAMemoryPosition & hSrcPos,
							const CUDAMemoryRegion & hCopyRegion );
	//cudaGraphAddMemcpyNodeFromSymbol()
	//cudaGraphAddMemcpyNodeToSymbol()
	
		// Execution Nodes
	Void CreateNodeKernel( CUDANodeKernel * outNode, CUDAKernel * pKernel, const CUDAKernelDimension & hGridSize, const CUDAKernelDimension & hBlockSize, Bool bCooperative = false );
	Void CreateNodeHostFunction( CUDANodeHostFunction * outNode, CUDAHostFunction pfHostFunction, Void * pUserData );
	
		// Event Nodes, cannot be used inside loops/conditionals
	Void CreateNodeEventRecord( CUDANodeEventRecord * outNode, CUDAEvent * pEvent );
	Void CreateNodeEventRequire( CUDANodeEventRequire * outNode, CUDAEvent * pEvent );
	
		// Semaphore Nodes
	//cudaGraphAddExternalSemaphoresSignalNode()
	//cudaGraphAddExternalSemaphoresWaitNode()
	
		// ChildGraph Nodes, graph is cloned
	Void CreateNodeChildGraph( CUDANodeChildGraph * outNode, CUDAGraph * pChildGraph );
	
	// Edges management
	SizeT GetDependencyCount() const;
	//cudaGraphGetEdges, Requires reverse-lookup tables ... not sure we want this ! Best to stay in "pure" mode !
	
	Void AddDependency( const CUDANode * pRequiredNode, const CUDANode * pDependentNode ) const;
	Void RemoveDependency( const CUDANode * pRequiredNode, const CUDANode * pDependentNode ) const;
	
	// Instanciation (ie. run-time compilation)
	Void CreateInstance( CUDAGraphInstance * outGraphInstance ) const;

private:
	friend class CUDAStream;
	friend class CUDAGraphInstance;
	
	Void * m_hGraph;
};

/////////////////////////////////////////////////////////////////////////////////
// The CUDAGraphInstance class
class CUDAGraphInstance
{
public:
	CUDAGraphInstance();
	~CUDAGraphInstance();
	
	// Deferred Creation / Destruction
	inline Bool IsCreated() const;
	Void Destroy();
	
	// Updating
		// There are intricate conditions for these to succeed ... see CUDA documentation : cudaGraphExecUpdate and cudaGraphExecChildGraphNodeSetParams)
		// Mostly, graph topology and node types must strictly match.
	Bool Update( CUDAGraph * pGraph ) const;
	Void Update( CUDANodeChildGraph * pNodeChildGraph, CUDAGraph * pChildGraph ) const; // Node is unchanged ! Only instance is affected !
	
	// Execution
		// Note that concurrent execution requires multiple instances
		// Any one instance can be executed only once at any time
	Void Upload( CUDAStream * pStream ) const; // Load without execution
	Void Execute( CUDAStream * pStream ) const; // Load if required, then executes
	
private:
	friend class CUDANodeMemSet;
	friend class CUDANodeMemCopy;
	friend class CUDANodeKernel;
	friend class CUDANodeHostFunction;
	friend class CUDANodeEventRecord;
	friend class CUDANodeEventRequire;

	friend class CUDAGraph;

	Void * m_hGraphInstance;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "CUDAGraph.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_CUDA_CUDAGRAPH_H

