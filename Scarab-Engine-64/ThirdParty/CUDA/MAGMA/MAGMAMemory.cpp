/////////////////////////////////////////////////////////////////////////////////
// File : MAGMAMemory.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : MAGMA Memory Containers
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#include <magma_v2.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "MAGMAMemory.h"

/////////////////////////////////////////////////////////////////////////////////
// MAGMAMemory implementation
MAGMAMemory::MAGMAMemory()
{
	m_bHasOwnerShip = false;
	m_iSize = 0;
	m_pMemory = NULL;
}
MAGMAMemory::~MAGMAMemory()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// MAGMAHostMemory implementation
MAGMAHostMemory::MAGMAHostMemory():
	MAGMAMemory()
{
	m_bIsPinned = false;
}
MAGMAHostMemory::~MAGMAHostMemory()
{
	if ( m_bHasOwnerShip )
		Free();
}

Void MAGMAHostMemory::Allocate( SizeT iSize )
{
	Assert( m_pMemory == NULL );
	
	Void * pHostMemory = NULL;
	
	magma_int_t iError = magma_malloc_cpu( &pHostMemory, iSize );
	Assert( iError == MAGMA_SUCCESS && pHostMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_iSize = iSize;
	m_pMemory = pHostMemory;
	m_bIsPinned = false;
}
Void MAGMAHostMemory::AllocatePinned( SizeT iSize )
{
	Assert( m_pMemory == NULL );
	
	Void * pHostMemory = NULL;
	
	magma_int_t iError = magma_malloc_pinned( &pHostMemory, iSize );
	Assert( iError == MAGMA_SUCCESS && pHostMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_iSize = iSize;
	m_pMemory = pHostMemory;
	m_bIsPinned = true;
}
Void MAGMAHostMemory::Free()
{
	if ( m_pMemory == NULL )
		return;
	Assert( m_bHasOwnerShip );
	
	magma_int_t iError;
	if ( m_bIsPinned )
		iError = magma_free_pinned( m_pMemory );
	else
		iError = magma_free_cpu( m_pMemory );
	Assert( iError == MAGMA_SUCCESS );
	
	m_bHasOwnerShip = false;
	m_iSize = 0;
	m_pMemory = NULL;
	m_bIsPinned = false;
}

/////////////////////////////////////////////////////////////////////////////////
// MAGMADeviceMemory implementation
MAGMADeviceMemory::MAGMADeviceMemory():
	MAGMAMemory()
{
	// nothing to do
}
MAGMADeviceMemory::~MAGMADeviceMemory()
{
	if ( m_bHasOwnerShip )
		Free();
}

Void MAGMADeviceMemory::Allocate( SizeT iSize )
{
	Assert( m_pMemory == NULL );
	
	Void * pDeviceMemory = NULL;
	
	magma_int_t iError = magma_malloc( &pDeviceMemory, iSize );
	Assert( iError == MAGMA_SUCCESS && pDeviceMemory != NULL );
	
	m_bHasOwnerShip = true;
	m_iSize = iSize;
	m_pMemory = pDeviceMemory;
}
Void MAGMADeviceMemory::Free()
{
	if ( m_pMemory == NULL )
		return;
	Assert( m_bHasOwnerShip );
	
	magma_int_t iError = magma_free( m_pMemory );
	Assert( iError == MAGMA_SUCCESS );
	
	m_bHasOwnerShip = false;
	m_iSize = 0;
	m_pMemory = NULL;
}


