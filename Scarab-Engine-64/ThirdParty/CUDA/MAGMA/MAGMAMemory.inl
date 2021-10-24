/////////////////////////////////////////////////////////////////////////////////
// File : MAGMAMemory.inl
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
// MAGMAMemory implementation
inline Bool MAGMAMemory::IsAllocated() const {
	return ( m_pMemory != NULL );
}
inline Bool MAGMAMemory::HasOwnerShip() const {
	return m_bHasOwnerShip;
}

inline SizeT MAGMAMemory::GetSize() const {
	Assert( m_pMemory != NULL );
	return m_iSize;
}
inline Void * MAGMAMemory::GetPointer() {
	Assert( m_pMemory != NULL );
	return m_pMemory;
}
inline const Void * MAGMAMemory::GetPointer() const {
	Assert( m_pMemory != NULL );
	return m_pMemory;
}

/////////////////////////////////////////////////////////////////////////////////
// MAGMAHostMemory implementation
inline Bool MAGMAHostMemory::IsHostMemory() const {
	return true;
}
inline Bool MAGMAHostMemory::IsDeviceMemory() const {
	return false;
}

/////////////////////////////////////////////////////////////////////////////////
// MAGMADeviceMemory implementation
inline Bool MAGMADeviceMemory::IsHostMemory() const {
	return false;
}
inline Bool MAGMADeviceMemory::IsDeviceMemory() const {
	return true;
}

	
	