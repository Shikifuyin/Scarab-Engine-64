/////////////////////////////////////////////////////////////////////////////////
// File : MAGMAContext.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : MAGMA Context management
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// MAGMAContext implementation
inline static MAGMAContext * MAGMAContext::GetInstance() {
	static MAGMAContext s_hInstance;
	return &s_hInstance;
}