/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Tools/WinGUIImage.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Images (Bitmap or Icon)
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// WinGUIImage implementation
inline Bool WinGUIImage::IsCreated() const {
	return ( m_hHandle != NULL );
}
inline Bool WinGUIImage::IsDeviceDependant() const {
	return m_bIsDeviceDependant;
}
inline Bool WinGUIImage::IsShared() const {
	return m_bShared;
}
inline WinGUIImageType WinGUIImage::GetType() const {
	return m_iType;
}

inline UInt WinGUIImage::GetDDWidth() const {
	DebugAssert( m_hHandle != NULL && m_bIsDeviceDependant );
	return m_iDDWidth;
}
inline UInt WinGUIImage::GetDDHeight() const {
	DebugAssert( m_hHandle != NULL && m_bIsDeviceDependant );
	return m_iDDHeight;
}

inline const WinGUIBitmapDescriptor * WinGUIImage::GetDIBDescriptor() const {
	DebugAssert( m_hHandle != NULL && !m_bIsDeviceDependant );
	return &m_hBitmapDesc;
}

