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
// WinGUIBitmap implementation
inline Bool WinGUIBitmap::IsCreated() const {
	return ( m_hHandle != NULL );
}
inline Bool WinGUIBitmap::IsDeviceDependant() const {
	return m_bIsDeviceDependant;
}
inline Bool WinGUIBitmap::IsShared() const {
	return m_bShared;
}

inline UInt WinGUIBitmap::GetDDWidth() const {
	DebugAssert( m_hHandle != NULL && m_bIsDeviceDependant );
	return m_iDDWidth;
}
inline UInt WinGUIBitmap::GetDDHeight() const {
	DebugAssert( m_hHandle != NULL && m_bIsDeviceDependant );
	return m_iDDHeight;
}

inline const WinGUIBitmapDescriptor * WinGUIBitmap::GetDIBDescriptor() const {
	DebugAssert( m_hHandle != NULL && !m_bIsDeviceDependant );
	return &m_hBitmapDesc;
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIIcon implementation
inline Bool WinGUIIcon::IsCreated() const {
	return ( m_hHandle != NULL );
}
inline Bool WinGUIIcon::IsShared() const {
	return m_bShared;
}

inline const WinGUIPoint * WinGUIIcon::GetHotSpot() const {
	return &m_hHotSpot;
}

inline WinGUIBitmap * WinGUIIcon::GetBitmapColor() {
	return &m_hBitmapColor;
}
inline WinGUIBitmap * WinGUIIcon::GetBitmapMask() {
	return &m_hBitmapMask;
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUICursor implementation
inline Bool WinGUICursor::IsCreated() const {
	return ( m_hHandle != NULL );
}
inline Bool WinGUICursor::IsShared() const {
	return m_bShared;
}

inline const WinGUIPoint * WinGUICursor::GetHotSpot() const {
	return &m_hHotSpot;
}

inline WinGUIBitmap * WinGUICursor::GetBitmapColor() {
	return &m_hBitmapColor;
}
inline WinGUIBitmap * WinGUICursor::GetBitmapMask() {
	return &m_hBitmapMask;
}

