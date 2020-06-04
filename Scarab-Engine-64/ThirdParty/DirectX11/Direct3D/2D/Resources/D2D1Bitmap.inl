/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/2D/Resources/D2D1Bitmap.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : D2D1 Dev-Dep Resource : Bitmaps.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// D2D1Bitmap implementation
inline Bool D2D1Bitmap::IsCreated() const {
    return ( m_pBitmap != NULL || m_bTemporaryDestroyed );
}
inline Void D2D1Bitmap::UpdateDataSource( const Void * pData ) {
    m_hCreationParameters.pData = pData;
}

inline PixelFormat D2D1Bitmap::GetPixelFormat() const {
    DebugAssert( IsCreated() );
    return m_hDesc.iFormat;
}
inline D2D1BitmapAlphaMode D2D1Bitmap::GetAlphaMode() const {
    DebugAssert( IsCreated() );
    return m_hDesc.iAlphaMode;
}

inline Float D2D1Bitmap::GetDpiX() const {
    DebugAssert( IsCreated() );
    return m_hDesc.fDpiX;
}
inline Float D2D1Bitmap::GetDpiY() const {
    DebugAssert( IsCreated() );
    return m_hDesc.fDpiY;
}

inline UInt D2D1Bitmap::GetWidth() const {
    DebugAssert( IsCreated() );
    return m_hDesc.iWidth;
}
inline UInt D2D1Bitmap::GetHeight() const {
    DebugAssert( IsCreated() );
    return m_hDesc.iHeight;
}

inline Float D2D1Bitmap::GetDIPWidth() const {
    DebugAssert( IsCreated() );
    return m_hDesc.fDIPWidth;
}
inline Float D2D1Bitmap::GetDIPHeight() const {
    DebugAssert( IsCreated() );
    return m_hDesc.fDIPHeight;
}

inline const D2D1BitmapDesc * D2D1Bitmap::GetDesc() const {
    DebugAssert( IsCreated() );
    return &m_hDesc;
}

