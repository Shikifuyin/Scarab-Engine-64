/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/Resources/D3D11Texture.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : GPU Resources : Textures.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// D3D11Texture implementation
inline Bool D3D11Texture::IsCreated() const {
    return ( m_pResource != NULL || m_bTemporaryDestroyed );
}
inline Void D3D11Texture::UpdateDataSource( const Void * pData ) {
    m_hCreationParameters.pData = pData;
}

inline PixelFormat D3D11Texture::GetFormat() const {
    return m_iFormat;
}
inline UInt D3D11Texture::GetStride() const {
    return PixelFormatBytes( m_iFormat );
}

inline UInt D3D11Texture::GetMipLevelCount() const {
    return m_iMipLevels;
}
inline Bool D3D11Texture::IsArray() const {
    return ( m_iArraySize > 1 );
}
inline UInt D3D11Texture::GetArraySize() const {
    return m_iArraySize;
}
inline UInt D3D11Texture::GetSubResourceCount() const {
    return ( m_iMipLevels * m_iArraySize );
}

/////////////////////////////////////////////////////////////////////////////////

inline UInt D3D11Texture::_GetBound( UInt iMaxBound, UInt iMipLevel ) {
    return Max<UInt>( 1, iMaxBound >> iMipLevel );
}
inline UInt D3D11Texture::_GetSubResourceIndex( const D3D11SubResourceIndex * pIndex, UInt iMipLevelCount, UInt iArrayCount ) {
    if ( pIndex == NULL )
        return 0;
    DebugAssert( pIndex->iMipSlice < iMipLevelCount );
    DebugAssert( pIndex->iArraySlice < iArrayCount );
    return ( (pIndex->iArraySlice * iMipLevelCount) + pIndex->iMipSlice );
}
inline UInt D3D11Texture::_GetSubResourceIndex( const D3D11SubResourceIndex * pIndex, D3D11TextureCubeFace iFace, UInt iMipLevelCount, UInt iArrayCount ) {
    if ( pIndex == NULL )
        return ( iFace * iMipLevelCount );
    DebugAssert( pIndex->iMipSlice < iMipLevelCount );
    DebugAssert( pIndex->iArraySlice < iArrayCount );
    UInt iAdjustedArraySlice = ( (pIndex->iArraySlice * 6) + iFace );
    return ( (iAdjustedArraySlice * iMipLevelCount) + pIndex->iMipSlice );
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11Texture1D implementation
inline D3D11ResourceType D3D11Texture1D::GetType() const {
    return D3D11RESOURCE_TEXTURE_1D;
}

inline UInt D3D11Texture1D::GetBound( UInt iDimension, UInt iMipLevel ) const {
    DebugAssert( IsCreated() );
    DebugAssert( iDimension < 1 );
    DebugAssert( iMipLevel < m_iMipLevels );
    return Max<UInt>( 1, m_iWidth >> iMipLevel );
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11Texture2D implementation
inline Bool D3D11Texture2D::IsBoundToBackBuffer( UInt * outBackBuffer ) const {
    DebugAssert( IsCreated() );
    if ( outBackBuffer != NULL )
        *outBackBuffer = m_iBoundToBackBuffer;
    return ( m_iBoundToBackBuffer != INVALID_OFFSET );
}

inline D3D11ResourceType D3D11Texture2D::GetType() const {
    return D3D11RESOURCE_TEXTURE_2D;
}

inline UInt D3D11Texture2D::GetBound( UInt iDimension, UInt iMipLevel ) const {
    DebugAssert( IsCreated() );
    DebugAssert( iDimension < 2 );
    DebugAssert( iMipLevel < m_iMipLevels );
    if ( iDimension == 0 )
        return Max<UInt>( 1, m_iWidth >> iMipLevel );
    else // iDimension == 1
        return Max<UInt>( 1, m_iHeight >> iMipLevel );
}

inline Bool D3D11Texture2D::IsMultiSampled() const {
    return ( m_iSampleCount > 1 );
}
inline UInt D3D11Texture2D::GetSampleCount() const {
    return m_iSampleCount;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11Texture3D implementation
inline D3D11ResourceType D3D11Texture3D::GetType() const {
    return D3D11RESOURCE_TEXTURE_3D;
}

inline UInt D3D11Texture3D::GetBound( UInt iDimension, UInt iMipLevel ) const {
    DebugAssert( IsCreated() );
    DebugAssert( iDimension < 3 );
    DebugAssert( iMipLevel < m_iMipLevels );
    if ( iDimension == 0 )
        return Max<UInt>( 1, m_iWidth >> iMipLevel );
    else if ( iDimension == 1 )
        return Max<UInt>( 1, m_iHeight >> iMipLevel );
    else // iDimension == 2
        return Max<UInt>( 1, m_iDepth >> iMipLevel );
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11TextureCube implementation
inline D3D11ResourceType D3D11TextureCube::GetType() const {
    return D3D11RESOURCE_TEXTURE_CUBE;
}

inline UInt D3D11TextureCube::GetBound( UInt iDimension, UInt iMipLevel ) const {
    DebugAssert( IsCreated() );
    DebugAssert( iDimension < 2 );
    DebugAssert( iMipLevel < m_iMipLevels );
    if ( iDimension == 0 )
        return Max<UInt>( 1, m_iWidth >> iMipLevel );
    else // iDimension == 1
        return Max<UInt>( 1, m_iHeight >> iMipLevel );
}

inline Bool D3D11TextureCube::IsCubeArray() const {
    DebugAssert( IsCreated() );
    return ( m_iArraySize > 6 );
}
inline UInt D3D11TextureCube::GetCubeCount() const {
    DebugAssert( IsCreated() );
    return ( m_iArraySize / 6 );
}
