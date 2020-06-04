/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/Resources/D3D11ResourceView.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : GPU Resources : Bind-Views.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// D3D11ResourceView implementation
inline Bool D3D11ResourceView::IsCreated() const {
    return ( m_pView != NULL || m_bTemporaryDestroyed );
}
inline Bool D3D11ResourceView::IsBoundToBackBuffer( UInt * outBackBuffer ) const {
    DebugAssert( IsCreated() );
    if ( m_iBoundToBackBuffer == INVALID_OFFSET && m_pResource->GetType() == D3D11RESOURCE_TEXTURE_2D )
        return ((D3D11Texture2D*)m_pResource)->IsBoundToBackBuffer( outBackBuffer );
    if ( outBackBuffer != NULL )
        *outBackBuffer = m_iBoundToBackBuffer;
    return ( m_iBoundToBackBuffer != INVALID_OFFSET );
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11RenderTargetView implementation
inline D3D11ResourceViewType D3D11RenderTargetView::GetType() const {
    return D3D11RESOURCEVIEW_RENDER_TARGET;
}

inline const D3D11RenderTargetViewDesc * D3D11RenderTargetView::GetDesc() const {
    DebugAssert( IsCreated() );
    return &m_hDesc;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11DepthStencilView implementation
inline D3D11ResourceViewType D3D11DepthStencilView::GetType() const {
    return D3D11RESOURCEVIEW_DEPTH_STENCIL;
}

inline const D3D11DepthStencilViewDesc * D3D11DepthStencilView::GetDesc() const {
    DebugAssert( IsCreated() );
    return &m_hDesc;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11ShaderView implementation
inline D3D11ResourceViewType D3D11ShaderView::GetType() const {
    return D3D11RESOURCEVIEW_SHADER;
}

inline const D3D11ShaderViewDesc * D3D11ShaderView::GetDesc() const {
    DebugAssert( IsCreated() );
    return &m_hDesc;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11UnorderedAccessView implementation
inline D3D11ResourceViewType D3D11UnorderedAccessView::GetType() const {
    return D3D11RESOURCEVIEW_UNORDERED_ACCESS;
}

inline const D3D11UnorderedAccessViewDesc * D3D11UnorderedAccessView::GetDesc() const {
    DebugAssert( IsCreated() );
    return &m_hDesc;
}

