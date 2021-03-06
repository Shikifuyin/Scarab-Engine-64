/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/D3D11Renderer.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Platform-dependant abstraction for 3D graphics.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// D3D11RendererCallbacks implementation
inline Void D3D11RendererCallbacks::SetUserData( Void * pUserData )  {
    m_pUserData = pUserData;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11Renderer implementation
inline D3D11Window * D3D11Renderer::GetWindow() const {
    return m_pWindow;
}

inline const D3D11SwapChainDesc * D3D11Renderer::GetSwapChainDesc() const {
    DebugAssert( m_pSwapChain != NULL );
    return &m_hSwapChainDesc;
}

inline Bool D3D11Renderer::IsIdle() const {
    return m_bIdleState;
}

