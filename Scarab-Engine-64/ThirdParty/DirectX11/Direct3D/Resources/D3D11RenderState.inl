/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/Resources/D3D11RenderState.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : GPU resources : Render states.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// D3D11RenderState implementation
inline Bool D3D11RenderState::IsCreated() const {
    return ( m_pRenderState != NULL || m_bTemporaryDestroyed );
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11SamplerState implementation
inline D3D11RenderStateType D3D11SamplerState::GetType() const {
    return D3D11RENDERSTATE_SAMPLER;
}

inline const D3D11SamplerStateDesc * D3D11SamplerState::GetDesc() const {
    return &m_hDesc;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11RasterizerState implementation
inline D3D11RenderStateType D3D11RasterizerState::GetType() const {
    return D3D11RENDERSTATE_RASTERIZER;
}

inline const D3D11RasterizerStateDesc * D3D11RasterizerState::GetDesc() const {
    return &m_hDesc;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11DepthStencilState implementation
inline D3D11RenderStateType D3D11DepthStencilState::GetType() const {
    return D3D11RENDERSTATE_DEPTHSTENCIL;
}

inline const D3D11DepthStencilStateDesc * D3D11DepthStencilState::GetDesc() const {
    return &m_hDesc;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11BlendState implementation
inline D3D11RenderStateType D3D11BlendState::GetType() const {
    return D3D11RENDERSTATE_BLEND;
}

inline const D3D11BlendStateDesc * D3D11BlendState::GetDesc() const {
    return &m_hDesc;
}

