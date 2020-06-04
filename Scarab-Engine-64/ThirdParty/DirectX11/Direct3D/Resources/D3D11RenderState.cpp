/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/Resources/D3D11RenderState.cpp
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
// Third-Party Includes
#pragma warning(disable:4005)

#define WIN32_LEAN_AND_MEAN
#include <d3d11.h>

#undef DebugAssert

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "D3D11RenderState.h"

#include "../D3D11Renderer.h"

/////////////////////////////////////////////////////////////////////////////////
// D3D11RenderState implementation
D3D11RenderState::D3D11RenderState( D3D11Renderer * pRenderer )
{
    m_pRenderer = pRenderer;

    m_pRenderState = NULL;

    m_bTemporaryDestroyed = false;
}
D3D11RenderState::~D3D11RenderState()
{
    // nothing to do
}

Void D3D11RenderState::Destroy()
{
    DebugAssert( IsCreated() );

    if ( m_bTemporaryDestroyed )
        m_bTemporaryDestroyed = false;
    else
        _NakedDestroy();
}

Void D3D11RenderState::OnDestroyDevice()
{
    DebugAssert( !m_bTemporaryDestroyed );

    if ( m_pRenderState != NULL ) {
        _NakedDestroy();
        m_bTemporaryDestroyed = true;
    }
}
Void D3D11RenderState::OnRestoreDevice()
{
    DebugAssert( m_pRenderState == NULL );

    if ( m_bTemporaryDestroyed ) {
        _NakedCreate();
        m_bTemporaryDestroyed = false;
    }
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11SamplerState implementation
D3D11SamplerState::D3D11SamplerState( D3D11Renderer * pRenderer ):
    D3D11RenderState( pRenderer )
{
    m_pSamplerState = NULL;

    m_hDesc.iFilterMode = D3D11SAMPLER_FILTER_MIN_MAG_MIP_LLL;
    m_hDesc.iWrapModeU = D3D11SAMPLER_WRAP_CLAMP;
    m_hDesc.iWrapModeV = D3D11SAMPLER_WRAP_CLAMP;
    m_hDesc.iWrapModeW = D3D11SAMPLER_WRAP_CLAMP;
    m_hDesc.arrBorderColor[0] = 1.0f;
    m_hDesc.arrBorderColor[1] = 1.0f;
    m_hDesc.arrBorderColor[2] = 1.0f;
    m_hDesc.arrBorderColor[3] = 1.0f;
    m_hDesc.fMinLOD = -FLOAT_INFINITE;
    m_hDesc.fMaxLOD = FLOAT_INFINITE;
    m_hDesc.fLODBias = 0.0f;
    m_hDesc.iMaxAnisotropy = 1;
    m_hDesc.iCompareFunction = D3D11SAMPLER_COMPARE_NEVER;
}
D3D11SamplerState::~D3D11SamplerState()
{
    if ( IsCreated() )
        Destroy();
}

Void D3D11SamplerState::Create( const D3D11SamplerStateDesc * pSamplerDesc )
{
    DebugAssert( !(IsCreated()) );

    MemCopy( &m_hDesc, pSamplerDesc, sizeof(D3D11SamplerStateDesc) );

    _NakedCreate();
}
Void D3D11SamplerState::Destroy()
{
    D3D11RenderState::Destroy();

    m_hDesc.iFilterMode = D3D11SAMPLER_FILTER_MIN_MAG_MIP_LLL;
    m_hDesc.iWrapModeU = D3D11SAMPLER_WRAP_CLAMP;
    m_hDesc.iWrapModeV = D3D11SAMPLER_WRAP_CLAMP;
    m_hDesc.iWrapModeW = D3D11SAMPLER_WRAP_CLAMP;
    m_hDesc.arrBorderColor[0] = 1.0f;
    m_hDesc.arrBorderColor[1] = 1.0f;
    m_hDesc.arrBorderColor[2] = 1.0f;
    m_hDesc.arrBorderColor[3] = 1.0f;
    m_hDesc.fMinLOD = -FLOAT_INFINITE;
    m_hDesc.fMaxLOD = FLOAT_INFINITE;
    m_hDesc.fLODBias = 0.0f;
    m_hDesc.iMaxAnisotropy = 1;
    m_hDesc.iCompareFunction = D3D11SAMPLER_COMPARE_NEVER;
}

/////////////////////////////////////////////////////////////////////////////////

Void D3D11SamplerState::_NakedCreate()
{
    D3D11_SAMPLER_DESC hD3D11Desc;
    m_hDesc.ConvertTo( &hD3D11Desc );

    m_pSamplerState = NULL;
    HRESULT hRes = m_pRenderer->m_pDevice->CreateSamplerState( &hD3D11Desc, (ID3D11SamplerState**)&m_pSamplerState );
    DebugAssert( hRes == S_OK && m_pSamplerState != NULL );

    m_pRenderState = NULL;
    hRes = ((ID3D11SamplerState*)m_pSamplerState)->QueryInterface( __uuidof(ID3D11DeviceChild), &m_pRenderState );
    DebugAssert( hRes == S_OK && m_pRenderState != NULL );
}
Void D3D11SamplerState::_NakedDestroy()
{
    ((ID3D11DeviceChild*)m_pRenderState)->Release();
    m_pRenderState = NULL;

    ((ID3D11SamplerState*)m_pSamplerState)->Release();
    m_pSamplerState = NULL;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11RasterizerState implementation
D3D11RasterizerState::D3D11RasterizerState( D3D11Renderer * pRenderer ):
    D3D11RenderState( pRenderer )
{
    m_pRasterizerState = NULL;

    m_hDesc.iFillMode = D3D11RASTERIZER_FILL_SOLID;
    m_hDesc.iCullMode = D3D11RASTERIZER_CULL_BACK;
    m_hDesc.bFrontCounterClockwise = true;
    m_hDesc.iDepthBias = 0;
    m_hDesc.fDepthBiasClamp = 0.0f;
    m_hDesc.fSlopeScaledDepthBias = 0.0f;
    m_hDesc.bDepthClipEnabled = true;
    m_hDesc.bScissorEnabled = false;
    m_hDesc.bMultisampleEnabled = false;
    m_hDesc.bAntialiasedLineEnabled = false;
}
D3D11RasterizerState::~D3D11RasterizerState()
{
    if ( IsCreated() )
        Destroy();
}

Void D3D11RasterizerState::Create( const D3D11RasterizerStateDesc * pRasterizerDesc )
{
    DebugAssert( !(IsCreated()) );

    MemCopy( &m_hDesc, pRasterizerDesc, sizeof(D3D11RasterizerStateDesc) );

    _NakedCreate();
}
Void D3D11RasterizerState::Destroy()
{
    D3D11RenderState::Destroy();

    m_hDesc.iFillMode = D3D11RASTERIZER_FILL_SOLID;
    m_hDesc.iCullMode = D3D11RASTERIZER_CULL_BACK;
    m_hDesc.bFrontCounterClockwise = true;
    m_hDesc.iDepthBias = 0;
    m_hDesc.fDepthBiasClamp = 0.0f;
    m_hDesc.fSlopeScaledDepthBias = 0.0f;
    m_hDesc.bDepthClipEnabled = true;
    m_hDesc.bScissorEnabled = false;
    m_hDesc.bMultisampleEnabled = false;
    m_hDesc.bAntialiasedLineEnabled = false;
}

/////////////////////////////////////////////////////////////////////////////////

Void D3D11RasterizerState::_NakedCreate()
{
    D3D11_RASTERIZER_DESC hD3D11Desc;
    m_hDesc.ConvertTo( &hD3D11Desc );

    m_pRasterizerState = NULL;
    HRESULT hRes = m_pRenderer->m_pDevice->CreateRasterizerState( &hD3D11Desc, (ID3D11RasterizerState**)&m_pRasterizerState );
    DebugAssert( hRes == S_OK && m_pRasterizerState != NULL );

    m_pRenderState = NULL;
    hRes = ((ID3D11RasterizerState*)m_pRasterizerState)->QueryInterface( __uuidof(ID3D11DeviceChild), &m_pRenderState );
    DebugAssert( hRes == S_OK && m_pRenderState != NULL );
}
Void D3D11RasterizerState::_NakedDestroy()
{
    ((ID3D11DeviceChild*)m_pRenderState)->Release();
    m_pRenderState = NULL;

    ((ID3D11RasterizerState*)m_pRasterizerState)->Release();
    m_pRasterizerState = NULL;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11DepthStencilState implementation
D3D11DepthStencilState::D3D11DepthStencilState( D3D11Renderer * pRenderer ):
    D3D11RenderState( pRenderer )
{
    m_pDepthStencilState = NULL;

    m_hDesc.bDepthEnabled = true;
    m_hDesc.iDepthWriteMask = D3D11DEPTH_WRITEMASK_ALL;
    m_hDesc.iDepthFunction = D3D11DEPTHSTENCIL_COMPARE_LESSER;
    m_hDesc.bStencilEnabled = false;
    m_hDesc.iStencilReadMask = 0xff;
    m_hDesc.iStencilWriteMask = 0xff;
    m_hDesc.hFrontFace.iOnStencilFail = D3D11STENCIL_OP_KEEP;
    m_hDesc.hFrontFace.iOnStencilDepthFail = D3D11STENCIL_OP_KEEP;
    m_hDesc.hFrontFace.iOnStencilPass = D3D11STENCIL_OP_KEEP;
    m_hDesc.hFrontFace.iStencilFunction = D3D11DEPTHSTENCIL_COMPARE_ALLWAYS;
    m_hDesc.hBackFace.iOnStencilFail = D3D11STENCIL_OP_KEEP;
    m_hDesc.hBackFace.iOnStencilDepthFail = D3D11STENCIL_OP_KEEP;
    m_hDesc.hBackFace.iOnStencilPass = D3D11STENCIL_OP_KEEP;
    m_hDesc.hBackFace.iStencilFunction = D3D11DEPTHSTENCIL_COMPARE_ALLWAYS;
}
D3D11DepthStencilState::~D3D11DepthStencilState()
{
    if ( IsCreated() )
        Destroy();
}

Void D3D11DepthStencilState::Create( const D3D11DepthStencilStateDesc * pDepthStencilDesc )
{
    DebugAssert( !(IsCreated()) );

    MemCopy( &m_hDesc, pDepthStencilDesc, sizeof(D3D11DepthStencilStateDesc) );

    _NakedCreate();
}
Void D3D11DepthStencilState::Destroy()
{
    D3D11RenderState::Destroy();

    m_hDesc.bDepthEnabled = true;
    m_hDesc.iDepthWriteMask = D3D11DEPTH_WRITEMASK_ALL;
    m_hDesc.iDepthFunction = D3D11DEPTHSTENCIL_COMPARE_LESSER;
    m_hDesc.bStencilEnabled = false;
    m_hDesc.iStencilReadMask = 0xff;
    m_hDesc.iStencilWriteMask = 0xff;
    m_hDesc.hFrontFace.iOnStencilFail = D3D11STENCIL_OP_KEEP;
    m_hDesc.hFrontFace.iOnStencilDepthFail = D3D11STENCIL_OP_KEEP;
    m_hDesc.hFrontFace.iOnStencilPass = D3D11STENCIL_OP_KEEP;
    m_hDesc.hFrontFace.iStencilFunction = D3D11DEPTHSTENCIL_COMPARE_ALLWAYS;
    m_hDesc.hBackFace.iOnStencilFail = D3D11STENCIL_OP_KEEP;
    m_hDesc.hBackFace.iOnStencilDepthFail = D3D11STENCIL_OP_KEEP;
    m_hDesc.hBackFace.iOnStencilPass = D3D11STENCIL_OP_KEEP;
    m_hDesc.hBackFace.iStencilFunction = D3D11DEPTHSTENCIL_COMPARE_ALLWAYS;
}

/////////////////////////////////////////////////////////////////////////////////

Void D3D11DepthStencilState::_NakedCreate()
{
    D3D11_DEPTH_STENCIL_DESC hD3D11Desc;
    m_hDesc.ConvertTo( &hD3D11Desc );

    m_pDepthStencilState = NULL;
    HRESULT hRes = m_pRenderer->m_pDevice->CreateDepthStencilState( &hD3D11Desc, (ID3D11DepthStencilState**)&m_pDepthStencilState );
    DebugAssert( hRes == S_OK && m_pDepthStencilState != NULL );

    m_pRenderState = NULL;
    hRes = ((ID3D11DepthStencilState*)m_pDepthStencilState)->QueryInterface( __uuidof(ID3D11DeviceChild), &m_pRenderState );
    DebugAssert( hRes == S_OK && m_pRenderState != NULL );
}
Void D3D11DepthStencilState::_NakedDestroy()
{
    ((ID3D11DeviceChild*)m_pRenderState)->Release();
    m_pRenderState = NULL;

    ((ID3D11DepthStencilState*)m_pDepthStencilState)->Release();
    m_pDepthStencilState = NULL;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11BlendState implementation
D3D11BlendState::D3D11BlendState( D3D11Renderer * pRenderer ):
    D3D11RenderState( pRenderer )
{
    m_pBlendState = NULL;

    m_hDesc.bAlphaToCoverageEnabled = false;
    m_hDesc.bIndependentBlendEnabled = false;
    for( UInt i = 0; i < D3D11RENDERER_MAX_RENDERTARGET_SLOTS; ++i ) {
        m_hDesc.arrRenderTargets[i].bBlendEnabled = false;
        m_hDesc.arrRenderTargets[i].iBlendSrc = D3D11BLEND_PARAM_ONE;
        m_hDesc.arrRenderTargets[i].iBlendSrcAlpha = D3D11BLEND_PARAM_ONE;
        m_hDesc.arrRenderTargets[i].iBlendDst = D3D11BLEND_PARAM_ZERO;
        m_hDesc.arrRenderTargets[i].iBlendDstAlpha = D3D11BLEND_PARAM_ZERO;
        m_hDesc.arrRenderTargets[i].iBlendOp = D3D11BLEND_OP_ADD;
        m_hDesc.arrRenderTargets[i].iBlendOpAlpha = D3D11BLEND_OP_ADD;
        m_hDesc.arrRenderTargets[i].iColorWriteMask = D3D11BLEND_COLORWRITEMASK_ALL;
    }
}
D3D11BlendState::~D3D11BlendState()
{
    if ( IsCreated() )
        Destroy();
}

Void D3D11BlendState::Create( const D3D11BlendStateDesc * pBlendDesc )
{
    DebugAssert( !(IsCreated()) );

    MemCopy( &m_hDesc, pBlendDesc, sizeof(D3D11BlendStateDesc) );

    _NakedCreate();
}
Void D3D11BlendState::Destroy()
{
    D3D11RenderState::Destroy();

    m_hDesc.bAlphaToCoverageEnabled = false;
    m_hDesc.bIndependentBlendEnabled = false;
    for ( UInt i = 0; i < D3D11RENDERER_MAX_RENDERTARGET_SLOTS; ++i ) {
        m_hDesc.arrRenderTargets[i].bBlendEnabled = false;
        m_hDesc.arrRenderTargets[i].iBlendSrc = D3D11BLEND_PARAM_ONE;
        m_hDesc.arrRenderTargets[i].iBlendSrcAlpha = D3D11BLEND_PARAM_ONE;
        m_hDesc.arrRenderTargets[i].iBlendDst = D3D11BLEND_PARAM_ZERO;
        m_hDesc.arrRenderTargets[i].iBlendDstAlpha = D3D11BLEND_PARAM_ZERO;
        m_hDesc.arrRenderTargets[i].iBlendOp = D3D11BLEND_OP_ADD;
        m_hDesc.arrRenderTargets[i].iBlendOpAlpha = D3D11BLEND_OP_ADD;
        m_hDesc.arrRenderTargets[i].iColorWriteMask = D3D11BLEND_COLORWRITEMASK_ALL;
    }
}

/////////////////////////////////////////////////////////////////////////////////

Void D3D11BlendState::_NakedCreate()
{
    D3D11_BLEND_DESC hD3D11Desc;
    m_hDesc.ConvertTo( &hD3D11Desc );

    m_pBlendState = NULL;
    HRESULT hRes = m_pRenderer->m_pDevice->CreateBlendState( &hD3D11Desc, (ID3D11BlendState**)&m_pBlendState );
    DebugAssert( hRes == S_OK && m_pBlendState != NULL );

    m_pRenderState = NULL;
    hRes = ((ID3D11BlendState*)m_pBlendState)->QueryInterface( __uuidof(ID3D11DeviceChild), &m_pRenderState );
    DebugAssert( hRes == S_OK && m_pRenderState != NULL );
}
Void D3D11BlendState::_NakedDestroy()
{
    ((ID3D11DeviceChild*)m_pRenderState)->Release();
    m_pRenderState = NULL;

    ((ID3D11BlendState*)m_pBlendState)->Release();
    m_pBlendState = NULL;
}

