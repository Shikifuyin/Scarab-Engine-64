/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/Resources/D3D11RenderState.h
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
// Header prelude
#ifndef SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_RESOURCES_D3D11RENDERSTATE_H
#define SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_RESOURCES_D3D11RENDERSTATE_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../D3D11Mappings.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Render states types
enum D3D11RenderStateType {
    D3D11RENDERSTATE_SAMPLER = 0,
    D3D11RENDERSTATE_RASTERIZER,
    D3D11RENDERSTATE_DEPTHSTENCIL,
    D3D11RENDERSTATE_BLEND
};

// Prototypes
class D3D11Renderer;

/////////////////////////////////////////////////////////////////////////////////
// The D3D11RenderState class
class D3D11RenderState
{
protected:
    D3D11RenderState( D3D11Renderer * pRenderer );
public:
    virtual ~D3D11RenderState();

    // Deferred construction
    inline Bool IsCreated() const;

    virtual Void Destroy();

    Void OnDestroyDevice();
    Void OnRestoreDevice();

    // Getters
    virtual D3D11RenderStateType GetType() const = 0;

protected:
    friend class D3D11Renderer;
    D3D11Renderer * m_pRenderer;

    Void * m_pRenderState;

    // Auto-Regen system
    virtual Void _NakedCreate() = 0;
    virtual Void _NakedDestroy() = 0;

    Bool m_bTemporaryDestroyed;
};

/////////////////////////////////////////////////////////////////////////////////
// The D3D11SamplerState class
class D3D11SamplerState : public D3D11RenderState
{
public:
    D3D11SamplerState( D3D11Renderer * pRenderer );
    virtual ~D3D11SamplerState();

    // Deferred construction
    Void Create( const D3D11SamplerStateDesc * pSamplerDesc );
    virtual Void Destroy();

    // Getters
    inline virtual D3D11RenderStateType GetType() const;

    inline const D3D11SamplerStateDesc * GetDesc() const;

protected:
    friend class D3D11Renderer;

    Void * m_pSamplerState;
    D3D11SamplerStateDesc m_hDesc;

    // Auto-Regen system
    virtual Void _NakedCreate();
    virtual Void _NakedDestroy();
};

/////////////////////////////////////////////////////////////////////////////////
// The D3D11RasterizerState class
class D3D11RasterizerState : public D3D11RenderState
{
public:
    D3D11RasterizerState( D3D11Renderer * pRenderer );
    virtual ~D3D11RasterizerState();

    // Deferred construction
    Void Create( const D3D11RasterizerStateDesc * pRasterizerDesc );
    virtual Void Destroy();

    // Getters
    inline virtual D3D11RenderStateType GetType() const;

    inline const D3D11RasterizerStateDesc * GetDesc() const;

protected:
    friend class D3D11Renderer;

    Void * m_pRasterizerState;
    D3D11RasterizerStateDesc m_hDesc;

    // Auto-Regen system
    virtual Void _NakedCreate();
    virtual Void _NakedDestroy();
};

/////////////////////////////////////////////////////////////////////////////////
// The D3D11DepthStencilState class
class D3D11DepthStencilState : public D3D11RenderState
{
public:
    D3D11DepthStencilState( D3D11Renderer * pRenderer );
    virtual ~D3D11DepthStencilState();

    // Deferred construction
    Void Create( const D3D11DepthStencilStateDesc * pDepthStencilDesc );
    virtual Void Destroy();

    // Getters
    inline virtual D3D11RenderStateType GetType() const;

    inline const D3D11DepthStencilStateDesc * GetDesc() const;

protected:
    friend class D3D11Renderer;

    Void * m_pDepthStencilState;
    D3D11DepthStencilStateDesc m_hDesc;

    // Auto-Regen system
    virtual Void _NakedCreate();
    virtual Void _NakedDestroy();
};

/////////////////////////////////////////////////////////////////////////////////
// The D3D11BlendState class
class D3D11BlendState : public D3D11RenderState
{
public:
    D3D11BlendState( D3D11Renderer * pRenderer );
    virtual ~D3D11BlendState();

    // Deferred construction
    Void Create( const D3D11BlendStateDesc * pBlendDesc );
    virtual Void Destroy();

    // Getters
    inline virtual D3D11RenderStateType GetType() const;

    inline const D3D11BlendStateDesc * GetDesc() const;

protected:
    friend class D3D11Renderer;

    Void * m_pBlendState;
    D3D11BlendStateDesc m_hDesc;

    // Auto-Regen system
    virtual Void _NakedCreate();
    virtual Void _NakedDestroy();
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "D3D11RenderState.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_RESOURCES_D3D11RENDERSTATE_H

