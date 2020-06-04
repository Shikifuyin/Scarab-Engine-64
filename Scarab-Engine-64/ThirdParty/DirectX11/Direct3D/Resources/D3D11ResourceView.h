/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/Resources/D3D11ResourceView.h
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
// Header prelude
#ifndef SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_RESOURCES_D3D11RESOURCEVIEW_H
#define SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_RESOURCES_D3D11RESOURCEVIEW_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "D3D11Buffer.h"
#include "D3D11Texture.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Resource types
enum D3D11ResourceViewType {
    D3D11RESOURCEVIEW_RENDER_TARGET = 0,
    D3D11RESOURCEVIEW_DEPTH_STENCIL,
    D3D11RESOURCEVIEW_SHADER,
    D3D11RESOURCEVIEW_UNORDERED_ACCESS
};

// Prototypes
class D3D11Renderer;

/////////////////////////////////////////////////////////////////////////////////
// The D3D11ResourceView class
class D3D11ResourceView
{
protected:
    D3D11ResourceView( D3D11Renderer * pRenderer );
public:
    virtual ~D3D11ResourceView();

    // Deferred construction
    inline Bool IsCreated() const;
    inline Bool IsBoundToBackBuffer( UInt * outBackBuffer = NULL ) const;

    Void Destroy();

    Void OnDestroyDevice();
    Void OnRestoreDevice();

    // Getters
    virtual D3D11ResourceViewType GetType() const = 0;

protected:
    friend class D3D11Renderer;
    D3D11Renderer * m_pRenderer;

    D3D11Resource * m_pResource;

    Void * m_pView;

    // Auto-Regen system
    virtual Void _NakedCreate() = 0;
    virtual Void _NakedDestroy() = 0;

    Bool m_bTemporaryDestroyed;
    UInt m_iBoundToBackBuffer;
};

/////////////////////////////////////////////////////////////////////////////////
// The D3D11RenderTargetView class
class D3D11RenderTargetView : public D3D11ResourceView
{
public:
    D3D11RenderTargetView( D3D11Renderer * pRenderer );
    virtual ~D3D11RenderTargetView();

    // Deferred construction
    Void AttachToBackBuffer( UInt iBackBuffer );
    Void Create( D3D11Buffer * pBuffer, UInt iIndex = 0, UInt iCount = INVALID_OFFSET );
    Void Create( D3D11Texture * pTexture, UInt iMipSlice = 0, UInt iArraySlice = 0, UInt iArraySliceCount = 1 );

    // Getters
    inline virtual D3D11ResourceViewType GetType() const;

    inline const D3D11RenderTargetViewDesc * GetDesc() const;

protected:
    friend class D3D11Renderer;

    Void * m_pRenderTargetView;
    D3D11RenderTargetViewDesc m_hDesc;

    // Auto-Regen system
    virtual Void _NakedCreate();
    virtual Void _NakedDestroy();
};

/////////////////////////////////////////////////////////////////////////////////
// The D3D11DepthStencilView class
class D3D11DepthStencilView : public D3D11ResourceView
{
public:
    D3D11DepthStencilView( D3D11Renderer * pRenderer );
    virtual ~D3D11DepthStencilView();

    // Deferred construction
    Void AttachToBackBuffer( UInt iBackBuffer, Bool bReadOnlyDepth = false, Bool bReadOnlyStencil = false );
    Void Create( D3D11Texture * pTexture, UInt iMipSlice = 0, UInt iArraySlice = 0, UInt iArraySliceCount = 1, Bool bReadOnlyDepth = false, Bool bReadOnlyStencil = false );

    // Getters
    inline virtual D3D11ResourceViewType GetType() const;

    inline const D3D11DepthStencilViewDesc * GetDesc() const;

protected:
    friend class D3D11Renderer;

    Void * m_pDepthStencilView;
    D3D11DepthStencilViewDesc m_hDesc;

    // Auto-Regen system
    virtual Void _NakedCreate();
    virtual Void _NakedDestroy();
};

/////////////////////////////////////////////////////////////////////////////////
// The D3D11ShaderView class
class D3D11ShaderView : public D3D11ResourceView
{
public:
    D3D11ShaderView( D3D11Renderer * pRenderer );
    virtual ~D3D11ShaderView();

    // Deferred construction (iArraySlice/iArraySliceCount = iFirstCube/iCubeCount for D3D11TextureCube)
    Void AttachToBackBuffer( UInt iBackBuffer );
    Void Create( D3D11Buffer * pBuffer, UInt iIndex = 0, UInt iCount = INVALID_OFFSET );
    Void Create( D3D11Texture * pTexture, UInt iMostDetailedMip = 0, UInt iMipLevelCount = INVALID_OFFSET, UInt iArraySlice = 0, UInt iArraySliceCount = 1 );

    // Getters
    inline virtual D3D11ResourceViewType GetType() const;

    inline const D3D11ShaderViewDesc * GetDesc() const;

protected:
    friend class D3D11Renderer;

    Void * m_pShaderView;
    D3D11ShaderViewDesc m_hDesc;

    // Auto-Regen system
    virtual Void _NakedCreate();
    virtual Void _NakedDestroy();
};

/////////////////////////////////////////////////////////////////////////////////
// The D3D11UnorderedAccessView class
class D3D11UnorderedAccessView : public D3D11ResourceView
{
public:
    D3D11UnorderedAccessView( D3D11Renderer * pRenderer );
    virtual ~D3D11UnorderedAccessView();

    // Deferred construction
    Void AttachToBackBuffer( UInt iBackBuffer );
    Void Create( D3D11Buffer * pBuffer, UInt iIndex = 0, UInt iCount = INVALID_OFFSET, Bool bAppendConsume = false, Bool bUseCounter = false );
    Void Create( D3D11Texture * pTexture, UInt iMipSlice = 0, UInt iArraySlice = 0, UInt iArraySliceCount = 1 );

    // Getters
    inline virtual D3D11ResourceViewType GetType() const;

    inline const D3D11UnorderedAccessViewDesc * GetDesc() const;

protected:
    friend class D3D11Renderer;

    Void * m_pUnorderedAccessView;
    D3D11UnorderedAccessViewDesc m_hDesc;

    // Auto-Regen system
    virtual Void _NakedCreate();
    virtual Void _NakedDestroy();
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "D3D11ResourceView.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_RESOURCES_D3D11RESOURCEVIEW_H

