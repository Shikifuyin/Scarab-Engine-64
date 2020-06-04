/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/Resources/D3D11ResourceView.cpp
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
// Third-Party Includes
#pragma warning(disable:4005)

#define WIN32_LEAN_AND_MEAN
#include <d3d11.h>

#undef DebugAssert

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "D3D11ResourceView.h"

#include "../D3D11Renderer.h"

/////////////////////////////////////////////////////////////////////////////////
// D3D11ResourceView implementation
D3D11ResourceView::D3D11ResourceView( D3D11Renderer * pRenderer )
{
    m_pRenderer = pRenderer;

    m_pResource = NULL;

    m_pView = NULL;

    m_bTemporaryDestroyed = false;
    m_iBoundToBackBuffer = INVALID_OFFSET;
}
D3D11ResourceView::~D3D11ResourceView()
{
    // nothing to do
}

Void D3D11ResourceView::Destroy()
{
    DebugAssert( IsCreated() );

    if ( m_bTemporaryDestroyed )
        m_bTemporaryDestroyed = false;
    else
        _NakedDestroy();
}

Void D3D11ResourceView::OnDestroyDevice()
{
    DebugAssert( !m_bTemporaryDestroyed );

    if ( m_pView != NULL ) {
        _NakedDestroy();
        m_bTemporaryDestroyed = true;
    }
}
Void D3D11ResourceView::OnRestoreDevice()
{
    DebugAssert( m_pView == NULL );

    if ( m_bTemporaryDestroyed ) {
        _NakedCreate();
        m_bTemporaryDestroyed = false;
    }
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11RenderTargetView implementation
D3D11RenderTargetView::D3D11RenderTargetView( D3D11Renderer * pRenderer ):
    D3D11ResourceView( pRenderer )
{
    m_pRenderTargetView = NULL;

    m_hDesc.iViewDimension = D3D11RENDERTARGETVIEW_DIM_UNKNOWN;
    m_hDesc.iFormat = PIXEL_FMT_UNKNOWN;
}
D3D11RenderTargetView::~D3D11RenderTargetView()
{
    if ( IsCreated() )
        Destroy();
}

Void D3D11RenderTargetView::AttachToBackBuffer( UInt iBackBuffer )
{
    DebugAssert( !(IsCreated()) );

    DebugAssert( iBackBuffer < m_pRenderer->m_hSwapChainDesc.iBufferCount );

    m_iBoundToBackBuffer = iBackBuffer;
    m_pResource = NULL;

    m_hDesc.iViewDimension = D3D11RENDERTARGETVIEW_DIM_TEXTURE2D;
    m_hDesc.iFormat = m_pRenderer->m_hSwapChainDesc.iFormat;
    m_hDesc.hTexture2D.iMipSlice = 0;

    _NakedCreate();
}
Void D3D11RenderTargetView::Create( D3D11Buffer * pBuffer, UInt iIndex, UInt iCount )
{
    DebugAssert( !(IsCreated()) );

    DebugAssert( (pBuffer->GetBinds() & D3D11RESOURCE_BIND_RENDER_TARGET) != 0 );

    DebugAssert( pBuffer->IsCreated() );
    DebugAssert( (iIndex + iCount) * pBuffer->GetStride() <= pBuffer->GetByteSize() );

    m_iBoundToBackBuffer = INVALID_OFFSET;
    m_pResource = pBuffer;

    m_hDesc.iViewDimension = D3D11RENDERTARGETVIEW_DIM_BUFFER;
    m_hDesc.hBuffer.iOffset = iIndex;
    m_hDesc.hBuffer.iSize = iCount;

    switch( pBuffer->GetType() ) {
        case D3D11RESOURCE_BUFFER_RAW:    m_hDesc.iFormat = PIXEL_FMT_R32; break;
        case D3D11RESOURCE_BUFFER_STRUCT: m_hDesc.iFormat = PIXEL_FMT_UNKNOWN; break;
        default: DebugAssert( false ); break;
    }

    _NakedCreate();
}
Void D3D11RenderTargetView::Create( D3D11Texture * pTexture, UInt iMipSlice, UInt iArraySlice, UInt iArraySliceCount )
{
    DebugAssert( !(IsCreated()) );

    DebugAssert( (pTexture->GetBinds() & D3D11RESOURCE_BIND_RENDER_TARGET) != 0 );

    DebugAssert( pTexture->IsCreated() );
    DebugAssert( iMipSlice < pTexture->GetMipLevelCount() );
    DebugAssert( iArraySlice + iArraySliceCount <= pTexture->GetArrayCount() );

    m_iBoundToBackBuffer = INVALID_OFFSET;
    m_pResource = pTexture;

    m_hDesc.iFormat = pTexture->GetFormat();

    switch( pTexture->GetType() ) {
        case D3D11RESOURCE_TEXTURE_1D:
            if ( pTexture->IsArray() ) {
                m_hDesc.iViewDimension = D3D11RENDERTARGETVIEW_DIM_TEXTURE1DARRAY;
                m_hDesc.hTexture1DArray.iMipSlice= iMipSlice;
                m_hDesc.hTexture1DArray.iArraySlice = iArraySlice;
                m_hDesc.hTexture1DArray.iArraySliceCount = iArraySliceCount;
            } else {
                m_hDesc.iViewDimension = D3D11RENDERTARGETVIEW_DIM_TEXTURE1D;
                m_hDesc.hTexture1D.iMipSlice = iMipSlice;
            }
            break;
        case D3D11RESOURCE_TEXTURE_2D:
            if ( pTexture->IsArray() ) {
                if ( ((D3D11Texture2D*)pTexture)->IsMultiSampled() ) {
                    m_hDesc.iViewDimension = D3D11RENDERTARGETVIEW_DIM_TEXTURE2DMSARRAY;
                    m_hDesc.hTexture2DMSArray.iArraySlice = iArraySlice;
                    m_hDesc.hTexture2DMSArray.iArraySliceCount = iArraySliceCount;
                } else {
                    m_hDesc.iViewDimension = D3D11RENDERTARGETVIEW_DIM_TEXTURE2DARRAY;
                    m_hDesc.hTexture2DArray.iMipSlice = iMipSlice;
                    m_hDesc.hTexture2DArray.iArraySlice = iArraySlice;
                    m_hDesc.hTexture2DArray.iArraySliceCount = iArraySliceCount;
                }
            } else {
                if ( ((D3D11Texture2D*)pTexture)->IsMultiSampled() ) {
                    m_hDesc.iViewDimension = D3D11RENDERTARGETVIEW_DIM_TEXTURE2DMS;
                    m_hDesc.hTexture2DMS._reserved = 0;
                } else {
                    m_hDesc.iViewDimension = D3D11RENDERTARGETVIEW_DIM_TEXTURE2D;
                    m_hDesc.hTexture2D.iMipSlice = iMipSlice;
                }
            }
            break;
        case D3D11RESOURCE_TEXTURE_3D:
            m_hDesc.iViewDimension = D3D11RENDERTARGETVIEW_DIM_TEXTURE3D;
            m_hDesc.hTexture3D.iMipSlice = iMipSlice;
            m_hDesc.hTexture3D.iDepthSlice = iArraySlice;
            m_hDesc.hTexture3D.iDepthSliceCount = iArraySliceCount;
            break;
        default: DebugAssert( false ); break;
    }

    _NakedCreate();
}

/////////////////////////////////////////////////////////////////////////////////

Void D3D11RenderTargetView::_NakedCreate()
{
    HRESULT hRes;

    D3D11_RENDER_TARGET_VIEW_DESC hD3D11Desc;
    m_hDesc.ConvertTo( &hD3D11Desc );

    if ( m_iBoundToBackBuffer == INVALID_OFFSET ) {
        m_pRenderTargetView = NULL;
        hRes = ((ID3D11Device*)(m_pRenderer->m_pDevice))->CreateRenderTargetView( (ID3D11Resource*)(m_pResource->m_pResource), &hD3D11Desc, (ID3D11RenderTargetView**)&m_pRenderTargetView );
        DebugAssert( hRes == S_OK && m_pRenderTargetView != NULL );
    } else {
        ID3D11Texture2D * pBackBuffer = NULL;
        hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->GetBuffer( m_iBoundToBackBuffer, __uuidof(ID3D11Texture2D), (Void**)&pBackBuffer );
        DebugAssert( hRes == S_OK && pBackBuffer != NULL );

        m_pRenderTargetView = NULL;
        hRes = ((ID3D11Device*)(m_pRenderer->m_pDevice))->CreateRenderTargetView( pBackBuffer, &hD3D11Desc, (ID3D11RenderTargetView**)&m_pRenderTargetView );
        DebugAssert( hRes == S_OK && m_pRenderTargetView != NULL );

        pBackBuffer->Release();
        pBackBuffer = NULL;
    }

    m_pView = NULL;
    hRes = ((ID3D11RenderTargetView*)m_pRenderTargetView)->QueryInterface( __uuidof(ID3D11View), &m_pView );
    DebugAssert( hRes == S_OK && m_pView != NULL );
}
Void D3D11RenderTargetView::_NakedDestroy()
{
    ((ID3D11View*)m_pView)->Release();
    m_pView = NULL;

    ((ID3D11RenderTargetView*)m_pRenderTargetView)->Release();
    m_pRenderTargetView = NULL;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11DepthStencilView implementation
D3D11DepthStencilView::D3D11DepthStencilView( D3D11Renderer * pRenderer ):
    D3D11ResourceView( pRenderer )
{
    m_pDepthStencilView = NULL;

    m_hDesc.iViewDimension = D3D11DEPTHSTENCILVIEW_DIM_UNKNOWN;
    m_hDesc.iFormat = PIXEL_FMT_UNKNOWN;
    m_hDesc.iFlags = 0;
}
D3D11DepthStencilView::~D3D11DepthStencilView()
{
    if ( IsCreated() )
        Destroy();
}

Void D3D11DepthStencilView::AttachToBackBuffer( UInt iBackBuffer, Bool bReadOnlyDepth, Bool bReadOnlyStencil )
{
    DebugAssert( !(IsCreated()) );

    DebugAssert( iBackBuffer < m_pRenderer->m_hSwapChainDesc.iBufferCount );

    m_iBoundToBackBuffer = iBackBuffer;
    m_pResource = NULL;

    m_hDesc.iViewDimension = D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2D;
    m_hDesc.iFormat = m_pRenderer->m_hSwapChainDesc.iFormat;
    m_hDesc.iFlags = 0;
    if ( bReadOnlyDepth )
        m_hDesc.iFlags |= D3D11DEPTHSTENCILVIEW_FLAG_READONLY_DEPTH;
    if ( bReadOnlyStencil )
        m_hDesc.iFlags |= D3D11DEPTHSTENCILVIEW_FLAG_READONLY_STENCIL;
    m_hDesc.hTexture2D.iMipSlice = 0;

    _NakedCreate();
}
Void D3D11DepthStencilView::Create( D3D11Texture * pTexture, UInt iMipSlice, UInt iArraySlice, UInt iArraySliceCount, Bool bReadOnlyDepth, Bool bReadOnlyStencil )
{
    DebugAssert( !(IsCreated()) );

    DebugAssert( (pTexture->GetBinds() & D3D11RESOURCE_BIND_DEPTH_STENCIL) != 0 );

    DebugAssert( pTexture->IsCreated() );
    DebugAssert( iMipSlice < pTexture->GetMipLevelCount() );
    DebugAssert( iArraySlice + iArraySliceCount <= pTexture->GetArrayCount() );

    m_iBoundToBackBuffer = INVALID_OFFSET;
    m_pResource = pTexture;

    m_hDesc.iFormat = pTexture->GetFormat();
    m_hDesc.iFlags = 0;
    if ( bReadOnlyDepth )
        m_hDesc.iFlags |= D3D11DEPTHSTENCILVIEW_FLAG_READONLY_DEPTH;
    if ( bReadOnlyStencil )
        m_hDesc.iFlags |= D3D11DEPTHSTENCILVIEW_FLAG_READONLY_STENCIL;

    switch( pTexture->GetType() ) {
        case D3D11RESOURCE_TEXTURE_1D:
            if ( pTexture->IsArray() ) {
                m_hDesc.iViewDimension = D3D11DEPTHSTENCILVIEW_DIM_TEXTURE1DARRAY;
                m_hDesc.hTexture1DArray.iMipSlice = iMipSlice;
                m_hDesc.hTexture1DArray.iArraySlice = iArraySlice;
                m_hDesc.hTexture1DArray.iArraySliceCount = iArraySliceCount;
            } else {
                m_hDesc.iViewDimension = D3D11DEPTHSTENCILVIEW_DIM_TEXTURE1D;
                m_hDesc.hTexture1D.iMipSlice = iMipSlice;
            }
            break;
        case D3D11RESOURCE_TEXTURE_2D:
            if ( pTexture->IsArray() ) {
                if ( ((D3D11Texture2D*)pTexture)->IsMultiSampled() ) {
                    m_hDesc.iViewDimension = D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2DMSARRAY;
                    m_hDesc.hTexture2DMSArray.iArraySlice = iArraySlice;
                    m_hDesc.hTexture2DMSArray.iArraySliceCount = iArraySliceCount;
                } else {
                    m_hDesc.iViewDimension = D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2DARRAY;
                    m_hDesc.hTexture2DArray.iMipSlice = iMipSlice;
                    m_hDesc.hTexture2DArray.iArraySlice = iArraySlice;
                    m_hDesc.hTexture2DArray.iArraySliceCount = iArraySliceCount;
                }
            } else {
                if ( ((D3D11Texture2D*)pTexture)->IsMultiSampled() ) {
                    m_hDesc.iViewDimension = D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2DMS;
                    m_hDesc.hTexture2DMS._reserved = 0;
                } else {
                    m_hDesc.iViewDimension = D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2D;
                    m_hDesc.hTexture2D.iMipSlice = iMipSlice;
                }
            }
            break;
        default: DebugAssert( false ); break;
    }

    _NakedCreate();
}

/////////////////////////////////////////////////////////////////////////////////

Void D3D11DepthStencilView::_NakedCreate()
{
    HRESULT hRes;

    D3D11_DEPTH_STENCIL_VIEW_DESC hD3D11Desc;
    m_hDesc.ConvertTo( &hD3D11Desc );

    if ( m_iBoundToBackBuffer == INVALID_OFFSET ) {
        m_pDepthStencilView = NULL;
        hRes = ((ID3D11Device*)(m_pRenderer->m_pDevice))->CreateDepthStencilView( (ID3D11Resource*)(m_pResource->m_pResource), &hD3D11Desc, (ID3D11DepthStencilView**)&m_pDepthStencilView );
        DebugAssert( hRes == S_OK && m_pDepthStencilView != NULL );
    } else {
        ID3D11Texture2D * pBackBuffer = NULL;
        hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->GetBuffer( m_iBoundToBackBuffer, __uuidof(ID3D11Texture2D), (Void**)&pBackBuffer );
        DebugAssert( hRes == S_OK && pBackBuffer != NULL );

        m_pDepthStencilView = NULL;
        hRes = ((ID3D11Device*)(m_pRenderer->m_pDevice))->CreateDepthStencilView( pBackBuffer, &hD3D11Desc, (ID3D11DepthStencilView**)&m_pDepthStencilView );
        DebugAssert( hRes == S_OK && m_pDepthStencilView != NULL );

        pBackBuffer->Release();
        pBackBuffer = NULL;
    }

    m_pView = NULL;
    hRes = ((ID3D11DepthStencilView*)m_pDepthStencilView)->QueryInterface( __uuidof(ID3D11View), &m_pView );
    DebugAssert( hRes == S_OK && m_pView != NULL );
}
Void D3D11DepthStencilView::_NakedDestroy()
{
    ((ID3D11View*)m_pView)->Release();
    m_pView = NULL;

    ((ID3D11DepthStencilView*)m_pDepthStencilView)->Release();
    m_pDepthStencilView = NULL;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11ShaderView implementation
D3D11ShaderView::D3D11ShaderView( D3D11Renderer * pRenderer ):
    D3D11ResourceView( pRenderer )
{
    m_pShaderView = NULL;

    m_hDesc.iViewDimension = D3D11SHADERVIEW_DIM_UNKNOWN;
    m_hDesc.iFormat = PIXEL_FMT_UNKNOWN;
}
D3D11ShaderView::~D3D11ShaderView()
{
    if ( IsCreated() )
        Destroy();
}

Void D3D11ShaderView::AttachToBackBuffer( UInt iBackBuffer )
{
    DebugAssert( !(IsCreated()) );

    DebugAssert( iBackBuffer < m_pRenderer->m_hSwapChainDesc.iBufferCount );

    m_iBoundToBackBuffer = iBackBuffer;
    m_pResource = NULL;

    m_hDesc.iViewDimension = D3D11SHADERVIEW_DIM_TEXTURE2D;
    m_hDesc.iFormat = m_pRenderer->m_hSwapChainDesc.iFormat;
    m_hDesc.hTexture2D.iMostDetailedMip = 0;
    m_hDesc.hTexture2D.iMipLevels = 1;

    _NakedCreate();
}
Void D3D11ShaderView::Create( D3D11Buffer * pBuffer, UInt iIndex, UInt iCount )
{
    DebugAssert( !(IsCreated()) );

    DebugAssert( (pBuffer->GetBinds() & D3D11RESOURCE_BIND_SHADER_INPUT) != 0 );

    DebugAssert( pBuffer->IsCreated() );
    DebugAssert( (iIndex + iCount) * pBuffer->GetStride() <= pBuffer->GetByteSize() );

    m_iBoundToBackBuffer = INVALID_OFFSET;
    m_pResource = pBuffer;

    m_hDesc.iViewDimension = D3D11SHADERVIEW_DIM_BUFFEREX;
    m_hDesc.hBufferEx.iOffset = iIndex;
    m_hDesc.hBufferEx.iSize = iCount;

    switch( pBuffer->GetType() ) {
        case D3D11RESOURCE_BUFFER_RAW:
            m_hDesc.iFormat = PIXEL_FMT_R32;
            m_hDesc.hBufferEx.iFlags = D3D11SHADERVIEW_BUFFEREXFLAG_RAW;
            break;
        case D3D11RESOURCE_BUFFER_STRUCT:
            m_hDesc.iFormat = PIXEL_FMT_UNKNOWN;
            m_hDesc.hBufferEx.iFlags = 0;
            break;
        default: DebugAssert( false ); break;
    }

    _NakedCreate();
}
Void D3D11ShaderView::Create( D3D11Texture * pTexture, UInt iMostDetailedMip, UInt iMipLevelCount, UInt iArraySlice, UInt iArraySliceCount )
{
    DebugAssert( !(IsCreated()) );

    DebugAssert( (pTexture->GetBinds() & D3D11RESOURCE_BIND_SHADER_INPUT) != 0 );

    DebugAssert( pTexture->IsCreated() );
    DebugAssert( iMostDetailedMip < pTexture->GetMipLevelCount() );
    if ( iMipLevelCount == INVALID_OFFSET )
        iMipLevelCount = ( pTexture->GetMipLevelCount() - iMostDetailedMip );
    DebugAssert( iMostDetailedMip + iMipLevelCount <= pTexture->GetMipLevelCount() );
    DebugAssert( iArraySlice + iArraySliceCount <= pTexture->GetArrayCount() );

    m_iBoundToBackBuffer = INVALID_OFFSET;
    m_pResource = pTexture;

    m_hDesc.iFormat = pTexture->GetFormat();

    switch( pTexture->GetType() ) {
        case D3D11RESOURCE_TEXTURE_1D:
            if ( pTexture->IsArray() ) {
                m_hDesc.iViewDimension = D3D11SHADERVIEW_DIM_TEXTURE1DARRAY;
                m_hDesc.hTexture1DArray.iMostDetailedMip = iMostDetailedMip;
                m_hDesc.hTexture1DArray.iMipLevels = iMipLevelCount;
                m_hDesc.hTexture1DArray.iArraySlice = iArraySlice;
                m_hDesc.hTexture1DArray.iArraySliceCount = iArraySliceCount;
            } else {
                m_hDesc.iViewDimension = D3D11SHADERVIEW_DIM_TEXTURE1D;
                m_hDesc.hTexture1D.iMostDetailedMip = iMostDetailedMip;
                m_hDesc.hTexture1D.iMipLevels = iMipLevelCount;
            }
            break;
        case D3D11RESOURCE_TEXTURE_2D:
            if ( pTexture->IsArray() ) {
                if ( ((D3D11Texture2D*)pTexture)->IsMultiSampled() ) {
                    m_hDesc.iViewDimension = D3D11SHADERVIEW_DIM_TEXTURE2DMSARRAY;
                    m_hDesc.hTexture2DMSArray.iArraySlice = iArraySlice;
                    m_hDesc.hTexture2DMSArray.iArraySliceCount = iArraySliceCount;
                } else {
                    m_hDesc.iViewDimension = D3D11SHADERVIEW_DIM_TEXTURE2DARRAY;
                    m_hDesc.hTexture2DArray.iMostDetailedMip = iMostDetailedMip;
                    m_hDesc.hTexture2DArray.iMipLevels = iMipLevelCount;
                    m_hDesc.hTexture2DArray.iArraySlice = iArraySlice;
                    m_hDesc.hTexture2DArray.iArraySliceCount = iArraySliceCount;
                }
            } else {
                if ( ((D3D11Texture2D*)pTexture)->IsMultiSampled() ) {
                    m_hDesc.iViewDimension = D3D11SHADERVIEW_DIM_TEXTURE2DMS;
                    m_hDesc.hTexture2DMS._reserved = 0;
                } else {
                    m_hDesc.iViewDimension = D3D11SHADERVIEW_DIM_TEXTURE2D;
                    m_hDesc.hTexture2D.iMostDetailedMip = iMostDetailedMip;
                    m_hDesc.hTexture2D.iMipLevels = iMipLevelCount;
                }
            }
            break;
        case D3D11RESOURCE_TEXTURE_3D:
            m_hDesc.iViewDimension = D3D11SHADERVIEW_DIM_TEXTURE3D;
            m_hDesc.hTexture3D.iMostDetailedMip = iMostDetailedMip;
            m_hDesc.hTexture3D.iMipLevels = iMipLevelCount;
            break;
        case D3D11RESOURCE_TEXTURE_CUBE:
            if ( ((D3D11TextureCube*)pTexture)->IsCubeArray() ) {
                m_hDesc.iViewDimension = D3D11SHADERVIEW_DIM_TEXTURECUBEARRAY;
                m_hDesc.hTextureCubeArray.iMostDetailedMip = iMostDetailedMip;
                m_hDesc.hTextureCubeArray.iMipLevels = iMipLevelCount;
                m_hDesc.hTextureCubeArray.iFirstFaceIndex = iArraySlice;
                m_hDesc.hTextureCubeArray.iCubeCount = ( iArraySliceCount / 6 );
            } else {
                m_hDesc.iViewDimension = D3D11SHADERVIEW_DIM_TEXTURECUBE;
                m_hDesc.hTextureCube.iMostDetailedMip = iMostDetailedMip;
                m_hDesc.hTextureCube.iMipLevels = iMipLevelCount;
            }
            break;
        default: DebugAssert( false ); break;
    }

    _NakedCreate();
}

/////////////////////////////////////////////////////////////////////////////////

Void D3D11ShaderView::_NakedCreate()
{
    HRESULT hRes;

    D3D11_SHADER_RESOURCE_VIEW_DESC hD3D11Desc;
    m_hDesc.ConvertTo( &hD3D11Desc );

    if ( m_iBoundToBackBuffer == INVALID_OFFSET ) {
        m_pShaderView = NULL;
        hRes = ((ID3D11Device*)(m_pRenderer->m_pDevice))->CreateShaderResourceView( (ID3D11Resource*)(m_pResource->m_pResource), &hD3D11Desc, (ID3D11ShaderResourceView**)&m_pShaderView );
        DebugAssert( hRes == S_OK && m_pShaderView != NULL );
    } else {
        ID3D11Texture2D * pBackBuffer = NULL;
        hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->GetBuffer( m_iBoundToBackBuffer, __uuidof(ID3D11Texture2D), (Void**)&pBackBuffer );
        DebugAssert( hRes == S_OK && pBackBuffer != NULL );

        m_pShaderView = NULL;
        hRes = ((ID3D11Device*)(m_pRenderer->m_pDevice))->CreateShaderResourceView( pBackBuffer, &hD3D11Desc, (ID3D11ShaderResourceView**)&m_pShaderView );
        DebugAssert( hRes == S_OK && m_pShaderView != NULL );

        pBackBuffer->Release();
        pBackBuffer = NULL;
    }

    m_pView = NULL;
    hRes = ((ID3D11ShaderResourceView*)m_pShaderView)->QueryInterface( __uuidof(ID3D11View), &m_pView );
    DebugAssert( hRes == S_OK && m_pView != NULL );
}
Void D3D11ShaderView::_NakedDestroy()
{
    ((ID3D11View*)m_pView)->Release();
    m_pView = NULL;

    ((ID3D11ShaderResourceView*)m_pShaderView)->Release();
    m_pShaderView = NULL;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11UnorderedAccessView implementation
D3D11UnorderedAccessView::D3D11UnorderedAccessView( D3D11Renderer * pRenderer ):
    D3D11ResourceView( pRenderer )
{
    m_pUnorderedAccessView = NULL;

    m_hDesc.iViewDimension = D3D11UNORDEREDACCESSVIEW_DIM_UNKNOWN;
    m_hDesc.iFormat = PIXEL_FMT_UNKNOWN;
}
D3D11UnorderedAccessView::~D3D11UnorderedAccessView()
{
    if ( IsCreated() )
        Destroy();
}

Void D3D11UnorderedAccessView::AttachToBackBuffer( UInt iBackBuffer )
{
    DebugAssert( !(IsCreated()) );

    DebugAssert( iBackBuffer < m_pRenderer->m_hSwapChainDesc.iBufferCount );

    m_iBoundToBackBuffer = iBackBuffer;
    m_pResource = NULL;

    m_hDesc.iViewDimension = D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE2D;
    m_hDesc.iFormat = m_pRenderer->m_hSwapChainDesc.iFormat;
    m_hDesc.hTexture2D.iMipSlice = 0;

    _NakedCreate();
}
Void D3D11UnorderedAccessView::Create( D3D11Buffer * pBuffer, UInt iIndex, UInt iCount, Bool bAppendConsume, Bool bUseCounter )
{
    DebugAssert( !(IsCreated()) );

    DebugAssert( (pBuffer->GetBinds() & D3D11RESOURCE_BIND_UNORDERED_ACCESS) != 0 );

    DebugAssert( pBuffer->IsCreated() );
    DebugAssert( (iIndex + iCount) * pBuffer->GetStride() <= pBuffer->GetByteSize() );

    m_iBoundToBackBuffer = INVALID_OFFSET;
    m_pResource = pBuffer;

    m_hDesc.iViewDimension = D3D11UNORDEREDACCESSVIEW_DIM_BUFFER;
    m_hDesc.hBuffer.iOffset = iIndex;
    m_hDesc.hBuffer.iSize = iCount;

    switch( pBuffer->GetType() ) {
        case D3D11RESOURCE_BUFFER_RAW:
            m_hDesc.iFormat = PIXEL_FMT_R32;
            m_hDesc.hBuffer.iFlags = D3D11UNORDEREDACCESSVIEW_BUFFERFLAG_RAW;
            break;
        case D3D11RESOURCE_BUFFER_STRUCT:
            m_hDesc.iFormat = PIXEL_FMT_UNKNOWN;
            m_hDesc.hBuffer.iFlags = 0;
            if ( bAppendConsume )
                m_hDesc.hBuffer.iFlags |= D3D11UNORDEREDACCESSVIEW_BUFFERFLAG_APPEND;
            if ( bUseCounter ) {
                DebugAssert( pBuffer->CanGPUWrite() );
                m_hDesc.hBuffer.iFlags |= D3D11UNORDEREDACCESSVIEW_BUFFERFLAG_COUNTER;
            }
            break;
        default: DebugAssert( false ); break;
    }

    _NakedCreate();
}
Void D3D11UnorderedAccessView::Create( D3D11Texture * pTexture, UInt iMipSlice, UInt iArraySlice, UInt iArraySliceCount )
{
    DebugAssert( !(IsCreated()) );

    DebugAssert( (pTexture->GetBinds() & D3D11RESOURCE_BIND_UNORDERED_ACCESS) != 0 );

    DebugAssert( pTexture->IsCreated() );
    DebugAssert( iMipSlice < pTexture->GetMipLevelCount() );
    DebugAssert( iArraySlice + iArraySliceCount <= pTexture->GetArrayCount() );

    m_iBoundToBackBuffer = INVALID_OFFSET;
    m_pResource = pTexture;

    m_hDesc.iFormat = pTexture->GetFormat();

    switch( pTexture->GetType() ) {
        case D3D11RESOURCE_TEXTURE_1D:
            if ( pTexture->IsArray() ) {
                m_hDesc.iViewDimension = D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE1DARRAY;
                m_hDesc.hTexture1DArray.iMipSlice = iMipSlice;
                m_hDesc.hTexture1DArray.iArraySlice = iArraySlice;
                m_hDesc.hTexture1DArray.iArraySliceCount = iArraySliceCount;
            } else {
                m_hDesc.iViewDimension = D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE1D;
                m_hDesc.hTexture1D.iMipSlice = iMipSlice;
            }
            break;
        case D3D11RESOURCE_TEXTURE_2D:
            DebugAssert( !(((D3D11Texture2D*)pTexture)->IsMultiSampled()) );
            if ( pTexture->IsArray() ) {
                m_hDesc.iViewDimension = D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE2DARRAY;
                m_hDesc.hTexture2DArray.iMipSlice = iMipSlice;
                m_hDesc.hTexture2DArray.iArraySlice = iArraySlice;
                m_hDesc.hTexture2DArray.iArraySliceCount = iArraySliceCount;
            } else {
                m_hDesc.iViewDimension = D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE2D;
                m_hDesc.hTexture2D.iMipSlice = iMipSlice;
            }
            break;
        case D3D11RESOURCE_TEXTURE_3D:
            m_hDesc.iViewDimension = D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE3D;
            m_hDesc.hTexture3D.iMipSlice = iMipSlice;
            m_hDesc.hTexture3D.iDepthSlice = iArraySlice;
            m_hDesc.hTexture3D.iDepthSliceCount = iArraySliceCount;
            break;
        default: DebugAssert( false ); break;
    }

    _NakedCreate();
}

/////////////////////////////////////////////////////////////////////////////////

Void D3D11UnorderedAccessView::_NakedCreate()
{
    HRESULT hRes;

    D3D11_UNORDERED_ACCESS_VIEW_DESC hD3D11Desc;
    m_hDesc.ConvertTo( &hD3D11Desc );

    if ( m_iBoundToBackBuffer == INVALID_OFFSET ) {
        m_pUnorderedAccessView = NULL;
        hRes = ((ID3D11Device*)(m_pRenderer->m_pDevice))->CreateUnorderedAccessView( (ID3D11Resource*)(m_pResource->m_pResource), &hD3D11Desc, (ID3D11UnorderedAccessView**)&m_pUnorderedAccessView );
        DebugAssert( hRes == S_OK && m_pUnorderedAccessView != NULL );
    } else {
        ID3D11Texture2D * pBackBuffer = NULL;
        hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->GetBuffer( m_iBoundToBackBuffer, __uuidof(ID3D11Texture2D), (Void**)&pBackBuffer );
        DebugAssert( hRes == S_OK && pBackBuffer != NULL );

        m_pUnorderedAccessView = NULL;
        hRes = ((ID3D11Device*)(m_pRenderer->m_pDevice))->CreateUnorderedAccessView( pBackBuffer, &hD3D11Desc, (ID3D11UnorderedAccessView**)&m_pUnorderedAccessView );
        DebugAssert( hRes == S_OK && m_pUnorderedAccessView != NULL );

        pBackBuffer->Release();
        pBackBuffer = NULL;
    }

    m_pView = NULL;
    hRes = ((ID3D11UnorderedAccessView*)m_pUnorderedAccessView)->QueryInterface( __uuidof(ID3D11View), &m_pView );
    DebugAssert( hRes == S_OK && m_pView != NULL );
}
Void D3D11UnorderedAccessView::_NakedDestroy()
{
    ((ID3D11View*)m_pView)->Release();
    m_pView = NULL;

    ((ID3D11UnorderedAccessView*)m_pUnorderedAccessView)->Release();
    m_pUnorderedAccessView = NULL;
}

