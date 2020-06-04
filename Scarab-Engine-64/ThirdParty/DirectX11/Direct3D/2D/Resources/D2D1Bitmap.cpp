/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/2D/Resources/D2D1Bitmap.cpp
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
// Third-Party Includes
#pragma warning(disable:4005)

#define WIN32_LEAN_AND_MEAN
#include <d2d1.h>

#undef DebugAssert

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "D2D1Bitmap.h"

#include "../D2D1RenderingContext.h"

/////////////////////////////////////////////////////////////////////////////////
// D2D1Bitmaps implementation
D2D1Bitmap::D2D1Bitmap( D2D1RenderingContext * pContext2D )
{
    m_pContext2D = pContext2D;

    m_pBitmap = NULL;

    m_hDesc.iFormat = PIXEL_FMT_UNKNOWN;
    m_hDesc.iAlphaMode = D2D1BITMAP_ALPHAMODE_UNKNOWN;
    m_hDesc.fDpiX = 0.0f;
    m_hDesc.fDpiY = 0.0f;
    m_hDesc.iWidth = 0;
    m_hDesc.iHeight = 0;
    m_hDesc.fDIPWidth = 0.0f;
    m_hDesc.fDIPHeight = 0.0f;

    m_bTemporaryDestroyed = false;
    m_hCreationParameters.pData = NULL;
    m_hCreationParameters.iDataPitch = 0;
}
D2D1Bitmap::~D2D1Bitmap()
{
    if ( IsCreated() )
        Destroy();
}

Void D2D1Bitmap::Create( const D2D1BitmapDesc * pDesc, const Void * pData, UInt iDataPitch )
{
    DebugAssert( !(IsCreated()) );

    static const Float s_fInv96 = 1.0f / 96.0f;
    const Float fConvertDIPX = ( pDesc->fDpiX * s_fInv96 );
    const Float fConvertDIPY = ( pDesc->fDpiY * s_fInv96 );
    const Float fConvertInvDIPX = 1.0f / fConvertDIPX;
    const Float fConvertInvDIPY = 1.0f / fConvertDIPY;

    MemCopy( &m_hDesc, pDesc, sizeof(D2D1BitmapDesc) );

    // Use pixel or DIP dimensions, pick the non-zero one
    // Both zero or none zero is an error (no implicit preference)
    if ( m_hDesc.iWidth != 0 ) {
        DebugAssert( m_hDesc.fDIPWidth == 0.0f );
        m_hDesc.fDIPWidth = ( fConvertInvDIPX * (Float)(m_hDesc.iWidth) );
    } else {
        DebugAssert( m_hDesc.fDIPWidth != 0.0f );
        m_hDesc.iWidth = (UInt)( fConvertDIPX * m_hDesc.fDIPWidth );
    }
    if ( m_hDesc.iHeight != 0 ) {
        DebugAssert( m_hDesc.fDIPHeight == 0.0f );
        m_hDesc.fDIPHeight = ( fConvertInvDIPY * (Float)(m_hDesc.iHeight) );
    } else {
        DebugAssert( m_hDesc.fDIPHeight != 0.0f );
        m_hDesc.iHeight = (UInt)( fConvertDIPY * m_hDesc.fDIPHeight );
    }

    m_hCreationParameters.pData = pData;
    m_hCreationParameters.iDataPitch = iDataPitch;

    _NakedCreate();
}
Void D2D1Bitmap::Destroy()
{
    DebugAssert( IsCreated() );

    if ( m_bTemporaryDestroyed )
        m_bTemporaryDestroyed = false;
    else
        _NakedDestroy();

    m_hDesc.iFormat = PIXEL_FMT_UNKNOWN;
    m_hDesc.iAlphaMode = D2D1BITMAP_ALPHAMODE_UNKNOWN;
    m_hDesc.fDpiX = 0.0f;
    m_hDesc.fDpiY = 0.0f;
    m_hDesc.iWidth = 0;
    m_hDesc.iHeight = 0;
    m_hDesc.fDIPWidth = 0.0f;
    m_hDesc.fDIPHeight = 0.0f;

    m_hCreationParameters.pData = NULL;
    m_hCreationParameters.iDataPitch = 0;
}

Void D2D1Bitmap::OnDestroyDevice()
{
    DebugAssert( !m_bTemporaryDestroyed );

    if ( m_pBitmap != NULL ) {
        _NakedDestroy();
        m_bTemporaryDestroyed = true;
    }
}
Void D2D1Bitmap::OnRestoreDevice()
{
    DebugAssert( m_pBitmap == NULL );

    if ( m_bTemporaryDestroyed ) {
        _NakedCreate();
        m_bTemporaryDestroyed = false;
    }
}

Void D2D1Bitmap::CopyFrom( const Void * pSrcData, UInt iSrcPitch, const D2D1RectangleI * pDstRect )
{
    DebugAssert( IsCreated() );

    // Destination rectangle
    D2D1_RECT_U hRect;

    if ( pDstRect != NULL ) {
        DebugAssert( pDstRect->iLeft < pDstRect->iRight && pDstRect->iRight < m_hDesc.iWidth );
        DebugAssert( pDstRect->iTop < pDstRect->iBottom && pDstRect->iBottom < m_hDesc.iHeight );
        hRect.left = pDstRect->iLeft;
        hRect.top = pDstRect->iTop;
        hRect.right = pDstRect->iRight;
        hRect.bottom = pDstRect->iBottom;
    } else {
        hRect.left = 0;
        hRect.top = 0;
        hRect.right = m_hDesc.iWidth - 1;
        hRect.bottom = m_hDesc.iHeight - 1;
    }

    // Perform Copy
    HRESULT hRes = ((ID2D1Bitmap*)m_pBitmap)->CopyFromMemory( &hRect, pSrcData, iSrcPitch );
    DebugAssert( hRes == S_OK );
}
Void D2D1Bitmap::CopyFrom( D2D1Bitmap * pSrcBitmap, const D2D1PointI * pDstPoint, const D2D1RectangleI * pSrcRect )
{
    DebugAssert( IsCreated() );
    DebugAssert( pSrcBitmap->IsCreated() );

    DebugAssert( m_hDesc.iFormat == pSrcBitmap->m_hDesc.iFormat );
    DebugAssert( m_hDesc.iAlphaMode == pSrcBitmap->m_hDesc.iAlphaMode );

    // Destination point
    D2D1_POINT_2U hDstPoint;
    if ( pDstPoint != NULL ) {
        DebugAssert( pDstPoint->iX < m_hDesc.iWidth );
        DebugAssert( pDstPoint->iY < m_hDesc.iHeight );
        hDstPoint.x = pDstPoint->iX;
        hDstPoint.y = pDstPoint->iY;
    } else {
        hDstPoint.x = 0;
        hDstPoint.y = 0;
    }

    // Source rectangle
    D2D1_RECT_U hSrcRect;
    if ( pSrcRect != NULL ) {
        DebugAssert( pSrcRect->iLeft < pSrcRect->iRight && pSrcRect->iRight < pSrcBitmap->m_hDesc.iWidth );
        DebugAssert( pSrcRect->iTop < pSrcRect->iBottom && pSrcRect->iBottom < pSrcBitmap->m_hDesc.iHeight );
        hSrcRect.left = pSrcRect->iLeft;
        hSrcRect.top = pSrcRect->iTop;
        hSrcRect.right = pSrcRect->iRight;
        hSrcRect.bottom = pSrcRect->iBottom;
    } else {
        hSrcRect.left = 0;
        hSrcRect.top = 0;
        hSrcRect.right = pSrcBitmap->m_hDesc.iWidth - 1;
        hSrcRect.bottom = pSrcBitmap->m_hDesc.iHeight - 1;
    }

    // Destination rectangle
    UInt iCopyWidth = ( hSrcRect.right + 1 - hSrcRect.left );
    UInt iCopyHeight = ( hSrcRect.bottom + 1 - hSrcRect.top );
    DebugAssert( hDstPoint.x + iCopyWidth <= m_hDesc.iWidth );
    DebugAssert( hDstPoint.y + iCopyHeight <= m_hDesc.iHeight );

    // Perform Copy
    HRESULT hRes = ((ID2D1Bitmap*)m_pBitmap)->CopyFromBitmap( &hDstPoint, (ID2D1Bitmap*)(pSrcBitmap->m_pBitmap), &hSrcRect );
    DebugAssert( hRes == S_OK );
}

/////////////////////////////////////////////////////////////////////////////////

Void D2D1Bitmap::_NakedCreate()
{
    D2D1_SIZE_U hSize;
    hSize.width = m_hDesc.iWidth;
    hSize.height = m_hDesc.iHeight;

    D2D1_BITMAP_PROPERTIES hD2D1Desc;
    hD2D1Desc.pixelFormat.format = (DXGI_FORMAT)( PixelFormatToDXGI[m_hDesc.iFormat] );
    hD2D1Desc.pixelFormat.alphaMode = (D2D1_ALPHA_MODE)( D2D1BitmapAlphaModeToD2D1[m_hDesc.iAlphaMode] );
    hD2D1Desc.dpiX = m_hDesc.fDpiX;
    hD2D1Desc.dpiY = m_hDesc.fDpiY;

    m_pBitmap = NULL;
    HRESULT hRes = ((ID2D1RenderTarget*)(m_pContext2D->m_pD2D1RenderingContext))->CreateBitmap( hSize, m_hCreationParameters.pData, m_hCreationParameters.iDataPitch, &hD2D1Desc, (ID2D1Bitmap**)&m_pBitmap );
    DebugAssert( hRes == S_OK && m_pBitmap != NULL );
}
Void D2D1Bitmap::_NakedDestroy()
{
    ((ID2D1Bitmap*)m_pBitmap)->Release();
    m_pBitmap = NULL;
}
