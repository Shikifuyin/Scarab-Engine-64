/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Tools/WinGUIImage.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Images (Bitmap or Icon)
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Third-Party Includes
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "WinGUIImage.h"
#include "../WinGUI.h"

/////////////////////////////////////////////////////////////////////////////////
// WinGUIBitmap implementation
WinGUIBitmap::WinGUIBitmap()
{
	m_bIsDeviceDependant = false;
	m_bShared = false;

	m_hHandle = NULL;

	m_iDDWidth = 0;
	m_iDDHeight = 0;

	MemZero( &m_hBitmapDesc, sizeof(WinGUIBitmapDescriptor) );
	m_pBitmapData = NULL;
	m_bLocked = false;
}
WinGUIBitmap::~WinGUIBitmap()
{
	// nothing to do
}

Void WinGUIBitmap::CreateDDBitmap( UInt iWidth, UInt iHeight )
{
	DebugAssert( m_hHandle == NULL );

	HWND hAppWindow = (HWND)( _GetAppWindowHandle() );
	HDC hDC = GetDC( hAppWindow );

	m_hHandle = CreateCompatibleBitmap( hDC, iWidth, iHeight );
	DebugAssert( m_hHandle != NULL );

	ReleaseDC( hAppWindow, hDC );

	// Setup
	m_bIsDeviceDependant = true;
	m_bShared = false;

	m_iDDWidth = iWidth;
	m_iDDHeight = iHeight;
}
Void WinGUIBitmap::CreateDDBitmapMask( UInt iWidth, UInt iHeight )
{
	DebugAssert( m_hHandle == NULL );

	m_hHandle = ::CreateBitmap( iWidth, iHeight, 1, 1, NULL );
	DebugAssert( m_hHandle != NULL );

	// Setup
	m_bIsDeviceDependant = true;
	m_bShared = false;

	m_iDDWidth = iWidth;
	m_iDDHeight = iHeight;
}

Void WinGUIBitmap::CreateDIBitmap( const WinGUIBitmapDescriptor & hDescriptor )
{
	DebugAssert( m_hHandle == NULL );

	BITMAPV5HEADER hBitmapInfo;
	hBitmapInfo.bV5Size = sizeof(BITMAPV5HEADER);

	hBitmapInfo.bV5Width = hDescriptor.iWidth;
	if ( hDescriptor.bBottomUpElseTopDown )
		hBitmapInfo.bV5Height = hDescriptor.iHeight;
	else
		hBitmapInfo.bV5Height = -((LONG)(hDescriptor.iHeight));

	hBitmapInfo.bV5Planes = 1;
	hBitmapInfo.bV5BitCount = (WORD)( hDescriptor.iBPP );

	switch( hDescriptor.iCompression ) {
		case WINGUI_BITMAP_RGB:      hBitmapInfo.bV5Compression = BI_RGB; break;
		case WINGUI_BITMAP_BITFIELD: hBitmapInfo.bV5Compression = BI_BITFIELDS; break;
		case WINGUI_BITMAP_JPEG:     hBitmapInfo.bV5Compression = BI_JPEG; break;
		case WINGUI_BITMAP_PNG:      hBitmapInfo.bV5Compression = BI_PNG; break;
		default: DebugAssert(false); break;
	}
	hBitmapInfo.bV5SizeImage = hDescriptor.iByteSize;

	hBitmapInfo.bV5XPelsPerMeter = hDescriptor.iPixelsPerMeterX;
	hBitmapInfo.bV5YPelsPerMeter = hDescriptor.iPixelsPerMeterY;

	hBitmapInfo.bV5ClrUsed = 0;
	hBitmapInfo.bV5ClrImportant = 0;

	hBitmapInfo.bV5RedMask = hDescriptor.iMaskRed;
	hBitmapInfo.bV5GreenMask = hDescriptor.iMaskGreen;
	hBitmapInfo.bV5BlueMask = hDescriptor.iMaskBlue;
	hBitmapInfo.bV5AlphaMask = hDescriptor.iMaskAlpha;

	switch( hDescriptor.iColorSpace ) {
		case WINGUI_BITMAP_SRGB:       hBitmapInfo.bV5CSType = LCS_sRGB; break;
		case WINGUI_BITMAP_CALIBRATED: hBitmapInfo.bV5CSType = LCS_CALIBRATED_RGB; break;
		default: DebugAssert(false); break;
	}

	hBitmapInfo.bV5Endpoints.ciexyzRed.ciexyzX = hDescriptor.hEndPoints.Red.iFixed2_30_X;
	hBitmapInfo.bV5Endpoints.ciexyzRed.ciexyzY = hDescriptor.hEndPoints.Red.iFixed2_30_Y;
	hBitmapInfo.bV5Endpoints.ciexyzRed.ciexyzZ = hDescriptor.hEndPoints.Red.iFixed2_30_Z;

	hBitmapInfo.bV5Endpoints.ciexyzGreen.ciexyzX = hDescriptor.hEndPoints.Green.iFixed2_30_X;
	hBitmapInfo.bV5Endpoints.ciexyzGreen.ciexyzY = hDescriptor.hEndPoints.Green.iFixed2_30_Y;
	hBitmapInfo.bV5Endpoints.ciexyzGreen.ciexyzZ = hDescriptor.hEndPoints.Green.iFixed2_30_Z;

	hBitmapInfo.bV5Endpoints.ciexyzBlue.ciexyzX = hDescriptor.hEndPoints.Blue.iFixed2_30_X;
	hBitmapInfo.bV5Endpoints.ciexyzBlue.ciexyzY = hDescriptor.hEndPoints.Blue.iFixed2_30_Y;
	hBitmapInfo.bV5Endpoints.ciexyzBlue.ciexyzZ = hDescriptor.hEndPoints.Blue.iFixed2_30_Z;

	hBitmapInfo.bV5GammaRed = hDescriptor.iGammaRed;
	hBitmapInfo.bV5GammaGreen = hDescriptor.iGammaGreen;
	hBitmapInfo.bV5GammaBlue = hDescriptor.iGammaBlue;

	switch( hDescriptor.iRenderingIntent ) {
		case WINGUI_BITMAP_COLORIMETRIC_ABS: hBitmapInfo.bV5Intent = LCS_GM_ABS_COLORIMETRIC; break;
		case WINGUI_BITMAP_COLORIMETRIC_REL: hBitmapInfo.bV5Intent = LCS_GM_GRAPHICS; break;
		case WINGUI_BITMAP_SATURATION:       hBitmapInfo.bV5Intent = LCS_GM_BUSINESS; break;
		case WINGUI_BITMAP_PERCEPTUAL:       hBitmapInfo.bV5Intent = LCS_GM_IMAGES; break;
		default: DebugAssert(false); break;
	}

	hBitmapInfo.bV5ProfileData = 0;
	hBitmapInfo.bV5ProfileSize = 0;
	hBitmapInfo.bV5Reserved = 0;

	HWND hAppWindow = (HWND)( _GetAppWindowHandle() );
	HDC hDC = GetDC( hAppWindow );

	m_hHandle = CreateDIBSection( hDC, (const BITMAPINFO *)&hBitmapInfo, DIB_RGB_COLORS, (Void**)&m_pBitmapData, NULL, 0 );
	DebugAssert( m_hHandle != NULL && m_pBitmapData != NULL );

	ReleaseDC( hAppWindow, hDC );

	// Setup
	m_bIsDeviceDependant = false;
	m_bShared = false;

	m_iDDWidth = 0;
	m_iDDHeight = 0;

	MemCopy( &m_hBitmapDesc, &hDescriptor, sizeof(WinGUIBitmapDescriptor) );

	m_bLocked = false;
}

Void WinGUIBitmap::LockDIB( Byte ** ppMemory )
{
	DebugAssert( m_hHandle != NULL && !m_bIsDeviceDependant && m_pBitmapData != NULL );
	DebugAssert( !m_bLocked );

	// Synchronize
	GdiFlush();

	// Lock
	m_bLocked = true;
	*ppMemory = m_pBitmapData;
}
Void WinGUIBitmap::UnlockDIB( Byte ** ppMemory )
{
	DebugAssert( m_hHandle != NULL && !m_bIsDeviceDependant && m_pBitmapData != NULL );
	DebugAssert( m_bLocked );

	// Unlock
	*ppMemory = NULL;
	m_bLocked = false;
}

Void WinGUIBitmap::LoadFromFile( const GChar * strFilename, const WinGUIImageLoadParameters & hLoadParams )
{
	DebugAssert( m_hHandle == NULL );

	UInt iFlags;
	UInt iWidth, iHeight;

	// Flags
	iFlags = LR_LOADFROMFILE;
	if ( hLoadParams.bMakeDIB )
		iFlags |= LR_CREATEDIBSECTION;
	if ( hLoadParams.bMonochrome )
		iFlags |= LR_MONOCHROME;
	if ( hLoadParams.bTrueVGA )
		iFlags |= LR_VGACOLOR;

	switch( hLoadParams.iResizeWidth ) {
		case WINGUI_IMAGE_RESIZE_KEEP: iWidth = 0; iFlags |= 0; break;
		case WINGUI_IMAGE_RESIZE_USER: iWidth = hLoadParams.iWidth; iFlags |= 0; break;
		default: DebugAssert(false); break;
	}
	switch( hLoadParams.iResizeHeight ) {
		case WINGUI_IMAGE_RESIZE_KEEP: iHeight = 0; iFlags |= 0; break;
		case WINGUI_IMAGE_RESIZE_USER: iHeight = hLoadParams.iHeight; iFlags |= 0; break;
		default: DebugAssert(false); break;
	}

	// Load File
	HBITMAP hBMP = (HBITMAP)( LoadImage(NULL, strFilename, IMAGE_BITMAP, iWidth, iHeight, iFlags) );
	DebugAssert( hBMP != NULL );

	// Setup
	_CreateFromHandle( hBMP, !(hLoadParams.bMakeDIB), false );
}
Void WinGUIBitmap::LoadFromResource( UInt iResourceID, const WinGUIImageLoadParameters & hLoadParams )
{
	DebugAssert( m_hHandle == NULL );

	UInt iFlags;
	UInt iWidth, iHeight;

	// Obtain Application Handle
	HWND hAppWindow = (HWND)( _GetAppWindowHandle() );
	HINSTANCE hInst = (HINSTANCE)( GetWindowLongPtr(hAppWindow, GWLP_HINSTANCE) );

	// Flags
	iFlags = 0;
	if ( hLoadParams.bMakeDIB )
		iFlags |= LR_CREATEDIBSECTION;
	if ( hLoadParams.bMonochrome )
		iFlags |= LR_MONOCHROME;
	if ( hLoadParams.bTrueVGA )
		iFlags |= LR_VGACOLOR;
	if ( hLoadParams.bSharedResource )
		iFlags |= LR_SHARED;

	// Type
	switch( hLoadParams.iResizeWidth ) {
		case WINGUI_IMAGE_RESIZE_KEEP: iWidth = 0; iFlags |= 0; break;
		case WINGUI_IMAGE_RESIZE_USER: iWidth = hLoadParams.iWidth; iFlags |= 0; break;
		default: DebugAssert(false); break;
	}
	switch( hLoadParams.iResizeHeight ) {
		case WINGUI_IMAGE_RESIZE_KEEP: iHeight = 0; iFlags |= 0; break;
		case WINGUI_IMAGE_RESIZE_USER: iHeight = hLoadParams.iHeight; iFlags |= 0; break;
		default: DebugAssert(false); break;
	}

	// Load File
	HBITMAP hBMP = (HBITMAP)( LoadImage(hInst, MAKEINTRESOURCE(iResourceID), IMAGE_BITMAP, iWidth, iHeight, iFlags) );
	DebugAssert( hBMP != NULL );

	// Setup
	_CreateFromHandle( hBMP, !(hLoadParams.bMakeDIB), hLoadParams.bSharedResource );
}

Void WinGUIBitmap::Destroy()
{
	DebugAssert( m_hHandle != NULL && !m_bShared && !m_bLocked );

	m_bIsDeviceDependant = false;

	DeleteObject( m_hHandle );
	m_hHandle = NULL;

	m_iDDWidth = 0;
	m_iDDHeight = 0;

	MemZero( &m_hBitmapDesc, sizeof(WinGUIBitmapDescriptor) );
	m_pBitmapData = NULL;
	m_bLocked = false;
}

Void WinGUIBitmap::BitBlit( const WinGUIRectangle & hDestRect, const WinGUIBitmap * pSrcBitmap, const WinGUIPoint & hSrcOrigin, WinGUIRasterOperation iOperation )
{
	DebugAssert( m_hHandle != NULL && !m_bLocked );
	DebugAssert( pSrcBitmap->m_hHandle != NULL && !(pSrcBitmap->m_bLocked) );

	// Retrieve Handles
	HBITMAP hDestBitmap = (HBITMAP)m_hHandle;
	HBITMAP hSrcBitmap = (HBITMAP)(pSrcBitmap->m_hHandle);

	// Raster Operation
	DWORD iROP = _ConvertRasterOperation( iOperation );

	// Get Window DC
	HWND hAppWindow = (HWND)( _GetAppWindowHandle() );
	HDC hDC = GetDC( hAppWindow );

	// Create Memory DCs
	HDC hDestMemoryDC = CreateCompatibleDC( hDC );
	HDC hSrcMemoryDC = CreateCompatibleDC( hDC );

	// Select Bitmaps
	HBITMAP hDestSaved = (HBITMAP)SelectObject( hDestMemoryDC, hDestBitmap );
	HBITMAP hSrcSaved = (HBITMAP)SelectObject( hSrcMemoryDC, hSrcBitmap );

	// Perform Operation
	BitBlt( hDestMemoryDC, hDestRect.iLeft, hDestRect.iTop, hDestRect.iWidth, hDestRect.iHeight,
			hSrcMemoryDC, hSrcOrigin.iX, hSrcOrigin.iY,
			iROP );

	// Release All
	SelectObject( hDestMemoryDC, hDestSaved );
	SelectObject( hSrcMemoryDC, hSrcSaved );
	DeleteDC( hDestMemoryDC );
	DeleteDC( hSrcMemoryDC );
	ReleaseDC( hAppWindow, hDC );
}
Void WinGUIBitmap::StretchBlit( const WinGUIRectangle & hDestRect, const WinGUIBitmap * pSrcBitmap, const WinGUIRectangle & hSrcRect, WinGUIRasterOperation iOperation )
{
	DebugAssert( m_hHandle != NULL && !m_bLocked );
	DebugAssert( pSrcBitmap->m_hHandle != NULL && !(pSrcBitmap->m_bLocked) );

	// Retrieve Handles
	HBITMAP hDestBitmap = (HBITMAP)m_hHandle;
	HBITMAP hSrcBitmap = (HBITMAP)(pSrcBitmap->m_hHandle);

	// Raster Operation
	DWORD iROP = _ConvertRasterOperation( iOperation );

	// Get Window DC
	HWND hAppWindow = (HWND)( _GetAppWindowHandle() );
	HDC hDC = GetDC( hAppWindow );

	// Create Memory DCs
	HDC hDestMemoryDC = CreateCompatibleDC( hDC );
	HDC hSrcMemoryDC = CreateCompatibleDC( hDC );

	// Select Bitmaps
	HBITMAP hDestSaved = (HBITMAP)SelectObject( hDestMemoryDC, hDestBitmap );
	HBITMAP hSrcSaved = (HBITMAP)SelectObject( hSrcMemoryDC, hSrcBitmap );

	// Perform Operation
	StretchBlt( hDestMemoryDC, hDestRect.iLeft, hDestRect.iTop, hDestRect.iWidth, hDestRect.iHeight,
				hSrcMemoryDC, hSrcRect.iLeft, hSrcRect.iTop, hSrcRect.iWidth, hSrcRect.iHeight,
				iROP );

	// Release All
	SelectObject( hDestMemoryDC, hDestSaved );
	SelectObject( hSrcMemoryDC, hSrcSaved );
	DeleteDC( hDestMemoryDC );
	DeleteDC( hSrcMemoryDC );
	ReleaseDC( hAppWindow, hDC );
}
Void WinGUIBitmap::MaskBlit( const WinGUIRectangle & hDestRect, const WinGUIBitmap * pSrcBitmap, const WinGUIPoint & hSrcOrigin,
						 	 const WinGUIBitmap * pMask, const WinGUIPoint & hMaskOrigin,
							 WinGUIRasterOperation iForegroundOP, WinGUIRasterOperation iBackgroundOP )
{
	DebugAssert( m_hHandle != NULL && !m_bLocked );
	DebugAssert( pSrcBitmap->m_hHandle != NULL && !(pSrcBitmap->m_bLocked) );
	DebugAssert( pMask->m_hHandle != NULL && !(pMask->m_bLocked) );

	// Retrieve Handles
	HBITMAP hDestBitmap = (HBITMAP)m_hHandle;
	HBITMAP hSrcBitmap = (HBITMAP)(pSrcBitmap->m_hHandle);
	HBITMAP hMaskBitmap = (HBITMAP)(pMask->m_hHandle);

	// Raster Operations
	DWORD iForegroundROP = _ConvertRasterOperation( iForegroundOP );
	DWORD iBackgroundROP = _ConvertRasterOperation( iBackgroundOP );
	DWORD iROP = MAKEROP4( iForegroundROP, iBackgroundROP );

	// Get Window DC
	HWND hAppWindow = (HWND)( _GetAppWindowHandle() );
	HDC hDC = GetDC( hAppWindow );

	// Create Memory DCs
	HDC hDestMemoryDC = CreateCompatibleDC( hDC );
	HDC hSrcMemoryDC = CreateCompatibleDC( hDC );

	// Select Bitmaps
	HBITMAP hDestSaved = (HBITMAP)SelectObject( hDestMemoryDC, hDestBitmap );
	HBITMAP hSrcSaved = (HBITMAP)SelectObject( hSrcMemoryDC, hSrcBitmap );

	// Perform Operation
	MaskBlt( hDestMemoryDC, hDestRect.iLeft, hDestRect.iTop, hDestRect.iWidth, hDestRect.iHeight,
			 hSrcMemoryDC, hSrcOrigin.iX, hSrcOrigin.iY,
			 hMaskBitmap, hMaskOrigin.iX, hMaskOrigin.iY,
			 iROP );

	// Release All
	SelectObject( hDestMemoryDC, hDestSaved );
	SelectObject( hSrcMemoryDC, hSrcSaved );
	DeleteDC( hDestMemoryDC );
	DeleteDC( hSrcMemoryDC );
	ReleaseDC( hAppWindow, hDC );
}
Void WinGUIBitmap::TransparentBlit( const WinGUIRectangle & hDestRect, const WinGUIBitmap * pSrcBitmap, const WinGUIRectangle & hSrcRect, UInt iKeyColor )
{
	DebugAssert( m_hHandle != NULL && !m_bLocked );
	DebugAssert( pSrcBitmap->m_hHandle != NULL && !(pSrcBitmap->m_bLocked) );

	// Retrieve Handles
	HBITMAP hDestBitmap = (HBITMAP)m_hHandle;
	HBITMAP hSrcBitmap = (HBITMAP)(pSrcBitmap->m_hHandle);

	// Get Window DC
	HWND hAppWindow = (HWND)( _GetAppWindowHandle() );
	HDC hDC = GetDC( hAppWindow );

	// Create Memory DCs
	HDC hDestMemoryDC = CreateCompatibleDC( hDC );
	HDC hSrcMemoryDC = CreateCompatibleDC( hDC );

	// Select Bitmaps
	HBITMAP hDestSaved = (HBITMAP)SelectObject( hDestMemoryDC, hDestBitmap );
	HBITMAP hSrcSaved = (HBITMAP)SelectObject( hSrcMemoryDC, hSrcBitmap );

	// Perform Operation
	TransparentBlt( hDestMemoryDC, hDestRect.iLeft, hDestRect.iTop, hDestRect.iWidth, hDestRect.iHeight,
					hSrcMemoryDC, hSrcRect.iLeft, hSrcRect.iTop, hSrcRect.iWidth, hSrcRect.iHeight,
					iKeyColor );

	// Release All
	SelectObject( hDestMemoryDC, hDestSaved );
	SelectObject( hSrcMemoryDC, hSrcSaved );
	DeleteDC( hDestMemoryDC );
	DeleteDC( hSrcMemoryDC );
	ReleaseDC( hAppWindow, hDC );
}

Void WinGUIBitmap::Render( WinGUIElement * pTarget, const WinGUIRectangle & hDestRect, const WinGUIPoint & hSrcOrigin, WinGUIRasterOperation iOperation )
{
	DebugAssert( m_hHandle != NULL && !m_bLocked );

	// Retrieve Handles
	HWND hTargetWnd = NULL;
	switch( pTarget->GetElementType() ) {
		case WINGUI_ELEMENT_WINDOW:    hTargetWnd = (HWND)( WinGUIElement::_GetHandle(pTarget) ); break;
		case WINGUI_ELEMENT_CONTAINER: hTargetWnd = (HWND)( WinGUIElement::_GetHandle(pTarget) ); break;
		default: DebugAssert(false); break;
	}

	HBITMAP hSrcBitmap = (HBITMAP)m_hHandle;

	// Raster Operation
	DWORD iROP = _ConvertRasterOperation( iOperation );

	// Get Window DC
	HDC hDC = GetDC( hTargetWnd );

	// Create Memory DCs
	HDC hSrcMemoryDC = CreateCompatibleDC( hDC );

	// Select Bitmaps
	HBITMAP hSrcSaved = (HBITMAP)SelectObject( hSrcMemoryDC, hSrcBitmap );

	// Perform Operation
	BitBlt( hDC, hDestRect.iLeft, hDestRect.iTop, hDestRect.iWidth, hDestRect.iHeight,
			hSrcMemoryDC, hSrcOrigin.iX, hSrcOrigin.iY,
			iROP );

	// Release All
	SelectObject( hSrcMemoryDC, hSrcSaved );
	DeleteDC( hSrcMemoryDC );
	ReleaseDC( hTargetWnd, hDC );
}
Void WinGUIBitmap::StretchRender( WinGUIElement * pTarget, const WinGUIRectangle & hDestRect, const WinGUIRectangle & hSrcRect, WinGUIRasterOperation iOperation )
{
	DebugAssert( m_hHandle != NULL && !m_bLocked );

	// Retrieve Handles
	HWND hTargetWnd = NULL;
	switch( pTarget->GetElementType() ) {
		case WINGUI_ELEMENT_WINDOW:    hTargetWnd = (HWND)( WinGUIElement::_GetHandle(pTarget) ); break;
		case WINGUI_ELEMENT_CONTAINER: hTargetWnd = (HWND)( WinGUIElement::_GetHandle(pTarget) ); break;
		default: DebugAssert(false); break;
	}

	HBITMAP hSrcBitmap = (HBITMAP)m_hHandle;

	// Raster Operation
	DWORD iROP = _ConvertRasterOperation( iOperation );

	// Get Window DC
	HDC hDC = GetDC( hTargetWnd );

	// Create Memory DCs
	HDC hSrcMemoryDC = CreateCompatibleDC( hDC );

	// Select Bitmaps
	HBITMAP hSrcSaved = (HBITMAP)SelectObject( hSrcMemoryDC, hSrcBitmap );

	// Perform Operation
	StretchBlt( hDC, hDestRect.iLeft, hDestRect.iTop, hDestRect.iWidth, hDestRect.iHeight,
				hSrcMemoryDC, hSrcRect.iLeft, hSrcRect.iTop, hSrcRect.iWidth, hSrcRect.iHeight,
				iROP );

	// Release All
	SelectObject( hSrcMemoryDC, hSrcSaved );
	DeleteDC( hSrcMemoryDC );
	ReleaseDC( hTargetWnd, hDC );
}
Void WinGUIBitmap::MaskRender( WinGUIElement * pTarget, const WinGUIRectangle & hDestRect, const WinGUIPoint & hSrcOrigin,
							  const WinGUIBitmap * pMask, const WinGUIPoint & hMaskOrigin,
							  WinGUIRasterOperation iForegroundOP, WinGUIRasterOperation iBackgroundOP )
{
	DebugAssert( m_hHandle != NULL && !m_bLocked );
	DebugAssert( pMask->m_hHandle != NULL && !(pMask->m_bLocked) );

	// Retrieve Handles
	HWND hTargetWnd = NULL;
	switch( pTarget->GetElementType() ) {
		case WINGUI_ELEMENT_WINDOW:    hTargetWnd = (HWND)( WinGUIElement::_GetHandle(pTarget) ); break;
		case WINGUI_ELEMENT_CONTAINER: hTargetWnd = (HWND)( WinGUIElement::_GetHandle(pTarget) ); break;
		default: DebugAssert(false); break;
	}

	HBITMAP hSrcBitmap = (HBITMAP)m_hHandle;
	HBITMAP hMaskBitmap = (HBITMAP)(pMask->m_hHandle);

	// Raster Operations
	DWORD iForegroundROP = _ConvertRasterOperation( iForegroundOP );
	DWORD iBackgroundROP = _ConvertRasterOperation( iBackgroundOP );
	DWORD iROP = MAKEROP4( iForegroundROP, iBackgroundROP );

	// Get Window DC
	HDC hDC = GetDC( hTargetWnd );

	// Create Memory DCs
	HDC hSrcMemoryDC = CreateCompatibleDC( hDC );

	// Select Bitmaps
	HBITMAP hSrcSaved = (HBITMAP)SelectObject( hSrcMemoryDC, hSrcBitmap );

	// Perform Operation
	MaskBlt( hDC, hDestRect.iLeft, hDestRect.iTop, hDestRect.iWidth, hDestRect.iHeight,
			 hSrcMemoryDC, hSrcOrigin.iX, hSrcOrigin.iY,
			 hMaskBitmap, hMaskOrigin.iX, hMaskOrigin.iY,
			 iROP );

	// Release All
	SelectObject( hSrcMemoryDC, hSrcSaved );
	DeleteDC( hSrcMemoryDC );
	ReleaseDC( hTargetWnd, hDC );
}
Void WinGUIBitmap::TransparentRender( WinGUIElement * pTarget, const WinGUIRectangle & hDestRect, const WinGUIRectangle & hSrcRect, UInt iKeyColor )
{
	DebugAssert( m_hHandle != NULL && !m_bLocked );

	// Retrieve Handles
	HWND hTargetWnd = NULL;
	switch( pTarget->GetElementType() ) {
		case WINGUI_ELEMENT_WINDOW:    hTargetWnd = (HWND)( WinGUIElement::_GetHandle(pTarget) ); break;
		case WINGUI_ELEMENT_CONTAINER: hTargetWnd = (HWND)( WinGUIElement::_GetHandle(pTarget) ); break;
		default: DebugAssert(false); break;
	}

	HBITMAP hSrcBitmap = (HBITMAP)m_hHandle;

	// Get Window DC
	HDC hDC = GetDC( hTargetWnd );

	// Create Memory DCs
	HDC hSrcMemoryDC = CreateCompatibleDC( hDC );

	// Select Bitmaps
	HBITMAP hSrcSaved = (HBITMAP)SelectObject( hSrcMemoryDC, hSrcBitmap );

	// Perform Operation
	TransparentBlt( hDC, hDestRect.iLeft, hDestRect.iTop, hDestRect.iWidth, hDestRect.iHeight,
					hSrcMemoryDC, hSrcRect.iLeft, hSrcRect.iTop, hSrcRect.iWidth, hSrcRect.iHeight,
					iKeyColor );

	// Release All
	SelectObject( hSrcMemoryDC, hSrcSaved );
	DeleteDC( hSrcMemoryDC );
	ReleaseDC( hTargetWnd, hDC );
}

/////////////////////////////////////////////////////////////////////////////////

DWord WinGUIBitmap::_ConvertRasterOperation( WinGUIRasterOperation iROP )
{
	switch( iROP ) {
		case WINGUI_RASTER_BLACK:             return BLACKNESS; break;
		case WINGUI_RASTER_WHITE:             return WHITENESS; break;
		case WINGUI_RASTER_COPY:              return SRCCOPY; break;
		case WINGUI_RASTER_NOTDST:            return DSTINVERT; break;
		case WINGUI_RASTER_NOTSRC:            return NOTSRCCOPY; break;
		case WINGUI_RASTER_AND:               return SRCAND; break;
		case WINGUI_RASTER_OR:                return SRCPAINT; break;
		case WINGUI_RASTER_XOR:               return SRCINVERT; break;
		case WINGUI_RASTER_NOTDST_AND_SRC:    return SRCERASE; break;
		case WINGUI_RASTER_DST_OR_NOTSRC:     return MERGEPAINT; break;
		case WINGUI_RASTER_NOTDST_AND_NOTSRC: return NOTSRCERASE; break;
		default: DebugAssert(false); break;
	};
	return 0;
}

Void * WinGUIBitmap::_GetAppWindowHandle()
{
	WinGUIWindow * pAppWindow = WinGUIFn->GetAppWindow();
	return WinGUIElement::_GetHandle(pAppWindow);
}
Void WinGUIBitmap::_CreateFromHandle( Void * hHandle, Bool bDeviceDependant, Bool bShared )
{
	DebugAssert( m_hHandle == NULL );

	m_hHandle = hHandle;
	DebugAssert( m_hHandle != NULL );

	// Setup
	m_bIsDeviceDependant = bDeviceDependant;
	m_bShared = bShared;

	m_iDDWidth = 0;
	m_iDDHeight = 0;
	if ( m_bIsDeviceDependant ) {
		BITMAP hBMP;
		GetObject( m_hHandle, sizeof(BITMAP), &hBMP );

		m_iDDWidth = hBMP.bmWidth;
		m_iDDHeight = hBMP.bmHeight;
	}

	MemZero( &m_hBitmapDesc, sizeof(WinGUIBitmapDescriptor) );
	m_pBitmapData = NULL;
	m_bLocked = false;

	if ( !m_bIsDeviceDependant ) {
		DIBSECTION hDIBSection;
		GetObject( m_hHandle, sizeof(DIBSECTION), &hDIBSection );

		hDIBSection.dsBmih.biWidth;

		m_hBitmapDesc.bBottomUpElseTopDown = ( hDIBSection.dsBmih.biHeight > 0 );
		m_hBitmapDesc.iWidth = hDIBSection.dsBmih.biWidth;
		m_hBitmapDesc.iHeight = hDIBSection.dsBmih.biHeight;
		m_hBitmapDesc.iBPP = (WinGUIBitmapBPP)( hDIBSection.dsBmih.biBitCount );

		switch( hDIBSection.dsBmih.biCompression ) {
			case BI_RGB:       m_hBitmapDesc.iCompression = WINGUI_BITMAP_RGB; break;
			case BI_BITFIELDS: m_hBitmapDesc.iCompression = WINGUI_BITMAP_BITFIELD; break;
			case BI_JPEG:      m_hBitmapDesc.iCompression = WINGUI_BITMAP_JPEG; break;
			case BI_PNG:       m_hBitmapDesc.iCompression = WINGUI_BITMAP_PNG; break;
			default: DebugAssert(false); break;
		}

		m_hBitmapDesc.iByteSize = hDIBSection.dsBmih.biSizeImage;
		m_hBitmapDesc.iPixelsPerMeterX = hDIBSection.dsBmih.biXPelsPerMeter;
		m_hBitmapDesc.iPixelsPerMeterY = hDIBSection.dsBmih.biYPelsPerMeter;
		m_hBitmapDesc.iMaskRed = hDIBSection.dsBitfields[0];
		m_hBitmapDesc.iMaskGreen = hDIBSection.dsBitfields[1];
		m_hBitmapDesc.iMaskBlue = hDIBSection.dsBitfields[2];
		m_hBitmapDesc.iMaskAlpha = 0;
		m_hBitmapDesc.iColorSpace = WINGUI_BITMAP_SRGB;
		m_hBitmapDesc.hEndPoints.Red.iFixed2_30_X = 0;
		m_hBitmapDesc.hEndPoints.Red.iFixed2_30_Y = 0;
		m_hBitmapDesc.hEndPoints.Red.iFixed2_30_Z = 0;
		m_hBitmapDesc.hEndPoints.Green.iFixed2_30_X = 0;
		m_hBitmapDesc.hEndPoints.Green.iFixed2_30_Y = 0;
		m_hBitmapDesc.hEndPoints.Green.iFixed2_30_Z = 0;
		m_hBitmapDesc.hEndPoints.Blue.iFixed2_30_X = 0;
		m_hBitmapDesc.hEndPoints.Blue.iFixed2_30_Y = 0;
		m_hBitmapDesc.hEndPoints.Blue.iFixed2_30_Z = 0;
		m_hBitmapDesc.iGammaRed = 0;
		m_hBitmapDesc.iGammaGreen = 0;
		m_hBitmapDesc.iGammaBlue = 0;
		m_hBitmapDesc.iRenderingIntent = WINGUI_BITMAP_COLORIMETRIC_ABS;

		m_pBitmapData = (Byte*)( hDIBSection.dsBm.bmBits );
	}
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIIcon implementation
WinGUIIcon::WinGUIIcon():
	m_hBitmapColor(), m_hBitmapMask()
{
	m_bShared = false;

	m_hHandle = NULL;

	m_hHotSpot.iX = 0;
	m_hHotSpot.iY = 0;
}
WinGUIIcon::~WinGUIIcon()
{
	// nothing to do
}

Void WinGUIIcon::Create( const WinGUIBitmap * pBitmapColor, const WinGUIBitmap * pBitmapMask, const WinGUIPoint & hHotSpot )
{
	DebugAssert( m_hHandle == NULL );
	DebugAssert( pBitmapColor->IsCreated() && pBitmapMask->IsCreated() );
	DebugAssert( pBitmapColor->IsDeviceDependant() && pBitmapMask->IsDeviceDependant() );

	// Obtain Application Handle
	HWND hAppWindow = (HWND)( _GetAppWindowHandle() );
	HINSTANCE hInst = (HINSTANCE)( GetWindowLongPtr(hAppWindow, GWLP_HINSTANCE) );

	// Check Required dimensions
	UInt iWidth = GetSystemMetrics( SM_CXICON );
	UInt iHeight = GetSystemMetrics( SM_CYICON );
	DebugAssert( iWidth == pBitmapColor->GetDDWidth() && iWidth == pBitmapMask->GetDDWidth() );
	DebugAssert( iHeight == pBitmapColor->GetDDHeight() && iHeight == pBitmapMask->GetDDHeight() );

	// Create Icon
	ICONINFO hIconInfo;
	hIconInfo.fIcon = TRUE;
	hIconInfo.xHotspot = hHotSpot.iX;
	hIconInfo.yHotspot = hHotSpot.iY;
	hIconInfo.hbmColor = (HBITMAP)( pBitmapColor->m_hHandle );
	hIconInfo.hbmMask = (HBITMAP)( pBitmapMask->m_hHandle );

	HICON hIcon = CreateIconIndirect( &hIconInfo );
	DebugAssert( hIcon != NULL );

	_CreateFromHandle( hIcon, false );
}
Void WinGUIIcon::Destroy()
{
	DebugAssert( m_hHandle != NULL && !m_bShared );

	m_hBitmapColor.Destroy();
	m_hBitmapMask.Destroy();

	DestroyIcon( (HICON)m_hHandle );
	m_hHandle = NULL;

	m_hHotSpot.iX = 0;
	m_hHotSpot.iY = 0;
}

Void WinGUIIcon::LoadFromFile( const GChar * strFilename, const WinGUIImageLoadParameters & hLoadParams )
{
	DebugAssert( m_hHandle == NULL );

	UInt iFlags;
	UInt iWidth, iHeight;

	// Flags
	iFlags = LR_LOADFROMFILE;
	if ( hLoadParams.bMonochrome )
		iFlags |= LR_MONOCHROME;
	if ( hLoadParams.bTrueVGA )
		iFlags |= LR_VGACOLOR;

	switch( hLoadParams.iResizeWidth ) {
		case WINGUI_IMAGE_RESIZE_KEEP:    iWidth = 0; iFlags |= 0; break;
		case WINGUI_IMAGE_RESIZE_DEFAULT: iWidth = 0; iFlags |= LR_DEFAULTSIZE; break;
		case WINGUI_IMAGE_RESIZE_USER:    iWidth = hLoadParams.iWidth; iFlags |= 0; break;
		default: DebugAssert(false); break;
	}
	switch( hLoadParams.iResizeHeight ) {
		case WINGUI_IMAGE_RESIZE_KEEP:    iHeight = 0; iFlags |= 0; break;
		case WINGUI_IMAGE_RESIZE_DEFAULT: iHeight = 0; iFlags |= LR_DEFAULTSIZE; break;
		case WINGUI_IMAGE_RESIZE_USER:    iHeight = hLoadParams.iHeight; iFlags |= 0; break;
		default: DebugAssert(false); break;
	}

	// Load File
	HICON hIcon = (HICON)( LoadImage(NULL, strFilename, IMAGE_ICON, iWidth, iHeight, iFlags) );
	DebugAssert( hIcon != NULL );

	// Setup
	_CreateFromHandle( hIcon, false );
}
Void WinGUIIcon::LoadFromResource( UInt iResourceID, const WinGUIImageLoadParameters & hLoadParams )
{
	DebugAssert( m_hHandle == NULL );

	UInt iFlags;
	UInt iWidth, iHeight;

	// Obtain Application Handle
	HWND hAppWindow = (HWND)( _GetAppWindowHandle() );
	HINSTANCE hInst = (HINSTANCE)( GetWindowLongPtr(hAppWindow, GWLP_HINSTANCE) );

	// Flags
	iFlags = 0;
	if ( hLoadParams.bMonochrome )
		iFlags |= LR_MONOCHROME;
	if ( hLoadParams.bTrueVGA )
		iFlags |= LR_VGACOLOR;
	if ( hLoadParams.bSharedResource )
		iFlags |= LR_SHARED;

	switch( hLoadParams.iResizeWidth ) {
		case WINGUI_IMAGE_RESIZE_KEEP:    iWidth = 0; iFlags |= 0; break;
		case WINGUI_IMAGE_RESIZE_DEFAULT: iWidth = 0; iFlags |= LR_DEFAULTSIZE; break;
		case WINGUI_IMAGE_RESIZE_USER:    iWidth = hLoadParams.iWidth; iFlags |= 0; break;
		default: DebugAssert(false); break;
	}
	switch( hLoadParams.iResizeHeight ) {
		case WINGUI_IMAGE_RESIZE_KEEP:    iHeight = 0; iFlags |= 0; break;
		case WINGUI_IMAGE_RESIZE_DEFAULT: iHeight = 0; iFlags |= LR_DEFAULTSIZE; break;
		case WINGUI_IMAGE_RESIZE_USER:    iHeight = hLoadParams.iHeight; iFlags |= 0; break;
		default: DebugAssert(false); break;
	}

	// Load File
	HICON hIcon = (HICON)( LoadImage(hInst, MAKEINTRESOURCE(iResourceID), IMAGE_ICON, iWidth, iHeight, iFlags) );
	DebugAssert( hIcon != NULL );

	// Setup
	_CreateFromHandle( hIcon, hLoadParams.bSharedResource );
}

/////////////////////////////////////////////////////////////////////////////////

Void * WinGUIIcon::_GetAppWindowHandle() const
{
	WinGUIWindow * pAppWindow = WinGUIFn->GetAppWindow();
	return WinGUIElement::_GetHandle(pAppWindow);
}
Void WinGUIIcon::_CreateFromHandle( Void * hHandle, Bool bShared )
{
	DebugAssert( m_hHandle == NULL );

	m_hHandle = hHandle;
	DebugAssert( m_hHandle != NULL );

	// Setup
	m_bShared = bShared;

	ICONINFOEX hIconInfos;
	GetIconInfoEx( (HICON)m_hHandle, &hIconInfos );
	DebugAssert( hIconInfos.fIcon == TRUE );

	m_hHotSpot.iX = hIconInfos.xHotspot;
	m_hHotSpot.iY = hIconInfos.yHotspot;

	m_hBitmapColor._CreateFromHandle( hIconInfos.hbmColor, true, bShared );
	m_hBitmapMask._CreateFromHandle( hIconInfos.hbmMask, true, bShared );
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUICursor implementation
WinGUICursor::WinGUICursor():
	m_hBitmapColor(), m_hBitmapMask()
{
	m_bShared = false;

	m_hHandle = NULL;

	m_hHotSpot.iX = 0;
	m_hHotSpot.iY = 0;
}
WinGUICursor::~WinGUICursor()
{
	// nothing to do
}

Void WinGUICursor::Create( const WinGUIBitmap * pBitmapColor, const WinGUIBitmap * pBitmapMask, const WinGUIPoint & hHotSpot )
{
	DebugAssert( m_hHandle == NULL );
	DebugAssert( pBitmapColor->IsCreated() && pBitmapMask->IsCreated() );
	DebugAssert( pBitmapColor->IsDeviceDependant() && pBitmapMask->IsDeviceDependant() );

	// Obtain Application Handle
	HWND hAppWindow = (HWND)( _GetAppWindowHandle() );
	HINSTANCE hInst = (HINSTANCE)( GetWindowLongPtr(hAppWindow, GWLP_HINSTANCE) );

	// Check Required dimensions
	UInt iWidth = GetSystemMetrics( SM_CXCURSOR );
	UInt iHeight = GetSystemMetrics( SM_CYCURSOR );
	DebugAssert( iWidth == pBitmapColor->GetDDWidth() && iWidth == pBitmapMask->GetDDWidth() );
	DebugAssert( iHeight == pBitmapColor->GetDDHeight() && iHeight == pBitmapMask->GetDDHeight() );

	// Create Cursor
	ICONINFO hIconInfo;
	hIconInfo.fIcon = FALSE;
	hIconInfo.xHotspot = hHotSpot.iX;
	hIconInfo.yHotspot = hHotSpot.iY;
	hIconInfo.hbmColor = (HBITMAP)( pBitmapColor->m_hHandle );
	hIconInfo.hbmMask = (HBITMAP)( pBitmapMask->m_hHandle );

	HCURSOR hCursor = CreateIconIndirect( &hIconInfo );
	DebugAssert( hCursor != NULL );

	_CreateFromHandle( hCursor, false );
}
Void WinGUICursor::Destroy()
{
	DebugAssert( m_hHandle != NULL && !m_bShared );

	m_hBitmapColor.Destroy();
	m_hBitmapMask.Destroy();

	DestroyCursor( (HCURSOR)m_hHandle );
	m_hHandle = NULL;

	m_hHotSpot.iX = 0;
	m_hHotSpot.iY = 0;
}

Void WinGUICursor::LoadFromFile( const GChar * strFilename, const WinGUIImageLoadParameters & hLoadParams )
{
	DebugAssert( m_hHandle == NULL );

	UInt iFlags;
	UInt iWidth, iHeight;

	// Flags
	iFlags = LR_LOADFROMFILE;
	if ( hLoadParams.bMonochrome )
		iFlags |= LR_MONOCHROME;
	if ( hLoadParams.bTrueVGA )
		iFlags |= LR_VGACOLOR;

	switch( hLoadParams.iResizeWidth ) {
		case WINGUI_IMAGE_RESIZE_KEEP:    iWidth = 0; iFlags |= 0; break;
		case WINGUI_IMAGE_RESIZE_DEFAULT: iWidth = 0; iFlags |= LR_DEFAULTSIZE; break;
		case WINGUI_IMAGE_RESIZE_USER:    iWidth = hLoadParams.iWidth; iFlags |= 0; break;
		default: DebugAssert(false); break;
	}
	switch( hLoadParams.iResizeHeight ) {
		case WINGUI_IMAGE_RESIZE_KEEP:    iHeight = 0; iFlags |= 0; break;
		case WINGUI_IMAGE_RESIZE_DEFAULT: iHeight = 0; iFlags |= LR_DEFAULTSIZE; break;
		case WINGUI_IMAGE_RESIZE_USER:    iHeight = hLoadParams.iHeight; iFlags |= 0; break;
		default: DebugAssert(false); break;
	}

	// Load File
	HCURSOR hCursor = (HCURSOR)( LoadImage(NULL, strFilename, IMAGE_CURSOR, iWidth, iHeight, iFlags) );
	DebugAssert( hCursor != NULL );

	// Setup
	_CreateFromHandle( hCursor, false );
}
Void WinGUICursor::LoadFromResource( UInt iResourceID, const WinGUIImageLoadParameters & hLoadParams )
{
	DebugAssert( m_hHandle == NULL );

	UInt iFlags;
	UInt iWidth, iHeight;

	// Obtain Application Handle
	HWND hAppWindow = (HWND)( _GetAppWindowHandle() );
	HINSTANCE hInst = (HINSTANCE)( GetWindowLongPtr(hAppWindow, GWLP_HINSTANCE) );

	// Flags
	iFlags = 0;
	if ( hLoadParams.bMonochrome )
		iFlags |= LR_MONOCHROME;
	if ( hLoadParams.bTrueVGA )
		iFlags |= LR_VGACOLOR;
	if ( hLoadParams.bSharedResource )
		iFlags |= LR_SHARED;

	switch( hLoadParams.iResizeWidth ) {
		case WINGUI_IMAGE_RESIZE_KEEP:    iWidth = 0; iFlags |= 0; break;
		case WINGUI_IMAGE_RESIZE_DEFAULT: iWidth = 0; iFlags |= LR_DEFAULTSIZE; break;
		case WINGUI_IMAGE_RESIZE_USER:    iWidth = hLoadParams.iWidth; iFlags |= 0; break;
		default: DebugAssert(false); break;
	}
	switch( hLoadParams.iResizeHeight ) {
		case WINGUI_IMAGE_RESIZE_KEEP:    iHeight = 0; iFlags |= 0; break;
		case WINGUI_IMAGE_RESIZE_DEFAULT: iHeight = 0; iFlags |= LR_DEFAULTSIZE; break;
		case WINGUI_IMAGE_RESIZE_USER:    iHeight = hLoadParams.iHeight; iFlags |= 0; break;
		default: DebugAssert(false); break;
	}

	// Load File
	HCURSOR hCursor = (HCURSOR)( LoadImage(hInst, MAKEINTRESOURCE(iResourceID), IMAGE_CURSOR, iWidth, iHeight, iFlags) );
	DebugAssert( hCursor != NULL );

	// Setup
	_CreateFromHandle( hCursor, hLoadParams.bSharedResource );
}

/////////////////////////////////////////////////////////////////////////////////

Void * WinGUICursor::_GetAppWindowHandle() const
{
	WinGUIWindow * pAppWindow = WinGUIFn->GetAppWindow();
	return WinGUIElement::_GetHandle(pAppWindow);
}
Void WinGUICursor::_CreateFromHandle( Void * hHandle, Bool bShared )
{
	DebugAssert( m_hHandle == NULL );

	m_hHandle = hHandle;
	DebugAssert( m_hHandle != NULL );

	// Setup
	m_bShared = bShared;

	ICONINFOEX hIconInfos;
	GetIconInfoEx( (HCURSOR)m_hHandle, &hIconInfos );
	DebugAssert( hIconInfos.fIcon == FALSE );

	m_hHotSpot.iX = hIconInfos.xHotspot;
	m_hHotSpot.iY = hIconInfos.yHotspot;

	m_hBitmapColor._CreateFromHandle( hIconInfos.hbmColor, true, bShared );
	m_hBitmapMask._CreateFromHandle( hIconInfos.hbmMask, true, bShared );
}
