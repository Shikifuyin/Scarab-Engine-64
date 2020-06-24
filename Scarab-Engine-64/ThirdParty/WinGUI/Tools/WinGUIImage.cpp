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
// WinGUIImage implementation
WinGUIImage::WinGUIImage()
{
	m_bIsDeviceDependant = false;
	m_bShared = false;

	m_hHandle = NULL;

	m_iType = WINGUI_IMAGE_BITMAP;
	m_iDDWidth = 0;
	m_iDDHeight = 0;

	MemZero( &m_hBitmapDesc, sizeof(WinGUIBitmapDescriptor) );
	m_pBitmapData = NULL;
	m_bLocked = false;
}
WinGUIImage::~WinGUIImage()
{
	// nothing to do
}

Void WinGUIImage::CreateDDBitmap( UInt iWidth, UInt iHeight )
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

	m_iType = WINGUI_IMAGE_BITMAP;
	m_iDDWidth = iWidth;
	m_iDDHeight = iHeight;
}
Void WinGUIImage::CreateDDBitmapMask( UInt iWidth, UInt iHeight )
{
	DebugAssert( m_hHandle == NULL );

	m_hHandle = ::CreateBitmap( iWidth, iHeight, 1, 1, NULL );
	DebugAssert( m_hHandle != NULL );

	// Setup
	m_bIsDeviceDependant = true;
	m_bShared = false;

	m_iType = WINGUI_IMAGE_BITMAP;
	m_iDDWidth = iWidth;
	m_iDDHeight = iHeight;
}

Void WinGUIImage::CreateDIBitmap( const WinGUIBitmapDescriptor & hDescriptor )
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

	m_iType = WINGUI_IMAGE_BITMAP;
	m_iDDWidth = 0;
	m_iDDHeight = 0;

	MemCopy( &m_hBitmapDesc, &hDescriptor, sizeof(WinGUIBitmapDescriptor) );

	m_bLocked = false;
}

Void WinGUIImage::LockDIB( Byte ** ppMemory )
{
	DebugAssert( m_hHandle != NULL && !m_bIsDeviceDependant && m_pBitmapData != NULL );
	DebugAssert( !m_bLocked );

	// Synchronize
	GdiFlush();

	// Lock
	m_bLocked = true;
	*ppMemory = m_pBitmapData;
}
Void WinGUIImage::UnlockDIB( Byte ** ppMemory )
{
	DebugAssert( m_hHandle != NULL && !m_bIsDeviceDependant && m_pBitmapData != NULL );
	DebugAssert( m_bLocked );

	// Unlock
	*ppMemory = NULL;
	m_bLocked = false;
}

Void WinGUIImage::LoadFromFile( const GChar * strFilename, const WinGUIImageLoadParameters & hLoadParams )
{
	DebugAssert( m_hHandle == NULL );

	UInt iType, iFlags;
	UInt iWidth, iHeight;

	// Flags
	iFlags = LR_LOADFROMFILE;
	if ( hLoadParams.bMakeDIB )
		iFlags |= LR_CREATEDIBSECTION;
	if ( hLoadParams.bMonochrome )
		iFlags |= LR_MONOCHROME;
	if ( hLoadParams.bTrueVGA )
		iFlags |= LR_VGACOLOR;

	// Type
	switch( hLoadParams.iType ) {
		case WINGUI_IMAGE_BITMAP:
			iType = IMAGE_BITMAP;
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
			break;
		case WINGUI_IMAGE_ICON:
			iType = IMAGE_ICON;
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
			break;
		case WINGUI_IMAGE_CURSOR:
			iType = IMAGE_CURSOR;
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
			break;
		default: DebugAssert(false); break;
	}

	// Load File
	m_hHandle = LoadImage( NULL, strFilename, iType, iWidth, iHeight, iFlags );
	DebugAssert( m_hHandle != NULL );

	// Setup
	m_bIsDeviceDependant = !(hLoadParams.bMakeDIB);
	m_bShared = false;

	m_iType = hLoadParams.iType;
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

	// Retrieve DIB Infos
	if ( hLoadParams.bMakeDIB ) {
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

	m_bLocked = false;
}
Void WinGUIImage::LoadFromResource( UInt iResourceID, const WinGUIImageLoadParameters & hLoadParams )
{
	DebugAssert( m_hHandle == NULL );

	UInt iType, iFlags;
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
	switch( hLoadParams.iType ) {
		case WINGUI_IMAGE_BITMAP:
			iType = IMAGE_BITMAP;
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
			break;
		case WINGUI_IMAGE_ICON:
			iType = IMAGE_ICON;
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
			break;
		case WINGUI_IMAGE_CURSOR:
			iType = IMAGE_CURSOR;
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
			break;
		default: DebugAssert(false); break;
	}

	// Load File
	m_hHandle = LoadImage( hInst, MAKEINTRESOURCE(iResourceID), iType, iWidth, iHeight, iFlags );
	DebugAssert( m_hHandle != NULL );

	// Setup
	m_bIsDeviceDependant = !(hLoadParams.bMakeDIB);
	m_bShared = hLoadParams.bSharedResource;

	m_iType = hLoadParams.iType;
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

	// Retrieve DIB Infos
	if ( hLoadParams.bMakeDIB ) {
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

	m_bLocked = false;
}

Void WinGUIImage::Destroy()
{
	DebugAssert( m_hHandle != NULL && !m_bShared && !m_bLocked );

	m_bIsDeviceDependant = false;

	switch( m_iType ) {
		case WINGUI_IMAGE_BITMAP: DeleteObject( m_hHandle ); break;
		case WINGUI_IMAGE_ICON:   DestroyIcon( (HICON)m_hHandle ); break;
		case WINGUI_IMAGE_CURSOR: DestroyCursor( (HCURSOR)m_hHandle ); break;
		default: DebugAssert(false); break;
	}
	m_hHandle = NULL;

	m_iType = WINGUI_IMAGE_BITMAP;
	m_iDDWidth = 0;
	m_iDDHeight = 0;

	MemZero( &m_hBitmapDesc, sizeof(WinGUIBitmapDescriptor) );
	m_pBitmapData = NULL;
	m_bLocked = false;
}

Void WinGUIImage::BitBlit( const WinGUIRectangle & hDestRect, const WinGUIImage * pSrcImage, const WinGUIPoint & hSrcOrigin, WinGUIRasterOperation iOperation )
{
	DebugAssert( m_hHandle != NULL && !m_bLocked );
	DebugAssert( pSrcImage->m_hHandle != NULL && !(pSrcImage->m_bLocked) );

	// Retrieve Handles
	HBITMAP hDestBitmap = (HBITMAP)m_hHandle;
	HBITMAP hSrcBitmap = (HBITMAP)(pSrcImage->m_hHandle);

	// Raster Operation
	DWORD iROP = 0;
	switch( iOperation ) {
		case WINGUI_RASTER_BLACK:             iROP = BLACKNESS; break;
		case WINGUI_RASTER_WHITE:             iROP = WHITENESS; break;
		case WINGUI_RASTER_COPY:              iROP = SRCCOPY; break;
		case WINGUI_RASTER_NOTDST:            iROP = DSTINVERT; break;
		case WINGUI_RASTER_NOTSRC:            iROP = NOTSRCCOPY; break;
		case WINGUI_RASTER_AND:               iROP = SRCAND; break;
		case WINGUI_RASTER_OR:                iROP = SRCPAINT; break;
		case WINGUI_RASTER_XOR:               iROP = SRCINVERT; break;
		case WINGUI_RASTER_NOTDST_AND_SRC:    iROP = SRCERASE; break;
		case WINGUI_RASTER_DST_OR_NOTSRC:     iROP = MERGEPAINT; break;
		case WINGUI_RASTER_NOTDST_AND_NOTSRC: iROP = NOTSRCERASE; break;
		default: DebugAssert(false); break;
	};

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
Void WinGUIImage::StretchBlit( const WinGUIRectangle & hDestRect, const WinGUIImage * pSrcImage, const WinGUIRectangle & hSrcRect, WinGUIRasterOperation iOperation )
{
	DebugAssert( m_hHandle != NULL && !m_bLocked );
	DebugAssert( pSrcImage->m_hHandle != NULL && !(pSrcImage->m_bLocked) );

	// Retrieve Handles
	HBITMAP hDestBitmap = (HBITMAP)m_hHandle;
	HBITMAP hSrcBitmap = (HBITMAP)(pSrcImage->m_hHandle);

	// Raster Operation
	DWORD iROP = 0;
	switch( iOperation ) {
		case WINGUI_RASTER_BLACK:             iROP = BLACKNESS; break;
		case WINGUI_RASTER_WHITE:             iROP = WHITENESS; break;
		case WINGUI_RASTER_COPY:              iROP = SRCCOPY; break;
		case WINGUI_RASTER_NOTDST:            iROP = DSTINVERT; break;
		case WINGUI_RASTER_NOTSRC:            iROP = NOTSRCCOPY; break;
		case WINGUI_RASTER_AND:               iROP = SRCAND; break;
		case WINGUI_RASTER_OR:                iROP = SRCPAINT; break;
		case WINGUI_RASTER_XOR:               iROP = SRCINVERT; break;
		case WINGUI_RASTER_NOTDST_AND_SRC:    iROP = SRCERASE; break;
		case WINGUI_RASTER_DST_OR_NOTSRC:     iROP = MERGEPAINT; break;
		case WINGUI_RASTER_NOTDST_AND_NOTSRC: iROP = NOTSRCERASE; break;
		default: DebugAssert(false); break;
	};

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
Void WinGUIImage::MaskBlit( const WinGUIRectangle & hDestRect, const WinGUIImage * pSrcImage, const WinGUIPoint & hSrcOrigin,
							const WinGUIImage * pMask, const WinGUIPoint & hMaskOrigin,
							WinGUIRasterOperation iForegroundOP, WinGUIRasterOperation iBackgroundOP )
{
	DebugAssert( m_hHandle != NULL && !m_bLocked );
	DebugAssert( pSrcImage->m_hHandle != NULL && !(pSrcImage->m_bLocked) );
	DebugAssert( pMask->m_hHandle != NULL && !(pMask->m_bLocked) );

	// Retrieve Handles
	HBITMAP hDestBitmap = (HBITMAP)m_hHandle;
	HBITMAP hSrcBitmap = (HBITMAP)(pSrcImage->m_hHandle);
	HBITMAP hMaskBitmap = (HBITMAP)(pMask->m_hHandle);

	// Raster Operations
	DWORD iForegroundROP = 0, iBackgroundROP = 0;
	switch( iForegroundOP ) {
		case WINGUI_RASTER_BLACK:             iForegroundROP = BLACKNESS; break;
		case WINGUI_RASTER_WHITE:             iForegroundROP = WHITENESS; break;
		case WINGUI_RASTER_COPY:              iForegroundROP = SRCCOPY; break;
		case WINGUI_RASTER_NOTDST:            iForegroundROP = DSTINVERT; break;
		case WINGUI_RASTER_NOTSRC:            iForegroundROP = NOTSRCCOPY; break;
		case WINGUI_RASTER_AND:               iForegroundROP = SRCAND; break;
		case WINGUI_RASTER_OR:                iForegroundROP = SRCPAINT; break;
		case WINGUI_RASTER_XOR:               iForegroundROP = SRCINVERT; break;
		case WINGUI_RASTER_NOTDST_AND_SRC:    iForegroundROP = SRCERASE; break;
		case WINGUI_RASTER_DST_OR_NOTSRC:     iForegroundROP = MERGEPAINT; break;
		case WINGUI_RASTER_NOTDST_AND_NOTSRC: iForegroundROP = NOTSRCERASE; break;
		default: DebugAssert(false); break;
	};
	switch( iBackgroundOP ) {
		case WINGUI_RASTER_BLACK:             iBackgroundROP = BLACKNESS; break;
		case WINGUI_RASTER_WHITE:             iBackgroundROP = WHITENESS; break;
		case WINGUI_RASTER_COPY:              iBackgroundROP = SRCCOPY; break;
		case WINGUI_RASTER_NOTDST:            iBackgroundROP = DSTINVERT; break;
		case WINGUI_RASTER_NOTSRC:            iBackgroundROP = NOTSRCCOPY; break;
		case WINGUI_RASTER_AND:               iBackgroundROP = SRCAND; break;
		case WINGUI_RASTER_OR:                iBackgroundROP = SRCPAINT; break;
		case WINGUI_RASTER_XOR:               iBackgroundROP = SRCINVERT; break;
		case WINGUI_RASTER_NOTDST_AND_SRC:    iBackgroundROP = SRCERASE; break;
		case WINGUI_RASTER_DST_OR_NOTSRC:     iBackgroundROP = MERGEPAINT; break;
		case WINGUI_RASTER_NOTDST_AND_NOTSRC: iBackgroundROP = NOTSRCERASE; break;
		default: DebugAssert(false); break;
	};
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
Void WinGUIImage::TransparentBlit( const WinGUIRectangle & hDestRect, const WinGUIImage * pSrcImage, const WinGUIRectangle & hSrcRect, UInt iKeyColor )
{
	DebugAssert( m_hHandle != NULL && !m_bLocked );
	DebugAssert( pSrcImage->m_hHandle != NULL && !(pSrcImage->m_bLocked) );

	// Retrieve Handles
	HBITMAP hDestBitmap = (HBITMAP)m_hHandle;
	HBITMAP hSrcBitmap = (HBITMAP)(pSrcImage->m_hHandle);

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

/////////////////////////////////////////////////////////////////////////////////

Void * WinGUIImage::_GetAppWindowHandle() const
{
	WinGUIWindow * pAppWindow = WinGUIFn->GetAppWindow();
	return WinGUIElement::_GetHandle(pAppWindow);
}

