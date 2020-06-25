/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Tools/WinGUIImageList.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Image Lists
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
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commctrl.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "WinGUIImageList.h"

#include "../WinGUIWindow.h"

/////////////////////////////////////////////////////////////////////////////////
// WinGUIImageList implementation
WinGUIImageList::WinGUIImageList()
{
	m_hHandle = NULL;

	m_bHasMasks = false;
	m_iWidth = 0;
	m_iHeight = 0;

	m_bIsDragging = false;
}
WinGUIImageList::~WinGUIImageList()
{
	// nothing to do
}

Void WinGUIImageList::Create( UInt iWidth, UInt iHeight, UInt iInitialImageCount, UInt iGrowCount )
{
	DebugAssert( m_hHandle == NULL );

	m_hHandle = ImageList_Create( iWidth, iHeight, ILC_COLORDDB, iInitialImageCount, iGrowCount );
	DebugAssert( m_hHandle != NULL );

	m_bHasMasks = false;
	m_iWidth = iWidth;
	m_iHeight = iHeight;

	m_bIsDragging = false;
}
Void WinGUIImageList::CreateMasked( UInt iWidth, UInt iHeight, UInt iInitialImageCount, UInt iGrowCount )
{
	DebugAssert( m_hHandle == NULL );

	m_hHandle = ImageList_Create( iWidth, iHeight, ILC_COLORDDB | ILC_MASK, iInitialImageCount, iGrowCount );
	DebugAssert( m_hHandle != NULL );

	m_bHasMasks = true;
	m_iWidth = iWidth;
	m_iHeight = iHeight;

	m_bIsDragging = false;
}

Void WinGUIImageList::CreateDuplicate( const WinGUIImageList * pList )
{
	DebugAssert( m_hHandle == NULL );
	DebugAssert( pList->m_hHandle != NULL );

	m_hHandle = ImageList_Duplicate( (HIMAGELIST)(pList->m_hHandle) );
	DebugAssert( m_hHandle != NULL );

	m_bHasMasks = pList->m_bHasMasks;
	m_iWidth = pList->m_iWidth;
	m_iHeight = pList->m_iHeight;

	m_bIsDragging = false;
}

Void WinGUIImageList::MergeImages( WinGUIImageList * pListA, UInt iIndexA, WinGUIImageList * pListB, UInt iIndexB, Int iDX, Int iDY )
{
	DebugAssert( m_hHandle == NULL );
	DebugAssert( pListA->m_hHandle != NULL && pListB->m_hHandle != NULL );
	DebugAssert( pListA->m_bHasMasks && pListB->m_bHasMasks );

	m_hHandle = ImageList_Merge( (HIMAGELIST)(pListA->m_hHandle), iIndexA, (HIMAGELIST)(pListB->m_hHandle), iIndexB, iDX, iDY );
	DebugAssert( m_hHandle != NULL );

	m_bHasMasks = true;
	m_iWidth = pListA->m_iWidth;
	m_iHeight = pListA->m_iHeight;

	m_bIsDragging = false;
}

Void WinGUIImageList::Destroy()
{
	DebugAssert( m_hHandle != NULL );

	ImageList_Destroy( (HIMAGELIST)m_hHandle );

	m_hHandle = NULL;

	m_bHasMasks = false;
	m_iWidth = 0;
	m_iHeight = 0;

	m_bIsDragging = false;
}

UInt WinGUIImageList::GetImageCount() const
{
	DebugAssert( m_hHandle != NULL );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	return ImageList_GetImageCount( hHandle );
}
Void WinGUIImageList::GetImage( UInt iIndex, WinGUIRectangle * outBoundingRect, WinGUIBitmap * outImage, WinGUIBitmap * outMask ) const
{
	DebugAssert( m_hHandle != NULL );
	DebugAssert( !(outImage->IsCreated()) );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;

	IMAGEINFO hImageInfo;
	ImageList_GetImageInfo( hHandle, iIndex, &hImageInfo );

	outBoundingRect->iLeft = hImageInfo.rcImage.left;
	outBoundingRect->iTop = hImageInfo.rcImage.top;
	outBoundingRect->iWidth = ( hImageInfo.rcImage.right - hImageInfo.rcImage.left );
	outBoundingRect->iHeight = ( hImageInfo.rcImage.bottom - hImageInfo.rcImage.top );

	outImage->_CreateFromHandle( hImageInfo.hbmImage, true, true );
	if ( m_bHasMasks && outMask != NULL ) {
		DebugAssert( !(outMask->IsCreated()) );
		outMask->_CreateFromHandle( hImageInfo.hbmMask, true, true );
	}
}

Void WinGUIImageList::MakeIcon( UInt iIndex, WinGUIIcon * outIcon, const WinGUIImageListDrawOptions & hOptions ) const
{
	DebugAssert( m_hHandle != NULL );
	DebugAssert( !(outIcon->IsCreated()) );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;

	// Modes
	UInt iFlags = 0;
		// Stencil Mode
	if ( hOptions.bStencilMode ) {
		DebugAssert( m_bHasMasks );
		iFlags |= ILD_MASK;
	}

		// Alpha Preserve
	if ( hOptions.bPreserveDestAlpha )
		iFlags |= ILD_PRESERVEALPHA;

		// Background
	if ( hOptions.bUseBackgroundColor )
		iFlags |= ILD_NORMAL;
	else {
		DebugAssert( m_bHasMasks );
		iFlags |= ILD_TRANSPARENT;
	}

		// Blending
	if ( hOptions.bUseBlending25 ) {
		DebugAssert( m_bHasMasks );
		iFlags |= ILD_BLEND25;
	} else if ( hOptions.bUseBlending50 ) {
		DebugAssert( m_bHasMasks );
		iFlags |= ILD_BLEND50;
	}

		// Overlay
	if ( hOptions.bUseOverlay ) {
		DebugAssert( hOptions.iOverlayIndex < 15 );
		iFlags |= INDEXTOOVERLAYMASK( 1 + hOptions.iOverlayIndex );
		if ( hOptions.bOverlayRequiresMask ) {
			DebugAssert( m_bHasMasks );
		} else
			iFlags |= ILD_IMAGE;
	}

		// Scaling
	if ( hOptions.bUseScaling )
		iFlags |= ILD_SCALE;
	if ( hOptions.bUseDPIScaling )
		iFlags |= ILD_DPISCALE;

	HICON hIcon = ImageList_GetIcon( hHandle, iIndex, iFlags );
	DebugAssert( hIcon != NULL );

	outIcon->_CreateFromHandle( hIcon, false );
}

UInt WinGUIImageList::AddImage( WinGUIBitmap * pImage, WinGUIBitmap * pMask )
{
	DebugAssert( m_hHandle != NULL );
	DebugAssert( pImage->IsCreated() && pImage->IsDeviceDependant() );
	DebugAssert( pImage->GetDDWidth() == m_iWidth && pImage->GetDDHeight() == m_iHeight );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;

	if ( m_bHasMasks && pMask != NULL ) {
		DebugAssert( pMask->IsCreated() && pMask->IsDeviceDependant() );
		DebugAssert( pMask->GetDDWidth() == m_iWidth && pMask->GetDDHeight() == m_iHeight );
		return ImageList_Add( hHandle, (HBITMAP)(pImage->m_hHandle), (HBITMAP)(pMask->m_hHandle) );
	} else
		return ImageList_Add( hHandle, (HBITMAP)(pImage->m_hHandle), NULL );
}
UInt WinGUIImageList::AddImageMasked( WinGUIBitmap * pImage, UInt iKeyColor )
{
	DebugAssert( m_hHandle != NULL && m_bHasMasks );
	DebugAssert( pImage->IsCreated() && pImage->IsDeviceDependant() );
	DebugAssert( pImage->GetDDWidth() == m_iWidth && pImage->GetDDHeight() == m_iHeight );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	return ImageList_AddMasked( hHandle, (HBITMAP)(pImage->m_hHandle), iKeyColor );
}
UInt WinGUIImageList::AddIcon( WinGUIIcon * pIcon )
{
	DebugAssert( m_hHandle != NULL && m_bHasMasks );
	DebugAssert( pIcon->IsCreated() );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	return ImageList_AddIcon( hHandle, (HICON)(pIcon->m_hHandle) );
}
UInt WinGUIImageList::AddCursor( WinGUICursor * pCursor )
{
	DebugAssert( m_hHandle != NULL && m_bHasMasks );
	DebugAssert( pCursor->IsCreated() );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	return ImageList_AddIcon( hHandle, (HCURSOR)(pCursor->m_hHandle) );
}

Void WinGUIImageList::ReplaceImage( UInt iIndex, WinGUIBitmap * pImage, WinGUIBitmap * pMask )
{
	DebugAssert( m_hHandle != NULL );
	DebugAssert( pImage->IsCreated() && pImage->IsDeviceDependant() );
	DebugAssert( pImage->GetDDWidth() == m_iWidth && pImage->GetDDHeight() == m_iHeight );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;

	if ( m_bHasMasks && pMask != NULL ) {
		DebugAssert( pMask->IsCreated() && pMask->IsDeviceDependant() );
		DebugAssert( pMask->GetDDWidth() == m_iWidth && pMask->GetDDHeight() == m_iHeight );
		ImageList_Replace( hHandle, iIndex, (HBITMAP)(pImage->m_hHandle), (HBITMAP)(pMask->m_hHandle) );
	} else
		ImageList_Replace( hHandle, iIndex, (HBITMAP)(pImage->m_hHandle), NULL );
}
Void WinGUIImageList::ReplaceIcon( UInt iIndex, WinGUIIcon * pIcon )
{
	DebugAssert( m_hHandle != NULL && m_bHasMasks );
	DebugAssert( pIcon->IsCreated() );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	ImageList_ReplaceIcon( hHandle, iIndex, (HICON)(pIcon->m_hHandle) );
}
Void WinGUIImageList::ReplaceCursor( UInt iIndex, WinGUICursor * pCursor )
{
	DebugAssert( m_hHandle != NULL && m_bHasMasks );
	DebugAssert( pCursor->IsCreated() );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	ImageList_ReplaceIcon( hHandle, iIndex, (HCURSOR)(pCursor->m_hHandle) );
}

Void WinGUIImageList::SwapImages( UInt iIndexA, UInt iIndexB )
{
	DebugAssert( m_hHandle != NULL );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	ImageList_Copy( hHandle, iIndexA, hHandle, iIndexB, ILCF_SWAP );
}
Void WinGUIImageList::CopyImage( UInt iIndexDst, UInt iIndexSrc )
{
	DebugAssert( m_hHandle != NULL );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	ImageList_Copy( hHandle, iIndexDst, hHandle, iIndexSrc, ILCF_MOVE );
}

Void WinGUIImageList::RemoveImage( UInt iIndex )
{
	DebugAssert( m_hHandle != NULL );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	ImageList_Remove( hHandle, iIndex );
}
Void WinGUIImageList::RemoveAll()
{
	DebugAssert( m_hHandle != NULL );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	ImageList_RemoveAll( hHandle );
}

UInt WinGUIImageList::GetBackgroundColor() const
{
	DebugAssert( m_hHandle != NULL );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	return ImageList_GetBkColor( hHandle );
}
UInt WinGUIImageList::SetBackgroundColor( UInt iColor )
{
	DebugAssert( m_hHandle != NULL );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	return ImageList_SetBkColor( hHandle, iColor );
}

Void WinGUIImageList::SetOverlayImage( UInt iImageIndex, UInt iOverlayIndex )
{
	DebugAssert( m_hHandle != NULL && m_bHasMasks );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	ImageList_SetOverlayImage( hHandle, iImageIndex, 1 + iOverlayIndex ); // Overlay indices are 1-based
}

Void WinGUIImageList::Draw( WinGUIBitmap * pTarget, const WinGUIPoint & hDestOrigin, UInt iSrcImage, const WinGUIRectangle & hSrcRect, const WinGUIImageListDrawOptions & hOptions )
{
	DebugAssert( m_hHandle != NULL );
	DebugAssert( pTarget->IsCreated() && pTarget->IsDeviceDependant() );

	// Retrieve Handles
	HBITMAP hDestBitmap = (HBITMAP)( pTarget->m_hHandle );

	// Get Window DC
	HWND hAppWindow = (HWND)( WinGUIBitmap::_GetAppWindowHandle() );
	HDC hDC = GetDC( hAppWindow );

	// Create Memory DC
	HDC hDestMemoryDC = CreateCompatibleDC( hDC );

	// Select Bitmap
	HBITMAP hDestSaved = (HBITMAP)SelectObject( hDestMemoryDC, hDestBitmap );

	// Perform Operation
	IMAGELISTDRAWPARAMS hDrawParams;
	hDrawParams.cbSize = sizeof(IMAGELISTDRAWPARAMS);
	hDrawParams.fStyle = 0;
	
		// Destination
	hDrawParams.hdcDst = hDestMemoryDC;
	hDrawParams.x = hDestOrigin.iX;
	hDrawParams.y = hDestOrigin.iY;

		// Source
	hDrawParams.himl = (HIMAGELIST)m_hHandle;
	hDrawParams.i = iSrcImage;
	hDrawParams.xBitmap = hSrcRect.iLeft;
	hDrawParams.yBitmap = hSrcRect.iTop;
	hDrawParams.cx = hSrcRect.iWidth;
	hDrawParams.cy = hSrcRect.iHeight;

		// Stencil Mode
	if ( hOptions.bStencilMode ) {
		DebugAssert( m_bHasMasks );
		hDrawParams.fStyle |= ILD_MASK;
	}

		// Alpha Preserve
	if ( hOptions.bPreserveDestAlpha )
		hDrawParams.fStyle |= ILD_PRESERVEALPHA;

		// Background
	if ( hOptions.bUseBackgroundColor ) {
		hDrawParams.fStyle |= ILD_NORMAL;
		if ( hOptions.bUseDefaultBackgroundColor )
			hDrawParams.rgbBk = CLR_DEFAULT;
		else
			hDrawParams.rgbBk = hOptions.iBackgroundColor;
	} else {
		DebugAssert( m_bHasMasks );
		hDrawParams.fStyle |= ILD_TRANSPARENT;
		hDrawParams.rgbBk = CLR_NONE;
	}

		// Blending
	hDrawParams.rgbFg = CLR_NONE;
	if ( hOptions.bUseBlending25 ) {
		DebugAssert( m_bHasMasks );
		hDrawParams.fStyle |= ILD_BLEND25;
		if ( hOptions.bBlendWithDestination )
			hDrawParams.rgbFg = CLR_NONE;
		else if ( hOptions.bUseDefaultBlendForegroundColor )
			hDrawParams.rgbFg = CLR_DEFAULT;
		else
			hDrawParams.rgbFg = hOptions.iBlendForegroundColor;
	} else if ( hOptions.bUseBlending50 ) {
		DebugAssert( m_bHasMasks );
		hDrawParams.fStyle |= ILD_BLEND50;
		if ( hOptions.bBlendWithDestination )
			hDrawParams.rgbFg = CLR_NONE;
		else if ( hOptions.bUseDefaultBlendForegroundColor )
			hDrawParams.rgbFg = CLR_DEFAULT;
		else
			hDrawParams.rgbFg = hOptions.iBlendForegroundColor;
	}

		// Alpha Blending
	hDrawParams.fState = ILS_NORMAL;
	if ( hOptions.bUseAlphaBlending ) {
		hDrawParams.fState |= ILS_ALPHA;
		hDrawParams.Frame = hOptions.iAlphaValue;
	}

		// Saturation
	if ( hOptions.bUseSaturation )
		hDrawParams.fState |= ILS_SATURATE;

		// Glow & Shadow not supported yet ...
	hDrawParams.crEffect = 0;

		// Raster Operation
	hDrawParams.dwRop = 0;
	if ( hOptions.bUseRasterOp ) {
		hDrawParams.fStyle |= ILD_ROP;
		hDrawParams.dwRop = WinGUIBitmap::_ConvertRasterOperation( hOptions.iRasterOp );
	}

		// Overlay
	if ( hOptions.bUseOverlay ) {
		DebugAssert( hOptions.iOverlayIndex < 15 );
		hDrawParams.fStyle |= INDEXTOOVERLAYMASK( 1 + hOptions.iOverlayIndex );
		if ( hOptions.bOverlayRequiresMask ) {
			DebugAssert( m_bHasMasks );
		} else
			hDrawParams.fStyle |= ILD_IMAGE;
	}

		// Scaling
	if ( hOptions.bUseScaling )
		hDrawParams.fStyle |= ILD_SCALE;
	if ( hOptions.bUseDPIScaling )
		hDrawParams.fStyle |= ILD_DPISCALE;

		// Draw
	ImageList_DrawIndirect( &hDrawParams );

	// Release All
	SelectObject( hDestMemoryDC, hDestSaved );
	DeleteDC( hDestMemoryDC );
	ReleaseDC( hAppWindow, hDC );
}
Void WinGUIImageList::Render( WinGUIElement * pTarget, const WinGUIPoint & hDestOrigin, UInt iSrcImage, const WinGUIRectangle & hSrcRect, const WinGUIImageListDrawOptions & hOptions )
{
	DebugAssert( m_hHandle != NULL );

	// Retrieve Handles
	HWND hTargetWnd = NULL;
	switch( pTarget->GetElementType() ) {
		case WINGUI_ELEMENT_WINDOW:    hTargetWnd = (HWND)( WinGUIElement::_GetHandle(pTarget) ); break;
		case WINGUI_ELEMENT_CONTAINER: hTargetWnd = (HWND)( WinGUIElement::_GetHandle(pTarget) ); break;
		default: DebugAssert(false); break;
	}

	// Get Window DC
	HDC hDC = GetDC( hTargetWnd );

	// Perform Operation
	IMAGELISTDRAWPARAMS hDrawParams;
	hDrawParams.cbSize = sizeof(IMAGELISTDRAWPARAMS);
	hDrawParams.fStyle = 0;
	
		// Destination
	hDrawParams.hdcDst = hDC;
	hDrawParams.x = hDestOrigin.iX;
	hDrawParams.y = hDestOrigin.iY;

		// Source
	hDrawParams.himl = (HIMAGELIST)m_hHandle;
	hDrawParams.i = iSrcImage;
	hDrawParams.xBitmap = hSrcRect.iLeft;
	hDrawParams.yBitmap = hSrcRect.iTop;
	hDrawParams.cx = hSrcRect.iWidth;
	hDrawParams.cy = hSrcRect.iHeight;

		// Stencil Mode
	if ( hOptions.bStencilMode ) {
		DebugAssert( m_bHasMasks );
		hDrawParams.fStyle |= ILD_MASK;
	}

		// Alpha Preserve
	if ( hOptions.bPreserveDestAlpha )
		hDrawParams.fStyle |= ILD_PRESERVEALPHA;

		// Background
	if ( hOptions.bUseBackgroundColor ) {
		hDrawParams.fStyle |= ILD_NORMAL;
		if ( hOptions.bUseDefaultBackgroundColor )
			hDrawParams.rgbBk = CLR_DEFAULT;
		else
			hDrawParams.rgbBk = hOptions.iBackgroundColor;
	} else {
		DebugAssert( m_bHasMasks );
		hDrawParams.fStyle |= ILD_TRANSPARENT;
		hDrawParams.rgbBk = CLR_NONE;
	}

		// Blending
	hDrawParams.rgbFg = CLR_NONE;
	if ( hOptions.bUseBlending25 ) {
		DebugAssert( m_bHasMasks );
		hDrawParams.fStyle |= ILD_BLEND25;
		if ( hOptions.bBlendWithDestination )
			hDrawParams.rgbFg = CLR_NONE;
		else if ( hOptions.bUseDefaultBlendForegroundColor )
			hDrawParams.rgbFg = CLR_DEFAULT;
		else
			hDrawParams.rgbFg = hOptions.iBlendForegroundColor;
	} else if ( hOptions.bUseBlending50 ) {
		DebugAssert( m_bHasMasks );
		hDrawParams.fStyle |= ILD_BLEND50;
		if ( hOptions.bBlendWithDestination )
			hDrawParams.rgbFg = CLR_NONE;
		else if ( hOptions.bUseDefaultBlendForegroundColor )
			hDrawParams.rgbFg = CLR_DEFAULT;
		else
			hDrawParams.rgbFg = hOptions.iBlendForegroundColor;
	}

		// Alpha Blending
	hDrawParams.fState = ILS_NORMAL;
	if ( hOptions.bUseAlphaBlending ) {
		hDrawParams.fState |= ILS_ALPHA;
		hDrawParams.Frame = hOptions.iAlphaValue;
	}

		// Saturation
	if ( hOptions.bUseSaturation )
		hDrawParams.fState |= ILS_SATURATE;

		// Glow & Shadow not supported yet ...
	hDrawParams.crEffect = 0;

		// Raster Operation
	hDrawParams.dwRop = 0;
	if ( hOptions.bUseRasterOp ) {
		hDrawParams.fStyle |= ILD_ROP;
		hDrawParams.dwRop = WinGUIBitmap::_ConvertRasterOperation( hOptions.iRasterOp );
	}

		// Overlay
	if ( hOptions.bUseOverlay ) {
		DebugAssert( hOptions.iOverlayIndex < 15 );
		hDrawParams.fStyle |= INDEXTOOVERLAYMASK( 1 + hOptions.iOverlayIndex );
		if ( hOptions.bOverlayRequiresMask ) {
			DebugAssert( m_bHasMasks );
		} else
			hDrawParams.fStyle |= ILD_IMAGE;
	}

		// Scaling
	if ( hOptions.bUseScaling )
		hDrawParams.fStyle |= ILD_SCALE;
	if ( hOptions.bUseDPIScaling )
		hDrawParams.fStyle |= ILD_DPISCALE;

		// Draw
	ImageList_DrawIndirect( &hDrawParams );

	// Release All
	ReleaseDC( hTargetWnd, hDC );
}

Void WinGUIImageList::DragBegin( UInt iImageIndex, const WinGUIPoint & hHotSpot )
{
	DebugAssert( m_hHandle != NULL );
	DebugAssert( !m_bIsDragging );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	ImageList_BeginDrag( hHandle, iImageIndex, hHotSpot.iX, hHotSpot.iY );

	m_bIsDragging = true;
}
Void WinGUIImageList::DragEnter( WinGUIWindow * pOwner, const WinGUIPoint & hPosition )
{
	DebugAssert( m_hHandle != NULL );
	DebugAssert( m_bIsDragging );
	DebugAssert( pOwner->IsCreated() );

	HWND hWnd = (HWND)( WinGUIElement::_GetHandle(pOwner) );

	ImageList_DragEnter( hWnd, hPosition.iX, hPosition.iY );
}
Void WinGUIImageList::DragEnd()
{
	DebugAssert( m_hHandle != NULL );
	DebugAssert( m_bIsDragging );

	ImageList_EndDrag();

	m_bIsDragging = false;
}
Void WinGUIImageList::DragLeave( WinGUIWindow * pOwner )
{
	DebugAssert( m_hHandle != NULL );
	DebugAssert( !m_bIsDragging );

	HWND hWnd = (HWND)( WinGUIElement::_GetHandle(pOwner) );

	ImageList_DragLeave( hWnd );
}

Void WinGUIImageList::DragShow( Bool bShow )
{
	DebugAssert( m_hHandle != NULL );
	DebugAssert( m_bIsDragging );

	ImageList_DragShowNolock( bShow ? TRUE : FALSE );
}
Void WinGUIImageList::DragMove( const WinGUIPoint & hPosition )
{
	DebugAssert( m_hHandle != NULL );
	DebugAssert( m_bIsDragging );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	ImageList_DragMove( hPosition.iX, hPosition.iY );
}

//Void WinGUIImageList::GetDragImage( WinGUIImageList * outImageList, WinGUIPoint * outDragPosition, WinGUIPoint * outHotSpot ) const
//{
//	DebugAssert( m_hHandle != NULL );
//	DebugAssert( m_bIsDragging );
//	DebugAssert( outImageList->m_hHandle == NULL );
//
//	POINT hPos, hHotSpot;
//	outImageList->m_hHandle = ImageList_GetDragImage( &hPos, &hHotSpot );
//
//	outImageList->m_iCount = 1;
//
//	outImageList->m_bHasMasks = m_bHasMasks;
//	outImageList->m_iWidth = m_iWidth;
//	outImageList->m_iHeight = m_iHeight;
//
//	outImageList->m_bIsDragging = true;
//
//	outDragPosition->iX = hPos.x;
//	outDragPosition->iY = hPos.y;
//	outHotSpot->iX = hHotSpot.x;
//	outHotSpot->iY = hHotSpot.y;
//}
Void WinGUIImageList::CombineDragImages( UInt iNewImageIndex, const WinGUIPoint & hNewHotSpot )
{
	DebugAssert( m_hHandle != NULL );
	DebugAssert( m_bIsDragging );

	HIMAGELIST hHandle = (HIMAGELIST)m_hHandle;
	ImageList_SetDragCursorImage( hHandle, iNewImageIndex, hNewHotSpot.iX, hNewHotSpot.iY );
}

