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
	switch( hOptions.iMode ) {
		case WINGUI_IMAGELIST_DRAW_NORMAL:      iFlags |= ILD_NORMAL; break;
		case WINGUI_IMAGELIST_DRAW_MASK:        iFlags |= ILD_MASK; break;
		case WINGUI_IMAGELIST_DRAW_BLEND25:     iFlags |= ILD_BLEND25; break;
		case WINGUI_IMAGELIST_DRAW_BLEND50:     iFlags |= ILD_BLEND50; break;
		case WINGUI_IMAGELIST_DRAW_TRANSPARENT: iFlags |= ILD_TRANSPARENT; break;
		default: DebugAssert(false); break;
	}
	if ( hOptions.bUseOverlay ) {
		DebugAssert( hOptions.iOverlayIndex < 15 );
		iFlags |= INDEXTOOVERLAYMASK(1 + hOptions.iOverlayIndex);
		if ( !(hOptions.bOverlayRequiresMask) )
			iFlags |= ILD_IMAGE;
	}
	if ( hOptions.bScaleElseClip ) {
		if ( hOptions.bUseDPIScaling )
			iFlags |= ILD_DPISCALE;
		else
			iFlags |= ILD_SCALE;
	}
	if ( hOptions.bPreserveDestAlpha )
		iFlags |= ILD_PRESERVEALPHA;

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

UInt WinGUIImageList::GetBackgroundColor()
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

