/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUITable.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : Table (ListView)
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
#include <windowsx.h>
#include <commctrl.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "WinGUITable.h"

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUITableModel implementation
WinGUITableModel::WinGUITableModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// Default Parameters
	m_hCreationParameters.bMakeVirtualTable = false;
	m_hCreationParameters.bHeadersInAllViews = false;
	m_hCreationParameters.iViewMode = WINGUI_TABLE_VIEW_LIST;

	m_hCreationParameters.bStaticColumnHeaders = false;
	m_hCreationParameters.bSnapColumnsWidth = false;
	m_hCreationParameters.bAutoSizeColumns = false;

	m_hCreationParameters.bEditableLabels = false;

	m_hCreationParameters.bSingleItemSelection = false;
	m_hCreationParameters.bAlwaysShowSelection = true;
	m_hCreationParameters.bBorderSelection = false;

	m_hCreationParameters.bSortAscending = false;
	m_hCreationParameters.bSortDescending = false;

	m_hCreationParameters.bAddCheckBoxes = false;
	m_hCreationParameters.bAutoCheckOnSelect = false;

	m_hCreationParameters.bHandleInfoTips = false;

	m_hCreationParameters.bHotTrackingSingleClick = true;
	m_hCreationParameters.bHotTrackingDoubleClick = false;
	m_hCreationParameters.bHotTrackSelection = true;
	m_hCreationParameters.bUnderlineHot = false;
	m_hCreationParameters.bUnderlineCold = false;

	m_hCreationParameters.bSharedImageList = false;
	m_hCreationParameters.bUseBackBuffer = true;
	m_hCreationParameters.bTransparentBackground = false;
	m_hCreationParameters.bTransparentShadowText = false;
}
WinGUITableModel::~WinGUITableModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUITable implementation
WinGUITable::WinGUITable( WinGUIElement * pParent, WinGUITableModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	m_bVirtualTable = false;
	m_iViewMode = WINGUI_TABLE_VIEW_LIST;
	m_bGroupMode = false;

	m_bHasCheckBoxes = false;
}
WinGUITable::~WinGUITable()
{
	// nothing to do
}

Void WinGUITable::SwitchViewMode( WinGUITableViewMode iViewMode )
{
	HWND hHandle = (HWND)m_hHandle;

	switch( iViewMode ) {
		case WINGUI_TABLE_VIEW_LIST:        ListView_SetView( hHandle, LV_VIEW_LIST ); break;
		case WINGUI_TABLE_VIEW_ICONS:       ListView_SetView( hHandle, LV_VIEW_ICON ); break;
		case WINGUI_TABLE_VIEW_ICONS_SMALL: ListView_SetView( hHandle, LV_VIEW_SMALLICON ); break;
		case WINGUI_TABLE_VIEW_DETAILED:    ListView_SetView( hHandle, LV_VIEW_DETAILS ); break;
		case WINGUI_TABLE_VIEW_TILES:       ListView_SetView( hHandle, LV_VIEW_TILE ); break;
		default: DebugAssert(false); break;
	}

	m_iViewMode = iViewMode;
}

Bool WinGUITable::IsUnicode() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ( ListView_GetUnicodeFormat(hHandle) != 0 );
}
Bool WinGUITable::IsANSI() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ( ListView_GetUnicodeFormat(hHandle) == 0 );
}

Void WinGUITable::SetUnicode()
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetUnicodeFormat( hHandle, TRUE );
}
Void WinGUITable::SetANSI()
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetUnicodeFormat( hHandle, FALSE );
}

Void WinGUITable::SetItemCount( UInt iPreAllocatedItemCount )
{
	HWND hHandle = (HWND)m_hHandle;

	if ( m_bVirtualTable ) {
		ListView_SetItemCountEx( hHandle, iPreAllocatedItemCount, LVSICF_NOINVALIDATEALL | LVSICF_NOSCROLL );
	} else {
		ListView_SetItemCount( hHandle, iPreAllocatedItemCount );
	}
}

Void WinGUITable::AdjustRequiredDimensions( UInt * pWidth, UInt * pHeight, UInt iItemCount ) const
{
	HWND hHandle = (HWND)m_hHandle;

	DWord dwResult = ListView_ApproximateViewRect( hHandle, *pWidth, *pHeight, iItemCount );
	*pWidth = (UInt)( LOWORD(dwResult) );
	*pHeight = (UInt)( HIWORD(dwResult) );
}

UInt WinGUITable::GetStringWidth( const GChar * strText ) const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetStringWidth( hHandle, strText );
}

Void WinGUITable::GetViewOrigin( WinGUIPoint * outOrigin ) const
{
	HWND hHandle = (HWND)m_hHandle;

	POINT hPt;
	ListView_GetOrigin( hHandle, &hPt );

	outOrigin->iX = hPt.x;
	outOrigin->iY = hPt.y;
}
Void WinGUITable::GetIconViewRect( WinGUIRectangle * outRectangle ) const
{
	DebugAssert( m_iViewMode == WINGUI_TABLE_VIEW_ICONS || m_iViewMode == WINGUI_TABLE_VIEW_ICONS_SMALL );

	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	ListView_GetViewRect( hHandle, &hRect );

	outRectangle->iLeft = hRect.left;
	outRectangle->iTop = hRect.top;
	outRectangle->iWidth = ( hRect.right - hRect.left );
	outRectangle->iHeight = ( hRect.bottom - hRect.top );
}

Bool WinGUITable::IsItemVisible( UInt iIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;
	return ( ListView_IsItemVisible(hHandle, iIndex) != FALSE );
}

UInt WinGUITable::GetFirstVisibleItem() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetTopIndex( hHandle );
}
UInt WinGUITable::GetVisibleItemCount() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetCountPerPage( hHandle );
}
Void WinGUITable::GetIconSpacing( UInt * outSpacingH, UInt * outSpacingV, Bool bSmallIcons ) const
{
	HWND hHandle = (HWND)m_hHandle;
	
	DWord dwSpacing = ListView_GetItemSpacing( hHandle, bSmallIcons ? TRUE : FALSE );

	*outSpacingH = LOWORD(dwSpacing);
	*outSpacingV = HIWORD(dwSpacing);
}
Void WinGUITable::SetIconSpacing( UInt iSpacingH, UInt iSpacingV )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetIconSpacing( hHandle, iSpacingH, iSpacingV );
}

Void WinGUITable::GetEmptyText( GChar * outText, UInt iMaxLength ) const
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_GetEmptyText( hHandle, outText, iMaxLength );
}

UInt WinGUITable::GetBackgroundColor() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetBkColor( hHandle );
}
Void WinGUITable::SetBackgroundColor( UInt iColor )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetBkColor( hHandle, iColor );
}

Void WinGUITable::SetBackgroundImage( const WinGUIBitmap * pImage, const WinGUIPointF & hRelativePos, Bool bUseTiling )
{
	DebugAssert( pImage->IsCreated() && pImage->IsDeviceDependant() );

	HWND hHandle = (HWND)m_hHandle;

	LVBKIMAGE hBkgImageInfos;
	hBkgImageInfos.hbm = (HBITMAP)( pImage->m_hHandle );
	hBkgImageInfos.pszImage = NULL;
	hBkgImageInfos.cchImageMax = 0;
	hBkgImageInfos.xOffsetPercent = (Int)( hRelativePos.fX * 100.0f );
	hBkgImageInfos.yOffsetPercent = (Int)( hRelativePos.fY * 100.0f );

	hBkgImageInfos.ulFlags = LVBKIF_SOURCE_HBITMAP;
	if ( bUseTiling )
		hBkgImageInfos.ulFlags |= LVBKIF_STYLE_TILE;

	ListView_SetBkImage( hHandle, &hBkgImageInfos );
}
Void WinGUITable::RemoveBackgroundImage()
{
	HWND hHandle = (HWND)m_hHandle;

	LVBKIMAGE hBkgImageInfos;
	hBkgImageInfos.hbm = NULL;
	hBkgImageInfos.pszImage = NULL;
	hBkgImageInfos.cchImageMax = 0;
	hBkgImageInfos.xOffsetPercent = 0;
	hBkgImageInfos.yOffsetPercent = 0;
	hBkgImageInfos.ulFlags = LVBKIF_SOURCE_NONE;

	ListView_SetBkImage( hHandle, &hBkgImageInfos );
}

UInt WinGUITable::GetTextBackgroundColor() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetTextBkColor( hHandle );
}
Void WinGUITable::SetTextBackgroundColor( UInt iColor )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetTextBkColor( hHandle, iColor );
}

UInt WinGUITable::GetTextColor() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetTextColor( hHandle );
}
Void WinGUITable::SetTextColor( UInt iColor )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetTextColor( hHandle, iColor );
}

UInt WinGUITable::GetBorderSelectionColor() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetOutlineColor( hHandle );
}
Void WinGUITable::SetBorderSelectionColor( UInt iColor )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetOutlineColor( hHandle, iColor );
}

Void WinGUITable::GetInsertionMarkMetrics( WinGUIRectangle * outRectangle, UInt * outColor ) const
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	ListView_GetInsertMarkRect( hHandle, &hRect );

	outRectangle->iLeft = hRect.left;
	outRectangle->iTop = hRect.top;
	outRectangle->iWidth = ( hRect.right - hRect.left );
	outRectangle->iHeight = ( hRect.bottom - hRect.top );
	*outColor = ListView_GetInsertMarkColor( hHandle );
}
Void WinGUITable::SetInsertionMarkColor( UInt iColor )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetInsertMarkColor( hHandle, iColor );
}

Void WinGUITable::GetGroupMetrics( WinGUIPoint * outBorderSizeLeftTop, WinGUIPoint * outBorderSizeRightBottom ) const
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;

	LVGROUPMETRICS hGroupMetrics;
	hGroupMetrics.cbSize = sizeof(LVGROUPMETRICS);
	hGroupMetrics.mask = LVGMF_BORDERSIZE; // Other fields are unimplemented

	ListView_GetGroupMetrics( hHandle, &hGroupMetrics );

	outBorderSizeLeftTop->iX = hGroupMetrics.Left;
	outBorderSizeLeftTop->iY = hGroupMetrics.Top;

	outBorderSizeRightBottom->iX = hGroupMetrics.Right;
	outBorderSizeRightBottom->iY = hGroupMetrics.Bottom;
}
Void WinGUITable::SetGroupMetrics( const WinGUIPoint & hBorderSizeLeftTop, const WinGUIPoint & hBorderSizeRightBottom )
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;

	LVGROUPMETRICS hGroupMetrics;
	hGroupMetrics.cbSize = sizeof(LVGROUPMETRICS);
	hGroupMetrics.mask = LVGMF_BORDERSIZE; // Other fields are unimplemented

	hGroupMetrics.Left = hBorderSizeLeftTop.iX;
	hGroupMetrics.Top = hBorderSizeLeftTop.iY;

	hGroupMetrics.Right = hBorderSizeRightBottom.iX;
	hGroupMetrics.Bottom = hBorderSizeRightBottom.iY;

	ListView_SetGroupMetrics( hHandle, &hGroupMetrics );
}

Void WinGUITable::GetGroupRect( WinGUIRectangle * outRectangle, UInt iGroupID, Bool bCollapsed, Bool bLabelOnly ) const
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;

	LONG iType = 0;
	if ( bLabelOnly )
		iType = LVGGR_LABEL;
	else if ( bCollapsed )
		iType = LVGGR_HEADER;
	else
		iType = LVGGR_GROUP;

	RECT hRect;
	ListView_GetGroupRect( hHandle, iGroupID, iType, &hRect );

	outRectangle->iLeft = hRect.left;
	outRectangle->iTop = hRect.top;
	outRectangle->iWidth = ( hRect.right - hRect.left );
	outRectangle->iHeight = ( hRect.bottom - hRect.top );
}

Void WinGUITable::GetItemPosition( WinGUIPoint * outPosition, UInt iIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	POINT hPt;
	ListView_GetItemPosition( hHandle, iIndex, &hPt );

	outPosition->iX = hPt.x;
	outPosition->iY = hPt.y;
}
Void WinGUITable::GetItemRect( WinGUIRectangle * outRectangle, UInt iIndex, Bool bIconOnly, Bool bLabelOnly ) const
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iCode = LVIR_BOUNDS;
	if ( bIconOnly )
		iCode |= LVIR_ICON;
	if ( bLabelOnly )
		iCode |= LVIR_LABEL;
	// Both combined = LVIR_SELECTBOUNDS

	RECT hRect;
	ListView_GetItemRect( hHandle, iIndex, &hRect, iCode );

	outRectangle->iLeft = hRect.left;
	outRectangle->iTop = hRect.top;
	outRectangle->iWidth = ( hRect.right - hRect.left );
	outRectangle->iHeight = ( hRect.bottom - hRect.top );
}
Void WinGUITable::GetSubItemRect( WinGUIRectangle * outRectangle, UInt iIndex, UInt iSubItem, Bool bIconOnly, Bool bLabelOnly ) const
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iCode = LVIR_BOUNDS;
	if ( bIconOnly )
		iCode |= LVIR_ICON;
	if ( bLabelOnly )
		iCode |= LVIR_LABEL;
	DebugAssert( iCode != LVIR_SELECTBOUNDS ); // Both combined = LVIR_SELECTBOUNDS, NOT ALLOWED HERE

	RECT hRect;
	ListView_GetSubItemRect( hHandle, iIndex, (1 + iSubItem), iCode, &hRect );

	outRectangle->iLeft = hRect.left;
	outRectangle->iTop = hRect.top;
	outRectangle->iWidth = ( hRect.right - hRect.left );
	outRectangle->iHeight = ( hRect.bottom - hRect.top );
}
Void WinGUITable::GetSubItemRect( WinGUIRectangle * outRectangle, UInt iGroupIndex, UInt iIndex, UInt iSubItem, Bool bIconOnly, Bool bLabelOnly ) const
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iCode = LVIR_BOUNDS;
	if ( bIconOnly )
		iCode |= LVIR_ICON;
	if ( bLabelOnly )
		iCode |= LVIR_LABEL;
	DebugAssert( iCode != LVIR_SELECTBOUNDS ); // Both combined = LVIR_SELECTBOUNDS, NOT ALLOWED HERE

	LVITEMINDEX hItemIndex;
	hItemIndex.iGroup = iGroupIndex;
	hItemIndex.iItem = iIndex;

	RECT hRect;
	ListView_GetItemIndexRect( hHandle, &hItemIndex, (1 + iSubItem), iCode, &hRect );

	outRectangle->iLeft = hRect.left;
	outRectangle->iTop = hRect.top;
	outRectangle->iWidth = ( hRect.right - hRect.left );
	outRectangle->iHeight = ( hRect.bottom - hRect.top );
}

Void WinGUITable::SetItemIconPosition( UInt iIndex, const WinGUIPoint & hPosition )
{
	DebugAssert( m_iViewMode == WINGUI_TABLE_VIEW_ICONS || m_iViewMode == WINGUI_TABLE_VIEW_ICONS_SMALL );

	HWND hHandle = (HWND)m_hHandle;

	ListView_SetItemPosition( hHandle, iIndex, hPosition.iX, hPosition.iY );
}

Void WinGUITable::GetTileMetrics( WinGUITableTileMetrics * outTileMetrics ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVTILEVIEWINFO hTileViewInfos;
	hTileViewInfos.cbSize = sizeof(LVTILEVIEWINFO);
	hTileViewInfos.dwMask = LVTVIM_TILESIZE | LVTVIM_COLUMNS | LVTVIM_LABELMARGIN;

	ListView_GetTileViewInfo( hHandle, &hTileViewInfos );

	_Convert_TileMetrics( outTileMetrics, &hTileViewInfos );
}
Void WinGUITable::SetTileMetrics( const WinGUITableTileMetrics * pTileMetrics )
{
	HWND hHandle = (HWND)m_hHandle;

	LVTILEVIEWINFO hTileViewInfos;
	hTileViewInfos.cbSize = sizeof(LVTILEVIEWINFO);
	hTileViewInfos.dwMask = LVTVIM_TILESIZE | LVTVIM_COLUMNS | LVTVIM_LABELMARGIN;

	_Convert_TileMetrics( &hTileViewInfos, pTileMetrics );

	ListView_SetTileViewInfo( hHandle, &hTileViewInfos );
}

Void WinGUITable::GetImageListIcons( WinGUIImageList * outImageList ) const
{
	DebugAssert( !(outImageList->IsCreated()) );

	HWND hHandle = (HWND)m_hHandle;

	HIMAGELIST hImageList = ListView_GetImageList( hHandle, LVSIL_NORMAL );
	outImageList->_CreateFromHandle( hImageList );
}
Void WinGUITable::SetImageListIcons( const WinGUIImageList * pImageList )
{
	DebugAssert( pImageList->IsCreated() );

	HWND hHandle = (HWND)m_hHandle;

	ListView_SetImageList( hHandle, (HIMAGELIST)(pImageList->m_hHandle), LVSIL_NORMAL );
}

Void WinGUITable::GetImageListSmallIcons( WinGUIImageList * outImageList ) const
{
	DebugAssert( !(outImageList->IsCreated()) );

	HWND hHandle = (HWND)m_hHandle;

	HIMAGELIST hImageList = ListView_GetImageList( hHandle, LVSIL_SMALL );
	outImageList->_CreateFromHandle( hImageList );
}
Void WinGUITable::SetImageListSmallIcons( const WinGUIImageList * pImageList )
{
	DebugAssert( pImageList->IsCreated() );

	HWND hHandle = (HWND)m_hHandle;

	ListView_SetImageList( hHandle, (HIMAGELIST)(pImageList->m_hHandle), LVSIL_SMALL );
}

Void WinGUITable::GetImageListGroupHeaders( WinGUIImageList * outImageList ) const
{
	DebugAssert( !(outImageList->IsCreated()) );

	HWND hHandle = (HWND)m_hHandle;

	HIMAGELIST hImageList = ListView_GetImageList( hHandle, LVSIL_GROUPHEADER );
	outImageList->_CreateFromHandle( hImageList );
}
Void WinGUITable::SetImageListGroupHeaders( const WinGUIImageList * pImageList )
{
	DebugAssert( pImageList->IsCreated() );

	HWND hHandle = (HWND)m_hHandle;

	HIMAGELIST hPrevious = ListView_SetImageList( hHandle, (HIMAGELIST)(pImageList->m_hHandle), LVSIL_GROUPHEADER );
	ImageList_Destroy( hPrevious );
}

Void WinGUITable::GetImageListStates( WinGUIImageList * outImageList ) const
{
	DebugAssert( !(outImageList->IsCreated()) );

	HWND hHandle = (HWND)m_hHandle;

	HIMAGELIST hImageList = ListView_GetImageList( hHandle, LVSIL_STATE );
	outImageList->_CreateFromHandle( hImageList );
}
Void WinGUITable::SetImageListStates( const WinGUIImageList * pImageList )
{
	DebugAssert( pImageList->IsCreated() );

	HWND hHandle = (HWND)m_hHandle;

	ListView_SetImageList( hHandle, (HIMAGELIST)(pImageList->m_hHandle), LVSIL_STATE );
}

Void WinGUITable::ForceRedraw( UInt iFirstItem, UInt iLastItem, Bool bImmediate )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_RedrawItems( hHandle, iFirstItem, iLastItem );
	if ( bImmediate )
		UpdateWindow( hHandle );
}

Void WinGUITable::Scroll( Int iScrollH, Int iScrollV )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_Scroll( hHandle, iScrollH, iScrollV );
}
Void WinGUITable::ScrollToItem( UInt iIndex, Bool bAllowPartial )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_EnsureVisible( hHandle, iIndex, bAllowPartial ? TRUE : FALSE );
}

Void WinGUITable::GetColumnInfos( WinGUITableColumnInfos * outInfos, UInt iIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT | LVCF_WIDTH | LVCF_TEXT | LVCF_SUBITEM | LVCF_IMAGE
		              | LVCF_ORDER | LVCF_MINWIDTH | LVCF_DEFAULTWIDTH | LVCF_IDEALWIDTH;

	hColumnInfos.pszText = outInfos->strHeaderText;
	hColumnInfos.cchTextMax = 64;

	ListView_GetColumn( hHandle, (1 + iIndex), &hColumnInfos );

	_Convert_ColumnInfos( outInfos, &hColumnInfos );
}

Void WinGUITable::GetColumnOrder( UInt * outOrderedIndices, UInt iCount ) const
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_GetColumnOrderArray( hHandle, iCount, outOrderedIndices );
}
Void WinGUITable::SetColumnOrder( const UInt * arrOrderedIndices, UInt iCount )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetColumnOrderArray( hHandle, iCount, arrOrderedIndices );
}

UInt WinGUITable::GetColumnWidth( UInt iIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetColumnWidth( hHandle, iIndex );
}
Void WinGUITable::SetColumnWidth( UInt iIndex, UInt iWidth )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetColumnWidth( hHandle, iIndex, iWidth );
}

UInt WinGUITable::GetSelectedColumn() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetSelectedColumn( hHandle );
}
Void WinGUITable::SelectColumn( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetSelectedColumn( hHandle, iIndex );
}

Void WinGUITable::AddColumn( UInt iIndex, const WinGUITableColumnInfos * pColumnInfos )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT | LVCF_WIDTH | LVCF_TEXT | LVCF_SUBITEM | LVCF_IMAGE
		              | LVCF_ORDER | LVCF_MINWIDTH | LVCF_DEFAULTWIDTH | LVCF_IDEALWIDTH;

	_Convert_ColumnInfos( &hColumnInfos, pColumnInfos );

	ListView_InsertColumn( hHandle, iIndex, &hColumnInfos );
}
Void WinGUITable::SetColumn( UInt iIndex, const WinGUITableColumnInfos * pColumnInfos )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT | LVCF_WIDTH | LVCF_TEXT | LVCF_SUBITEM | LVCF_IMAGE
		              | LVCF_ORDER | LVCF_MINWIDTH | LVCF_DEFAULTWIDTH | LVCF_IDEALWIDTH;

	_Convert_ColumnInfos( &hColumnInfos, pColumnInfos );

	ListView_SetColumn( hHandle, iIndex, &hColumnInfos );
}
Void WinGUITable::RemoveColumn( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_DeleteColumn( hHandle, iIndex );
}

UInt WinGUITable::GetItemCount() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetItemCount( hHandle );
}

UInt WinGUITable::GetMultiSelectMark() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetSelectionMark( hHandle );
}
Void WinGUITable::SetMultiSelectMark( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetSelectionMark( hHandle, iIndex );
}

UInt WinGUITable::GetSelectedItemCount() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetSelectedCount( hHandle );
}

UInt WinGUITable::GetInsertionMark( Bool * outInsertAfter ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVINSERTMARK hInsertMark;
	hInsertMark.cbSize = sizeof(LVINSERTMARK);
	ListView_GetInsertMark( hHandle, &hInsertMark );

	*outInsertAfter = ( (hInsertMark.dwFlags & LVIM_AFTER) != 0 );
	return hInsertMark.iItem;
}
UInt WinGUITable::GetInsertionMark( Bool * outInsertAfter, const WinGUIPoint & hPoint ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVINSERTMARK hInsertMark;
	hInsertMark.cbSize = sizeof(LVINSERTMARK);

	POINT hPt;
	hPt.x = hPoint.iX;
	hPt.y = hPoint.iY;

	ListView_InsertMarkHitTest( hHandle, &hPt, &hInsertMark );

	*outInsertAfter = ( (hInsertMark.dwFlags & LVIM_AFTER) != 0 );
	return hInsertMark.iItem;
}
Void WinGUITable::SetInsertionMark( UInt iIndex, Bool bInsertAfter )
{
	HWND hHandle = (HWND)m_hHandle;

	LVINSERTMARK hInsertMark;
	hInsertMark.cbSize = sizeof(LVINSERTMARK);
	hInsertMark.iItem = iIndex;
	hInsertMark.dwFlags = 0;
	if ( bInsertAfter )
		hInsertMark.dwFlags |= LVIM_AFTER;

	ListView_SetInsertMark( hHandle, &hInsertMark );
}

Void WinGUITable::AddItem( UInt iIndex, const WinGUITableItemInfos * pItemInfos )
{
	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_GROUPID | LVIF_TEXT | LVIF_PARAM | LVIF_COLUMNS | LVIF_COLFMT | LVIF_INDENT | LVIF_IMAGE | LVIF_STATE;
	hItemInfos.stateMask = 0xffffffff;

	static UInt arrTempIndices[32];
	hItemInfos.puColumns = arrTempIndices;
	static Int arrTempFormats[32];
	hItemInfos.piColFmt = arrTempFormats;

	_Convert_ItemInfos( &hItemInfos, pItemInfos );

	hItemInfos.iItem = iIndex;
	hItemInfos.iSubItem = 0;

	ListView_InsertItem( hHandle, &hItemInfos );
}
Void WinGUITable::SetItem( UInt iItemIndex, const WinGUITableItemInfos * pItemInfos )
{
	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_GROUPID | LVIF_TEXT | LVIF_PARAM | LVIF_COLUMNS | LVIF_COLFMT | LVIF_INDENT | LVIF_IMAGE | LVIF_STATE;
	hItemInfos.stateMask = 0xffffffff;

	static UInt arrTempIndices[32];
	hItemInfos.puColumns = arrTempIndices;
	static Int arrTempFormats[32];
	hItemInfos.piColFmt = arrTempFormats;

	_Convert_ItemInfos( &hItemInfos, pItemInfos );

	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;

	ListView_SetItem( hHandle, &hItemInfos );
}
Void WinGUITable::SetSubItem( UInt iItemIndex, UInt iSubItemIndex, const WinGUITableItemInfos * pItemInfos )
{
	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_GROUPID | LVIF_TEXT | LVIF_COLUMNS | LVIF_COLFMT | LVIF_INDENT | LVIF_IMAGE;
	hItemInfos.stateMask = 0;

	static UInt arrTempIndices[32];
	hItemInfos.puColumns = arrTempIndices;
	static Int arrTempFormats[32];
	hItemInfos.piColFmt = arrTempFormats;

	_Convert_ItemInfos( &hItemInfos, pItemInfos );

	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = ( 1 + iSubItemIndex );

	ListView_SetItem( hHandle, &hItemInfos );
}
Void WinGUITable::RemoveItem( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_DeleteItem( hHandle, iIndex );
}
Void WinGUITable::RemoveAllItems()
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_DeleteAllItems( hHandle );
}

Void WinGUITable::EnableGroups( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_EnableGroupView( hHandle, bEnable ? TRUE : FALSE );

	m_bGroupMode = bEnable;
}

Bool WinGUITable::HasGroup( UInt iGroupID ) const
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;
	return ( ListView_HasGroup(hHandle, iGroupID) != FALSE );
}

UInt WinGUITable::GetGroupCount() const
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetGroupCount( hHandle );
}
UInt WinGUITable::GetFocusedGroup() const
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetFocusedGroup( hHandle );
}

Void WinGUITable::GetGroupInfosByID( WinGUITableGroupInfos * outInfos, UInt iGroupID ) const
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;

	LVGROUP hGroupInfos;
	hGroupInfos.cbSize = sizeof(LVGROUP);
	hGroupInfos.mask = LVGF_GROUPID | LVGF_ITEMS | LVGF_HEADER | LVGF_FOOTER | LVGF_ALIGN | LVGF_STATE |
		LVGF_SUBTITLE | LVGF_TASK | LVGF_DESCRIPTIONTOP | LVGF_DESCRIPTIONBOTTOM |
		LVGF_TITLEIMAGE | LVGF_EXTENDEDIMAGE | LVGF_SUBSET; // | LVGF_SUBSETITEMS;
	hGroupInfos.stateMask = 0xff;

	hGroupInfos.pszHeader = outInfos->strHeaderText;
	hGroupInfos.cchHeader = 64;
	hGroupInfos.pszFooter = outInfos->strFooterText;
	hGroupInfos.cchFooter = 64;
	hGroupInfos.pszSubtitle = outInfos->strSubTitleText;
	hGroupInfos.cchSubtitle = 64;
	hGroupInfos.pszTask = outInfos->strTaskLinkText;
	hGroupInfos.cchTask = 64;
	hGroupInfos.pszDescriptionTop = outInfos->strTopDescriptionText;
	hGroupInfos.cchDescriptionTop = 64;
	hGroupInfos.pszDescriptionBottom = outInfos->strBottomDescriptionText;
	hGroupInfos.cchDescriptionBottom = 64;
	hGroupInfos.pszSubsetTitle = outInfos->strSubSetTitleText;
	hGroupInfos.cchSubsetTitle = 64;

	ListView_GetGroupInfo( hHandle, iGroupID, &hGroupInfos );

	_Convert_GroupInfos( outInfos, &hGroupInfos );
}
Void WinGUITable::GetGroupInfosByIndex( WinGUITableGroupInfos * outInfos, UInt iIndex ) const
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;

	LVGROUP hGroupInfos;
	hGroupInfos.cbSize = sizeof(LVGROUP);
	hGroupInfos.mask = LVGF_GROUPID | LVGF_ITEMS | LVGF_HEADER | LVGF_FOOTER | LVGF_ALIGN | LVGF_STATE |
		LVGF_SUBTITLE | LVGF_TASK | LVGF_DESCRIPTIONTOP | LVGF_DESCRIPTIONBOTTOM |
		LVGF_TITLEIMAGE | LVGF_EXTENDEDIMAGE | LVGF_SUBSET; // | LVGF_SUBSETITEMS;
	hGroupInfos.stateMask = 0xff;

	hGroupInfos.pszHeader = outInfos->strHeaderText;
	hGroupInfos.cchHeader = 64;
	hGroupInfos.pszFooter = outInfos->strFooterText;
	hGroupInfos.cchFooter = 64;
	hGroupInfos.pszSubtitle = outInfos->strSubTitleText;
	hGroupInfos.cchSubtitle = 64;
	hGroupInfos.pszTask = outInfos->strTaskLinkText;
	hGroupInfos.cchTask = 64;
	hGroupInfos.pszDescriptionTop = outInfos->strTopDescriptionText;
	hGroupInfos.cchDescriptionTop = 64;
	hGroupInfos.pszDescriptionBottom = outInfos->strBottomDescriptionText;
	hGroupInfos.cchDescriptionBottom = 64;
	hGroupInfos.pszSubsetTitle = outInfos->strSubSetTitleText;
	hGroupInfos.cchSubsetTitle = 64;

	ListView_GetGroupInfoByIndex( hHandle, iIndex, &hGroupInfos );

	_Convert_GroupInfos( outInfos, &hGroupInfos );
}

Void WinGUITable::AddGroup( UInt iIndex, const WinGUITableGroupInfos * pGroupInfos )
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;

	LVGROUP hGroupInfos;
	hGroupInfos.cbSize = sizeof(LVGROUP);
	hGroupInfos.mask = LVGF_GROUPID | LVGF_ITEMS | LVGF_HEADER | LVGF_FOOTER | LVGF_ALIGN | LVGF_STATE |
		LVGF_SUBTITLE | LVGF_TASK | LVGF_DESCRIPTIONTOP | LVGF_DESCRIPTIONBOTTOM |
		LVGF_TITLEIMAGE | LVGF_EXTENDEDIMAGE | LVGF_SUBSET; // | LVGF_SUBSETITEMS;
	hGroupInfos.stateMask = 0xff;

	_Convert_GroupInfos( &hGroupInfos, pGroupInfos );

	ListView_InsertGroup( hHandle, iIndex, &hGroupInfos );
}
Void WinGUITable::AddGroup( const WinGUITableGroupInfos * pGroupInfos, WinGUITableGroupComparator pfComparator, Void * pUserData )
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;

	LVINSERTGROUPSORTED hInsertSorted;
	hInsertSorted.lvGroup.cbSize = sizeof(LVGROUP);
	hInsertSorted.lvGroup.mask = LVGF_GROUPID | LVGF_ITEMS | LVGF_HEADER | LVGF_FOOTER | LVGF_ALIGN | LVGF_STATE |
		LVGF_SUBTITLE | LVGF_TASK | LVGF_DESCRIPTIONTOP | LVGF_DESCRIPTIONBOTTOM |
		LVGF_TITLEIMAGE | LVGF_EXTENDEDIMAGE | LVGF_SUBSET; // | LVGF_SUBSETITEMS;
	hInsertSorted.lvGroup.stateMask = 0xff;

	_Convert_GroupInfos( &(hInsertSorted.lvGroup), pGroupInfos );

	hInsertSorted.pfnGroupCompare = pfComparator;
	hInsertSorted.pvData = pUserData;

	ListView_InsertGroupSorted( hHandle, &hInsertSorted );
}
Void WinGUITable::SetGroup( UInt iGroupID, const WinGUITableGroupInfos * pGroupInfos )
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;

	LVGROUP hGroupInfos;
	hGroupInfos.cbSize = sizeof(LVGROUP);
	hGroupInfos.mask = LVGF_GROUPID | LVGF_ITEMS | LVGF_HEADER | LVGF_FOOTER | LVGF_ALIGN | LVGF_STATE |
		LVGF_SUBTITLE | LVGF_TASK | LVGF_DESCRIPTIONTOP | LVGF_DESCRIPTIONBOTTOM |
		LVGF_TITLEIMAGE | LVGF_EXTENDEDIMAGE | LVGF_SUBSET; // | LVGF_SUBSETITEMS;
	hGroupInfos.stateMask = 0xff;

	_Convert_GroupInfos( &hGroupInfos, pGroupInfos );

	ListView_SetGroupInfo( hHandle, iGroupID, &hGroupInfos );
}
Void WinGUITable::RemoveGroup( UInt iGroupID )
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;
	ListView_RemoveGroup( hHandle, iGroupID );
}
Void WinGUITable::RemoveAllGroups()
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;
	ListView_RemoveAllGroups( hHandle );
}

Void WinGUITable::GetItemInfos( WinGUITableItemInfos * outInfos, UInt iIndex, UInt iSubItem ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_GROUPID | LVIF_TEXT | LVIF_PARAM | LVIF_COLUMNS | LVIF_COLFMT | LVIF_INDENT | LVIF_IMAGE | LVIF_STATE;
	hItemInfos.stateMask = 0xffffffff;

	hItemInfos.iItem = iIndex;
	hItemInfos.iSubItem = 0;
	if ( iSubItem != INVALID_OFFSET )
		hItemInfos.iSubItem = ( 1 + iSubItem );

	hItemInfos.pszText = outInfos->strLabelText;
	hItemInfos.cchTextMax = 64;

	hItemInfos.cColumns = 32;
	static UInt arrTempIndices[32];
	hItemInfos.puColumns = arrTempIndices;
	static Int arrTempFormats[32];
	hItemInfos.piColFmt = arrTempFormats;

	ListView_GetItem( hHandle, &hItemInfos );

	_Convert_ItemInfos( outInfos, &hItemInfos );
}

Void WinGUITable::GetTileInfos( WinGUITableTileInfos * outInfos, UInt iIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVTILEINFO hTileInfos;
	hTileInfos.cbSize = sizeof(LVTILEINFO);
	hTileInfos.iItem = iIndex;

	hTileInfos.cColumns = 32;
	static UInt arrTempIndices[32];
	hTileInfos.puColumns = arrTempIndices;
	static Int arrTempFormats[32];
	hTileInfos.piColFmt = arrTempFormats;

	ListView_GetTileInfo( hHandle, &hTileInfos );

	_Convert_TileInfos( outInfos, &hTileInfos );
}
Void WinGUITable::SetTileInfos( UInt iIndex, const WinGUITableTileInfos * pTileInfos )
{
	HWND hHandle = (HWND)m_hHandle;

	LVTILEINFO hTileInfos;
	hTileInfos.cbSize = sizeof(LVTILEINFO);

	hTileInfos.cColumns = 32;
	static UInt arrTempIndices[32];
	hTileInfos.puColumns = arrTempIndices;
	static Int arrTempFormats[32];
	hTileInfos.piColFmt = arrTempFormats;

	_Convert_TileInfos( &hTileInfos, pTileInfos );

	hTileInfos.iItem = iIndex;

	ListView_SetTileInfo( hHandle, &hTileInfos );
}

Bool WinGUITable::IsItemChecked( UInt iIndex ) const
{
	DebugAssert( m_bHasCheckBoxes == true );

	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetCheckState( hHandle, iIndex );
}
Void WinGUITable::CheckItem( UInt iIndex, Bool bChecked )
{
	DebugAssert( m_bHasCheckBoxes == true );

	HWND hHandle = (HWND)m_hHandle;
	ListView_SetCheckState( hHandle, iIndex, bChecked ? TRUE : FALSE );
}

Void * WinGUITable::EditItemLabelStart( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_EditLabel( hHandle, iIndex );
}
Void WinGUITable::EditItemLabelCancel()
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_CancelEditLabel( hHandle );
}

Void WinGUITable::SetItemLabelText( UInt iItemIndex, UInt iSubItemIndex, GChar * strLabelText )
{
	HWND hHandle = (HWND)m_hHandle;

	if ( iSubItemIndex == INVALID_OFFSET )
		iSubItemIndex = 0;
	else
		++iSubItemIndex;

	ListView_SetItemText( hHandle, iItemIndex, iSubItemIndex, strLabelText );
}

Void WinGUITable::UpdateItem( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_Update( hHandle, iIndex );
}

UInt WinGUITable::AssignItemID( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_MapIndexToID( hHandle, iIndex );
}
UInt WinGUITable::GetItemFromID( UInt iUniqueID )
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_MapIDToIndex( hHandle, iUniqueID );
}

UInt WinGUITable::GetIncrementalSearchStringLength() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ( ListView_GetISearchString( hHandle, NULL ) + 1 ); // Add NULLBYTE
}
Void WinGUITable::GetIncrementalSearchString( GChar * outSearchString ) const
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_GetISearchString( hHandle, outSearchString );
}

UInt WinGUITable::SearchItem( const GChar * strLabel, Bool bExact, UInt iStartIndex, Bool bWrapAround ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVFINDINFO hInfos;
	hInfos.flags = LVFI_STRING;
	if ( !bExact )
		hInfos.flags |= LVFI_PARTIAL;
	if ( bWrapAround )
		hInfos.flags |= LVFI_WRAP;
	hInfos.psz = strLabel;

	return ListView_FindItem( hHandle, ((Int)iStartIndex) - 1, &hInfos );
}
UInt WinGUITable::SearchItem( Void * pUserData, UInt iStartIndex, Bool bWrapAround ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVFINDINFO hInfos;
	hInfos.flags = LVFI_PARAM;
	if ( bWrapAround )
		hInfos.flags |= LVFI_WRAP;
	hInfos.lParam = (LPARAM)pUserData;

	return ListView_FindItem( hHandle, ((Int)iStartIndex) - 1, &hInfos );
}
UInt WinGUITable::SearchItem( const WinGUIPoint * pPoint, KeyCode iDirection, UInt iStartIndex, Bool bWrapAround ) const
{
	DebugAssert( m_iViewMode == WINGUI_TABLE_VIEW_ICONS || m_iViewMode == WINGUI_TABLE_VIEW_ICONS_SMALL );

	HWND hHandle = (HWND)m_hHandle;

	LVFINDINFO hInfos;
	hInfos.flags = LVFI_NEARESTXY;
	if ( bWrapAround )
		hInfos.flags |= LVFI_WRAP;
	hInfos.pt.x = pPoint->iX;
	hInfos.pt.y = pPoint->iY;
	switch( iDirection ) {
		case KEYCODE_UP:    hInfos.vkDirection = VK_UP; break;
		case KEYCODE_DOWN:  hInfos.vkDirection = VK_DOWN; break;
		case KEYCODE_LEFT:  hInfos.vkDirection = VK_LEFT; break;
		case KEYCODE_RIGHT: hInfos.vkDirection = VK_RIGHT; break;
		default: DebugAssert(false); break;
	}

	return ListView_FindItem( hHandle, ((Int)iStartIndex) - 1, &hInfos );
}

UInt WinGUITable::SearchNextItem( UInt iStartIndex, Bool bReverse, Bool bSameGroup, Bool bVisible, Bool bHasFocus, Bool bSelected, Bool bCutMarked, Bool bDropHighlight ) const
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iFlags = LVNI_ALL;
	if ( bReverse )
		iFlags |= LVNI_PREVIOUS;

	if ( bSameGroup )
		iFlags |= LVNI_SAMEGROUPONLY;
	if ( bVisible )
		iFlags |= LVNI_VISIBLEONLY;
	if ( bHasFocus )
		iFlags |= LVNI_FOCUSED;
	if ( bSelected )
		iFlags |= LVNI_SELECTED;
	if ( bCutMarked )
		iFlags |= LVNI_CUT;
	if ( bDropHighlight )
		iFlags |= LVNI_DROPHILITED;

	return ListView_GetNextItem( hHandle, ((Int)iStartIndex) - 1, iFlags );
}
UInt WinGUITable::SearchNextItem( UInt iStartIndex, KeyCode iDirection, Bool bSameGroup, Bool bVisible, Bool bHasFocus, Bool bSelected, Bool bCutMarked, Bool bDropHighlight ) const
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iFlags = 0;
	switch( iDirection ) {
		case KEYCODE_UP:    iFlags |= LVNI_ABOVE; break;
		case KEYCODE_DOWN:  iFlags |= LVNI_BELOW; break;
		case KEYCODE_LEFT:  iFlags |= LVNI_TOLEFT; break;
		case KEYCODE_RIGHT: iFlags |= LVNI_TORIGHT; break;
		default: DebugAssert(false); break;
	}

	if ( bSameGroup )
		iFlags |= LVNI_SAMEGROUPONLY;
	if ( bVisible )
		iFlags |= LVNI_VISIBLEONLY;
	if ( bHasFocus )
		iFlags |= LVNI_FOCUSED;
	if ( bSelected )
		iFlags |= LVNI_SELECTED;
	if ( bCutMarked )
		iFlags |= LVNI_CUT;
	if ( bDropHighlight )
		iFlags |= LVNI_DROPHILITED;

	return ListView_GetNextItem( hHandle, ((Int)iStartIndex) - 1, iFlags );
}

UInt WinGUITable::SearchNextItem( UInt iStartGroupIndex, UInt iStartIndex, Bool bReverse, Bool bSameGroup, Bool bVisible, Bool bHasFocus, Bool bSelected, Bool bCutMarked, Bool bDropHighlight ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVITEMINDEX hItemIndex;
	hItemIndex.iGroup = iStartGroupIndex;
	hItemIndex.iItem = ((Int)iStartIndex) - 1;

	UInt iFlags = 0;
	if ( bReverse )
		iFlags |= LVNI_PREVIOUS;

	if ( bSameGroup )
		iFlags |= LVNI_SAMEGROUPONLY;
	if ( bVisible )
		iFlags |= LVNI_VISIBLEONLY;
	if ( bHasFocus )
		iFlags |= LVNI_FOCUSED;
	if ( bSelected )
		iFlags |= LVNI_SELECTED;
	if ( bCutMarked )
		iFlags |= LVNI_CUT;
	if ( bDropHighlight )
		iFlags |= LVNI_DROPHILITED;

	return ListView_GetNextItemIndex( hHandle, &hItemIndex, iFlags );
}
UInt WinGUITable::SearchNextItem( UInt iStartGroupIndex, UInt iStartIndex, KeyCode iDirection, Bool bSameGroup, Bool bVisible, Bool bHasFocus, Bool bSelected, Bool bCutMarked, Bool bDropHighlight ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVITEMINDEX hItemIndex;
	hItemIndex.iGroup = iStartGroupIndex;
	hItemIndex.iItem = ((Int)iStartIndex) - 1;

	UInt iFlags = 0;
	switch( iDirection ) {
		case KEYCODE_UP:    iFlags |= LVNI_ABOVE; break;
		case KEYCODE_DOWN:  iFlags |= LVNI_BELOW; break;
		case KEYCODE_LEFT:  iFlags |= LVNI_TOLEFT; break;
		case KEYCODE_RIGHT: iFlags |= LVNI_TORIGHT; break;
		default: DebugAssert(false); break;
	}

	if ( bSameGroup )
		iFlags |= LVNI_SAMEGROUPONLY;
	if ( bVisible )
		iFlags |= LVNI_VISIBLEONLY;
	if ( bHasFocus )
		iFlags |= LVNI_FOCUSED;
	if ( bSelected )
		iFlags |= LVNI_SELECTED;
	if ( bCutMarked )
		iFlags |= LVNI_CUT;
	if ( bDropHighlight )
		iFlags |= LVNI_DROPHILITED;

	return ListView_GetNextItemIndex( hHandle, &hItemIndex, iFlags );
}

UInt WinGUITable::GetHotItem() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetHotItem( hHandle );
}
Void WinGUITable::SetHotItem( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetHotItem( hHandle, iIndex );
}

UInt WinGUITable::GetHoverTime() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetHoverTime( hHandle );
}
Void WinGUITable::SetHoverTime( UInt iTimeMS )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetHoverTime( hHandle, iTimeMS );
}

Void WinGUITable::HitTest( WinGUITableHitTestResult * outResult, const WinGUIPoint & hPoint ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVHITTESTINFO hHitTestInfos;
	hHitTestInfos.pt.x = hPoint.iX;
	hHitTestInfos.pt.y = hPoint.iY;

	ListView_HitTestEx( hHandle, &hHitTestInfos );

	outResult->hPoint.iX = hPoint.iX;
	outResult->hPoint.iY = hPoint.iY;

	outResult->iGroupIndex = hHitTestInfos.iGroup;
	outResult->iItemIndex = hHitTestInfos.iItem;
	outResult->iSubItemIndex = ( hHitTestInfos.iSubItem - 1 );

	outResult->bOutsideAbove = ( (hHitTestInfos.flags & LVHT_ABOVE) != 0 );
	outResult->bOutsideBelow = ( (hHitTestInfos.flags & LVHT_BELOW) != 0 );
	outResult->bOutsideLeft = ( (hHitTestInfos.flags & LVHT_TOLEFT) != 0 );
	outResult->bOutsideRight = ( (hHitTestInfos.flags & LVHT_TORIGHT) != 0 );

	outResult->bInsideNoWhere = ( (hHitTestInfos.flags & LVHT_NOWHERE) != 0 );

	outResult->bOnItem = ( (hHitTestInfos.flags & LVHT_EX_ONCONTENTS) != 0 );
	outResult->bOnItemIcon = ( (hHitTestInfos.flags & LVHT_ONITEMICON) != 0 );
	outResult->bOnItemLabel = ( (hHitTestInfos.flags & LVHT_ONITEMLABEL) != 0 );
	outResult->bOnItemStateIcon = ( (hHitTestInfos.flags & LVHT_ONITEMSTATEICON) != 0 );

	outResult->bOnGroup = ( (hHitTestInfos.flags & LVHT_EX_GROUP) != 0 );
	outResult->bOnGroupHeader = ( (hHitTestInfos.flags & LVHT_EX_GROUP_HEADER) != 0 );
	outResult->bOnGroupFooter = ( (hHitTestInfos.flags & LVHT_EX_GROUP_FOOTER) != 0 );
	outResult->bOnGroupBackground = ( (hHitTestInfos.flags & LVHT_EX_GROUP_BACKGROUND) != 0 );
	outResult->bOnGroupExpandCollapse = ( (hHitTestInfos.flags & LVHT_EX_GROUP_COLLAPSE) != 0 );
	outResult->bOnGroupStateIcon = ( (hHitTestInfos.flags & LVHT_EX_GROUP_STATEICON) != 0 );
	outResult->bOnGroupSubSetLink = ( (hHitTestInfos.flags & LVHT_EX_GROUP_SUBSETLINK) != 0 );

	outResult->bOnFooter = ( (hHitTestInfos.flags & LVHT_EX_FOOTER) != 0 );
}

Void WinGUITable::SortGroups( WinGUITableGroupComparator pfComparator, Void * pUserData )
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;
	ListView_SortGroups( hHandle, pfComparator, pUserData );
}
Void WinGUITable::SortItemsByIndex( WinGUITableItemComparator pfComparator, Void * pUserData )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SortItemsEx( hHandle, pfComparator, pUserData );
}
Void WinGUITable::SortItemsByData( WinGUITableItemComparator pfComparator, Void * pUserData )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SortItems( hHandle, pfComparator, pUserData );
}

Void WinGUITable::SetInfoTip( UInt iItemIndex, UInt iSubItemIndex, GChar * strInfoText )
{
	HWND hHandle = (HWND)m_hHandle;

	LVSETINFOTIP hInfoTip;
	hInfoTip.cbSize = sizeof(LVSETINFOTIP);
	hInfoTip.dwFlags = 0;
	hInfoTip.iItem = iItemIndex;
	hInfoTip.iSubItem = 0;
	if ( iSubItemIndex != INVALID_OFFSET )
		hInfoTip.iSubItem = ( 1 + iSubItemIndex );
	hInfoTip.pszText = strInfoText;

	ListView_SetInfoTip( hHandle, &hInfoTip );
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUITable::_Convert_ColumnInfos( WinGUITableColumnInfos * outColumnInfos, const Void * pColumnInfos ) const
{
	const LVCOLUMN * pDesc = (const LVCOLUMN *)pColumnInfos;

	outColumnInfos->iOrderIndex = pDesc->iOrder;
	outColumnInfos->iSubItemIndex = ( pDesc->iSubItem - 1 );

	switch( pDesc->fmt & LVCFMT_JUSTIFYMASK ) {
		case LVCFMT_LEFT:   outColumnInfos->iRowsTextAlign = WINGUI_TABLE_TEXT_ALIGN_LEFT; break;
		case LVCFMT_RIGHT:  outColumnInfos->iRowsTextAlign = WINGUI_TABLE_TEXT_ALIGN_RIGHT; break;
		case LVCFMT_CENTER: outColumnInfos->iRowsTextAlign = WINGUI_TABLE_TEXT_ALIGN_CENTER; break;
		default: DebugAssert(false); break;
	}

	outColumnInfos->bHeaderSplitButton = ( (pDesc->fmt & LVCFMT_SPLITBUTTON) != 0 );

	outColumnInfos->bHeaderHasImage = ( (pDesc->fmt & LVCFMT_COL_HAS_IMAGES) != 0 );
	outColumnInfos->bRowsHaveImages = ( (pDesc->fmt & LVCFMT_IMAGE) != 0 );
	outColumnInfos->bIsImageOnRight = ( (pDesc->fmt & LVCFMT_BITMAP_ON_RIGHT) != 0 );
	outColumnInfos->iImageListIndex = pDesc->iImage;

	outColumnInfos->bFixedWidth = ( (pDesc->fmt & LVCFMT_FIXED_WIDTH) != 0 );
	outColumnInfos->bFixedAspectRatio = ( (pDesc->fmt & LVCFMT_FIXED_RATIO) != 0 );
	outColumnInfos->iWidth = pDesc->cx;
	outColumnInfos->iMinWidth = pDesc->cxMin;
	outColumnInfos->iDefaultWidth = pDesc->cxDefault;
	outColumnInfos->iIdealWidth = pDesc->cxIdeal;
}
Void WinGUITable::_Convert_ColumnInfos( Void * outColumnInfos, const WinGUITableColumnInfos * pColumnInfos ) const
{
	LVCOLUMN * outDesc = (LVCOLUMN*)outColumnInfos;

	outDesc->iOrder = pColumnInfos->iOrderIndex;
	outDesc->iSubItem = ( 1 + pColumnInfos->iSubItemIndex );

	outDesc->pszText = pColumnInfos->strHeaderText;
	outDesc->cchTextMax = 64;

	outDesc->fmt = 0;
	switch( pColumnInfos->iRowsTextAlign ) {
		case WINGUI_TABLE_TEXT_ALIGN_LEFT:   outDesc->fmt |= LVCFMT_LEFT; break;
		case WINGUI_TABLE_TEXT_ALIGN_RIGHT:  outDesc->fmt |= LVCFMT_RIGHT; break;
		case WINGUI_TABLE_TEXT_ALIGN_CENTER: outDesc->fmt |= LVCFMT_CENTER; break;
		default: DebugAssert(false); break;
	}

	if ( pColumnInfos->bHeaderSplitButton )
		outDesc->fmt |= LVCFMT_SPLITBUTTON;

	if ( pColumnInfos->bHeaderHasImage )
		outDesc->fmt |= LVCFMT_COL_HAS_IMAGES;
	if ( pColumnInfos->bRowsHaveImages )
		outDesc->fmt |= LVCFMT_IMAGE;
	if ( pColumnInfos->bIsImageOnRight )
		outDesc->fmt |= LVCFMT_BITMAP_ON_RIGHT;
	outDesc->iImage = pColumnInfos->iImageListIndex;

	if ( pColumnInfos->bFixedWidth )
		outDesc->fmt |= LVCFMT_FIXED_WIDTH;
	if ( pColumnInfos->bFixedAspectRatio )
		outDesc->fmt |= LVCFMT_FIXED_RATIO;
	outDesc->cx = pColumnInfos->iWidth;
	outDesc->cxMin = pColumnInfos->iMinWidth;
	outDesc->cxDefault = pColumnInfos->iDefaultWidth;
	outDesc->cxIdeal = pColumnInfos->iIdealWidth;
}

Void WinGUITable::_Convert_GroupInfos( WinGUITableGroupInfos * outGroupInfos, const Void * pGroupInfos ) const
{
	const LVGROUP * pDesc = (const LVGROUP *)pGroupInfos;

	outGroupInfos->iGroupID = pDesc->iGroupId;

	outGroupInfos->iFirstItemIndex = pDesc->iFirstItem;
	outGroupInfos->iItemCount = pDesc->cItems;

	outGroupInfos->bHasSubSets = ( (pDesc->state & LVGS_SUBSETED) != 0 );

	outGroupInfos->bHasHeader = ( (pDesc->state & LVGS_NOHEADER) == 0 );

	switch( pDesc->uAlign & 0x07 ) {
		case LVGA_HEADER_LEFT:   outGroupInfos->iHeaderTextAlign = WINGUI_TABLE_TEXT_ALIGN_LEFT; break;
		case LVGA_HEADER_RIGHT:  outGroupInfos->iHeaderTextAlign = WINGUI_TABLE_TEXT_ALIGN_RIGHT; break;
		case LVGA_HEADER_CENTER: outGroupInfos->iHeaderTextAlign = WINGUI_TABLE_TEXT_ALIGN_CENTER; break;
		default: DebugAssert(false); break;
	}
	switch( pDesc->uAlign & 0x38 ) {
		case LVGA_FOOTER_LEFT:   outGroupInfos->iFooterTextAlign = WINGUI_TABLE_TEXT_ALIGN_LEFT; break;
		case LVGA_FOOTER_RIGHT:  outGroupInfos->iFooterTextAlign = WINGUI_TABLE_TEXT_ALIGN_RIGHT; break;
		case LVGA_FOOTER_CENTER: outGroupInfos->iFooterTextAlign = WINGUI_TABLE_TEXT_ALIGN_CENTER; break;
		default: DebugAssert(false); break;
	}

	outGroupInfos->iTitleImageIndex = pDesc->iTitleImage;
	outGroupInfos->iExtendedImageIndex = pDesc->iExtendedImage;

	outGroupInfos->bCanCollapse = ( (pDesc->state & LVGS_COLLAPSIBLE) != 0 );
	outGroupInfos->bCollapsed = ( (pDesc->state & LVGS_COLLAPSED) != 0 );

	outGroupInfos->bHasFocus = ( (pDesc->state & LVGS_FOCUSED) != 0 );
	outGroupInfos->bSubSetHasFocus = ( (pDesc->state & LVGS_SUBSETLINKFOCUSED) != 0 );
	outGroupInfos->bIsSelected = ( (pDesc->state & LVGS_SELECTED) != 0 );

	outGroupInfos->bHidden = ( (pDesc->state & LVGS_HIDDEN) != 0 );
}
Void WinGUITable::_Convert_GroupInfos( Void * outGroupInfos, const WinGUITableGroupInfos * pGroupInfos ) const
{
	LVGROUP * outDesc = (LVGROUP*)outGroupInfos;

	outDesc->iGroupId = pGroupInfos->iGroupID;

	outDesc->iFirstItem = pGroupInfos->iFirstItemIndex;
	outDesc->cItems = pGroupInfos->iItemCount;

	outDesc->state = 0;
	if ( pGroupInfos->bHasSubSets )
		outDesc->state |= LVGS_SUBSETED;

	if ( !(pGroupInfos->bHasHeader) )
		outDesc->state |= LVGS_NOHEADER;
	outDesc->pszHeader = pGroupInfos->strHeaderText;
	outDesc->cchHeader = 64;

	outDesc->pszFooter = pGroupInfos->strFooterText;
	outDesc->cchFooter = 64;

	outDesc->pszSubtitle = pGroupInfos->strSubTitleText;
	outDesc->cchSubtitle = 64;
	outDesc->pszTask = pGroupInfos->strTaskLinkText;
	outDesc->cchTask = 64;
	outDesc->pszDescriptionTop = pGroupInfos->strTopDescriptionText;
	outDesc->cchDescriptionTop = 64;
	outDesc->pszDescriptionBottom = pGroupInfos->strBottomDescriptionText;
	outDesc->cchDescriptionBottom = 64;

	outDesc->pszSubsetTitle = pGroupInfos->strSubSetTitleText;
	outDesc->cchSubsetTitle = 64;

	outDesc->uAlign = 0;
	switch( pGroupInfos->iHeaderTextAlign ) {
		case WINGUI_TABLE_TEXT_ALIGN_LEFT:   outDesc->uAlign |= LVGA_HEADER_LEFT; break;
		case WINGUI_TABLE_TEXT_ALIGN_RIGHT:  outDesc->uAlign |= LVGA_HEADER_RIGHT; break;
		case WINGUI_TABLE_TEXT_ALIGN_CENTER: outDesc->uAlign |= LVGA_HEADER_CENTER; break;
		default: DebugAssert(false); break;
	}
	switch( pGroupInfos->iFooterTextAlign ) {
		case WINGUI_TABLE_TEXT_ALIGN_LEFT:   outDesc->uAlign |= LVGA_FOOTER_LEFT; break;
		case WINGUI_TABLE_TEXT_ALIGN_RIGHT:  outDesc->uAlign |= LVGA_FOOTER_RIGHT; break;
		case WINGUI_TABLE_TEXT_ALIGN_CENTER: outDesc->uAlign |= LVGA_FOOTER_CENTER; break;
		default: DebugAssert(false); break;
	}

	outDesc->iTitleImage = pGroupInfos->iTitleImageIndex;
	outDesc->iExtendedImage = pGroupInfos->iExtendedImageIndex;

	if ( pGroupInfos->bCanCollapse )
		outDesc->state |= LVGS_COLLAPSIBLE;
	if ( pGroupInfos->bCollapsed )
		outDesc->state |= LVGS_COLLAPSED;

	if ( pGroupInfos->bHasFocus )
		outDesc->state |= LVGS_FOCUSED;
	if ( pGroupInfos->bSubSetHasFocus )
		outDesc->state |= LVGS_SUBSETLINKFOCUSED;
	if ( pGroupInfos->bIsSelected )
		outDesc->state |= LVGS_SELECTED;

	if ( pGroupInfos->bHidden )
		outDesc->state |= LVGS_HIDDEN;
}

Void WinGUITable::_Convert_ItemInfos( WinGUITableItemInfos * outItemInfos, const Void * pItemInfos ) const
{
	const LVITEM * pDesc = (const LVITEM *)pItemInfos;

	outItemInfos->bIsSubItem = ( pDesc->iSubItem != 0 );
	outItemInfos->iItemIndex = pDesc->iItem;
	outItemInfos->iSubItemIndex = INVALID_OFFSET;
	if ( outItemInfos->bIsSubItem )
		outItemInfos->iSubItemIndex = ( pDesc->iSubItem - 1 );

	outItemInfos->iParentGroupID = pDesc->iGroupId;

	outItemInfos->pUserData = (Void*)( pDesc->lParam );

	outItemInfos->iColumnCount = pDesc->cColumns;
	for ( UInt i = 0; i < pDesc->cColumns; ++i ) {
		outItemInfos->arrColumns[i].iIndex = ( pDesc->puColumns[i] - 1 );
		outItemInfos->arrColumns[i].bLineBreak = ( (pDesc->piColFmt[i] & LVCFMT_LINE_BREAK) != 0 );
		outItemInfos->arrColumns[i].bFill = ( (pDesc->piColFmt[i] & LVCFMT_FILL) != 0 );
		outItemInfos->arrColumns[i].bAllowWrap = ( (pDesc->piColFmt[i] & LVCFMT_WRAP) != 0 );
		outItemInfos->arrColumns[i].bNoTitle = ( (pDesc->piColFmt[i] & LVCFMT_NO_TITLE) != 0 );
	}

	outItemInfos->iIndentDepth = pDesc->iIndent;

	outItemInfos->iIconImage = pDesc->iImage;
	outItemInfos->iOverlayImage = ( pDesc->state & LVIS_OVERLAYMASK ) >> 8;
	if ( outItemInfos->iOverlayImage == 0 )
		outItemInfos->iOverlayImage = INVALID_OFFSET;
	else
		--(outItemInfos->iOverlayImage);
	outItemInfos->iStateImage = ( pDesc->state & LVIS_STATEIMAGEMASK ) >> 12;
	if ( outItemInfos->iStateImage == 0 )
		outItemInfos->iStateImage = INVALID_OFFSET;
	else
		--(outItemInfos->iStateImage);

	outItemInfos->bHasFocus = ( (pDesc->state & LVIS_FOCUSED) != 0 );
	outItemInfos->bSelected = ( (pDesc->state & LVIS_SELECTED) != 0 );

	outItemInfos->bCutMarked = ( (pDesc->state & LVIS_CUT) != 0 );
	outItemInfos->bDropHighlight = ( (pDesc->state & LVIS_DROPHILITED) != 0 );
}
Void WinGUITable::_Convert_ItemInfos( Void * outItemInfos, const WinGUITableItemInfos * pItemInfos ) const
{
	LVITEM * outDesc = (LVITEM*)outItemInfos;

	outDesc->iItem = pItemInfos->iItemIndex;
	outDesc->iSubItem = 0;
	if ( pItemInfos->bIsSubItem )
		outDesc->iSubItem = ( 1 + pItemInfos->iSubItemIndex );

	outDesc->iGroupId = pItemInfos->iParentGroupID;

	outDesc->pszText = pItemInfos->strLabelText;
	outDesc->cchTextMax = 64;

	outDesc->lParam = (LPARAM)( pItemInfos->pUserData );

	outDesc->cColumns = pItemInfos->iColumnCount;
	for ( UInt i = 0; i < pItemInfos->iColumnCount; ++i ) {
		outDesc->puColumns[i] = ( 1 + pItemInfos->arrColumns[i].iIndex );
		outDesc->piColFmt[i] = 0;
		if ( pItemInfos->arrColumns[i].bLineBreak )
			outDesc->piColFmt[i] |= LVCFMT_LINE_BREAK;
		if ( pItemInfos->arrColumns[i].bFill )
			outDesc->piColFmt[i] |= LVCFMT_FILL;
		if ( pItemInfos->arrColumns[i].bAllowWrap )
			outDesc->piColFmt[i] |= LVCFMT_WRAP;
		if ( pItemInfos->arrColumns[i].bNoTitle )
			outDesc->piColFmt[i] |= LVCFMT_NO_TITLE;
	}

	outDesc->iIndent = pItemInfos->iIndentDepth;

	outDesc->iImage = pItemInfos->iIconImage;

	outDesc->state = 0;
	if ( pItemInfos->iOverlayImage != INVALID_OFFSET )
		outDesc->state |= ( ((1 + pItemInfos->iOverlayImage) & 0x0f) << 8 );
	if ( pItemInfos->iStateImage != INVALID_OFFSET )
		outDesc->state |= ( ((1 + pItemInfos->iStateImage) & 0x0f) << 12 );

	if ( pItemInfos->bHasFocus )
		outDesc->state |= LVIS_FOCUSED;
	if ( pItemInfos->bSelected )
		outDesc->state |= LVIS_SELECTED;

	if ( pItemInfos->bCutMarked )
		outDesc->state |= LVIS_CUT;
	if ( pItemInfos->bDropHighlight )
		outDesc->state |= LVIS_DROPHILITED;
}

Void WinGUITable::_Convert_TileInfos( WinGUITableTileInfos * outTileInfos, const Void * pTileInfos ) const
{
	const LVTILEINFO * pDesc = (const LVTILEINFO *)pTileInfos;

	outTileInfos->iItemIndex = pDesc->iItem;

	outTileInfos->iColumnCount = pDesc->cColumns;
	for ( UInt i = 0; i < pDesc->cColumns; ++i ) {
		outTileInfos->arrColumns[i].iIndex = ( pDesc->puColumns[i] - 1 );
		outTileInfos->arrColumns[i].bLineBreak = ( (pDesc->piColFmt[i] & LVCFMT_LINE_BREAK) != 0 );
		outTileInfos->arrColumns[i].bFill = ( (pDesc->piColFmt[i] & LVCFMT_FILL) != 0 );
		outTileInfos->arrColumns[i].bAllowWrap = ( (pDesc->piColFmt[i] & LVCFMT_WRAP) != 0 );
		outTileInfos->arrColumns[i].bNoTitle = ( (pDesc->piColFmt[i] & LVCFMT_NO_TITLE) != 0 );
	}
}
Void WinGUITable::_Convert_TileInfos( Void * outTileInfos, const WinGUITableTileInfos * pTileInfos ) const
{
	LVTILEINFO * outDesc = (LVTILEINFO*)outTileInfos;

	outDesc->iItem = pTileInfos->iItemIndex;

	outDesc->cColumns = pTileInfos->iColumnCount;
	for ( UInt i = 0; i < pTileInfos->iColumnCount; ++i ) {
		outDesc->puColumns[i] = ( 1 + pTileInfos->arrColumns[i].iIndex );
		outDesc->piColFmt[i] = 0;
		if ( pTileInfos->arrColumns[i].bLineBreak )
			outDesc->piColFmt[i] |= LVCFMT_LINE_BREAK;
		if ( pTileInfos->arrColumns[i].bFill )
			outDesc->piColFmt[i] |= LVCFMT_FILL;
		if ( pTileInfos->arrColumns[i].bAllowWrap )
			outDesc->piColFmt[i] |= LVCFMT_WRAP;
		if ( pTileInfos->arrColumns[i].bNoTitle )
			outDesc->piColFmt[i] |= LVCFMT_NO_TITLE;
	}
}

Void WinGUITable::_Convert_TileMetrics( WinGUITableTileMetrics * outTileMetrics, const Void * pTileMetrics ) const
{
	const LVTILEVIEWINFO * pDesc = (const LVTILEVIEWINFO *)pTileMetrics;

	switch( pDesc->dwFlags ) {
		case LVTVIF_AUTOSIZE:    outTileMetrics->iSizeMode = WINGUI_TABLE_TILES_AUTOSIZE; break;
		case LVTVIF_FIXEDWIDTH:  outTileMetrics->iSizeMode = WINGUI_TABLE_TILES_FIXED_WIDTH; break;
		case LVTVIF_FIXEDHEIGHT: outTileMetrics->iSizeMode = WINGUI_TABLE_TILES_FIXED_HEIGHT; break;
		case LVTVIF_FIXEDSIZE:   outTileMetrics->iSizeMode = WINGUI_TABLE_TILES_FIXED_SIZE; break;
		default: DebugAssert( false ); break;
	}

	outTileMetrics->iWidth = pDesc->sizeTile.cx;
	outTileMetrics->iHeight = pDesc->sizeTile.cy;
	outTileMetrics->iMaxItemLabelLines = pDesc->cLines;
	
	outTileMetrics->hItemLabelMargin.iLeft = pDesc->rcLabelMargin.left;
	outTileMetrics->hItemLabelMargin.iTop = pDesc->rcLabelMargin.top;
	outTileMetrics->hItemLabelMargin.iWidth = ( pDesc->rcLabelMargin.right - pDesc->rcLabelMargin.left );
	outTileMetrics->hItemLabelMargin.iHeight = ( pDesc->rcLabelMargin.bottom - pDesc->rcLabelMargin.top );
}
Void WinGUITable::_Convert_TileMetrics( Void * outTileMetrics, const WinGUITableTileMetrics * pTileMetrics ) const
{
	LVTILEVIEWINFO * outDesc = (LVTILEVIEWINFO*)outTileMetrics;

	switch( pTileMetrics->iSizeMode ) {
		case WINGUI_TABLE_TILES_AUTOSIZE:     outDesc->dwFlags = LVTVIF_AUTOSIZE; break;
		case WINGUI_TABLE_TILES_FIXED_WIDTH:  outDesc->dwFlags = LVTVIF_FIXEDWIDTH; break;
		case WINGUI_TABLE_TILES_FIXED_HEIGHT: outDesc->dwFlags = LVTVIF_FIXEDHEIGHT; break;
		case WINGUI_TABLE_TILES_FIXED_SIZE:   outDesc->dwFlags = LVTVIF_FIXEDSIZE; break;
		default: DebugAssert( false ); break;
	}

	outDesc->sizeTile.cx = pTileMetrics->iWidth;
	outDesc->sizeTile.cy = pTileMetrics->iHeight;
	outDesc->cLines = pTileMetrics->iMaxItemLabelLines;

	outDesc->rcLabelMargin.left = pTileMetrics->hItemLabelMargin.iLeft;
	outDesc->rcLabelMargin.top = pTileMetrics->hItemLabelMargin.iTop;
	outDesc->rcLabelMargin.right = ( pTileMetrics->hItemLabelMargin.iLeft + pTileMetrics->hItemLabelMargin.iWidth );
	outDesc->rcLabelMargin.bottom = ( pTileMetrics->hItemLabelMargin.iTop + pTileMetrics->hItemLabelMargin.iHeight );
}

Void WinGUITable::_Create()
{
	DebugAssert( m_hHandle == NULL );

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUITableModel * pModel = (WinGUITableModel*)m_pModel;

	// Compute Layout
    const WinGUILayout * pLayout = pModel->GetLayout();

    WinGUIRectangle hParentRect;
    m_pParent->GetClientRect( &hParentRect );

    WinGUIRectangle hWindowRect;
    pLayout->ComputeLayout( &hWindowRect, hParentRect );

	// Get Creation Parameters
    const WinGUITableParameters * pParameters = pModel->GetCreationParameters();

	// Save State
	m_bVirtualTable = pParameters->bMakeVirtualTable;
	m_iViewMode = pParameters->iViewMode;
	m_bHasCheckBoxes = pParameters->bAddCheckBoxes;

    // Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE );
	DWord dwStyleEx = LVS_EX_LABELTIP;

	if ( pParameters->bMakeVirtualTable )
		dwStyle |= LVS_OWNERDATA;
	if ( pParameters->bHeadersInAllViews )
		dwStyleEx |= LVS_EX_HEADERINALLVIEWS;

	switch( pParameters->iViewMode ) {
		case WINGUI_TABLE_VIEW_LIST:
			dwStyle |= LVS_LIST;

			if ( pParameters->bHeadersInAllViews ) {
				if ( pParameters->bStaticColumnHeaders )
					dwStyle |= LVS_NOSORTHEADER;
				if ( pParameters->bSnapColumnsWidth )
					dwStyleEx |= LVS_EX_COLUMNSNAPPOINTS;
				if ( pParameters->bAutoSizeColumns )
					dwStyleEx |= LVS_EX_AUTOSIZECOLUMNS;
			}
			break;
		case WINGUI_TABLE_VIEW_ICONS:
			dwStyle |= LVS_ICON;
			if ( pParameters->hIconsMode.iAlign == WINGUI_TABLE_ICONS_ALIGN_TOP )
				dwStyle |= LVS_ALIGNTOP;
			else if ( pParameters->hIconsMode.iAlign == WINGUI_TABLE_ICONS_ALIGN_LEFT )
				dwStyle |= LVS_ALIGNLEFT;

			if ( pParameters->hIconsMode.bAutoArrange )
				dwStyle |= LVS_AUTOARRANGE;

			if ( pParameters->hIconsMode.bHideLabels )
				dwStyleEx |= LVS_EX_HIDELABELS;
			else if ( pParameters->hIconsMode.bNoLabelWrap )
				dwStyle |= LVS_NOLABELWRAP;

			if ( pParameters->hIconsMode.bJustifiedColumns )
				dwStyleEx |= LVS_EX_JUSTIFYCOLUMNS;
			if ( pParameters->hIconsMode.bSnapToGrid )
				dwStyleEx |= LVS_EX_SNAPTOGRID;

			if ( pParameters->hIconsMode.bSimpleSelection )
				dwStyleEx |= LVS_EX_SIMPLESELECT;

			if ( pParameters->bHeadersInAllViews ) {
				if ( pParameters->bStaticColumnHeaders )
					dwStyle |= LVS_NOSORTHEADER;
				if ( pParameters->bSnapColumnsWidth )
					dwStyleEx |= LVS_EX_COLUMNSNAPPOINTS;
				if ( pParameters->bAutoSizeColumns )
					dwStyleEx |= LVS_EX_AUTOSIZECOLUMNS;
				if ( pParameters->hIconsMode.bColumnOverflow )
					dwStyleEx |= LVS_EX_COLUMNOVERFLOW;
			}
			break;
		case WINGUI_TABLE_VIEW_ICONS_SMALL:
			dwStyle |= LVS_SMALLICON;
			if ( pParameters->hSmallIconsMode.iAlign == WINGUI_TABLE_ICONS_ALIGN_TOP )
				dwStyle |= LVS_ALIGNTOP;
			else if ( pParameters->hSmallIconsMode.iAlign == WINGUI_TABLE_ICONS_ALIGN_LEFT )
				dwStyle |= LVS_ALIGNLEFT;

			if ( pParameters->hSmallIconsMode.bAutoArrange )
				dwStyle |= LVS_AUTOARRANGE;

			if ( pParameters->hSmallIconsMode.bHideLabels )
				dwStyleEx |= LVS_EX_HIDELABELS;
			else if ( pParameters->hSmallIconsMode.bNoLabelWrap )
				dwStyle |= LVS_NOLABELWRAP;

			if ( pParameters->hSmallIconsMode.bJustifiedColumns )
				dwStyleEx |= LVS_EX_JUSTIFYCOLUMNS;
			if ( pParameters->hSmallIconsMode.bSnapToGrid )
				dwStyleEx |= LVS_EX_SNAPTOGRID;

			if ( pParameters->hSmallIconsMode.bSimpleSelection )
				dwStyleEx |= LVS_EX_SIMPLESELECT;

			if ( pParameters->bHeadersInAllViews ) {
				if ( pParameters->bStaticColumnHeaders )
					dwStyle |= LVS_NOSORTHEADER;
				if ( pParameters->bSnapColumnsWidth )
					dwStyleEx |= LVS_EX_COLUMNSNAPPOINTS;
				if ( pParameters->bAutoSizeColumns )
					dwStyleEx |= LVS_EX_AUTOSIZECOLUMNS;
				if ( pParameters->hSmallIconsMode.bColumnOverflow )
					dwStyleEx |= LVS_EX_COLUMNOVERFLOW;
			}
			break;
		case WINGUI_TABLE_VIEW_DETAILED:
			dwStyle |= LVS_REPORT;
			if ( pParameters->hDetailedMode.bNoColumnHeaders )
				dwStyle |= LVS_NOCOLUMNHEADER;
			else {
				if ( pParameters->bStaticColumnHeaders )
					dwStyle |= LVS_NOSORTHEADER;
				if ( pParameters->bSnapColumnsWidth )
					dwStyleEx |= LVS_EX_COLUMNSNAPPOINTS;
				if ( pParameters->hDetailedMode.bHeaderDragNDrop )
					dwStyleEx |= LVS_EX_HEADERDRAGDROP;
			}

			if ( pParameters->bAutoSizeColumns )
				dwStyleEx |= LVS_EX_AUTOSIZECOLUMNS;
			
			if ( pParameters->hDetailedMode.bFullRowSelection )
				dwStyleEx |= LVS_EX_FULLROWSELECT;
			if ( pParameters->hDetailedMode.bShowGridLines )
				dwStyleEx |= LVS_EX_GRIDLINES;
			if ( pParameters->hDetailedMode.bSubItemImages )
				dwStyleEx |= LVS_EX_SUBITEMIMAGES;
			break;
		case WINGUI_TABLE_VIEW_TILES:
			dwStyle |= LVS_LIST; // Can't set at creation, delay

			if ( pParameters->bHeadersInAllViews ) {
				if ( pParameters->bStaticColumnHeaders )
					dwStyle |= LVS_NOSORTHEADER;
				if ( pParameters->bSnapColumnsWidth )
					dwStyleEx |= LVS_EX_COLUMNSNAPPOINTS;
				if ( pParameters->bAutoSizeColumns )
					dwStyleEx |= LVS_EX_AUTOSIZECOLUMNS;
				if ( pParameters->hTilesMode.bColumnOverflow )
					dwStyleEx |= LVS_EX_COLUMNOVERFLOW;
			}
			break;
		default: DebugAssert(false); break;
	}

	if ( pParameters->bEditableLabels )
		dwStyle |= LVS_EDITLABELS;

	if ( pParameters->bSingleItemSelection )
		dwStyle |= LVS_SINGLESEL;
	if ( pParameters->bAlwaysShowSelection )
		dwStyle |= LVS_SHOWSELALWAYS;
	if ( pParameters->bBorderSelection )
		dwStyleEx |= LVS_EX_BORDERSELECT;

	if ( !m_bVirtualTable ) {
		if ( pParameters->bSortAscending )
			dwStyle |= LVS_SORTASCENDING;
		if ( pParameters->bSortDescending )
			dwStyle |= LVS_SORTDESCENDING;
	}

	if ( pParameters->bAddCheckBoxes ) {
		dwStyleEx |= LVS_EX_CHECKBOXES;
		if ( pParameters->bAutoCheckOnSelect )
			dwStyleEx |= LVS_EX_AUTOCHECKSELECT;
	}

	if ( pParameters->bHandleInfoTips )
		dwStyleEx |= LVS_EX_INFOTIP;

	if ( pParameters->bHotTrackingSingleClick ) {
		dwStyleEx |= LVS_EX_ONECLICKACTIVATE;
		if ( pParameters->bHotTrackSelection )
			dwStyleEx |= LVS_EX_TRACKSELECT;

		if ( pParameters->bUnderlineHot )
			dwStyleEx |= LVS_EX_UNDERLINEHOT;
	} else if ( pParameters->bHotTrackingDoubleClick ) {
		dwStyleEx |= LVS_EX_TWOCLICKACTIVATE;
		if ( pParameters->bHotTrackSelection )
			dwStyleEx |= LVS_EX_TRACKSELECT;

		if ( pParameters->bUnderlineHot )
			dwStyleEx |= LVS_EX_UNDERLINEHOT;
		else if ( pParameters->bUnderlineCold )
			dwStyleEx |= LVS_EX_UNDERLINECOLD;
	}

	if ( pParameters->bSharedImageList )
		dwStyle |= LVS_SHAREIMAGELISTS;
	if ( pParameters->bUseBackBuffer )
		dwStyleEx |= LVS_EX_DOUBLEBUFFER;
	if ( pParameters->bTransparentBackground ) {
		dwStyleEx |= LVS_EX_TRANSPARENTBKGND;
		if ( pParameters->bTransparentShadowText )
			dwStyleEx |= LVS_EX_TRANSPARENTSHADOWTEXT;
	}

    // Window creation
	m_hHandle = CreateWindowEx (
		dwStyleEx,
		WC_LISTVIEW,
		TEXT(""),
		dwStyle,
		hWindowRect.iLeft, hWindowRect.iTop,
        hWindowRect.iWidth, hWindowRect.iHeight,
		hParentWnd,
		(HMENU)m_iResourceID,
		(HINSTANCE)( GetWindowLongPtr(hParentWnd,GWLP_HINSTANCE) ),
		NULL
	);
	DebugAssert( m_hHandle != NULL );

	// Set Tile Mode after creation
	if ( pParameters->iViewMode == WINGUI_TABLE_VIEW_TILES )
		ListView_SetView( (HWND)m_hHandle, LV_VIEW_TILE );

	// Populate the list
	UInt iCount = pModel->GetItemCount();
	for( UInt i = 0; i < iCount; ++i ) {

	}

	// Done
	_SaveElementToHandle();
}
Void WinGUITable::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUITable::_DispatchEvent( Int iNotificationCode, Void * pParameters )
{
    // Get Model
	WinGUIButtonModel * pModel = (WinGUIButtonModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		//case NM_SETFOCUS:  return pModel->OnFocusGained(); break;
		//case NM_KILLFOCUS: return pModel->OnFocusLost(); break;

		//case LVN_KEYDOWN: return pModel->OnKeyPress(); break;
		//case NM_RETURN:   return pModel->OnKeyPressEnter(); break;

		//case NM_CLICK:   return pModel->OnClickLeftItem(); break;
		//case NM_RCLICK:  return pModel->OnClickRightItem(); break;
		//case NM_DBLCLK:  return pModel->OnDblClickLeftItem(); break;
		//case NM_RDBLCLK: return pModel->OnDblClickRightItem(); break;
		//case NM_HOVER:   return pModel->OnHoverItem(); break;

		//case LVN_COLUMNCLICK: return pModel->OnClickHeader(); break;
		//case LVN_LINKCLICK:   return pModel->OnClickLink(); break;

		//case LVN_BEGINDRAG:      return pModel->OnDragLeftStart(); break;
		//case LVN_BEGINRDRAG:     return pModel->OnDragRightStart(); break;
		//case LVN_BEGINSCROLL:    return pModel->OnScrollStart(); break;
		//case LVN_ENDSCROLL:      return pModel->OnScrollEnd(); break;
		//case LVN_BEGINLABELEDIT: return pModel->OnLabelEditStart(); break;
		//case LVN_ENDLABELEDIT:   return pModel->OnLabelEditEnd(); break;

		//case LVN_INSERTITEM:     return pModel->OnInsertItem(); break;
		//case LVN_DELETEALLITEMS: return pModel->OnDeleteAllItems(); break;
		//case LVN_DELETEITEM:     return pModel->OnDeleteItem(); break;

		//case LVN_MARQUEEBEGIN: return pModel->OnSelectionBoxStart(); break;

		//case LVN_ITEMACTIVATE: return pModel->OnItemActivation(); break;
		//case LVN_ITEMCHANGED:  return pModel->OnItemChange(); break;
		//case LVN_ITEMCHANGING: return pModel->OnItemChanging(); break;

		//case LVN_INCREMENTALSEARCH: return pModel->OnSearch(); break;

		//case LVN_GETEMPTYMARKUP: return pModel->OnGetEmptyMarkup(); break;
		//case LVN_GETINFOTIP:     return pModel->OnGetInfoTip(); break;

		//case LVN_GETDISPINFO: return pModel->OnGetDisplayInfo(); break;
		//case LVN_SETDISPINFO: return pModel->OnSetDisplayInfo(); break;

		//case LVN_ODCACHEHINT:     return pModel->OnRequestCache(); break;
		//case LVN_ODSTATECHANGED : return pModel->OnRequestUpdate(); break;
		//case LVN_ODFINDITEM:      return pModel->OnRequestSearch(); break;

		default: break;
	}

	// Unhandled
	return false;
}

