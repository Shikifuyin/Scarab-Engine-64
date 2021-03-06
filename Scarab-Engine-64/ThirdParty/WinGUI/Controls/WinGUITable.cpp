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
	m_hCreationParameters.bVirtualTable = false;

	m_hCreationParameters.bHasBackBuffer = false;
	m_hCreationParameters.bHasSharedImageLists = false;

	m_hCreationParameters.iItemCallBackMode = 0;
	m_hCreationParameters.iStateCallBackMode = 0;

	m_hCreationParameters.iViewMode = WINGUI_TABLE_VIEW_LIST;
	m_hCreationParameters.bGroupMode = false;
	m_hCreationParameters.bHasHeadersInAllViews = false;

	m_hCreationParameters.bHasColumnHeaders = true;
	m_hCreationParameters.bHasStaticColumnHeaders = false;
	m_hCreationParameters.bHasDraggableColumnHeaders = false;
	m_hCreationParameters.bHasIconColumnOverflowButton = false;

	m_hCreationParameters.bHasCheckBoxes = false;
	m_hCreationParameters.bHasIconLabels = true;
	m_hCreationParameters.bHasEditableLabels = false;
	m_hCreationParameters.bHasSubItemImages = false;

	m_hCreationParameters.bSingleItemSelection = false;
	m_hCreationParameters.bIconSimpleSelection = false;

	m_hCreationParameters.bAutoSortAscending = false;
	m_hCreationParameters.bAutoSortDescending = false;

	m_hCreationParameters.bHasHotTrackingSingleClick = true;
	m_hCreationParameters.bHasHotTrackingDoubleClick = false;
	m_hCreationParameters.bHasHotTrackingSelection = true;

	m_hCreationParameters.bHasInfoTips = false;
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

	m_bHasBackBuffer = false;
	m_bHasSharedImageLists = false;

	m_iItemCallBackMode = 0;
	m_iStateCallBackMode = 0;

	m_iViewMode = WINGUI_TABLE_VIEW_LIST;
	m_bGroupMode = false;
	m_bHasHeadersInAllViews = false;

	m_bHasColumnHeaders = true;
	m_bHasStaticColumnHeaders = false;
	m_bHasDraggableColumnHeaders = false;
	m_bHasIconColumnOverflowButton = false;

	m_bHasCheckBoxes = false;
	m_bHasIconLabels = true;
	m_bHasEditableLabels = false;
	m_bHasSubItemImages = false;

	m_bSingleItemSelection = false;
	m_bIconSimpleSelection = false;

	m_bAutoSortAscending = false;
	m_bAutoSortDescending = false;

	m_bHasHotTrackingSingleClick = false;
	m_bHasHotTrackingDoubleClick = false;
	m_bHasHotTrackingSelection = false;

	m_bHasInfoTips = false;

	m_hEditLabelHandle = NULL;
	m_iEditLabelItemIndex = INVALID_OFFSET;

	m_iColumnCount = 0;
}
WinGUITable::~WinGUITable()
{
	// nothing to do
}

// General Settings /////////////////////////////////////////////////////////////
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

Void WinGUITable::SetAllocatedItemCount( UInt iPreAllocatedItemCount )
{
	HWND hHandle = (HWND)m_hHandle;

	if ( m_bVirtualTable ) {
		ListView_SetItemCountEx( hHandle, iPreAllocatedItemCount, LVSICF_NOINVALIDATEALL | LVSICF_NOSCROLL );
	} else {
		ListView_SetItemCount( hHandle, iPreAllocatedItemCount );
	}
}

Void WinGUITable::UseBackBuffer( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_DOUBLEBUFFER;
	DWORD dwValue = bEnable ? LVS_EX_DOUBLEBUFFER : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );

	m_bHasBackBuffer = bEnable;
}

Void WinGUITable::UseSharedImageLists( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwStyle = GetWindowLong( hHandle, GWL_STYLE );
	if ( bEnable ) {
		dwStyle |= LVS_SHAREIMAGELISTS;
	} else {
		dwStyle &= (~LVS_SHAREIMAGELISTS);
	}
	SetWindowLong( hHandle, GWL_STYLE, dwStyle );

	m_bHasSharedImageLists = bEnable;
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

// Callback Settings ////////////////////////////////////////////////////////////
Void WinGUITable::SetItemCallBackMode( WinGUITableItemCallBackMode iMode )
{
	DebugAssert( GetItemCount() == 0 );

	// Disable Auto-Sorting
	if ( iMode & WINGUI_TABLE_ITEMCALLBACK_LABELS )
		ToggleAutoSorting( false );

	m_iItemCallBackMode = iMode;
}

Void WinGUITable::SetStateCallBackMode( WinGUITableStateCallBackMode iMode )
{
	DebugAssert( GetItemCount() == 0 );

	HWND hHandle = (HWND)m_hHandle;

	UInt iMask = 0;
	if ( iMode & WINGUI_TABLE_STATECALLBACK_IMAGE_OVERLAY )
		iMask |= LVIS_OVERLAYMASK;
	if ( iMode & WINGUI_TABLE_STATECALLBACK_IMAGE_STATE )
		iMask |= LVIS_STATEIMAGEMASK;
	if ( iMode & WINGUI_TABLE_STATECALLBACK_SELECTION )
		iMask |= LVIS_SELECTED;
	if ( iMode & WINGUI_TABLE_STATECALLBACK_FOCUS )
		iMask |= LVIS_FOCUSED;
	if ( iMode & WINGUI_TABLE_STATECALLBACK_CUTMARK )
		iMask |= LVIS_CUT;
	if ( iMode & WINGUI_TABLE_STATECALLBACK_DROPHIGHLIGHT )
		iMask |= LVIS_DROPHILITED;

	ListView_SetCallbackMask( hHandle, iMask );

	m_iStateCallBackMode = iMode;
}

Void WinGUITable::UpdateItem( UInt iItemIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_Update( hHandle, iItemIndex );
}
Void WinGUITable::ForceRedraw( UInt iFirstItem, UInt iLastItem, Bool bImmediate )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_RedrawItems( hHandle, iFirstItem, iLastItem );
	if ( bImmediate )
		UpdateWindow( hHandle );
}

// View Modes ///////////////////////////////////////////////////////////////////
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

Void WinGUITable::ToggleGroupMode( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_EnableGroupView( hHandle, bEnable ? TRUE : FALSE );

	m_bGroupMode = bEnable;
}

Void WinGUITable::ToggleHeadersInAllViews( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_HEADERINALLVIEWS;
	DWORD dwValue = bEnable ? LVS_EX_HEADERINALLVIEWS : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );

	m_bHasHeadersInAllViews = bEnable;
}

// Options //////////////////////////////////////////////////////////////////////
Void WinGUITable::ToggleColumnHeaders( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwStyle = GetWindowLong( hHandle, GWL_STYLE );
	if ( bEnable ) {
		dwStyle &= (~LVS_NOCOLUMNHEADER);
	} else {
		dwStyle |= LVS_NOCOLUMNHEADER;
	}
	SetWindowLong( hHandle, GWL_STYLE, dwStyle );

	m_bHasColumnHeaders = bEnable;
}

Void WinGUITable::ToggleStaticColumnHeaders( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwStyle = GetWindowLong( hHandle, GWL_STYLE );
	if ( bEnable ) {
		dwStyle |= LVS_NOSORTHEADER;
	} else {
		dwStyle &= (~LVS_NOSORTHEADER);
	}
	SetWindowLong( hHandle, GWL_STYLE, dwStyle );

	m_bHasStaticColumnHeaders = bEnable;
}

Void WinGUITable::ToggleDraggableColumnHeaders( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_HEADERDRAGDROP;
	DWORD dwValue = bEnable ? LVS_EX_HEADERDRAGDROP : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );

	m_bHasDraggableColumnHeaders = bEnable;
}

Void WinGUITable::ToggleIconColumnOverflowButton( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_COLUMNOVERFLOW;
	DWORD dwValue = bEnable ? LVS_EX_COLUMNOVERFLOW : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );

	m_bHasIconColumnOverflowButton = bEnable;
}

Void WinGUITable::ToggleCheckBoxes( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_CHECKBOXES;
	DWORD dwValue = bEnable ? LVS_EX_CHECKBOXES : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );

	m_bHasCheckBoxes = bEnable;
}
Void WinGUITable::ToggleAutoCheckOnSelect( Bool bEnable )
{
	DebugAssert( m_bHasCheckBoxes );

	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_AUTOCHECKSELECT;
	DWORD dwValue = bEnable ? LVS_EX_AUTOCHECKSELECT : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );
}

Void WinGUITable::ToggleIconLabels( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_CHECKBOXES;
	DWORD dwValue = bEnable ? 0 : LVS_EX_HIDELABELS;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );

	m_bHasIconLabels = bEnable;
}
Void WinGUITable::PreventIconLabelWrap( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwStyle = GetWindowLong( hHandle, GWL_STYLE );
	if ( bEnable ) {
		dwStyle |= LVS_NOLABELWRAP;
	} else {
		dwStyle &= (~LVS_NOLABELWRAP);
	}
	SetWindowLong( hHandle, GWL_STYLE, dwStyle );
}

Void WinGUITable::ToggleEditableLabels( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwStyle = GetWindowLong( hHandle, GWL_STYLE );
	if ( bEnable ) {
		dwStyle |= LVS_EDITLABELS;
	} else {
		dwStyle &= (~LVS_EDITLABELS);
	}
	SetWindowLong( hHandle, GWL_STYLE, dwStyle );

	m_bHasEditableLabels = bEnable;
}

Void WinGUITable::ToggleSubItemImages( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_SUBITEMIMAGES;
	DWORD dwValue = bEnable ? LVS_EX_SUBITEMIMAGES : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );
}

Void WinGUITable::ToggleSingleItemSelection( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwStyle = GetWindowLong( hHandle, GWL_STYLE );
	if ( bEnable ) {
		dwStyle |= LVS_SINGLESEL;
	} else {
		dwStyle &= (~LVS_SINGLESEL);
	}
	SetWindowLong( hHandle, GWL_STYLE, dwStyle );

	m_bSingleItemSelection = bEnable;
}

Void WinGUITable::ToggleIconSimpleSelection( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_SIMPLESELECT;
	DWORD dwValue = bEnable ? LVS_EX_SIMPLESELECT : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );

	m_bIconSimpleSelection = bEnable;
}

Void WinGUITable::ToggleAutoSorting( Bool bEnable, Bool bAscendingElseDescending )
{
	DebugAssert( !m_bVirtualTable );

	HWND hHandle = (HWND)m_hHandle;

	DWORD dwStyle = GetWindowLong( hHandle, GWL_STYLE );
	if ( bEnable ) {
		if ( bAscendingElseDescending ) {
			dwStyle |= LVS_SORTASCENDING;
			dwStyle &= (~LVS_SORTDESCENDING);
			m_bAutoSortAscending = true;
			m_bAutoSortDescending = false;
		} else {
			dwStyle &= (~LVS_SORTASCENDING);
			dwStyle |= LVS_SORTDESCENDING;
			m_bAutoSortAscending = false;
			m_bAutoSortDescending = true;
		}
	} else {
		dwStyle &= (~LVS_SORTASCENDING);
		dwStyle &= (~LVS_SORTDESCENDING);
		m_bAutoSortAscending = false;
		m_bAutoSortDescending = false;
	}
	SetWindowLong( hHandle, GWL_STYLE, dwStyle );
}

Void WinGUITable::ToggleHotTracking( Bool bEnable, Bool bUseSingleClick )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_ONECLICKACTIVATE | LVS_EX_TWOCLICKACTIVATE;
	DWORD dwValue = 0;
	m_bHasHotTrackingSingleClick = false;
	m_bHasHotTrackingDoubleClick = false;
	if ( bEnable ) {
		if ( bUseSingleClick ) {
			dwValue = LVS_EX_ONECLICKACTIVATE;
			m_bHasHotTrackingSingleClick = true;
		} else {
			dwValue = LVS_EX_TWOCLICKACTIVATE;
			m_bHasHotTrackingDoubleClick = true;
		}
	}
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );
}
Void WinGUITable::ToggleHotTrackingSelection( Bool bEnable )
{
	DebugAssert( m_bHasHotTrackingSingleClick || m_bHasHotTrackingDoubleClick );

	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_TRACKSELECT;
	DWORD dwValue = bEnable ? LVS_EX_TRACKSELECT : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );

	m_bHasHotTrackingSelection = bEnable;
}
Void WinGUITable::ToggleHotTrackingUnderline( Bool bEnable, Bool bUnderlineHotElseCold )
{
	DebugAssert( m_bHasHotTrackingSingleClick || m_bHasHotTrackingDoubleClick );

	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_UNDERLINEHOT | LVS_EX_UNDERLINECOLD;
	DWORD dwValue = 0;
	if ( bEnable ) {
		if ( bUnderlineHotElseCold )
			dwValue = LVS_EX_UNDERLINEHOT;
		else {
			DebugAssert( m_bHasHotTrackingDoubleClick );
			dwValue = LVS_EX_UNDERLINECOLD;
		}
	}
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );
}

Void WinGUITable::ToggleInfoTips( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_INFOTIP;
	DWORD dwValue = bEnable ? LVS_EX_INFOTIP : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );

	m_bHasInfoTips = bEnable;
}

// Visual Settings //////////////////////////////////////////////////////////////
Void WinGUITable::ToggleTransparentBackground( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_TRANSPARENTBKGND;
	DWORD dwValue = bEnable ? LVS_EX_TRANSPARENTBKGND : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );
}
Void WinGUITable::ShowGridLines( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_GRIDLINES;
	DWORD dwValue = bEnable ? LVS_EX_GRIDLINES : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );
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

Void WinGUITable::GetBackgroundImage( WinGUIBitmap * outImage, WinGUIPointF * outRelativePos, Bool * outIsTiled ) const
{
	DebugAssert( !(outImage->IsCreated()) );

	HWND hHandle = (HWND)m_hHandle;

	LVBKIMAGE hBkgImageInfos;
	hBkgImageInfos.pszImage = NULL;
	hBkgImageInfos.cchImageMax = 0;
	hBkgImageInfos.ulFlags = LVBKIF_SOURCE_HBITMAP;

	ListView_GetBkImage( hHandle, &hBkgImageInfos );
	DebugAssert( hBkgImageInfos.hbm != NULL );

	outImage->_CreateFromHandle( hBkgImageInfos.hbm, true, true );

	outRelativePos->fX = ( 0.01f * (Float)(hBkgImageInfos.xOffsetPercent) );
	outRelativePos->fY = ( 0.01f * (Float)(hBkgImageInfos.yOffsetPercent) );
	*outIsTiled = ( (hBkgImageInfos.ulFlags & LVBKIF_STYLE_TILE) != 0 );
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

Void WinGUITable::ToggleTransparentShadowText( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_TRANSPARENTSHADOWTEXT;
	DWORD dwValue = bEnable ? LVS_EX_TRANSPARENTSHADOWTEXT : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );
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

Void WinGUITable::AutoSizeColumns( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_AUTOSIZECOLUMNS;
	DWORD dwValue = bEnable ? LVS_EX_AUTOSIZECOLUMNS : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );
}
Void WinGUITable::SnapColumnWidths( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_COLUMNSNAPPOINTS;
	DWORD dwValue = bEnable ? LVS_EX_COLUMNSNAPPOINTS : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );
}
Void WinGUITable::JustifyIconColumns( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_JUSTIFYCOLUMNS;
	DWORD dwValue = bEnable ? LVS_EX_JUSTIFYCOLUMNS : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );
}

Void WinGUITable::SetIconAlignment( WinGUITableIconsAlign iAlign )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwStyle = GetWindowLong( hHandle, GWL_STYLE );
	switch( iAlign ) {
		case WINGUI_TABLE_ICONS_ALIGN_DEFAULT:
			dwStyle &= (~LVS_ALIGNTOP);
			dwStyle &= (~LVS_ALIGNLEFT);
			break;
		case WINGUI_TABLE_ICONS_ALIGN_TOP:
			dwStyle |= LVS_ALIGNTOP;
			dwStyle &= (~LVS_ALIGNLEFT);
			break;
		case WINGUI_TABLE_ICONS_ALIGN_LEFT:
			dwStyle &= (~LVS_ALIGNTOP);
			dwStyle |= LVS_ALIGNLEFT;
			break;
		default: DebugAssert(false); break;
	}
	SetWindowLong( hHandle, GWL_STYLE, dwStyle );
}
Void WinGUITable::SnapIconsToGrid( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_SNAPTOGRID;
	DWORD dwValue = bEnable ? LVS_EX_SNAPTOGRID : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );
}
Void WinGUITable::AutoArrangeIcons( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwStyle = GetWindowLong( hHandle, GWL_STYLE );
	if ( bEnable ) {
		dwStyle |= LVS_AUTOARRANGE;
	} else {
		dwStyle &= (~LVS_AUTOARRANGE);
	}
	SetWindowLong( hHandle, GWL_STYLE, dwStyle );
}

Void WinGUITable::ToggleAlwaysShowSelection( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwStyle = GetWindowLong( hHandle, GWL_STYLE );
	if ( bEnable ) {
		dwStyle |= LVS_SHOWSELALWAYS;
	} else {
		dwStyle &= (~LVS_SHOWSELALWAYS);
	}
	SetWindowLong( hHandle, GWL_STYLE, dwStyle );
}
Void WinGUITable::ToggleFullRowSelection( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_FULLROWSELECT;
	DWORD dwValue = bEnable ? LVS_EX_FULLROWSELECT : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );
}

Void WinGUITable::ToggleBorderSelection( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwMask = LVS_EX_BORDERSELECT;
	DWORD dwValue = bEnable ? LVS_EX_BORDERSELECT : 0;
	ListView_SetExtendedListViewStyleEx( hHandle, dwMask, dwValue );
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

UInt WinGUITable::GetInsertionMarkColor() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetInsertMarkColor( hHandle );
}
Void WinGUITable::SetInsertionMarkColor( UInt iColor )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetInsertMarkColor( hHandle, iColor );
}

Void WinGUITable::GetEmptyText( GChar * outText, UInt iMaxLength ) const
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_GetEmptyText( hHandle, outText, iMaxLength );
}

// Metrics //////////////////////////////////////////////////////////////////////
Void WinGUITable::GetViewOrigin( WinGUIPoint * outOrigin ) const
{
	HWND hHandle = (HWND)m_hHandle;

	POINT hPt;
	ListView_GetOrigin( hHandle, &hPt );

	outOrigin->iX = hPt.x;
	outOrigin->iY = hPt.y;
}
Void WinGUITable::GetViewRect( WinGUIRectangle * outRectangle ) const
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

Void WinGUITable::GetRequiredDimensions( UInt * pWidth, UInt * pHeight, UInt iItemCount ) const
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

Void WinGUITable::GetItemPosition( WinGUIPoint * outPosition, UInt iItemIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	POINT hPt;
	ListView_GetItemPosition( hHandle, iItemIndex, &hPt );

	outPosition->iX = hPt.x;
	outPosition->iY = hPt.y;
}
Void WinGUITable::GetItemRect( WinGUIRectangle * outRectangle, UInt iItemIndex, Bool bIconOnly, Bool bLabelOnly ) const
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iCode = LVIR_BOUNDS;
	if ( bIconOnly )
		iCode |= LVIR_ICON;
	if ( bLabelOnly )
		iCode |= LVIR_LABEL;
	// Both combined = LVIR_SELECTBOUNDS

	RECT hRect;
	ListView_GetItemRect( hHandle, iItemIndex, &hRect, iCode );

	outRectangle->iLeft = hRect.left;
	outRectangle->iTop = hRect.top;
	outRectangle->iWidth = ( hRect.right - hRect.left );
	outRectangle->iHeight = ( hRect.bottom - hRect.top );
}
Void WinGUITable::GetSubItemRect( WinGUIRectangle * outRectangle, UInt iItemIndex, UInt iSubItemIndex, Bool bIconOnly, Bool bLabelOnly ) const
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iCode = LVIR_BOUNDS;
	if ( bIconOnly )
		iCode |= LVIR_ICON;
	if ( bLabelOnly )
		iCode |= LVIR_LABEL;
	DebugAssert( iCode != LVIR_SELECTBOUNDS ); // Both combined = LVIR_SELECTBOUNDS, NOT ALLOWED HERE

	RECT hRect;
	ListView_GetSubItemRect( hHandle, iItemIndex, iSubItemIndex, iCode, &hRect );

	outRectangle->iLeft = hRect.left;
	outRectangle->iTop = hRect.top;
	outRectangle->iWidth = ( hRect.right - hRect.left );
	outRectangle->iHeight = ( hRect.bottom - hRect.top );
}
Void WinGUITable::GetSubItemRect( WinGUIRectangle * outRectangle, UInt iGroupIndex, UInt iItemIndex, UInt iSubItemIndex, Bool bIconOnly, Bool bLabelOnly ) const
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
	hItemIndex.iItem = iItemIndex;

	RECT hRect;
	ListView_GetItemIndexRect( hHandle, &hItemIndex, iSubItemIndex, iCode, &hRect );

	outRectangle->iLeft = hRect.left;
	outRectangle->iTop = hRect.top;
	outRectangle->iWidth = ( hRect.right - hRect.left );
	outRectangle->iHeight = ( hRect.bottom - hRect.top );
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

Void WinGUITable::GetInsertionMarkMetrics( WinGUIRectangle * outRectangle ) const
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	ListView_GetInsertMarkRect( hHandle, &hRect );

	outRectangle->iLeft = hRect.left;
	outRectangle->iTop = hRect.top;
	outRectangle->iWidth = ( hRect.right - hRect.left );
	outRectangle->iHeight = ( hRect.bottom - hRect.top );
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
	outResult->iSubItemIndex = hHitTestInfos.iSubItem;

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

// Scroll Operations ////////////////////////////////////////////////////////////
Void WinGUITable::Scroll( Int iScrollH, Int iScrollV )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_Scroll( hHandle, iScrollH, iScrollV );
}
Void WinGUITable::ScrollToItem( UInt iItemIndex, Bool bAllowPartial )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_EnsureVisible( hHandle, iItemIndex, bAllowPartial ? TRUE : FALSE );
}

// Column Operations ////////////////////////////////////////////////////////////
Void WinGUITable::AddColumn( UInt iColumnIndex, GChar * strHeaderText, UInt iSubItemIndex, UInt iOrderIndex, UInt iDefaultWidth )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT | LVCF_SUBITEM | LVCF_ORDER | LVCF_TEXT | LVCF_IMAGE
		              | LVCF_WIDTH | LVCF_MINWIDTH | LVCF_DEFAULTWIDTH | LVCF_IDEALWIDTH;
	hColumnInfos.fmt = LVCFMT_LEFT;
	hColumnInfos.iSubItem = iSubItemIndex;
	hColumnInfos.iOrder = iOrderIndex;
	hColumnInfos.pszText = strHeaderText;
	hColumnInfos.iImage = I_IMAGENONE;
	hColumnInfos.cx = iDefaultWidth;
	hColumnInfos.cxMin = iDefaultWidth;
	hColumnInfos.cxDefault = iDefaultWidth;
	hColumnInfos.cxIdeal = iDefaultWidth;

	ListView_InsertColumn( hHandle, iColumnIndex, &hColumnInfos );

	++m_iColumnCount;
}
Void WinGUITable::RemoveColumn( UInt iColumnIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_DeleteColumn( hHandle, iColumnIndex );

	--m_iColumnCount;
}

UInt WinGUITable::GetColumnSubItem( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_SUBITEM;
	hColumnInfos.iSubItem = INVALID_OFFSET;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	return hColumnInfos.iSubItem;
}
Void WinGUITable::SetColumnSubItem( UInt iColumnIndex, UInt iSubItemIndex )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_SUBITEM;
	hColumnInfos.iSubItem = iSubItemIndex;

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
}

UInt WinGUITable::GetColumnOrderIndex( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_ORDER;
	hColumnInfos.iOrder = INVALID_OFFSET;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	return hColumnInfos.iOrder;
}
Void WinGUITable::SetColumnOrderIndex( UInt iColumnIndex, UInt iOrderIndex )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_ORDER;
	hColumnInfos.iOrder = iOrderIndex;

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
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

Void WinGUITable::GetColumnHeaderText( GChar * outHeaderText, UInt iMaxLength, UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_TEXT;
	hColumnInfos.pszText = outHeaderText;
	hColumnInfos.cchTextMax = iMaxLength;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );
}
Void WinGUITable::SetColumnHeaderText( UInt iColumnIndex, GChar * strHeaderText )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_TEXT;
	hColumnInfos.pszText = strHeaderText;

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
}

Bool WinGUITable::HasColumnHeaderImage( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT;
	hColumnInfos.fmt = 0;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	return ( (hColumnInfos.fmt & LVCFMT_COL_HAS_IMAGES) != 0 );
}
Void WinGUITable::ToggleColumnHeaderImage( UInt iColumnIndex, Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT;
	hColumnInfos.fmt = 0;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	if ( bEnable )
		hColumnInfos.fmt |= LVCFMT_COL_HAS_IMAGES;
	else
		hColumnInfos.fmt &= (~LVCFMT_COL_HAS_IMAGES);

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
}
UInt WinGUITable::GetColumnHeaderImage( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT | LVCF_IMAGE;
	hColumnInfos.fmt = 0;
	hColumnInfos.iImage = INVALID_OFFSET;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );
	DebugAssert( (hColumnInfos.fmt & LVCFMT_COL_HAS_IMAGES) != 0 );

	if ( hColumnInfos.iImage == I_IMAGENONE )
		return INVALID_OFFSET;
	return hColumnInfos.iImage;
}
Void WinGUITable::SetColumnHeaderImage( UInt iColumnIndex, UInt iImageIndex )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_IMAGE;
	if ( iImageIndex == INVALID_OFFSET )
		hColumnInfos.iImage = I_IMAGENONE;
	else
		hColumnInfos.iImage = iImageIndex;

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
}

Bool WinGUITable::HasColumnHeaderSplitButton( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT;
	hColumnInfos.fmt = 0;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	return ( (hColumnInfos.fmt & LVCFMT_SPLITBUTTON) != 0 );
}
Void WinGUITable::ToggleColumnHeaderSplitButton( UInt iColumnIndex, Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT;
	hColumnInfos.fmt = 0;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	if ( bEnable )
		hColumnInfos.fmt |= LVCFMT_SPLITBUTTON;
	else
		hColumnInfos.fmt &= (~LVCFMT_SPLITBUTTON);

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
}

WinGUITableTextAlign WinGUITable::GetColumnRowTextAlign( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT;
	hColumnInfos.fmt = 0;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	switch( hColumnInfos.fmt & LVCFMT_JUSTIFYMASK ) {
		case LVCFMT_LEFT:   return WINGUI_TABLE_TEXT_ALIGN_LEFT; break;
		case LVCFMT_RIGHT:  return WINGUI_TABLE_TEXT_ALIGN_RIGHT; break;
		case LVCFMT_CENTER: return WINGUI_TABLE_TEXT_ALIGN_CENTER; break;
		default: DebugAssert(false); break;
	}
	return WINGUI_TABLE_TEXT_ALIGN_LEFT; // Should never get here
}
Void WinGUITable::SetColumnRowTextAlign( UInt iColumnIndex, WinGUITableTextAlign iAlign )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT;
	hColumnInfos.fmt = 0;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	hColumnInfos.fmt &= (~LVCFMT_JUSTIFYMASK);
	switch( iAlign ) {
		case WINGUI_TABLE_TEXT_ALIGN_LEFT:   hColumnInfos.fmt |= LVCFMT_LEFT; break;
		case WINGUI_TABLE_TEXT_ALIGN_RIGHT:  hColumnInfos.fmt |= LVCFMT_RIGHT; break;
		case WINGUI_TABLE_TEXT_ALIGN_CENTER: hColumnInfos.fmt |= LVCFMT_CENTER; break;
		default: DebugAssert(false); break;
	}

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
}

Bool WinGUITable::HasColumnRowImages( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT;
	hColumnInfos.fmt = 0;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	return ( (hColumnInfos.fmt & LVCFMT_IMAGE) != 0 );
}
Void WinGUITable::ToggleColumnRowImages( UInt iColumnIndex, Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT;
	hColumnInfos.fmt = 0;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	if ( bEnable )
		hColumnInfos.fmt |= LVCFMT_IMAGE;
	else
		hColumnInfos.fmt &= (~LVCFMT_IMAGE);

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
}

Bool WinGUITable::HasColumnRightAlignedRowImages( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT;
	hColumnInfos.fmt = 0;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	return ( (hColumnInfos.fmt & LVCFMT_BITMAP_ON_RIGHT) != 0 );
}
Void WinGUITable::ToggleColumnRightAlignedRowImages( UInt iColumnIndex, Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT;
	hColumnInfos.fmt = 0;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	if ( bEnable )
		hColumnInfos.fmt |= LVCFMT_BITMAP_ON_RIGHT;
	else
		hColumnInfos.fmt &= (~LVCFMT_BITMAP_ON_RIGHT);

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
}

Bool WinGUITable::HasColumnFixedWidth( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT;
	hColumnInfos.fmt = 0;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	return ( (hColumnInfos.fmt & LVCFMT_FIXED_WIDTH) != 0 );
}
Void WinGUITable::ToggleColumnFixedWidth( UInt iColumnIndex, Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT;
	hColumnInfos.fmt = 0;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	if ( bEnable )
		hColumnInfos.fmt |= LVCFMT_FIXED_WIDTH;
	else
		hColumnInfos.fmt &= (~LVCFMT_FIXED_WIDTH);

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
}

Bool WinGUITable::HasColumnFixedRatio( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT;
	hColumnInfos.fmt = 0;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	return ( (hColumnInfos.fmt & LVCFMT_FIXED_RATIO) != 0 );
}
Void WinGUITable::ToggleColumnFixedRatio( UInt iColumnIndex, Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT;
	hColumnInfos.fmt = 0;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	if ( bEnable )
		hColumnInfos.fmt |= LVCFMT_FIXED_RATIO;
	else
		hColumnInfos.fmt &= (~LVCFMT_FIXED_RATIO);

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
}
	
UInt WinGUITable::GetColumnWidth( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_WIDTH;
	hColumnInfos.cx = INVALID_OFFSET;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	return hColumnInfos.cx;
}
Void WinGUITable::SetColumnWidth( UInt iColumnIndex, UInt iWidth )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_WIDTH;
	hColumnInfos.cx = iWidth;

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
}

UInt WinGUITable::GetColumnMinWidth( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_MINWIDTH;
	hColumnInfos.cxMin = INVALID_OFFSET;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	return hColumnInfos.cxMin;
}
Void WinGUITable::SetColumnMinWidth( UInt iColumnIndex, UInt iMinWidth )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_MINWIDTH;
	hColumnInfos.cxMin = iMinWidth;

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
}

UInt WinGUITable::GetColumnDefaultWidth( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_DEFAULTWIDTH;
	hColumnInfos.cxDefault = INVALID_OFFSET;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	return hColumnInfos.cxDefault;
}
Void WinGUITable::SetColumnDefaultWidth( UInt iColumnIndex, UInt iDefaultWidth )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_DEFAULTWIDTH;
	hColumnInfos.cxDefault = iDefaultWidth;

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
}

UInt WinGUITable::GetColumnIdealWidth( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_IDEALWIDTH;
	hColumnInfos.cxIdeal = INVALID_OFFSET;

	ListView_GetColumn( hHandle, iColumnIndex, &hColumnInfos );

	return hColumnInfos.cxIdeal;
}
Void WinGUITable::SetColumnIdealWidth( UInt iColumnIndex, UInt iIdealWidth )
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_IDEALWIDTH;
	hColumnInfos.cxIdeal = iIdealWidth;

	ListView_SetColumn( hHandle, iColumnIndex, &hColumnInfos );
}

UInt WinGUITable::GetColumnCurrentWidth( UInt iColumnIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetColumnWidth( hHandle, iColumnIndex );
}
Void WinGUITable::SetColumnCurrentWidth( UInt iColumnIndex, UInt iCurrentWidth )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetColumnWidth( hHandle, iColumnIndex, iCurrentWidth );
}
Void WinGUITable::AutoSizeColumnCurrentWidth( UInt iColumnIndex, Bool bFitHeaderText )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetColumnWidth( hHandle, iColumnIndex, bFitHeaderText ? LVSCW_AUTOSIZE_USEHEADER : LVSCW_AUTOSIZE );
}

UInt WinGUITable::GetSelectedColumn() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetSelectedColumn( hHandle );
}
Void WinGUITable::SelectColumn( UInt iColumnIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetSelectedColumn( hHandle, iColumnIndex );
}

// Group Operations /////////////////////////////////////////////////////////////
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
Void WinGUITable::GetGroupByID( WinGUITableGroupInfos * outGroupInfos, UInt iGroupID ) const
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;

	LVGROUP hGroupInfos;
	hGroupInfos.cbSize = sizeof(LVGROUP);
	hGroupInfos.mask = LVGF_GROUPID | LVGF_ITEMS | LVGF_HEADER | LVGF_FOOTER | LVGF_ALIGN | LVGF_STATE |
		LVGF_SUBTITLE | LVGF_TASK | LVGF_DESCRIPTIONTOP | LVGF_DESCRIPTIONBOTTOM |
		LVGF_TITLEIMAGE | LVGF_EXTENDEDIMAGE | LVGF_SUBSET; // | LVGF_SUBSETITEMS;
	hGroupInfos.stateMask = 0xff;

	hGroupInfos.pszHeader = outGroupInfos->strHeaderText;
	hGroupInfos.cchHeader = 64;
	hGroupInfos.pszFooter = outGroupInfos->strFooterText;
	hGroupInfos.cchFooter = 64;
	hGroupInfos.pszSubtitle = outGroupInfos->strSubTitleText;
	hGroupInfos.cchSubtitle = 64;
	hGroupInfos.pszTask = outGroupInfos->strTaskLinkText;
	hGroupInfos.cchTask = 64;
	hGroupInfos.pszDescriptionTop = outGroupInfos->strTopDescriptionText;
	hGroupInfos.cchDescriptionTop = 64;
	hGroupInfos.pszDescriptionBottom = outGroupInfos->strBottomDescriptionText;
	hGroupInfos.cchDescriptionBottom = 64;
	hGroupInfos.pszSubsetTitle = outGroupInfos->strSubSetTitleText;
	hGroupInfos.cchSubsetTitle = 64;

	ListView_GetGroupInfo( hHandle, iGroupID, &hGroupInfos );

	_Convert_GroupInfos( outGroupInfos, &hGroupInfos );
}
Void WinGUITable::GetGroupByIndex( WinGUITableGroupInfos * outGroupInfos, UInt iGroupIndex ) const
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;

	LVGROUP hGroupInfos;
	hGroupInfos.cbSize = sizeof(LVGROUP);
	hGroupInfos.mask = LVGF_GROUPID | LVGF_ITEMS | LVGF_HEADER | LVGF_FOOTER | LVGF_ALIGN | LVGF_STATE |
		LVGF_SUBTITLE | LVGF_TASK | LVGF_DESCRIPTIONTOP | LVGF_DESCRIPTIONBOTTOM |
		LVGF_TITLEIMAGE | LVGF_EXTENDEDIMAGE | LVGF_SUBSET; // | LVGF_SUBSETITEMS;
	hGroupInfos.stateMask = 0xff;

	hGroupInfos.pszHeader = outGroupInfos->strHeaderText;
	hGroupInfos.cchHeader = 64;
	hGroupInfos.pszFooter = outGroupInfos->strFooterText;
	hGroupInfos.cchFooter = 64;
	hGroupInfos.pszSubtitle = outGroupInfos->strSubTitleText;
	hGroupInfos.cchSubtitle = 64;
	hGroupInfos.pszTask = outGroupInfos->strTaskLinkText;
	hGroupInfos.cchTask = 64;
	hGroupInfos.pszDescriptionTop = outGroupInfos->strTopDescriptionText;
	hGroupInfos.cchDescriptionTop = 64;
	hGroupInfos.pszDescriptionBottom = outGroupInfos->strBottomDescriptionText;
	hGroupInfos.cchDescriptionBottom = 64;
	hGroupInfos.pszSubsetTitle = outGroupInfos->strSubSetTitleText;
	hGroupInfos.cchSubsetTitle = 64;

	ListView_GetGroupInfoByIndex( hHandle, iGroupIndex, &hGroupInfos );

	_Convert_GroupInfos( outGroupInfos, &hGroupInfos );
}

Void WinGUITable::AddGroup( UInt iGroupIndex, const WinGUITableGroupInfos * pGroupInfos )
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

	ListView_InsertGroup( hHandle, iGroupIndex, &hGroupInfos );
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

Void WinGUITable::ExpandGroup( UInt iGroupID )
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;
	ListView_SetGroupState( hHandle, iGroupID, LVGS_COLLAPSED, 0 );
}
Void WinGUITable::CollapseGroup( UInt iGroupID )
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;
	ListView_SetGroupState( hHandle, iGroupID, LVGS_COLLAPSED, LVGS_COLLAPSED );
}

UInt WinGUITable::GetFocusedGroup() const
{
	DebugAssert( m_bGroupMode );

	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetFocusedGroup( hHandle );
}

// Item Operations //////////////////////////////////////////////////////////////
Bool WinGUITable::IsItemVisible( UInt iItemIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;
	return ( ListView_IsItemVisible(hHandle, iItemIndex) != FALSE );
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

Void WinGUITable::SetItemIconPosition( UInt iItemIndex, const WinGUIPoint & hPosition )
{
	DebugAssert( m_iViewMode == WINGUI_TABLE_VIEW_ICONS || m_iViewMode == WINGUI_TABLE_VIEW_ICONS_SMALL );

	HWND hHandle = (HWND)m_hHandle;

	ListView_SetItemPosition( hHandle, iItemIndex, hPosition.iX, hPosition.iY );
}

UInt WinGUITable::GetItemCount() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetItemCount( hHandle );
}

Void WinGUITable::AddItem( UInt iItemIndex )
{
	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;

	static GChar arrTempLabel[64];

	hItemInfos.mask = LVIF_TEXT;
	hItemInfos.pszText = arrTempLabel;
	hItemInfos.cchTextMax = 64;
	StringFn->NCopy( hItemInfos.pszText, TEXT("_Uninitialized_"), 63 );

	if ( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_LABELS) != 0 )
		hItemInfos.pszText = LPSTR_TEXTCALLBACK;
	
	if ( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_IMAGES) != 0 ) {
		hItemInfos.mask |= LVIF_IMAGE;
		hItemInfos.iImage = I_IMAGECALLBACK;
	}

	if ( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_GROUPIDS) != 0 ) {
		hItemInfos.mask |= LVIF_GROUPID;
		hItemInfos.iGroupId = I_GROUPIDCALLBACK;
	}

	if ( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_INDENTATION) != 0 ) {
		hItemInfos.mask |= LVIF_INDENT;
		hItemInfos.iIndent = I_INDENTCALLBACK;
	}

	if ( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_COLUMNS) != 0 ) {
		hItemInfos.mask |= LVIF_COLUMNS;
		hItemInfos.cColumns = I_COLUMNSCALLBACK;
	}

	ListView_InsertItem( hHandle, &hItemInfos );
}
Void WinGUITable::RemoveItem( UInt iItemIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_DeleteItem( hHandle, iItemIndex );
}
Void WinGUITable::RemoveAllItems()
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_DeleteAllItems( hHandle );
}

Void WinGUITable::GetItemLabel( GChar * outLabelText, UInt iMaxLength, UInt iItemIndex, UInt iSubItemIndex ) const
{
	DebugAssert( !m_bVirtualTable );
	DebugAssert( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_LABELS) == 0 );

	HWND hHandle = (HWND)m_hHandle;
	ListView_GetItemText( hHandle, iItemIndex, iSubItemIndex, outLabelText, iMaxLength );
}
Void WinGUITable::SetItemLabel( UInt iItemIndex, UInt iSubItemIndex, GChar * strLabelText )
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_LABELS) == 0 );

	HWND hHandle = (HWND)m_hHandle;
	ListView_SetItemText( hHandle, iItemIndex, iSubItemIndex, strLabelText );
}

UInt WinGUITable::GetItemIcon( UInt iItemIndex, UInt iSubItemIndex ) const
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_IMAGES) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_IMAGE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = iSubItemIndex;
	hItemInfos.iImage = INVALID_OFFSET;

	ListView_GetItem( hHandle, &hItemInfos );

	if ( hItemInfos.iImage == I_IMAGENONE )
		return INVALID_OFFSET;
	return hItemInfos.iImage;
}
Void WinGUITable::SetItemIcon( UInt iItemIndex, UInt iSubItemIndex, UInt iIconIndex )
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_IMAGES) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_IMAGE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = iSubItemIndex;
	if ( iIconIndex == INVALID_OFFSET )
		hItemInfos.iImage = I_IMAGENONE;
	else
		hItemInfos.iImage = iIconIndex;

	ListView_SetItem( hHandle, &hItemInfos );
}

Void * WinGUITable::GetItemData( UInt iItemIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_PARAM;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.lParam = NULL;

	ListView_GetItem( hHandle, &hItemInfos );

	return (Void*)( hItemInfos.lParam );
}
Void WinGUITable::SetItemData( UInt iItemIndex, Void * pData )
{
	HWND hHandle = (HWND)m_hHandle;
	
	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_PARAM;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.lParam = (LPARAM)pData;

	ListView_SetItem( hHandle, &hItemInfos );
}

UInt WinGUITable::GetItemGroupID( UInt iItemIndex ) const
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_GROUPIDS) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_GROUPID;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.iGroupId = INVALID_OFFSET;

	ListView_GetItem( hHandle, &hItemInfos );

	return hItemInfos.iGroupId;
}
Void WinGUITable::SetItemGroupID( UInt iItemIndex, UInt iGroupID )
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_GROUPIDS) == 0 );

	HWND hHandle = (HWND)m_hHandle;
	
	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_GROUPID;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	if ( iGroupID == INVALID_OFFSET )
		hItemInfos.iGroupId = I_GROUPIDNONE;
	else
		hItemInfos.iGroupId = iGroupID;

	ListView_SetItem( hHandle, &hItemInfos );
}

UInt WinGUITable::GetItemIndentation( UInt iItemIndex ) const
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_INDENTATION) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_INDENT;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.iIndent = INVALID_OFFSET;

	ListView_GetItem( hHandle, &hItemInfos );

	return hItemInfos.iIndent;
}
Void WinGUITable::SetItemIndentation( UInt iItemIndex, UInt iIndentation )
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_INDENTATION) == 0 );

	HWND hHandle = (HWND)m_hHandle;
	
	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_INDENT;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.iIndent = iIndentation;

	ListView_SetItem( hHandle, &hItemInfos );
}

UInt WinGUITable::GetItemColumnCount( UInt iItemIndex ) const
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_COLUMNS) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_COLUMNS;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.cColumns = 0;
	hItemInfos.puColumns = NULL;

	ListView_GetItem( hHandle, &hItemInfos );

	return hItemInfos.cColumns;
}

Void WinGUITable::GetItemColumnIndices( UInt * outColumnIndices, UInt iMaxColumns, UInt iItemIndex ) const
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_COLUMNS) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_COLUMNS;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.cColumns = 0;
	hItemInfos.puColumns = NULL;

	ListView_GetItem( hHandle, &hItemInfos );

	UInt iMinCount = Min<UInt>( hItemInfos.cColumns, iMaxColumns );
	for( UInt i = 0; i < iMinCount; ++i )
		outColumnIndices[i] = hItemInfos.puColumns[i];
}
Void WinGUITable::SetItemColumnIndices( UInt iItemIndex, const UInt * arrColumnIndices, UInt iColumnCount )
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_COLUMNS) == 0 );
	DebugAssert( iColumnCount <= 32 );

	HWND hHandle = (HWND)m_hHandle;
	
	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_COLUMNS;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.cColumns = iColumnCount;

	static UInt arrTempIndices[32];
	hItemInfos.puColumns = arrTempIndices;
	for( UInt i = 0; i < iColumnCount; ++i )
		hItemInfos.puColumns[i] = arrColumnIndices[i];

	ListView_SetItem( hHandle, &hItemInfos );
}

Void WinGUITable::GetItemColumnFormats( WinGUITableItemColumnFormat * outColumnFormats, UInt iMaxColumns, UInt iItemIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_COLFMT;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.cColumns = 0;
	hItemInfos.piColFmt = NULL;

	ListView_GetItem( hHandle, &hItemInfos );

	UInt iMinCount = Min<UInt>( hItemInfos.cColumns, iMaxColumns );
	for( UInt i = 0; i < iMinCount; ++i )
		_Convert_ItemColumnFormat( outColumnFormats + i, hItemInfos.piColFmt[i] );
}
Void WinGUITable::SetItemColumnFormats( UInt iItemIndex, const WinGUITableItemColumnFormat * arrColumnFormats, UInt iColumnCount )
{
	DebugAssert( iColumnCount <= 32 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_COLFMT;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.cColumns = iColumnCount;

	static Int arrTempFormats[32];
	hItemInfos.piColFmt = arrTempFormats;
	for( UInt i = 0; i < iColumnCount; ++i )
		_Convert_ItemColumnFormat( hItemInfos.piColFmt + i, arrColumnFormats + i );

	ListView_SetItem( hHandle, &hItemInfos );
}

Void WinGUITable::GetItemState( WinGUITableItemState * outItemState, UInt iItemIndex ) const
{
	DebugAssert( m_iStateCallBackMode == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_STATE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.stateMask = 0xffffffff;
	hItemInfos.state = 0;

	ListView_GetItem( hHandle, &hItemInfos );

	_Convert_ItemState( outItemState, hItemInfos.state );
}
Void WinGUITable::SetItemState( UInt iItemIndex, const WinGUITableItemState * pItemState )
{
	DebugAssert( m_iStateCallBackMode == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_STATE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.stateMask = 0xffffffff;

	_Convert_ItemState( &(hItemInfos.state), pItemState );

	ListView_SetItem( hHandle, &hItemInfos );
}

UInt WinGUITable::GetItemOverlayImage( UInt iItemIndex ) const
{
	DebugAssert( (m_iStateCallBackMode & WINGUI_TABLE_STATECALLBACK_IMAGE_OVERLAY) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_STATE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.stateMask = LVIS_OVERLAYMASK;
	hItemInfos.state = 0;

	ListView_GetItem( hHandle, &hItemInfos );

	return ( (hItemInfos.state & LVIS_OVERLAYMASK) >> 8 );
}
Void WinGUITable::SetItemOverlayImage( UInt iItemIndex, UInt iOverlayImage )
{
	DebugAssert( (m_iStateCallBackMode & WINGUI_TABLE_STATECALLBACK_IMAGE_OVERLAY) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_STATE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.stateMask = LVIS_OVERLAYMASK;
	hItemInfos.state = INDEXTOOVERLAYMASK(iOverlayImage & 0x0f);

	ListView_SetItem( hHandle, &hItemInfos );
}

UInt WinGUITable::GetItemStateImage( UInt iItemIndex ) const
{
	DebugAssert( (m_iStateCallBackMode & WINGUI_TABLE_STATECALLBACK_IMAGE_STATE) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_STATE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.stateMask = LVIS_STATEIMAGEMASK;
	hItemInfos.state = 0;

	ListView_GetItem( hHandle, &hItemInfos );

	return ( (hItemInfos.state & LVIS_STATEIMAGEMASK) >> 12 );
}
Void WinGUITable::SetItemStateImage( UInt iItemIndex, UInt iStateImage )
{
	DebugAssert( (m_iStateCallBackMode & WINGUI_TABLE_STATECALLBACK_IMAGE_STATE) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_STATE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.stateMask = LVIS_OVERLAYMASK;
	hItemInfos.state = INDEXTOSTATEIMAGEMASK(iStateImage & 0x0f);

	ListView_SetItem( hHandle, &hItemInfos );
}

Bool WinGUITable::IsItemFocused( UInt iItemIndex ) const
{
	DebugAssert( (m_iStateCallBackMode & WINGUI_TABLE_STATECALLBACK_FOCUS) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_STATE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.stateMask = LVIS_FOCUSED;
	hItemInfos.state = 0;

	ListView_GetItem( hHandle, &hItemInfos );

	return ( (hItemInfos.state & LVIS_FOCUSED) != 0 );
}
Void WinGUITable::FocusItem( UInt iItemIndex, Bool bHasFocus )
{
	DebugAssert( (m_iStateCallBackMode & WINGUI_TABLE_STATECALLBACK_FOCUS) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_STATE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.stateMask = LVIS_FOCUSED;
	hItemInfos.state = bHasFocus ? LVIS_FOCUSED : 0;

	ListView_SetItem( hHandle, &hItemInfos );
}

Bool WinGUITable::IsItemSelected( UInt iItemIndex ) const
{
	DebugAssert( (m_iStateCallBackMode & WINGUI_TABLE_STATECALLBACK_SELECTION) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_STATE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.stateMask = LVIS_SELECTED;
	hItemInfos.state = 0;

	ListView_GetItem( hHandle, &hItemInfos );

	return ( (hItemInfos.state & LVIS_SELECTED) != 0 );
}
Void WinGUITable::SelectItem( UInt iItemIndex, Bool bSelect )
{
	DebugAssert( (m_iStateCallBackMode & WINGUI_TABLE_STATECALLBACK_SELECTION) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_STATE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.stateMask = LVIS_SELECTED;
	hItemInfos.state = bSelect ? LVIS_SELECTED : 0;

	ListView_SetItem( hHandle, &hItemInfos );
}

Bool WinGUITable::IsItemCutMarked( UInt iItemIndex ) const
{
	DebugAssert( (m_iStateCallBackMode & WINGUI_TABLE_STATECALLBACK_CUTMARK) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_STATE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.stateMask = LVIS_CUT;
	hItemInfos.state = 0;

	ListView_GetItem( hHandle, &hItemInfos );

	return ( (hItemInfos.state & LVIS_CUT) != 0 );
}
Void WinGUITable::SetItemCutMarked( UInt iItemIndex, Bool bCutMark )
{
	DebugAssert( (m_iStateCallBackMode & WINGUI_TABLE_STATECALLBACK_CUTMARK) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_STATE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.stateMask = LVIS_CUT;
	hItemInfos.state = bCutMark ? LVIS_CUT : 0;

	ListView_SetItem( hHandle, &hItemInfos );
}

Bool WinGUITable::IsItemDropHighlighted( UInt iItemIndex ) const
{
	DebugAssert( (m_iStateCallBackMode & WINGUI_TABLE_STATECALLBACK_DROPHIGHLIGHT) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_STATE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.stateMask = LVIS_DROPHILITED;
	hItemInfos.state = 0;

	ListView_GetItem( hHandle, &hItemInfos );

	return ( (hItemInfos.state & LVIS_DROPHILITED) != 0 );
}
Void WinGUITable::SetItemDropHighlighted( UInt iItemIndex, Bool bCutMark )
{
	DebugAssert( (m_iStateCallBackMode & WINGUI_TABLE_STATECALLBACK_DROPHIGHLIGHT) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	LVITEM hItemInfos;
	hItemInfos.mask = LVIF_STATE;
	hItemInfos.iItem = iItemIndex;
	hItemInfos.iSubItem = 0;
	hItemInfos.stateMask = LVIS_DROPHILITED;
	hItemInfos.state = bCutMark ? LVIS_DROPHILITED : 0;

	ListView_SetItem( hHandle, &hItemInfos );
}

UInt WinGUITable::GetSelectedItemCount() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetSelectedCount( hHandle );
}
UInt WinGUITable::GetSelectedItems( UInt * outItemIndices, UInt iMaxIndices ) const
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iItemCount = 0;
	UInt iFlags = LVNI_ALL | LVNI_SELECTED;

	UInt iResult = ListView_GetNextItem( hHandle, -1, iFlags );
	while( iResult != INVALID_OFFSET ) {
		if ( iItemCount >= iMaxIndices )
			break;
		outItemIndices[iItemCount++] = iResult;
		iResult = ListView_GetNextItem( hHandle, iResult, iFlags );
	}

	return iItemCount;
}

UInt WinGUITable::GetMultiSelectMark() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetSelectionMark( hHandle );
}
Void WinGUITable::SetMultiSelectMark( UInt iItemIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_SetSelectionMark( hHandle, iItemIndex );
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
Void WinGUITable::SetInsertionMark( UInt iItemIndex, Bool bInsertAfter )
{
	HWND hHandle = (HWND)m_hHandle;

	LVINSERTMARK hInsertMark;
	hInsertMark.cbSize = sizeof(LVINSERTMARK);
	hInsertMark.iItem = iItemIndex;
	hInsertMark.dwFlags = 0;
	if ( bInsertAfter )
		hInsertMark.dwFlags |= LVIM_AFTER;

	ListView_SetInsertMark( hHandle, &hInsertMark );
}

Bool WinGUITable::IsItemChecked( UInt iItemIndex ) const
{
	DebugAssert( m_bHasCheckBoxes == true );

	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetCheckState( hHandle, iItemIndex );
}
Void WinGUITable::CheckItem( UInt iItemIndex, Bool bChecked )
{
	DebugAssert( m_bHasCheckBoxes == true );

	HWND hHandle = (HWND)m_hHandle;
	ListView_SetCheckState( hHandle, iItemIndex, bChecked ? TRUE : FALSE );
}

UInt WinGUITable::AssignItemID( UInt iItemIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_MapIndexToID( hHandle, iItemIndex );
}
UInt WinGUITable::GetItemFromID( UInt iUniqueID )
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_MapIDToIndex( hHandle, iUniqueID );
}

// Item Label Edition ///////////////////////////////////////////////////////////
Void WinGUITable::GetEditItemLabel( WinGUITextEdit * outTextEdit )
{
	DebugAssert( m_bHasEditableLabels && (m_hEditLabelHandle != NULL) );
	outTextEdit->_CreateFromHandle( m_hEditLabelHandle );
}
Void WinGUITable::ReleaseEditItemLabel( WinGUITextEdit * pTextEdit )
{
	DebugAssert( m_bHasEditableLabels && (m_hEditLabelHandle != NULL) );
	pTextEdit->_Release();
}

Void WinGUITable::EditItemLabelStart( WinGUITextEdit * outTextEdit, UInt iItemIndex )
{
	DebugAssert( m_bHasEditableLabels && (m_hEditLabelHandle == NULL) );
	DebugAssert( !(outTextEdit->IsCreated()) );

	HWND hHandle = (HWND)m_hHandle;

	SetFocus( hHandle ); // Required

	m_hEditLabelHandle = ListView_EditLabel( hHandle, iItemIndex );
	DebugAssert( m_hEditLabelHandle != NULL );

	m_iEditLabelItemIndex = iItemIndex;

	outTextEdit->_CreateFromHandle( m_hEditLabelHandle );
}
Void WinGUITable::EditItemLabelEnd( WinGUITextEdit * pTextEdit )
{
	DebugAssert( m_bHasEditableLabels && (m_hEditLabelHandle != NULL) );
	DebugAssert( !pTextEdit->IsCreated() );

	m_hEditLabelHandle = NULL; // We do NOT own this handle, let the ListView destroy it
	m_iEditLabelItemIndex = INVALID_OFFSET;

	pTextEdit->_Release();
}
Void WinGUITable::EditItemLabelCancel( WinGUITextEdit * pTextEdit )
{
	DebugAssert( m_bHasEditableLabels && (m_hEditLabelHandle != NULL) );
	DebugAssert( !pTextEdit->IsCreated() );

	HWND hHandle = (HWND)m_hHandle;

	ListView_CancelEditLabel( hHandle );

	m_hEditLabelHandle = NULL; // We do NOT own this handle, let the ListView destroy it
	m_iEditLabelItemIndex = INVALID_OFFSET;

	pTextEdit->_Release();
}

// Tile Operations //////////////////////////////////////////////////////////////
Void WinGUITable::GetTile( WinGUITableTileInfos * outInfos, UInt iItemIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVTILEINFO hTileInfos;
	hTileInfos.cbSize = sizeof(LVTILEINFO);
	hTileInfos.iItem = iItemIndex;

	hTileInfos.cColumns = 32;
	static UInt arrTempIndices[32];
	hTileInfos.puColumns = arrTempIndices;
	static Int arrTempFormats[32];
	hTileInfos.piColFmt = arrTempFormats;

	ListView_GetTileInfo( hHandle, &hTileInfos );

	_Convert_TileInfos( outInfos, &hTileInfos );
}
Void WinGUITable::SetTile( UInt iItemIndex, const WinGUITableTileInfos * pTileInfos )
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

	hTileInfos.iItem = iItemIndex;

	ListView_SetTileInfo( hHandle, &hTileInfos );
}

// Search Operations ////////////////////////////////////////////////////////////
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

UInt WinGUITable::SearchItem( const GChar * strLabel, UInt iStartIndex, const WinGUITableSearchOptions & hSearchOptions ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVFINDINFO hInfos;
	hInfos.psz = strLabel;

	switch( hSearchOptions.iMode ) {
		case WINGUI_TABLE_SEARCH_STRING:    hInfos.flags = LVFI_STRING; break;
		case WINGUI_TABLE_SEARCH_SUBSTRING: hInfos.flags = LVFI_STRING | LVFI_PARTIAL; break;
		default: DebugAssert(false); break;
	}
	if ( hSearchOptions.bWrapAround )
		hInfos.flags |= LVFI_WRAP;

	return ListView_FindItem( hHandle, ((Int)iStartIndex) - 1, &hInfos );
}
UInt WinGUITable::SearchItem( Void * pUserData, UInt iStartIndex, const WinGUITableSearchOptions & hSearchOptions ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVFINDINFO hInfos;
	hInfos.lParam = (LPARAM)pUserData;
	hInfos.flags = LVFI_PARAM;

	DebugAssert( hSearchOptions.iMode == WINGUI_TABLE_SEARCH_USERDATA );
	if ( hSearchOptions.bWrapAround )
		hInfos.flags |= LVFI_WRAP;

	return ListView_FindItem( hHandle, ((Int)iStartIndex) - 1, &hInfos );
}
UInt WinGUITable::SearchItem( const WinGUIPoint * pPoint, UInt iStartIndex, const WinGUITableSearchOptions & hSearchOptions ) const
{
	DebugAssert( m_iViewMode == WINGUI_TABLE_VIEW_ICONS || m_iViewMode == WINGUI_TABLE_VIEW_ICONS_SMALL );

	HWND hHandle = (HWND)m_hHandle;

	LVFINDINFO hInfos;
	hInfos.pt.x = pPoint->iX;
	hInfos.pt.y = pPoint->iY;
	hInfos.flags = LVFI_NEARESTXY;

	DebugAssert( hSearchOptions.iMode == WINGUI_TABLE_SEARCH_SPATIAL );
	switch( hSearchOptions.iSpatialDirection ) {
		case WINGUI_TABLE_SEARCH_SPATIAL_UP:    hInfos.vkDirection = VK_UP; break;
		case WINGUI_TABLE_SEARCH_SPATIAL_DOWN:  hInfos.vkDirection = VK_DOWN; break;
		case WINGUI_TABLE_SEARCH_SPATIAL_LEFT:  hInfos.vkDirection = VK_LEFT; break;
		case WINGUI_TABLE_SEARCH_SPATIAL_RIGHT: hInfos.vkDirection = VK_RIGHT; break;
		default: DebugAssert(false); break;
	}

	return ListView_FindItem( hHandle, ((Int)iStartIndex) - 1, &hInfos );
}

UInt WinGUITable::SearchNextItem( UInt iStartIndex, const WinGUITableSearchNextOptions & hSearchOptions ) const
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iFlags = 0;
	if ( hSearchOptions.bSpatialSearch ) {
		switch( hSearchOptions.iSpatialDirection ) {
			case WINGUI_TABLE_SEARCH_SPATIAL_UP:    iFlags |= LVNI_ABOVE; break;
			case WINGUI_TABLE_SEARCH_SPATIAL_DOWN:  iFlags |= LVNI_BELOW; break;
			case WINGUI_TABLE_SEARCH_SPATIAL_LEFT:  iFlags |= LVNI_TOLEFT; break;
			case WINGUI_TABLE_SEARCH_SPATIAL_RIGHT: iFlags |= LVNI_TORIGHT; break;
			default: DebugAssert(false); break;
		}
	} else {
		iFlags = LVNI_ALL;
		if ( hSearchOptions.bReverseSearch )
			iFlags |= LVNI_PREVIOUS;
	}

	if ( hSearchOptions.bSameGroup )
		iFlags |= LVNI_SAMEGROUPONLY;
	if ( hSearchOptions.bVisible )
		iFlags |= LVNI_VISIBLEONLY;
	if ( hSearchOptions.bHasFocus )
		iFlags |= LVNI_FOCUSED;
	if ( hSearchOptions.bSelected )
		iFlags |= LVNI_SELECTED;
	if ( hSearchOptions.bCutMarked )
		iFlags |= LVNI_CUT;
	if ( hSearchOptions.bDropHighlight )
		iFlags |= LVNI_DROPHILITED;

	return ListView_GetNextItem( hHandle, ((Int)iStartIndex) - 1, iFlags );
}
UInt WinGUITable::SearchNextItem( UInt iStartGroupIndex, UInt iStartIndex, const WinGUITableSearchNextOptions & hSearchOptions ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVITEMINDEX hItemIndex;
	hItemIndex.iGroup = iStartGroupIndex;
	hItemIndex.iItem = ((Int)iStartIndex) - 1;

	UInt iFlags = 0;
	if ( hSearchOptions.bSpatialSearch ) {
		switch( hSearchOptions.iSpatialDirection ) {
			case WINGUI_TABLE_SEARCH_SPATIAL_UP:    iFlags |= LVNI_ABOVE; break;
			case WINGUI_TABLE_SEARCH_SPATIAL_DOWN:  iFlags |= LVNI_BELOW; break;
			case WINGUI_TABLE_SEARCH_SPATIAL_LEFT:  iFlags |= LVNI_TOLEFT; break;
			case WINGUI_TABLE_SEARCH_SPATIAL_RIGHT: iFlags |= LVNI_TORIGHT; break;
			default: DebugAssert(false); break;
		}
	} else {
		iFlags = LVNI_ALL;
		if ( hSearchOptions.bReverseSearch )
			iFlags |= LVNI_PREVIOUS;
	}

	if ( hSearchOptions.bSameGroup )
		iFlags |= LVNI_SAMEGROUPONLY;
	if ( hSearchOptions.bVisible )
		iFlags |= LVNI_VISIBLEONLY;
	if ( hSearchOptions.bHasFocus )
		iFlags |= LVNI_FOCUSED;
	if ( hSearchOptions.bSelected )
		iFlags |= LVNI_SELECTED;
	if ( hSearchOptions.bCutMarked )
		iFlags |= LVNI_CUT;
	if ( hSearchOptions.bDropHighlight )
		iFlags |= LVNI_DROPHILITED;

	return ListView_GetNextItemIndex( hHandle, &hItemIndex, iFlags );
}

// Sorting Operations ///////////////////////////////////////////////////////////
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

Void WinGUITable::ArrangeIcons( WinGUITableIconsAlign iAlign, Bool bSnapToGrid )
{
	DebugAssert( m_iViewMode == WINGUI_TABLE_VIEW_ICONS || m_iViewMode == WINGUI_TABLE_VIEW_ICONS_SMALL );

	HWND hHandle = (HWND)m_hHandle;

	UInt iCode = 0;
	if ( bSnapToGrid )
		iCode = LVA_SNAPTOGRID;
	else {
		switch( iAlign ) {
			case WINGUI_TABLE_ICONS_ALIGN_DEFAULT: iCode = LVA_DEFAULT; break;
			case WINGUI_TABLE_ICONS_ALIGN_TOP:     iCode = LVA_ALIGNTOP; break;
			case WINGUI_TABLE_ICONS_ALIGN_LEFT:    iCode = LVA_ALIGNLEFT; break;
			default: DebugAssert(false); break;
		}
	}
	
	ListView_Arrange( hHandle, iCode );
}

// Hot Tracking /////////////////////////////////////////////////////////////////
UInt WinGUITable::GetHotItem() const
{
	DebugAssert( m_bHasHotTrackingSingleClick || m_bHasHotTrackingDoubleClick );

	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetHotItem( hHandle );
}
Void WinGUITable::SetHotItem( UInt iIndex )
{
	DebugAssert( m_bHasHotTrackingSingleClick || m_bHasHotTrackingDoubleClick );

	HWND hHandle = (HWND)m_hHandle;
	ListView_SetHotItem( hHandle, iIndex );
}

UInt WinGUITable::GetHoverTime() const
{
	DebugAssert( m_bHasHotTrackingSingleClick || m_bHasHotTrackingDoubleClick );

	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetHoverTime( hHandle );
}
Void WinGUITable::SetHoverTime( UInt iTimeMS )
{
	DebugAssert( m_bHasHotTrackingSingleClick || m_bHasHotTrackingDoubleClick );

	HWND hHandle = (HWND)m_hHandle;
	ListView_SetHoverTime( hHandle, iTimeMS );
}

Void WinGUITable::GetHotCursor( WinGUICursor * outCursor ) const
{
	DebugAssert( m_bHasHotTrackingSingleClick || m_bHasHotTrackingDoubleClick );
	DebugAssert( !(outCursor->IsCreated()) );

	HWND hHandle = (HWND)m_hHandle;

	HCURSOR hCursor = ListView_GetHotCursor( hHandle );
	DebugAssert( hCursor != NULL );

	outCursor->_CreateFromHandle( hCursor, true );
}
Void WinGUITable::SetHotCursor( const WinGUICursor * pCursor )
{
	DebugAssert( m_bHasHotTrackingSingleClick || m_bHasHotTrackingDoubleClick );
	DebugAssert( pCursor->IsCreated() );

	HWND hHandle = (HWND)m_hHandle;
	ListView_SetHotCursor( hHandle, (HCURSOR)(pCursor->m_hHandle) );
}

// Work Areas ///////////////////////////////////////////////////////////////////
UInt WinGUITable::GetMaxWorkAreasCount() const
{
	return LV_MAX_WORKAREAS;
}
UInt WinGUITable::GetWorkAreasCount() const
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iCount = 0;
	ListView_GetNumberOfWorkAreas( hHandle, &iCount );
	return iCount;
}

Void WinGUITable::GetWorkAreas( WinGUIRectangle * outWorkAreas, UInt iMaxCount ) const
{
	HWND hHandle = (HWND)m_hHandle;

	static RECT arrWorkAreasTemp[LV_MAX_WORKAREAS];
	ListView_GetWorkAreas( hHandle, iMaxCount, arrWorkAreasTemp );

	for( UInt i = 0; i < iMaxCount; ++i ) {
		outWorkAreas[i].iLeft = arrWorkAreasTemp[i].left;
		outWorkAreas[i].iTop = arrWorkAreasTemp[i].top;
		outWorkAreas[i].iWidth = ( arrWorkAreasTemp[i].right - arrWorkAreasTemp[i].left );
		outWorkAreas[i].iHeight = ( arrWorkAreasTemp[i].bottom - arrWorkAreasTemp[i].top );
	}
}
Void WinGUITable::SetWorkAreas( const WinGUIRectangle * arrWorkAreas, UInt iCount )
{
	HWND hHandle = (HWND)m_hHandle;

	static RECT arrWorkAreasTemp[LV_MAX_WORKAREAS];
	for( UInt i = 0; i < iCount; ++i ) {
		arrWorkAreasTemp[i].left = arrWorkAreas[i].iLeft;
		arrWorkAreasTemp[i].top = arrWorkAreas[i].iTop;
		arrWorkAreasTemp[i].right = ( arrWorkAreas[i].iLeft + arrWorkAreas[i].iWidth );
		arrWorkAreasTemp[i].bottom = ( arrWorkAreas[i].iTop + arrWorkAreas[i].iHeight );
	}

	ListView_SetWorkAreas( hHandle, iCount, arrWorkAreasTemp );
}

// Drag & Drop //////////////////////////////////////////////////////////////////
Void WinGUITable::CreateDragImageList( WinGUIImageList * outDragImageList, WinGUIPoint * outInitialPosition, UInt iItemIndex )
{
	DebugAssert( !(outDragImageList->IsCreated()) );

	HWND hHandle = (HWND)m_hHandle;

	POINT hPt;
	HIMAGELIST hImageList = ListView_CreateDragImage( hHandle, iItemIndex, &hPt );
	DebugAssert( hImageList != NULL );

	outDragImageList->_CreateFromHandle( hImageList );
	// outDragImageList->m_bIsDragging = true; ???

	outInitialPosition->iX = hPt.x;
	outInitialPosition->iY = hPt.y;
}

// Tool/Info Tips ///////////////////////////////////////////////////////////////
Void WinGUITable::SetInfoTip( UInt iItemIndex, UInt iSubItemIndex, GChar * strInfoText )
{
	DebugAssert( m_bHasInfoTips );

	HWND hHandle = (HWND)m_hHandle;

	LVSETINFOTIP hInfoTip;
	hInfoTip.cbSize = sizeof(LVSETINFOTIP);
	hInfoTip.dwFlags = 0;
	hInfoTip.iItem = iItemIndex;
	hInfoTip.iSubItem = iSubItemIndex;
	hInfoTip.pszText = strInfoText;

	ListView_SetInfoTip( hHandle, &hInfoTip );
}

/////////////////////////////////////////////////////////////////////////////////

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

Void WinGUITable::_Convert_ItemState( WinGUITableItemState * outItemState, UInt iItemState ) const
{
	outItemState->iOverlayImage = ( iItemState & LVIS_OVERLAYMASK ) >> 8;
	outItemState->iStateImage = ( iItemState & LVIS_STATEIMAGEMASK ) >> 12;

	outItemState->bHasFocus = ( (iItemState & LVIS_FOCUSED) != 0 );
	outItemState->bSelected = ( (iItemState & LVIS_SELECTED) != 0 );

	outItemState->bCutMarked = ( (iItemState & LVIS_CUT) != 0 );
	outItemState->bDropHighlight = ( (iItemState & LVIS_DROPHILITED) != 0 );
}
Void WinGUITable::_Convert_ItemState( UInt * outItemState, const WinGUITableItemState * pItemState ) const
{
	*outItemState = 0;

	*outItemState |= ( (pItemState->iOverlayImage & 0x0f) << 8 );
	*outItemState |= ( (pItemState->iStateImage & 0x0f) << 12 );

	if ( pItemState->bHasFocus )
		*outItemState |= LVIS_FOCUSED;
	if ( pItemState->bSelected )
		*outItemState |= LVIS_SELECTED;

	if ( pItemState->bCutMarked )
		*outItemState |= LVIS_CUT;
	if ( pItemState->bDropHighlight )
		*outItemState |= LVIS_DROPHILITED;
}

Void WinGUITable::_Convert_ItemColumnFormat( WinGUITableItemColumnFormat * outItemColumnFormat, Int iColFormat ) const
{
	outItemColumnFormat->bLineBreak = ( (iColFormat & LVCFMT_LINE_BREAK) != 0 );
	outItemColumnFormat->bFill = ( (iColFormat & LVCFMT_FILL) != 0 );
	outItemColumnFormat->bAllowWrap = ( (iColFormat & LVCFMT_WRAP) != 0 );
	outItemColumnFormat->bNoTitle = ( (iColFormat & LVCFMT_NO_TITLE) != 0 );
}
Void WinGUITable::_Convert_ItemColumnFormat( Int * outColFormat, const WinGUITableItemColumnFormat * pItemColumnFormat ) const
{
	*outColFormat = 0;
	if ( pItemColumnFormat->bLineBreak )
		*outColFormat |= LVCFMT_LINE_BREAK;
	if ( pItemColumnFormat->bFill )
		*outColFormat |= LVCFMT_FILL;
	if ( pItemColumnFormat->bAllowWrap )
		*outColFormat |= LVCFMT_WRAP;
	if ( pItemColumnFormat->bNoTitle )
		*outColFormat |= LVCFMT_NO_TITLE;
}

Void WinGUITable::_Convert_TileInfos( WinGUITableTileInfos * outTileInfos, const Void * pTileInfos ) const
{
	const LVTILEINFO * pDesc = (const LVTILEINFO *)pTileInfos;

	outTileInfos->iItemIndex = pDesc->iItem;

	outTileInfos->iColumnCount = pDesc->cColumns;
	for( UInt i = 0; i < pDesc->cColumns; ++i ) {
		outTileInfos->arrColumnIndices[i] = pDesc->puColumns[i];
		_Convert_ItemColumnFormat( outTileInfos->arrColumnFormats + i, pDesc->piColFmt[i] );
	}
}
Void WinGUITable::_Convert_TileInfos( Void * outTileInfos, const WinGUITableTileInfos * pTileInfos ) const
{
	LVTILEINFO * outDesc = (LVTILEINFO*)outTileInfos;

	outDesc->iItem = pTileInfos->iItemIndex;

	outDesc->cColumns = pTileInfos->iColumnCount;
	for( UInt i = 0; i < pTileInfos->iColumnCount; ++i ) {
		outDesc->puColumns[i] = pTileInfos->arrColumnIndices[i];
		_Convert_ItemColumnFormat( outDesc->piColFmt + i, pTileInfos->arrColumnFormats + i );
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
	m_bVirtualTable = pParameters->bVirtualTable;

	m_bHasBackBuffer = pParameters->bHasBackBuffer;
	m_bHasSharedImageLists = pParameters->bHasSharedImageLists;

	m_iItemCallBackMode = pParameters->iItemCallBackMode;
	m_iStateCallBackMode = pParameters->iStateCallBackMode;

	m_iViewMode = pParameters->iViewMode;
	m_bGroupMode = pParameters->bGroupMode;
	m_bHasHeadersInAllViews = pParameters->bHasHeadersInAllViews;

	m_bHasColumnHeaders = pParameters->bHasColumnHeaders;
	m_bHasStaticColumnHeaders = pParameters->bHasStaticColumnHeaders;
	m_bHasDraggableColumnHeaders = pParameters->bHasDraggableColumnHeaders;
	m_bHasIconColumnOverflowButton = pParameters->bHasIconColumnOverflowButton;

	m_bHasCheckBoxes = pParameters->bHasCheckBoxes;
	m_bHasIconLabels = pParameters->bHasIconLabels;
	m_bHasEditableLabels = pParameters->bHasEditableLabels;
	m_bHasSubItemImages = pParameters->bHasSubItemImages;

	m_bSingleItemSelection = pParameters->bSingleItemSelection;
	m_bIconSimpleSelection = pParameters->bIconSimpleSelection;

	if ( !m_bVirtualTable ) {
		m_bAutoSortAscending = pParameters->bAutoSortAscending;
		m_bAutoSortDescending = pParameters->bAutoSortDescending;
		DebugAssert( !m_bAutoSortAscending || !m_bAutoSortDescending );
	}

	m_bHasHotTrackingSingleClick = pParameters->bHasHotTrackingSingleClick;
	m_bHasHotTrackingDoubleClick = pParameters->bHasHotTrackingDoubleClick;
	DebugAssert( !m_bHasHotTrackingSingleClick || !m_bHasHotTrackingDoubleClick );
	if ( m_bHasHotTrackingSingleClick || m_bHasHotTrackingDoubleClick )
		m_bHasHotTrackingSelection = pParameters->bHasHotTrackingSelection;

	m_bHasInfoTips = pParameters->bHasInfoTips;

    // Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE );
	DWord dwStyleEx = LVS_EX_LABELTIP;

	if ( m_bVirtualTable )
		dwStyle |= LVS_OWNERDATA;

	if ( m_bHasBackBuffer )
		dwStyleEx |= LVS_EX_DOUBLEBUFFER;
	if ( m_bHasSharedImageLists )
		dwStyle |= LVS_SHAREIMAGELISTS;
	
	switch( m_iViewMode ) {
		case WINGUI_TABLE_VIEW_LIST:        dwStyle |= LVS_LIST; break;
		case WINGUI_TABLE_VIEW_ICONS:       dwStyle |= LVS_ICON; break;
		case WINGUI_TABLE_VIEW_ICONS_SMALL: dwStyle |= LVS_SMALLICON; break;
		case WINGUI_TABLE_VIEW_DETAILED:    dwStyle |= LVS_REPORT; break;
		case WINGUI_TABLE_VIEW_TILES:       dwStyle |= LVS_LIST; break; // Can't set at creation, delay
		default: DebugAssert(false); break;
	}
	// Can't set group mode at creation, delay
	if ( m_bHasHeadersInAllViews )
		dwStyleEx |= LVS_EX_HEADERINALLVIEWS;

	if ( !m_bHasColumnHeaders )
		dwStyle |= LVS_NOCOLUMNHEADER;
	if ( m_bHasStaticColumnHeaders )
		dwStyle |= LVS_NOSORTHEADER;
	if ( m_bHasDraggableColumnHeaders )
		dwStyleEx |= LVS_EX_HEADERDRAGDROP;
	if ( m_bHasIconColumnOverflowButton )
		dwStyleEx |= LVS_EX_COLUMNOVERFLOW;

	if ( m_bHasCheckBoxes )
		dwStyleEx |= LVS_EX_CHECKBOXES;
	if ( !m_bHasIconLabels )
		dwStyleEx |= LVS_EX_HIDELABELS;
	if ( m_bHasEditableLabels )
		dwStyle |= LVS_EDITLABELS;
	if ( m_bHasSubItemImages )
		dwStyleEx |= LVS_EX_SUBITEMIMAGES;

	if ( m_bSingleItemSelection )
		dwStyle |= LVS_SINGLESEL;
	if ( m_bIconSimpleSelection )
		dwStyleEx |= LVS_EX_SIMPLESELECT;

	if ( !m_bVirtualTable ) {
		if ( m_bAutoSortAscending )
			dwStyle |= LVS_SORTASCENDING;
		else if ( m_bAutoSortDescending )
			dwStyle |= LVS_SORTDESCENDING;
	}

	if ( m_bHasHotTrackingSingleClick ) {
		dwStyleEx |= LVS_EX_ONECLICKACTIVATE;
		if ( m_bHasHotTrackingSelection )
			dwStyleEx |= LVS_EX_TRACKSELECT;
	} else if ( m_bHasHotTrackingDoubleClick ) {
		dwStyleEx |= LVS_EX_TWOCLICKACTIVATE;
		if ( m_bHasHotTrackingSelection )
			dwStyleEx |= LVS_EX_TRACKSELECT;
	}

	if ( m_bHasInfoTips )
		dwStyleEx |= LVS_EX_INFOTIP;

    // Window creation
	m_hHandle = CreateWindowEx (
		0,
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

	// Apply extended styles after creation
	ListView_SetExtendedListViewStyle( (HWND)m_hHandle, dwStyleEx );

	// Set Tile Mode after creation
	if ( m_iViewMode == WINGUI_TABLE_VIEW_TILES )
		SwitchViewMode( WINGUI_TABLE_VIEW_TILES );

	// Set Group Mode after creation
	if ( m_bGroupMode )
		ToggleGroupMode( true );

	// Start with an empty list

	// Done
	_SaveElementToHandle();
	_RegisterSubClass();
}
Void WinGUITable::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

	// Remove SubClass
	_UnregisterSubClass();

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUITable::_DispatchEvent( Int iNotificationCode, Void * pParameters )
{
    // Get Model
	WinGUITableModel * pModel = (WinGUITableModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		// Focus
		case NM_SETFOCUS:  return pModel->OnFocusGained(); break;
		case NM_KILLFOCUS: return pModel->OnFocusLost(); break;

		// Keyboard
		case LVN_KEYDOWN: {
				NMLVKEYDOWN * pParams = (NMLVKEYDOWN*)pParameters;
				KeyCode iKey = KeyCodeFromWin32[pParams->wVKey];
				return pModel->OnKeyPress( iKey );
			} break;
		case NM_RETURN: return pModel->OnKeyPressEnter(); break;

		// Mouse
		case NM_CLICK: {
				NMITEMACTIVATE * pParams = (NMITEMACTIVATE*)pParameters;

				WinGUIPoint hMousePosition;
				hMousePosition.iX = pParams->ptAction.x;
				hMousePosition.iY = pParams->ptAction.y;

				return pModel->OnClickLeft( pParams->iItem, pParams->iSubItem, hMousePosition );
			} break;
		case NM_RCLICK: {
				NMITEMACTIVATE * pParams = (NMITEMACTIVATE*)pParameters;

				WinGUIPoint hMousePosition;
				hMousePosition.iX = pParams->ptAction.x;
				hMousePosition.iY = pParams->ptAction.y;

				return pModel->OnClickRight( pParams->iItem, pParams->iSubItem, hMousePosition );
			} break;
		case NM_DBLCLK: {
				NMITEMACTIVATE * pParams = (NMITEMACTIVATE*)pParameters;

				WinGUIPoint hMousePosition;
				hMousePosition.iX = pParams->ptAction.x;
				hMousePosition.iY = pParams->ptAction.y;

				return pModel->OnDblClickLeft( pParams->iItem, pParams->iSubItem, hMousePosition );
			} break;
		case NM_RDBLCLK: {
				NMITEMACTIVATE * pParams = (NMITEMACTIVATE*)pParameters;

				WinGUIPoint hMousePosition;
				hMousePosition.iX = pParams->ptAction.x;
				hMousePosition.iY = pParams->ptAction.y;

				return pModel->OnDblClickRight( pParams->iItem, pParams->iSubItem, hMousePosition );
			} break;
		case NM_HOVER: {
				// Must be an old message, mouse position is not provided !
				POINT hPt;
				GetCursorPos( &hPt );
				ScreenToClient( (HWND)m_hHandle, &hPt );

				WinGUIPoint hMousePosition;
				hMousePosition.iX = hPt.x;
				hMousePosition.iY = hPt.y;

				return pModel->OnHover( hMousePosition );
			} break;

		// Scrolling
		case LVN_BEGINSCROLL: {
				NMLVSCROLL * pParams = (NMLVSCROLL*)pParameters;
				WinGUIPoint hScrollPoint;
				hScrollPoint.iX = pParams->dx;
				hScrollPoint.iY = pParams->dy;
				return pModel->OnScrollStart( hScrollPoint );
			} break;
		case LVN_ENDSCROLL: {
				NMLVSCROLL * pParams = (NMLVSCROLL*)pParameters;
				WinGUIPoint hScrollPoint;
				hScrollPoint.iX = pParams->dx;
				hScrollPoint.iY = pParams->dy;
				return pModel->OnScrollEnd( hScrollPoint );
			} break;

		// Empty Table Text
		case LVN_GETEMPTYMARKUP: {
				NMLVEMPTYMARKUP * pParams = (NMLVEMPTYMARKUP*)pParameters;
				Bool bCentered = false;
				Bool bSetMarkup = pModel->OnRequestEmptyText( pParams->szMarkup, L_MAX_URL_LENGTH, &bCentered );
				pParams->dwFlags = bCentered ? EMF_CENTERED : 0;
				return bSetMarkup;
			} break;

		// Column Headers Interactions
		case LVN_COLUMNCLICK: {
				NMLISTVIEW * pParams = (NMLISTVIEW*)pParameters;
				return pModel->OnColumnHeaderClick( pParams->iSubItem );
			} break;

		// Group Interactions
		case LVN_LINKCLICK: {
				NMLVLINK * pParams = (NMLVLINK*)pParameters;
				////////////////////////////////////////////
				return pModel->OnGroupLinkClick( pParams->iItem, pParams->iSubItem );
			} break;

		// Item Interactions
		case LVN_GETDISPINFO: {
				NMLVDISPINFO * pParams = (NMLVDISPINFO*)pParameters;
				UInt iItemIndex = pParams->item.iItem;
				UInt iSubItemIndex = pParams->item.iSubItem;
				UInt iMask = pParams->item.mask;

				Void * pItemData = NULL;
				if ( iMask & LVIF_PARAM )
					pItemData = (Void*)( pParams->item.lParam );

				// Request Item Label Text
				if ( (iMask & LVIF_TEXT) != 0 ) {
					pParams->item.pszText = pModel->OnRequestItemLabel( iItemIndex, iSubItemIndex, pItemData );
				}

				// Request Item Icon Image Index
				if ( (iMask & LVIF_IMAGE) != 0 ) {
					UInt iImage = pModel->OnRequestItemIconImage( iItemIndex, iSubItemIndex, pItemData );
					pParams->item.iImage = (iImage != INVALID_OFFSET) ? iImage : I_IMAGENONE;
				}

				// Request Item GroupID
				if ( (iMask & LVIF_GROUPID) != 0 ) {
					UInt iGroupID = pModel->OnRequestItemGroupID( iItemIndex, iSubItemIndex, pItemData );
					pParams->item.iGroupId = (iGroupID != INVALID_OFFSET) ? iGroupID : I_GROUPIDNONE;
				}

				// Request Item Indentation
				if ( (iMask & LVIF_INDENT) != 0 ) {
					pParams->item.iIndent = pModel->OnRequestItemIndentation( iItemIndex, iSubItemIndex, pItemData );
				}

				// Request Item Columns
				if ( (iMask & LVIF_COLUMNS) != 0 ) {
					pParams->item.cColumns = pModel->OnRequestItemColumnCount( iItemIndex, iSubItemIndex, pItemData );
					pParams->item.puColumns = pModel->OnRequestItemColumnIndices( iItemIndex, iSubItemIndex, pItemData );
				}

				// Request Item State
				if ( (iMask & LVIF_STATE) != 0 ) {
					UInt iStateMask = pParams->item.stateMask;

					// Request Overlay Image Index
					if ( (iStateMask & LVIS_OVERLAYMASK) != 0 ) {
						UInt iOverlayImageIndex = pModel->OnRequestItemOverlayImage( iItemIndex, iSubItemIndex, pItemData );
						pParams->item.state |= INDEXTOOVERLAYMASK( iOverlayImageIndex & 0x0f );
					}

					// Request State Image Index
					if ( (iStateMask & LVIS_STATEIMAGEMASK) != 0 ) {
						UInt iStateImageIndex = pModel->OnRequestItemStateImage( iItemIndex, iSubItemIndex, pItemData );
						pParams->item.state |= INDEXTOSTATEIMAGEMASK( iStateImageIndex & 0x0f );
					}

					// Request Focus State
					if ( (iStateMask & LVIS_FOCUSED) != 0 ) {
						Bool bHasFocus = pModel->OnRequestItemFocusState( iItemIndex, iSubItemIndex, pItemData );
						pParams->item.state |= ( bHasFocus ? LVIS_FOCUSED : 0 );
					}

					// Request Selection State
					if ( (iStateMask & LVIS_SELECTED) != 0 ) {
						Bool bIsSelected = pModel->OnRequestItemSelectState( iItemIndex, iSubItemIndex, pItemData );
						pParams->item.state |= ( bIsSelected ? LVIS_SELECTED : 0 );
					}

					// Request CutMark State
					if ( (iStateMask & LVIS_CUT) != 0 ) {
						Bool bIsCutMarked = pModel->OnRequestItemCutMarkState( iItemIndex, iSubItemIndex, pItemData );
						pParams->item.state |= ( bIsCutMarked ? LVIS_CUT : 0 );
					}

					// Request DropHighlight State
					if ( (iStateMask & LVIS_DROPHILITED) != 0 ) {
						Bool bIsDropHighlighted = pModel->OnRequestItemDropHighlightState( iItemIndex, iSubItemIndex, pItemData );
						pParams->item.state |= ( bIsDropHighlighted ? LVIS_DROPHILITED : 0 );
					}
				}

				return true;
			} break;
		case LVN_SETDISPINFO: {
				NMLVDISPINFO * pParams = (NMLVDISPINFO*)pParameters;
				UInt iItemIndex = pParams->item.iItem;
				UInt iSubItemIndex = pParams->item.iSubItem;
				UInt iMask = pParams->item.mask;

				Void * pItemData = NULL;
				if ( iMask & LVIF_PARAM )
					pItemData = (Void*)( pParams->item.lParam );

				// Update Item Label Text
				if ( (iMask & LVIF_TEXT) != 0 ) {
					pModel->OnUpdateItemLabel( iItemIndex, iSubItemIndex, pItemData, pParams->item.pszText );
				}

				// Update Item Icon Image Index
				if ( (iMask & LVIF_IMAGE) != 0 ) {
					UInt iImage = (pParams->item.iImage != I_IMAGENONE) ? pParams->item.iImage : INVALID_OFFSET;
					pModel->OnUpdateItemIconImage( iItemIndex, iSubItemIndex, pItemData, iImage );
				}

				// Update Item GroupID
				if ( (iMask & LVIF_GROUPID) != 0 ) {
					UInt iGroupId = (pParams->item.iGroupId != I_GROUPIDNONE) ? pParams->item.iGroupId : INVALID_OFFSET;
					pModel->OnUpdateItemGroupID( iItemIndex, iSubItemIndex, pItemData, iGroupId );
				}

				// Update Item Indentation
				if ( (iMask & LVIF_INDENT) != 0 ) {
					pModel->OnUpdateItemIndentation( iItemIndex, iSubItemIndex, pItemData, pParams->item.iIndent );
				}

				// Update Item Columns
				if ( (iMask & LVIF_COLUMNS) != 0 ) {
					pModel->OnUpdateItemColumnIndices( iItemIndex, iSubItemIndex, pItemData, pParams->item.puColumns, pParams->item.cColumns );
				}

				// Update Item State
				if ( (iMask & LVIF_STATE) != 0 ) {
					UInt iStateMask = pParams->item.stateMask;

					// Update Overlay Image Index
					if ( (iStateMask & LVIS_OVERLAYMASK) != 0 ) {
						UInt iOverlayImageIndex = ( (pParams->item.state & LVIS_OVERLAYMASK) >> 8 );
						pModel->OnUpdateItemOverlayImage( iItemIndex, iSubItemIndex, pItemData, iOverlayImageIndex );
					}

					// Update State Image Index
					if ( (iStateMask & LVIS_STATEIMAGEMASK) != 0 ) {
						UInt iStateImageIndex = ( (pParams->item.state & LVIS_STATEIMAGEMASK) >> 12 );
						pModel->OnUpdateItemStateImage( iItemIndex, iSubItemIndex, pItemData, iStateImageIndex );
					}

					// Update Focus State
					if ( (iStateMask & LVIS_FOCUSED) != 0 ) {
						Bool bHasFocus = ( (pParams->item.state & LVIS_FOCUSED) != 0 );
						pModel->OnUpdateItemFocusState( iItemIndex, iSubItemIndex, pItemData, bHasFocus );
					}

					// Update Selection State
					if ( (iStateMask & LVIS_SELECTED) != 0 ) {
						Bool bIsSelected = ( (pParams->item.state & LVIS_SELECTED) != 0 );
						pModel->OnUpdateItemSelectState( iItemIndex, iSubItemIndex, pItemData, bIsSelected );
					}

					// Update CutMark State
					if ( (iStateMask & LVIS_CUT) != 0 ) {
						Bool bIsCutMarked = ( (pParams->item.state & LVIS_CUT) != 0 );
						pModel->OnUpdateItemCutMarkState( iItemIndex, iSubItemIndex, pItemData, bIsCutMarked );
					}

					// Update DropHighlight State
					if ( (iStateMask & LVIS_DROPHILITED) != 0 ) {
						Bool bIsDropHighlighted = ( (pParams->item.state & LVIS_DROPHILITED) != 0 );
						pModel->OnUpdateItemDropHighlightState( iItemIndex, iSubItemIndex, pItemData, bIsDropHighlighted );
					}
				}

				return true;
			} break;

		case LVN_INSERTITEM: {
				NMLISTVIEW * pParams = (NMLISTVIEW*)pParameters;
				return pModel->OnAddItem( pParams->iItem );
			} break;
		case LVN_DELETEITEM: {
				NMLISTVIEW * pParams = (NMLISTVIEW*)pParameters;
				return pModel->OnRemoveItem( pParams->iItem, (Void*)(pParams->lParam) );
			} break;
		case LVN_DELETEALLITEMS: {
				NMLISTVIEW * pParams = (NMLISTVIEW*)pParameters;
				return pModel->OnRemoveAllItems(); // return false to receive subsequent LVN_DELETEITEM notifications
			} break;

		case LVN_ITEMACTIVATE: {
				NMITEMACTIVATE * pParams = (NMITEMACTIVATE*)pParameters;

				WinGUITableItemState hOldState;
				_Convert_ItemState( &hOldState, pParams->uOldState );
				WinGUITableItemState hNewState;
				_Convert_ItemState( &hNewState, pParams->uNewState );

				WinGUIPoint hHotPoint;
				hHotPoint.iX = pParams->ptAction.x;
				hHotPoint.iY = pParams->ptAction.y;

				Bool bShiftPressed = ( (pParams->uKeyFlags & LVKF_SHIFT) != 0 );
				Bool bCtrlPressed = ( (pParams->uKeyFlags & LVKF_CONTROL) != 0 );
				Bool bAltPressed = ( (pParams->uKeyFlags & LVKF_ALT) != 0 );

				return pModel->OnItemActivation( pParams->iItem, hOldState, hNewState, hHotPoint, bShiftPressed, bCtrlPressed, bAltPressed );
			} break;

		case LVN_ITEMCHANGING: {
				NMLISTVIEW * pParams = (NMLISTVIEW*)pParameters;

				WinGUITableItemState hOldState;
				_Convert_ItemState( &hOldState, pParams->uOldState );
				WinGUITableItemState hNewState;
				_Convert_ItemState( &hNewState, pParams->uNewState );

				WinGUIPoint hHotPoint;
				hHotPoint.iX = pParams->ptAction.x;
				hHotPoint.iY = pParams->ptAction.y;

				return pModel->OnItemChanging( pParams->iItem, pParams->iSubItem, (Void*)(pParams->lParam), hOldState, hNewState, hHotPoint ); // return false to allow the change
			} break;
		case LVN_ITEMCHANGED: {
				NMLISTVIEW * pParams = (NMLISTVIEW*)pParameters;

				WinGUITableItemState hOldState;
				_Convert_ItemState( &hOldState, pParams->uOldState );
				WinGUITableItemState hNewState;
				_Convert_ItemState( &hNewState, pParams->uNewState );

				WinGUIPoint hHotPoint;
				hHotPoint.iX = pParams->ptAction.x;
				hHotPoint.iY = pParams->ptAction.y;

				return pModel->OnItemChanged( pParams->iItem, pParams->iSubItem, (Void*)(pParams->lParam), hOldState, hNewState, hHotPoint );
			} break;

		// BoundingBox Selection
		case LVN_MARQUEEBEGIN: return pModel->OnBoundingBoxSelection(); break; // Return false to allow selection

		// Hot Tracking Selection
		case LVN_HOTTRACK: {
				NMLISTVIEW * pParams = (NMLISTVIEW*)pParameters;

				WinGUIPoint hHotPoint;
				hHotPoint.iX = pParams->ptAction.x;
				hHotPoint.iY = pParams->ptAction.y;

				return pModel->OnHotTrackSelection( (UInt*)&(pParams->iItem), pParams->iSubItem, hHotPoint ); // Return false to allow selection
			} break;

		// Editable Labels
		case LVN_BEGINLABELEDIT: {
				NMLVDISPINFO * pParams = (NMLVDISPINFO*)pParameters;

				m_hEditLabelHandle = ListView_GetEditControl( (HWND)m_hHandle );
				m_iEditLabelItemIndex = pParams->item.iItem;

				return pModel->OnLabelEditStart(); // Return false to allow edition
			} break;
		case LVN_ENDLABELEDIT: {
				NMLVDISPINFO * pParams = (NMLVDISPINFO*)pParameters;

				if ( pParams->item.pszText == NULL ) {
					Bool bHandled = pModel->OnLabelEditCancel();

					m_hEditLabelHandle = NULL;
					m_iEditLabelItemIndex = INVALID_OFFSET;

					return bHandled;
				} else {
					Bool bAllowModification = pModel->OnLabelEditEnd( pParams->item.pszText );

					m_hEditLabelHandle = NULL;
					m_iEditLabelItemIndex = INVALID_OFFSET;

					SetWindowLongPtr( (HWND)(_GetHandle(m_pParent)), DWLP_MSGRESULT, bAllowModification ? TRUE : FALSE );
					return true;
				}
			} break;

		// Incremental Search
		case LVN_INCREMENTALSEARCH: {
				NMLVFINDITEM * pParams = (NMLVFINDITEM*)pParameters;

				// Ask for Search Options, Give opportunity for use search
				UInt iSearchResult = INVALID_OFFSET;
				UInt iStartItemIndex = 0;
				WinGUITableSearchOptions hSearchOptions;
				Bool bUserSearchDone = pModel->OnIncrementalSearch( &iSearchResult, &iStartItemIndex, &hSearchOptions );

				// User has performed the search himself
				if ( bUserSearchDone ) {
					pParams->lvfi.flags = LVFI_PARAM;
					pParams->lvfi.lParam = (LPARAM)iSearchResult;
					return true;
				}

				// User has set default string search parameters
				pParams->iStart = ( (Int)iStartItemIndex - 1 );
				pParams->lvfi.flags = 0;
				switch( hSearchOptions.iMode ) {
					case WINGUI_TABLE_SEARCH_STRING:
						pParams->lvfi.flags = LVFI_STRING;
						if ( hSearchOptions.bWrapAround )
							pParams->lvfi.flags |= LVFI_WRAP;
						break;
					case WINGUI_TABLE_SEARCH_SUBSTRING:
						pParams->lvfi.flags = LVFI_STRING | LVFI_PARTIAL;
						if ( hSearchOptions.bWrapAround )
							pParams->lvfi.flags |= LVFI_WRAP;
						break;
					default: DebugAssert(false); break;
				}

				return true;
			} break;

		// Drag & Drop
		case LVN_BEGINDRAG: {
				NMLISTVIEW * pParams = (NMLISTVIEW*)pParameters;
				return pModel->OnDragLeftStart( pParams->iItem );
			} break;
		case LVN_BEGINRDRAG: {
				NMLISTVIEW * pParams = (NMLISTVIEW*)pParameters;
				return pModel->OnDragRightStart( pParams->iItem );
			} break;

		// Info Tips
		case LVN_GETINFOTIP: {
				NMLVGETINFOTIP * pParams = (NMLVGETINFOTIP*)pParameters;

				// Retrieve Item Text if needed
				Bool bFullItemTextVisible = ( (pParams->dwFlags & LVGIT_UNFOLDED) != 0 );
				if ( !bFullItemTextVisible ) {
					if ( (m_iItemCallBackMode & WINGUI_TABLE_ITEMCALLBACK_LABELS) == 0 )
						GetItemLabel( pParams->pszText, pParams->cchTextMax, pParams->iItem, 0 );
					else {
						const GChar * strLabel = pModel->OnRequestItemLabel( pParams->iItem, 0, (Void*)(pParams->lParam) );
						StringFn->NCopy( pParams->pszText, strLabel, pParams->cchTextMax - 1 );
					}
				}

				// Setup for info tip text append
				UInt iCurrentLength = StringFn->Length( pParams->pszText );
				GChar * outAppendText = ( pParams->pszText + iCurrentLength );

				return pModel->OnRequestInfoTip( outAppendText, pParams->cchTextMax - iCurrentLength, pParams->iItem );
			} break;

		// TODO : Check those when HeaderControl gets implemented
		// LVN_COLUMNDROPDOWN      // When using a HeaderControl as child of the listview ...
		// LVN_COLUMNOVERFLOWCLICK // When using a HeaderControl as child of the listview ...

		// TODO : Virtual Tables (Unimplemented yet)
		//case LVN_ODCACHEHINT:     return pModel->OnRequestCache(); break;
		//case LVN_ODSTATECHANGED : return pModel->OnRequestUpdate(); break;
		//case LVN_ODFINDITEM:      return pModel->OnRequestSearch(); break;

		default: break;
	}

	// Unhandled
	return false;
}


