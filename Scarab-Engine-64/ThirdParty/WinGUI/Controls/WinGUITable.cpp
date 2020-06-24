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

Void WinGUITable::AdjustRequiredDimensions( UInt * pWidth, UInt * pHeight, UInt iItemCount ) const
{
	HWND hHandle = (HWND)m_hHandle;

	DWord dwResult = ListView_ApproximateViewRect( hHandle, *pWidth, *pHeight, iItemCount );
	*pWidth = (UInt)( LOWORD(dwResult) );
	*pHeight = (UInt)( HIWORD(dwResult) );
}

UInt WinGUITable::GetVisibleItemCount() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetCountPerPage( hHandle );
}

Void WinGUITable::GetEmptyText( GChar * outText, UInt iMaxLength ) const
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_GetEmptyText( hHandle, outText, iMaxLength );
}

//UInt WinGUITable::GetBackgroundColor() const
//{
//	HWND hHandle = (HWND)m_hHandle;
//	return ListView_GetBkColor( hHandle );
//}

Void WinGUITable::MakeItemVisible( UInt iIndex, Bool bAllowPartial )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_EnsureVisible( hHandle, iIndex, bAllowPartial ? TRUE : FALSE );
}

Void WinGUITable::GetColumnInfos( UInt iIndex, WinGUITableColumnInfos * outInfos ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVCOLUMN hColumnInfos;
	hColumnInfos.mask = LVCF_FMT | LVCF_WIDTH | LVCF_TEXT | LVCF_SUBITEM | LVCF_IMAGE | LVCF_ORDER | LVCF_MINWIDTH | LVCF_DEFAULTWIDTH | LVCF_IDEALWIDTH;
	hColumnInfos.pszText = outInfos->strHeaderText;
	hColumnInfos.cchTextMax = 64;

	ListView_GetColumn( hHandle, iIndex, &hColumnInfos );

	outInfos->iOrderIndex = hColumnInfos.iOrder;
	outInfos->iSubItemIndex = hColumnInfos.iSubItem;

	switch( hColumnInfos.fmt & LVCFMT_JUSTIFYMASK ) {
		case LVCFMT_LEFT:   outInfos->iRowsTextAlign = WINGUI_TABLE_TEXT_ALIGN_LEFT; break;
		case LVCFMT_RIGHT:  outInfos->iRowsTextAlign = WINGUI_TABLE_TEXT_ALIGN_RIGHT; break;
		case LVCFMT_CENTER: outInfos->iRowsTextAlign = WINGUI_TABLE_TEXT_ALIGN_CENTER; break;
		default: DebugAssert(false); break;
	}

	outInfos->bHeaderSplitButton = ( (hColumnInfos.fmt & LVCFMT_SPLITBUTTON) != 0 );

	outInfos->bHeaderHasImage = ( (hColumnInfos.fmt & LVCFMT_COL_HAS_IMAGES) != 0 );
	outInfos->bRowsHaveImages = ( (hColumnInfos.fmt & LVCFMT_IMAGE) != 0 );
	outInfos->bIsImageOnRight = ( (hColumnInfos.fmt & LVCFMT_BITMAP_ON_RIGHT) != 0 );
	outInfos->iImageListIndex = hColumnInfos.iImage;

	outInfos->bFixedWidth = ( (hColumnInfos.fmt & LVCFMT_FIXED_WIDTH) != 0 );
	outInfos->bFixedAspectRatio = ( (hColumnInfos.fmt & LVCFMT_FIXED_RATIO) != 0 );
	outInfos->iWidth = hColumnInfos.cx;
	outInfos->iMinWidth = hColumnInfos.cxMin;
	outInfos->iDefaultWidth = hColumnInfos.cxDefault;
	outInfos->iIdealWidth = hColumnInfos.cxIdeal;
}
Void WinGUITable::GetColumnOrder( UInt * outOrderedIndices, UInt iCount ) const
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_GetColumnOrderArray( hHandle, iCount, outOrderedIndices );
}
UInt WinGUITable::GetColumnWidth( UInt iIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetColumnWidth( hHandle, iIndex );
}

Void WinGUITable::RemoveColumn( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ListView_DeleteColumn( hHandle, iIndex );
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
}

UInt WinGUITable::GetGroupCount() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetGroupCount( hHandle );
}
UInt WinGUITable::GetFocusedGroup() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ListView_GetFocusedGroup( hHandle );
}

Bool WinGUITable::IsItemChecked( UInt iIndex ) const
{
	DebugAssert( m_bHasCheckBoxes == true );

	HWND hHandle = (HWND)m_hHandle;
	ListView_GetCheckState( hHandle, iIndex );
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
UInt WinGUITable::SearchItem( Void * pData, UInt iStartIndex, Bool bWrapAround ) const
{
	HWND hHandle = (HWND)m_hHandle;

	LVFINDINFO hInfos;
	hInfos.flags = LVFI_PARAM;
	if ( bWrapAround )
		hInfos.flags |= LVFI_WRAP;
	hInfos.lParam = (LPARAM)pData;

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

/////////////////////////////////////////////////////////////////////////////////

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


