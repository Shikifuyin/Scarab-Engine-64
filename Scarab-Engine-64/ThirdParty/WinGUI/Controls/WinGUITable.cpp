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
	// nothing to do
}
WinGUITable::~WinGUITable()
{
	// nothing to do
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

	// Save if this is a virtual table or not
	m_bVirtualTable = pParameters->bMakeVirtualTable;

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

Bool WinGUITable::_DispatchEvent( Int iNotificationCode )
{
    // Get Model
	WinGUIButtonModel * pModel = (WinGUIButtonModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		case LVN_BEGINDRAG:      return pModel->OnStartDragLeft(); break;
		case LVN_BEGINRDRAG:     return pModel->OnStartDragRight(); break;
		case LVN_BEGINSCROLL:    return pModel->OnStartScroll(); break;
		case LVN_BEGINLABELEDIT: return pModel->OnLabelEdit(); break;
		case LVN_COLUMNCLICK:    return pModel->OnHeaderClick(); break;


		default: break;
	}

	// Unhandled
	return false;
}


