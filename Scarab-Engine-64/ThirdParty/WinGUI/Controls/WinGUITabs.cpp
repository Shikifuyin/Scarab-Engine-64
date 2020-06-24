/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUITabs.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : Tabs
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
#include "WinGUITabs.h"

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUITabsModel implementation
WinGUITabsModel::WinGUITabsModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// Default Parameters
	m_hCreationParameters.bSingleLine = true;
	m_hCreationParameters.bFixedWidth = true;
	m_hCreationParameters.iTabCount = 2;

	StringFn->Copy( m_hCreationParameters.arrTabs[0].strLabel, TEXT("Tab1") );
	m_hCreationParameters.arrTabs[0].pUserData = NULL;

	StringFn->Copy( m_hCreationParameters.arrTabs[1].strLabel, TEXT("Tab2") );
	m_hCreationParameters.arrTabs[1].pUserData = NULL;
}
WinGUITabsModel::~WinGUITabsModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUITabs implementation
WinGUITabs::WinGUITabs( WinGUIElement * pParent, WinGUITabsModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	m_pSelectedTabPane = NULL;
}
WinGUITabs::~WinGUITabs()
{
	// nothing to do
}

Void WinGUITabs::GetDisplayArea( WinGUIRectangle * outDisplayArea ) const
{
	HWND hHandle = (HWND)m_hHandle;
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

	RECT hRect;
	::GetClientRect( hParentWnd, &hRect );

	TabCtrl_AdjustRect( hHandle, FALSE, &hRect );

	outDisplayArea->iLeft = hRect.left;
	outDisplayArea->iTop = hRect.top;
	outDisplayArea->iWidth = ( hRect.right - hRect.left );
	outDisplayArea->iHeight = ( hRect.bottom - hRect.top );
}

Void WinGUITabs::SwitchSelectedTabPane( WinGUIContainer * pSelectedTabPane )
{
	// Previous Selection : Set Invisible and Throw Back
	if ( m_pSelectedTabPane != NULL ) {
		HWND hHandle = (HWND)( _GetHandle(m_pSelectedTabPane) );
		SetWindowPos( hHandle, HWND_BOTTOM, 0, 0, 0, 0, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE );
		m_pSelectedTabPane->SetVisible( false );
	}

	// New Selection : Set Visible and Bring Front
	m_pSelectedTabPane = pSelectedTabPane;
	HWND hHandle = (HWND)( _GetHandle(m_pSelectedTabPane) );
	SetWindowPos( hHandle, HWND_TOP, 0, 0, 0, 0, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE );
	m_pSelectedTabPane->SetVisible( true );
}

UInt WinGUITabs::GetTabsRowCount() const
{
	HWND hHandle = (HWND)m_hHandle;
	return TabCtrl_GetRowCount( hHandle );
}

Void WinGUITabs::SetMinTabWidth( UInt iWidth )
{
	HWND hHandle = (HWND)m_hHandle;
	TabCtrl_SetMinTabWidth( hHandle, (Int)iWidth );
}
Void WinGUITabs::SetTabButtonPadding( UInt iHPadding, UInt iVPadding )
{
	HWND hHandle = (HWND)m_hHandle;
	TabCtrl_SetPadding( hHandle, iHPadding, iVPadding );
}

Void WinGUITabs::AddTab( UInt iIndex, GChar * strLabel, Void * pUserData )
{
	HWND hHandle = (HWND)m_hHandle;

	TCITEM hTabItem;
	hTabItem.mask = TCIF_TEXT | TCIF_IMAGE | TCIF_PARAM;
	hTabItem.pszText = strLabel;
	hTabItem.iImage = -1;
	hTabItem.lParam = (LPARAM)pUserData;

	TabCtrl_InsertItem( hHandle, iIndex, &hTabItem );
}
Void WinGUITabs::RemoveTab( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	TabCtrl_DeleteItem( hHandle, iIndex );
}
Void WinGUITabs::RemoveAllTabs()
{
	HWND hHandle = (HWND)m_hHandle;
	TabCtrl_DeleteAllItems( hHandle );
}

Void WinGUITabs::UpdateTab( UInt iIndex, GChar * strLabel, Void * pUserData )
{
	HWND hHandle = (HWND)m_hHandle;

	TCITEM hTabItem;
	hTabItem.mask = 0;
	if ( strLabel != NULL ) {
		hTabItem.mask |= TCIF_TEXT;
		hTabItem.pszText = strLabel;
	}
	if ( pUserData != NULL ) {
		hTabItem.mask |= TCIF_PARAM;
		hTabItem.lParam = (LPARAM)pUserData;
	}
	if ( hTabItem.mask == 0 )
		return;

	TabCtrl_SetItem( hHandle, iIndex, &hTabItem );
}

UInt WinGUITabs::GetSelectedTab() const
{
	HWND hHandle = (HWND)m_hHandle;
	return TabCtrl_GetCurSel( hHandle );
}
Void WinGUITabs::SelectTab( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	TabCtrl_SetCurSel( hHandle, iIndex );
}
Void WinGUITabs::UnselectAll( Bool bKeepCurrent )
{
	HWND hHandle = (HWND)m_hHandle;
	TabCtrl_DeselectAll( hHandle, bKeepCurrent ? TRUE : FALSE );
}

//WinGUIToolTip * WinGUITabs::GetToolTip() const
//{
//	HWND hHandle = (HWND)m_hHandle;
//	HWND hToolTipHandle = TabCtrl_GetToolTips( hHandle );
//	return (WinGUIToolTip*)( _GetElementFromHandle(hToolTipHandle) );
//}
//Void WinGUITabs::SetToolTip( WinGUIToolTip * pToolTip )
//{
//	HWND hHandle = (HWND)m_hHandle;
//	HWND hToolTipHandle = (HWND)( _GetHandle(pToolTip) );
//	TabCtrl_SetToolTips( hHandle, hToolTipHandle );
//}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUITabs::_Create()
{
	DebugAssert( m_hHandle == NULL );

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUITabsModel * pModel = (WinGUITabsModel*)m_pModel;

	// Compute Layout
    const WinGUILayout * pLayout = pModel->GetLayout();

    WinGUIRectangle hParentRect;
    m_pParent->GetClientRect( &hParentRect );

    WinGUIRectangle hWindowRect;
    pLayout->ComputeLayout( &hWindowRect, hParentRect );

	// Get Creation Parameters
    WinGUITabsParameters * pParameters = pModel->GetCreationParameters();

	// Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | TCS_TABS );
	if ( pParameters->bSingleLine )
		dwStyle |= TCS_SINGLELINE;
	if ( pParameters->bFixedWidth )
		dwStyle |= TCS_FIXEDWIDTH;

    // Window creation
	m_hHandle = CreateWindowEx (
		0,
		WC_TABCONTROL,
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

	// Add Tabs
	UInt iTabCount = pParameters->iTabCount;
	for( UInt i = 0; i < iTabCount; ++i ) {
		TCITEM hTabItem;
		hTabItem.mask = TCIF_TEXT | TCIF_IMAGE | TCIF_PARAM;
		hTabItem.pszText = pParameters->arrTabs[i].strLabel;
		hTabItem.iImage = -1;
		hTabItem.lParam = (LPARAM)( pParameters->arrTabs[i].pUserData );

		TabCtrl_InsertItem( (HWND)m_hHandle, i, &hTabItem );
	}

	// Done
	_SaveElementToHandle();
}
Void WinGUITabs::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUITabs::_DispatchEvent( Int iNotificationCode, Void * pParameters )
{
    // Get Model
	WinGUITabsModel * pModel = (WinGUITabsModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		case TCN_FOCUSCHANGE: return pModel->OnFocusChange(); break;

		case TCN_KEYDOWN: {
			NMTCKEYDOWN * pParams = (NMTCKEYDOWN*)pParameters;
			KeyCode iKey = KeyCodeFromWin32[pParams->wVKey & 0xff];
			UInt iRepeatCount = ( pParams->flags & 0x0000ffff );
			Bool bPreviouslyDown = ( (pParams->flags & 0x40000000) != 0 );
			return pModel->OnKeyPress( iKey, iRepeatCount, bPreviouslyDown );
		} break;

		case NM_CLICK: return pModel->OnClick(); break;
		case NM_RCLICK: return pModel->OnRightClick(); break;
		case NM_DBLCLK: return pModel->OnDblClick(); break;
		case NM_RDBLCLK: return pModel->OnRightDblClick(); break;

		case TCN_SELCHANGING: return pModel->OnPreventSelect(); break;
		case TCN_SELCHANGE:   return pModel->OnSelect(); break;
		default: break;
	}

	// Unhandled
	return false;
}

