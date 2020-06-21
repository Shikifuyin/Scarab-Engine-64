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
	// nothing to do
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
		m_pSelectedTabPane->SetVisible( false );
		HWND hHandle = (HWND)( _GetHandle(m_pSelectedTabPane) );
		SetWindowPos( hHandle, HWND_BOTTOM, 0, 0, 0, 0, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE );
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

	WinGUITabsModel * pModel = (WinGUITabsModel*)m_pModel;
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

	// Get Parent Dimension
	RECT hClientArea;
	::GetClientRect( hParentWnd, &hClientArea );

	m_hHandle = CreateWindowEx (
		0, WC_TABCONTROL, TEXT(""),
		WS_VISIBLE | WS_CHILD | TCS_TABS | TCS_SINGLELINE | TCS_FIXEDWIDTH,
		0, 0, hClientArea.right, hClientArea.bottom,
		hParentWnd, (HMENU)m_iResourceID,
		(HINSTANCE)( GetWindowLongPtr(hParentWnd,GWLP_HINSTANCE) ),
		NULL
	);
	DebugAssert( m_hHandle != NULL );

	// Add Tabs
	UInt iTabCount = pModel->GetTabCount();
	for( UInt i = 0; i < iTabCount; ++i ) {
		TCITEM hTabItem;
		hTabItem.mask = TCIF_TEXT | TCIF_IMAGE | TCIF_PARAM;
		hTabItem.pszText = pModel->GetTabLabel(i);
		hTabItem.iImage = -1;
		hTabItem.lParam = (LPARAM)( pModel->GetTabUserData(i) );

		TabCtrl_InsertItem( (HWND)m_hHandle, i, &hTabItem );
	}

	// Done
	_SaveElementToHandle();
}
Void WinGUITabs::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUITabs::_DispatchEvent( Int iNotificationCode )
{
	WinGUITabsModel * pModel = (WinGUITabsModel*)m_pModel;

	// Dispatch Event to our Model
	switch( iNotificationCode ) {
		case TCN_SELCHANGING:
			return false; // Allow selection to change
			break;
		case TCN_SELCHANGE: {
			return pModel->OnTabSelect();
		} break;
		default: break;
	}

	// Unhandled
	return false;
}

