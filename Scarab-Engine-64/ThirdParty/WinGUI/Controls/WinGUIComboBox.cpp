/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIComboBox.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : ComboBox
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
#include "WinGUIComboBox.h"

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUIComboBoxModel implementation
WinGUIComboBoxModel::WinGUIComboBoxModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// Default Parameters
	m_hCreationParameters.iType = WINGUI_COMBOBOX_BUTTON;
	m_hCreationParameters.iCase = WINGUI_COMBOBOX_CASE_BOTH;
	m_hCreationParameters.iInitialSelectedItem = 0;
	m_hCreationParameters.bAllowHorizontalScroll = false;
	m_hCreationParameters.bAutoSort = false;
	m_hCreationParameters.bEnableTabStop = true;
}
WinGUIComboBoxModel::~WinGUIComboBoxModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIComboBox implementation
WinGUIComboBox::WinGUIComboBox( WinGUIElement * pParent, WinGUIComboBoxModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	// nothing to do
}
WinGUIComboBox::~WinGUIComboBox()
{
	// nothing to do
}

Void WinGUIComboBox::Enable()
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_Enable( hHandle, TRUE );
}
Void WinGUIComboBox::Disable()
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_Enable( hHandle, FALSE );
}

UInt WinGUIComboBox::GetMinVisibleItems() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ComboBox_GetMinVisible( hHandle );
}
Void WinGUIComboBox::SetMinVisibleItems( UInt iCount )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_SetMinVisible( hHandle, iCount );
}

UInt WinGUIComboBox::GetSelectionHeight() const
{
	HWND hHandle = (HWND)m_hHandle;
	return (UInt)( SendMessage(hHandle, CB_GETITEMHEIGHT, -1, 0) );
}
Void WinGUIComboBox::SetSelectionHeight( UInt iHeight )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_SetItemHeight( hHandle, -1, iHeight );
}
UInt WinGUIComboBox::GetListItemHeight() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ComboBox_GetItemHeight( hHandle );
}
Void WinGUIComboBox::SetListItemHeight( UInt iHeight )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_SetItemHeight( hHandle, 0, iHeight );
}

Void WinGUIComboBox::SetTextLimit( UInt iMaxLength )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_LimitText( hHandle, iMaxLength );
}

UInt WinGUIComboBox::GetItemCount() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ComboBox_GetCount( hHandle );
}
UInt WinGUIComboBox::GetItemStringLength( UInt iIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;
	return ComboBox_GetLBTextLen( hHandle, iIndex );
}
Void WinGUIComboBox::GetItemString( UInt iIndex, GChar * outBuffer ) const
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_GetLBText( hHandle, iIndex, outBuffer );
}

UInt WinGUIComboBox::SearchItem( const GChar * strItem, UInt iStartIndex, Bool bExact ) const
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iResult;
	if ( bExact )
		iResult = ComboBox_FindStringExact( hHandle, ((Int)iStartIndex) - 1, strItem );
	else
		iResult = ComboBox_FindString( hHandle, ((Int)iStartIndex) - 1, strItem );

	return (iResult != CB_ERR) ? iResult : INVALID_OFFSET;
}

UInt WinGUIComboBox::AddItem( const GChar * strItem )
{
	HWND hHandle = (HWND)m_hHandle;
	return ComboBox_AddString( hHandle, strItem );
}
Void WinGUIComboBox::AddItem( UInt iIndex, const GChar * strItem )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_InsertString( hHandle, iIndex, strItem );
}
Void WinGUIComboBox::RemoveItem( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_DeleteString( hHandle, iIndex );
}
Void WinGUIComboBox::RemoveAllItems()
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_ResetContent( hHandle );
}

Void * WinGUIComboBox::GetItemData( UInt iIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;
	return (Void*)( ComboBox_GetItemData(hHandle, iIndex) );
}
Void WinGUIComboBox::SetItemData( UInt iIndex, Void * pUserData )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_SetItemData( hHandle, iIndex, pUserData );
}

UInt WinGUIComboBox::GetSelectedItem() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ComboBox_GetCurSel( hHandle );
}
UInt WinGUIComboBox::GetSelectedItemStringLength() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ComboBox_GetTextLength( hHandle );
}
Void WinGUIComboBox::GetSelectedItemString( GChar * outBuffer, UInt iMaxLength ) const
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_GetText( hHandle, outBuffer, iMaxLength );
}

Void WinGUIComboBox::SelectItem( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_SetCurSel( hHandle, iIndex );
}
UInt WinGUIComboBox::SelectItem( const GChar * strItem, UInt iStartIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	UInt iResult = ComboBox_SelectString( hHandle, iStartIndex, strItem );
	return (iResult != CB_ERR) ? iResult : INVALID_OFFSET;
}

Void WinGUIComboBox::SetSelectionText( const GChar * strText )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_SetText( hHandle, strText );
}

Void WinGUIComboBox::GetCueText( GChar * outText, UInt iMaxLength ) const
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_GetCueBannerText( hHandle, outText, iMaxLength );
}
Void WinGUIComboBox::SetCueText( const GChar * strText )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_SetCueBannerText( hHandle, strText );
}

UInt WinGUIComboBox::AddFiles( GChar * strPath, Bool bIncludeSubDirs )
{
	HWND hHandle = (HWND)m_hHandle;
	return ComboBox_Dir( hHandle, bIncludeSubDirs ? DDL_DIRECTORY : 0, strPath );
}
Void WinGUIComboBox::MakeDirectoryList( GChar * strPath, Bool bIncludeSubDirs, WinGUIStatic * pDisplay )
{
	HWND hHandle = (HWND)m_hHandle;

	Int iStaticDisplayID = 0;
	if ( pDisplay != NULL )
		iStaticDisplayID = _GetResourceID( pDisplay );

	DlgDirListComboBox( hHandle, strPath, m_iResourceID, iStaticDisplayID, bIncludeSubDirs ? DDL_DIRECTORY : 0 );
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIComboBox::_Create()
{
	DebugAssert( m_hHandle == NULL );

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUIComboBoxModel * pModel = (WinGUIComboBoxModel*)m_pModel;

	// Compute Layout
    const WinGUILayout * pLayout = pModel->GetLayout();

    WinGUIRectangle hParentRect;
    m_pParent->GetClientRect( &hParentRect );

    WinGUIRectangle hWindowRect;
    pLayout->ComputeLayout( &hWindowRect, hParentRect );

	// Get Creation Parameters
    const WinGUIComboBoxParameters * pParameters = pModel->GetCreationParameters();

    // Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE | CBS_NOINTEGRALHEIGHT );
	switch( pParameters->iType ) {
		case WINGUI_COMBOBOX_BUTTON:
			dwStyle |= CBS_DROPDOWNLIST;
			break;
		case WINGUI_COMBOBOX_EDIT:
			dwStyle |= CBS_DROPDOWN;
			if ( pParameters->bAllowHorizontalScroll )
				dwStyle |= CBS_AUTOHSCROLL;
			break;
		case WINGUI_COMBOBOX_LIST:
			dwStyle |= CBS_SIMPLE;
			if ( pParameters->bAllowHorizontalScroll )
				dwStyle |= CBS_AUTOHSCROLL;
			break;
		default: DebugAssert(false); break;
	}
	switch( pParameters->iCase ) {
		case WINGUI_COMBOBOX_CASE_BOTH:  dwStyle |= 0; break;
		case WINGUI_COMBOBOX_CASE_LOWER: dwStyle |= CBS_LOWERCASE; break;
		case WINGUI_COMBOBOX_CASE_UPPER: dwStyle |= CBS_UPPERCASE; break;
		default: DebugAssert(false); break;
	}
	if ( pParameters->bAutoSort )
		dwStyle |= CBS_SORT;
	if ( pParameters->bEnableTabStop )
		dwStyle |= WS_TABSTOP;

    // Window creation
	m_hHandle = CreateWindowEx (
		0,
		WC_COMBOBOX,
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
		AddItem( pModel->GetItemString(i) );
		SetItemData( i, pModel->GetItemData(i) );
	}
	SelectItem( pParameters->iInitialSelectedItem );

	// Done
	_SaveElementToHandle();
}
Void WinGUIComboBox::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUIComboBox::_DispatchEvent( Int iNotificationCode, Void * pParameters )
{
    // Get Model
	WinGUIComboBoxModel * pModel = (WinGUIComboBoxModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		case CBN_SETFOCUS:  return pModel->OnFocusGained(); break;
		case CBN_KILLFOCUS: return pModel->OnFocusLost(); break;

		case CBN_DBLCLK: return pModel->OnDblClick(); break;

		case CBN_EDITCHANGE:   return pModel->OnTextChange(); break;
		case CBN_SELCHANGE:    return pModel->OnSelectionChange(); break;
		case CBN_SELENDOK:     return pModel->OnSelectionOK(); break;
		case CBN_SELENDCANCEL: return pModel->OnSelectionCancel(); break;
		default: break;
	}

	// Unhandled
	return false;
}


