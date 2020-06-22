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
	// nothing to do
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
UInt WinGUIComboBox::GetListItemHeight() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ComboBox_GetItemHeight( hHandle );
}
Void WinGUIComboBox::SetSelectionHeight( UInt iHeight )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_SetItemHeight( hHandle, -1, iHeight );
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

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIComboBox::_Create()
{
	DebugAssert( m_hHandle == NULL );

	WinGUIComboBoxModel * pModel = (WinGUIComboBoxModel*)m_pModel;
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    const WinGUIRectangle * pRect = pModel->GetRectangle();

	DWord dwStyle = CBS_NOINTEGRALHEIGHT;
	switch( pModel->GetType() ) {
		case WINGUI_COMBOBOX_BUTTON:
			dwStyle = CBS_DROPDOWNLIST;
			break;
		case WINGUI_COMBOBOX_EDIT:
			dwStyle = CBS_DROPDOWN;
			if ( pModel->AllowHorizScroll() )
				dwStyle |= CBS_AUTOHSCROLL;
			break;
		case WINGUI_COMBOBOX_LIST:
			dwStyle = CBS_SIMPLE;
			if ( pModel->AllowHorizScroll() )
				dwStyle |= CBS_AUTOHSCROLL;
			break;
		default: DebugAssert(false); break;
	}
	switch( pModel->GetTextCase() ) {
		case WINGUI_COMBOBOX_CASE_BOTH:  dwStyle |= 0; break;
		case WINGUI_COMBOBOX_CASE_LOWER: dwStyle |= CBS_LOWERCASE; break;
		case WINGUI_COMBOBOX_CASE_UPPER: dwStyle |= CBS_UPPERCASE; break;
		default: DebugAssert(false); break;
	}
	if ( pModel->AutoSort() ) {
		dwStyle |= CBS_SORT;
	}

	m_hHandle = CreateWindowEx (
		0, WC_COMBOBOX, TEXT(""),
		WS_VISIBLE | WS_CHILD | WS_TABSTOP | dwStyle,
		pRect->iLeft, pRect->iTop,
        pRect->iWidth, pRect->iHeight,
		hParentWnd, (HMENU)m_iResourceID,
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
	SelectItem( pModel->GetInitialSelectedItem() );

	// Done
	_SaveElementToHandle();
}
Void WinGUIComboBox::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUIComboBox::_DispatchEvent( Int iNotificationCode )
{
	WinGUIComboBoxModel * pModel = (WinGUIComboBoxModel*)m_pModel;

	// Dispatch Event to our Model
	switch( iNotificationCode ) {
		case CBN_DBLCLK:
			return pModel->OnDblClick();
			break;
		case CBN_EDITCHANGE:
			return pModel->OnTextChange();
			break;
		case CBN_SELCHANGE:
			return pModel->OnSelectionChange();
			break;
		case CBN_SELENDOK:
			return pModel->OnSelectionOK();
			break;
		case CBN_SELENDCANCEL:
			return pModel->OnSelectionCancel();
			break;
		default: break;
	}

	// Unhandled
	return false;
}


