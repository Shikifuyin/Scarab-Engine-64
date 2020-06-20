/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUICheckBox.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : CheckBox
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
#include "WinGUICheckBox.h"

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUICheckBoxModel implementation
WinGUICheckBoxModel::WinGUICheckBoxModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// nothing to do
}
WinGUICheckBoxModel::~WinGUICheckBoxModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUICheckBox implementation
WinGUICheckBox::WinGUICheckBox( WinGUIElement * pParent, WinGUICheckBoxModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	// nothing to do
}
WinGUICheckBox::~WinGUICheckBox()
{
	// nothing to do
}

Void WinGUICheckBox::Enable()
{
	Button_Enable( (HWND)m_hHandle, TRUE );
}
Void WinGUICheckBox::Disable()
{
	Button_Enable( (HWND)m_hHandle, FALSE );
}

UInt WinGUICheckBox::GetTextLength() const
{
	return Button_GetTextLength( (HWND)m_hHandle );
}
Void WinGUICheckBox::GetText( GChar * outText, UInt iMaxLength ) const
{
	Button_GetText( (HWND)m_hHandle, outText, iMaxLength );
}
Void WinGUICheckBox::SetText( const GChar * strText )
{
	Button_SetText( (HWND)m_hHandle, strText );
}

Bool WinGUICheckBox::IsChecked() const
{
	UInt iState = Button_GetCheck( (HWND)m_hHandle );
	if ( iState == BST_CHECKED )
		return true;
	if ( iState == BST_UNCHECKED )
		return false;

	// Undetermined, Should never happen
	DebugAssert( false );
	return false;
}
Void WinGUICheckBox::Check()
{
	Button_SetCheck( (HWND)m_hHandle, BST_CHECKED );
}
Void WinGUICheckBox::Uncheck()
{
	Button_SetCheck( (HWND)m_hHandle, BST_UNCHECKED );
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUICheckBox::_Create()
{
	DebugAssert( m_hHandle == NULL );

	WinGUICheckBoxModel * pModel = (WinGUICheckBoxModel*)m_pModel;
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

	m_hHandle = CreateWindowEx (
		0, WC_BUTTON, pModel->GetText(),
		WS_VISIBLE | WS_CHILD | WS_TABSTOP | BS_AUTOCHECKBOX,
		pModel->GetPositionX(),	pModel->GetPositionY(),
		pModel->GetWidth(), pModel->GetHeight(),
		hParentWnd, (HMENU)m_iResourceID,
		(HINSTANCE)( GetWindowLongPtr(hParentWnd,GWLP_HINSTANCE) ),
		NULL
	);
	DebugAssert( m_hHandle != NULL );

	// Done
	_SaveElementToHandle();
}
Void WinGUICheckBox::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUICheckBox::_DispatchEvent( Int iNotificationCode )
{
	WinGUICheckBoxModel * pModel = (WinGUICheckBoxModel*)m_pModel;

	// Dispatch Event to our Model
	switch( iNotificationCode ) {
		case BN_CLICKED:
			return pModel->OnClick();
			break;
		case BN_DBLCLK:
			return pModel->OnDblClick();
			break;
		default: break;
	}

	// Unhandled
	return false;
}

