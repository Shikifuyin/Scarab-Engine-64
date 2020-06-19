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

/////////////////////////////////////////////////////////////////////////////////
// WinGUICheckBoxModel implementation
WinGUICheckBoxModel::WinGUICheckBoxModel():
	WinGUIControlModel()
{
}
WinGUICheckBoxModel::~WinGUICheckBoxModel()
{
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUICheckBox implementation
WinGUICheckBox::WinGUICheckBox( WinGUICheckBoxModel * pModel ):
	WinGUIControl(pModel)
{
}
WinGUICheckBox::~WinGUICheckBox()
{
}

Void WinGUICheckBox::Enable()
{
	Button_Enable( (HWND)m_hButtonWnd, TRUE );
}
Void WinGUICheckBox::Disable()
{
	Button_Enable( (HWND)m_hButtonWnd, FALSE );
}

UInt WinGUICheckBox::GetTextLength() const
{
	return Button_GetTextLength( (HWND)m_hButtonWnd );
}
Void WinGUICheckBox::GetText( GChar * outText, UInt iMaxLength ) const
{
	Button_GetText( (HWND)m_hButtonWnd, outText, iMaxLength );
}
Void WinGUICheckBox::SetText( const GChar * strText )
{
	Button_SetText( (HWND)m_hButtonWnd, strText );
}

Bool WinGUICheckBox::IsChecked() const
{
	UInt iState = IsDlgButtonChecked( (HWND)m_hButtonWnd, m_iButtonID );
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
	CheckDlgButton( (HWND)m_hButtonWnd, m_iButtonID, BST_CHECKED );
}
Void WinGUICheckBox::Uncheck()
{
	CheckDlgButton( (HWND)m_hButtonWnd, m_iButtonID, BST_UNCHECKED );
}

/////////////////////////////////////////////////////////////////////////////////

UIntPtr __stdcall WinGUICheckBox::_MessageCallback_Virtual( Void * hWnd, UInt message, UIntPtr wParam, UIntPtr lParam )
{
	// Translate Button Messages
	switch( message ) {
		case WM_COMMAND: {

		} break;
		default: break;
	}
}

