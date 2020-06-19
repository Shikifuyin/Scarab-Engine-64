/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIRadioButton.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : Radio Button
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
#include "WinGUIRadioButton.h"

/////////////////////////////////////////////////////////////////////////////////
// WinGUIRadioButtonModel implementation
WinGUIRadioButtonModel::WinGUIRadioButtonModel():
	WinGUIControlModel()
{
}
WinGUIRadioButtonModel::~WinGUIRadioButtonModel()
{
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIRadioButton implementation
WinGUIRadioButton::WinGUIRadioButton( WinGUIRadioButtonModel * pModel ):
	WinGUIControl(pModel)
{
}
WinGUIRadioButton::~WinGUIRadioButton()
{
}

Void WinGUIRadioButton::Initialize()
{
	// Get Initialization values from Model
	m_iButtonID = m_pModel->GetIdentifier();
	const GChar * strText = m_pModel->GetText();
	UInt iPositionX = m_pModel->GetPositionX();
	UInt iPositionY = m_pModel->GetPositionY();
	UInt iWidth = m_pModel->GetWidth();
	UInt iHeight = m_pModel->GetHeight();

	// Create the Control
	m_hButtonWnd = CreateWindowEx(
		0, TEXT("BUTTON"), strText,
		WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_RADIOBUTTON, 
		iPositionX, iPositionY, iWidth, iHeight,
		m_hParentWnd, (HMENU)m_iButtonID, m_hAppInstance, this
	);
	DebugAssert( m_hButtonWnd != NULL );

	if ( m_pModel->IsChecked() )
		CheckDlgButton( (HWND)m_hButtonWnd, m_iButtonID, BST_CHECKED );
}

Void WinGUIRadioButton::Enable()
{
	Button_Enable( (HWND)m_hButtonWnd, TRUE );
}
Void WinGUIRadioButton::Disable()
{
	Button_Enable( (HWND)m_hButtonWnd, FALSE );
}

UInt WinGUIRadioButton::GetTextLength() const
{
	return Button_GetTextLength( (HWND)m_hButtonWnd );
}
Void WinGUIRadioButton::GetText( GChar * outText, UInt iMaxLength ) const
{
	Button_GetText( (HWND)m_hButtonWnd, outText, iMaxLength );
}
Void WinGUIRadioButton::SetText( const GChar * strText )
{
	Button_SetText( (HWND)m_hButtonWnd, strText );
}

Bool WinGUIRadioButton::IsChecked() const
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
Void WinGUIRadioButton::Check()
{
	UInt iCount = m_pRadioButtonGroup->GetButtonCount();
	for ( UInt i = 0; i < iCount; ++i ) {
		WinGUIRadioButton * pButton = m_pRadioButtonGroup->GetButton(i);
		if ( pButton == this )
			CheckDlgButton( (HWND)m_hButtonWnd, m_iButtonID, BST_CHECKED );
		else
			CheckDlgButton( (HWND)(pButton->m_hButtonWnd), pButton->m_iButtonID, BST_UNCHECKED );
	}
}

/////////////////////////////////////////////////////////////////////////////////

UIntPtr __stdcall WinGUIRadioButton::_MessageCallback_Virtual( Void * hWnd, UInt message, UIntPtr wParam, UIntPtr lParam )
{
	// Translate Button Messages
	switch( message ) {
		case WM_COMMAND: {

		} break;
		default: break;
	}
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIRadioButtonGroup implementation
WinGUIRadioButtonGroup::WinGUIRadioButtonGroup()
{
	m_iButtonCount = 0;

	for( UInt i = 0; i < WINGUI_RADIO_BUTTON_MAX_GROUP_SIZE; ++i )
		m_arrRadioButtons[i] = NULL;
}
WinGUIRadioButtonGroup::~WinGUIRadioButtonGroup()
{
	for( UInt i = 0; i < m_iButtonCount; ++i )
		m_arrRadioButtons[i]->SetGroup( NULL );
}

Void WinGUIRadioButtonGroup::AddButton( WinGUIRadioButton * pButton )
{
	DebugAssert( m_iButtonCount < WINGUI_RADIO_BUTTON_MAX_GROUP_SIZE );
	m_arrRadioButtons[m_iButtonCount] = pButton;
	++m_iButtonCount;
}

