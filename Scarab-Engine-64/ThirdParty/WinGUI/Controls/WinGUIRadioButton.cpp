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

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUIRadioButtonModel implementation
WinGUIRadioButtonModel::WinGUIRadioButtonModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// nothing to do
}
WinGUIRadioButtonModel::~WinGUIRadioButtonModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIRadioButton implementation
WinGUIRadioButton::WinGUIRadioButton( WinGUIElement * pParent, WinGUIRadioButtonModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	// nothing to do
}
WinGUIRadioButton::~WinGUIRadioButton()
{
	// nothing to do
}

Void WinGUIRadioButton::Enable()
{
	Button_Enable( (HWND)m_hHandle, TRUE );
}
Void WinGUIRadioButton::Disable()
{
	Button_Enable( (HWND)m_hHandle, FALSE );
}

UInt WinGUIRadioButton::GetTextLength() const
{
	return Button_GetTextLength( (HWND)m_hHandle );
}
Void WinGUIRadioButton::GetText( GChar * outText, UInt iMaxLength ) const
{
	Button_GetText( (HWND)m_hHandle, outText, iMaxLength );
}
Void WinGUIRadioButton::SetText( const GChar * strText )
{
	Button_SetText( (HWND)m_hHandle, strText );
}

Bool WinGUIRadioButton::IsChecked() const
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
Void WinGUIRadioButton::Check()
{
	UInt iCount = m_pRadioButtonGroup->GetButtonCount();
	for ( UInt i = 0; i < iCount; ++i ) {
		WinGUIRadioButton * pButton = m_pRadioButtonGroup->GetButton(i);
		if ( pButton == this )
			Button_SetCheck( (HWND)(_GetHandle(pButton)), BST_CHECKED );
		else
			Button_SetCheck( (HWND)(_GetHandle(pButton)), BST_UNCHECKED );
	}
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIRadioButton::_Create()
{
	DebugAssert( m_hHandle == NULL );

	WinGUIRadioButtonModel * pModel = (WinGUIRadioButtonModel*)m_pModel;
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    const WinGUIRectangle * pRect = pModel->GetRectangle();

	m_hHandle = CreateWindowEx (
		0, WC_BUTTON, pModel->GetText(),
		WS_VISIBLE | WS_CHILD | WS_TABSTOP | BS_AUTORADIOBUTTON,
		pRect->iLeft, pRect->iTop,
        pRect->iWidth, pRect->iHeight,
		hParentWnd, (HMENU)m_iResourceID,
		(HINSTANCE)( GetWindowLongPtr(hParentWnd,GWLP_HINSTANCE) ),
		NULL
	);
	DebugAssert( m_hHandle != NULL );

	// Done
	_SaveElementToHandle();
}
Void WinGUIRadioButton::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUIRadioButton::_DispatchEvent( Int iNotificationCode )
{
	WinGUIRadioButtonModel * pModel = (WinGUIRadioButtonModel*)m_pModel;

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

