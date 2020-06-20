/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIButton.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : Button
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
#include "WinGUIButton.h"

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUIButtonModel implementation
WinGUIButtonModel::WinGUIButtonModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// nothing to do
}
WinGUIButtonModel::~WinGUIButtonModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIButton implementation
WinGUIButton::WinGUIButton( WinGUIElement * pParent, WinGUIButtonModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	// nothing to do
}
WinGUIButton::~WinGUIButton()
{
	// nothing to do
}

Void WinGUIButton::Enable()
{
	HWND hHandle = (HWND)m_hHandle;
	Button_Enable( hHandle, TRUE );
}
Void WinGUIButton::Disable()
{
	HWND hHandle = (HWND)m_hHandle;
	Button_Enable( hHandle, FALSE );
}

UInt WinGUIButton::GetTextLength() const
{
	HWND hHandle = (HWND)m_hHandle;
	return Button_GetTextLength( hHandle );
}
Void WinGUIButton::GetText( GChar * outText, UInt iMaxLength ) const
{
	HWND hHandle = (HWND)m_hHandle;
	Button_GetText( hHandle, outText, iMaxLength );
}
Void WinGUIButton::SetText( const GChar * strText )
{
	HWND hHandle = (HWND)m_hHandle;
	Button_SetText( hHandle, strText );
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIButton::_Create()
{
	DebugAssert( m_hHandle == NULL );

	WinGUIButtonModel * pModel = (WinGUIButtonModel*)m_pModel;
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

	m_hHandle = CreateWindowEx (
		0, L"BUTTON", pModel->GetText(),
		WS_VISIBLE | WS_CHILD | WS_TABSTOP | BS_PUSHBUTTON,
		pModel->GetPositionX(),	pModel->GetPositionY(),
		pModel->GetWidth(), pModel->GetHeight(),
		hParentWnd, (HMENU)m_iResourceID,
		(HINSTANCE)( GetWindowLongPtr(hParentWnd,GWLP_HINSTANCE) ),
		this
	);
	DebugAssert( m_hHandle != NULL );

	// Done
	_SaveElementToHandle();
}
Void WinGUIButton::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUIButton::_DispatchEvent( Int iNotificationCode )
{
	WinGUIButtonModel * pModel = (WinGUIButtonModel*)m_pModel;

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

