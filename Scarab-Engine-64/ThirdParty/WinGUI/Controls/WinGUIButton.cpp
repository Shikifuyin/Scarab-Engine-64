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
	// Default Parameters
	StringFn->Copy( m_hCreationParameters.strLabel, TEXT("Button") );

	m_hCreationParameters.bCenterLabel = true;
	m_hCreationParameters.bEnableTabStop = true;
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

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUIButtonModel * pModel = (WinGUIButtonModel*)m_pModel;

	// Compute Layout
    const WinGUILayout * pLayout = pModel->GetLayout();

    WinGUIRectangle hParentRect;
    m_pParent->GetClientRect( &hParentRect );

    WinGUIRectangle hWindowRect;
    pLayout->ComputeLayout( &hWindowRect, hParentRect );

	// Get Creation Parameters
    const WinGUIButtonParameters * pParameters = pModel->GetCreationParameters();

    // Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON );
	if ( pParameters->bCenterLabel )
		dwStyle |= ( BS_CENTER | BS_VCENTER );
	if ( pParameters->bEnableTabStop )
		dwStyle |= WS_TABSTOP;

    // Window creation
	m_hHandle = CreateWindowEx (
		0,
		WC_BUTTON,
		pParameters->strLabel,
		dwStyle,
		hWindowRect.iLeft, hWindowRect.iTop,
        hWindowRect.iWidth, hWindowRect.iHeight,
		hParentWnd,
		(HMENU)m_iResourceID,
		(HINSTANCE)( GetWindowLongPtr(hParentWnd,GWLP_HINSTANCE) ),
		NULL
	);
	DebugAssert( m_hHandle != NULL );

	// Done
	_SaveElementToHandle();
}
Void WinGUIButton::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUIButton::_DispatchEvent( Int iNotificationCode )
{
    // Get Model
	WinGUIButtonModel * pModel = (WinGUIButtonModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		case BN_CLICKED: return pModel->OnClick(); break;
		case BN_DBLCLK:  return pModel->OnDblClick(); break;
		default: break;
	}

	// Unhandled
	return false;
}

