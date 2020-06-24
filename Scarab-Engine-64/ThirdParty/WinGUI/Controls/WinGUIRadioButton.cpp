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
	// Default Parameters
	StringFn->Copy( m_hCreationParameters.strLabel, TEXT("RadioButton") );

	m_hCreationParameters.bEnableTabStop = true;
	m_hCreationParameters.bEnableNotify = false;

	// Radio Button Group
	m_pRadioButtonGroup = NULL;
}
WinGUIRadioButtonModel::~WinGUIRadioButtonModel()
{
	// nothing to do
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

Void WinGUIRadioButtonGroup::AddButton( WinGUIRadioButtonModel * pButton )
{
	DebugAssert( m_iButtonCount < WINGUI_RADIO_BUTTON_MAX_GROUP_SIZE );
	m_arrRadioButtons[m_iButtonCount] = pButton;
	++m_iButtonCount;
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
	WinGUIRadioButtonModel * pModel = (WinGUIRadioButtonModel*)m_pModel;
	WinGUIRadioButtonGroup * pGroup = pModel->GetGroup();

	UInt iCount = pGroup->GetButtonCount();
	for ( UInt i = 0; i < iCount; ++i ) {
		WinGUIRadioButton * pButton = (WinGUIRadioButton*)( pGroup->GetButton(i)->GetController() );
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

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUIRadioButtonModel * pModel = (WinGUIRadioButtonModel*)m_pModel;

	// Compute Layout
    const WinGUILayout * pLayout = pModel->GetLayout();

    WinGUIRectangle hParentRect;
    m_pParent->GetClientRect( &hParentRect );

    WinGUIRectangle hWindowRect;
    pLayout->ComputeLayout( &hWindowRect, hParentRect );

	// Get Creation Parameters
    const WinGUIRadioButtonParameters * pParameters = pModel->GetCreationParameters();

	// Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON );
	if ( pParameters->bEnableTabStop )
		dwStyle |= WS_TABSTOP;
	if ( pParameters->bEnableNotify )
		dwStyle |= BS_NOTIFY;

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
Void WinGUIRadioButton::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUIRadioButton::_DispatchEvent( Int iNotificationCode, Void * pParameters )
{
    // Get Model
	WinGUIRadioButtonModel * pModel = (WinGUIRadioButtonModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		case BN_SETFOCUS:  return pModel->OnFocusGained(); break;
		case BN_KILLFOCUS: return pModel->OnFocusLost(); break;

		case BCN_HOTITEMCHANGE: {
			NMBCHOTITEM * pParams = (NMBCHOTITEM *)pParameters;
			if ( pParams->dwFlags & HICF_ENTERING )
				return pModel->OnMouseHovering();
			else if ( pParams->dwFlags & HICF_LEAVING )
				return pModel->OnMouseLeaving();
		} break;

		case BN_CLICKED: return pModel->OnClick(); break;
		case BN_DBLCLK:  return pModel->OnDblClick(); break;
		default: break;
	}

	// Unhandled
	return false;
}



