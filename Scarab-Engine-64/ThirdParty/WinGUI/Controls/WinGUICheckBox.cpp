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
	// Default Parameters
	StringFn->Copy( m_hCreationParameters.strLabel, TEXT("CheckBox") );

	m_hCreationParameters.bEnableTabStop = true;
	m_hCreationParameters.bEnableNotify = false;
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

Void WinGUICheckBox::GetIdealSize( WinGUIPoint * outSize ) const
{
	HWND hHandle = (HWND)m_hHandle;

	SIZE hSize;
	Button_GetIdealSize( hHandle, &hSize );

	outSize->iX = hSize.cx;
	outSize->iY = hSize.cy;
}

Void WinGUICheckBox::GetTextMargin( WinGUIRectangle * outRectMargin ) const
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	Button_GetTextMargin( hHandle, &hRect );

	outRectMargin->iLeft = hRect.left;
	outRectMargin->iTop = hRect.top;
	outRectMargin->iWidth = ( hRect.right - hRect.left );
	outRectMargin->iHeight = ( hRect.bottom - hRect.top );
}
Void WinGUICheckBox::SetTextMargin( const WinGUIRectangle & hRectMargin )
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	hRect.left = hRectMargin.iLeft;
	hRect.top = hRectMargin.iTop;
	hRect.right = ( hRectMargin.iLeft + hRectMargin.iWidth );
	hRect.bottom = ( hRectMargin.iTop + hRectMargin.iHeight );

	Button_SetTextMargin( hHandle, &hRect );
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

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUICheckBoxModel * pModel = (WinGUICheckBoxModel*)m_pModel;

	// Compute Layout
    const WinGUILayout * pLayout = pModel->GetLayout();

    WinGUIRectangle hParentRect;
    m_pParent->GetClientRect( &hParentRect );

    WinGUIRectangle hWindowRect;
    pLayout->ComputeLayout( &hWindowRect, hParentRect );

	// Get Creation Parameters
    const WinGUICheckBoxParameters * pParameters = pModel->GetCreationParameters();

	// Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX );
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
	_RegisterSubClass();
}
Void WinGUICheckBox::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

	// Remove SubClass
	_UnregisterSubClass();

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUICheckBox::_DispatchEvent( Int iNotificationCode, Void * pParameters )
{
    // Get Model
	WinGUICheckBoxModel * pModel = (WinGUICheckBoxModel*)m_pModel;

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

