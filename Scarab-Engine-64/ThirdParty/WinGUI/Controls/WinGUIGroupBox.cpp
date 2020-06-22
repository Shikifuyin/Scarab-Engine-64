/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIGroupBox.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : GroupBox
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
#include "WinGUIGroupBox.h"

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUIGroupBoxModel implementation
WinGUIGroupBoxModel::WinGUIGroupBoxModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// Default Parameters
	StringFn->Copy( m_hCreationParameters.strLabel, TEXT("GroupBox") );
}
WinGUIGroupBoxModel::~WinGUIGroupBoxModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIGroupBox implementation
WinGUIGroupBox::WinGUIGroupBox( WinGUIElement * pParent, WinGUIGroupBoxModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	// nothing to do
}
WinGUIGroupBox::~WinGUIGroupBox()
{
	// nothing to do
}

UInt WinGUIGroupBox::GetTextLength() const
{
	HWND hHandle = (HWND)m_hHandle;
	return Button_GetTextLength( hHandle );
}
Void WinGUIGroupBox::GetText( GChar * outText, UInt iMaxLength ) const
{
	HWND hHandle = (HWND)m_hHandle;
	Button_GetText( hHandle, outText, iMaxLength );
}
Void WinGUIGroupBox::SetText( const GChar * strText )
{
	HWND hHandle = (HWND)m_hHandle;
	Button_SetText( hHandle, strText );
}

Void WinGUIGroupBox::ComputeClientArea( WinGUIRectangle * outClientArea, Int iPadding ) const
{
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );
	HWND hGroupWnd = (HWND)m_hHandle;

	RECT hRect;
    ::GetWindowRect(hGroupWnd, &hRect);
    MapWindowPoints( NULL, hParentWnd, (POINT*)&hRect, 2 );

	RECT hBorder = { 4, 8, 4, 4 };
	OffsetRect( &hBorder, iPadding, iPadding );
	MapDialogRect( hParentWnd, &hBorder );

	outClientArea->iLeft = ( hRect.left + hBorder.left );
	outClientArea->iTop = ( hRect.top + hBorder.top );
	outClientArea->iWidth = ( hRect.right - hBorder.right ) - outClientArea->iLeft;
	outClientArea->iHeight = ( hRect.bottom - hBorder.bottom ) - outClientArea->iTop;
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIGroupBox::_Create()
{
	DebugAssert( m_hHandle == NULL );

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUIGroupBoxModel * pModel = (WinGUIGroupBoxModel*)m_pModel;

	// Compute Layout
    const WinGUILayout * pLayout = pModel->GetLayout();

    WinGUIRectangle hParentRect;
    m_pParent->GetClientRect( &hParentRect );

    WinGUIRectangle hWindowRect;
    pLayout->ComputeLayout( &hWindowRect, hParentRect );

	// Get Creation Parameters
    const WinGUIGroupBoxParameters * pParameters = pModel->GetCreationParameters();

	// Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE | BS_GROUPBOX );

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
Void WinGUIGroupBox::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUIGroupBox::_DispatchEvent( Int iNotificationCode )
{
	// nothing to do
	return false;
}

