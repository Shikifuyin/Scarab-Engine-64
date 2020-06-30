/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUIElement.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Element Base Interface
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

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "WinGUIElement.h"

/////////////////////////////////////////////////////////////////////////////////
// WinGUIElementModel implementation
WinGUIElementModel::WinGUIElementModel( Int iResourceID )
{
	m_iResourceID = iResourceID;

	m_pController = NULL;
}
WinGUIElementModel::~WinGUIElementModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIElement implementation
WinGUIElement::WinGUIElement( WinGUIElement * pParent, WinGUIElementModel * pModel )
{
	// Link with Model
	m_pModel = pModel;
	if ( m_pModel != NULL )
		m_pModel->m_pController = this;

	// Parent Link
	m_pParent = pParent;

	// Start uninitialized
	m_hHandle = NULL;

	// Pick Resource Identifier
	m_iResourceID = INVALID_OFFSET;
	if ( m_pModel != NULL )
		m_iResourceID = m_pModel->m_iResourceID;
}
WinGUIElement::~WinGUIElement()
{
	DebugAssert( m_hHandle == NULL );

	// Unlink with Model
	m_pModel->m_pController = NULL;
}

Bool WinGUIElement::IsVisible() const
{
    return ( IsWindowVisible((HWND)m_hHandle) != FALSE );
}
Void WinGUIElement::SetVisible( Bool bVisible )
{
    if ( bVisible ) {
		ShowWindow( (HWND)m_hHandle, SW_SHOW );
        UpdateWindow( (HWND)m_hHandle );
	} else {
		ShowWindow( (HWND)m_hHandle, SW_HIDE );
	}
}

Void WinGUIElement::GiveFocus()
{
	SetFocus( (HWND)m_hHandle );
}

Void WinGUIElement::GetWindowRect( WinGUIRectangle * outRectangle ) const
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	::GetWindowRect( hHandle, &hRect );

	outRectangle->iLeft = hRect.left;
	outRectangle->iTop = hRect.top;
	outRectangle->iWidth = ( hRect.right - hRect.left );
	outRectangle->iHeight = ( hRect.bottom - hRect.top );
}
Void WinGUIElement::GetClientRect( WinGUIRectangle * outRectangle ) const
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	::GetClientRect( hHandle, &hRect );

	outRectangle->iLeft = hRect.left;
	outRectangle->iTop = hRect.top;
	outRectangle->iWidth = ( hRect.right - hRect.left );
	outRectangle->iHeight = ( hRect.bottom - hRect.top );
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIElement::_ApplyDefaultFont( Void * hFont )
{
	DebugAssert( hFont != NULL );
	SendMessage( (HWND)m_hHandle, WM_SETFONT, (WPARAM)hFont, (LPARAM)TRUE );
}

Void WinGUIElement::_SaveElementToHandle() const
{
	DebugAssert( m_hHandle != NULL );
	SetWindowLongPtr( (HWND)m_hHandle, GWLP_USERDATA, (LONG_PTR)this );
}
WinGUIElement * WinGUIElement::_GetElementFromHandle( Void * hHandle )
{
	WinGUIElement * pElement = (WinGUIElement*)( GetWindowLongPtr((HWND)hHandle, GWLP_USERDATA) );
	if ( pElement == NULL )
		return NULL;
	DebugAssert( pElement->m_hHandle == hHandle );
	return pElement;
}

