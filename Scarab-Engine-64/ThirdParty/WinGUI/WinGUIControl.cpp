/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUIControl.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Element : Controls
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
#include "WinGUIControl.h"

/////////////////////////////////////////////////////////////////////////////////
// WinGUIControlModel implementation
WinGUIControlModel::WinGUIControlModel( Int iResourceID ):
	WinGUIElementModel(iResourceID)
{
	// nothing to do
}
WinGUIControlModel::~WinGUIControlModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIControl implementation
WinGUIControl::WinGUIControl( WinGUIElement * pParent, WinGUIControlModel * pModel ):
	WinGUIElement(pParent, pModel)
{
	// nothing to do
}
WinGUIControl::~WinGUIControl()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////

UIntPtr __stdcall WinGUIControl::_SubClassCallback_Static( Void * hHandle, UInt iMessage, UIntPtr wParam, UIntPtr lParam, UIntPtr iSubClassID, UIntPtr iRefData )
{
	DebugAssert( iSubClassID == 0 );
	WinGUIControl * pThis = (WinGUIControl*)iRefData;
    return pThis->_SubClassCallback_Virtual( hHandle, iMessage, wParam, lParam );
}
UIntPtr __stdcall WinGUIControl::_SubClassCallback_Virtual( Void * hHandle, UInt iMessage, UIntPtr wParam, UIntPtr lParam )
{
	// Get Model
    WinGUIControlModel * pModel = (WinGUIControlModel*)m_pModel;

	// Dispatch Message
	switch( iMessage ) {
		case WM_LBUTTONDOWN: {
				WinGUIPoint hPoint;
                hPoint.iX = (UInt)( GET_X_LPARAM(lParam) );
                hPoint.iY = (UInt)( GET_Y_LPARAM(lParam) );
				pModel->OnMousePress( hPoint, KEYCODE_MOUSELEFT );
			} break;
		case WM_RBUTTONDOWN: {
				WinGUIPoint hPoint;
                hPoint.iX = (UInt)( GET_X_LPARAM(lParam) );
                hPoint.iY = (UInt)( GET_Y_LPARAM(lParam) );
				pModel->OnMousePress( hPoint, KEYCODE_MOUSERIGHT );
			} break;
		case WM_LBUTTONUP: {
				WinGUIPoint hPoint;
                hPoint.iX = (UInt)( GET_X_LPARAM(lParam) );
                hPoint.iY = (UInt)( GET_Y_LPARAM(lParam) );
				pModel->OnMouseRelease( hPoint, KEYCODE_MOUSELEFT );
			} break;
		case WM_RBUTTONUP: {
				WinGUIPoint hPoint;
                hPoint.iX = (UInt)( GET_X_LPARAM(lParam) );
                hPoint.iY = (UInt)( GET_Y_LPARAM(lParam) );
				pModel->OnMouseRelease( hPoint, KEYCODE_MOUSERIGHT );
			} break;
		default: break;
	}

	// Done
    return DefSubclassProc( (HWND)hHandle, iMessage, wParam, lParam );
}

Void WinGUIControl::_RegisterSubClass()
{
	SetWindowSubclass( (HWND)m_hHandle, (SUBCLASSPROC)_SubClassCallback_Static, 0, (DWORD_PTR)this );
}
Void WinGUIControl::_UnregisterSubClass()
{
	RemoveWindowSubclass( (HWND)m_hHandle, (SUBCLASSPROC)_SubClassCallback_Static, 0 );
}
