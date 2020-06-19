/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUIContainer.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Container Base Interface
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
#include <commctrl.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "WinGUIContainer.h"
#include "WinGUIControl.h"

/////////////////////////////////////////////////////////////////////////////////
// WinGUIContainerModel implementation
WinGUIContainerModel::WinGUIContainerModel( Int iResourceID ):
	WinGUIElementModel(iResourceID)
{
}
WinGUIContainerModel::~WinGUIContainerModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIContainer implementation
WinGUIContainer::WinGUIContainer( WinGUIContainerModel * pModel ):
	WinGUIElement(pModel)
{
	m_pParent = NULL;

	m_iChildCount = 0;
	for( UInt i = 0; i < WINGUI_CONTAINER_MAX_CHILDREN; ++i )
		m_arrChildren[i] = NULL;
}
WinGUIContainer::~WinGUIContainer()
{
	// nothing to do
}

WinGUIElement * WinGUIContainer::GetChildByID( Int iResourceID ) const
{
	for ( UInt i = 0; i < m_iChildCount; ++i ) {
		WinGUIElement * pChild = m_arrChildren[i];
		if ( pChild->m_iResourceID == iResourceID )
			return pChild;
	}
	return NULL;
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIContainer::_Create()
{
}
Void WinGUIContainer::_Destroy()
{
}

WinGUIControl * WinGUIContainer::_SearchControl( Int iResourceID ) const
{
    // Lookup Children
    for ( UInt i = 0; i < m_iChildCount; ++i ) {
        WinGUIElement * pChild = m_arrChildren[i];

        // Check for Control case
        if ( pChild->GetElementType() == WINGUI_ELEMENT_CONTROL ) {
            WinGUIControl * pControl = (WinGUIControl*)pChild;
            if ( pControl->m_iResourceID == iResourceID )
                return pControl;
        }

        // Recurse on Containers
        if ( pChild->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
            WinGUIContainer * pContainer = (WinGUIContainer*)pChild;
            WinGUIControl * pControl = pContainer->_SearchControl( iResourceID );
            if ( pControl != NULL )
                return pControl;
        }
    }

    // Not Found
    return NULL;
}

UIntPtr __stdcall WinGUIContainer::_MessageCallback_Static( Void * hHandle, UInt iMessage, UIntPtr wParam, UIntPtr lParam )
{
	static WinGUIContainer * s_pThis = NULL;
    if ( s_pThis != NULL )
        return s_pThis->_MessageCallback_Virtual( hHandle, iMessage, wParam, lParam );

    if ( iMessage == WM_CREATE )
        s_pThis = (WinGUIContainer*)( ((LPCREATESTRUCT)lParam)->lpCreateParams );

    return DefWindowProc( (HWND)hHandle, iMessage, wParam, lParam );
}
UIntPtr __stdcall WinGUIContainer::_MessageCallback_Virtual( Void * hHandle, UInt iMessage, UIntPtr wParam, UIntPtr lParam )
{
    WinGUIContainerModel * pModel = (WinGUIContainerModel*)m_pModel;

    switch( iMessage ) {
        // Keyboard messages
        //case WM_SYSKEYDOWN:
        //case WM_KEYDOWN: {
        //        // Handle key press
        //        KeyCode iKey = KeyCodeFromWin32[wParam & 0xff];
        //        if ( m_pModel->OnKeyPress(iKey) )
        //            return 0;
        //    } break;
        //case WM_SYSKEYUP:
        //case WM_KEYUP: {
        //        // Handle key release
        //        KeyCode iKey = KeyCodeFromWin32[wParam & 0xff];
        //        if ( m_pModel->OnKeyRelease(iKey) )
        //            return 0;
        //    } break;

        // Mouse messages
        //case WM_MOUSEMOVE: {
        //        UInt iX = (UInt)(LOWORD(lParam));
        //        UInt iY = (UInt)(HIWORD(lParam));
        //        m_pModel->OnMouseMove( iX, iY );
        //        return 0;
        //    } break;
        //case WM_MOUSEWHEEL: {
        //        POINT pt; // WM_MOUSEWHEEL needs screen->client conversion
        //        pt.x = (UInt)(LOWORD(lParam));
        //        pt.y = (UInt)(HIWORD(lParam));
        //        ScreenToClient( (HWND)hWnd, &pt );
        //        Int iWheelDelta = (Int)( (Short)(HIWORD(wParam)) ) / WHEEL_DELTA;
        //        if ( m_pModel->OnMouseWheel((UInt)pt.x, (UInt)pt.y, iWheelDelta) )
        //            return 0;
        //    } break;
        //case WM_LBUTTONDOWN: {
        //        UInt iX = (UInt)(LOWORD(lParam));
        //        UInt iY = (UInt)(HIWORD(lParam));
        //        if ( m_pModel->OnMousePress(iX, iY, KEYCODE_MOUSELEFT) )
        //            return 0;
        //    } break;
        //case WM_RBUTTONDOWN: {
        //        UInt iX = (UInt)(LOWORD(lParam));
        //        UInt iY = (UInt)(HIWORD(lParam));
        //        if ( m_pModel->OnMousePress(iX, iY, KEYCODE_MOUSERIGHT) )
        //            return 0;
        //    } break;
        //case WM_MBUTTONDOWN: {
        //        UInt iX = (UInt)(LOWORD(lParam));
        //        UInt iY = (UInt)(HIWORD(lParam));
        //        if ( m_pModel->OnMousePress(iX, iY, KEYCODE_MOUSEMIDDLE) )
        //            return 0;
        //    } break;
        //case WM_XBUTTONDOWN: {
        //        UInt iX = (UInt)(LOWORD(lParam));
        //        UInt iY = (UInt)(HIWORD(lParam));
        //        KeyCode iButton = KeyCodeFromWin32[wParam & 0xff];
        //        if ( m_pModel->OnMousePress(iX, iY, iButton) )
        //            return 0;
        //    } break;
        //case WM_LBUTTONUP: {
        //        UInt iX = (UInt)(LOWORD(lParam));
        //        UInt iY = (UInt)(HIWORD(lParam));
        //        if ( m_pModel->OnMouseRelease(iX, iY, KEYCODE_MOUSELEFT) )
        //            return 0;
        //    } break;
        //case WM_RBUTTONUP: {
        //        UInt iX = (UInt)(LOWORD(lParam));
        //        UInt iY = (UInt)(HIWORD(lParam));
        //        if ( m_pModel->OnMouseRelease(iX, iY, KEYCODE_MOUSERIGHT) )
        //            return 0;
        //    } break;
        //case WM_MBUTTONUP: {
        //        UInt iX = (UInt)(LOWORD(lParam));
        //        UInt iY = (UInt)(HIWORD(lParam));
        //        if ( m_pModel->OnMouseRelease(iX, iY, KEYCODE_MOUSEMIDDLE) )
        //            return 0;
        //    } break;
        //case WM_XBUTTONUP: {
        //        UInt iX = (UInt)(LOWORD(lParam));
        //        UInt iY = (UInt)(HIWORD(lParam));
        //        KeyCode iButton = KeyCodeFromWin32[wParam & 0xff];
        //        if ( m_pModel->OnMouseRelease(iX, iY, iButton) )
        //            return 0;
        //    } break;
        //case WM_LBUTTONDBLCLK: {
        //        UInt iX = (UInt)(LOWORD(lParam));
        //        UInt iY = (UInt)(HIWORD(lParam));
        //        if ( m_pModel->OnMouseDblClick(iX, iY, KEYCODE_MOUSELEFT) )
        //            return 0;
        //    } break;
        //case WM_RBUTTONDBLCLK: {
        //        UInt iX = (UInt)(LOWORD(lParam));
        //        UInt iY = (UInt)(HIWORD(lParam));
        //        if ( m_pModel->OnMouseDblClick(iX, iY, KEYCODE_MOUSERIGHT) )
        //            return 0;
        //    } break;
        //case WM_MBUTTONDBLCLK: {
        //        UInt iX = (UInt)(LOWORD(lParam));
        //        UInt iY = (UInt)(HIWORD(lParam));
        //        if ( m_pModel->OnMouseDblClick(iX, iY, KEYCODE_MOUSEMIDDLE) )
        //            return 0;
        //    } break;
        //case WM_XBUTTONDBLCLK: {
        //        UInt iX = (UInt)(LOWORD(lParam));
        //        UInt iY = (UInt)(HIWORD(lParam));
        //        KeyCode iButton = KeyCodeFromWin32[wParam & 0xff];
        //        if ( m_pModel->OnMouseDblClick(iX, iY, iButton) )
        //            return 0;
        //    } break;

        // Paint message
        //case WM_PAINT: {
        //        PAINTSTRUCT ps;
        //        HDC hdc = BeginPaint((HWND)hWnd, &ps);
        //        m_pModel->OnDraw();
        //        EndPaint((HWND)hWnd, &ps);
        //    } break;

        // Moving / Sizing messages
        //case WM_ENTERSIZEMOVE: {
        //        m_pModel->OnEnterMoveSize();
        //    } break;
        //case WM_EXITSIZEMOVE: {
        //        m_pModel->OnExitMoveSize();
        //    } break;
        //case WM_MOVE: {
        //        UInt iX = (UInt)(LOWORD(iLParam));
        //        UInt iY = (UInt)(HIWORD(iLParam));
        //        m_pModel->OnMove( iX, iY );
        //    } break;
        //case WM_SIZE: {
        //        UInt iWidth = (UInt)(LOWORD(iLParam));
        //        UInt iHeight = (UInt)(HIWORD(iLParam));
        //        m_pModel->OnResize( iWidth, iHeight );
        //    } break;

        // Command messages
        case WM_COMMAND: {
                // Triggered by Child Control, find which one
                HWND hCallerHandle = (HWND)lParam;
                Int iCallerResourceID = (Int)( LOWORD(wParam) );
                Int iNotificationCode = (Int)( HIWORD(wParam) );

                // Search the Caller
                WinGUIControl * pControl = _SearchControl( iCallerResourceID );

                // Dispatch message to the Control
                Bool bDone = pControl->_DispatchEvent( iNotificationCode );
                if ( bDone )
                    return 0;
            } break;

        // Notify messages
        case WM_NOTIFY: {
                // Triggered by Child Control
                HWND hCallerHandle = (HWND)lParam;
                Int iCallerResourceID = (Int)( LOWORD(wParam) );
                Int iNotificationCode = (Int)( HIWORD(wParam) );
                
                // Search the Caller
                WinGUIControl * pControl = _SearchControl( iCallerResourceID );

                // Dispatch message to the Control
                Bool bDone = pControl->_DispatchEvent( iNotificationCode );
                if ( bDone )
                    return 0;
            } break;

        // Menu messages
        //case WM_ENTERMENULOOP: {
        //        // nothing to do
        //    } break;
        //case WM_EXITMENULOOP: {
        //        // nothing to do
        //    } break;

        // Exit sequence
        case WM_CLOSE: {
                pModel->OnClose();
                return 0;
            } break;
        case WM_DESTROY: {
                //PostQuitMessage(0);
            } break;
        case WM_QUIT: {
                // Message loop exit-case, never goes here
            } break;
        default: break;
    }

    // Message wasn't handled by application
    return DefWindowProc( (HWND)hHandle, iMessage, wParam, lParam );
}

