/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Events/WinGUIEventProcessor.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Event Processor
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
#include "WinGUIEventProcessor.h"

/////////////////////////////////////////////////////////////////////////////////
// WinGUIEventProcessorModel implementation
WinGUIEventProcessorModel::WinGUIEventProcessorModel()
{
}
WinGUIEventProcessorModel::~WinGUIEventProcessorModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIEventProcessor implementation
WinGUIEventProcessor::WinGUIEventProcessor( WinGUIEventProcessorModel * pModel )
{
    m_pModel = pModel;
}
WinGUIEventProcessor::~WinGUIEventProcessor()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////

UIntPtr __stdcall WinGUIEventProcessor::_MessageCallback_Static( Void * hWnd, UInt message, UIntPtr wParam, UIntPtr lParam )
{
	static WinGUIEventProcessor * s_pThis = NULL;
    if ( s_pThis != NULL )
        return s_pThis->_MessageCallback_Virtual( hWnd, message, wParam, lParam );

    if ( message == WM_CREATE )
        s_pThis = (WinGUIEventProcessor*)( ((LPCREATESTRUCT)lParam)->lpCreateParams );

    return DefWindowProc( (HWND)hWnd, message, wParam, lParam );
}
UIntPtr __stdcall WinGUIEventProcessor::_MessageCallback_Virtual( Void * hWnd, UInt message, UIntPtr wParam, UIntPtr lParam )
{
    switch( message ) {
        // Keyboard messages
        case WM_SYSKEYDOWN:
        case WM_KEYDOWN: {
                // Handle key press
                KeyCode iKey = KeyCodeFromWin32[wParam & 0xff];
                if ( m_pModel->OnKeyPress(iKey) )
                    return 0;
            } break;
        case WM_SYSKEYUP:
        case WM_KEYUP: {
                // Handle key release
                KeyCode iKey = KeyCodeFromWin32[wParam & 0xff];
                if ( m_pModel->OnKeyRelease(iKey) )
                    return 0;
            } break;

        // Mouse messages
        case WM_MOUSEMOVE: {
                UInt iX = (UInt)(LOWORD(lParam));
                UInt iY = (UInt)(HIWORD(lParam));
                m_pModel->OnMouseMove( iX, iY );
                return 0;
            } break;
        case WM_MOUSEWHEEL: {
                POINT pt; // WM_MOUSEWHEEL needs screen->client conversion
                pt.x = (UInt)(LOWORD(lParam));
                pt.y = (UInt)(HIWORD(lParam));
                ScreenToClient( (HWND)hWnd, &pt );
                Int iWheelDelta = (Int)( (Short)(HIWORD(wParam)) ) / WHEEL_DELTA;
                if ( m_pModel->OnMouseWheel((UInt)pt.x, (UInt)pt.y, iWheelDelta) )
                    return 0;
            } break;
        case WM_LBUTTONDOWN: {
                UInt iX = (UInt)(LOWORD(lParam));
                UInt iY = (UInt)(HIWORD(lParam));
                if ( m_pModel->OnMousePress(iX, iY, KEYCODE_MOUSELEFT) )
                    return 0;
            } break;
        case WM_RBUTTONDOWN: {
                UInt iX = (UInt)(LOWORD(lParam));
                UInt iY = (UInt)(HIWORD(lParam));
                if ( m_pModel->OnMousePress(iX, iY, KEYCODE_MOUSERIGHT) )
                    return 0;
            } break;
        case WM_MBUTTONDOWN: {
                UInt iX = (UInt)(LOWORD(lParam));
                UInt iY = (UInt)(HIWORD(lParam));
                if ( m_pModel->OnMousePress(iX, iY, KEYCODE_MOUSEMIDDLE) )
                    return 0;
            } break;
        case WM_XBUTTONDOWN: {
                UInt iX = (UInt)(LOWORD(lParam));
                UInt iY = (UInt)(HIWORD(lParam));
                KeyCode iButton = KeyCodeFromWin32[wParam & 0xff];
                if ( m_pModel->OnMousePress(iX, iY, iButton) )
                    return 0;
            } break;
        case WM_LBUTTONUP: {
                UInt iX = (UInt)(LOWORD(lParam));
                UInt iY = (UInt)(HIWORD(lParam));
                if ( m_pModel->OnMouseRelease(iX, iY, KEYCODE_MOUSELEFT) )
                    return 0;
            } break;
        case WM_RBUTTONUP: {
                UInt iX = (UInt)(LOWORD(lParam));
                UInt iY = (UInt)(HIWORD(lParam));
                if ( m_pModel->OnMouseRelease(iX, iY, KEYCODE_MOUSERIGHT) )
                    return 0;
            } break;
        case WM_MBUTTONUP: {
                UInt iX = (UInt)(LOWORD(lParam));
                UInt iY = (UInt)(HIWORD(lParam));
                if ( m_pModel->OnMouseRelease(iX, iY, KEYCODE_MOUSEMIDDLE) )
                    return 0;
            } break;
        case WM_XBUTTONUP: {
                UInt iX = (UInt)(LOWORD(lParam));
                UInt iY = (UInt)(HIWORD(lParam));
                KeyCode iButton = KeyCodeFromWin32[wParam & 0xff];
                if ( m_pModel->OnMouseRelease(iX, iY, iButton) )
                    return 0;
            } break;
        case WM_LBUTTONDBLCLK: {
                UInt iX = (UInt)(LOWORD(lParam));
                UInt iY = (UInt)(HIWORD(lParam));
                if ( m_pModel->OnMouseDblClick(iX, iY, KEYCODE_MOUSELEFT) )
                    return 0;
            } break;
        case WM_RBUTTONDBLCLK: {
                UInt iX = (UInt)(LOWORD(lParam));
                UInt iY = (UInt)(HIWORD(lParam));
                if ( m_pModel->OnMouseDblClick(iX, iY, KEYCODE_MOUSERIGHT) )
                    return 0;
            } break;
        case WM_MBUTTONDBLCLK: {
                UInt iX = (UInt)(LOWORD(lParam));
                UInt iY = (UInt)(HIWORD(lParam));
                if ( m_pModel->OnMouseDblClick(iX, iY, KEYCODE_MOUSEMIDDLE) )
                    return 0;
            } break;
        case WM_XBUTTONDBLCLK: {
                UInt iX = (UInt)(LOWORD(lParam));
                UInt iY = (UInt)(HIWORD(lParam));
                KeyCode iButton = KeyCodeFromWin32[wParam & 0xff];
                if ( m_pModel->OnMouseDblClick(iX, iY, iButton) )
                    return 0;
            } break;

        // Paint message
        case WM_PAINT: {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint((HWND)hWnd, &ps);
                m_pModel->OnDraw();
                EndPaint((HWND)hWnd, &ps);
            } break;

        // Moving / Sizing messages
        case WM_ENTERSIZEMOVE: {
                m_pModel->OnEnterMoveSize();
            } break;
        case WM_EXITSIZEMOVE: {
                m_pModel->OnExitMoveSize();
            } break;
        case WM_MOVE: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                m_pModel->OnMove( iX, iY );
            } break;
        case WM_SIZE: {
                UInt iWidth = (UInt)(LOWORD(iLParam));
                UInt iHeight = (UInt)(HIWORD(iLParam));
                m_pModel->OnResize( iWidth, iHeight );
            } break;

        // Command messages
        case WM_COMMAND: {

            } break;

        // Notify messages
        case WM_NOTIFY: {
                
            } break;

        // Menu messages
        case WM_ENTERMENULOOP: {
                // nothing to do
            } break;
        case WM_EXITMENULOOP: {
                // nothing to do
            } break;

        // Exit sequence
        case WM_CLOSE: {
                m_pModel->OnClose();
                return 0;
            } break;
        case WM_DESTROY: {
                PostQuitMessage(0);
            } break;
        case WM_QUIT: {
                // Message loop exit-case, never goes here
            } break;
        default: break;
    }

    // Message wasn't handled by application
    return DefWindowProc( (HWND)hWnd, message, wParam, lParam );
}
