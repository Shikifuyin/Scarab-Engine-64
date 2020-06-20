/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUIWindow.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Element : Windows
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
#include "WinGUIWindow.h"
#include "WinGUIControl.h"

/////////////////////////////////////////////////////////////////////////////////
// WinGUIWindowModel implementation
WinGUIWindowModel::WinGUIWindowModel( Int iResourceID ):
	WinGUIElementModel(iResourceID)
{
	// nothing to do
}
WinGUIWindowModel::~WinGUIWindowModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIWindow implementation
WinGUIWindow::WinGUIWindow( WinGUIWindowModel * pModel ):
	WinGUIElement(NULL, pModel)
{
    // Children Links
	m_iChildCount = 0;
	for( UInt i = 0; i < WINGUI_WINDOW_MAX_CHILDREN; ++i )
		m_arrChildren[i] = NULL;
}
WinGUIWindow::~WinGUIWindow()
{
	// nothing to do
}

WinGUIElement * WinGUIWindow::GetChildByID( Int iResourceID ) const
{
	for ( UInt i = 0; i < m_iChildCount; ++i ) {
		WinGUIElement * pChild = m_arrChildren[i];
		if ( _GetResourceID(pChild) == iResourceID )
			return pChild;
	}
	return NULL;
}

Bool WinGUIWindow::IsVisible() const
{
    return ( IsWindowVisible((HWND)m_hHandle) != FALSE );
}
Void WinGUIWindow::SetVisible( Bool bVisible )
{
    if ( bVisible ) {
        ShowWindow( (HWND)m_hHandle, SW_SHOW );
        UpdateWindow( (HWND)m_hHandle );
    } else
        ShowWindow( (HWND)m_hHandle, SW_HIDE );
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIWindow::_Create()
{
    DebugAssert( m_hHandle == NULL );

    WinGUIWindowModel * pModel = (WinGUIWindowModel*)m_pModel;

    // Build Style
    DWord dwWindowStyle = ( WS_OVERLAPPED | WS_CAPTION | WS_CLIPCHILDREN | WS_CLIPSIBLINGS );
    if ( pModel->HasSystemMenu() ) {
        dwWindowStyle |= WS_SYSMENU;
        if ( pModel->HasMinimizeButton() )
            dwWindowStyle |= WS_MINIMIZEBOX;
        if ( pModel->HasMaximizeButton() )
            dwWindowStyle |= WS_MAXIMIZEBOX;
    }
    if ( pModel->AllowResizing() )
        dwWindowStyle |= WS_SIZEBOX;

    // Window region
    RECT rectWindow;
    SetRect( &rectWindow, pModel->GetPositionX(), pModel->GetPositionY(),
                          pModel->GetPositionX() + pModel->GetWidth(),
                          pModel->GetPositionY() + pModel->GetHeight() );
    AdjustWindowRect( &rectWindow, dwWindowStyle, FALSE );

    // Window class
    WNDCLASSEX winClass;
    winClass.cbSize = sizeof(WNDCLASSEX);
    winClass.cbClsExtra = 0;
	winClass.cbWndExtra = 0;
    winClass.hInstance = GetModuleHandle( NULL );
	winClass.style = CS_DBLCLKS;
	winClass.lpfnWndProc = (WNDPROC)( _MessageCallback_Static );
	winClass.lpszClassName = pModel->GetClassNameID();
    winClass.lpszMenuName = NULL;
	winClass.hbrBackground = (HBRUSH)( COLOR_WINDOW + 1 );
	winClass.hCursor = (HCURSOR)( LoadImage(NULL, IDC_ARROW, IMAGE_CURSOR, 0, 0, LR_DEFAULTSIZE) );
	winClass.hIcon = (HICON)( LoadImage(NULL, IDI_APPLICATION, IMAGE_ICON, 0, 0, LR_DEFAULTSIZE) );
	winClass.hIconSm = (HICON)( LoadImage(NULL, IDI_APPLICATION, IMAGE_ICON, 0, 0, LR_DEFAULTSIZE) );
	RegisterClassEx( &winClass );

    // Window creation
    m_hHandle = CreateWindowEx (
        0, pModel->GetClassNameID(), pModel->GetTitle(), dwWindowStyle,
        pModel->GetPositionX(), pModel->GetPositionY(),
        (rectWindow.right - rectWindow.left), (rectWindow.bottom - rectWindow.top),
		NULL, NULL,
        GetModuleHandle(NULL),
        (Void*)this
	);
    DebugAssert( m_hHandle != NULL );

    // Done
    _SaveElementToHandle();
}
Void WinGUIWindow::_Destroy()
{
    DebugAssert( m_hHandle != NULL );

    WinGUIWindowModel * pModel = (WinGUIWindowModel*)m_pModel;

    DestroyWindow( (HWND)m_hHandle );
    m_hHandle = NULL;

    UnregisterClass( pModel->GetClassNameID(), GetModuleHandle(NULL) );
}

UIntPtr __stdcall WinGUIWindow::_MessageCallback_Static( Void * hHandle, UInt iMessage, UIntPtr wParam, UIntPtr lParam )
{
	static WinGUIWindow * s_pThis = NULL;
    if ( s_pThis != NULL )
        return s_pThis->_MessageCallback_Virtual( hHandle, iMessage, wParam, lParam );

    if ( iMessage == WM_CREATE )
        s_pThis = (WinGUIWindow*)( ((LPCREATESTRUCT)lParam)->lpCreateParams );

    return DefWindowProc( (HWND)hHandle, iMessage, wParam, lParam );
}
UIntPtr __stdcall WinGUIWindow::_MessageCallback_Virtual( Void * hHandle, UInt iMessage, UIntPtr wParam, UIntPtr lParam )
{
    WinGUIWindowModel * pModel = (WinGUIWindowModel*)m_pModel;

    // Dispatch Message
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
                HWND hCallerHandle = (HWND)lParam;
                Int iCallerResourceID = (Int)( LOWORD(wParam) );
                Int iNotificationCode = (Int)( HIWORD(wParam) );

                // Retrieve Caller Element
                WinGUIElement * pCallerElement = _GetElementFromHandle( hCallerHandle );
                DebugAssert( _GetResourceID(pCallerElement) == iCallerResourceID );

                // Dispatch message to the Control
                if ( pCallerElement->GetElementType() == WINGUI_ELEMENT_CONTROL ) {
                    WinGUIControl * pCallerControl = (WinGUIControl*)pCallerElement;
                    if ( pCallerControl->_DispatchEvent( iNotificationCode ) )
                        return 0;
                }
            } break;

        // Notify messages
        case WM_NOTIFY: {
                HWND hCallerHandle = (HWND)lParam;
                Int iCallerResourceID = (Int)( LOWORD(wParam) );
                Int iNotificationCode = (Int)( HIWORD(wParam) );
                
                // Retrieve Caller Element
                WinGUIElement * pCallerElement = _GetElementFromHandle( hCallerHandle );
                DebugAssert( _GetResourceID(pCallerElement) == iCallerResourceID );

                // Dispatch message to the Control
                if ( pCallerElement->GetElementType() == WINGUI_ELEMENT_CONTROL ) {
                    WinGUIControl * pCallerControl = (WinGUIControl*)pCallerElement;
                    if ( pCallerControl->_DispatchEvent( iNotificationCode ) )
                        return 0;
                }
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
                PostQuitMessage(0);
            } break;
        default: break;
    }

    // Message wasn't handled by application
    return DefWindowProc( (HWND)hHandle, iMessage, wParam, lParam );
}

Void WinGUIWindow::_AppendChild( WinGUIElement * pElement )
{
    DebugAssert( m_iChildCount < WINGUI_WINDOW_MAX_CHILDREN );

    m_arrChildren[m_iChildCount] = pElement;
    ++m_iChildCount;
}
Void WinGUIWindow::_RemoveChild( WinGUIElement * pElement )
{
    UInt iIndex;
    for( iIndex = 0; iIndex < m_iChildCount; ++iIndex ) {
        if ( m_arrChildren[iIndex] == pElement )
            break;
    }
    DebugAssert( iIndex < m_iChildCount );

    m_arrChildren[iIndex] = m_arrChildren[m_iChildCount - 1];
    m_arrChildren[m_iChildCount - 1] = NULL;
    --m_iChildCount;
}

