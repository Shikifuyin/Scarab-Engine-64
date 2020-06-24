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
    // Default Size
    m_hCreationParameters.hClientRect.iLeft = 100;
    m_hCreationParameters.hClientRect.iTop = 100;
    m_hCreationParameters.hClientRect.iWidth = 800;
    m_hCreationParameters.hClientRect.iHeight = 600;

    // Default Class Name
    StringFn->Copy( m_hCreationParameters.strClassName, TEXT("ApplicationWindow") );

    // Default Title
    StringFn->Copy( m_hCreationParameters.strTitle, TEXT("Scarab-Engine-64 Application") );

    // Default Parameters
    m_hCreationParameters.bHasSystemMenu = true;
    m_hCreationParameters.bHasMinimizeButton = true;
    m_hCreationParameters.bHasMaximizeButton = false; // Default to fixed
    m_hCreationParameters.bAllowResizing = false;     // size window

    m_hCreationParameters.bClipChildren = false;  // Allow Tabs to work properly
    m_hCreationParameters.bClipSibblings = false; // Not needed most of the time
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

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIWindow::_Create()
{
    DebugAssert( m_hHandle == NULL );

    // Get Model
    WinGUIWindowModel * pModel = (WinGUIWindowModel*)m_pModel;

    // No-Parent case, Don't use Layout

    // Get Creation Parameters
    const WinGUIWindowParameters * pParameters = pModel->GetCreationParameters();

    // Build Style
    DWord dwWindowStyle = ( WS_OVERLAPPED | WS_CAPTION );
    if ( pParameters->bHasSystemMenu ) {
        dwWindowStyle |= WS_SYSMENU;
        if ( pParameters->bHasMinimizeButton )
            dwWindowStyle |= WS_MINIMIZEBOX;
        if ( pParameters->bHasMaximizeButton )
            dwWindowStyle |= WS_MAXIMIZEBOX;
    }
    if ( pParameters->bAllowResizing )
        dwWindowStyle |= WS_SIZEBOX;
    if ( pParameters->bClipChildren )
        dwWindowStyle |= WS_CLIPCHILDREN;
    if ( pParameters->bClipSibblings )
        dwWindowStyle |= WS_CLIPSIBLINGS;

    // Compute Window Rect
    RECT rectWindow;
    SetRect( &rectWindow, 0, 0, pParameters->hClientRect.iWidth, pParameters->hClientRect.iHeight );
    AdjustWindowRect( &rectWindow, dwWindowStyle, FALSE );

    // Register Window class
    WNDCLASSEX winClass;
    winClass.cbSize = sizeof(WNDCLASSEX);
    winClass.cbClsExtra = 0;
	winClass.cbWndExtra = 0;
    winClass.hInstance = GetModuleHandle( NULL );
	winClass.style = CS_DBLCLKS;
	winClass.lpfnWndProc = (WNDPROC)( _MessageCallback_Static );
	winClass.lpszClassName = pParameters->strClassName;
    winClass.lpszMenuName = NULL;
	winClass.hbrBackground = (HBRUSH)( COLOR_3DFACE + 1 );
	winClass.hCursor = (HCURSOR)( LoadImage(NULL, IDC_ARROW, IMAGE_CURSOR, 0, 0, LR_DEFAULTSIZE) );
	winClass.hIcon = (HICON)( LoadImage(NULL, IDI_APPLICATION, IMAGE_ICON, 0, 0, LR_DEFAULTSIZE) );
	winClass.hIconSm = (HICON)( LoadImage(NULL, IDI_APPLICATION, IMAGE_ICON, 0, 0, LR_DEFAULTSIZE) );
	RegisterClassEx( &winClass );

    // Window creation
    m_hHandle = CreateWindowEx (
        WS_EX_CONTROLPARENT,
        pParameters->strClassName,
        pParameters->strTitle,
        dwWindowStyle,
        pParameters->hClientRect.iLeft, pParameters->hClientRect.iTop,
        (rectWindow.right - rectWindow.left), (rectWindow.bottom - rectWindow.top),
		NULL, // No Parent
        NULL, // No Menu
        GetModuleHandle( NULL ),
        (Void*)this
	);
    DebugAssert( m_hHandle != NULL );

    // Done
    _SaveElementToHandle();
}
Void WinGUIWindow::_Destroy()
{
    DebugAssert( m_hHandle != NULL );

    // Get Model
    WinGUIWindowModel * pModel = (WinGUIWindowModel*)m_pModel;

    // Window destruction
    DestroyWindow( (HWND)m_hHandle );
    m_hHandle = NULL;

    // Unregister window class
    UnregisterClass( pModel->GetCreationParameters()->strClassName, GetModuleHandle(NULL) );
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
    // Get Model
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
        //        HDC hdc = BeginPaint((HWND)m_hHandle, &ps);
        //        //m_pModel->OnDraw();
        //        EndPaint((HWND)m_hHandle, &ps);
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
                Int iNotificationCode = (Int)( HIWORD(wParam) );
                Int iCallerResourceID = (Int)( LOWORD(wParam) );

                // Check for Menu / Accelerator
                if ( hCallerHandle == NULL )
                    break;

                // Control Notification
                WinGUIElement * pCallerElement = _GetElementFromHandle( hCallerHandle );
                if ( pCallerElement != NULL ) {
                    DebugAssert( _GetResourceID( pCallerElement ) == iCallerResourceID );

                    // Dispatch message to the Control
                    if ( pCallerElement->GetElementType() == WINGUI_ELEMENT_CONTROL ) {
                        WinGUIControl * pCallerControl = (WinGUIControl *)pCallerElement;
                        if ( pCallerControl->_DispatchEvent( iNotificationCode, NULL ) )
                            return 0;
                    }
                }
            } break;

        // Notify messages
        case WM_NOTIFY: {
                NMHDR * pHeader = (NMHDR*)lParam;
                HWND hCallerHandle = pHeader->hwndFrom;
                Int iCallerResourceID = pHeader->idFrom;
                Int iNotificationCode = pHeader->code;
                
                // Retrieve Caller Element
                WinGUIElement * pCallerElement = _GetElementFromHandle( hCallerHandle );
                if ( pCallerElement != NULL ) {
                    DebugAssert( _GetResourceID( pCallerElement ) == iCallerResourceID );

                    // Dispatch message to the Control
                    if ( pCallerElement->GetElementType() == WINGUI_ELEMENT_CONTROL ) {
                        WinGUIControl * pCallerControl = (WinGUIControl *)pCallerElement;
                        if ( pCallerControl->_DispatchEvent( iNotificationCode, (Void*)lParam ) )
                            return 0;
                    }
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
                if ( pModel->OnClose() )
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

