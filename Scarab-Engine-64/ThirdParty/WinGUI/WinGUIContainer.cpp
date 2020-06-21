/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUIContainer.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Element : Containers
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

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUIContainerModel implementation
WinGUIContainerModel::WinGUIContainerModel( Int iResourceID ):
	WinGUIElementModel(iResourceID)
{
	// nothing to do
}
WinGUIContainerModel::~WinGUIContainerModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIContainer implementation
WinGUIContainer::WinGUIContainer( WinGUIElement * pParent, WinGUIContainerModel * pModel ):
	WinGUIElement(pParent, pModel)
{
    // Children Links
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
		if ( _GetResourceID(pChild) == iResourceID )
			return pChild;
	}
	return NULL;
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIContainer::_Create()
{
    DebugAssert( m_hHandle == NULL );

    WinGUIContainerModel * pModel = (WinGUIContainerModel*)m_pModel;
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Build Style
    DWord dwWindowStyle = ( WS_VISIBLE | WS_CHILD | WS_BORDER | WS_CLIPCHILDREN | WS_CLIPSIBLINGS );
    if ( pModel->AllowResizing() )
        dwWindowStyle |= WS_SIZEBOX;

    // Window region
    const WinGUIRectangle * pRect = pModel->GetRectangle();

    // Window class
    WNDCLASSEX winClass;
    winClass.cbSize = sizeof(WNDCLASSEX);
    winClass.cbClsExtra = 0;
	winClass.cbWndExtra = 0;
    winClass.hInstance = GetModuleHandle( NULL );
	winClass.style = CS_DBLCLKS | CS_PARENTDC;
	winClass.lpfnWndProc = (WNDPROC)( _MessageCallback_Static );
	winClass.lpszClassName = pModel->GetClassNameID();
    winClass.lpszMenuName = NULL;
	winClass.hbrBackground = (HBRUSH)( COLOR_WINDOW + 1 );
	winClass.hCursor = NULL;
	winClass.hIcon = NULL;
	winClass.hIconSm = NULL;
	RegisterClassEx( &winClass );

    // Window creation
    m_hHandle = CreateWindowEx (
        WS_EX_CONTROLPARENT, pModel->GetClassNameID(), NULL, dwWindowStyle,
        pRect->iLeft, pRect->iTop,
        pRect->iWidth, pRect->iHeight,
		hParentWnd, (HMENU)m_iResourceID,
        GetModuleHandle(NULL),
        (Void*)this
	);
    DebugAssert( m_hHandle != NULL );    

    // Done
    _SaveElementToHandle();
}
Void WinGUIContainer::_Destroy()
{
    DebugAssert( m_hHandle != NULL );

    WinGUIContainerModel * pModel = (WinGUIContainerModel*)m_pModel;

    DestroyWindow( (HWND)m_hHandle );
    m_hHandle = NULL;

    UnregisterClass( pModel->GetClassNameID(), GetModuleHandle(NULL) );
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
                Int iCallerResourceID = (Int)( LOWORD(wParam) );
                Int iNotificationCode = (Int)( HIWORD(wParam) );

                // Retrieve Caller Element
                WinGUIElement * pCallerElement = _GetElementFromHandle( hCallerHandle );
                if ( pCallerElement != NULL ) {
                    DebugAssert( _GetResourceID( pCallerElement ) == iCallerResourceID );

                    // Dispatch message to the Control
                    if ( pCallerElement->GetElementType() == WINGUI_ELEMENT_CONTROL ) {
                        WinGUIControl * pCallerControl = (WinGUIControl *)pCallerElement;
                        if ( pCallerControl->_DispatchEvent( iNotificationCode ) )
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
                        if ( pCallerControl->_DispatchEvent( iNotificationCode ) )
                            return 0;
                    }
                }
            } break;

        default: break;
    }

    // Message wasn't handled by application
    return DefWindowProc( (HWND)hHandle, iMessage, wParam, lParam );
}

Void WinGUIContainer::_AppendChild( WinGUIElement * pElement )
{
    DebugAssert( m_iChildCount < WINGUI_CONTAINER_MAX_CHILDREN );

    m_arrChildren[m_iChildCount] = pElement;
    ++m_iChildCount;
}
Void WinGUIContainer::_RemoveChild( WinGUIElement * pElement )
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

