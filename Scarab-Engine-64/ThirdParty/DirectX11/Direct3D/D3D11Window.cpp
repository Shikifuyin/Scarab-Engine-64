/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/D3D11Window.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Platform-dependant window implementation.
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
#pragma warning(disable:4005)

#define WIN32_LEAN_AND_MEAN
//#define INITGUID // Needed for DXGIDebug
#include <d3d11.h>
#include <dxgidebug.h>

#undef DebugAssert

//typedef HRESULT (__stdcall * _DXGIGetDebugInterface)( const IID & rIID, void ** ppDebug ); // Needed for DXGIDebug

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "D3D11Window.h"

#include "D3D11Renderer.h"

/////////////////////////////////////////////////////////////////////////////////
// D3D11WindowCallbacks implementation
D3D11WindowCallbacks::D3D11WindowCallbacks( Void * pUserData )
{
    m_pUserData = pUserData;
}
D3D11WindowCallbacks::~D3D11WindowCallbacks()
{
    // nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11Window implementation
D3D11Window::D3D11Window( const GChar * strTitle, UInt iX, UInt iY, UInt iWidth, UInt iHeight, D3D11WindowCallbacks * pCallbacks )
{
    // Renderer bind
    m_pRenderer = NULL;

    // Window management
    StringFn->NCopy( m_strTitle, strTitle, D3D11WINDOW_TITLE_SIZE - 1 );
    m_pCallbacks = pCallbacks;

    m_hInstance = (HINSTANCE)( GetModuleHandle(NULL) );
    m_hWindow = NULL;
    m_hWindowMonitor = NULL;
    m_hDefaultWindowCursor = NULL;
    m_hWindowCursor = NULL;
    m_hWindowMenu = NULL;

    m_bActive = false;
    m_bMinimized = false;
    m_bMaximized = false;
    m_bIsInSizeMove = false;

    _CreateWindow( iX, iY, iWidth, iHeight );

    // Display Enumeration
    m_pDXGIFactory = NULL;
    HRESULT hRes = CreateDXGIFactory( __uuidof(IDXGIFactory), &m_pDXGIFactory );
    DebugAssert( hRes == S_OK && m_pDXGIFactory != NULL );

    m_iAdapterCount = 0;
    m_arrOutputCounts[0] = 0;
    m_arrDisplayModeCounts[0] = 0;

    m_iAdapter = INVALID_OFFSET;
    m_iOutput = INVALID_OFFSET;
    m_iDisplayMode = INVALID_OFFSET;
    m_hAdapterDesc.iIndex = INVALID_OFFSET;
    m_hOutputDesc.iIndex = INVALID_OFFSET;
    m_hDisplayModeDesc.iIndex = INVALID_OFFSET;
    m_pAdapter = NULL;
    m_pOutput = NULL;

    _Enumerate();

    _SelectAdapter( 0 );

    m_bFullScreen = false;
    m_bFullScreenWindowed = false;
    m_iFullScreenWindowedSavedX = 0;
    m_iFullScreenWindowedSavedY = 0;
    m_iFullScreenWindowedSavedWidth = 0;
    m_iFullScreenWindowedSavedHeight = 0;
    m_dwFullScreenWindowedSavedStyle = 0;
    m_bInDisplayModeSwitchTransition = false;
}
D3D11Window::~D3D11Window()
{
    // Renderer bind
    DebugAssert( m_pRenderer == NULL );

    // Display modes management
    if ( m_iAdapterCount > 0 )
        _ClearEnumeration();

    if ( m_pDXGIFactory != NULL ) {
        ((IDXGIFactory*)m_pDXGIFactory)->Release();
        m_pDXGIFactory = NULL;
    }

    // Window management
    if ( m_hWindow != NULL )
        _DestroyWindow();

    m_hInstance = NULL;

    // Memory objects report for DXGI /////////////////////////////////////
    //HMODULE hModule = GetModuleHandle( TEXT("dxgidebug.dll") );
    //_DXGIGetDebugInterface pfProc = (_DXGIGetDebugInterface)( GetProcAddress( hModule, "DXGIGetDebugInterface" ) );

    //IDXGIDebug * pDebug = NULL;
    //HRESULT hRes = pfProc( __uuidof(IDXGIDebug), (void**)&pDebug );
    //DebugAssert( hRes == S_OK && pDebug != NULL );
    //hRes = pDebug->ReportLiveObjects( DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_ALL );
    //DebugAssert( hRes == S_OK );
    //pDebug->Release();
    //pDebug = NULL;
    ///////////////////////////////////////////////////////////////////////
}

Void D3D11Window::SetTitle( const GChar * strTitle )
{
    StringFn->NCopy( m_strTitle, strTitle, D3D11WINDOW_TITLE_SIZE - 1 );
    
    BOOL bRes = SetWindowText( (HWND)m_hWindow, m_strTitle );
    DebugAssert( bRes == TRUE );
}

Void D3D11Window::GetScreenSize( UInt * outWidth, UInt * outHeight ) const
{
    *outWidth = (UInt)( GetSystemMetrics(SM_CXSCREEN) );
    *outHeight = (UInt)( GetSystemMetrics(SM_CYSCREEN) );
}
Void D3D11Window::GetWindowRect( D3D11Rectangle * outWindowRect ) const
{
    RECT hRECT;
    BOOL bRes = ::GetWindowRect( (HWND)m_hWindow, &hRECT );
    DebugAssert( bRes == TRUE );

    outWindowRect->iLeft = hRECT.left;
    outWindowRect->iTop = hRECT.top;
    outWindowRect->iRight = hRECT.right;
    outWindowRect->iBottom = hRECT.bottom;
}
Void D3D11Window::GetClientRect( D3D11Rectangle * outClientRect ) const
{
    RECT hRECT;
    BOOL bRes = ::GetClientRect( (HWND)m_hWindow, &hRECT );
    DebugAssert( bRes == TRUE );

    outClientRect->iLeft = hRECT.left;
    outClientRect->iTop = hRECT.top;
    outClientRect->iRight = hRECT.right;
    outClientRect->iBottom = hRECT.bottom;
}

UInt D3D11Window::GetDpi() const
{
    return GetDpiForWindow( (HWND)m_hWindow );
}

Bool D3D11Window::IsVisible() const
{
    return ( IsWindowVisible((HWND)m_hWindow) == TRUE );
}
Void D3D11Window::SetVisible( Bool bVisible ) const
{
    ShowWindow( (HWND)m_hWindow, (bVisible) ? SW_SHOW : SW_HIDE );
}
//Void D3D11Window::Update() const
//{
//    BOOL bRes = UpdateWindow( (HWND)m_hWindow );
//    DebugAssert( bRes == TRUE );
//}

Void D3D11Window::MessageLoop() const
{
    HACCEL hAccel = NULL;

    MSG msg;
    while( true ) {
        if ( PeekMessage(&msg, NULL, 0u, 0u, PM_REMOVE) != FALSE ) {
            if ( msg.message == WM_QUIT )
                break;
            if ( hAccel == NULL || TranslateAccelerator((HWND)m_hWindow, hAccel, &msg) == 0 ) {
                TranslateMessage( &msg );
		        DispatchMessage( &msg );
            }
        } else
            m_pCallbacks->OnIdleTime();
    }

    if ( hAccel != NULL )
        DestroyAcceleratorTable( hAccel );
}
Void D3D11Window::Exit() const
{
    BOOL bRes = PostMessage( (HWND)m_hWindow, WM_CLOSE, 0, 0 );
    DebugAssert( bRes == TRUE );
}

Void D3D11Window::ClipCursor( Bool bClip, Bool bClipToClientRect ) const
{
    if ( !bClip ) {
        BOOL bRes = ::ClipCursor( NULL );
        DebugAssert( bRes == TRUE );
        return;
    }
    RECT rectClip;
    if ( bClipToClientRect )
        ::GetClientRect( (HWND)m_hWindow, &rectClip );
    else
        ::GetWindowRect( (HWND)m_hWindow, &rectClip );
    BOOL bRes = ::ClipCursor( &rectClip );
    DebugAssert( bRes == TRUE );
}
Void D3D11Window::ShowCursor( Bool bShow ) const
{
    ::ShowCursor( (bShow) ? TRUE : FALSE );
}
Void D3D11Window::WarpCursor( UInt iX, UInt iY ) const
{
    BOOL bRes = SetCursorPos( iX, iY );
    DebugAssert( bRes == TRUE );
}
Void D3D11Window::SetCursor( const DWord * pCursorRGBA, UInt iWidth, UInt iHeight, DWord dwKeyColor, UInt iHotspotX, UInt iHotspotY )
{
    HCURSOR hNewCursor = (HCURSOR)m_hDefaultWindowCursor;
    if ( pCursorRGBA != NULL ) {
        DebugAssert( iWidth > 0 && iHeight > 0 );

        HDC hDC = GetDC( NULL );
        HDC hMaskDC = CreateCompatibleDC( hDC );
        HDC hColorDC = CreateCompatibleDC( hDC );
        HBITMAP hMaskBmp = CreateCompatibleBitmap( hMaskDC, iWidth, iHeight );
        HBITMAP hColorBmp = CreateCompatibleBitmap( hColorDC, iWidth, iHeight );
        HBITMAP hOldMaskBmp = (HBITMAP)( SelectObject(hMaskDC, hMaskBmp) );
        HBITMAP hOldColorBmp = (HBITMAP)( SelectObject(hColorDC, hColorBmp) );

        DWord dwColor;
        for( UInt iY = 0; iY < iHeight; ++iY ) {
            for( UInt iX = 0; iX < iWidth; ++iX ) {
                dwColor = pCursorRGBA[ (iY * iWidth) + iX ];
                if ( dwColor == dwKeyColor ) {
                    SetPixel( hMaskDC, iX, iY, RGB(0xff,0xff,0xff) );
                    SetPixel( hColorDC, iX, iY, RGB(0,0,0) );
                } else {
                    SetPixel( hMaskDC, iX, iY, RGB(0,0,0) );
                    SetPixel( hColorDC, iX, iY, dwColor );
                }
            }
        }

        SelectObject(hMaskDC, hOldMaskBmp);
        SelectObject(hColorDC, hOldColorBmp);
        DeleteDC( hMaskDC );
        DeleteDC( hColorDC );
        ReleaseDC( NULL, hDC );

        ICONINFO hIconInfos;
        hIconInfos.fIcon = FALSE;
        hIconInfos.xHotspot = iHotspotX;
        hIconInfos.yHotspot = iHotspotY;
        hIconInfos.hbmMask = hMaskBmp;
        hIconInfos.hbmColor = hColorBmp;
        hNewCursor = CreateIconIndirect( &hIconInfos );
    }

    HCURSOR hOldCursor = ::SetCursor( hNewCursor );
    DebugAssert( hOldCursor == m_hWindowCursor );

    if ( m_hWindowCursor != hNewCursor ) {
        if ( m_hWindowCursor != m_hDefaultWindowCursor )
            DestroyCursor( (HCURSOR)m_hWindowCursor );
        m_hWindowCursor = hNewCursor;
    }
}

Bool D3D11Window::GetClipboardData( GChar * outText, UInt iMaxLength ) const
{
#if ( defined(UNICODE) || defined(_UNICODE) )
    UInt iFormat = CF_UNICODETEXT;
#else
    UInt iFormat = CF_TEXT;
#endif

    // Check data availability
    if ( !(IsClipboardFormatAvailable(iFormat)) )
        return false;

    // Open Clipboard
    BOOL bOk = OpenClipboard( (HWND)m_hWindow );
    if ( !bOk )
        return false;

    // Get clipboard data handle
    HGLOBAL hGlobalData = ::GetClipboardData( iFormat );

    // Retrieve data
    GChar * strGlobalMem = (GChar*)( GlobalLock(hGlobalData) );
    DebugAssert( strGlobalMem != NULL );

    StringFn->NCopy( outText, strGlobalMem, iMaxLength );

    GlobalUnlock( hGlobalData );

    // Close clipboard
    bOk = CloseClipboard();
    DebugAssert( bOk );

    return true;
}
Bool D3D11Window::SetClipboardData( const GChar * strText, UInt iLength ) const
{
#if ( defined(UNICODE) || defined(_UNICODE) )
    UInt iFormat = CF_UNICODETEXT;
#else
    UInt iFormat = CF_TEXT;
#endif

    if ( iLength == INVALID_OFFSET )
        iLength = StringFn->Length( strText );

    // Open Clipboard
    BOOL bOk = OpenClipboard( (HWND)m_hWindow );
    if ( !bOk )
        return false;

    // Reset clipboard content
    bOk = EmptyClipboard();
    DebugAssert( bOk );

    // Allocate global memory
    HGLOBAL hGlobalData = GlobalAlloc( GMEM_MOVEABLE, (iLength + 1) * sizeof(GChar) );

    // Fill global memory
    GChar * strGlobalMem = (GChar*)( GlobalLock(hGlobalData) );
    DebugAssert( strGlobalMem != NULL );

    StringFn->NCopy( strGlobalMem, strText, iLength );

    GlobalUnlock( hGlobalData ); 

    // Set clipboard data
    HANDLE hClipboardData = ::SetClipboardData( iFormat, hGlobalData );
    DebugAssert( hClipboardData != NULL );

    // Close clipboard
    bOk = CloseClipboard();
    DebugAssert( bOk );

    return true;
}

UInt D3D11Window::MatchDisplayMode( D3D11DisplayModeDesc * outMatchedDisplayMode, const D3D11DisplayModeDesc * pDisplayMode, UInt iOutput, UInt iAdapter )
{
    if ( iAdapter == INVALID_OFFSET )
        iAdapter = m_iAdapter;
    if ( iOutput == INVALID_OFFSET )
        iOutput = m_iOutput;

    UInt iMatchedDisplayMode = _MatchDisplayMode( iAdapter, iOutput, pDisplayMode );

    UInt iOutputIndex = ( iAdapter * D3D11WINDOW_MAX_OUTPUTS ) + iOutput;
    UInt iDisplayModeIndex = ( iOutputIndex * D3D11WINDOW_MAX_DISPLAYMODES ) + iMatchedDisplayMode;
    
    MemCopy( outMatchedDisplayMode, m_arrDisplayModes + iDisplayModeIndex, sizeof(D3D11DisplayModeDesc) );

    return iMatchedDisplayMode;
}

Void D3D11Window::SwitchAdapter( UInt iAdapter, UInt iOutput, UInt iDisplayMode )
{
    _SwitchAdapter( iAdapter, iOutput, iDisplayMode );
}
Void D3D11Window::SwitchOutput( UInt iOutput, UInt iDisplayMode )
{
    _SwitchOutput( iOutput, iDisplayMode );
}
Void D3D11Window::SwitchDisplayMode( UInt iDisplayMode )
{
    _SwitchDisplayMode( iDisplayMode );
}

Bool D3D11Window::IsFullScreen( UInt * outOutput ) const
{
    DebugAssert( m_pRenderer != NULL );
    DebugAssert( m_pRenderer->m_pSwapChain != NULL );
    if ( outOutput != NULL )
        *outOutput = INVALID_OFFSET;
    if ( m_bInDisplayModeSwitchTransition )
        return false;

    if ( outOutput != NULL )
        *outOutput = m_iOutput;
    return m_bFullScreen;
}
Void D3D11Window::SetFullScreen( Bool bFullScreen, UInt iOutput, UInt iDisplayMode )
{
    DebugAssert( m_pRenderer != NULL );
    DebugAssert( m_pRenderer->m_pSwapChain != NULL );
    if ( m_bInDisplayModeSwitchTransition )
        return;

    if ( m_bFullScreenWindowed )
        SetFullScreenWindowed( false );

    if ( m_bFullScreen == bFullScreen ) {
        if ( m_bFullScreen && iOutput != INVALID_OFFSET )
            _SwitchOutput( iOutput, iDisplayMode );
        return;
    }

    if ( iOutput != INVALID_OFFSET )
        _SelectOutput( iOutput, iDisplayMode );
    _ToggleFullScreen( bFullScreen );
}
Void D3D11Window::SetFullScreenWindowed( Bool bFullScreenWindowed )
{
    DebugAssert( m_pRenderer != NULL );
    DebugAssert( m_pRenderer->m_pSwapChain != NULL );
    if ( m_bInDisplayModeSwitchTransition )
        return;
    if ( m_bFullScreenWindowed == bFullScreenWindowed )
        return;

    // Switch display mode
    D3D11DisplayModeDesc hDesc, hFound;
    hDesc.iFormat = m_hDisplayModeDesc.iFormat;
    hDesc.iRefreshRateNumerator = m_hDisplayModeDesc.iRefreshRateNumerator;
    hDesc.iRefreshRateDenominator = m_hDisplayModeDesc.iRefreshRateDenominator;
    hDesc.iScanlineOrdering = D3D11DISPLAYMODE_SCANLINE_UNDEFINED;
    hDesc.iScaling = D3D11DISPLAYMODE_SCALING_UNDEFINED;

    if ( bFullScreenWindowed ) {
        // Ensure windowed mode
        if ( m_bFullScreen )
            SetFullScreen( false );

        m_iFullScreenWindowedSavedWidth = m_hDisplayModeDesc.iWidth;
        m_iFullScreenWindowedSavedHeight = m_hDisplayModeDesc.iHeight;

        hDesc.iWidth = GetSystemMetrics( SM_CXSCREEN );
        hDesc.iHeight = GetSystemMetrics( SM_CYSCREEN );
    } else {
        // Must be windowed mode
        DebugAssert( !m_bFullScreen );

        hDesc.iWidth = m_iFullScreenWindowedSavedWidth;
        hDesc.iHeight = m_iFullScreenWindowedSavedHeight;

        m_iFullScreenWindowedSavedWidth = 0;
        m_iFullScreenWindowedSavedHeight = 0;
    }

    UInt iDisplayMode = MatchDisplayMode( &hFound, &hDesc );
    DebugAssert( hFound.iWidth == hDesc.iWidth );
    DebugAssert( hFound.iHeight == hDesc.iHeight );
    DebugAssert( hFound.iFormat == hDesc.iFormat );
    DebugAssert( hFound.iRefreshRateNumerator == hDesc.iRefreshRateNumerator );
    DebugAssert( hFound.iRefreshRateDenominator == hDesc.iRefreshRateDenominator );

    _SwitchDisplayMode( iDisplayMode );

    // Update window style, position & topmost status
    if ( bFullScreenWindowed ) {
        RECT hRect;
        BOOL bOk = ::GetWindowRect( (HWND)m_hWindow, &hRect );
        DebugAssert( bOk );
        m_iFullScreenWindowedSavedX = hRect.left;
        m_iFullScreenWindowedSavedY = hRect.top;

        m_dwFullScreenWindowedSavedStyle = SetWindowLong( (HWND)m_hWindow, GWL_STYLE, WS_POPUP | WS_VISIBLE );
        DebugAssert( m_dwFullScreenWindowedSavedStyle != 0 );

        bOk = SetWindowPos( (HWND)m_hWindow, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOREDRAW );
        DebugAssert( bOk );
    } else {
        DWord dwPrevStyle = SetWindowLong( (HWND)m_hWindow, GWL_STYLE, m_dwFullScreenWindowedSavedStyle );
        DebugAssert( dwPrevStyle != 0 );

        BOOL bOk = SetWindowPos( (HWND)m_hWindow, HWND_TOP, m_iFullScreenWindowedSavedX, m_iFullScreenWindowedSavedY, 0, 0, SWP_NOSIZE | SWP_NOREDRAW );
        DebugAssert( bOk );

        m_dwFullScreenWindowedSavedStyle = 0;
        m_iFullScreenWindowedSavedX = 0;
        m_iFullScreenWindowedSavedY = 0;
    }

    // Done
    m_bFullScreenWindowed = bFullScreenWindowed;
}

Void D3D11Window::MoveWindow( UInt iX, UInt iY ) const
{
    DebugAssert( m_pRenderer != NULL );
    DebugAssert( m_pRenderer->m_pSwapChain != NULL );
    if ( m_bInDisplayModeSwitchTransition )
        return;
    DebugAssert( !m_bFullScreen );

    BOOL bRes = SetWindowPos( (HWND)m_hWindow, HWND_TOP, iX, iY, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOREDRAW );
    DebugAssert( bRes == TRUE );
}

Void D3D11Window::GetGammaCaps( D3D11GammaCaps * outGammaCaps ) const
{
    DebugAssert( m_pRenderer != NULL );
    DebugAssert( m_pRenderer->m_pSwapChain != NULL );
    if ( m_bInDisplayModeSwitchTransition )
        return;
    DebugAssert( m_bFullScreen );

    DXGI_GAMMA_CONTROL_CAPABILITIES dxgiGammaCaps;
    HRESULT hRes = ((IDXGIOutput*)m_pOutput)->GetGammaControlCapabilities( &dxgiGammaCaps );
    DebugAssert( hRes == S_OK );

    outGammaCaps->ConvertFrom( &dxgiGammaCaps );
}
Void D3D11Window::GetGammaControl( D3D11GammaControl * outGammaControl ) const
{
    DebugAssert( m_pRenderer != NULL );
    DebugAssert( m_pRenderer->m_pSwapChain != NULL );
    if ( m_bInDisplayModeSwitchTransition )
        return;
    DebugAssert( m_bFullScreen );

    DXGI_GAMMA_CONTROL dxgiGammaControl;
    HRESULT hRes = ((IDXGIOutput*)m_pOutput)->GetGammaControl( &dxgiGammaControl );
    DebugAssert( hRes == S_OK );

    outGammaControl->ConvertFrom( &dxgiGammaControl );
}
Void D3D11Window::SetGammaControl( const D3D11GammaControl * pGammaControl ) const
{
    DebugAssert( m_pRenderer != NULL );
    DebugAssert( m_pRenderer->m_pSwapChain != NULL );
    if ( m_bInDisplayModeSwitchTransition )
        return;
    DebugAssert( m_bFullScreen );

    DXGI_GAMMA_CONTROL dxgiGammaControl;
    pGammaControl->ConvertTo( &dxgiGammaControl );

    HRESULT hRes = ((IDXGIOutput*)m_pOutput)->SetGammaControl( &dxgiGammaControl );
    DebugAssert( hRes == S_OK );
}

Void D3D11Window::WaitForVSync() const
{
    DebugAssert( m_pOutput != NULL );

    HRESULT hRes = ((IDXGIOutput*)m_pOutput)->WaitForVBlank();
    DebugAssert( hRes == S_OK );
}

/////////////////////////////////////////////////////////////////////////////////

Void D3D11Window::_CreateWindow( UInt iX, UInt iY, UInt iWidth, UInt iHeight )
{
    DebugAssert( m_hInstance != NULL );
    DebugAssert( m_hWindow == NULL );

    // Default style
    DWord dwWindowStyle = ( WS_OVERLAPPED | WS_CAPTION | WS_MINIMIZEBOX ); 

    // Load default cursor
    m_hDefaultWindowCursor = LoadCursor( NULL, IDC_ARROW );
    m_hWindowCursor = m_hDefaultWindowCursor;

    // Window region
    RECT rectWindow;
    SetRect( &rectWindow, iX, iY, iX + iWidth, iY + iHeight );
    AdjustWindowRect( &rectWindow, dwWindowStyle, FALSE );

    // Window class
    WNDCLASSEX winClass;
    winClass.cbSize = sizeof(WNDCLASSEX);
    winClass.cbClsExtra = 0;
	winClass.cbWndExtra = 0;
    winClass.hInstance = (HINSTANCE)m_hInstance;
	winClass.style = CS_DBLCLKS;
	winClass.lpfnWndProc = (WNDPROC)_MsgProc_Static;
	winClass.lpszClassName = TEXT("D3D11Window_Class");
    winClass.lpszMenuName = NULL;
	winClass.hbrBackground = (HBRUSH)( GetStockObject(BLACK_BRUSH) );
	winClass.hIcon = LoadIcon( NULL, IDI_APPLICATION );
	winClass.hCursor = (HCURSOR)m_hWindowCursor;
	winClass.hIconSm = LoadIcon( NULL, IDI_APPLICATION );
	RegisterClassEx( &winClass );

    // Window creation
    m_hWindow = CreateWindowEx (
        0, TEXT("D3D11Window_Class"), m_strTitle, dwWindowStyle, iX, iY,
        (rectWindow.right - rectWindow.left), (rectWindow.bottom - rectWindow.top),
		NULL, NULL, (HINSTANCE)m_hInstance, (Void*)this
	);
    DebugAssert( m_hWindow != NULL );

    // Save window's monitor
    m_hWindowMonitor = MonitorFromWindow( (HWND)m_hWindow, MONITOR_DEFAULTTOPRIMARY );

    // Save window's cursor : already done

    // No menu allowed
    m_hWindowMenu = NULL;

    // Save window's state
    m_bActive = true;
    m_bMinimized = ( IsIconic((HWND)m_hWindow) == TRUE );
    m_bMaximized = ( IsZoomed((HWND)m_hWindow) == TRUE );
    m_bIsInSizeMove = false;
}
Void D3D11Window::_DestroyWindow()
{
    DebugAssert( m_hWindow != NULL );

    // Clear window's state
    m_bActive = false;
    m_bMinimized = false;
    m_bMaximized = false;
    m_bIsInSizeMove = false;

    // No menu allowed
    m_hWindowMenu = NULL;

    // Clear window's cursor : nothing to do

    // Clear window's monitor
    m_hWindowMonitor = NULL;

    // Destroy window
    BOOL bRes = DestroyWindow( (HWND)m_hWindow );
    DebugAssert( bRes == TRUE );
    
    bRes = UnregisterClass( TEXT("D3D11Window_Class"), NULL );
    DebugAssert( bRes == TRUE );

    m_hWindow = NULL;

    // Destroy window's cursor
    if ( m_hWindowCursor != m_hDefaultWindowCursor )
        DestroyCursor( (HCURSOR)m_hWindowCursor );
    m_hWindowCursor = NULL;
    m_hDefaultWindowCursor = NULL;
}

Int64 __stdcall D3D11Window::_MsgProc_Static( Void * hWindow, UInt iMsg, UInt64 iWParam, Int64 iLParam )
{
    static D3D11Window * s_pThis = NULL;
    if ( s_pThis != NULL )
        return s_pThis->_MsgProc_Virtual( hWindow, iMsg, iWParam, iLParam );

    if ( iMsg == WM_CREATE )
        s_pThis = (D3D11Window *)(((LPCREATESTRUCT)iLParam)->lpCreateParams);

    return DefWindowProc( (HWND)hWindow, iMsg, iWParam, iLParam );
}
Int64 D3D11Window::_MsgProc_Virtual( Void * hWindow, UInt iMsg, UInt64 iWParam, Int64 iLParam )
{
    switch( iMsg ) {
        // Keyboard messages
        case WM_SYSKEYDOWN:
        case WM_KEYDOWN: {
                // Handle key press
                KeyCode iKey = KeyCodeFromWin32[iWParam & 0xff];
                if ( m_pCallbacks->OnKeyPress(iKey) )
                    return 0;
            } break;
        case WM_SYSKEYUP:
        case WM_KEYUP: {
                // Handle key release
                KeyCode iKey = KeyCodeFromWin32[iWParam & 0xff];
                if ( m_pCallbacks->OnKeyRelease(iKey) )
                    return 0;
            } break;

        // Mouse messages
        case WM_MOUSEMOVE: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                m_pCallbacks->OnMouseMove( iX, iY );
                return 0;
            } break;
        case WM_MOUSEWHEEL: {
                POINT pt; // WM_MOUSEWHEEL needs screen->client conversion
                pt.x = (UInt)(LOWORD(iLParam));
                pt.y = (UInt)(HIWORD(iLParam));
                ScreenToClient( (HWND)m_hWindow, &pt );
                Int iWheelDelta = (Int)( (Short)(HIWORD(iWParam)) ) / WHEEL_DELTA;
                if ( m_pCallbacks->OnMouseWheel((UInt)pt.x, (UInt)pt.y, iWheelDelta) )
                    return 0;
            } break;
        case WM_LBUTTONDOWN: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                if ( m_pCallbacks->OnMousePress(iX, iY, KEYCODE_MOUSELEFT) )
                    return 0;
            } break;
        case WM_RBUTTONDOWN: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                if ( m_pCallbacks->OnMousePress(iX, iY, KEYCODE_MOUSERIGHT) )
                    return 0;
            } break;
        case WM_MBUTTONDOWN: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                if ( m_pCallbacks->OnMousePress(iX, iY, KEYCODE_MOUSEMIDDLE) )
                    return 0;
            } break;
        case WM_XBUTTONDOWN: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                KeyCode iButton = KeyCodeFromWin32[iWParam & 0xff];
                if ( m_pCallbacks->OnMousePress(iX, iY, iButton) )
                    return 0;
            } break;
        case WM_LBUTTONUP: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                if ( m_pCallbacks->OnMouseRelease(iX, iY, KEYCODE_MOUSELEFT) )
                    return 0;
            } break;
        case WM_RBUTTONUP: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                if ( m_pCallbacks->OnMouseRelease(iX, iY, KEYCODE_MOUSERIGHT) )
                    return 0;
            } break;
        case WM_MBUTTONUP: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                if ( m_pCallbacks->OnMouseRelease(iX, iY, KEYCODE_MOUSEMIDDLE) )
                    return 0;
            } break;
        case WM_XBUTTONUP: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                KeyCode iButton = KeyCodeFromWin32[iWParam & 0xff];
                if ( m_pCallbacks->OnMouseRelease(iX, iY, iButton) )
                    return 0;
            } break;
        case WM_LBUTTONDBLCLK: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                if ( m_pCallbacks->OnMouseDblClick(iX, iY, KEYCODE_MOUSELEFT) )
                    return 0;
            } break;
        case WM_RBUTTONDBLCLK: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                if ( m_pCallbacks->OnMouseDblClick(iX, iY, KEYCODE_MOUSERIGHT) )
                    return 0;
            } break;
        case WM_MBUTTONDBLCLK: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                if ( m_pCallbacks->OnMouseDblClick(iX, iY, KEYCODE_MOUSEMIDDLE) )
                    return 0;
            } break;
        case WM_XBUTTONDBLCLK: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                KeyCode iButton = KeyCodeFromWin32[iWParam & 0xff];
                if ( m_pCallbacks->OnMouseDblClick(iX, iY, iButton) )
                    return 0;
            } break;

        // Activation messages
        case WM_ACTIVATEAPP: {
                if ( !m_bActive && iWParam == TRUE ) {
                    m_bActive = true;
                    m_pCallbacks->OnResume();
                } else if ( m_bActive && iWParam == FALSE ) {
                    m_bActive = false;
                    m_pCallbacks->OnPause();
                }
            } break;
        case WM_ACTIVATE: {
                if ( LOWORD(iWParam) == WA_ACTIVE || LOWORD(iWParam) == WA_CLICKACTIVE ) {
                    if ( HIWORD(iWParam) == 0 ) // not minimized
                        m_pCallbacks->OnWindowFocus();
                }
            } break;
        //case WM_MOUSEACTIVATE: {
        //        return MA_ACTIVATEANDEAT;
        //    } break;

        // Paint message
        case WM_PAINT: {
                m_pCallbacks->OnWindowDraw();
            } break;

        // Moving / Sizing messages
        case WM_ENTERSIZEMOVE: {
                m_bIsInSizeMove = true;
            } break;
        case WM_EXITSIZEMOVE: {
                m_bIsInSizeMove = false;
            } break;
        case WM_MOVE: {
                UInt iX = (UInt)(LOWORD(iLParam));
                UInt iY = (UInt)(HIWORD(iLParam));
                m_pCallbacks->OnWindowMove( iX, iY );

                // Check for a continuous move
                if ( m_bIsInSizeMove ) {
                    // Don't update anything yet ...
                    break;
                }

                // Update SwapChain & output/monitor
                if ( m_pRenderer != NULL && m_pRenderer->m_pSwapChain != NULL ) {
                    if ( m_bInDisplayModeSwitchTransition ) {
                        m_pRenderer->m_pCallbacks->OnDestroySwapChain();

                        HRESULT hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->ResizeBuffers( 0, m_hDisplayModeDesc.iWidth, m_hDisplayModeDesc.iHeight,
                                                                                                      DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH );
                        DebugAssert( hRes == S_OK );

                        m_pRenderer->_UpdateSwapChainDesc();
                        m_bFullScreen = !(m_pRenderer->m_hSwapChainDesc.bWindowed);

                        m_pRenderer->m_pCallbacks->OnRestoreSwapChain();

                        // Check SwapChain's mode matches our selection
                        DebugAssert( m_hDisplayModeDesc.iScanlineOrdering == m_pRenderer->m_hSwapChainDesc.iScanlineOrdering );
                        DebugAssert( m_hDisplayModeDesc.iScaling == m_pRenderer->m_hSwapChainDesc.iScaling );
                        DebugAssert( m_hDisplayModeDesc.iFormat == m_pRenderer->m_hSwapChainDesc.iFormat );
                        DebugAssert( m_hDisplayModeDesc.iWidth == m_pRenderer->m_hSwapChainDesc.iWidth );
                        DebugAssert( m_hDisplayModeDesc.iHeight == m_pRenderer->m_hSwapChainDesc.iHeight );
                        //DebugAssert( m_hDisplayModeDesc.iRefreshRateNumerator == m_pRenderer->m_hSwapChainDesc.iRefreshRateNumerator );
                        //DebugAssert( m_hDisplayModeDesc.iRefreshRateDenominator == m_pRenderer->m_hSwapChainDesc.iRefreshRateDenominator );
                    }
                    if ( !m_bFullScreen ) {
                        if ( _UpdateOutputFromWindow() ) {
                            m_hWindowMonitor = (HMONITOR)(m_hOutputDesc.pMonitor);
                            m_pCallbacks->OnWindowMonitorChange( m_hWindowMonitor );
                        }
                    }
                } else {
                    HMONITOR hNewMonitor = MonitorFromWindow( (HWND)m_hWindow, MONITOR_DEFAULTTOPRIMARY );
                    if ( m_hWindowMonitor != hNewMonitor ) {
                        m_hWindowMonitor = hNewMonitor;
                        m_pCallbacks->OnWindowMonitorChange( m_hWindowMonitor );
                    }
                }
            } break;
        case WM_SIZE: {
                UInt iWidth = (UInt)(LOWORD(iLParam));
                UInt iHeight = (UInt)(HIWORD(iLParam));
                m_pCallbacks->OnWindowResize( iWidth, iHeight );

                // Update window's state
                if ( iWParam == SIZE_MINIMIZED ) {
                    if ( !m_bMinimized )
                        m_pCallbacks->OnPause();
                    m_bMinimized = true;
                    m_bMaximized = false;
                } else if ( iWParam == SIZE_MAXIMIZED ) {
                    if ( m_bMinimized )
                        m_pCallbacks->OnResume();
                    m_bMinimized = false;
                    m_bMaximized = true;
                } else if ( iWParam == SIZE_RESTORED ) {
                    if ( m_bMinimized )
                        m_pCallbacks->OnResume();
                    m_bMinimized = false;
                    m_bMaximized = false;
                }

                // Check for a continuous resize
                if ( m_bIsInSizeMove ) {
                    // Don't update anything yet ...
                    // Should never happen since we never give sizing borders
                    break;
                }

                // Update SwapChain & output/monitor
                if ( m_pRenderer != NULL && m_pRenderer->m_pSwapChain != NULL ) {
                    if ( m_bInDisplayModeSwitchTransition ) {
                        m_pRenderer->m_pCallbacks->OnDestroySwapChain();

                        HRESULT hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->ResizeBuffers( 0, m_hDisplayModeDesc.iWidth, m_hDisplayModeDesc.iHeight,
                                                                                                      DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH );
                        DebugAssert( hRes == S_OK );

                        m_pRenderer->_UpdateSwapChainDesc();
                        m_bFullScreen = !(m_pRenderer->m_hSwapChainDesc.bWindowed);

                        m_pRenderer->m_pCallbacks->OnRestoreSwapChain();

                        // Check SwapChain's mode matches our selection
                        DebugAssert( m_hDisplayModeDesc.iScanlineOrdering == m_pRenderer->m_hSwapChainDesc.iScanlineOrdering );
                        DebugAssert( m_hDisplayModeDesc.iScaling == m_pRenderer->m_hSwapChainDesc.iScaling );
                        DebugAssert( m_hDisplayModeDesc.iFormat == m_pRenderer->m_hSwapChainDesc.iFormat );
                        DebugAssert( m_hDisplayModeDesc.iWidth == m_pRenderer->m_hSwapChainDesc.iWidth );
                        DebugAssert( m_hDisplayModeDesc.iHeight == m_pRenderer->m_hSwapChainDesc.iHeight );
                        //DebugAssert( m_hDisplayModeDesc.iRefreshRateNumerator == m_pRenderer->m_hSwapChainDesc.iRefreshRateNumerator );
                        //DebugAssert( m_hDisplayModeDesc.iRefreshRateDenominator == m_pRenderer->m_hSwapChainDesc.iRefreshRateDenominator );
                    }
                    if ( !m_bFullScreen ) {
                        if ( _UpdateOutputFromWindow() ) {
                            m_hWindowMonitor = (HMONITOR)(m_hOutputDesc.pMonitor);
                            m_pCallbacks->OnWindowMonitorChange( m_hWindowMonitor );
                        }
                    }
                } else {
                    HMONITOR hNewMonitor = MonitorFromWindow( (HWND)m_hWindow, MONITOR_DEFAULTTOPRIMARY );
                    if ( m_hWindowMonitor != hNewMonitor ) {
                        m_hWindowMonitor = hNewMonitor;
                        m_pCallbacks->OnWindowMonitorChange( m_hWindowMonitor );
                    }
                }

                // Done
                return 1;
            } break;

        // Menu messages
        case WM_ENTERMENULOOP: {
                // nothing to do
            } break;
        case WM_EXITMENULOOP: {
                // nothing to do
            } break;
        case WM_MENUCHAR: {
                // A menu is active and the user presses an unbinded key
                return MAKELRESULT( 0, MNC_CLOSE );
            } break;
        case WM_NCHITTEST: {
                // Prevent the user from selecting the menu in fullscreen mode
                if ( !(m_pRenderer->m_hSwapChainDesc.bWindowed) )
                    return HTCLIENT;
            } break;

        // Misc messages
        case WM_SETCURSOR: {
                if ( m_bActive && !(m_pRenderer->m_hSwapChainDesc.bWindowed) ) {
                    // Let us handle the cursor in fullscreen
                    return 1;
                }
            } break;
        case WM_GETMINMAXINFO: {
                // Retrieve Min-Max infos
                ( (MINMAXINFO*)iLParam )->ptMinTrackSize.x = D3D11WINDOW_MIN_WIDTH;
                ( (MINMAXINFO*)iLParam )->ptMinTrackSize.y = D3D11WINDOW_MIN_HEIGHT;
            } break;
        case WM_ERASEBKGND: {
                // Don't clear background
                return 1;
            } break;
        case WM_SYSCOMMAND: {
                UInt iSysCmd = ( iWParam & 0xFFF0 );
	            // Prevent screensaver or monitor powersave mode from starting
	            if ( iSysCmd == SC_SCREENSAVE || iSysCmd == SC_MONITORPOWER )
		            return 0;
                // Prevent moving / sizing / menu in fullscreen
                if ( !(m_pRenderer->m_hSwapChainDesc.bWindowed) ) {
                    if ( iSysCmd == SC_MOVE || iSysCmd == SC_SIZE || iSysCmd == SC_MAXIMIZE || iSysCmd == SC_KEYMENU )
		                return 0;
                }
            } break;

        // Exit sequence
        case WM_CLOSE: {
                PostQuitMessage( 0 );
                return 0;
            } break;
        case WM_DESTROY: {
                // Window gets destroyed, nothing to do
            } break;
        case WM_QUIT: {
                // Message loop exit-case, never goes here
            } break;
        default: break;
    }

    // Message wasn't handled by application
    return DefWindowProc( (HWND)hWindow, iMsg, iWParam, iLParam );
}

Void * D3D11Window::_GetAdapter( UInt iAdapter, D3D11AdapterDesc * outDesc ) const
{
    DebugAssert( m_pDXGIFactory != NULL );

    IDXGIAdapter * pAdapter = NULL;
    HRESULT hRes = ((IDXGIFactory*)m_pDXGIFactory)->EnumAdapters( iAdapter, &pAdapter );
    if ( hRes == DXGI_ERROR_NOT_FOUND ) {
        if ( pAdapter != NULL ) {
            pAdapter->Release();
            pAdapter = NULL;
        }
        return NULL;
    }
    DebugAssert( hRes == S_OK && pAdapter != NULL );

    if ( outDesc != NULL ) {
        DXGI_ADAPTER_DESC hDXGIDesc;
        hRes = pAdapter->GetDesc( &hDXGIDesc );
        DebugAssert( hRes == S_OK );

        outDesc->ConvertFrom( &hDXGIDesc, iAdapter );
    }

    return pAdapter;
}
Void * D3D11Window::_GetOutput( Void * pAdapter, UInt iOutput, D3D11OutputDesc * outDesc ) const
{
    DebugAssert( m_pDXGIFactory != NULL );

    IDXGIOutput * pOutput = NULL;
    HRESULT hRes = ((IDXGIAdapter*)pAdapter)->EnumOutputs( iOutput, &pOutput );
    if ( hRes == DXGI_ERROR_NOT_FOUND ) {
        if ( pOutput != NULL ) {
            pOutput->Release();
            pOutput = NULL;
        }
        return NULL;
    }
    DebugAssert( hRes == S_OK && pOutput != NULL );

    if ( outDesc != NULL ) {
        DXGI_OUTPUT_DESC hDXGIDesc;
        hRes = pOutput->GetDesc( &hDXGIDesc );
        DebugAssert( hRes == S_OK );

        outDesc->ConvertFrom( &hDXGIDesc, iOutput );
    }

    return pOutput;
}
UInt D3D11Window::_FilterDisplayModes( D3D11DisplayModeDesc * arrDisplayModes, UInt iCount ) const
{
    static Float s_fRegularScreen = 4.0f / 3.0f;
    static Float s_fWideScreen = 16.0f / 9.0f;
    static Float s_arrRefreshRates[3] = { 59.0f, 60.0f, 75.0f };

    UInt iFilteredCount = 0;
    for( UInt i = 0; i < iCount; ++i ) {
        // Remove any unwanted scanline ordering
        if ( arrDisplayModes[i].iScanlineOrdering != D3D11DISPLAYMODE_SCANLINE_PROGRESSIVE )
            continue;

        // Remove centered & stretched modes (ugly)
        if ( arrDisplayModes[i].iScaling != D3D11DISPLAYMODE_SCALING_UNDEFINED )
            continue;

        // Format is already filtered by caller (_Enumerate)

        // Remove any non-standard aspect ratio
        Float fAspectRatio = ( ((Float)(arrDisplayModes[i].iWidth)) / (Float)(arrDisplayModes[i].iHeight) );
        Float fTest = ( fAspectRatio <= s_fRegularScreen ) ? (s_fRegularScreen - fAspectRatio) : (fAspectRatio - s_fRegularScreen);
        if ( fTest > 1.0e-3 ) {
            fTest = ( fAspectRatio <= s_fWideScreen ) ? (s_fWideScreen - fAspectRatio) : (fAspectRatio - s_fWideScreen);
            if ( fTest > 1.0e-3 )
               continue;
        }

        // Remove any resolution below 800x600
        if ( arrDisplayModes[i].iWidth < 800 || arrDisplayModes[i].iHeight < 600 )
            continue;

        // Remove any non-standard refresh rate
        Float fRefreshRate = ( ((Float)(arrDisplayModes[i].iRefreshRateNumerator)) / (Float)(arrDisplayModes[i].iRefreshRateDenominator) );
        fTest = ( fRefreshRate <= s_arrRefreshRates[0] ) ? (s_arrRefreshRates[0] - fRefreshRate) : (fRefreshRate - s_arrRefreshRates[0]);
        if ( fTest > 0.5f ) {
            fTest = ( fRefreshRate <= s_arrRefreshRates[1] ) ? (s_arrRefreshRates[1] - fRefreshRate) : (fRefreshRate - s_arrRefreshRates[1]);
            if ( fTest > 0.5f ) {
                fTest = ( fRefreshRate <= s_arrRefreshRates[2] ) ? (s_arrRefreshRates[2] - fRefreshRate) : (fRefreshRate - s_arrRefreshRates[2]);
                if ( fTest > 0.5f )
                    continue;
            }
        }

        // Passed !
        if ( i > iFilteredCount )
            MemCopy( arrDisplayModes + iFilteredCount, arrDisplayModes + i, sizeof(D3D11DisplayModeDesc) );
        ++iFilteredCount;
    }

    return iFilteredCount;
}
Int D3D11Window::_CompareDisplayModes( const D3D11DisplayModeDesc * pLeft, const D3D11DisplayModeDesc * pRight ) const
{
    if ( pLeft->iScanlineOrdering < pRight->iScanlineOrdering )
        return +1;
    if ( pLeft->iScanlineOrdering > pRight->iScanlineOrdering )
        return -1;

    if ( pLeft->iScaling < pRight->iScaling )
        return +1;
    if ( pLeft->iScaling > pRight->iScaling )
        return -1;

    //if ( pLeft->Format < pRight->Format )
    //    return +1;
    //if ( pLeft->Format > pRight->Format )
    //    return -1;

    if ( pLeft->iWidth < pRight->iWidth )
        return +1;
    if ( pLeft->iWidth > pRight->iWidth )
        return -1;
    if ( pLeft->iHeight < pRight->iHeight )
        return +1;
    if ( pLeft->iHeight > pRight->iHeight )
        return -1;

    Float fRefreshRateLeft = ( ((Float)(pLeft->iRefreshRateNumerator)) / (Float)(pLeft->iRefreshRateDenominator) );
    Float fRefreshRateRight = ( ((Float)(pRight->iRefreshRateNumerator)) / (Float)(pRight->iRefreshRateDenominator) );
    if ( fRefreshRateLeft < fRefreshRateRight )
        return +1;
    if ( fRefreshRateLeft > fRefreshRateRight )
        return -1;

    return 0;
}
Void D3D11Window::_SortDisplayModes( D3D11DisplayModeDesc * outDisplayModes, const D3D11DisplayModeDesc * arrDisplayModes, UInt iCount ) const
{
    for( UInt i = 0; i < iCount; ++i ) {
        // Pick first element
        const D3D11DisplayModeDesc * pDisplayMode = ( arrDisplayModes + i );

        // Sorted insertion
        UInt iInsert;
        for( iInsert = 0; iInsert < i; ++iInsert ) {
            Int iCmp = _CompareDisplayModes( outDisplayModes + iInsert, pDisplayMode );
            if ( iCmp < 0 ) {
                UInt iShiftCount = ( i - iInsert );
                for( UInt k = 0; k < iShiftCount; ++k )
                    MemCopy( outDisplayModes + i - k, outDisplayModes + i - k - 1, sizeof(D3D11DisplayModeDesc) );
                MemCopy( outDisplayModes + iInsert, pDisplayMode, sizeof(D3D11DisplayModeDesc) );
                break;
            }
        }
        if ( iInsert == i )
            MemCopy( outDisplayModes + iInsert, pDisplayMode, sizeof(D3D11DisplayModeDesc) );
    }
}
UInt D3D11Window::_MatchDisplayMode( UInt iAdapter, UInt iOutput, const D3D11DisplayModeDesc * pDesc ) const
{
    // Aspect ratios
    static Float s_fRegularScreen = 4.0f / 3.0f;
    static Float s_fWideScreen = 16.0f / 9.0f;

    // Precompute aspect ratio & refresh rate
    Float fAspectRatio = 0.0f;
    if ( pDesc->iWidth != 0 && pDesc->iHeight != 0 )
        fAspectRatio = ( ((Float)(pDesc->iWidth)) / (Float)(pDesc->iHeight) );

    Float fRefreshRate = 0.0f;
    if ( pDesc->iRefreshRateNumerator != 0 && pDesc->iRefreshRateDenominator != 0 )
        fRefreshRate = ( ((Float)(pDesc->iRefreshRateNumerator)) / (Float)(pDesc->iRefreshRateDenominator) );

    // Precompute best mark that can be reach given input
    DWord iBestMarkAvailable = ( pDesc->iScanlineOrdering != D3D11DISPLAYMODE_SCANLINE_UNDEFINED ) ? 0x0e : 0x04;
    iBestMarkAvailable <<= 4;
    iBestMarkAvailable += ( pDesc->iScaling != D3D11DISPLAYMODE_SCALING_UNDEFINED ) ? 0x0e : 0x04;
    iBestMarkAvailable <<= 4;
    iBestMarkAvailable += ( pDesc->iFormat != PIXEL_FMT_UNKNOWN ) ? 0x0e : 0x06;
    iBestMarkAvailable <<= 4;
    iBestMarkAvailable += ( fAspectRatio != 0.0f ) ? 0x0e : 0x06;
    iBestMarkAvailable <<= 4;
    iBestMarkAvailable += ( pDesc->iWidth != 0.0f ) ? 0x07 : 0x05;
    iBestMarkAvailable += ( pDesc->iHeight != 0.0f ) ? 0x07 : 0x05;
    iBestMarkAvailable <<= 4;
    iBestMarkAvailable += ( fRefreshRate != 0.0f ) ? 0x0e : 0x06;

    // Check all display modes for current adapter/output
    UInt iAdapterIndex = iAdapter;
    UInt iOutputIndex = ( iAdapterIndex * D3D11WINDOW_MAX_OUTPUTS ) + iOutput;
    UInt iDisplayModeIndex = ( iOutputIndex * D3D11WINDOW_MAX_DISPLAYMODES );
    UInt iCount = m_arrDisplayModeCounts[iOutputIndex];

    UInt iBest = INVALID_OFFSET;
    DWord iBestMark = 0;

    for( UInt i = 0; i < iCount; ++i ) {
        const D3D11DisplayModeDesc * pCurrent = ( m_arrDisplayModes + iDisplayModeIndex + i );
        DWord iCurrentMark = 0;

        // Rate scanline ordering
        if ( pDesc->iScanlineOrdering != D3D11DISPLAYMODE_SCANLINE_UNDEFINED ) {
            if ( pCurrent->iScanlineOrdering == pDesc->iScanlineOrdering )
                iCurrentMark += 0x0e;
        } else {
            if ( pCurrent->iScanlineOrdering == D3D11DISPLAYMODE_SCANLINE_PROGRESSIVE )
                iCurrentMark += 0x04;
            else
                iCurrentMark += 0x02;
        }

        iCurrentMark <<= 4;

        // Rate scaling mode
        if ( pDesc->iScaling != D3D11DISPLAYMODE_SCALING_UNDEFINED ) {
            if ( pCurrent->iScaling == pDesc->iScaling )
                iCurrentMark += 0x0e;
        } else {
            if ( pCurrent->iScaling == D3D11DISPLAYMODE_SCALING_CENTERED )
                iCurrentMark += 0x04;
            else
                iCurrentMark += 0x02;
        }

        iCurrentMark <<= 4;

        // Rate format
        if ( pDesc->iFormat != PIXEL_FMT_UNKNOWN ) {
            if ( pCurrent->iFormat == pDesc->iFormat )
                iCurrentMark += 0x0e;
        } else {
            if ( pCurrent->iFormat == PIXEL_FMT_RGBA8UN )
                iCurrentMark += 0x06;
            else if ( pCurrent->iFormat == PIXEL_FMT_BGRA8UN )
                iCurrentMark += 0x04;
            else
                iCurrentMark += 0x02;
        }

        iCurrentMark <<= 4;

        // Rate aspect ratio
        Float fCurrentAspectRatio = ( ((Float)(pCurrent->iWidth)) / (Float)(pCurrent->iHeight) );
        if ( fAspectRatio != 0.0f ) {
            Float fTest = (fCurrentAspectRatio <= fAspectRatio) ? (fAspectRatio - fCurrentAspectRatio) : (fCurrentAspectRatio - fAspectRatio);
            if ( fTest < 1.0e-3 )
                iCurrentMark += 0x0e;
        } else {
            Float fTest = (fCurrentAspectRatio <= s_fWideScreen) ? (s_fWideScreen - fCurrentAspectRatio) : (fCurrentAspectRatio - s_fWideScreen);
            if ( fTest < 1.0e-3 )
                iCurrentMark += 0x06;
            else {
                fTest = (fCurrentAspectRatio <= s_fRegularScreen) ? (s_fRegularScreen - fCurrentAspectRatio) : (fCurrentAspectRatio - s_fRegularScreen);
                if ( fTest < 1.0e-3 )
                    iCurrentMark += 0x04;
                else
                    iCurrentMark += 0x02;
            }
        }

        iCurrentMark <<= 4;

        // Rate resolution
        if ( pDesc->iWidth != 0 ) {
            if ( pCurrent->iWidth == pDesc->iWidth )
                iCurrentMark += 0x07;
            else if ( pCurrent->iWidth < pDesc->iWidth ) {
                Float fDelta = ( ((Float)(pCurrent->iWidth)) / (Float)(pDesc->iWidth) ); // [0,MaxW[ => [0,1[
                iCurrentMark += 0x01 + (DWord)( fDelta * (Float)0x06 );                  // [0,1[ => [0x01,0x07[
            }
        } else {
            if ( pCurrent->iWidth == 1920 )
                iCurrentMark += 0x05;
            else if ( pCurrent->iWidth == 1600 )
                iCurrentMark += 0x04;
            else if ( pCurrent->iWidth == 1280 )
                iCurrentMark += 0x03;
            else if ( pCurrent->iWidth == 1024 )
                iCurrentMark += 0x02;
            else
                iCurrentMark += 0x01;
        }

        if ( pDesc->iHeight != 0 ) {
            if ( pCurrent->iHeight == pDesc->iHeight )
                iCurrentMark += 0x07;
            else if ( pCurrent->iHeight < pDesc->iHeight ) {
                Float fDelta = ( ((Float)(pCurrent->iHeight)) / (Float)(pDesc->iHeight) ); // [0,MaxH[ => [0,1[
                iCurrentMark += 0x01 + (DWord)( fDelta * (Float)0x06 );                    // [0,1[ => [0x01,0x07[
            }
        } else {
            if ( pCurrent->iHeight == 1080 )
                iCurrentMark += 0x05;
            else if ( pCurrent->iHeight == 900 )
                iCurrentMark += 0x04;
            else if ( pCurrent->iHeight == 960 )
                iCurrentMark += 0x03;
            else if ( pCurrent->iHeight == 768 )
                iCurrentMark += 0x02;
            else
                iCurrentMark += 0x01;
        }

        iCurrentMark <<= 4;

        // Rate refresh rate
        Float fCurrentRefreshRate = ( ((Float)(pCurrent->iRefreshRateNumerator)) / (Float)(pCurrent->iRefreshRateDenominator) );
        if ( fRefreshRate != 0.0f ) {
            Float fTest = (fCurrentRefreshRate <= fRefreshRate) ? (fRefreshRate - fCurrentRefreshRate) : (fCurrentRefreshRate - fRefreshRate);
            if ( fTest < 0.5f )
                iCurrentMark += 0x0e;
        } else {
            Float fTest = (fCurrentRefreshRate <= 60.0f) ? (60.0f - fCurrentRefreshRate) : (fCurrentRefreshRate - 60.0f);
            if ( fTest < 0.5f )
                iCurrentMark += 0x06;
            else {
                fTest = (fCurrentRefreshRate <= 59.0f) ? (59.0f - fCurrentRefreshRate) : (fCurrentRefreshRate - 59.0f);
                if ( fTest < 0.5f )
                    iCurrentMark += 0x04;
                else
                    iCurrentMark += 0x02;
            }
        }

        // Check if we have a better mark than before
        if ( iCurrentMark >= iBestMark ) {
            iBest = i;
            iBestMark = iCurrentMark;
        }

        // Check if we found a perfect match
        if ( iBestMark == iBestMarkAvailable )
            break;
    }

    DebugAssert( iBest != INVALID_OFFSET );
    return iBest;
}

Void D3D11Window::_Enumerate()
{
    DebugAssert( m_iAdapterCount == 0 );

    UInt iAdapterIndex, iOutputIndex, iDisplayModeIndex;
    HRESULT hRes;

    // Display formats (by preference)
    static const UInt s_iDisplayModeFormatCount = 3;
    static const DXGI_FORMAT s_arrDisplayModeFormats[s_iDisplayModeFormatCount] = {
        DXGI_FORMAT_R8G8B8A8_UNORM,      // Allows Flip-Sequential SwapChains,
        DXGI_FORMAT_B8G8R8A8_UNORM,      // (DXGI 1.2+ only) but you have to pass
        DXGI_FORMAT_R16G16B16A16_FLOAT   // on multisampling.
//        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
//        DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
//        DXGI_FORMAT_R10G10B10A2_UNORM,
//        DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM // No multisampling allowed
    };
    DXGI_FORMAT iDisplayModeFormat;
    UInt iDisplayModeFlags = 0; // DXGI_ENUM_MODES_INTERLACED | DXGI_ENUM_MODES_SCALING | DXGI_ENUM_MODES_STEREO
    DXGI_MODE_DESC arrTmpDisplayModes[D3D11WINDOW_MAX_DISPLAYMODES];
    D3D11DisplayModeDesc arrFoundDisplayModes[D3D11WINDOW_MAX_DISPLAYMODES];

    // Enumerate adapters
    UInt iAdapter = 0;
    m_iAdapterCount = 0;
    while( m_iAdapterCount < D3D11WINDOW_MAX_ADAPTERS ) {
        // Get current adapter
        iAdapterIndex = m_iAdapterCount;
        IDXGIAdapter * pAdapter = (IDXGIAdapter*)_GetAdapter( iAdapter, m_arrAdapters + iAdapterIndex );
        if ( pAdapter == NULL )
            break;

        // Check adapter support : try to create a device, IDXGIAdapter::CheckInterfaceSupport is deprecated (DX10 only)
        D3D_FEATURE_LEVEL iActualFeatureLevel, iFeatureLevel = D3D_FEATURE_LEVEL_11_0;
        ID3D11Device * pTmpDevice = NULL;
        ID3D11DeviceContext * pTmpImmediateContext = NULL;
        hRes = D3D11CreateDevice( pAdapter, D3D_DRIVER_TYPE_UNKNOWN, NULL, D3D11_CREATE_DEVICE_BGRA_SUPPORT, &iFeatureLevel, 1,
                                  D3D11_SDK_VERSION, &pTmpDevice, &iActualFeatureLevel, &pTmpImmediateContext );
        if ( hRes == DXGI_ERROR_UNSUPPORTED ) {
            // Next adapter
            pAdapter->Release();
            pAdapter = NULL;
            ++iAdapter;
            continue;
        }
        DebugAssert( hRes == S_OK && pTmpDevice != NULL && pTmpImmediateContext != NULL && iActualFeatureLevel >= iFeatureLevel );
        pTmpImmediateContext->Release();
        pTmpImmediateContext = NULL;
        pTmpDevice->Release();
        pTmpDevice = NULL;

        // Store adapter
        ++m_iAdapterCount;

        // Enumerate outputs
        UInt iOutput = 0;
        m_arrOutputCounts[iAdapterIndex] = 0;
        while( m_arrOutputCounts[iAdapterIndex] < D3D11WINDOW_MAX_OUTPUTS ) {
            // Get current output
            iOutputIndex = ( iAdapterIndex * D3D11WINDOW_MAX_OUTPUTS ) + m_arrOutputCounts[iAdapterIndex];
            IDXGIOutput * pOutput = (IDXGIOutput*)_GetOutput( pAdapter, iOutput, m_arrOutputs + iOutputIndex );
            if ( pOutput == NULL )
                break;

            // Check output support

            // Store output
            ++(m_arrOutputCounts[iAdapterIndex]);

            // Enumerate display modes
            UInt iCount;
            m_arrDisplayModeCounts[iOutputIndex] = 0;
            for( UInt i = 0; i < s_iDisplayModeFormatCount; ++i ) {
                // Get current display mode format
                iDisplayModeFormat = s_arrDisplayModeFormats[i];

                // Get display mode list
                iCount = 0;
                hRes = pOutput->GetDisplayModeList( iDisplayModeFormat, iDisplayModeFlags, &iCount, NULL );
                DebugAssert( hRes == S_OK );
                if ( iCount == 0 )
                    continue;
                DebugAssert( iCount <= D3D11WINDOW_MAX_DISPLAYMODES );

                hRes = pOutput->GetDisplayModeList( iDisplayModeFormat, iDisplayModeFlags, &iCount, arrTmpDisplayModes );
                DebugAssert( hRes == S_OK );

                for ( UInt j = 0; j < iCount; ++j )
                    arrFoundDisplayModes[j].ConvertFrom( arrTmpDisplayModes + j, j );

                // Filter display mode list
                iCount = _FilterDisplayModes( arrFoundDisplayModes, iCount );

                // Sort & Store display mode list
                DebugAssert( (m_arrDisplayModeCounts[iOutputIndex] + iCount) <= D3D11WINDOW_MAX_DISPLAYMODES );

                iDisplayModeIndex = ( iOutputIndex * D3D11WINDOW_MAX_DISPLAYMODES ) + m_arrDisplayModeCounts[iOutputIndex];
                _SortDisplayModes( m_arrDisplayModes + iDisplayModeIndex, arrFoundDisplayModes, iCount );
                m_arrDisplayModeCounts[iOutputIndex] += iCount;
            }
            DebugAssert( m_arrDisplayModeCounts[iOutputIndex] > 0 );
            DebugAssert( m_arrDisplayModeCounts[iOutputIndex] <= D3D11WINDOW_MAX_DISPLAYMODES );

            // Next output
            pOutput->Release();
            pOutput = NULL;
            ++iOutput;
        }
		// Allow adapters without output ...
        //DebugAssert( m_arrOutputCounts[iAdapterIndex] > 0 );
        DebugAssert( m_arrOutputCounts[iAdapterIndex] <= D3D11WINDOW_MAX_OUTPUTS );

        // Next adapter
        pAdapter->Release();
        pAdapter = NULL;
        ++iAdapter;
    }
    DebugAssert( m_iAdapterCount > 0 );
    DebugAssert( m_iAdapterCount <= D3D11WINDOW_MAX_ADAPTERS );
}
Void D3D11Window::_ClearEnumeration()
{
    DebugAssert( m_iAdapterCount > 0 );

    // Unselect everything
    m_hDisplayModeDesc.iIndex = INVALID_OFFSET;
    m_iDisplayMode = INVALID_OFFSET;

    if ( m_pOutput != NULL ) {
        ((IDXGIOutput*)m_pOutput)->Release();
        m_pOutput = NULL;
    }
    m_hOutputDesc.iIndex = INVALID_OFFSET;
    m_iOutput = INVALID_OFFSET;

    if ( m_pAdapter != NULL ) {
        ((IDXGIAdapter*)m_pAdapter)->Release();
        m_pAdapter = NULL;
    }
    m_hAdapterDesc.iIndex = INVALID_OFFSET;
    m_iAdapter = INVALID_OFFSET;

    // Reset arrays
    m_arrDisplayModeCounts[0] = 0;
    m_arrOutputCounts[0] = 0;
    m_iAdapterCount = 0;
}

Void D3D11Window::_SelectAdapter( UInt iAdapter, UInt iLandingOutput, UInt iLandingDisplayMode )
{
    DebugAssert( m_iAdapterCount > 0 );
    DebugAssert( iAdapter < m_iAdapterCount );
    if ( m_iAdapter == iAdapter )
        return;
    UInt iAdapterIndex = iAdapter;

    // Default display mode
    D3D11DisplayModeDesc hDefaultDisplayMode;
    hDefaultDisplayMode.iScanlineOrdering = D3D11DISPLAYMODE_SCANLINE_UNDEFINED;
    hDefaultDisplayMode.iScaling = D3D11DISPLAYMODE_SCALING_UNDEFINED;
    hDefaultDisplayMode.iFormat = PIXEL_FMT_RGBA8UN;
    hDefaultDisplayMode.iWidth = 1024;
    hDefaultDisplayMode.iHeight = 768;
    hDefaultDisplayMode.iRefreshRateNumerator = 0;
    hDefaultDisplayMode.iRefreshRateDenominator = 0;

    // Landing output
    if ( iLandingOutput == INVALID_OFFSET )
        iLandingOutput = 0;
    DebugAssert( m_arrOutputCounts[iAdapterIndex] > 0 );
    DebugAssert( iLandingOutput < m_arrOutputCounts[iAdapterIndex] );
    UInt iLandingOutputIndex = ( iAdapterIndex * D3D11WINDOW_MAX_OUTPUTS ) + iLandingOutput;

    // Landing display mode
    if ( iLandingDisplayMode != INVALID_OFFSET ) {
        DebugAssert( m_arrDisplayModeCounts[iLandingOutputIndex] > 0 );
        DebugAssert( iLandingDisplayMode < m_arrDisplayModeCounts[iLandingOutputIndex] );
    } else {
        const D3D11DisplayModeDesc * pRefDisplayMode = &hDefaultDisplayMode;
        if ( m_iAdapter != INVALID_OFFSET && m_iOutput != INVALID_OFFSET && m_iDisplayMode != INVALID_OFFSET ) {
            UInt iOldAdapterIndex = m_iAdapter;
            UInt iOldOutputIndex = ( iOldAdapterIndex * D3D11WINDOW_MAX_OUTPUTS ) + m_iOutput;
            pRefDisplayMode = ( m_arrDisplayModes + ( iOldOutputIndex * D3D11WINDOW_MAX_DISPLAYMODES ) + m_iDisplayMode );
        }
        iLandingDisplayMode = _MatchDisplayMode( iAdapter, iLandingOutput, pRefDisplayMode );
    }
    UInt iLandingDisplayModeIndex = ( iLandingOutputIndex * D3D11WINDOW_MAX_DISPLAYMODES ) + iLandingDisplayMode;

    // Unselect display mode
    m_hDisplayModeDesc.iIndex = INVALID_OFFSET;
    m_iDisplayMode = INVALID_OFFSET;

    // Unselect output
    if ( m_pOutput != NULL ) {
        ((IDXGIOutput*)m_pOutput)->Release();
        m_pOutput = NULL;
    }
    m_hOutputDesc.iIndex = INVALID_OFFSET;
    m_iOutput = INVALID_OFFSET;

    // Switch adapter
    if ( m_pAdapter != NULL ) {
        ((IDXGIAdapter*)m_pAdapter)->Release();
        m_pAdapter = NULL;
    }
    m_iAdapter = iAdapter;
    MemCopy( &m_hAdapterDesc, m_arrAdapters + iAdapterIndex, sizeof(D3D11AdapterDesc) );
    m_pAdapter = _GetAdapter( m_hAdapterDesc.iIndex );

    // Select landing output
    m_iOutput = iLandingOutput;
    MemCopy( &m_hOutputDesc, m_arrOutputs + iLandingOutputIndex, sizeof(D3D11OutputDesc) );
    m_pOutput = _GetOutput( m_pAdapter, m_hOutputDesc.iIndex );

    // Select landing display mode
    m_iDisplayMode = iLandingDisplayMode;
    MemCopy( &m_hDisplayModeDesc, m_arrDisplayModes + iLandingDisplayModeIndex, sizeof(D3D11DisplayModeDesc) );
}
Void D3D11Window::_SelectOutput( UInt iOutput, UInt iLandingDisplayMode )
{
    DebugAssert( m_iAdapter != INVALID_OFFSET );
    UInt iAdapterIndex = m_iAdapter;

    DebugAssert( m_arrOutputCounts[iAdapterIndex] > 0 );
    DebugAssert( iOutput < m_arrOutputCounts[iAdapterIndex] );
    if ( m_iOutput == iOutput )
        return;
    UInt iOutputIndex = ( iAdapterIndex * D3D11WINDOW_MAX_OUTPUTS ) + iOutput;

    // Default display mode
    D3D11DisplayModeDesc hDefaultDisplayMode;
    hDefaultDisplayMode.iScanlineOrdering = D3D11DISPLAYMODE_SCANLINE_UNDEFINED;
    hDefaultDisplayMode.iScaling = D3D11DISPLAYMODE_SCALING_UNDEFINED;
    hDefaultDisplayMode.iFormat = PIXEL_FMT_RGBA8UN;
    hDefaultDisplayMode.iWidth = 1024;
    hDefaultDisplayMode.iHeight = 768;
    hDefaultDisplayMode.iRefreshRateNumerator = 0;
    hDefaultDisplayMode.iRefreshRateDenominator = 0;

    // Landing display mode
    if ( iLandingDisplayMode != INVALID_OFFSET ) {
        DebugAssert( m_arrDisplayModeCounts[iOutputIndex] > 0 );
        DebugAssert( iLandingDisplayMode < m_arrDisplayModeCounts[iOutputIndex] );
    } else {
        const D3D11DisplayModeDesc * pRefDisplayMode = &hDefaultDisplayMode;
        if ( m_iOutput != INVALID_OFFSET && m_iDisplayMode != INVALID_OFFSET ) {
            UInt iOldOutputIndex = ( iAdapterIndex * D3D11WINDOW_MAX_OUTPUTS ) + m_iOutput;
            pRefDisplayMode = ( m_arrDisplayModes + ( iOldOutputIndex * D3D11WINDOW_MAX_DISPLAYMODES ) + m_iDisplayMode );
        }
        iLandingDisplayMode = _MatchDisplayMode( m_iAdapter, iOutput, pRefDisplayMode );
    }
    UInt iLandingDisplayModeIndex = ( iOutputIndex * D3D11WINDOW_MAX_DISPLAYMODES ) + iLandingDisplayMode;

    // Unselect display mode
    m_hDisplayModeDesc.iIndex = INVALID_OFFSET;
    m_iDisplayMode = INVALID_OFFSET;

    // Switch output
    if ( m_pOutput != NULL ) {
        ((IDXGIOutput*)m_pOutput)->Release();
        m_pOutput = NULL;
    }
    m_iOutput = iOutput;
    MemCopy( &m_hOutputDesc, m_arrOutputs + iOutputIndex, sizeof(D3D11OutputDesc) );
    m_pOutput = _GetOutput( m_pAdapter, m_hOutputDesc.iIndex );

    // Select landing display mode
    m_iDisplayMode = iLandingDisplayMode;
    MemCopy( &m_hDisplayModeDesc, m_arrDisplayModes + iLandingDisplayModeIndex, sizeof(D3D11DisplayModeDesc) );
}
Void D3D11Window::_SelectDisplayMode( UInt iDisplayMode )
{
    DebugAssert( m_iAdapter != INVALID_OFFSET );
    DebugAssert( m_iOutput != INVALID_OFFSET );
    UInt iAdapterIndex = m_iAdapter;
    UInt iOutputIndex = ( iAdapterIndex * D3D11WINDOW_MAX_OUTPUTS ) + m_iOutput;

    DebugAssert( m_arrDisplayModeCounts[iOutputIndex] > 0 );
    DebugAssert( iDisplayMode < m_arrDisplayModeCounts[iOutputIndex] );
    if ( m_iDisplayMode == iDisplayMode )
        return;
    UInt iDisplayModeIndex = ( iOutputIndex * D3D11WINDOW_MAX_DISPLAYMODES ) + iDisplayMode;

    // Switch display mode
    m_iDisplayMode = iDisplayMode;
    MemCopy( &m_hDisplayModeDesc, m_arrDisplayModes + iDisplayModeIndex, sizeof( D3D11DisplayModeDesc ) );
}

Void D3D11Window::_SwitchAdapter( UInt iAdapter, UInt iLandingOutput, UInt iLandingDisplayMode )
{
    DebugAssert( m_pRenderer != NULL );
    DebugAssert( m_pRenderer->m_pSwapChain != NULL );
    if ( m_bInDisplayModeSwitchTransition )
        return;
    if ( m_iAdapter == iAdapter )
        return;

    m_pRenderer->_DestroySwapChain();
    m_pRenderer->_DestroyDevice();

    _SelectAdapter( iAdapter, iLandingOutput, iLandingDisplayMode );

    m_pRenderer->_CreateDevice();
    m_pRenderer->_CreateSwapChain();

    DXGI_MODE_DESC hDXGIMode;
    UInt iModeIndex;
    m_hDisplayModeDesc.ConvertTo( &hDXGIMode, &iModeIndex );

    // Start transition
    m_bInDisplayModeSwitchTransition = true;

    HRESULT hRes;
    if ( m_bFullScreen )
        hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->SetFullscreenState( TRUE, (IDXGIOutput*)m_pOutput );
    else
        hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->SetFullscreenState( FALSE, NULL );
    DebugAssert( hRes == S_OK );

    hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->ResizeTarget( &hDXGIMode );
    DebugAssert( hRes == S_OK );

    // End transition
    m_bInDisplayModeSwitchTransition = false;
}
Void D3D11Window::_SwitchOutput( UInt iOutput, UInt iLandingDisplayMode )
{
    DebugAssert( m_pRenderer != NULL );
    DebugAssert( m_pRenderer->m_pSwapChain != NULL );
    if ( m_bInDisplayModeSwitchTransition )
        return;
    DebugAssert( m_bFullScreen );
    if ( m_iOutput == iOutput )
        return;

    _SelectOutput( iOutput, iLandingDisplayMode );

    DXGI_MODE_DESC hDXGIMode;
    UInt iModeIndex;
    m_hDisplayModeDesc.ConvertTo( &hDXGIMode, &iModeIndex );

    // Start transition
    m_bInDisplayModeSwitchTransition = true;

    HRESULT hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->SetFullscreenState( TRUE, (IDXGIOutput*)m_pOutput );
    DebugAssert( hRes == S_OK );

    hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->ResizeTarget( &hDXGIMode );
    DebugAssert( hRes == S_OK );

    // End transition
    m_bInDisplayModeSwitchTransition = false;
}
Void D3D11Window::_SwitchDisplayMode( UInt iDisplayMode )
{
    DebugAssert( m_pRenderer != NULL );
    DebugAssert( m_pRenderer->m_pSwapChain != NULL );
    if ( m_bInDisplayModeSwitchTransition )
        return;
    if ( m_iDisplayMode == iDisplayMode )
        return;

    _SelectDisplayMode( iDisplayMode );

    DXGI_MODE_DESC hDXGIMode;
    UInt iModeIndex;
    m_hDisplayModeDesc.ConvertTo( &hDXGIMode, &iModeIndex );

    // Start transition
    m_bInDisplayModeSwitchTransition = true;

    HRESULT hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->ResizeTarget( &hDXGIMode );
    DebugAssert( hRes == S_OK );

    // End transition
    m_bInDisplayModeSwitchTransition = false;
}
Void D3D11Window::_ToggleFullScreen( Bool bFullscreen )
{
    DebugAssert( m_pRenderer != NULL );
    DebugAssert( m_pRenderer->m_pSwapChain != NULL );
    if ( m_bInDisplayModeSwitchTransition )
        return;
    if ( m_bFullScreen == bFullscreen )
        return;

    DXGI_MODE_DESC hDXGIMode;
    UInt iModeIndex;
    m_hDisplayModeDesc.ConvertTo( &hDXGIMode, &iModeIndex );

    // Start transition
    m_bInDisplayModeSwitchTransition = true;

    HRESULT hRes;
    if ( bFullscreen )
        hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->SetFullscreenState( TRUE, (IDXGIOutput*)m_pOutput );
    else
        hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->SetFullscreenState( FALSE, NULL );
    DebugAssert( hRes == S_OK );

    hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->ResizeTarget( &hDXGIMode );
    DebugAssert( hRes == S_OK );

    // End transition
    m_bInDisplayModeSwitchTransition = false;
}
Bool D3D11Window::_UpdateOutputFromWindow()
{
    DebugAssert( m_pRenderer != NULL );
    DebugAssert( m_pRenderer->m_pSwapChain != NULL );
    DebugAssert( !m_bFullScreen );

    // Retrieve output containing our window
    IDXGIOutput * pOutput = NULL;
    HRESULT hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->GetContainingOutput( &pOutput );
    DebugAssert( hRes == S_OK && pOutput != NULL );

    DXGI_OUTPUT_DESC hOutputDesc;
    hRes = pOutput->GetDesc( &hOutputDesc );
    DebugAssert( hRes == S_OK );

    pOutput->Release();
    pOutput = NULL;

    // Retrieve output index
    UInt iNewOutput = INVALID_OFFSET;
    UInt iAdapterIndex = m_iAdapter;
    UInt iOutputIndex = ( iAdapterIndex * D3D11WINDOW_MAX_OUTPUTS );
    for( UInt i = 0; i < m_arrOutputCounts[iAdapterIndex]; ++i ) {
        //if ( StringFn->NCmp(hOutputDesc.DeviceName, m_arrOutputs[iOutputIndex + i].DeviceName, 31) == 0 ) {
        if ( hOutputDesc.Monitor == (HMONITOR)(m_arrOutputs[iOutputIndex + i].pMonitor) ) {
            iNewOutput = i;
            break;
        }
    }
    DebugAssert( iNewOutput != INVALID_OFFSET );

    // No change case
    if ( m_iOutput == iNewOutput )
        return false;

    // Switch output accordingly
    _SelectOutput( iNewOutput );

    DXGI_MODE_DESC hDXGIMode;
    UInt iModeIndex;
    m_hDisplayModeDesc.ConvertTo( &hDXGIMode, &iModeIndex );

    // Start transition
    m_bInDisplayModeSwitchTransition = true;

    hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->SetFullscreenState( FALSE, NULL );
    DebugAssert( hRes == S_OK );

    hRes = ((IDXGISwapChain*)(m_pRenderer->m_pSwapChain))->ResizeTarget( &hDXGIMode );
    DebugAssert( hRes == S_OK );

    // End transition
    m_bInDisplayModeSwitchTransition = false;

    return true;
}

