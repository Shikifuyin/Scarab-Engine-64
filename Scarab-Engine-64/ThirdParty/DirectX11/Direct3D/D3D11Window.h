/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/D3D11Window.h
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
// Header prelude
#ifndef SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_D3D11WINDOW_H
#define SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_D3D11WINDOW_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../../System/System.h"

#include "D3D11Mappings.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define D3D11WINDOW_TITLE_SIZE 256

#define D3D11WINDOW_MIN_WIDTH  200
#define D3D11WINDOW_MIN_HEIGHT 200

// Display Enumeration
#define D3D11WINDOW_MAX_ADAPTERS     4   // Allowed video cards
#define D3D11WINDOW_MAX_OUTPUTS      4   // Allowed outputs per video card
#define D3D11WINDOW_MAX_DISPLAYMODES 256 // Allowed display modes per output

#define D3D11WINDOW_MAX_OUTPUTS_TOTAL      (D3D11WINDOW_MAX_OUTPUTS * D3D11WINDOW_MAX_ADAPTERS)
#define D3D11WINDOW_MAX_DISPLAYMODES_TOTAL (D3D11WINDOW_MAX_DISPLAYMODES * D3D11WINDOW_MAX_OUTPUTS_TOTAL)

// Prototypes
class D3D11Window;
class D3D11Renderer;

/////////////////////////////////////////////////////////////////////////////////
// The D3D11WindowCallbacks class
class D3D11WindowCallbacks
{
public:
    D3D11WindowCallbacks( Void * pUserData = NULL );
    virtual ~D3D11WindowCallbacks();

    inline Void SetUserData( Void * pUserData );

    // Main loop
    virtual Void OnPause() = 0;
    virtual Void OnResume() = 0;

    virtual Void OnIdleTime() = 0;

    // Window
    virtual Void OnWindowMonitorChange( Void * pMonitor ) = 0;
    virtual Void OnWindowMove( UInt iX, UInt iY ) = 0;
    virtual Void OnWindowResize( UInt iWidth, UInt iHeight ) = 0;

    virtual Void OnWindowFocus() = 0;

    virtual Void OnWindowDraw() = 0;

    // Keyboard
    virtual Bool OnKeyPress( KeyCode iKey ) = 0;
    virtual Bool OnKeyRelease( KeyCode iKey ) = 0;

    // Mouse
    virtual Void OnMouseMove( UInt iX, UInt iY ) = 0;
    virtual Bool OnMousePress( UInt iX, UInt iY, KeyCode iKey ) = 0;
    virtual Bool OnMouseRelease( UInt iX, UInt iY, KeyCode iKey ) = 0;
    virtual Bool OnMouseDblClick( UInt iX, UInt iY, KeyCode iKey ) = 0;
    virtual Bool OnMouseWheel( UInt iX, UInt iY, Int iWheelDelta ) = 0;

protected:
    Void * m_pUserData;
};

/////////////////////////////////////////////////////////////////////////////////
// The D3D11Window class
class D3D11Window
{
public:
    D3D11Window( const GChar * strTitle, UInt iX, UInt iY, UInt iWidth, UInt iHeight, D3D11WindowCallbacks * pCallbacks );
    ~D3D11Window();

    // Window methods
    inline Bool IsActive() const;
    inline Bool IsMinimized() const;
    inline Bool IsMaximized() const;
    inline Bool IsInSizeMove() const;

    inline const GChar * GetTitle() const;
    Void SetTitle( const GChar * strTitle );

    Void GetScreenSize( UInt * outWidth, UInt * outHeight ) const;
    Void GetWindowRect( D3D11Rectangle * outWindowRect ) const;
    Void GetClientRect( D3D11Rectangle * outClientRect ) const;

    UInt GetDpi() const;

    Bool IsVisible() const;
    Void SetVisible( Bool bVisible ) const;
    //Void Update() const;

    Void MessageLoop() const;
    Void Exit() const;

    // Cursor methods
    Void ClipCursor( Bool bClip, Bool bClipToClientRect ) const;
    Void ShowCursor( Bool bShow ) const;
    Void WarpCursor( UInt iX, UInt iY ) const;
    Void SetCursor( const DWord * pCursorRGBA = NULL, UInt iWidth = 0, UInt iHeight = 0, DWord dwKeyColor = 0,
                    UInt iHotspotX = 0, UInt iHotspotY = 0 );

    // Clipboard methods
    Bool GetClipboardData( GChar * outText, UInt iMaxLength ) const;
    Bool SetClipboardData( const GChar * strText, UInt iLength = INVALID_OFFSET ) const;

    // Display Enumeration
    inline UInt GetAdapterCount() const;
    inline const D3D11AdapterDesc * GetAdapterDesc( UInt iAdapter ) const;

    inline UInt GetCurrentAdapter() const;
    inline const D3D11AdapterDesc * GetCurrentAdapterDesc() const;

    inline UInt GetOutputCount( UInt iAdapter = INVALID_OFFSET ) const;
    inline const D3D11OutputDesc * GetOutputDesc( UInt iOutput, UInt iAdapter = INVALID_OFFSET ) const;

    inline UInt GetCurrentOutput() const;
    inline const D3D11OutputDesc * GetCurrentOutputDesc() const;

    inline UInt GetDisplayModeCount( UInt iOutput = INVALID_OFFSET, UInt iAdapter = INVALID_OFFSET ) const;
    inline const D3D11DisplayModeDesc * GetDisplayModeDesc( UInt iDisplayMode, UInt iOutput = INVALID_OFFSET, UInt iAdapter = INVALID_OFFSET ) const;

    inline UInt GetCurrentDisplayMode() const;
    inline const D3D11DisplayModeDesc * GetCurrentDisplayModeDesc() const;

    UInt MatchDisplayMode( D3D11DisplayModeDesc * outMatchedDisplayMode, const D3D11DisplayModeDesc * pDisplayMode, UInt iOutput = INVALID_OFFSET, UInt iAdapter = INVALID_OFFSET );

    Void SwitchAdapter( UInt iAdapter, UInt iOutput = INVALID_OFFSET, UInt iDisplayMode = INVALID_OFFSET );
    Void SwitchOutput( UInt iOutput, UInt iDisplayMode = INVALID_OFFSET );
    Void SwitchDisplayMode( UInt iDisplayMode );

    Bool IsFullScreen( UInt * outOutput = NULL ) const;
    inline Bool IsFullScreenWindowed() const;
    Void SetFullScreen( Bool bFullScreen, UInt iOutput = INVALID_OFFSET, UInt iDisplayMode = INVALID_OFFSET );
    Void SetFullScreenWindowed( Bool bFullScreenWindowed );

    Void MoveWindow( UInt iX, UInt iY ) const;

    Void GetGammaCaps( D3D11GammaCaps * outGammaCaps ) const;
    Void GetGammaControl( D3D11GammaControl * outGammaControl ) const;
    Void SetGammaControl( const D3D11GammaControl * pGammaControl ) const;

    Void WaitForVSync() const;
    
private:
    // Renderer bind
    friend class D3D11Renderer;
    D3D11Renderer * m_pRenderer;

    // Window management
    Void _CreateWindow( UInt iX, UInt iY, UInt iWidth, UInt iHeight );
    Void _DestroyWindow();

    static Int64 __stdcall _MsgProc_Static( Void * hWindow, UInt iMsg, UInt64 iWParam, Int64 iLParam );
    Int64 _MsgProc_Virtual( Void * hWindow, UInt iMsg, UInt64 iWParam, Int64 iLParam );

    GChar m_strTitle[D3D11WINDOW_TITLE_SIZE];
    D3D11WindowCallbacks * m_pCallbacks;

    Void * m_hInstance;
    Void * m_hWindow;
    Void * m_hWindowMonitor;
    Void * m_hDefaultWindowCursor;
    Void * m_hWindowCursor;
    Void * m_hWindowMenu;

    Bool m_bActive;
    Bool m_bMinimized, m_bMaximized;
    Bool m_bIsInSizeMove;

    // Display modes management
    Void * _GetAdapter( UInt iAdapter, D3D11AdapterDesc * outDesc = NULL ) const;
    Void * _GetOutput( Void * pAdapter, UInt iOutput, D3D11OutputDesc * outDesc = NULL ) const;
    UInt _FilterDisplayModes( D3D11DisplayModeDesc * arrDisplayModes, UInt iCount ) const;
    Int _CompareDisplayModes( const D3D11DisplayModeDesc * pLeft, const D3D11DisplayModeDesc * pRight ) const;
    Void _SortDisplayModes( D3D11DisplayModeDesc * outDisplayModes, const D3D11DisplayModeDesc * arrDisplayModes, UInt iCount ) const;
    UInt _MatchDisplayMode( UInt iAdapter, UInt iOutput, const D3D11DisplayModeDesc * pDesc ) const;

    Void _Enumerate();
    Void _ClearEnumeration();

    Void _SelectAdapter( UInt iAdapter, UInt iLandingOutput = INVALID_OFFSET, UInt iLandingDisplayMode = INVALID_OFFSET );
    Void _SelectOutput( UInt iOutput, UInt iLandingDisplayMode = INVALID_OFFSET );
    Void _SelectDisplayMode( UInt iDisplayMode );

    Void * m_pDXGIFactory;

    UInt m_iAdapterCount;
    D3D11AdapterDesc m_arrAdapters[D3D11WINDOW_MAX_ADAPTERS];
    UInt m_arrOutputCounts[D3D11WINDOW_MAX_ADAPTERS];
    D3D11OutputDesc m_arrOutputs[D3D11WINDOW_MAX_OUTPUTS_TOTAL];
    UInt m_arrDisplayModeCounts[D3D11WINDOW_MAX_OUTPUTS_TOTAL];
    D3D11DisplayModeDesc m_arrDisplayModes[D3D11WINDOW_MAX_DISPLAYMODES_TOTAL];

    UInt m_iAdapter, m_iOutput, m_iDisplayMode;
    D3D11AdapterDesc m_hAdapterDesc;
    D3D11OutputDesc m_hOutputDesc;
    D3D11DisplayModeDesc m_hDisplayModeDesc;
    Void * m_pAdapter;
    Void * m_pOutput;

    Void _SwitchAdapter( UInt iAdapter, UInt iLandingOutput = INVALID_OFFSET, UInt iLandingDisplayMode = INVALID_OFFSET );
    Void _SwitchOutput( UInt iOutput, UInt iLandingDisplayMode = INVALID_OFFSET );
    Void _SwitchDisplayMode( UInt iDisplayMode );
    Void _ToggleFullScreen( Bool bFullscreen );
    Bool _UpdateOutputFromWindow();

    Bool m_bFullScreen;
    Bool m_bFullScreenWindowed;
    UInt m_iFullScreenWindowedSavedX;
    UInt m_iFullScreenWindowedSavedY;
    UInt m_iFullScreenWindowedSavedWidth;
    UInt m_iFullScreenWindowedSavedHeight;
    DWord m_dwFullScreenWindowedSavedStyle;
    Bool m_bInDisplayModeSwitchTransition;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "D3D11Window.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_D3D11WINDOW_H

