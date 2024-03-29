/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUI.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Main Interface
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
#include <objbase.h>

// Enable Visual Styles, Use Common Controls version 6.0 (No need for a manifest)
#pragma comment( linker, \
    "\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' \
    processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"" )

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "WinGUI.h"

/////////////////////////////////////////////////////////////////////////////////
// Helpers

// Font Enumeration Proc
int CALLBACK _EnumFontExProc( const LOGFONT * lpelfe, const TEXTMETRIC * lpntme, DWORD FontType, LPARAM lParam )
{
    // Only accept TRUETYPE
    if ( FontType != TRUETYPE_FONTTYPE )
        return 1;

    // Require Normal Weight
    if ( lpelfe->lfWeight != FW_NORMAL )
        return 1;

    // Reject Italic/Underlined/StruckOut
    if ( lpelfe->lfItalic || lpelfe->lfUnderline || lpelfe->lfStrikeOut )
        return 1;

    // Require Variable Pitch
    if ( lpelfe->lfPitchAndFamily & 0x03 != VARIABLE_PITCH )
        return 1;

    // Ensure Charset is proper
    if ( lpelfe->lfCharSet != ANSI_CHARSET )
        return 1;

    // Require at least Basic-Latin and Latin-1 UNICODE subset
    const NEWTEXTMETRICEX * pNTM = (const NEWTEXTMETRICEX *)lpntme;
    if ( (pNTM->ntmFontSig.fsUsb[0] & 0x03) != 0x03 )
        return 1;

    // We should have the proper font at this point
    MemCopy( (LOGFONT*)lParam, lpelfe, sizeof(LOGFONT) );
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUI implementation
WinGUI::WinGUI()
{
	m_pAppWindow = NULL;

    m_iFontWidth = 6;
    m_iFontHeight = 16;
    m_pDefaultFont = NULL;
}
WinGUI::~WinGUI()
{
	// nothing to do
}

Void WinGUI::CreateAppWindow( WinGUIWindowModel * pModel )
{
    DebugAssert( m_pAppWindow == NULL );

    // Required by some stuff
    CoInitialize(NULL);

    // Create App Window
    m_pAppWindow = (WinGUIWindow*)( SystemFn->MemAlloc(sizeof(WinGUIWindow)) );
    *m_pAppWindow = WinGUIWindow( pModel );

    m_pAppWindow->_Create();

    // Init Common Controls Here
    INITCOMMONCONTROLSEX hICCX;
    hICCX.dwSize = sizeof(INITCOMMONCONTROLSEX);
    hICCX.dwICC = ICC_STANDARD_CLASSES  // Button, Edit, Static, ListBox, ComboBox, ScrollBar
                | ICC_USEREX_CLASSES    // ComboBoxEx
                | ICC_PROGRESS_CLASS    // ProgressBar
                | ICC_BAR_CLASSES       // ToolBar, StatusBar, TrackBar, Tooltip
                | ICC_TAB_CLASSES       // Tabs, Tooltip
                | ICC_LISTVIEW_CLASSES; // ListView, HeaderControl
    InitCommonControlsEx( &hICCX );

    // Create Default Font
    LOGFONT hFont;
    StringFn->NCopy( hFont.lfFaceName, TEXT("Segoe UI"), 31 );
    hFont.lfCharSet = DEFAULT_CHARSET;
    hFont.lfPitchAndFamily = 0;

    HDC hDC = GetDC( (HWND)(m_pAppWindow->m_hHandle) );
    EnumFontFamiliesEx( hDC, &hFont, _EnumFontExProc, (LPARAM)&hFont, 0 );
    ReleaseDC( (HWND)(m_pAppWindow->m_hHandle), hDC );

    m_pDefaultFont = CreateFont( m_iFontHeight, m_iFontWidth, hFont.lfEscapement, hFont.lfOrientation, hFont.lfWeight,
                                 hFont.lfItalic, hFont.lfUnderline, hFont.lfStrikeOut, hFont.lfCharSet,
                                 hFont.lfOutPrecision, hFont.lfClipPrecision, hFont.lfQuality,
                                 hFont.lfPitchAndFamily, TEXT("Segoe UI") );
    DebugAssert( m_pDefaultFont != NULL );
}
Void WinGUI::DestroyAppWindow()
{
    DebugAssert( m_pAppWindow != NULL );

    // Destroy Default Font
    DeleteObject( m_pDefaultFont );
    m_pDefaultFont = NULL;

    // Destroy App Window
    while( m_pAppWindow->GetChildCount() > 0 )
        DestroyElement( m_pAppWindow->GetChild(0) );

    m_pAppWindow->_Destroy();

    m_pAppWindow->~WinGUIWindow();
    SystemFn->MemFree( m_pAppWindow );
    m_pAppWindow = NULL;

    // Required by some stuff
    CoUninitialize();
}

Int WinGUI::MessageLoop( Void (*pfIdleFunction)(Void*), Void * pUserData ) const
{
    MSG hMessage;

    while( true ) {
        if ( PeekMessage(&hMessage, NULL, 0u, 0u, PM_REMOVE) != FALSE ) {
            if ( hMessage.message == WM_QUIT )
                break;
            TranslateMessage( &hMessage );
		    DispatchMessage( &hMessage );
        } else {
            if ( pfIdleFunction != NULL ) 
                pfIdleFunction( pUserData );
        }
    }

    return (Int)( hMessage.wParam ); // Exit Code from PostQuitMessage
}

WinGUIMessageBoxResponse WinGUI::SpawnMessageBox( const GChar * strTitle, const GChar * strText, const WinGUIMessageBoxOptions & hOptions ) const
{
    // Parse Options
    UInt iOptions = MB_SETFOREGROUND;
    switch( hOptions.iType ) {
        case WINGUI_MESSAGEBOX_OK:
            iOptions = MB_OK;
            iOptions |= MB_DEFBUTTON1;
            break;
        case WINGUI_MESSAGEBOX_OKCANCEL:
            iOptions = MB_OKCANCEL;
            if ( hOptions.iDefaultResponse == WINGUI_MESSAGEBOX_RESPONSE_OK )
                iOptions |= MB_DEFBUTTON1;
            else if ( hOptions.iDefaultResponse == WINGUI_MESSAGEBOX_RESPONSE_CANCEL )
                iOptions |= MB_DEFBUTTON2;
            break;
        case WINGUI_MESSAGEBOX_YESNO:
            iOptions = MB_YESNO;
            if ( hOptions.iDefaultResponse == WINGUI_MESSAGEBOX_RESPONSE_YES )
                iOptions |= MB_DEFBUTTON1;
            else if ( hOptions.iDefaultResponse == WINGUI_MESSAGEBOX_RESPONSE_NO )
                iOptions |= MB_DEFBUTTON2;
            break;
        case WINGUI_MESSAGEBOX_YESNOCANCEL:
            iOptions = MB_YESNOCANCEL;
            if ( hOptions.iDefaultResponse == WINGUI_MESSAGEBOX_RESPONSE_YES )
                iOptions |= MB_DEFBUTTON1;
            else if ( hOptions.iDefaultResponse == WINGUI_MESSAGEBOX_RESPONSE_NO )
                iOptions |= MB_DEFBUTTON2;
            else if ( hOptions.iDefaultResponse == WINGUI_MESSAGEBOX_RESPONSE_CANCEL )
                iOptions |= MB_DEFBUTTON3;
            break;
        case WINGUI_MESSAGEBOX_RETRYCANCEL:
            iOptions = MB_RETRYCANCEL;
            if ( hOptions.iDefaultResponse == WINGUI_MESSAGEBOX_RESPONSE_RETRY )
                iOptions |= MB_DEFBUTTON1;
            else if ( hOptions.iDefaultResponse == WINGUI_MESSAGEBOX_RESPONSE_CANCEL )
                iOptions |= MB_DEFBUTTON2;
            break;
        default: DebugAssert(false); break;
    }
    switch( hOptions.iIcon ) {
        case WINGUI_MESSAGEBOX_ICON_NONE:    iOptions |= 0; break;
        case WINGUI_MESSAGEBOX_ICON_INFO:    iOptions |= MB_ICONINFORMATION; break;
        case WINGUI_MESSAGEBOX_ICON_WARNING: iOptions |= MB_ICONWARNING; break;
        case WINGUI_MESSAGEBOX_ICON_ERROR:   iOptions |= MB_ICONERROR; break;
        default: DebugAssert(false); break;
    }
    if ( hOptions.bMustAnswer )
        iOptions |= MB_SYSTEMMODAL;
    else
        iOptions |= MB_APPLMODAL;

    // Retrieve App Window Handle
    HWND hWnd = (HWND)( m_pAppWindow->m_hHandle );

    // Spawn the MessageBox
    Int iResult = MessageBox( hWnd, strText, strTitle, iOptions );

    // Parse Result
    switch( iResult ) {
        case IDCANCEL: return WINGUI_MESSAGEBOX_RESPONSE_CANCEL; break;
        case IDOK:     return WINGUI_MESSAGEBOX_RESPONSE_OK; break;
        case IDYES:    return WINGUI_MESSAGEBOX_RESPONSE_YES; break;
        case IDNO:     return WINGUI_MESSAGEBOX_RESPONSE_NO; break;
        case IDRETRY:  return WINGUI_MESSAGEBOX_RESPONSE_RETRY; break;
        default: DebugAssert(false); break;
    }

    // Should never happen
    return WINGUI_MESSAGEBOX_RESPONSE_CANCEL;
}

WinGUIContainer * WinGUI::CreateContainer( WinGUIElement * pParent, WinGUIContainerModel * pModel ) const
{
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Create Element
    WinGUIContainer * pContainer = (WinGUIContainer*)( SystemFn->MemAlloc(sizeof(WinGUIContainer)) );
    *pContainer = WinGUIContainer( pParent, pModel );

    ((WinGUIElement*)pContainer)->_Create();
    ((WinGUIElement*)pContainer)->_ApplyDefaultFont( m_pDefaultFont );

    // Add Child Links to Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_AppendChild( pContainer );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_AppendChild( pContainer );
    }

    // Done
    return pContainer;
}

WinGUIStatic * WinGUI::CreateStatic( WinGUIElement * pParent, WinGUIStaticModel * pModel ) const
{
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Create Element
    WinGUIStatic * pStatic = (WinGUIStatic*)( SystemFn->MemAlloc(sizeof(WinGUIStatic)) );
    *pStatic = WinGUIStatic( pParent, pModel );

    ((WinGUIElement*)pStatic)->_Create();
    ((WinGUIElement*)pStatic)->_ApplyDefaultFont( m_pDefaultFont );

    // Add Child Links to Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_AppendChild( pStatic );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_AppendChild( pStatic );
    }

    // Done
    return pStatic;
}
WinGUIGroupBox * WinGUI::CreateGroupBox( WinGUIElement * pParent, WinGUIGroupBoxModel * pModel ) const
{
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Create Element
    WinGUIGroupBox * pGroupBox = (WinGUIGroupBox*)( SystemFn->MemAlloc(sizeof(WinGUIGroupBox)) );
    *pGroupBox = WinGUIGroupBox( pParent, pModel );

    ((WinGUIElement*)pGroupBox)->_Create();
    ((WinGUIElement*)pGroupBox)->_ApplyDefaultFont( m_pDefaultFont );

    // Add Child Links to Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_AppendChild( pGroupBox );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_AppendChild( pGroupBox );
    }

    // Done
    return pGroupBox;
}

WinGUIStatusBar * WinGUI::CreateStatusBar( WinGUIElement * pParent, WinGUIStatusBarModel * pModel ) const
{
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Create Element
    WinGUIStatusBar * pStatusBar = (WinGUIStatusBar*)( SystemFn->MemAlloc(sizeof(WinGUIStatusBar)) );
    *pStatusBar = WinGUIStatusBar( pParent, pModel );

    ((WinGUIElement*)pStatusBar)->_Create();
    ((WinGUIElement*)pStatusBar)->_ApplyDefaultFont( m_pDefaultFont );

    // Add Child Links to Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_AppendChild( pStatusBar );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_AppendChild( pStatusBar );
    }

    // Done
    return pStatusBar;
}
WinGUIProgressBar * WinGUI::CreateProgressBar( WinGUIElement * pParent, WinGUIProgressBarModel * pModel ) const
{
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Create Element
    WinGUIProgressBar * pProgressBar = (WinGUIProgressBar*)( SystemFn->MemAlloc(sizeof(WinGUIProgressBar)) );
    *pProgressBar = WinGUIProgressBar( pParent, pModel );

    ((WinGUIElement*)pProgressBar)->_Create();
    ((WinGUIElement*)pProgressBar)->_ApplyDefaultFont( m_pDefaultFont );

    // Add Child Links to Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_AppendChild( pProgressBar );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_AppendChild( pProgressBar );
    }

    // Done
    return pProgressBar;
}

WinGUIButton * WinGUI::CreateButton( WinGUIElement * pParent, WinGUIButtonModel * pModel ) const
{
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Create Element
    WinGUIButton * pButton = (WinGUIButton*)( SystemFn->MemAlloc(sizeof(WinGUIButton)) );
    *pButton = WinGUIButton( pParent, pModel );

    ((WinGUIElement*)pButton)->_Create();
    ((WinGUIElement*)pButton)->_ApplyDefaultFont( m_pDefaultFont );

    // Add Child Links to Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_AppendChild( pButton );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_AppendChild( pButton );
    }

    // Done
    return pButton;
}
WinGUICheckBox * WinGUI::CreateCheckBox( WinGUIElement * pParent, WinGUICheckBoxModel * pModel ) const
{
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Create Element
    WinGUICheckBox * pCheckBox = (WinGUICheckBox*)( SystemFn->MemAlloc(sizeof(WinGUICheckBox)) );
    *pCheckBox = WinGUICheckBox( pParent, pModel );

    ((WinGUIElement*)pCheckBox)->_Create();
    ((WinGUIElement*)pCheckBox)->_ApplyDefaultFont( m_pDefaultFont );

    // Add Child Links to Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_AppendChild( pCheckBox );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_AppendChild( pCheckBox );
    }

    // Done
    return pCheckBox;
}
WinGUIRadioButton * WinGUI::CreateRadioButton( WinGUIElement * pParent, WinGUIRadioButtonModel * pModel ) const
{
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Create Element
    WinGUIRadioButton * pRadioButton = (WinGUIRadioButton*)( SystemFn->MemAlloc(sizeof(WinGUIRadioButton)) );
    *pRadioButton = WinGUIRadioButton( pParent, pModel );

    ((WinGUIElement*)pRadioButton)->_Create();
    ((WinGUIElement*)pRadioButton)->_ApplyDefaultFont( m_pDefaultFont );

    // Add Child Links to Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_AppendChild( pRadioButton );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_AppendChild( pRadioButton );
    }

    // Done
    return pRadioButton;
}
WinGUISliderBar * WinGUI::CreateSliderBar( WinGUIElement * pParent, WinGUISliderBarModel * pModel ) const
{
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Create Element
    WinGUISliderBar * pSliderBar = (WinGUISliderBar*)( SystemFn->MemAlloc(sizeof(WinGUISliderBar)) );
    *pSliderBar = WinGUISliderBar( pParent, pModel );

    ((WinGUIElement*)pSliderBar)->_Create();
    ((WinGUIElement*)pSliderBar)->_ApplyDefaultFont( m_pDefaultFont );

    // Add Child Links to Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_AppendChild( pSliderBar );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_AppendChild( pSliderBar );
    }

    // Done
    return pSliderBar;
}

WinGUITabs * WinGUI::CreateTabs( WinGUIElement * pParent, WinGUITabsModel * pModel ) const
{
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Create Element
    WinGUITabs * pTabs = (WinGUITabs*)( SystemFn->MemAlloc(sizeof(WinGUITabs)) );
    *pTabs = WinGUITabs( pParent, pModel );

    ((WinGUIElement*)pTabs)->_Create();
    ((WinGUIElement*)pTabs)->_ApplyDefaultFont( m_pDefaultFont );

    // Add Child Links to Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_AppendChild( pTabs );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_AppendChild( pTabs );
    }

    // Done
    return pTabs;
}

WinGUITextEdit * WinGUI::CreateTextEdit( WinGUIElement * pParent, WinGUITextEditModel * pModel ) const
{
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Create Element
    WinGUITextEdit * pTextEdit = (WinGUITextEdit*)( SystemFn->MemAlloc(sizeof(WinGUITextEdit)) );
    *pTextEdit = WinGUITextEdit( pParent, pModel );

    ((WinGUIElement*)pTextEdit)->_Create();
    ((WinGUIElement*)pTextEdit)->_ApplyDefaultFont( m_pDefaultFont );

    // Add Child Links to Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_AppendChild( pTextEdit );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_AppendChild( pTextEdit );
    }

    // Done
    return pTextEdit;
}
WinGUIComboBox * WinGUI::CreateComboBox( WinGUIElement * pParent, WinGUIComboBoxModel * pModel ) const
{
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Create Element
    WinGUIComboBox * pComboBox = (WinGUIComboBox*)( SystemFn->MemAlloc(sizeof(WinGUIComboBox)) );
    *pComboBox = WinGUIComboBox( pParent, pModel );

    ((WinGUIElement*)pComboBox)->_Create();
    ((WinGUIElement*)pComboBox)->_ApplyDefaultFont( m_pDefaultFont );

    // Add Child Links to Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_AppendChild( pComboBox );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_AppendChild( pComboBox );
    }

    // Done
    return pComboBox;
}

WinGUITable * WinGUI::CreateTable( WinGUIElement * pParent, WinGUITableModel * pModel ) const
{
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Create Element
    WinGUITable * pTable = (WinGUITable*)( SystemFn->MemAlloc(sizeof(WinGUITable)) );
    *pTable = WinGUITable( pParent, pModel );

    ((WinGUIElement*)pTable)->_Create();
    ((WinGUIElement*)pTable)->_ApplyDefaultFont( m_pDefaultFont );

    // Add Child Links to Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_AppendChild( pTable );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_AppendChild( pTable );
    }

    // Done
    return pTable;
}

WinGUIToolTip * WinGUI::CreateToolTip( WinGUIElement * pParent, WinGUIToolTipModel * pModel ) const
{
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Create Element
    WinGUIToolTip * pToolTip = (WinGUIToolTip*)( SystemFn->MemAlloc(sizeof(WinGUIToolTip)) );
    *pToolTip = WinGUIToolTip( pParent, pModel );

    ((WinGUIElement*)pToolTip)->_Create();
    ((WinGUIElement*)pToolTip)->_ApplyDefaultFont( m_pDefaultFont );

    // Add Child Links to Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_AppendChild( pToolTip );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_AppendChild( pToolTip );
    }

    // Done
    return pToolTip;
}

Void WinGUI::DestroyElement( WinGUIElement * pElement ) const
{
    // Retrieve Parent
    WinGUIElement * pParent = pElement->GetParent();
    DebugAssert( pParent != NULL ); // Don't use this to destroy app window !
    DebugAssert( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW || pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER );

    // Remove Child Links from Parent
    if ( pParent->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pParent;
        pWindow->_RemoveChild( pElement );
    } else if ( pParent->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pParent;
        pContainer->_RemoveChild( pElement );
    }

    // Recurse
    if ( pElement->GetElementType() == WINGUI_ELEMENT_WINDOW ) {
        WinGUIWindow * pWindow = (WinGUIWindow*)pElement;
        while( pWindow->GetChildCount() > 0 )
            DestroyElement( pWindow->GetChild(0) );
    } else if ( pElement->GetElementType() == WINGUI_ELEMENT_CONTAINER ) {
        WinGUIContainer * pContainer = (WinGUIContainer*)pElement;
        while( pContainer->GetChildCount() > 0 )
            DestroyElement( pContainer->GetChild(0) );
    }

    // Destroy Element
    pElement->_Destroy();

    pElement->~WinGUIElement();
    SystemFn->MemFree( pElement );
    pElement = NULL;
}

Void WinGUI::ShowCursor( Bool bShow ) const
{
    ::ShowCursor( bShow ? TRUE : FALSE );
}
Void WinGUI::SetCursor( WinGUICursor * pCursor ) const
{
    ::SetCursor( (HCURSOR)(pCursor->m_hHandle) );
}

/////////////////////////////////////////////////////////////////////////////////
