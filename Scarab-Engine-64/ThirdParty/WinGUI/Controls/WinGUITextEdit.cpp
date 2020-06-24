/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUITextEdit.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : Text Edit (Single Line)
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
#include "WinGUITextEdit.h"

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUITextEditModel implementation
WinGUITextEditModel::WinGUITextEditModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// Default Parameters
	StringFn->Copy( m_hCreationParameters.strInitialText, TEXT("TextEdit") );

	m_hCreationParameters.iAlign = WINGUI_TEXTEDIT_ALIGN_LEFT;
	m_hCreationParameters.iCase = WINGUI_TEXTEDIT_CASE_BOTH;
	m_hCreationParameters.iMode = WINGUI_TEXTEDIT_MODE_TEXT;

	m_hCreationParameters.bAllowHorizontalScroll = true;
	m_hCreationParameters.bDontHideSelection = false;
	m_hCreationParameters.bReadOnly = false;

	m_hCreationParameters.bEnableTabStop = true;
}
WinGUITextEditModel::~WinGUITextEditModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUITextEdit implementation
WinGUITextEdit::WinGUITextEdit( WinGUIElement * pParent, WinGUITextEditModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	// nothing to do
}
WinGUITextEdit::~WinGUITextEdit()
{
	// nothing to do
}

Void WinGUITextEdit::Enable()
{
	HWND hHandle = (HWND)m_hHandle;
	Edit_Enable( hHandle, TRUE );
}
Void WinGUITextEdit::Disable()
{
	HWND hHandle = (HWND)m_hHandle;
	Edit_Enable( hHandle, FALSE );
}

Bool WinGUITextEdit::CanUndo() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ( Edit_CanUndo(hHandle) != FALSE );
}
Void WinGUITextEdit::Undo()
{
	HWND hHandle = (HWND)m_hHandle;
	if ( Edit_CanUndo(hHandle) )
		Edit_Undo( hHandle );
}

Bool WinGUITextEdit::WasModified() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ( Edit_GetModify(hHandle) != FALSE );
}
Void  WinGUITextEdit::SetReadOnly( Bool bReadOnly )
{
	HWND hHandle = (HWND)m_hHandle;
	Edit_SetReadOnly( hHandle, bReadOnly ? TRUE : FALSE );
}

UInt WinGUITextEdit::GetTextLength() const
{
	HWND hHandle = (HWND)m_hHandle;
	return Edit_GetTextLength( hHandle );
}
Void WinGUITextEdit::GetText( GChar * outText, UInt iMaxLength ) const
{
	HWND hHandle = (HWND)m_hHandle;
	Edit_GetText( hHandle, outText, iMaxLength );
}
Void WinGUITextEdit::SetText( const GChar * strText )
{
	HWND hHandle = (HWND)m_hHandle;
	Edit_SetText( hHandle, strText );
}

Void WinGUITextEdit::SetTextLimit( UInt iMaxLength )
{
	HWND hHandle = (HWND)m_hHandle;
	Edit_LimitText( hHandle, iMaxLength );
}

Void WinGUITextEdit::GetSelection( UInt * outStartIndex, UInt * outLength ) const
{
	HWND hHandle = (HWND)m_hHandle;
	DWord dwResult = Edit_GetSel( hHandle );
	*outStartIndex = (UInt)( LOWORD(dwResult) );
	*outLength = ( (UInt)(HIWORD(dwResult)) ) - *outStartIndex;
}
Void WinGUITextEdit::SetSelection( UInt iStart, UInt iLength )
{
	HWND hHandle = (HWND)m_hHandle;
	Edit_SetSel( hHandle, iStart, iStart + iLength );
}
Void WinGUITextEdit::ReplaceSelection( const GChar * strText )
{
	HWND hHandle = (HWND)m_hHandle;
	Edit_ReplaceSel( hHandle, strText );
}

Void WinGUITextEdit::GetCueText( GChar * outText, UInt iMaxLength ) const
{
	HWND hHandle = (HWND)m_hHandle;
	Edit_GetCueBannerText( hHandle, outText, iMaxLength );
}
Void WinGUITextEdit::SetCueText( const GChar * strText, Bool bOnFocus )
{
	HWND hHandle = (HWND)m_hHandle;
	Edit_SetCueBannerTextFocused( hHandle, strText, bOnFocus ? TRUE : FALSE );
}

Void WinGUITextEdit::ShowBalloonTip( const GChar * strTitle, const GChar * strText, WinGUITextEditBalloonTipIcon iIcon )
{
	HWND hHandle = (HWND)m_hHandle;
	EDITBALLOONTIP hBalloonTip;
	hBalloonTip.cbStruct = sizeof(EDITBALLOONTIP);
	hBalloonTip.pszTitle = strTitle;
	hBalloonTip.pszText = strText;
	switch( iIcon ) {
		case WINGUI_TEXTEDIT_BALLOONTIP_ICON_NONE:    hBalloonTip.ttiIcon = TTI_NONE; break;
		case WINGUI_TEXTEDIT_BALLOONTIP_ICON_INFO:	  hBalloonTip.ttiIcon = TTI_INFO; break;
		case WINGUI_TEXTEDIT_BALLOONTIP_ICON_WARNING: hBalloonTip.ttiIcon = TTI_WARNING; break;
		case WINGUI_TEXTEDIT_BALLOONTIP_ICON_ERROR:   hBalloonTip.ttiIcon = TTI_ERROR; break;
		default: DebugAssert(false); break;
	}
	Edit_ShowBalloonTip( hHandle, &hBalloonTip );
}
Void WinGUITextEdit::HideBalloonTip()
{
	HWND hHandle = (HWND)m_hHandle;
	Edit_HideBalloonTip( hHandle );
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUITextEdit::_Create()
{
	DebugAssert( m_hHandle == NULL );

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUITextEditModel * pModel = (WinGUITextEditModel*)m_pModel;

	// Compute Layout
    const WinGUILayout * pLayout = pModel->GetLayout();

    WinGUIRectangle hParentRect;
    m_pParent->GetClientRect( &hParentRect );

    WinGUIRectangle hWindowRect;
    pLayout->ComputeLayout( &hWindowRect, hParentRect );

	// Get Creation Parameters
    const WinGUITextEditParameters * pParameters = pModel->GetCreationParameters();

	// Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE | WS_BORDER );
	switch( pParameters->iAlign ) {
		case WINGUI_TEXTEDIT_ALIGN_LEFT:   dwStyle |= ES_LEFT; break;
		case WINGUI_TEXTEDIT_ALIGN_RIGHT:  dwStyle |= ES_RIGHT; break;
		case WINGUI_TEXTEDIT_ALIGN_CENTER: dwStyle |= ES_CENTER; break;
		default: DebugAssert(false); break;
	}
	switch( pParameters->iCase ) {
		case WINGUI_TEXTEDIT_CASE_BOTH:  dwStyle |= 0; break;
		case WINGUI_TEXTEDIT_CASE_LOWER: dwStyle |= ES_LOWERCASE; break;
		case WINGUI_TEXTEDIT_CASE_UPPER: dwStyle |= ES_UPPERCASE; break;
		default: DebugAssert(false); break;
	}
	switch( pParameters->iMode ) {
		case WINGUI_TEXTEDIT_MODE_TEXT:     dwStyle |= 0; break;
		case WINGUI_TEXTEDIT_MODE_NUMERIC:  dwStyle |= ES_NUMBER; break;
		case WINGUI_TEXTEDIT_MODE_PASSWORD: dwStyle |= ES_PASSWORD; break;
		default: DebugAssert(false); break;
	}
	if ( pParameters->bAllowHorizontalScroll )
		dwStyle |= ES_AUTOHSCROLL;
	if ( pParameters->bDontHideSelection )
		dwStyle |= ES_NOHIDESEL;
	if ( pParameters->bReadOnly )
		dwStyle |= ES_READONLY;
	if ( pParameters->bEnableTabStop )
		dwStyle |= WS_TABSTOP;

    // Window creation
	m_hHandle = CreateWindowEx (
		0,
		WC_EDIT,
		pParameters->strInitialText,
		dwStyle,
		hWindowRect.iLeft, hWindowRect.iTop,
        hWindowRect.iWidth, hWindowRect.iHeight,
		hParentWnd,
		(HMENU)m_iResourceID,
		(HINSTANCE)( GetWindowLongPtr(hParentWnd,GWLP_HINSTANCE) ),
		NULL
	);
	DebugAssert( m_hHandle != NULL );

	// Done
	_SaveElementToHandle();
}
Void WinGUITextEdit::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUITextEdit::_DispatchEvent( Int iNotificationCode, Void * pParameters )
{
    // Get Model
	WinGUITextEditModel * pModel = (WinGUITextEditModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		case EN_SETFOCUS:  return pModel->OnFocusGained(); break;
		case EN_KILLFOCUS: return pModel->OnFocusLost(); break;

		case EN_CHANGE: return pModel->OnTextChange(); break;
		default: break;
	}

	// Unhandled
	return false;
}

