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
	// nothing to do
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

	WinGUITextEditModel * pModel = (WinGUITextEditModel*)m_pModel;
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

	DWord dwStyle = 0;
	if ( pModel->DontHideSelection() )
		dwStyle |= ES_NOHIDESEL;
	if ( pModel->AllowHorizScroll() )
		dwStyle |= ES_AUTOHSCROLL;
	if ( pModel->IsReadOnly() )
		dwStyle |= ES_READONLY;
	switch( pModel->GetTextAlign() ) {
		case WINGUI_TEXTEDIT_ALIGN_LEFT:   dwStyle |= ES_LEFT; break;
		case WINGUI_TEXTEDIT_ALIGN_RIGHT:  dwStyle |= ES_RIGHT; break;
		case WINGUI_TEXTEDIT_ALIGN_CENTER: dwStyle |= ES_CENTER; break;
		default: DebugAssert(false); break;
	}
	switch( pModel->GetTextCase() ) {
		case WINGUI_TEXTEDIT_CASE_BOTH:  dwStyle |= 0; break;
		case WINGUI_TEXTEDIT_CASE_LOWER: dwStyle |= ES_LOWERCASE; break;
		case WINGUI_TEXTEDIT_CASE_UPPER: dwStyle |= ES_UPPERCASE; break;
		default: DebugAssert(false); break;
	}
	switch( pModel->GetTextMode() ) {
		case WINGUI_TEXTEDIT_MODE_TEXT:     dwStyle |= 0; break;
		case WINGUI_TEXTEDIT_MODE_NUMERIC:  dwStyle |= ES_NUMBER; break;
		case WINGUI_TEXTEDIT_MODE_PASSWORD: dwStyle |= ES_PASSWORD; break;
		default: DebugAssert(false); break;
	}

	m_hHandle = CreateWindowEx (
		0, WC_EDIT, pModel->GetInitialText(),
		WS_VISIBLE | WS_CHILD | WS_TABSTOP | WS_BORDER | dwStyle,
		pModel->GetPositionX(),	pModel->GetPositionY(),
		pModel->GetWidth(), pModel->GetHeight(),
		hParentWnd, (HMENU)m_iResourceID,
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

	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUITextEdit::_DispatchEvent( Int iNotificationCode )
{
	WinGUITextEditModel * pModel = (WinGUITextEditModel*)m_pModel;

	// Dispatch Event to our Model
	switch( iNotificationCode ) {
		case EN_CHANGE:
			return pModel->OnTextChange();
			break;
		default: break;
	}

	// Unhandled
	return false;
}

