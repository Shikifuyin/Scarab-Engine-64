/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIToolTip.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : ToolTip
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
#include "WinGUIToolTip.h"

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUIToolTipModel implementation
WinGUIToolTipModel::WinGUIToolTipModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// Default Parameters
	m_hCreationParameters.bAlwaysTip = true;
	m_hCreationParameters.bBalloonTip = false;
	m_hCreationParameters.bBalloonCloseButton = false;
	m_hCreationParameters.bNoSlidingAnimation = false;
	m_hCreationParameters.bNoFadingAnimation = false;
}
WinGUIToolTipModel::~WinGUIToolTipModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIToolTip implementation
WinGUIToolTip::WinGUIToolTip( WinGUIElement * pParent, WinGUIToolTipModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	// nothing to do
}
WinGUIToolTip::~WinGUIToolTip()
{
	// nothing to do
}

Void WinGUIToolTip::Enable()
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TTM_ACTIVATE, (WPARAM)TRUE, (LPARAM)0 );
}
Void WinGUIToolTip::Disable()
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TTM_ACTIVATE, (WPARAM)FALSE, (LPARAM)0 );
}

Void WinGUIToolTip::Show()
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TTM_POPUP, (WPARAM)0, (LPARAM)0 );
}
Void WinGUIToolTip::Hide()
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TTM_POP, (WPARAM)0, (LPARAM)0 );
}

Void WinGUIToolTip::ForceRedraw()
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TTM_UPDATE, (WPARAM)0, (LPARAM)0 );
}

UInt WinGUIToolTip::GetBackgroundColor() const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, TTM_GETTIPBKCOLOR, (WPARAM)0, (LPARAM)0 );
}
Void WinGUIToolTip::SetBackgroundColor( UInt iColor )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TTM_SETTIPBKCOLOR, (WPARAM)iColor, (LPARAM)0 );
}

UInt WinGUIToolTip::GetTextColor() const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, TTM_GETTIPTEXTCOLOR, (WPARAM)0, (LPARAM)0 );
}
Void WinGUIToolTip::SetTextColor( UInt iColor )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TTM_SETTIPTEXTCOLOR, (WPARAM)iColor, (LPARAM)0 );
}

Void WinGUIToolTip::GetMargin( WinGUIPoint * outMarginLeftTop, WinGUIPoint * outMarginRightBottom ) const
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	SendMessage( hHandle, TTM_GETMARGIN, (WPARAM)0, (LPARAM)&hRect );

	outMarginLeftTop->iX = hRect.left;
	outMarginLeftTop->iY = hRect.top;
	outMarginRightBottom->iX = hRect.right;
	outMarginRightBottom->iY = hRect.bottom;
}
Void WinGUIToolTip::SetMargin( const WinGUIPoint & hMarginLeftTop, const WinGUIPoint & hMarginRightBottom )
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	hRect.left = hMarginLeftTop.iX;
	hRect.top = hMarginLeftTop.iY;
	hRect.right = hMarginRightBottom.iX;
	hRect.bottom = hMarginRightBottom.iY;

	SendMessage( hHandle, TTM_SETMARGIN, (WPARAM)0, (LPARAM)&hRect );
}

Void WinGUIToolTip::ToolTipRectToTextRect( WinGUIRectangle * pRect ) const
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	hRect.left = pRect->iLeft;
	hRect.top = pRect->iTop;
	hRect.right = ( pRect->iLeft + pRect->iWidth );
	hRect.bottom = ( pRect->iTop + pRect->iHeight );

	SendMessage( hHandle, TTM_ADJUSTRECT, (WPARAM)FALSE, (LPARAM)&hRect );

	pRect->iLeft = hRect.left;
	pRect->iTop = hRect.top;
	pRect->iWidth = ( hRect.right - hRect.left );
	pRect->iHeight = ( hRect.bottom - hRect.top );
}
Void WinGUIToolTip::TextRectToToolTipRect( WinGUIRectangle * pRect ) const
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	hRect.left = pRect->iLeft;
	hRect.top = pRect->iTop;
	hRect.right = ( pRect->iLeft + pRect->iWidth );
	hRect.bottom = ( pRect->iTop + pRect->iHeight );

	SendMessage( hHandle, TTM_ADJUSTRECT, (WPARAM)TRUE, (LPARAM)&hRect );

	pRect->iLeft = hRect.left;
	pRect->iTop = hRect.top;
	pRect->iWidth = ( hRect.right - hRect.left );
	pRect->iHeight = ( hRect.bottom - hRect.top );
}

UInt WinGUIToolTip::GetMaxWidth() const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, TTM_GETMAXTIPWIDTH, (WPARAM)0, (LPARAM)0 );
}
Void WinGUIToolTip::SetMaxWidth( UInt iMaxWidth )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TTM_SETMAXTIPWIDTH, (WPARAM)0, (LPARAM)iMaxWidth );
}

Void WinGUIToolTip::GetTitle( GChar * outTitle, UInt iMaxLength, WinGUIToolTipIcon * outIcon ) const
{
	HWND hHandle = (HWND)m_hHandle;

	TTGETTITLE hTitle;
	hTitle.dwSize = sizeof(TTGETTITLE);
	hTitle.uTitleBitmap = 0;
	hTitle.pszTitle = outTitle;
	hTitle.cch = iMaxLength;

	SendMessage( hHandle, TTM_GETTITLE, (WPARAM)0, (LPARAM)&hTitle );

	switch( hTitle.uTitleBitmap ) {
		case TTI_NONE:    *outIcon = WINGUI_TOOLTIP_ICON_NONE; break;
		case TTI_INFO:    *outIcon = WINGUI_TOOLTIP_ICON_INFO; break;
		case TTI_WARNING: *outIcon = WINGUI_TOOLTIP_ICON_WARNING; break;
		case TTI_ERROR:   *outIcon = WINGUI_TOOLTIP_ICON_ERROR; break;
		default: DebugAssert(false); break;
	}
}
Void WinGUIToolTip::SetTitle( const GChar * strTitle, WinGUIToolTipIcon iIcon )
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iIconID = 0;
	switch( iIcon ) {
		case WINGUI_TOOLTIP_ICON_NONE:    iIconID = TTI_NONE; break;
		case WINGUI_TOOLTIP_ICON_INFO:    iIconID = TTI_INFO; break;
		case WINGUI_TOOLTIP_ICON_WARNING: iIconID = TTI_WARNING; break;
		case WINGUI_TOOLTIP_ICON_ERROR:   iIconID = TTI_ERROR; break;
		default: DebugAssert(false); break;
	}

	SendMessage( hHandle, TTM_SETTITLE, (WPARAM)iIconID, (LPARAM)strTitle );
}

Void WinGUIToolTip::GetTimers( WinGUIToolTipTimers * outTimers ) const
{
	HWND hHandle = (HWND)m_hHandle;

	outTimers->iDelayMS = SendMessage( hHandle, TTM_GETDELAYTIME, (WPARAM)TTDT_INITIAL, (LPARAM)0 );
	outTimers->iDurationMS = SendMessage( hHandle, TTM_GETDELAYTIME, (WPARAM)TTDT_AUTOPOP, (LPARAM)0 );
	outTimers->iIntervalMS = SendMessage( hHandle, TTM_GETDELAYTIME, (WPARAM)TTDT_RESHOW, (LPARAM)0 );
}
Void WinGUIToolTip::SetTimers( const WinGUIToolTipTimers & hTimers )
{
	HWND hHandle = (HWND)m_hHandle;

	SendMessage( hHandle, TTM_SETDELAYTIME, (WPARAM)TTDT_INITIAL, (LPARAM)LOWORD(hTimers.iDelayMS) );
	SendMessage( hHandle, TTM_SETDELAYTIME, (WPARAM)TTDT_AUTOPOP, (LPARAM)LOWORD(hTimers.iDurationMS) );
	SendMessage( hHandle, TTM_SETDELAYTIME, (WPARAM)TTDT_RESHOW, (LPARAM)LOWORD(hTimers.iIntervalMS) );
}

Void WinGUIToolTip::RegisterTool( const WinGUIToolTipInfos & hToolTipInfos )
{
	HWND hHandle = (HWND)m_hHandle;

	TOOLINFO hToolInfos;
	_Convert_ToolTipInfos( &hToolInfos, &hToolTipInfos );

	SendMessage( hHandle, TTM_ADDTOOL, (WPARAM)0, (LPARAM)&hToolInfos );
}
Void WinGUIToolTip::UnregisterTool( WinGUIElement * pToolElement )
{
	DebugAssert( m_pParent == pToolElement->GetParent() );

	HWND hHandle = (HWND)m_hHandle;

	TOOLINFO hToolInfos;
	hToolInfos.cbSize = sizeof(TOOLINFO);
	hToolInfos.lpReserved = NULL;
	hToolInfos.uFlags = TTF_IDISHWND;

	hToolInfos.hwnd = (HWND)( _GetHandle(m_pParent) );
	hToolInfos.uId = (UINT_PTR)( _GetHandle(pToolElement) );

	SendMessage( hHandle, TTM_DELTOOL, (WPARAM)0, (LPARAM)&hToolInfos );
}

UInt WinGUIToolTip::GetToolCount() const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, TTM_GETTOOLCOUNT, (WPARAM)0, (LPARAM)0 );
}

Bool WinGUIToolTip::GetTool( WinGUIToolTipInfos * outToolTipInfos, GChar * outToolTipText, UInt iMaxLength, UInt iToolIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	GChar strBuffer[128]; // Large enough to be safe
	TOOLINFO hToolInfos;
	hToolInfos.cbSize = sizeof(TOOLINFO);
	hToolInfos.lpReserved = NULL;
	hToolInfos.lpszText = strBuffer;

	BOOL bValid = SendMessage( hHandle, TTM_ENUMTOOLS, (WPARAM)iToolIndex, (LPARAM)&hToolInfos );
	if ( !bValid )
		return false;

	_Convert_ToolTipInfos( outToolTipInfos, &hToolInfos );

	if ( outToolTipText != NULL )
		StringFn->NCopy( outToolTipText, strBuffer, iMaxLength - 1 );

	return true;
}

Void WinGUIToolTip::GetToolSize( WinGUIPoint * outSize, const WinGUIToolTipInfos & hToolTipInfos ) const
{
	HWND hHandle = (HWND)m_hHandle;

	TOOLINFO hToolInfo;
	_Convert_ToolTipInfos( &hToolInfo, &hToolTipInfos );

	DWORD dwSize = SendMessage( hHandle, TTM_GETBUBBLESIZE, (WPARAM)0, (LPARAM)&hToolInfo );

	outSize->iX = LOWORD(dwSize);
	outSize->iY = HIWORD(dwSize);
}
Void WinGUIToolTip::GetToolText( GChar * outToolTipText, UInt iMaxLength, WinGUIElement * pToolElement ) const
{
	DebugAssert( m_pParent == pToolElement->GetParent() );

	HWND hHandle = (HWND)m_hHandle;

	TOOLINFO hToolInfos;
	hToolInfos.cbSize = sizeof(TOOLINFO);
	hToolInfos.lpReserved = NULL;
	hToolInfos.uFlags = TTF_IDISHWND;
	
	hToolInfos.hwnd = (HWND)( _GetHandle(m_pParent) );
	hToolInfos.uId = (UINT_PTR)( _GetHandle(pToolElement) );

	hToolInfos.lpszText = outToolTipText;

	SendMessage( hHandle, TTM_GETTEXT, (WPARAM)iMaxLength, (LPARAM)&hToolInfos );
}

Bool WinGUIToolTip::HasCurrentTool() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ( SendMessage(hHandle, TTM_GETCURRENTTOOL, (WPARAM)0, (LPARAM)NULL) != 0 );
}
Void WinGUIToolTip::GetCurrentTool( WinGUIToolTipInfos * outToolTipInfos, GChar * outToolTipText, UInt iMaxLength ) const
{
	HWND hHandle = (HWND)m_hHandle;

	GChar strBuffer[128]; // Large enough to be safe
	TOOLINFO hToolInfos;
	hToolInfos.cbSize = sizeof(TOOLINFO);
	hToolInfos.lpReserved = NULL;
	hToolInfos.lpszText = strBuffer;

	SendMessage( hHandle, TTM_GETCURRENTTOOL, (WPARAM)0, (LPARAM)&hToolInfos );

	_Convert_ToolTipInfos( outToolTipInfos, &hToolInfos );

	if ( outToolTipText != NULL )
		StringFn->NCopy( outToolTipText, strBuffer, iMaxLength - 1 );
}

Void WinGUIToolTip::ToggleTracking( WinGUIElement * pToolElement, Bool bEnable )
{
	DebugAssert( m_pParent == pToolElement->GetParent() );

	HWND hHandle = (HWND)m_hHandle;

	TOOLINFO hToolInfos;
	hToolInfos.cbSize = sizeof(TOOLINFO);
	hToolInfos.lpReserved = NULL;
	hToolInfos.uFlags = TTF_IDISHWND;

	hToolInfos.hwnd = (HWND)( _GetHandle(m_pParent) );
	hToolInfos.uId = (UINT_PTR)( _GetHandle(pToolElement) );

	SendMessage( hHandle, TTM_TRACKACTIVATE, (WPARAM)(bEnable ? TRUE : FALSE), (LPARAM)&hToolInfos );
}
Void WinGUIToolTip::SetTrackPosition( const WinGUIPoint & hPosition )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwPosition = ( hPosition.iY & 0xffff );
	dwPosition <<= 16;
	dwPosition |= ( hPosition.iX & 0xffff );

	SendMessage( hHandle, TTM_TRACKPOSITION, (WPARAM)0, (LPARAM)dwPosition );
}

Bool WinGUIToolTip::HitTest( WinGUIToolTipInfos * outToolTipInfos, const WinGUIPoint & hPosition, WinGUIElement * pToolElement ) const
{
	DebugAssert( m_pParent == pToolElement->GetParent() );

	HWND hHandle = (HWND)m_hHandle;

	TTHITTESTINFO hHitTestInfos;
	hHitTestInfos.pt.x = hPosition.iX;
	hHitTestInfos.pt.y = hPosition.iY;
	hHitTestInfos.hwnd = (HWND)( _GetHandle(pToolElement) );
	hHitTestInfos.ti.cbSize = sizeof(TOOLINFO);
	hHitTestInfos.ti.lpReserved = NULL;

	BOOL bHit = SendMessage( hHandle, TTM_HITTEST, (WPARAM)0, (LPARAM)&hHitTestInfos );

	if ( bHit )
		_Convert_ToolTipInfos( outToolTipInfos, &(hHitTestInfos.ti) );

	return ( bHit != FALSE );
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIToolTip::_Convert_ToolTipInfos( WinGUIToolTipInfos * outToolTipInfos, const Void * pToolTipInfos ) const
{
	const TOOLINFO * pDesc = (const TOOLINFO *)pToolTipInfos;

	if ( (pDesc->uFlags & TTF_TRACK) != 0 ) {
		if ( (pDesc->uFlags & TTF_ABSOLUTE) != 0 )
			outToolTipInfos->iTrackingMode = WINGUI_TOOLTIP_TRACKING_ABSOLUTE;
		else
			outToolTipInfos->iTrackingMode = WINGUI_TOOLTIP_TRACKING_RELATIVE;
	} else
		outToolTipInfos->iTrackingMode = WINGUI_TOOLTIP_TRACKING_DISABLED;

	outToolTipInfos->bCentered = false;
	if ( (pDesc->uFlags & TTF_CENTERTIP) != 0 )
		outToolTipInfos->bCentered = true;

	outToolTipInfos->bForwardMouseEvents = false;
	if ( (pDesc->uFlags & TTF_TRANSPARENT) != 0 )
		outToolTipInfos->bForwardMouseEvents = true;

	HWND hParentWnd = pDesc->hwnd;
	DebugAssert( hParentWnd == _GetHandle(m_pParent) );

	HWND hToolWnd = (HWND)( pDesc->uId );
	outToolTipInfos->pToolElement = _GetElementFromHandle( hToolWnd );
	DebugAssert( outToolTipInfos->pToolElement != NULL );

	outToolTipInfos->pUserData = (Void*)( pDesc->lParam );
}
Void WinGUIToolTip::_Convert_ToolTipInfos( Void * outToolTipInfos, const WinGUIToolTipInfos * pToolTipInfos ) const
{
	DebugAssert( m_pParent == pToolTipInfos->pToolElement->GetParent() );

	TOOLINFO * outDesc = (TOOLINFO*)outToolTipInfos;

	outDesc->cbSize = sizeof(TOOLINFO);
	outDesc->lpReserved = NULL;
	outDesc->uFlags = ( TTF_SUBCLASS | TTF_IDISHWND | TTF_PARSELINKS );

	switch( pToolTipInfos->iTrackingMode ) {
		case WINGUI_TOOLTIP_TRACKING_DISABLED: outDesc->uFlags |= 0; break;
		case WINGUI_TOOLTIP_TRACKING_RELATIVE: outDesc->uFlags |= TTF_TRACK; break;
		case WINGUI_TOOLTIP_TRACKING_ABSOLUTE: outDesc->uFlags |= (TTF_TRACK | TTF_ABSOLUTE); break;
		default: DebugAssert(false); break;
	}
	if ( pToolTipInfos->bCentered )
		outDesc->uFlags |= TTF_CENTERTIP;
	if ( pToolTipInfos->bForwardMouseEvents )
		outDesc->uFlags |= TTF_TRANSPARENT;

	outDesc->hinst = NULL; // Don't use string resources
	//outDesc->rect; // Not needed, we use a HWND as Identifier

	outDesc->hwnd = (HWND)( _GetHandle(m_pParent) );
	outDesc->uId = (UINT_PTR)( _GetHandle(pToolTipInfos->pToolElement) );

	outDesc->lpszText = LPSTR_TEXTCALLBACK;
	outDesc->lParam = (LPARAM)( pToolTipInfos->pUserData );
}

Void WinGUIToolTip::_Create()
{
	DebugAssert( m_hHandle == NULL );

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUIToolTipModel * pModel = (WinGUIToolTipModel*)m_pModel;

	// Don't use Layout

	// Get Creation Parameters
    const WinGUIToolTipParameters * pParameters = pModel->GetCreationParameters();

    // Build Style
	DWord dwStyle = ( WS_POPUP | TTS_NOPREFIX | TTS_USEVISUALSTYLE );
	if ( pParameters->bAlwaysTip )
		dwStyle |= TTS_ALWAYSTIP;
	if ( pParameters->bBalloonTip ) {
		dwStyle |= TTS_BALLOON;
		if ( pParameters->bBalloonCloseButton )
			dwStyle |= TTS_CLOSE;
	}
	if ( pParameters->bNoSlidingAnimation )
		dwStyle |= TTS_NOANIMATE;
	if ( pParameters->bNoFadingAnimation )
		dwStyle |= TTS_NOFADE;

    // Window creation
	m_hHandle = CreateWindowEx (
		WS_EX_TOOLWINDOW,
		TOOLTIPS_CLASS,
		NULL,
		dwStyle,
		CW_USEDEFAULT, CW_USEDEFAULT,
        CW_USEDEFAULT, CW_USEDEFAULT,
		hParentWnd,
		(HMENU)m_iResourceID,
		(HINSTANCE)( GetWindowLongPtr(hParentWnd,GWLP_HINSTANCE) ),
		NULL
	);
	DebugAssert( m_hHandle != NULL );

	// Done
	_SaveElementToHandle();
}
Void WinGUIToolTip::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUIToolTip::_DispatchEvent( Int iNotificationCode, Void * pParameters )
{
    // Get Model
	WinGUIToolTipModel * pModel = (WinGUIToolTipModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		
		// Show/Hide
		case TTN_SHOW: return pModel->OnShowTip(); break;
		case TTN_POP:  return pModel->OnHideTip(); break;

		// Link
		case TTN_LINKCLICK: return pModel->OnLinkClick(); break;

		// ToolTip Text Requests
		case TTN_GETDISPINFO: {
				NMTTDISPINFO * pParams = (NMTTDISPINFO*)pParameters;
				DebugAssert( pParams->hinst == NULL );
				DebugAssert( (pParams->uFlags & TTF_IDISHWND) != 0 );

				HWND hToolWnd = (HWND)( pParams->hdr.idFrom );
				WinGUIElement * pTool = _GetElementFromHandle( hToolWnd );

				pParams->lpszText = pModel->OnRequestToolTipText( pTool, (Void*)(pParams->lParam) );

				return true;
			} break;

		default: break;
	}

	// Unhandled
	return false;
}

