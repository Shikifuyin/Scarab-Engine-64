/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIStatusBar.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : StatusBar
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
#include "WinGUIStatusBar.h"

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUIProgressBarModel implementation
WinGUIStatusBarModel::WinGUIStatusBarModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// Default Parameters
	m_hCreationParameters.bHasSizingGrip = false;
	m_hCreationParameters.bEnableToolTips = false;
}
WinGUIStatusBarModel::~WinGUIStatusBarModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIStatusBar implementation
WinGUIStatusBar::WinGUIStatusBar( WinGUIElement * pParent, WinGUIStatusBarModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	// nothing to do
}
WinGUIStatusBar::~WinGUIStatusBar()
{
	// nothing to do
}

Bool WinGUIStatusBar::IsUnicode() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ( SendMessage(hHandle, SB_GETUNICODEFORMAT, (WPARAM)0, (LPARAM)0) != 0 );
}
Bool WinGUIStatusBar::IsANSI() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ( SendMessage(hHandle, SB_GETUNICODEFORMAT, (WPARAM)0, (LPARAM)0) == 0 );
}

Void WinGUIStatusBar::SetUnicode()
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, SB_SETUNICODEFORMAT, (WPARAM)TRUE, (LPARAM)0 );
}
Void WinGUIStatusBar::SetANSI()
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, SB_SETUNICODEFORMAT, (WPARAM)FALSE, (LPARAM)0 );
}

UInt WinGUIStatusBar::SetBackgroundColor( UInt iColor )
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, SB_SETBKCOLOR, (WPARAM)0, (LPARAM)iColor );
}

Void WinGUIStatusBar::GetPartIcon( WinGUIIcon * outIcon,  UInt iPartIndex )
{
	DebugAssert( !(outIcon->IsCreated()) );

	HWND hHandle = (HWND)m_hHandle;

	HICON hIcon = (HICON)( SendMessage(hHandle, SB_GETICON, (WPARAM)iPartIndex, (LPARAM)0) );

	outIcon->_CreateFromHandle( hIcon, true );
}
Void WinGUIStatusBar::SetPartIcon( UInt iPartIndex, const WinGUIIcon * pIcon )
{
	DebugAssert( pIcon->IsCreated() );

	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, SB_SETICON, (WPARAM)iPartIndex, (LPARAM)(pIcon->m_hHandle) );
}

Void WinGUIStatusBar::GetBorders( UInt * outHorizontalBorderWidth, UInt * outVerticalBorderWidth, UInt * outSeparatorsWidth ) const
{
	HWND hHandle = (HWND)m_hHandle;

	UInt arrWidths[3];
	SendMessage( hHandle, SB_GETBORDERS, (WPARAM)0, (LPARAM)arrWidths );

	*outHorizontalBorderWidth = arrWidths[0];
	*outVerticalBorderWidth = arrWidths[1];
	*outSeparatorsWidth = arrWidths[2];
}

Void WinGUIStatusBar::SetMinHeight( UInt iMinHeight )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, SB_SETMINHEIGHT, (WPARAM)iMinHeight, (LPARAM)0 );
	SendMessage( hHandle, WM_SIZE, (WPARAM)0, (LPARAM)0 ); // Redraw immediately
}

Bool WinGUIStatusBar::IsSinglePart() const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, SB_ISSIMPLE, (WPARAM)0, (LPARAM)0 );
}
Void WinGUIStatusBar::SetSinglePart( Bool bSetSinglePart )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, SB_SIMPLE, (WPARAM)(bSetSinglePart ? TRUE : FALSE), (LPARAM)0 );
}

UInt WinGUIStatusBar::GetPartCount() const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, SB_GETPARTS, (WPARAM)0, (LPARAM)NULL );
}
Void WinGUIStatusBar::GetParts( UInt * outPartEdges, UInt iMaxParts ) const
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, SB_GETPARTS, (WPARAM)iMaxParts, (LPARAM)outPartEdges );
}
Void WinGUIStatusBar::SetParts( const UInt * arrPartEdges, UInt iPartCount )
{
	DebugAssert( iPartCount < 256 );

	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, SB_SETPARTS, (WPARAM)iPartCount, (LPARAM)arrPartEdges );
}

Void WinGUIStatusBar::GetPartRect( WinGUIRectangle * outPartRect, UInt iPartIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	SendMessage( hHandle, SB_GETRECT, (WPARAM)iPartIndex, (LPARAM)&hRect );

	outPartRect->iLeft = hRect.left;
	outPartRect->iTop = hRect.top;
	outPartRect->iWidth = ( hRect.right - hRect.left );
	outPartRect->iHeight = ( hRect.bottom - hRect.top );
}

UInt WinGUIStatusBar::GetPartTextLength( WinGUIStatusBarDrawMode * outDrawMode, UInt iPartIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	DWord dwResult = SendMessage( hHandle, SB_GETTEXTLENGTH, (WPARAM)iPartIndex, (LPARAM)NULL );

	UInt iLength = LOWORD(dwResult);
	UInt iDrawMode = HIWORD(dwResult);

	switch( iDrawMode ) {
		case SBT_NOBORDERS: *outDrawMode = WINGUI_STATUSBAR_DRAW_NOBORDER; break;
		case 0:             *outDrawMode = WINGUI_STATUSBAR_DRAW_SINKBORDER; break;
		case SBT_POPOUT:    *outDrawMode = WINGUI_STATUSBAR_DRAW_RAISEBORDER; break;
		default: DebugAssert(false); break;
	}

	return iLength;
}
Void WinGUIStatusBar::GetPartText( GChar * outPartText, UInt iPartIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, SB_GETTEXT, (WPARAM)iPartIndex, (LPARAM)outPartText );
}
Void WinGUIStatusBar::SetPartText( UInt iPartIndex, const GChar * strPartText, WinGUIStatusBarDrawMode iDrawMode ) const
{
	HWND hHandle = (HWND)m_hHandle;

	Word wParam = 0;
	switch( iDrawMode ) {
		case WINGUI_STATUSBAR_DRAW_NOBORDER:    wParam = SBT_NOBORDERS; break;
		case WINGUI_STATUSBAR_DRAW_SINKBORDER:  wParam = 0; break;
		case WINGUI_STATUSBAR_DRAW_RAISEBORDER: wParam = SBT_POPOUT; break;
		default: DebugAssert(false); break;
	}
	wParam <<= 8;
	wParam |= ( iPartIndex & 0xff );

	SendMessage( hHandle, SB_SETTEXT, (WPARAM)wParam, (LPARAM)strPartText );
}

Void WinGUIStatusBar::GetPartTipText( GChar * outPartTipText, UInt iMaxLength, UInt iPartIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	DWord dwParam = ( iMaxLength & 0xffff );
	dwParam <<= 16;
	dwParam |= ( iPartIndex & 0xffff );

	SendMessage( hHandle, SB_GETTIPTEXT, (WPARAM)dwParam, (LPARAM)outPartTipText );
}
Void WinGUIStatusBar::SetPartTipText( UInt iPartIndex, const GChar * strPartTipText ) const
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, SB_SETTIPTEXT, (WPARAM)iPartIndex, (LPARAM)strPartTipText );
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIStatusBar::_Create()
{
	DebugAssert( m_hHandle == NULL );

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUIStatusBarModel * pModel = (WinGUIStatusBarModel*)m_pModel;

    // Don't use Layout

	// Get Creation Parameters
    const WinGUIStatusBarParameters * pParameters = pModel->GetCreationParameters();

    // Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE );
	if ( pParameters->bHasSizingGrip )
		dwStyle |= SBARS_SIZEGRIP;
	if ( pParameters->bEnableToolTips )
		dwStyle |= SBARS_TOOLTIPS;

    // Window creation
	m_hHandle = CreateWindowEx (
		0,
		STATUSCLASSNAME,
		NULL,
		dwStyle,
		0, 0, // Position ignored
        0, 0, // Size ignored
		hParentWnd,
		(HMENU)m_iResourceID,
		(HINSTANCE)( GetWindowLongPtr(hParentWnd,GWLP_HINSTANCE) ),
		NULL
	);
	DebugAssert( m_hHandle != NULL );

	// Done
	_SaveElementToHandle();
}
Void WinGUIStatusBar::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUIStatusBar::_DispatchEvent( Int iNotificationCode, Void * pParameters )
{
    // Get Model
	WinGUIStatusBarModel * pModel = (WinGUIStatusBarModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		// Nothing to do !
		default: break;
	}

	// Unhandled
	return false;
}

