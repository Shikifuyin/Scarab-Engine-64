/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIProgressBar.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : ProgressBar
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
#include "WinGUIProgressBar.h"

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUIProgressBarModel implementation
WinGUIProgressBarModel::WinGUIProgressBarModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// Default Parameters
	m_hCreationParameters.bPendingMode = false;
	m_hCreationParameters.bSmoothWrap = false;
	m_hCreationParameters.bVertical = false;
}
WinGUIProgressBarModel::~WinGUIProgressBarModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIProgressBar implementation
WinGUIProgressBar::WinGUIProgressBar( WinGUIElement * pParent, WinGUIProgressBarModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	m_bPendingMode = false;
}
WinGUIProgressBar::~WinGUIProgressBar()
{
	// nothing to do
}

Void WinGUIProgressBar::TogglePendingMode( Bool bEnable )
{
	HWND hHandle = (HWND)m_hHandle;

	DWORD dwStyle = GetWindowLong( hHandle, GWL_STYLE );
	if ( bEnable ) {
		dwStyle |= PBS_MARQUEE;
	} else {
		dwStyle &= (~PBS_MARQUEE);
	}
	SetWindowLong( hHandle, GWL_STYLE, dwStyle );

	m_bPendingMode = bEnable;
}

UInt WinGUIProgressBar::GetBackgroundColor() const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, PBM_GETBKCOLOR, (WPARAM)0, (LPARAM)0 );
}
Void WinGUIProgressBar::SetBackgroundColor( UInt iColor )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, PBM_SETBKCOLOR, (WPARAM)0, (LPARAM)iColor );
}

UInt WinGUIProgressBar::GetBarColor() const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, PBM_GETBARCOLOR, (WPARAM)0, (LPARAM)0 );
}
Void WinGUIProgressBar::SetBarColor( UInt iColor )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, PBM_SETBARCOLOR, (WPARAM)0, (LPARAM)iColor );
}

Void WinGUIProgressBar::GetRange( Int * outLowerBound, Int * outUpperBound ) const
{
	HWND hHandle = (HWND)m_hHandle;

	PBRANGE hRange;
	SendMessage( hHandle, PBM_GETRANGE, (WPARAM)0, (LPARAM)&hRange );

	*outLowerBound = hRange.iLow;
	*outUpperBound = hRange.iHigh;
}
Void WinGUIProgressBar::SetRange( Int iLowerBound, Int iUpperBound )
{
	DebugAssert( iLowerBound < iUpperBound );

	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, PBM_SETRANGE32, (WPARAM)iLowerBound, (LPARAM)iUpperBound );
}

WinGUIProgressBarState WinGUIProgressBar::GetState() const
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iState = SendMessage( hHandle, PBM_GETSTATE, (WPARAM)0, (LPARAM)0 );

	switch( iState ) {
		case PBST_NORMAL: return WINGUI_PROGRESSBAR_INPROGRESS; break;
		case PBST_PAUSED: return WINGUI_PROGRESSBAR_PAUSED; break;
		case PBST_ERROR:  return WINGUI_PROGRESSBAR_ERROR; break;
		default: DebugAssert(false); break;
	}
	return WINGUI_PROGRESSBAR_ERROR; // Should never happen
}
WinGUIProgressBarState WinGUIProgressBar::SetState( WinGUIProgressBarState iState )
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iTmp = 0;
	switch( iState ) {
		case WINGUI_PROGRESSBAR_INPROGRESS: iTmp = PBST_NORMAL; break;
		case WINGUI_PROGRESSBAR_PAUSED:     iTmp = PBST_PAUSED; break;
		case WINGUI_PROGRESSBAR_ERROR:      iTmp = PBST_ERROR; break;
		default: DebugAssert(false); break;
	}

	iTmp = SendMessage( hHandle, PBM_SETSTATE, (WPARAM)iTmp, (LPARAM)0 );

	switch( iTmp ) {
		case PBST_NORMAL: return WINGUI_PROGRESSBAR_INPROGRESS; break;
		case PBST_PAUSED: return WINGUI_PROGRESSBAR_PAUSED; break;
		case PBST_ERROR:  return WINGUI_PROGRESSBAR_ERROR; break;
		default: DebugAssert(false); break;
	}
	return WINGUI_PROGRESSBAR_ERROR; // Should never happen
}

Int WinGUIProgressBar::GetBarPosition() const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, PBM_GETPOS, (WPARAM)0, (LPARAM)0 );
}
Int WinGUIProgressBar::SetBarPosition( Int iPosition )
{
	DebugAssert( !m_bPendingMode );

	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, PBM_SETPOS, (WPARAM)iPosition, (LPARAM)0 );
}

Int WinGUIProgressBar::Progress( Int iDelta )
{
	DebugAssert( !m_bPendingMode );

	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, PBM_DELTAPOS, (WPARAM)iDelta, (LPARAM)0 );
}

Int WinGUIProgressBar::GetStep() const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, PBM_GETSTEP, (WPARAM)0, (LPARAM)0 );
}
Int WinGUIProgressBar::SetStep( Int iStep )
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, PBM_SETSTEP, (WPARAM)iStep, (LPARAM)0 );
}

Int WinGUIProgressBar::Step()
{
	DebugAssert( !m_bPendingMode );

	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, PBM_STEPIT, (WPARAM)0, (LPARAM)0 );
}

Void WinGUIProgressBar::TogglePendingAnimation( Bool bEnable, UInt iUpdateIntervalMS )
{
	DebugAssert( m_bPendingMode );

	HWND hHandle = (HWND)m_hHandle;

	SendMessage( hHandle, PBM_SETMARQUEE, (WPARAM)(bEnable ? TRUE : FALSE), (LPARAM)iUpdateIntervalMS );
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIProgressBar::_Create()
{
	DebugAssert( m_hHandle == NULL );

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUIProgressBarModel * pModel = (WinGUIProgressBarModel*)m_pModel;

	// Compute Layout
    const WinGUILayout * pLayout = pModel->GetLayout();

    WinGUIRectangle hParentRect;
    m_pParent->GetClientRect( &hParentRect );

    WinGUIRectangle hWindowRect;
    pLayout->ComputeLayout( &hWindowRect, hParentRect );

	// Get Creation Parameters
    const WinGUIProgressBarParameters * pParameters = pModel->GetCreationParameters();

	// Save State
	m_bPendingMode = pParameters->bPendingMode;

    // Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE );
	if ( pParameters->bPendingMode )
		dwStyle |= PBS_MARQUEE;
	if ( pParameters->bSmoothWrap)
		dwStyle |= PBS_SMOOTHREVERSE;
	if ( pParameters->bVertical )
		dwStyle |= PBS_VERTICAL;

    // Window creation
	m_hHandle = CreateWindowEx (
		0,
		PROGRESS_CLASS,
		NULL,
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
Void WinGUIProgressBar::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUIProgressBar::_DispatchEvent( Int iNotificationCode, Void * pParameters )
{
    // Get Model
	WinGUIProgressBarModel * pModel = (WinGUIProgressBarModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		// Nothing to do !
		default: break;
	}

	// Unhandled
	return false;
}



