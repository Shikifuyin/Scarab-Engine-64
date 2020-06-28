/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUISliderBar.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : SliderBar
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
#include "WinGUISliderBar.h"

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUISliderBarModel implementation
WinGUISliderBarModel::WinGUISliderBarModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// Default Parameters
	StringFn->Copy( m_hCreationParameters.strLabel, TEXT("SliderBar") );
	m_hCreationParameters.bTransparentBackground = false;
	m_hCreationParameters.iTickMarks = WINGUI_SLIDERBAR_TICKMARKS_NONE;

	m_hCreationParameters.bAllowSelectionRange = false;
	m_hCreationParameters.bResizableSlider = false;
	m_hCreationParameters.bNoSlider = false;

	m_hCreationParameters.bReversedValues = false;
	m_hCreationParameters.bReverseEdgeMapping = false;

	m_hCreationParameters.bVertical = false;

	m_hCreationParameters.bEnableToolTips = false;
}
WinGUISliderBarModel::~WinGUISliderBarModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUISliderBar implementation
WinGUISliderBar::WinGUISliderBar( WinGUIElement * pParent, WinGUISliderBarModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	// nothing to do
}
WinGUISliderBar::~WinGUISliderBar()
{
	// nothing to do
}

Bool WinGUISliderBar::IsUnicode() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ( SendMessage(hHandle, TBM_GETUNICODEFORMAT, (WPARAM)0, (LPARAM)0) != 0 );
}
Bool WinGUISliderBar::IsANSI() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ( SendMessage(hHandle, TBM_GETUNICODEFORMAT, (WPARAM)0, (LPARAM)0) == 0 );
}

Void WinGUISliderBar::SetUnicode()
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TBM_SETUNICODEFORMAT, (WPARAM)TRUE, (LPARAM)0 );
}
Void WinGUISliderBar::SetANSI()
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TBM_SETUNICODEFORMAT, (WPARAM)FALSE, (LPARAM)0 );
}

Void WinGUISliderBar::GetTrackRect( WinGUIRectangle * outTrackRect ) const
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	SendMessage( hHandle, TBM_GETCHANNELRECT, (WPARAM)0, (LPARAM)&hRect );

	outTrackRect->iLeft = hRect.left;
	outTrackRect->iTop = hRect.top;
	outTrackRect->iWidth = ( hRect.right - hRect.left );
	outTrackRect->iHeight = ( hRect.bottom - hRect.top );
}
Void WinGUISliderBar::GetSliderRect( WinGUIRectangle * outSliderRect ) const
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	SendMessage( hHandle, TBM_GETTHUMBRECT, (WPARAM)0, (LPARAM)&hRect );

	outSliderRect->iLeft = hRect.left;
	outSliderRect->iTop = hRect.top;
	outSliderRect->iWidth = ( hRect.right - hRect.left );
	outSliderRect->iHeight = ( hRect.bottom - hRect.top );
}

UInt WinGUISliderBar::GetSliderWidth() const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, TBM_GETTHUMBLENGTH, (WPARAM)0, (LPARAM)0 );
}
Void WinGUISliderBar::SetSliderWidth( UInt iWidth )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TBM_SETTHUMBLENGTH, (WPARAM)iWidth, (LPARAM)0 );
}

Void WinGUISliderBar::GetRange( UInt * outMinPosition, UInt * outMaxPosition ) const
{
	HWND hHandle = (HWND)m_hHandle;
	*outMinPosition = SendMessage( hHandle, TBM_GETRANGEMIN, (WPARAM)0, (LPARAM)0 );
	*outMaxPosition = SendMessage( hHandle, TBM_GETRANGEMAX, (WPARAM)0, (LPARAM)0 );
}
Void WinGUISliderBar::SetRangeMin( UInt iMinPosition, Bool bRedraw )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TBM_SETRANGEMIN, (WPARAM)(bRedraw ? TRUE : FALSE), (LPARAM)iMinPosition );
}
Void WinGUISliderBar::SetRangeMax( UInt iMaxPosition, Bool bRedraw )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TBM_SETRANGEMAX, (WPARAM)(bRedraw ? TRUE : FALSE), (LPARAM)iMaxPosition );
}

UInt WinGUISliderBar::GetSmallSlideAmount() const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, TBM_GETLINESIZE, (WPARAM)0, (LPARAM)0 );
}
Void WinGUISliderBar::SetSmallSlideAmount( UInt iDeltaPos ) const
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TBM_SETLINESIZE, (WPARAM)0, (LPARAM)iDeltaPos );
}

UInt WinGUISliderBar::GetLargeSlideAmount() const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, TBM_GETPAGESIZE, (WPARAM)0, (LPARAM)0 );
}
Void WinGUISliderBar::SetLargeSlideAmount( UInt iDeltaPos ) const
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TBM_SETPAGESIZE, (WPARAM)0, (LPARAM)iDeltaPos );
}

UInt WinGUISliderBar::GetSliderPosition() const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, TBM_GETPOS, (WPARAM)0, (LPARAM)0 );
}
Void WinGUISliderBar::SetSliderPosition( UInt iPosition, Bool bRedraw ) const
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TBM_SETPOS, (WPARAM)(bRedraw ? TRUE : FALSE), (LPARAM)iPosition );
}

Void WinGUISliderBar::SetTickMarksFrequency( UInt iFrequency )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TBM_SETTICFREQ, (WPARAM)iFrequency, (LPARAM)0 );
}

UInt WinGUISliderBar::GetTickMarksCount() const
{
	HWND hHandle = (HWND)m_hHandle;
	UInt iCount = SendMessage( hHandle, TBM_GETNUMTICS, (WPARAM)0, (LPARAM)0 );

	// WINGUI_SLIDERBAR_TICKMARKS_NONE case
	if ( iCount == 0 )
		return 0;

	// Remove that +2 for end points making everything confusing !
	return iCount - 2;
}

UInt WinGUISliderBar::GetTickMarkPosition( UInt iTickMark ) const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, TBM_GETTIC, (WPARAM)iTickMark, (LPARAM)0 );
}
UInt * WinGUISliderBar::GetTickMarkPositions() const
{
	HWND hHandle = (HWND)m_hHandle;
	return (UInt*)( SendMessage(hHandle, TBM_GETPTICS, (WPARAM)0, (LPARAM)0) );
}

UInt WinGUISliderBar::GetTickMarkClientPos( UInt iTickMark ) const
{
	HWND hHandle = (HWND)m_hHandle;
	return SendMessage( hHandle, TBM_GETTICPOS, (WPARAM)iTickMark, (LPARAM)0 );
}


Void WinGUISliderBar::AddTickMark( UInt iPosition )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TBM_SETTIC, (WPARAM)0, (LPARAM)iPosition );
}
Void WinGUISliderBar::ClearTickMarks( Bool bRedraw )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TBM_CLEARTICS, (WPARAM)(bRedraw ? TRUE : FALSE), (LPARAM)0 );
}

Void WinGUISliderBar::GetSelectionRange( UInt * outSelectionStart, UInt * outSelectionEnd ) const
{
	HWND hHandle = (HWND)m_hHandle;
	*outSelectionStart = SendMessage( hHandle, TBM_GETSELSTART, (WPARAM)0, (LPARAM)0 );
	*outSelectionEnd = SendMessage( hHandle, TBM_GETSELEND, (WPARAM)0, (LPARAM)0 );
}
Void WinGUISliderBar::SetSelectionStart( UInt iStartPosition, Bool bRedraw )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TBM_SETSELSTART, (WPARAM)(bRedraw ? TRUE : FALSE), (LPARAM)iStartPosition );
}
Void WinGUISliderBar::SetSelectionEnd( UInt iEndPosition, Bool bRedraw )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TBM_SETSELEND, (WPARAM)(bRedraw ? TRUE : FALSE), (LPARAM)iEndPosition );
}

Void WinGUISliderBar::ClearSelection( Bool bRedraw )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, TBM_CLEARSEL, (WPARAM)(bRedraw ? TRUE : FALSE), (LPARAM)0 );
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUISliderBar::_Create()
{
	DebugAssert( m_hHandle == NULL );

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUISliderBarModel * pModel = (WinGUISliderBarModel*)m_pModel;

	// Compute Layout
    const WinGUILayout * pLayout = pModel->GetLayout();

    WinGUIRectangle hParentRect;
    m_pParent->GetClientRect( &hParentRect );

    WinGUIRectangle hWindowRect;
    pLayout->ComputeLayout( &hWindowRect, hParentRect );

	// Get Creation Parameters
    const WinGUISliderBarParameters * pParameters = pModel->GetCreationParameters();

    // Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE | TBS_NOTIFYBEFOREMOVE );
	if ( pParameters->bTransparentBackground )
		dwStyle |= TBS_TRANSPARENTBKGND;
	switch( pParameters->iTickMarks ) {
		case WINGUI_SLIDERBAR_TICKMARKS_NONE:
			dwStyle |= TBS_NOTICKS;
			break;
		case WINGUI_SLIDERBAR_TICKMARKS_BELOWRIGHT:
			dwStyle |= TBS_AUTOTICKS;
			if ( pParameters->bVertical )
				dwStyle |= TBS_RIGHT;
			else
				dwStyle |= TBS_BOTTOM;
			break;
		case WINGUI_SLIDERBAR_TICKMARKS_ABOVELEFT:
			dwStyle |= TBS_AUTOTICKS;
			if ( pParameters->bVertical )
				dwStyle |= TBS_LEFT;
			else
				dwStyle |= TBS_TOP;
			break;
		case WINGUI_SLIDERBAR_TICKMARKS_BOTH:
			dwStyle |= ( TBS_AUTOTICKS | TBS_BOTH );
			break;
		default: DebugAssert(false); break;
	}

	if ( pParameters->bAllowSelectionRange )
		dwStyle |= TBS_ENABLESELRANGE;
	if ( pParameters->bResizableSlider )
		dwStyle |= TBS_FIXEDLENGTH;
	if ( pParameters->bNoSlider )
		dwStyle |= TBS_NOTHUMB;

	if ( pParameters->bReversedValues )
		dwStyle |= TBS_REVERSED;
	if ( pParameters->bReverseEdgeMapping )
		dwStyle |= TBS_DOWNISLEFT;

	if ( pParameters->bVertical )
		dwStyle |= TBS_VERT;
	else
		dwStyle |= TBS_HORZ;
	
	if ( pParameters->bEnableToolTips )
		dwStyle |= TBS_TOOLTIPS;

    // Window creation
	m_hHandle = CreateWindowEx (
		0,
		TRACKBAR_CLASS,
		pParameters->strLabel,
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
Void WinGUISliderBar::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUISliderBar::_DispatchEvent( Int iNotificationCode, Void * pParameters )
{
    // Get Model
	WinGUISliderBarModel * pModel = (WinGUISliderBarModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		case TRBN_THUMBPOSCHANGING: {
				NMTRBTHUMBPOSCHANGING * pParams = (NMTRBTHUMBPOSCHANGING*)pParameters;
				switch( pParams->nReason ) {
						// Instant Slide
					case TB_TOP:           return pModel->OnSlideToMin( pParams->dwPos ); break;
					case TB_BOTTOM:        return pModel->OnSlideToMax( pParams->dwPos ); break;

					case TB_LINEUP:        return pModel->OnSlideSmallAmount( pParams->dwPos, -1 ); break;
					case TB_LINEDOWN:      return pModel->OnSlideSmallAmount( pParams->dwPos, +1 ); break;

					case TB_PAGEUP:        return pModel->OnSlideLargeAmount( pParams->dwPos, -1 ); break;
					case TB_PAGEDOWN:      return pModel->OnSlideLargeAmount( pParams->dwPos, +1 ); break;

						// Progressive Slide
					case TB_THUMBTRACK:    return pModel->OnSlide( pParams->dwPos ); break;

						// Slide End
					case TB_ENDTRACK:      return pModel->OnSlideEnd( pParams->dwPos, false ); break;
					case TB_THUMBPOSITION: return pModel->OnSlideEnd( pParams->dwPos, true ); break;

					default: DebugAssert(false); break;
				}
			} break;
		default: break;
	}

	// Unhandled
	return false;
}



