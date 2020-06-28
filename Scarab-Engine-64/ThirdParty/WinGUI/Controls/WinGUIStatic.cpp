/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIStatic.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : Static Text/Graphics
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
#include "WinGUIStatic.h"

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUIStaticModel implementation
WinGUIStaticModel::WinGUIStaticModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// Default Parameters
	m_hCreationParameters.iType = WINGUI_STATIC_TEXT;
	m_hCreationParameters.bAddSunkenBorder = false;

	StringFn->Copy( m_hCreationParameters.hText.strLabel, TEXT("StaticText") );
	m_hCreationParameters.hText.iAlign = WINGUI_STATIC_TEXT_ALIGN_LEFT;
	m_hCreationParameters.hText.iEllipsis = WINGUI_STATIC_TEXT_ELLIPSIS_NONE;

	m_hCreationParameters.bEnableNotify = false;
}
WinGUIStaticModel::~WinGUIStaticModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIStatic implementation
WinGUIStatic::WinGUIStatic( WinGUIElement * pParent, WinGUIStaticModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	// nothing to do
}
WinGUIStatic::~WinGUIStatic()
{
	// nothing to do
}

Void WinGUIStatic::Enable()
{
	HWND hHandle = (HWND)m_hHandle;
	Static_Enable( hHandle, TRUE );
}
Void WinGUIStatic::Disable()
{
	HWND hHandle = (HWND)m_hHandle;
	Static_Enable( hHandle, FALSE );
}

UInt WinGUIStatic::GetTextLength() const
{
	HWND hHandle = (HWND)m_hHandle;
	return Static_GetTextLength( hHandle );
}
Void WinGUIStatic::GetText( GChar * outText, UInt iMaxLength ) const
{
	HWND hHandle = (HWND)m_hHandle;
	Static_GetText( hHandle, outText, iMaxLength );
}
Void WinGUIStatic::SetText( const GChar * strText )
{
	HWND hHandle = (HWND)m_hHandle;
	Static_SetText( hHandle, strText );
}

Void WinGUIStatic::GetIcon( WinGUIIcon * outIcon ) const
{
	DebugAssert( !(outIcon->IsCreated()) );

	HWND hHandle = (HWND)m_hHandle;

	HICON hIcon = Static_GetIcon( hHandle, NULL );

	outIcon->_CreateFromHandle( hIcon, true );
}
Void WinGUIStatic::SetIcon( WinGUIIcon * pIcon )
{
	DebugAssert( pIcon->IsCreated() );

	HWND hHandle = (HWND)m_hHandle;

	Static_SetIcon( hHandle, (HICON)(pIcon->m_hHandle) );
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIStatic::_Create()
{
	DebugAssert( m_hHandle == NULL );

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUIStaticModel * pModel = (WinGUIStaticModel*)m_pModel;

	// Compute Layout
    const WinGUILayout * pLayout = pModel->GetLayout();

    WinGUIRectangle hParentRect;
    m_pParent->GetClientRect( &hParentRect );

    WinGUIRectangle hWindowRect;
    pLayout->ComputeLayout( &hWindowRect, hParentRect );

	// Get Creation Parameters
    const WinGUIStaticParameters * pParameters = pModel->GetCreationParameters();

	// Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE | SS_NOPREFIX );
	const GChar * strValue = NULL;
	switch( pParameters->iType ) {
		case WINGUI_STATIC_FRAME:
			switch( pParameters->hFrame.iFrameType ) {
				case WINGUI_STATIC_FRAME_ETCHED:       dwStyle |= SS_ETCHEDFRAME; break;
				case WINGUI_STATIC_FRAME_ETCHED_HORIZ: dwStyle |= SS_ETCHEDHORZ; break;
				case WINGUI_STATIC_FRAME_ETCHED_VERT:  dwStyle |= SS_ETCHEDVERT; break;
				default: DebugAssert(false); break;
			}
			break;
		case WINGUI_STATIC_RECT:
			switch( pParameters->hRect.iRectType ) {
				case WINGUI_STATIC_RECT_HOLLOW_BLACK: dwStyle |= SS_BLACKFRAME; break;
				case WINGUI_STATIC_RECT_HOLLOW_GRAY:  dwStyle |= SS_GRAYFRAME; break;
				case WINGUI_STATIC_RECT_HOLLOW_WHITE: dwStyle |= SS_WHITEFRAME; break;
				case WINGUI_STATIC_RECT_FILLED_BLACK: dwStyle |= SS_BLACKRECT; break;
				case WINGUI_STATIC_RECT_FILLED_GRAY:  dwStyle |= SS_GRAYRECT; break;
				case WINGUI_STATIC_RECT_FILLED_WHITE: dwStyle |= SS_WHITERECT; break;
				default: DebugAssert(false); break;
			}
			break;
		case WINGUI_STATIC_TEXT:
			strValue = pParameters->hText.strLabel;
			switch( pParameters->hText.iAlign ) {
				case WINGUI_STATIC_TEXT_ALIGN_LEFT:   dwStyle |= SS_LEFT; break;
				case WINGUI_STATIC_TEXT_ALIGN_RIGHT:  dwStyle |= SS_RIGHT; break;
				case WINGUI_STATIC_TEXT_ALIGN_CENTER: dwStyle |= SS_CENTER; break;
				default: DebugAssert(false); break;
			}
			switch( pParameters->hText.iEllipsis ) {
				case WINGUI_STATIC_TEXT_ELLIPSIS_NONE: dwStyle |= 0; break;
				case WINGUI_STATIC_TEXT_ELLIPSIS_END:  dwStyle |= SS_ENDELLIPSIS; break;
				case WINGUI_STATIC_TEXT_ELLIPSIS_WORD: dwStyle |= SS_WORDELLIPSIS; break;
				case WINGUI_STATIC_TEXT_ELLIPSIS_PATH: dwStyle |= SS_PATHELLIPSIS; break;
				default: DebugAssert(false); break;
			}
			break;
		case WINGUI_STATIC_BITMAP:
			dwStyle |= SS_BITMAP;
			strValue = pParameters->hBitmap.strResourceName;
			switch( pParameters->hBitmap.iInfos ) {
				case WINGUI_STATIC_IMAGE_DEFAULT:      dwStyle |= 0; break;
				case WINGUI_STATIC_IMAGE_CENTERED:     dwStyle |= SS_CENTERIMAGE; break;
				case WINGUI_STATIC_IMAGE_FIT:		   dwStyle |= SS_REALSIZECONTROL; break;
				case WINGUI_STATIC_IMAGE_FIT_CENTERED: dwStyle |= SS_REALSIZECONTROL | SS_CENTERIMAGE; break;
				default: DebugAssert(false); break;
			}
			break;
		case WINGUI_STATIC_ICON:
			dwStyle |= SS_ICON;
			strValue = pParameters->hIcon.strResourceName;
			switch( pParameters->hIcon.iInfos ) {
				case WINGUI_STATIC_IMAGE_DEFAULT:      dwStyle |= 0; break;
				case WINGUI_STATIC_IMAGE_CENTERED:     dwStyle |= SS_CENTERIMAGE; break;
				case WINGUI_STATIC_IMAGE_FIT:		   dwStyle |= SS_REALSIZEIMAGE; break;
				case WINGUI_STATIC_IMAGE_FIT_CENTERED: dwStyle |= SS_REALSIZEIMAGE | SS_CENTERIMAGE; break;
				default: DebugAssert(false); break;
			}
			break;
		default: DebugAssert(false); break;
	}
	if ( pParameters->bAddSunkenBorder )
		dwStyle |= SS_SUNKEN;
	if ( pParameters->bEnableNotify )
		dwStyle |= SS_NOTIFY;

    // Window creation
	m_hHandle = CreateWindowEx (
		0,
		WC_STATIC,
		strValue,
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
Void WinGUIStatic::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUIStatic::_DispatchEvent( Int iNotificationCode, Void * pParameters )
{
	// Get Model
	WinGUIStaticModel * pModel = (WinGUIStaticModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		case STN_CLICKED: return pModel->OnClick(); break;
		case STN_DBLCLK:  return pModel->OnDblClick(); break;
		default: break;
	}

	// Unhandled
	return false;
}

