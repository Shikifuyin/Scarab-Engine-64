/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIButton.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : Button
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
#include "WinGUIButton.h"

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUIButtonModel implementation
WinGUIButtonModel::WinGUIButtonModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// Default Parameters
	StringFn->Copy( m_hCreationParameters.strLabel, TEXT("Button") );

	m_hCreationParameters.bCenterLabel = true;
	m_hCreationParameters.bEnableTabStop = true;
	m_hCreationParameters.bEnableNotify = false;
}
WinGUIButtonModel::~WinGUIButtonModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIButton implementation
WinGUIButton::WinGUIButton( WinGUIElement * pParent, WinGUIButtonModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	// nothing to do
}
WinGUIButton::~WinGUIButton()
{
	// nothing to do
}

Void WinGUIButton::Enable()
{
	HWND hHandle = (HWND)m_hHandle;
	Button_Enable( hHandle, TRUE );
}
Void WinGUIButton::Disable()
{
	HWND hHandle = (HWND)m_hHandle;
	Button_Enable( hHandle, FALSE );
}

Void WinGUIButton::GetIdealSize( WinGUIPoint * outSize ) const
{
	HWND hHandle = (HWND)m_hHandle;

	SIZE hSize;
	Button_GetIdealSize( hHandle, &hSize );

	outSize->iX = hSize.cx;
	outSize->iY = hSize.cy;
}

Void WinGUIButton::GetTextMargin( WinGUIRectangle * outRectMargin ) const
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	Button_GetTextMargin( hHandle, &hRect );

	outRectMargin->iLeft = hRect.left;
	outRectMargin->iTop = hRect.top;
	outRectMargin->iWidth = ( hRect.right - hRect.left );
	outRectMargin->iHeight = ( hRect.bottom - hRect.top );
}
Void WinGUIButton::SetTextMargin( const WinGUIRectangle & hRectMargin )
{
	HWND hHandle = (HWND)m_hHandle;

	RECT hRect;
	hRect.left = hRectMargin.iLeft;
	hRect.top = hRectMargin.iTop;
	hRect.right = ( hRectMargin.iLeft + hRectMargin.iWidth );
	hRect.bottom = ( hRectMargin.iTop + hRectMargin.iHeight );

	Button_SetTextMargin( hHandle, &hRect );
}

UInt WinGUIButton::GetTextLength() const
{
	HWND hHandle = (HWND)m_hHandle;
	return Button_GetTextLength( hHandle );
}
Void WinGUIButton::GetText( GChar * outText, UInt iMaxLength ) const
{
	HWND hHandle = (HWND)m_hHandle;
	Button_GetText( hHandle, outText, iMaxLength );
}
Void WinGUIButton::SetText( const GChar * strText )
{
	HWND hHandle = (HWND)m_hHandle;
	Button_SetText( hHandle, strText );
}

Void WinGUIButton::GetImageList( WinGUIImageList * outImageList, WinGUIRectangle * outImageMargin, WinGUIButtonImageAlignment * outAlignment ) const
{
	DebugAssert( !(outImageList->IsCreated()) );

	HWND hHandle = (HWND)m_hHandle;

	BUTTON_IMAGELIST hImgList;
	Button_GetImageList( hHandle, &hImgList );

	outImageList->_CreateFromHandle( hImgList.himl );

	outImageMargin->iLeft = hImgList.margin.left;
	outImageMargin->iTop = hImgList.margin.top;
	outImageMargin->iWidth = ( hImgList.margin.right - hImgList.margin.left );
	outImageMargin->iHeight = ( hImgList.margin.bottom - hImgList.margin.top );

	switch( hImgList.uAlign ) {
		case BUTTON_IMAGELIST_ALIGN_LEFT:   *outAlignment = WINGUI_BUTTON_IMAGE_ALIGN_LEFT; break;
		case BUTTON_IMAGELIST_ALIGN_RIGHT:  *outAlignment = WINGUI_BUTTON_IMAGE_ALIGN_RIGHT; break;
		case BUTTON_IMAGELIST_ALIGN_TOP:    *outAlignment = WINGUI_BUTTON_IMAGE_ALIGN_TOP; break;
		case BUTTON_IMAGELIST_ALIGN_BOTTOM: *outAlignment = WINGUI_BUTTON_IMAGE_ALIGN_BOTTOM; break;
		case BUTTON_IMAGELIST_ALIGN_CENTER: *outAlignment = WINGUI_BUTTON_IMAGE_ALIGN_CENTER; break;
		default: DebugAssert(false); break;
	}
}
Void WinGUIButton::SetImageList( const WinGUIImageList * pImageList, const WinGUIRectangle & hImageMargin, WinGUIButtonImageAlignment iAlignment )
{
	DebugAssert( pImageList->IsCreated() );

	HWND hHandle = (HWND)m_hHandle;

	BUTTON_IMAGELIST hImgList;
	hImgList.himl = (HIMAGELIST)( pImageList->m_hHandle );

	hImgList.margin.left = hImageMargin.iLeft;
	hImgList.margin.top = hImageMargin.iTop;
	hImgList.margin.right = ( hImageMargin.iLeft + hImageMargin.iWidth );
	hImgList.margin.bottom = ( hImageMargin.iTop + hImageMargin.iHeight );

	switch( iAlignment ) {
		case WINGUI_BUTTON_IMAGE_ALIGN_LEFT:   hImgList.uAlign = BUTTON_IMAGELIST_ALIGN_LEFT; break;
		case WINGUI_BUTTON_IMAGE_ALIGN_RIGHT:  hImgList.uAlign = BUTTON_IMAGELIST_ALIGN_RIGHT; break;
		case WINGUI_BUTTON_IMAGE_ALIGN_TOP:    hImgList.uAlign = BUTTON_IMAGELIST_ALIGN_TOP; break;
		case WINGUI_BUTTON_IMAGE_ALIGN_BOTTOM: hImgList.uAlign = BUTTON_IMAGELIST_ALIGN_BOTTOM; break;
		case WINGUI_BUTTON_IMAGE_ALIGN_CENTER: hImgList.uAlign = BUTTON_IMAGELIST_ALIGN_CENTER; break;
		default: DebugAssert(false); break;
	}

	Button_SetImageList( hHandle, &hImgList );
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIButton::_Create()
{
	DebugAssert( m_hHandle == NULL );

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUIButtonModel * pModel = (WinGUIButtonModel*)m_pModel;

	// Compute Layout
    const WinGUILayout * pLayout = pModel->GetLayout();

    WinGUIRectangle hParentRect;
    m_pParent->GetClientRect( &hParentRect );

    WinGUIRectangle hWindowRect;
    pLayout->ComputeLayout( &hWindowRect, hParentRect );

	// Get Creation Parameters
    const WinGUIButtonParameters * pParameters = pModel->GetCreationParameters();

    // Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON );
	if ( pParameters->bCenterLabel )
		dwStyle |= ( BS_CENTER | BS_VCENTER );
	if ( pParameters->bEnableTabStop )
		dwStyle |= WS_TABSTOP;
	if ( pParameters->bEnableNotify )
		dwStyle |= BS_NOTIFY;

    // Window creation
	m_hHandle = CreateWindowEx (
		0,
		WC_BUTTON,
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
	_RegisterSubClass();
}
Void WinGUIButton::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

	// Remove SubClass
	_UnregisterSubClass();

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUIButton::_DispatchEvent( Int iNotificationCode, Void * pParameters )
{
    // Get Model
	WinGUIButtonModel * pModel = (WinGUIButtonModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		case BN_SETFOCUS:  return pModel->OnFocusGained(); break;
		case BN_KILLFOCUS: return pModel->OnFocusLost(); break;

		case BCN_HOTITEMCHANGE: {
			NMBCHOTITEM * pParams = (NMBCHOTITEM *)pParameters;
			if ( pParams->dwFlags & HICF_ENTERING )
				return pModel->OnMouseHovering();
			else if ( pParams->dwFlags & HICF_LEAVING )
				return pModel->OnMouseLeaving();
		} break;

		case BN_CLICKED: return pModel->OnClick(); break;
		case BN_DBLCLK:  return pModel->OnDblClick(); break;
		default: break;
	}

	// Unhandled
	return false;
}

