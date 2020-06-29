/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIComboBox.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : ComboBox
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
#include "WinGUIComboBox.h"

#pragma warning(disable:4312) // Int to HMENU cast

/////////////////////////////////////////////////////////////////////////////////
// WinGUIComboBoxModel implementation
WinGUIComboBoxModel::WinGUIComboBoxModel( Int iResourceID ):
	WinGUIControlModel(iResourceID)
{
	// Default Parameters
	m_hCreationParameters.iItemCallBackMode = 0;

	m_hCreationParameters.iType = WINGUI_COMBOBOX_BUTTON;
	m_hCreationParameters.iCase = WINGUI_COMBOBOX_CASE_BOTH;
	m_hCreationParameters.iInitialSelectedItem = 0;
	m_hCreationParameters.bAllowHorizontalScroll = false;
	m_hCreationParameters.bItemTextEllipsis = true;
	m_hCreationParameters.bCaseSensitiveSearch = false;
	m_hCreationParameters.bAutoSort = false;
	m_hCreationParameters.bEnableTabStop = true;
}
WinGUIComboBoxModel::~WinGUIComboBoxModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIComboBox implementation
WinGUIComboBox::WinGUIComboBox( WinGUIElement * pParent, WinGUIComboBoxModel * pModel ):
	WinGUIControl(pParent, pModel)
{
	m_iItemCallBackMode = 0;
}
WinGUIComboBox::~WinGUIComboBox()
{
	// nothing to do
}

Bool WinGUIComboBox::IsUnicode() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ( SendMessage(hHandle, CBEM_GETUNICODEFORMAT, (WPARAM)0, (LPARAM)0) != 0 );
}
Bool WinGUIComboBox::IsANSI() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ( SendMessage(hHandle, CBEM_GETUNICODEFORMAT, (WPARAM)0, (LPARAM)0) == 0 );
}

Void WinGUIComboBox::SetUnicode()
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, CBEM_SETUNICODEFORMAT, (WPARAM)TRUE, (LPARAM)0 );
}
Void WinGUIComboBox::SetANSI()
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, CBEM_SETUNICODEFORMAT, (WPARAM)FALSE, (LPARAM)0 );
}

Void WinGUIComboBox::Enable()
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_Enable( hHandle, TRUE );
}
Void WinGUIComboBox::Disable()
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_Enable( hHandle, FALSE );
}

Void WinGUIComboBox::Expand()
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_ShowDropdown( hHandle, TRUE );
}
Void WinGUIComboBox::Collapse()
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_ShowDropdown( hHandle, FALSE );
}

Void WinGUIComboBox::GetImageList( WinGUIImageList * outImageList ) const
{
	DebugAssert( !(outImageList->IsCreated()) );

	HWND hHandle = (HWND)m_hHandle;

	HIMAGELIST hImgList = (HIMAGELIST)( SendMessage(hHandle, CBEM_GETIMAGELIST, (WPARAM)0, (LPARAM)0) );
	outImageList->_CreateFromHandle( hImgList );
}
Void WinGUIComboBox::SetImageList( const WinGUIImageList * pImageList )
{
	DebugAssert( pImageList->IsCreated() );

	HWND hHandle = (HWND)m_hHandle;

	HIMAGELIST hImgList = (HIMAGELIST)( pImageList->m_hHandle );
	SendMessage( hHandle, CBEM_SETIMAGELIST, (WPARAM)0, (LPARAM)hImgList );
}

UInt WinGUIComboBox::GetBoxItemHeight() const
{
	HWND hHandle = (HWND)m_hHandle;
	return (UInt)( SendMessage(hHandle, CB_GETITEMHEIGHT, (WPARAM)-1, (LPARAM)0) );
}
Void WinGUIComboBox::SetBoxItemHeight( UInt iHeight )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_SetItemHeight( hHandle, -1, iHeight );
}

UInt WinGUIComboBox::GetListItemHeight() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ComboBox_GetItemHeight( hHandle );
}
Void WinGUIComboBox::SetListItemHeight( UInt iHeight )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_SetItemHeight( hHandle, 0, iHeight );
}

UInt WinGUIComboBox::GetMinVisibleItems() const
{
	HWND hHandle = (HWND)m_hHandle;

	HWND hComboBoxHandle = (HWND)( SendMessage(hHandle, CBEM_GETCOMBOCONTROL, (WPARAM)0, (LPARAM)0) );
	return ComboBox_GetMinVisible( hComboBoxHandle );
}
Void WinGUIComboBox::SetMinVisibleItems( UInt iCount )
{
	HWND hHandle = (HWND)m_hHandle;

	HWND hComboBoxHandle = (HWND)( SendMessage(hHandle, CBEM_GETCOMBOCONTROL, (WPARAM)0, (LPARAM)0) );
	ComboBox_SetMinVisible( hComboBoxHandle, iCount );
}

Void WinGUIComboBox::SetTextLimit( UInt iMaxLength )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_LimitText( hHandle, iMaxLength );
}

UInt WinGUIComboBox::GetItemCount() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ComboBox_GetCount( hHandle );
}

Void WinGUIComboBox::AddItem( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;

	COMBOBOXEXITEM hItemInfos;
	hItemInfos.iItem = iIndex;

	static GChar arrTempLabel[64];

	hItemInfos.mask = CBEIF_TEXT;
	hItemInfos.pszText = arrTempLabel;
	hItemInfos.cchTextMax = 64;
	StringFn->NCopy( hItemInfos.pszText, TEXT("_Uninitialized_"), 63 );

	if ( (m_iItemCallBackMode & WINGUI_COMBOBOX_ITEMCALLBACK_LABELS) != 0 )
		hItemInfos.pszText = LPSTR_TEXTCALLBACK;

	if ( (m_iItemCallBackMode & WINGUI_COMBOBOX_ITEMCALLBACK_IMAGES) != 0 ) {
		hItemInfos.mask |= ( CBEIF_IMAGE | CBEIF_SELECTEDIMAGE );
		hItemInfos.iImage = I_IMAGECALLBACK;
		hItemInfos.iSelectedImage = I_IMAGECALLBACK;
	}

	if ( (m_iItemCallBackMode & WINGUI_COMBOBOX_ITEMCALLBACK_OVERLAY) != 0 ) {
		hItemInfos.mask |= CBEIF_OVERLAY;
		hItemInfos.iOverlay = I_IMAGECALLBACK;
	}

	if ( (m_iItemCallBackMode & WINGUI_COMBOBOX_ITEMCALLBACK_INDENTATION) != 0 ) {
		hItemInfos.mask |= CBEIF_INDENT;
		hItemInfos.iIndent = I_INDENTCALLBACK;
	}

	SendMessage( hHandle, CBEM_INSERTITEM, (WPARAM)0, (LPARAM)&hItemInfos );
}
Void WinGUIComboBox::RemoveItem( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	SendMessage( hHandle, CBEM_DELETEITEM, (WPARAM)iIndex, (LPARAM)0 );
}
Void WinGUIComboBox::RemoveAllItems()
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_ResetContent( hHandle );
}

Void WinGUIComboBox::GetItemLabel( GChar * outLabelText, UInt iMaxLength, UInt iIndex ) const
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_COMBOBOX_ITEMCALLBACK_LABELS) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	COMBOBOXEXITEM hItemInfos;
	hItemInfos.mask = CBEIF_TEXT;
	hItemInfos.iItem = iIndex;
	hItemInfos.pszText = outLabelText;
	hItemInfos.cchTextMax = iMaxLength;

	SendMessage( hHandle, CBEM_GETITEM, (WPARAM)0, (LPARAM)&hItemInfos );
}
Void WinGUIComboBox::SetItemLabel( UInt iIndex, GChar * strLabelText )
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_COMBOBOX_ITEMCALLBACK_LABELS) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	COMBOBOXEXITEM hItemInfos;
	hItemInfos.mask = CBEIF_TEXT;
	hItemInfos.iItem = iIndex;
	hItemInfos.pszText = strLabelText;

	SendMessage( hHandle, CBEM_SETITEM, (WPARAM)0, (LPARAM)&hItemInfos );
}

UInt WinGUIComboBox::GetItemImage( UInt iIndex ) const
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_COMBOBOX_ITEMCALLBACK_IMAGES) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	COMBOBOXEXITEM hItemInfos;
	hItemInfos.mask = CBEIF_IMAGE;
	hItemInfos.iItem = iIndex;
	hItemInfos.iImage = I_IMAGENONE;

	SendMessage( hHandle, CBEM_GETITEM, (WPARAM)0, (LPARAM)&hItemInfos );

	if ( hItemInfos.iImage == I_IMAGENONE )
		return INVALID_OFFSET;
	return hItemInfos.iImage;
}
Void WinGUIComboBox::SetItemImage( UInt iIndex, UInt iImageIndex )
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_COMBOBOX_ITEMCALLBACK_IMAGES) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	COMBOBOXEXITEM hItemInfos;
	hItemInfos.mask = CBEIF_IMAGE;
	hItemInfos.iItem = iIndex;
	if ( iImageIndex == INVALID_OFFSET )
		hItemInfos.iImage = I_IMAGENONE;
	else
		hItemInfos.iImage = iImageIndex;

	SendMessage( hHandle, CBEM_SETITEM, (WPARAM)0, (LPARAM)&hItemInfos );
}

UInt WinGUIComboBox::GetItemImageSelected( UInt iIndex ) const
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_COMBOBOX_ITEMCALLBACK_IMAGES) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	COMBOBOXEXITEM hItemInfos;
	hItemInfos.mask = CBEIF_SELECTEDIMAGE;
	hItemInfos.iItem = iIndex;
	hItemInfos.iSelectedImage = I_IMAGENONE;

	SendMessage( hHandle, CBEM_GETITEM, (WPARAM)0, (LPARAM)&hItemInfos );

	if ( hItemInfos.iSelectedImage == I_IMAGENONE )
		return INVALID_OFFSET;
	return hItemInfos.iSelectedImage;
}
Void WinGUIComboBox::SetItemImageSelected( UInt iIndex, UInt iImageIndex )
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_COMBOBOX_ITEMCALLBACK_IMAGES) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	COMBOBOXEXITEM hItemInfos;
	hItemInfos.mask = CBEIF_SELECTEDIMAGE;
	hItemInfos.iItem = iIndex;
	if ( iImageIndex == INVALID_OFFSET )
		hItemInfos.iSelectedImage = I_IMAGENONE;
	else
		hItemInfos.iSelectedImage = iImageIndex;

	SendMessage( hHandle, CBEM_SETITEM, (WPARAM)0, (LPARAM)&hItemInfos );
}

Void * WinGUIComboBox::GetItemData( UInt iIndex ) const
{
	HWND hHandle = (HWND)m_hHandle;

	COMBOBOXEXITEM hItemInfos;
	hItemInfos.mask = CBEIF_LPARAM;
	hItemInfos.iItem = iIndex;
	hItemInfos.lParam = NULL;

	SendMessage( hHandle, CBEM_GETITEM, (WPARAM)0, (LPARAM)&hItemInfos );

	return (Void*)( hItemInfos.lParam );
}
Void WinGUIComboBox::SetItemData( UInt iIndex, Void * pData )
{
	HWND hHandle = (HWND)m_hHandle;
	
	COMBOBOXEXITEM hItemInfos;
	hItemInfos.mask = CBEIF_LPARAM;
	hItemInfos.iItem = iIndex;
	hItemInfos.lParam = (LPARAM)pData;

	SendMessage( hHandle, CBEM_SETITEM, (WPARAM)0, (LPARAM)&hItemInfos );
}

UInt WinGUIComboBox::GetItemOverlayImage( UInt iIndex ) const
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_COMBOBOX_ITEMCALLBACK_OVERLAY) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	COMBOBOXEXITEM hItemInfos;
	hItemInfos.mask = CBEIF_OVERLAY;
	hItemInfos.iItem = iIndex;
	hItemInfos.iOverlay = I_IMAGENONE;

	SendMessage( hHandle, CBEM_GETITEM, (WPARAM)0, (LPARAM)&hItemInfos );

	if ( hItemInfos.iOverlay == I_IMAGENONE )
		return INVALID_OFFSET;
	return hItemInfos.iOverlay;
}
Void WinGUIComboBox::SetItemOverlayImage( UInt iIndex, UInt iOverlayImage )
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_COMBOBOX_ITEMCALLBACK_OVERLAY) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	COMBOBOXEXITEM hItemInfos;
	hItemInfos.mask = CBEIF_OVERLAY;
	hItemInfos.iItem = iIndex;
	if ( iOverlayImage == INVALID_OFFSET )
		hItemInfos.iOverlay = I_IMAGENONE;
	else
		hItemInfos.iOverlay = iOverlayImage;

	SendMessage( hHandle, CBEM_SETITEM, (WPARAM)0, (LPARAM)&hItemInfos );
}

UInt WinGUIComboBox::GetItemIndentation( UInt iIndex ) const
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_COMBOBOX_ITEMCALLBACK_INDENTATION) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	COMBOBOXEXITEM hItemInfos;
	hItemInfos.mask = CBEIF_INDENT;
	hItemInfos.iItem = iIndex;
	hItemInfos.iIndent = 0;

	SendMessage( hHandle, CBEM_GETITEM, (WPARAM)0, (LPARAM)&hItemInfos );

	return hItemInfos.iIndent;
}
Void WinGUIComboBox::SetItemIndentation( UInt iIndex, UInt iIndentation )
{
	DebugAssert( (m_iItemCallBackMode & WINGUI_COMBOBOX_ITEMCALLBACK_INDENTATION) == 0 );

	HWND hHandle = (HWND)m_hHandle;

	COMBOBOXEXITEM hItemInfos;
	hItemInfos.mask = CBEIF_INDENT;
	hItemInfos.iItem = iIndex;
	hItemInfos.iIndent = iIndentation;

	SendMessage( hHandle, CBEM_SETITEM, (WPARAM)0, (LPARAM)&hItemInfos );
}

UInt WinGUIComboBox::GetSelectedItem() const
{
	HWND hHandle = (HWND)m_hHandle;
	return ComboBox_GetCurSel( hHandle );
}
Void WinGUIComboBox::SelectItem( UInt iIndex )
{
	HWND hHandle = (HWND)m_hHandle;
	ComboBox_SetCurSel( hHandle, iIndex );
}

UInt WinGUIComboBox::SearchItem( const GChar * strItem, UInt iStartIndex, Bool bExact ) const
{
	HWND hHandle = (HWND)m_hHandle;

	UInt iResult;
	if ( bExact )
		iResult = ComboBox_FindStringExact( hHandle, ((Int)iStartIndex) - 1, strItem );
	else {
		//HWND hComboBoxHandle = (HWND)( SendMessage(hHandle, CBEM_GETCOMBOCONTROL, (WPARAM)0, (LPARAM)0) );
		iResult = ComboBox_FindString( hHandle, ((Int)iStartIndex) - 1, strItem );
	}

	return (iResult != CB_ERR) ? iResult : INVALID_OFFSET;
}

Void WinGUIComboBox::GetCueText( GChar * outText, UInt iMaxLength ) const
{
	HWND hHandle = (HWND)m_hHandle;

	HWND hComboBoxHandle = (HWND)( SendMessage(hHandle, CBEM_GETCOMBOCONTROL, (WPARAM)0, (LPARAM)0) );
	ComboBox_GetCueBannerText( hComboBoxHandle, outText, iMaxLength );
}
Void WinGUIComboBox::SetCueText( const GChar * strText )
{
	HWND hHandle = (HWND)m_hHandle;

	HWND hComboBoxHandle = (HWND)( SendMessage(hHandle, CBEM_GETCOMBOCONTROL, (WPARAM)0, (LPARAM)0) );
	ComboBox_SetCueBannerText( hComboBoxHandle, strText );
}

/////////////////////////////////////////////////////////////////////////////////

Void WinGUIComboBox::_Create()
{
	DebugAssert( m_hHandle == NULL );

	// Get Parent Handle
	HWND hParentWnd = (HWND)( _GetHandle(m_pParent) );

    // Get Model
    WinGUIComboBoxModel * pModel = (WinGUIComboBoxModel*)m_pModel;

	// Compute Layout
    const WinGUILayout * pLayout = pModel->GetLayout();

    WinGUIRectangle hParentRect;
    m_pParent->GetClientRect( &hParentRect );

    WinGUIRectangle hWindowRect;
    pLayout->ComputeLayout( &hWindowRect, hParentRect );

	// Get Creation Parameters
    const WinGUIComboBoxParameters * pParameters = pModel->GetCreationParameters();

	// Save State
	m_iItemCallBackMode = pParameters->iItemCallBackMode;

    // Build Style
	DWord dwStyle = ( WS_CHILD | WS_VISIBLE );
	DWord dwStyleEx = 0;

	switch( pParameters->iType ) {
		case WINGUI_COMBOBOX_BUTTON:
			dwStyle |= CBS_DROPDOWNLIST;
			break;
		case WINGUI_COMBOBOX_EDIT:
			dwStyle |= CBS_DROPDOWN;
			if ( pParameters->bAllowHorizontalScroll )
				dwStyle |= CBS_AUTOHSCROLL;
			break;
		case WINGUI_COMBOBOX_LIST:
			dwStyle |= CBS_SIMPLE;
			if ( pParameters->bAllowHorizontalScroll )
				dwStyle |= CBS_AUTOHSCROLL;
			break;
		default: DebugAssert(false); break;
	}
	switch( pParameters->iCase ) {
		case WINGUI_COMBOBOX_CASE_BOTH:  dwStyle |= 0; break;
		case WINGUI_COMBOBOX_CASE_LOWER: dwStyle |= CBS_LOWERCASE; break;
		case WINGUI_COMBOBOX_CASE_UPPER: dwStyle |= CBS_UPPERCASE; break;
		default: DebugAssert(false); break;
	}
	if ( pParameters->bItemTextEllipsis )
		dwStyleEx |= CBES_EX_TEXTENDELLIPSIS;
	if ( pParameters->bCaseSensitiveSearch )
		dwStyleEx |= CBES_EX_CASESENSITIVE;
	if ( pParameters->bAutoSort )
		dwStyle |= CBS_SORT;
	if ( pParameters->bEnableTabStop )
		dwStyle |= WS_TABSTOP;

    // Window creation
	m_hHandle = CreateWindowEx (
		dwStyleEx,
		WC_COMBOBOXEX,
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

	// Start with an empty list

	// Done
	_SaveElementToHandle();
}
Void WinGUIComboBox::_Destroy()
{
	DebugAssert( m_hHandle != NULL );

    // Window destruction
	DestroyWindow( (HWND)m_hHandle );
	m_hHandle = NULL;
}

Bool WinGUIComboBox::_DispatchEvent( Int iNotificationCode, Void * pParameters )
{
    // Get Model
	WinGUIComboBoxModel * pModel = (WinGUIComboBoxModel*)m_pModel;

	// Dispatch Event to the Model
	switch( iNotificationCode ) {
		// ComboBox Notifications
		case CBN_SETFOCUS:  return pModel->OnFocusGained(); break;
		case CBN_KILLFOCUS: return pModel->OnFocusLost(); break;

		case CBN_DBLCLK: return pModel->OnDblClick(); break;

		case CBN_DROPDOWN: return pModel->OnExpand(); break;
		case CBN_CLOSEUP:  return pModel->OnCollapse(); break;

		case CBN_SELCHANGE:    return pModel->OnSelectionChange(); break;
		case CBN_SELENDOK:     return pModel->OnSelectionOK(); break;
		case CBN_SELENDCANCEL: return pModel->OnSelectionCancel(); break;

		// ComboBoxEx Notifications
		case CBEN_INSERTITEM: {
				NMCOMBOBOXEX * pParams = (NMCOMBOBOXEX*)pParameters;
				return pModel->OnItemAdded( pParams->ceItem.iItem, pParams->ceItem.pszText, (Void*)(pParams->ceItem.lParam) );
			} break;
		case CBEN_DELETEITEM: {
				NMCOMBOBOXEX * pParams = (NMCOMBOBOXEX*)pParameters;
				return pModel->OnItemRemoved( pParams->ceItem.iItem, pParams->ceItem.pszText, (Void*)(pParams->ceItem.lParam) );
			} break;

		case CBEN_BEGINEDIT: return pModel->OnEditStart(); break;
		case CBEN_ENDEDIT: {
				NMCBEENDEDIT * pParams = (NMCBEENDEDIT*)pParameters;
				Bool bTextEditChanged = ( pParams->fChanged != FALSE );
				UInt iSelectedItem = pParams->iNewSelection;
				if ( (pParams->iWhy & CBENF_ESCAPE) != 0 )
					return pModel->OnEditCancel( pParams->szText, bTextEditChanged, iSelectedItem ); // Return false to allow modification
				else
					return pModel->OnEditEnd( pParams->szText, bTextEditChanged, iSelectedItem ); // Return false to allow modification
			} break;

		case CBEN_GETDISPINFO: {
				NMCOMBOBOXEX * pParams = (NMCOMBOBOXEX*)pParameters;

				UInt iItemIndex = pParams->ceItem.iItem;
				UInt iMask = pParams->ceItem.mask;

				Void * pItemData = NULL;
				if ( iMask & CBEIF_LPARAM )
					pItemData = (Void*)( pParams->ceItem.lParam );

				// Request Item Label Text
				if ( (iMask & CBEIF_TEXT) != 0 ) {
					pParams->ceItem.pszText = pModel->OnRequestItemLabel( iItemIndex, pItemData );
				}

				// Request Item Image Index
				if ( (iMask & CBEIF_IMAGE) != 0 ) {
					UInt iImage = pModel->OnRequestItemImage( iItemIndex, pItemData );
					pParams->ceItem.iImage = (iImage != INVALID_OFFSET) ? iImage : I_IMAGENONE;
				}

				// Request Item Selected Image Index
				if ( (iMask & CBEIF_SELECTEDIMAGE) != 0 ) {
					UInt iImageSelected = pModel->OnRequestItemImageSelected( iItemIndex, pItemData );
					pParams->ceItem.iSelectedImage = (iImageSelected != INVALID_OFFSET) ? iImageSelected : I_IMAGENONE;
				}

				// Request Item Overlay Image Index
				if ( (iMask & CBEIF_OVERLAY) != 0 ) {
					UInt iOverlay = pModel->OnRequestItemOverlayImage( iItemIndex, pItemData );
					pParams->ceItem.iOverlay = (iOverlay != INVALID_OFFSET) ? iOverlay : I_IMAGENONE;
				}

				// Request Item Indentation
				if ( (iMask & CBEIF_INDENT) != 0 ) {
					pParams->ceItem.iIndent = pModel->OnRequestItemIndentation( iItemIndex, pItemData );
				}

				return false;
			} break;

		default: break;
	}

	// Unhandled
	return false;
}


