/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Tools/WinGUIImageList.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Image Lists
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

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "WinGUIImageList.h"

/////////////////////////////////////////////////////////////////////////////////
// WinGUIImageList implementation
WinGUIImageList::WinGUIImageList()
{
	m_hHandle = NULL;

	m_bIsMasked = false;
}
WinGUIImageList::~WinGUIImageList()
{
	// nothing to do
}

Void WinGUIImageList::Create( UInt iWidth, UInt iHeight, WinGUIImageListColorDepth iColorDepth, UInt iInitialImageCount, UInt iGrowCount )
{
	DebugAssert( m_hHandle == NULL );

	UInt iFlags = 0;
	switch( iColorDepth ) {
		case WINGUI_IMAGELIST_COLOR_4:  iFlags |= ILC_COLOR4; break;
		case WINGUI_IMAGELIST_COLOR_8:  iFlags |= ILC_COLOR8; break;
		case WINGUI_IMAGELIST_COLOR_16: iFlags |= ILC_COLOR16; break;
		case WINGUI_IMAGELIST_COLOR_24: iFlags |= ILC_COLOR24; break;
		case WINGUI_IMAGELIST_COLOR_32: iFlags |= ILC_COLOR32; break;
		default: DebugAssert(false); break;
	}

	m_hHandle = ImageList_Create( iWidth, iHeight, iFlags, iInitialImageCount, iGrowCount );
	DebugAssert( m_hHandle != NULL );

	m_bIsMasked = false;
}
Void WinGUIImageList::CreateMasked( UInt iWidth, UInt iHeight, WinGUIImageListColorDepth iColorDepth, UInt iInitialImageCount, UInt iGrowCount )
{
	DebugAssert( m_hHandle == NULL );

	UInt iFlags = ILC_MASK;
	switch( iColorDepth ) {
		case WINGUI_IMAGELIST_COLOR_4:  iFlags |= ILC_COLOR4; break;
		case WINGUI_IMAGELIST_COLOR_8:  iFlags |= ILC_COLOR8; break;
		case WINGUI_IMAGELIST_COLOR_16: iFlags |= ILC_COLOR16; break;
		case WINGUI_IMAGELIST_COLOR_24: iFlags |= ILC_COLOR24; break;
		case WINGUI_IMAGELIST_COLOR_32: iFlags |= ILC_COLOR32; break;
		default: DebugAssert(false); break;
	}

	m_hHandle = ImageList_Create( iWidth, iHeight, iFlags, iInitialImageCount, iGrowCount );
	DebugAssert( m_hHandle != NULL );

	m_bIsMasked = true;
}
Void WinGUIImageList::Destroy()
{
	DebugAssert( m_hHandle != NULL );

	ImageList_Destroy( (HIMAGELIST)m_hHandle );

	m_hHandle = NULL;
	m_bIsMasked = false;
}

Void WinGUIImageList::AddImage()
{
	DebugAssert( m_hHandle != NULL );


}
