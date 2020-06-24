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

	m_bHasMasks = false;
}
WinGUIImageList::~WinGUIImageList()
{
	// nothing to do
}

Void WinGUIImageList::Create( UInt iWidth, UInt iHeight, UInt iInitialImageCount, UInt iGrowCount )
{
	DebugAssert( m_hHandle == NULL );

	m_hHandle = ImageList_Create( iWidth, iHeight, ILC_COLORDDB, iInitialImageCount, iGrowCount );
	DebugAssert( m_hHandle != NULL );

	m_bHasMasks = false;
}
Void WinGUIImageList::CreateMasked( UInt iWidth, UInt iHeight, UInt iInitialImageCount, UInt iGrowCount )
{
	DebugAssert( m_hHandle == NULL );


	m_hHandle = ImageList_Create( iWidth, iHeight, ILC_COLORDDB | ILC_MASK, iInitialImageCount, iGrowCount );
	DebugAssert( m_hHandle != NULL );

	m_bHasMasks = true;
}
Void WinGUIImageList::Destroy()
{
	DebugAssert( m_hHandle != NULL );

	ImageList_Destroy( (HIMAGELIST)m_hHandle );

	m_hHandle = NULL;
	m_bHasMasks = false;
}

Void WinGUIImageList::AddImage( WinGUIImage * pImage, WinGUIImage * pMask )
{
	DebugAssert( m_hHandle != NULL );

	//////////////////////////////
}
