/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUIControl.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Controls Base Interface
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
#include "WinGUIControl.h"

/////////////////////////////////////////////////////////////////////////////////
// WinGUIControlModel implementation
WinGUIControlModel::WinGUIControlModel( Int iResourceID ):
	WinGUIElementModel(iResourceID)
{
}
WinGUIControlModel::~WinGUIControlModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIControl implementation
WinGUIControl::WinGUIControl( WinGUIControlModel * pModel ):
	WinGUIElement(pModel)
{
	m_pParent = NULL;
}
WinGUIControl::~WinGUIControl()
{
	// nothing to do
}

