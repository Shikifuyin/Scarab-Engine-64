/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUILayout.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Layouts
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// WinGUIManualLayout implementation
inline WinGUILayoutType WinGUIManualLayout::GetType() const {
	return WINGUI_LAYOUT_MANUAL;
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIGridLayout implementation
inline WinGUILayoutType WinGUIGridLayout::GetType() const {
	return WINGUI_LAYOUT_GRID;
}

