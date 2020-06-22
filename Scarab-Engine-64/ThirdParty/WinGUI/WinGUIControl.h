/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUIControl.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Element : Controls
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_WINGUI_WINGUICONTROL_H
#define SCARAB_THIRDPARTY_WINGUI_WINGUICONTROL_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "WinGUIContainer.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Prototypes
class WinGUIControlModel;
class WinGUIControl;

class WinGUIWindow;
class WinGUIContainer;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIControlModel class
class WinGUIControlModel : public WinGUIElementModel
{
public:
	WinGUIControlModel( Int iResourceID );
	virtual ~WinGUIControlModel();

protected:

};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIControl class
class WinGUIControl : public WinGUIElement
{
public:
	WinGUIControl( WinGUIElement * pParent, WinGUIControlModel * pModel );
	virtual ~WinGUIControl();

	// Type
	inline virtual WinGUIElementType GetElementType() const;

protected:
	friend class WinGUIWindow;
	friend class WinGUIContainer;

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode ) = 0;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIControl.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_WINGUICONTROL_H

