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

	// Events
	virtual Void OnMousePress( const WinGUIPoint & hPoint, KeyCode iKey )   {}
	virtual Void OnMouseRelease( const WinGUIPoint & hPoint, KeyCode iKey ) {}

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

	// Sub-Classing
	static UIntPtr __stdcall _SubClassCallback_Static( Void * hHandle, UInt iMessage, UIntPtr wParam, UIntPtr lParam, UIntPtr iSubClassID, UIntPtr iRefData );
    UIntPtr __stdcall _SubClassCallback_Virtual( Void * hHandle, UInt iMessage, UIntPtr wParam, UIntPtr lParam );

	virtual Void _RegisterSubClass();
	virtual Void _UnregisterSubClass();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode, Void * pParameters ) = 0;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIControl.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_WINGUICONTROL_H

