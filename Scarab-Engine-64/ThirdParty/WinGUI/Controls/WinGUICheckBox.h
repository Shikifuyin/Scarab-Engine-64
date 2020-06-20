/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUICheckBox.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : CheckBox
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
#ifndef SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUICHECKBOX_H
#define SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUICHECKBOX_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../WinGUIControl.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// The WinGUICheckBoxModel class
class WinGUICheckBoxModel : public WinGUIControlModel
{
public:
	WinGUICheckBoxModel( Int iResourceID );
	virtual ~WinGUICheckBoxModel();

	// Events
	virtual Bool OnClick() = 0;
	virtual Bool OnDblClick() = 0;

	// View
	virtual const GChar * GetText() const = 0;

	virtual UInt GetPositionX() const = 0;
	virtual UInt GetPositionY() const = 0;
	virtual UInt GetWidth() const = 0;
	virtual UInt GetHeight() const = 0;

protected:

};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUICheckBox class
class WinGUICheckBox : public WinGUIControl
{
public:
	WinGUICheckBox( WinGUIElement * pParent, WinGUICheckBoxModel * pModel );
	virtual ~WinGUICheckBox();

	// Enable / Disable
	Void Enable();
	Void Disable();

	// Label Text
	UInt GetTextLength() const;
	Void GetText( GChar * outText, UInt iMaxLength ) const;
	Void SetText( const GChar * strText );

	// State
	Bool IsChecked() const;
	Void Check();
	Void Uncheck();

private:
    // Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode );
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUICheckBox.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUICHECKBOX_H

