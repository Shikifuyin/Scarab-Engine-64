/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIRadioButton.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : Radio Button
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
#ifndef SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUIRADIOBUTTON_H
#define SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUIRADIOBUTTON_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../WinGUIControl.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define WINGUI_RADIO_BUTTON_MAX_GROUP_SIZE 8 // If you need more, use a combo box !

// Prototypes
class WinGUIRadioButtonModel;
class WinGUIRadioButton;
class WinGUIRadioButtonGroup;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIRadioButtonModel class
class WinGUIRadioButtonModel : public WinGUIControlModel
{
public:
	WinGUIRadioButtonModel( Int iResourceID );
	virtual ~WinGUIRadioButtonModel();

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
// The WinGUIRadioButton class
class WinGUIRadioButton : public WinGUIControl
{
public:
	WinGUIRadioButton( WinGUIElement * pParent, WinGUIRadioButtonModel * pModel );
	virtual ~WinGUIRadioButton();

	// Radio Button Group Access
	inline WinGUIRadioButtonGroup * GetGroup() const;
	inline Void SetGroup( WinGUIRadioButtonGroup * pGroup );

	// Enable / Disable
	Void Enable();
	Void Disable();

	// Button Text
	UInt GetTextLength() const;
	Void GetText( GChar * outText, UInt iMaxLength ) const;
	Void SetText( const GChar * strText );

	// State
	Bool IsChecked() const;
	Void Check(); // Auto-Uncheck other Buttons in the Group

private:
    // Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode );

	// Radio Button Group
	friend class WinGUIRadioButtonGroup;
	WinGUIRadioButtonGroup * m_pRadioButtonGroup;
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIRadioButtonGroup class
class WinGUIRadioButtonGroup
{
public:
	WinGUIRadioButtonGroup();
	~WinGUIRadioButtonGroup();

	inline UInt GetButtonCount() const;
	inline WinGUIRadioButton * GetButton( UInt iIndex ) const;

	Void AddButton( WinGUIRadioButton * pButton );

private:
	UInt m_iButtonCount;
	WinGUIRadioButton * m_arrRadioButtons[WINGUI_RADIO_BUTTON_MAX_GROUP_SIZE];
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIRadioButton.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUIRADIOBUTTON_H

