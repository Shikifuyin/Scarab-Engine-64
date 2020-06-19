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
	WinGUIRadioButtonModel();
	virtual ~WinGUIRadioButtonModel();


protected:

};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIRadioButton class
class WinGUIRadioButton : public WinGUIControl
{
public:
	WinGUIRadioButton( WinGUIRadioButtonModel * pModel );
	virtual ~WinGUIRadioButton();

	// Initialization
	virtual Void Initialize();

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

protected:
    // Event-Handling interface
	virtual UIntPtr __stdcall _MessageCallback_Virtual( Void * hWnd, UInt message, UIntPtr wParam, UIntPtr lParam );

	// Button Handles
	Void * m_hButtonWnd; // HWND
	Int m_iButtonID;

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

