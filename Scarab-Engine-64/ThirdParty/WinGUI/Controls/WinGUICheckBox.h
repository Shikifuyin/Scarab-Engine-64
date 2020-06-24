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

// Creation Parameters
typedef struct _wingui_checkbox_parameters {
	GChar strLabel[64];
	Bool bEnableTabStop;
	Bool bEnableNotify;
} WinGUICheckBoxParameters;

// Prototypes
class WinGUICheckBoxModel;
class WinGUICheckBox;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUICheckBoxModel class
class WinGUICheckBoxModel : public WinGUIControlModel
{
public:
	WinGUICheckBoxModel( Int iResourceID );
	virtual ~WinGUICheckBoxModel();

	// Creation Parameters
	inline const WinGUICheckBoxParameters * GetCreationParameters() const;

	// Events
	virtual Bool OnFocusGained() { return false; }
	virtual Bool OnFocusLost() { return false; }

	virtual Bool OnMouseHovering() { return false; }
	virtual Bool OnMouseLeaving() { return false; }

	virtual Bool OnClick() { return false; }
	virtual Bool OnDblClick() { return false; }

protected:
	WinGUICheckBoxParameters m_hCreationParameters;
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
	virtual Bool _DispatchEvent( Int iNotificationCode, Void * pParameters );
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUICheckBox.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUICHECKBOX_H

