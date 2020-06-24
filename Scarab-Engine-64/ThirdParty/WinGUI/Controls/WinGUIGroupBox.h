/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIGroupBox.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : GroupBox
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
#ifndef SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUIGROUPBOX_H
#define SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUIGROUPBOX_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../WinGUIControl.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Creation Parameters
typedef struct _wingui_groupbox_parameters {
	GChar strLabel[64];
} WinGUIGroupBoxParameters;

// Prototypes
class WinGUIGroupBoxModel;
class WinGUIGroupBox;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIGroupBoxModel class
class WinGUIGroupBoxModel : public WinGUIControlModel
{
public:
	WinGUIGroupBoxModel( Int iResourceID );
	virtual ~WinGUIGroupBoxModel();

	// Creation Parameters
	inline const WinGUIGroupBoxParameters * GetCreationParameters() const;

	// Events
	virtual Bool OnMouseHovering() { return false; }
	virtual Bool OnMouseLeaving() { return false; }

protected:
	WinGUIGroupBoxParameters m_hCreationParameters;
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIGroupBox class
class WinGUIGroupBox : public WinGUIControl
{
public:
	WinGUIGroupBox( WinGUIElement * pParent, WinGUIGroupBoxModel * pModel );
	virtual ~WinGUIGroupBox();

	// Label Text
	UInt GetTextLength() const;
	Void GetText( GChar * outText, UInt iMaxLength ) const;
	Void SetText( const GChar * strText );

	// Client Area
	Void ComputeClientArea( WinGUIRectangle * outClientArea, Int iPadding ) const;

private:
	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode, Void * pParameters );
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIGroupBox.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUIGROUPBOX_H

