/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIButton.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : Button
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
#ifndef SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUIBUTTON_H
#define SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUIBUTTON_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../WinGUIControl.h"

#include "../Tools/WinGUIImageList.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Creation Parameters
typedef struct _wingui_button_parameters {
	GChar strLabel[64];
	Bool bCenterLabel;
	Bool bEnableTabStop;
	Bool bEnableNotify;
} WinGUIButtonParameters;

// Button states (match image list)
enum WinGUIButtonState {
	WINGUI_BUTTON_STATE_NORMAL    = 1,
	WINGUI_BUTTON_STATE_HOT       = 2,
	WINGUI_BUTTON_STATE_PRESSED   = 3,
	WINGUI_BUTTON_STATE_DISABLED  = 4,
	WINGUI_BUTTON_STATE_DEFAULTED = 5,
	WINGUI_BUTTON_STATE_STYLUSHOT = 6 // Tablets only
};

// Image alignment
enum WinGUIButtonImageAlignment {
	WINGUI_BUTTON_IMAGE_ALIGN_LEFT = 0,
	WINGUI_BUTTON_IMAGE_ALIGN_RIGHT,
	WINGUI_BUTTON_IMAGE_ALIGN_TOP,
	WINGUI_BUTTON_IMAGE_ALIGN_BOTTOM,
	WINGUI_BUTTON_IMAGE_ALIGN_CENTER
};

// Prototypes
class WinGUIButtonModel;
class WinGUIButton;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIButtonModel class
class WinGUIButtonModel : public WinGUIControlModel
{
public:
	WinGUIButtonModel( Int iResourceID );
	virtual ~WinGUIButtonModel();

	// Creation Parameters
	inline const WinGUIButtonParameters * GetCreationParameters() const;

	// Events
	virtual Bool OnFocusGained() { return false; }
	virtual Bool OnFocusLost() { return false; }

	virtual Bool OnMouseHovering() { return false; }
	virtual Bool OnMouseLeaving() { return false; }

	virtual Bool OnClick() { return false; }
	virtual Bool OnDblClick() { return false; }

protected:
	WinGUIButtonParameters m_hCreationParameters;
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIButton class
class WinGUIButton : public WinGUIControl
{
public:
	WinGUIButton( WinGUIElement * pParent, WinGUIButtonModel * pModel );
	virtual ~WinGUIButton();

	// Enable / Disable
	Void Enable();
	Void Disable();

	// Metrics
	Void GetIdealSize( WinGUIPoint * outSize ) const;

	Void GetTextMargin( WinGUIRectangle * outRectMargin ) const;
	Void SetTextMargin( const WinGUIRectangle & hRectMargin );

	// Label Text
	UInt GetTextLength() const;
	Void GetText( GChar * outText, UInt iMaxLength ) const;
	Void SetText( const GChar * strText );

	// Image List
	Void GetImageList( WinGUIImageList * outImageList, WinGUIRectangle * outImageMargin, WinGUIButtonImageAlignment * outAlignment ) const;
	Void SetImageList( const WinGUIImageList * pImageList, const WinGUIRectangle & hImageMargin, WinGUIButtonImageAlignment iAlignment );

private:
	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode, Void * pParameters );
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIButton.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUIBUTTON_H

