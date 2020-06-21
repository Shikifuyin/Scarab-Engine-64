/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUITextEdit.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : Text Edit (Single Line)
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
#ifndef SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITEXTEDIT_H
#define SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITEXTEDIT_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../WinGUIControl.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// TextEdit Properties
enum WinGUITextEditAlign {
	WINGUI_TEXTEDIT_ALIGN_LEFT = 0,
	WINGUI_TEXTEDIT_ALIGN_RIGHT,
	WINGUI_TEXTEDIT_ALIGN_CENTER
};
enum WinGUITextEditCase {
	WINGUI_TEXTEDIT_CASE_BOTH = 0,
	WINGUI_TEXTEDIT_CASE_LOWER,
	WINGUI_TEXTEDIT_CASE_UPPER
};
enum WinGUITextEditMode {
	WINGUI_TEXTEDIT_MODE_TEXT = 0,
	WINGUI_TEXTEDIT_MODE_NUMERIC,
	WINGUI_TEXTEDIT_MODE_PASSWORD
};

// Balloon Tips Support
enum WinGUITextEditBalloonTipIcon {
	WINGUI_TEXTEDIT_BALLOONTIP_ICON_NONE = 0,
	WINGUI_TEXTEDIT_BALLOONTIP_ICON_INFO,
	WINGUI_TEXTEDIT_BALLOONTIP_ICON_WARNING,
	WINGUI_TEXTEDIT_BALLOONTIP_ICON_ERROR
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUITextEditModel class
class WinGUITextEditModel : public WinGUIControlModel
{
public:
	WinGUITextEditModel( Int iResourceID );
	virtual ~WinGUITextEditModel();

	// Events
	virtual Bool OnTextChange() = 0;

	// View
	virtual const GChar * GetInitialText() const = 0;

	virtual const WinGUIRectangle * GetRectangle() const = 0;

	virtual Bool DontHideSelection() const = 0;
	virtual Bool AllowHorizScroll() const = 0;
	virtual Bool IsReadOnly() const = 0;

	virtual WinGUITextEditAlign GetTextAlign() const = 0;
	virtual WinGUITextEditCase GetTextCase() const = 0;
	virtual WinGUITextEditMode GetTextMode() const = 0;

protected:

};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUITextEdit class
class WinGUITextEdit : public WinGUIControl
{
public:
	WinGUITextEdit( WinGUIElement * pParent, WinGUITextEditModel * pModel );
	virtual ~WinGUITextEdit();

	// Enable / Disable
	Void Enable();
	Void Disable();

	// Undo Queue
	Bool CanUndo() const;
	Void Undo();

	// State
	Bool WasModified() const;
	Void SetReadOnly( Bool bReadOnly );

	// Text Access
	UInt GetTextLength() const;
	Void GetText( GChar * outText, UInt iMaxLength ) const;
	Void SetText( const GChar * strText );

	Void SetTextLimit( UInt iMaxLength );

	// Selection Access
	Void GetSelection( UInt * outStartIndex, UInt * outLength ) const; // outStartIndex = Caret Position when no selection
	Void SetSelection( UInt iStart, UInt iLength );
	Void ReplaceSelection( const GChar * strText );                    // Insert at caret position when no selection

	// Text Cues
	Void GetCueText( GChar * outText, UInt iMaxLength ) const;
	Void SetCueText( const GChar * strText, Bool bOnFocus );

	// Ballon Tips
	Void ShowBalloonTip( const GChar * strTitle, const GChar * strText, WinGUITextEditBalloonTipIcon iIcon );
	Void HideBalloonTip();

private:
	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode );
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUITextEdit.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITEXTEDIT_H

