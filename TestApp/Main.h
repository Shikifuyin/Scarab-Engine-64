/////////////////////////////////////////////////////////////////////////////////
// File : Main.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Test Entry Point
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "ThirdParty/WinGUI/WinGUI.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Some Resource IDs
#define RESID_BUTTON_TEST 101

/////////////////////////////////////////////////////////////////////////////////
// The MyButtonModel class
class MyButtonModel : public WinGUIButtonModel
{
public:
	MyButtonModel():WinGUIButtonModel(RESID_BUTTON_TEST) {}
	~MyButtonModel() {}

	// Events
	virtual Bool OnClick() {
		WinGUIMessageBoxOptions hOptions;
		hOptions.iType = WINGUI_MESSAGEBOX_OK;
		hOptions.iIcon = WINGUI_MESSAGEBOX_ICON_INFO;
		hOptions.iDefaultResponse = WINGUI_MESSAGEBOX_RESPONSE_OK;
		hOptions.bMustAnswer = true;

		WinGUIFn->SpawnMessageBox( TEXT("Sample Message"), TEXT("Hello !"), hOptions );

		return true;
	}
	virtual Bool OnDblClick() { return false; }

	// View
	virtual const GChar * GetText() const { return TEXT("Press Me !"); }

	virtual UInt GetPositionX() const { return 10; }
	virtual UInt GetPositionY() const { return 10; }
	virtual UInt GetWidth() const { return 100; }
	virtual UInt GetHeight() const { return 30; }
};

/////////////////////////////////////////////////////////////////////////////////
// The MyWindowModel class
class MyWindowModel : public WinGUIWindowModel
{
public:
	MyWindowModel():WinGUIWindowModel(0) {
		StringFn->NCopy( m_strClassName, TEXT("MainAppWindow"), 31 );
	}
	~MyWindowModel() {}

	// Events
	virtual Bool OnClose() {
		WinGUIFn->DestroyAppWindow();
		return true;
	}

	// View
	virtual const GChar * GetClassNameID() const { return m_strClassName; }

	virtual const GChar * GetTitle() const { return TEXT("Sample GUI Application"); }

	virtual UInt GetPositionX() const { return 100; }
	virtual UInt GetPositionY() const { return 100; }
	virtual UInt GetWidth() const { return 640; }
	virtual UInt GetHeight() const { return 480; }

	virtual Bool HasSystemMenu() const { return true; }
	virtual Bool HasMinimizeButton() const { return true; }
	virtual Bool HasMaximizeButton() const { return false; }

	virtual Bool AllowResizing() const { return false; }

private:
	GChar m_strClassName[32];
};
