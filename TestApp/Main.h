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
#define RESID_BUTTON_TEST   101

#define RESID_CHECKBOX_TEST 102

#define RESID_GROUPBOX_TEST 103

#define RESID_RADIOBUTTON_A_TEST 104
#define RESID_RADIOBUTTON_B_TEST 105

#define RESID_TEXTEDIT_TEST 106

// Prototypes
class MyApplication;

/////////////////////////////////////////////////////////////////////////////////
// The MyWindowModel class
class MyWindowModel : public WinGUIWindowModel
{
public:
	MyWindowModel( MyApplication * pApplication );
	~MyWindowModel();

	// Events
	virtual Bool OnClose();

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
	MyApplication * m_pApplication;
	GChar m_strClassName[32];
};

/////////////////////////////////////////////////////////////////////////////////
// The MyButtonModel class
class MyButtonModel : public WinGUIButtonModel
{
public:
	MyButtonModel( MyApplication * pApplication );
	~MyButtonModel();

	// Events
	virtual Bool OnClick();
	virtual Bool OnDblClick() { return false; }

	// View
	virtual const GChar * GetText() const { return TEXT("Press Me !"); }

	virtual UInt GetPositionX() const { return 10; }
	virtual UInt GetPositionY() const { return 10; }
	virtual UInt GetWidth() const { return 100; }
	virtual UInt GetHeight() const { return 30; }

private:
	MyApplication * m_pApplication;
};

/////////////////////////////////////////////////////////////////////////////////
// The MyCheckBoxModel class
class MyCheckBoxModel : public WinGUICheckBoxModel
{
public:
	MyCheckBoxModel( MyApplication * pApplication );
	~MyCheckBoxModel();

	// Events
	virtual Bool OnClick();
	virtual Bool OnDblClick() { return false; }

	// View
	virtual const GChar * GetText() const { return TEXT("Enable TextEdit"); }

	virtual UInt GetPositionX() const { return 10; }
	virtual UInt GetPositionY() const { return 50; }
	virtual UInt GetWidth() const { return 100; }
	virtual UInt GetHeight() const { return 30; }

private:
	MyApplication * m_pApplication;
};

/////////////////////////////////////////////////////////////////////////////////
// The MyGroupBoxModel class
class MyGroupBoxModel : public WinGUIGroupBoxModel
{
public:
	MyGroupBoxModel( MyApplication * pApplication );
	~MyGroupBoxModel();

	// View
	virtual const GChar * GetText() const { return TEXT("Choose One :"); }

	virtual UInt GetPositionX() const { return 10; }
	virtual UInt GetPositionY() const { return 80; }
	virtual UInt GetWidth() const { return 100; }
	virtual UInt GetHeight() const { return 100; }

private:
	MyApplication * m_pApplication;
};

/////////////////////////////////////////////////////////////////////////////////
// The MyRadioButtonModelA class
class MyRadioButtonModelA : public WinGUIRadioButtonModel
{
public:
	MyRadioButtonModelA( MyApplication * pApplication );
	~MyRadioButtonModelA();

	// Events
	virtual Bool OnClick() { return false; }
	virtual Bool OnDblClick() { return false; }

	// View
	virtual const GChar * GetText() const { return TEXT("Option A"); }

	virtual UInt GetPositionX() const;
	virtual UInt GetPositionY() const;
	virtual UInt GetWidth() const;
	virtual UInt GetHeight() const;

private:
	MyApplication * m_pApplication;
};

/////////////////////////////////////////////////////////////////////////////////
// The MyRadioButtonModelB class
class MyRadioButtonModelB : public WinGUIRadioButtonModel
{
public:
	MyRadioButtonModelB( MyApplication * pApplication );
	~MyRadioButtonModelB();

	// Events
	virtual Bool OnClick() { return false; }
	virtual Bool OnDblClick() { return false; }

	// View
	virtual const GChar * GetText() const { return TEXT("Option B"); }

	virtual UInt GetPositionX() const;
	virtual UInt GetPositionY() const;
	virtual UInt GetWidth() const;
	virtual UInt GetHeight() const;

private:
	MyApplication * m_pApplication;
};

/////////////////////////////////////////////////////////////////////////////////
// The MyTextEditModel class
class MyTextEditModel : public WinGUITextEditModel
{
public:
	MyTextEditModel( MyApplication * pApplication );
	~MyTextEditModel();

	// Events
	virtual Bool OnTextChange() { return false; }

	// View
	virtual const GChar * GetInitialText() const { return TEXT(""); }

	virtual UInt GetPositionX() const { return 10; }
	virtual UInt GetPositionY() const { return 180; }
	virtual UInt GetWidth() const { return 200; }
	virtual UInt GetHeight() const { return 30; }

	virtual Bool DontHideSelection() const { return false; }
	virtual Bool AllowHorizScroll() const { return true; }
	virtual Bool IsReadOnly() const { return false; }

	virtual WinGUITextEditAlign GetTextAlign() const { return WINGUI_TEXTEDIT_ALIGN_LEFT; }
	virtual WinGUITextEditCase GetTextCase() const { return WINGUI_TEXTEDIT_CASE_BOTH; }
	virtual WinGUITextEditMode GetTextMode() const { return WINGUI_TEXTEDIT_MODE_TEXT; }

private:
	MyApplication * m_pApplication;
};

/////////////////////////////////////////////////////////////////////////////////
// The MyApplication class
class MyApplication
{
public:
	MyApplication();
	~MyApplication();

    MyWindowModel m_hAppWindowModel;
    
	MyButtonModel m_hButtonModel;

    MyCheckBoxModel m_hCheckBoxModel;

	MyGroupBoxModel m_hGroupBoxModel;
	WinGUIRadioButtonGroup m_hRadioButtonGroup;
    MyRadioButtonModelA m_hRadioButtonModelA;
    MyRadioButtonModelB m_hRadioButtonModelB;

    MyTextEditModel m_hTextEditModel;
};
