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

#define RESID_RADIOBUTTON_A_TEST 103
#define RESID_RADIOBUTTON_B_TEST 104

// Prototypes
class MyButtonModel;
class MyCheckBoxModel;
class MyWindowModel;
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
	virtual Bool OnClick() { return false; }
	virtual Bool OnDblClick() { return false; }

	// View
	virtual const GChar * GetText() const { return TEXT("Above Button does something."); }

	virtual UInt GetPositionX() const { return 10; }
	virtual UInt GetPositionY() const { return 50; }
	virtual UInt GetWidth() const { return 100; }
	virtual UInt GetHeight() const { return 30; }

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

	virtual UInt GetPositionX() const { return 10; }
	virtual UInt GetPositionY() const { return 80; }
	virtual UInt GetWidth() const { return 100; }
	virtual UInt GetHeight() const { return 30; }

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

	virtual UInt GetPositionX() const { return 10; }
	virtual UInt GetPositionY() const { return 110; }
	virtual UInt GetWidth() const { return 100; }
	virtual UInt GetHeight() const { return 30; }

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

	WinGUIRadioButtonGroup m_hRadioButtonGroup;
    MyRadioButtonModelA m_hRadioButtonModelA;
    MyRadioButtonModelB m_hRadioButtonModelB;
};
