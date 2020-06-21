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
#define RESID_TABS_TEST 100

#define RESID_CONTAINER_LEFT_TEST 101

#define RESID_BUTTON_TEST 110
#define RESID_GROUPBOX_TEST 111
#define RESID_RADIOBUTTON_A_TEST 112
#define RESID_RADIOBUTTON_B_TEST 113

#define RESID_CONTAINER_RIGHT_TEST 102

#define RESID_CHECKBOX_TEST 120
#define RESID_TEXTEDIT_TEST 121

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

	virtual const WinGUIRectangle * GetRectangle() const {
		static WinGUIRectangle hRect = { 100, 100, 800, 600 };
		return &hRect;
	}

	virtual Bool HasSystemMenu() const { return true; }
	virtual Bool HasMinimizeButton() const { return true; }
	virtual Bool HasMaximizeButton() const { return false; }

	virtual Bool AllowResizing() const { return false; }

private:
	MyApplication * m_pApplication;
	GChar m_strClassName[32];
};

/////////////////////////////////////////////////////////////////////////////////
// The MyTabsModel class
class MyTabsModel : public WinGUITabsModel
{
public:
	MyTabsModel( MyApplication * pApplication );
	~MyTabsModel();

	// Events
	virtual Bool OnTabSelect();

	// View
	virtual UInt GetTabCount() const { return 2; }
	virtual GChar * GetTabLabel( UInt iTabIndex ) const {
		static GChar strLabels[2][32] = {
			TEXT("A Tab"),
			TEXT("Another Tab")
		};
		DebugAssert( iTabIndex < 2 );
		return strLabels[iTabIndex];
	}
	virtual Void * GetTabUserData( UInt iTabIndex ) const { return NULL; }

private:
	MyApplication * m_pApplication;
};

/////////////////////////////////////////////////////////////////////////////////
// The MyContainerModelLeft class
class MyContainerModelLeft : public WinGUIContainerModel
{
public:
	MyContainerModelLeft( MyApplication * pApplication );
	~MyContainerModelLeft();

	// View
	virtual const GChar * GetClassNameID() const { return m_strClassName; }

	virtual const WinGUIRectangle * GetRectangle() const;

	virtual Bool AllowResizing() const { return false; }

private:
	MyApplication * m_pApplication;
	GChar m_strClassName[32];
};

/////////////////////////////////////////////////////////////////////////////////
// The MyContainerModelRight class
class MyContainerModelRight : public WinGUIContainerModel
{
public:
	MyContainerModelRight( MyApplication * pApplication );
	~MyContainerModelRight();

	// View
	virtual const GChar * GetClassNameID() const { return m_strClassName; }

	virtual const WinGUIRectangle * GetRectangle() const;

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

	virtual const WinGUIRectangle * GetRectangle() const {
		static WinGUIRectangle hRect = { 10, 10, 100, 20 };
		return &hRect;
	}

private:
	MyApplication * m_pApplication;
};

///////////////////////////////////////////////////////////////////////////////////
//// The MyGroupBoxModel class
//class MyGroupBoxModel : public WinGUIGroupBoxModel
//{
//public:
//	MyGroupBoxModel( MyApplication * pApplication );
//	~MyGroupBoxModel();
//
//	// View
//	virtual const GChar * GetText() const { return TEXT("Choose One :"); }
//
//	virtual const WinGUIRectangle * GetRectangle() const {
//		static WinGUIRectangle hRect = { 10, 40, 100, 100 };
//		return &hRect;
//	}
//
//private:
//	MyApplication * m_pApplication;
//};

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

	virtual const WinGUIRectangle * GetRectangle() const {
		static WinGUIRectangle hRect = { 10, 40, 100, 20 };
		return &hRect;
	}

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

	virtual const WinGUIRectangle * GetRectangle() const {
		static WinGUIRectangle hRect = { 10, 60, 100, 20 };
		return &hRect;
	}

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

	virtual const WinGUIRectangle * GetRectangle() const {
		static WinGUIRectangle hRect = { 10, 10, 100, 20 };
		return &hRect;
	}

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

	virtual const WinGUIRectangle * GetRectangle() const {
		static WinGUIRectangle hRect = { 10, 40, 100, 20 };
		return &hRect;
	}

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

	MyTabsModel m_hTabsModel;
    
	MyContainerModelLeft m_hContainerModelLeft;

	MyButtonModel m_hButtonModel;
	//MyGroupBoxModel m_hGroupBoxModel;
    MyRadioButtonModelA m_hRadioButtonModelA;
    MyRadioButtonModelB m_hRadioButtonModelB;
	WinGUIRadioButtonGroup m_hRadioButtonGroup;

	MyContainerModelRight m_hContainerModelRight;

    MyCheckBoxModel m_hCheckBoxModel;
    MyTextEditModel m_hTextEditModel;
};
