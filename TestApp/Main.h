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
#define RESID_TABS_TEST 1000

#define RESID_CONTAINER_LEFT_TEST 1001

#define RESID_BUTTON_TEST 1010
#define RESID_GROUPBOX_TEST 1011
#define RESID_RADIOBUTTON_A_TEST 1012
#define RESID_RADIOBUTTON_B_TEST 1013
#define RESID_STATIC_TEXT_TEST 1014

#define RESID_CONTAINER_RIGHT_TEST 1002

#define RESID_CHECKBOX_TEST 1020
#define RESID_TEXTEDIT_TEST 1021
#define RESID_COMBOBOX_TEST 1022
#define RESID_STATIC_RECT_TEST 1023

// Prototypes
class MyApplication;

/////////////////////////////////////////////////////////////////////////////////
// The MyWindowModel class
class MyWindowModel : public WinGUIWindowModel
{
public:
	MyWindowModel( MyApplication * pApplication );
	~MyWindowModel();

	// Layout
	virtual const WinGUILayout * GetLayout() const;

	// Events
	virtual Bool OnClose();

private:
	MyApplication * m_pApplication;
};

/////////////////////////////////////////////////////////////////////////////////
// The MyTabsModel class
class MyTabsModel : public WinGUITabsModel
{
public:
	MyTabsModel( MyApplication * pApplication );
	~MyTabsModel();

	// Layout
	virtual const WinGUILayout * GetLayout() const;

	// Events
	virtual Bool OnTabSelect();

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

	// Layout
	virtual const WinGUILayout * GetLayout() const;

private:
	MyApplication * m_pApplication;
};

/////////////////////////////////////////////////////////////////////////////////
// The MyContainerModelRight class
class MyContainerModelRight : public WinGUIContainerModel
{
public:
	MyContainerModelRight( MyApplication * pApplication );
	~MyContainerModelRight();

	// Layout
	virtual const WinGUILayout * GetLayout() const;

private:
	MyApplication * m_pApplication;
};

/////////////////////////////////////////////////////////////////////////////////
// The MyButtonModel class
class MyButtonModel : public WinGUIButtonModel
{
public:
	MyButtonModel( MyApplication * pApplication );
	~MyButtonModel();

	// Layout
	virtual const WinGUILayout * GetLayout() const;

	// Events
	virtual Bool OnClick();

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

	// Layout
	virtual const WinGUILayout * GetLayout() const;

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

	// Layout
	virtual const WinGUILayout * GetLayout() const;

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

	// Layout
	virtual const WinGUILayout * GetLayout() const;

private:
	MyApplication * m_pApplication;
};

/////////////////////////////////////////////////////////////////////////////////
// The MyStaticTextModel class
class MyStaticTextModel : public WinGUIStaticModel
{
public:
	MyStaticTextModel( MyApplication * pApplication );
	~MyStaticTextModel();

	// Layout
	virtual const WinGUILayout * GetLayout() const;

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

	// Layout
	virtual const WinGUILayout * GetLayout() const;

	// Events
	virtual Bool OnClick();

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

	// Layout
	virtual const WinGUILayout * GetLayout() const;

private:
	MyApplication * m_pApplication;
};

/////////////////////////////////////////////////////////////////////////////////
// The MyComboBoxModel class
class MyComboBoxModel : public WinGUIComboBoxModel
{
public:
	MyComboBoxModel( MyApplication * pApplication );
	~MyComboBoxModel();

	// Layout
	virtual const WinGUILayout * GetLayout() const;

	// Content Data
	virtual UInt GetItemCount() const						 { return 4; }
	virtual const GChar * GetItemString( UInt iIndex ) const { DebugAssert( iIndex < 4 ); return m_arrLabels[iIndex]; }
	virtual Void * GetItemData( UInt iIndex ) const			 { DebugAssert( iIndex < 4 ); return (Void*)(m_arrData[iIndex]); }

	// Events
	virtual Bool OnSelectionOK();

private:
	MyApplication * m_pApplication;

	// Content Data
	GChar m_arrLabels[4][32];
	GChar m_arrData[4][32];
};

/////////////////////////////////////////////////////////////////////////////////
// The MyStaticRectModel class
class MyStaticRectModel : public WinGUIStaticModel
{
public:
	MyStaticRectModel( MyApplication * pApplication );
	~MyStaticRectModel();

	// Layout
	virtual const WinGUILayout * GetLayout() const;

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
	MyGroupBoxModel m_hGroupBoxModel;
    MyRadioButtonModelA m_hRadioButtonModelA;
    MyRadioButtonModelB m_hRadioButtonModelB;
	WinGUIRadioButtonGroup m_hRadioButtonGroup;
	MyStaticTextModel m_hStaticTextModel;

	MyContainerModelRight m_hContainerModelRight;

    MyCheckBoxModel m_hCheckBoxModel;
    MyTextEditModel m_hTextEditModel;
    MyComboBoxModel m_hComboBoxModel;
	MyStaticRectModel m_hStaticRectModel;
};
