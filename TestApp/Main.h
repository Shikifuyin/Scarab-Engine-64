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

#define RESID_CONTAINER_CENTER_TEST 1002

#define RESID_TABLE_TEST 1020

#define RESID_CONTAINER_RIGHT_TEST 1003

#define RESID_CHECKBOX_TEST 1030
#define RESID_TEXTEDIT_TEST 1031
#define RESID_COMBOBOX_TEST 1032
#define RESID_STATIC_RECT_TEST 1033
#define RESID_PROGRESSBAR_TEST 1034

#define RESID_STATUSBAR_TEST 1004

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
	virtual Bool OnSelect();

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
// The MyContainerModelCenter class
class MyContainerModelCenter : public WinGUIContainerModel
{
public:
	MyContainerModelCenter( MyApplication * pApplication );
	~MyContainerModelCenter();

	// Layout
	virtual const WinGUILayout * GetLayout() const;

private:
	MyApplication * m_pApplication;
};

/////////////////////////////////////////////////////////////////////////////////
// The MyTableModel class
class MyTableModel : public WinGUITableModel
{
public:
	MyTableModel( MyApplication * pApplication );
	~MyTableModel();

	Void Initialize();

	// Layout
	virtual const WinGUILayout * GetLayout() const;

	// Callback Events
	virtual GChar * OnRequestItemLabel( UInt iItemIndex, UInt iSubItemIndex, Void * pItemData );
	virtual Void OnUpdateItemLabel( UInt iItemIndex, UInt iSubItemIndex, Void * pItemData, const GChar * strItemLabel );

private:
	MyApplication * m_pApplication;

	struct _column {
		GChar strLabel[64];
	} m_arrColumn[4];

	struct _item {
		struct _subitem {
			GChar strLabel[64];
		} arrSubItems[4];
	} m_arrItems[4];
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

	Void Initialize();

	// Layout
	virtual const WinGUILayout * GetLayout() const;

	// Events
	virtual Bool OnSelectionOK();

	// Item Callback Events
	virtual Void OnRequestItemLabel( GChar * outBuffer, UInt iMaxLength, UInt iItemIndex, Void * pItemData );

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
// The MyProgressBarModel class
class MyProgressBarModel : public WinGUIProgressBarModel
{
public:
	MyProgressBarModel( MyApplication * pApplication );
	~MyProgressBarModel();

	// Layout
	virtual const WinGUILayout * GetLayout() const;

private:
	MyApplication * m_pApplication;
};

/////////////////////////////////////////////////////////////////////////////////
// The MyStatusBarModel class
class MyStatusBarModel : public WinGUIStatusBarModel
{
public:
	MyStatusBarModel( MyApplication * pApplication );
	~MyStatusBarModel();

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

	MyContainerModelCenter m_hContainerModelCenter;

	MyTableModel m_hTableModel;

	MyContainerModelRight m_hContainerModelRight;

    MyCheckBoxModel m_hCheckBoxModel;
    MyTextEditModel m_hTextEditModel;
    MyComboBoxModel m_hComboBoxModel;
	MyStaticRectModel m_hStaticRectModel;
	MyProgressBarModel m_hProgressBarModel;

	MyStatusBarModel m_hStatusBarModel;
};
