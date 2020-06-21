/////////////////////////////////////////////////////////////////////////////////
// File : Main.cpp
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
// Third-Party Includes
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "Main.h"

/////////////////////////////////////////////////////////////////////////////////
// MyWindowModel implementation
MyWindowModel::MyWindowModel( MyApplication * pApplication ):
	WinGUIWindowModel(0)
{
	m_pApplication = pApplication;
	StringFn->NCopy( m_strClassName, TEXT("MainAppWindow"), 31 );
}
MyWindowModel::~MyWindowModel()
{
	// nothing to do
}

Bool MyWindowModel::OnClose()
{
	WinGUIFn->DestroyAppWindow();
	return true;
}

/////////////////////////////////////////////////////////////////////////////////
// MyTabsModel implementation
MyTabsModel::MyTabsModel( MyApplication * pApplication ):
	WinGUITabsModel(RESID_TABS_TEST)
{
	m_pApplication = pApplication;
}
MyTabsModel::~MyTabsModel()
{
	// nothing to do
}

Bool MyTabsModel::OnTabSelect()
{
	WinGUITabs * pTabs = (WinGUITabs*)m_pView;
	UInt iSelected = pTabs->GetSelectedTab();

	WinGUIContainer * pTabPane = NULL;
	switch( iSelected ) {
		case 0: pTabPane = (WinGUIContainer*)( m_pApplication->m_hContainerModelLeft.GetView() ); break;
		case 1: pTabPane = (WinGUIContainer*)( m_pApplication->m_hContainerModelRight.GetView() ); break;
		default: DebugAssert(false); break;
	}

	pTabs->SwitchSelectedTabPane( pTabPane );

	return true;
}

/////////////////////////////////////////////////////////////////////////////////
// MyContainerModelLeft implementation
MyContainerModelLeft::MyContainerModelLeft( MyApplication * pApplication ):
	WinGUIContainerModel(RESID_CONTAINER_LEFT_TEST)
{
	m_pApplication = pApplication;
	StringFn->NCopy( m_strClassName, TEXT("LeftContainer"), 31 );
}
MyContainerModelLeft::~MyContainerModelLeft()
{
	// nothing to do
}

const WinGUIRectangle * MyContainerModelLeft::GetRectangle() const
{
	static WinGUIRectangle hDisplayRect;
	((WinGUITabs*)(m_pApplication->m_hTabsModel.GetView()))->GetDisplayArea( &hDisplayRect );
	return &hDisplayRect;
}

/////////////////////////////////////////////////////////////////////////////////
// MyContainerModelRight implementation
MyContainerModelRight::MyContainerModelRight( MyApplication * pApplication ):
	WinGUIContainerModel(RESID_CONTAINER_RIGHT_TEST)
{
	m_pApplication = pApplication;
	StringFn->NCopy( m_strClassName, TEXT("RightContainer"), 31 );
}
MyContainerModelRight::~MyContainerModelRight()
{
	// nothing to do
}

const WinGUIRectangle * MyContainerModelRight::GetRectangle() const
{
	static WinGUIRectangle hDisplayRect;
	((WinGUITabs*)(m_pApplication->m_hTabsModel.GetView()))->GetDisplayArea( &hDisplayRect );
	return &hDisplayRect;
}

/////////////////////////////////////////////////////////////////////////////////
// MyButtonModel implementation
MyButtonModel::MyButtonModel( MyApplication * pApplication ):
	WinGUIButtonModel(RESID_BUTTON_TEST)
{
	m_pApplication = pApplication;
}
MyButtonModel::~MyButtonModel()
{
	// nothing to do
}

Bool MyButtonModel::OnClick()
{
	WinGUICheckBox * pCheckBox = (WinGUICheckBox*)(m_pApplication->m_hCheckBoxModel.GetView());
	WinGUIRadioButton * pRadioButton = (WinGUIRadioButton*)(m_pApplication->m_hRadioButtonModelA.GetView());
	if ( pRadioButton->IsChecked() ) {
		WinGUIMessageBoxOptions hOptions;
		hOptions.iType = WINGUI_MESSAGEBOX_OK;
		hOptions.iIcon = WINGUI_MESSAGEBOX_ICON_INFO;
		hOptions.iDefaultResponse = WINGUI_MESSAGEBOX_RESPONSE_OK;
		hOptions.bMustAnswer = true;

		WinGUIFn->SpawnMessageBox( TEXT( "Sample Message" ), TEXT( "Hello !" ), hOptions );

		return true;
	}

	return false;
}

///////////////////////////////////////////////////////////////////////////////////
//// MyGroupBoxModel implementation
//MyGroupBoxModel::MyGroupBoxModel( MyApplication * pApplication ):
//	WinGUIGroupBoxModel(RESID_GROUPBOX_TEST)
//{
//	m_pApplication = pApplication;
//}
//MyGroupBoxModel::~MyGroupBoxModel()
//{
//	// nothing to do
//}

/////////////////////////////////////////////////////////////////////////////////
// MyRadioButtonModelA implementation
MyRadioButtonModelA::MyRadioButtonModelA( MyApplication * pApplication ):
	WinGUIRadioButtonModel(RESID_RADIOBUTTON_A_TEST)
{
	m_pApplication = pApplication;
}
MyRadioButtonModelA::~MyRadioButtonModelA()
{
	// nothing to do
}

//const WinGUIRectangle * MyRadioButtonModelA::GetRectangle() const {
//	static WinGUIRectangle hRect;
//	((WinGUIGroupBox *)(m_pApplication->m_hGroupBoxModel.GetView()))->ComputeClientArea( &hRect, 8 );
//	hRect.iHeight >>= 1;
//	return &hRect;
//}

/////////////////////////////////////////////////////////////////////////////////
// MyRadioButtonModelB implementation
MyRadioButtonModelB::MyRadioButtonModelB( MyApplication * pApplication ):
	WinGUIRadioButtonModel(RESID_RADIOBUTTON_B_TEST)
{
	m_pApplication = pApplication;
}
MyRadioButtonModelB::~MyRadioButtonModelB()
{
	// nothing to do
}

//const WinGUIRectangle * MyRadioButtonModelB::GetRectangle() const {
//	static WinGUIRectangle hRect;
//	((WinGUIGroupBox *)(m_pApplication->m_hGroupBoxModel.GetView()))->ComputeClientArea( &hRect, 8 );
//	hRect.iHeight >>= 1;
//	hRect.iTop += hRect.iHeight;
//	return &hRect;
//}

/////////////////////////////////////////////////////////////////////////////////
// MyCheckBoxModel implementation
MyCheckBoxModel::MyCheckBoxModel( MyApplication * pApplication ):
	WinGUICheckBoxModel(RESID_CHECKBOX_TEST)
{
	m_pApplication = pApplication;
}
MyCheckBoxModel::~MyCheckBoxModel()
{
	// nothing to do
}

Bool MyCheckBoxModel::OnClick()
{
	WinGUICheckBox * pCheckBox = (WinGUICheckBox*)( GetView() );
	WinGUITextEdit * pTextEdit = (WinGUITextEdit*)( m_pApplication->m_hTextEditModel.GetView() );

	if ( pCheckBox->IsChecked() )
		pTextEdit->SetReadOnly( false );
	else
		pTextEdit->SetReadOnly( true );

	return true;
}

/////////////////////////////////////////////////////////////////////////////////
// MyTextEditModel implementation
MyTextEditModel::MyTextEditModel( MyApplication * pApplication ):
	WinGUITextEditModel(RESID_TEXTEDIT_TEST)
{
	m_pApplication = pApplication;
}
MyTextEditModel::~MyTextEditModel()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// MyApplication implementation
MyApplication::MyApplication():
	m_hAppWindowModel(this),

	m_hTabsModel(this),

	m_hContainerModelLeft(this),

	m_hButtonModel(this),
	//m_hGroupBoxModel(this),
	m_hRadioButtonModelA(this),
	m_hRadioButtonModelB(this),
	m_hRadioButtonGroup(),

	m_hContainerModelRight(this),

	m_hCheckBoxModel(this),
	m_hTextEditModel(this)
{
	// App Window
	WinGUIFn->CreateAppWindow( &m_hAppWindowModel );
	WinGUIWindow * pAppWindow = WinGUIFn->GetAppWindow();

	// Tabs
	WinGUITabs * pTabs = WinGUIFn->CreateTabs( pAppWindow, &m_hTabsModel );

	// Left Container
	WinGUIContainer * pContainerLeft = WinGUIFn->CreateContainer( pAppWindow, &m_hContainerModelLeft );

	// A Button
	WinGUIFn->CreateButton( pContainerLeft, &m_hButtonModel );

	// A GroupBox
	//WinGUIFn->CreateGroupBox( pContainerLeft, &m_hGroupBoxModel );

	// A couple Radio Buttons
	WinGUIRadioButton * pRadioButtonA = WinGUIFn->CreateRadioButton( pContainerLeft, &m_hRadioButtonModelA );
	WinGUIRadioButton * pRadioButtonB = WinGUIFn->CreateRadioButton( pContainerLeft, &m_hRadioButtonModelB );
	m_hRadioButtonGroup.AddButton( pRadioButtonA );
	m_hRadioButtonGroup.AddButton( pRadioButtonB );
	pRadioButtonA->SetGroup( &m_hRadioButtonGroup );
	pRadioButtonB->SetGroup( &m_hRadioButtonGroup );
	pRadioButtonA->Check();

	// Right Container
	WinGUIContainer * pContainerRight = WinGUIFn->CreateContainer( pAppWindow, &m_hContainerModelRight );

	// A CheckBox
	WinGUICheckBox * pCheckBox = WinGUIFn->CreateCheckBox( pContainerRight, &m_hCheckBoxModel );
	pCheckBox->Check();

	// A TextEdit
	WinGUITextEdit * pTextEdit = WinGUIFn->CreateTextEdit( pContainerRight, &m_hTextEditModel );
	pTextEdit->SetCueText( TEXT("Type Stuff"), false );
	pTextEdit->SetTextLimit( 32 );

	// Done
	pTabs->SelectTab( 0 );
	pTabs->SwitchSelectedTabPane( pContainerLeft );

	pAppWindow->SetVisible( true );
}
MyApplication::~MyApplication()
{
	// nothing to do
}

/////////////////////////////////////////////////////////////////////////////////
// Entry Point
int APIENTRY wWinMain( _In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow )
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    MyApplication hApplication;

    return WinGUIFn->MessageLoop();
}

