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
}

Bool MyWindowModel::OnClose()
{
	WinGUIFn->DestroyAppWindow();
	return true;
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
// MyGroupBoxModel implementation
MyGroupBoxModel::MyGroupBoxModel( MyApplication * pApplication ):
	WinGUIGroupBoxModel(RESID_GROUPBOX_TEST)
{
	m_pApplication = pApplication;
}
MyGroupBoxModel::~MyGroupBoxModel()
{
	// nothing to do
}

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

UInt MyRadioButtonModelA::GetPositionX() const
{
	UInt iLeft, iTop, iRight, iBottom;
	((WinGUIGroupBox *)(m_pApplication->m_hGroupBoxModel.GetView()))->GetClientArea( &iLeft, &iTop, &iRight, &iBottom, 8 );
	return iLeft;
}
UInt MyRadioButtonModelA::GetPositionY() const
{
	UInt iLeft, iTop, iRight, iBottom;
	((WinGUIGroupBox *)(m_pApplication->m_hGroupBoxModel.GetView()))->GetClientArea( &iLeft, &iTop, &iRight, &iBottom, 8 );
	return iTop;
}
UInt MyRadioButtonModelA::GetWidth() const
{
	UInt iLeft, iTop, iRight, iBottom;
	((WinGUIGroupBox *)(m_pApplication->m_hGroupBoxModel.GetView()))->GetClientArea( &iLeft, &iTop, &iRight, &iBottom, 8 );
	return iRight - iLeft;
}
UInt MyRadioButtonModelA::GetHeight() const
{
	UInt iLeft, iTop, iRight, iBottom;
	((WinGUIGroupBox *)(m_pApplication->m_hGroupBoxModel.GetView()))->GetClientArea( &iLeft, &iTop, &iRight, &iBottom, 8 );
	return (iBottom - iTop) >> 1;
}

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

UInt MyRadioButtonModelB::GetPositionX() const
{
	UInt iLeft, iTop, iRight, iBottom;
	((WinGUIGroupBox *)(m_pApplication->m_hGroupBoxModel.GetView()))->GetClientArea( &iLeft, &iTop, &iRight, &iBottom, 8 );
	return iLeft;
}
UInt MyRadioButtonModelB::GetPositionY() const
{
	UInt iLeft, iTop, iRight, iBottom;
	((WinGUIGroupBox *)(m_pApplication->m_hGroupBoxModel.GetView()))->GetClientArea( &iLeft, &iTop, &iRight, &iBottom, 8 );
	return iTop + ( (iBottom-iTop) >> 1 );
}
UInt MyRadioButtonModelB::GetWidth() const
{
	UInt iLeft, iTop, iRight, iBottom;
	((WinGUIGroupBox *)(m_pApplication->m_hGroupBoxModel.GetView()))->GetClientArea( &iLeft, &iTop, &iRight, &iBottom, 8 );
	return iRight - iLeft;
}
UInt MyRadioButtonModelB::GetHeight() const
{
	UInt iLeft, iTop, iRight, iBottom;
	((WinGUIGroupBox *)(m_pApplication->m_hGroupBoxModel.GetView()))->GetClientArea( &iLeft, &iTop, &iRight, &iBottom, 8 );
	return (iBottom - iTop) >> 1;
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

	m_hButtonModel(this),

	m_hCheckBoxModel(this),

	m_hGroupBoxModel(this),
	m_hRadioButtonGroup(),
	m_hRadioButtonModelA(this),
	m_hRadioButtonModelB(this),

	m_hTextEditModel(this)
{
	WinGUIFn->CreateAppWindow( &m_hAppWindowModel );
	WinGUIWindow * pAppWindow = WinGUIFn->GetAppWindow();

	WinGUIFn->CreateButton( pAppWindow, &m_hButtonModel );

	WinGUICheckBox * pCheckBox = WinGUIFn->CreateCheckBox( pAppWindow, &m_hCheckBoxModel );
	pCheckBox->Check();

	WinGUIFn->CreateGroupBox( pAppWindow, &m_hGroupBoxModel );

	WinGUIRadioButton * pRadioButtonA = WinGUIFn->CreateRadioButton( pAppWindow, &m_hRadioButtonModelA );
	WinGUIRadioButton * pRadioButtonB = WinGUIFn->CreateRadioButton( pAppWindow, &m_hRadioButtonModelB );
	m_hRadioButtonGroup.AddButton( pRadioButtonA );
	m_hRadioButtonGroup.AddButton( pRadioButtonB );
	pRadioButtonA->SetGroup( &m_hRadioButtonGroup );
	pRadioButtonB->SetGroup( &m_hRadioButtonGroup );
	pRadioButtonA->Check();

	WinGUITextEdit * pTextEdit = WinGUIFn->CreateTextEdit( pAppWindow, &m_hTextEditModel );
	pTextEdit->SetCueText( TEXT("Type Stuff"), false );
	pTextEdit->SetTextLimit( 32 );

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

