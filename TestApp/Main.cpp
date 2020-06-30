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

	m_hCreationParameters.hClientRect.iLeft = 100;
	m_hCreationParameters.hClientRect.iTop = 100;
	m_hCreationParameters.hClientRect.iWidth = 800;
	m_hCreationParameters.hClientRect.iHeight = 600;

	StringFn->Copy( m_hCreationParameters.strClassName, TEXT("MainWindow") );
	StringFn->Copy( m_hCreationParameters.strTitle, TEXT("Sample GUI Application") );
	m_hCreationParameters.bHasSystemMenu = true;
	m_hCreationParameters.bHasMinimizeButton = true;
	m_hCreationParameters.bHasMaximizeButton = false;
	m_hCreationParameters.bAllowResizing = false;
	m_hCreationParameters.bClipChildren = false;
	m_hCreationParameters.bClipSibblings = false;
}
MyWindowModel::~MyWindowModel()
{
	// nothing to do
}

const WinGUILayout * MyWindowModel::GetLayout() const
{
	// Top-Most Windows don't use layouts
	return NULL;
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

	m_hCreationParameters.bSingleLine = true;
	m_hCreationParameters.bFixedWidth = true;

	m_hCreationParameters.iTabCount = 3;

	StringFn->Copy( m_hCreationParameters.arrTabs[0].strLabel, TEXT("Left Tab") );
	m_hCreationParameters.arrTabs[0].pUserData = NULL;
	StringFn->Copy( m_hCreationParameters.arrTabs[1].strLabel, TEXT("Center Tab") );
	m_hCreationParameters.arrTabs[1].pUserData = NULL;
	StringFn->Copy( m_hCreationParameters.arrTabs[2].strLabel, TEXT("Right Tab") );
	m_hCreationParameters.arrTabs[2].pUserData = NULL;
}
MyTabsModel::~MyTabsModel()
{
	// nothing to do
}

const WinGUILayout * MyTabsModel::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	hLayout.UseScalingPosition = true;
	hLayout.ScalingPosition.fX = 0.0f;
	hLayout.ScalingPosition.fY = 0.0f;

	hLayout.UseScalingSize = true;
	hLayout.ScalingSize.fX = 1.0f;
	hLayout.ScalingSize.fY = 1.0f;

	return &hLayout;
}

Bool MyTabsModel::OnSelect()
{
	WinGUITabs * pTabs = (WinGUITabs*)m_pController;
	UInt iSelected = pTabs->GetSelectedTab();

	WinGUIContainer * pTabPane = NULL;
	switch( iSelected ) {
		case 0: pTabPane = (WinGUIContainer*)( m_pApplication->m_hContainerModelLeft.GetController() ); break;
		case 1: pTabPane = (WinGUIContainer*)( m_pApplication->m_hContainerModelCenter.GetController() ); break;
		case 2: pTabPane = (WinGUIContainer*)( m_pApplication->m_hContainerModelRight.GetController() ); break;
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

	StringFn->Copy( m_hCreationParameters.strClassName, TEXT("LeftContainer") );
	m_hCreationParameters.bAllowResizing = false;
	m_hCreationParameters.bClipChildren = true;
	m_hCreationParameters.bClipSibblings = true;
}
MyContainerModelLeft::~MyContainerModelLeft()
{
	// nothing to do
}

const WinGUILayout * MyContainerModelLeft::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	WinGUITabs * pTabs = (WinGUITabs *)(m_pApplication->m_hTabsModel.GetController());

	WinGUIRectangle hRect;
	pTabs->GetDisplayArea( &hRect );

	hLayout.UseScalingPosition = false;
	hLayout.FixedPosition.iX = hRect.iLeft;
	hLayout.FixedPosition.iY = hRect.iTop;

	hLayout.UseScalingSize = false;
	hLayout.FixedSize.iX = hRect.iWidth;
	hLayout.FixedSize.iY = hRect.iHeight;

	return &hLayout;
}

/////////////////////////////////////////////////////////////////////////////////
// MyButtonModel implementation
MyButtonModel::MyButtonModel( MyApplication * pApplication ):
	WinGUIButtonModel(RESID_BUTTON_TEST)
{
	m_pApplication = pApplication;

	StringFn->Copy( m_hCreationParameters.strLabel, TEXT("Press Me !") );
	m_hCreationParameters.bCenterLabel = true;
	m_hCreationParameters.bEnableTabStop = true;
}
MyButtonModel::~MyButtonModel()
{
	// nothing to do
}

const WinGUILayout * MyButtonModel::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	hLayout.UseScalingPosition = false;
	hLayout.FixedPosition.iX = 10;
	hLayout.FixedPosition.iY = 10;

	hLayout.UseScalingSize = false;
	hLayout.FixedSize.iX = 100;
	hLayout.FixedSize.iY = 40;

	return &hLayout;
}

Bool MyButtonModel::OnClick()
{
	WinGUIRadioButton * pRadioButton = (WinGUIRadioButton*)(m_pApplication->m_hRadioButtonModelA.GetController());

	WinGUIMessageBoxOptions hOptions;
	hOptions.iType = WINGUI_MESSAGEBOX_OK;
	hOptions.iIcon = WINGUI_MESSAGEBOX_ICON_INFO;
	hOptions.iDefaultResponse = WINGUI_MESSAGEBOX_RESPONSE_OK;
	hOptions.bMustAnswer = true;

	if ( pRadioButton->IsChecked() )
		WinGUIFn->SpawnMessageBox( TEXT( "Sample Message" ), TEXT( "Hello !" ), hOptions );
	else
		WinGUIFn->SpawnMessageBox( TEXT( "Sample Message" ), TEXT( "GoodBye !" ), hOptions );

	return true;
}

/////////////////////////////////////////////////////////////////////////////////
// MyGroupBoxModel implementation
MyGroupBoxModel::MyGroupBoxModel( MyApplication * pApplication ):
	WinGUIGroupBoxModel(RESID_GROUPBOX_TEST)
{
	m_pApplication = pApplication;

	StringFn->Copy( m_hCreationParameters.strLabel, TEXT("Choose One :") );
}
MyGroupBoxModel::~MyGroupBoxModel()
{
	// nothing to do
}

const WinGUILayout * MyGroupBoxModel::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	hLayout.UseScalingPosition = false;
	hLayout.FixedPosition.iX = 10;
	hLayout.FixedPosition.iY = 60;

	hLayout.UseScalingSize = false;
	hLayout.FixedSize.iX = 140;
	hLayout.FixedSize.iY = 100;

	return &hLayout;
}

/////////////////////////////////////////////////////////////////////////////////
// MyRadioButtonModelA implementation
MyRadioButtonModelA::MyRadioButtonModelA( MyApplication * pApplication ):
	WinGUIRadioButtonModel(RESID_RADIOBUTTON_A_TEST)
{
	m_pApplication = pApplication;

	StringFn->Copy( m_hCreationParameters.strLabel, TEXT("Say Hello !") );
	m_hCreationParameters.bEnableTabStop = true;
}
MyRadioButtonModelA::~MyRadioButtonModelA()
{
	// nothing to do
}

const WinGUILayout * MyRadioButtonModelA::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	WinGUIGroupBox * pGroupBox = (WinGUIGroupBox*)( m_pApplication->m_hGroupBoxModel.GetController() );
	WinGUIRectangle hRect;
	pGroupBox->ComputeClientArea( &hRect, 8 );

	hLayout.UseScalingPosition = false;
	hLayout.FixedPosition.iX = hRect.iLeft;
	hLayout.FixedPosition.iY = hRect.iTop;

	hLayout.UseScalingSize = false;
	hLayout.FixedSize.iX = hRect.iWidth;
	hLayout.FixedSize.iY = hRect.iHeight >> 1;

	return &hLayout;
}

/////////////////////////////////////////////////////////////////////////////////
// MyRadioButtonModelB implementation
MyRadioButtonModelB::MyRadioButtonModelB( MyApplication * pApplication ):
	WinGUIRadioButtonModel(RESID_RADIOBUTTON_B_TEST)
{
	m_pApplication = pApplication;

	StringFn->Copy( m_hCreationParameters.strLabel, TEXT("Say GoodBye !") );
	m_hCreationParameters.bEnableTabStop = true;
}
MyRadioButtonModelB::~MyRadioButtonModelB()
{
	// nothing to do
}

const WinGUILayout * MyRadioButtonModelB::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	WinGUIGroupBox * pGroupBox = (WinGUIGroupBox*)( m_pApplication->m_hGroupBoxModel.GetController() );
	WinGUIRectangle hRect;
	pGroupBox->ComputeClientArea( &hRect, 8 );

	hLayout.UseScalingPosition = false;
	hLayout.FixedPosition.iX = hRect.iLeft;
	hLayout.FixedPosition.iY = hRect.iTop + (hRect.iHeight >> 1);

	hLayout.UseScalingSize = false;
	hLayout.FixedSize.iX = hRect.iWidth;
	hLayout.FixedSize.iY = hRect.iHeight >> 1;

	return &hLayout;
}

/////////////////////////////////////////////////////////////////////////////////
// MyStaticTextModel implementation
MyStaticTextModel::MyStaticTextModel( MyApplication * pApplication ):
	WinGUIStaticModel(RESID_STATIC_TEXT_TEST)
{
	m_pApplication = pApplication;

	m_hCreationParameters.iType = WINGUI_STATIC_TEXT;
	m_hCreationParameters.bAddSunkenBorder = true;

	StringFn->Copy( m_hCreationParameters.hText.strLabel, TEXT("Not Dynamic ?") );
	m_hCreationParameters.hText.iAlign = WINGUI_STATIC_TEXT_ALIGN_CENTER;
	m_hCreationParameters.hText.iEllipsis = WINGUI_STATIC_TEXT_ELLIPSIS_NONE;
}
MyStaticTextModel::~MyStaticTextModel()
{
	// nothing to do
}

const WinGUILayout * MyStaticTextModel::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	hLayout.UseScalingPosition = false;
	hLayout.FixedPosition.iX = 10;
	hLayout.FixedPosition.iY = 200;

	hLayout.UseScalingSize = false;
	hLayout.FixedSize.iX = 100;
	hLayout.FixedSize.iY = 40;

	return &hLayout;
}

/////////////////////////////////////////////////////////////////////////////////
// MyContainerModelCenter implementation
MyContainerModelCenter::MyContainerModelCenter( MyApplication * pApplication ):
	WinGUIContainerModel(RESID_CONTAINER_CENTER_TEST)
{
	m_pApplication = pApplication;

	StringFn->Copy( m_hCreationParameters.strClassName, TEXT("CenterContainer") );
	m_hCreationParameters.bAllowResizing = false;
	m_hCreationParameters.bClipChildren = true;
	m_hCreationParameters.bClipSibblings = true;
}
MyContainerModelCenter::~MyContainerModelCenter()
{
	// nothing to do
}

const WinGUILayout * MyContainerModelCenter::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	WinGUITabs * pTabs = (WinGUITabs *)(m_pApplication->m_hTabsModel.GetController());

	WinGUIRectangle hRect;
	pTabs->GetDisplayArea( &hRect );

	hLayout.UseScalingPosition = false;
	hLayout.FixedPosition.iX = hRect.iLeft;
	hLayout.FixedPosition.iY = hRect.iTop;

	hLayout.UseScalingSize = false;
	hLayout.FixedSize.iX = hRect.iWidth;
	hLayout.FixedSize.iY = hRect.iHeight;

	return &hLayout;
}

/////////////////////////////////////////////////////////////////////////////////
// MyTableModel implementation
MyTableModel::MyTableModel( MyApplication * pApplication ):
	WinGUITableModel(RESID_TABLE_TEST)
{
	m_pApplication = pApplication;

	m_hCreationParameters.bVirtualTable = false;

	m_hCreationParameters.bHasBackBuffer = false;
	m_hCreationParameters.bHasSharedImageLists = false;

	m_hCreationParameters.iItemCallBackMode = WINGUI_TABLE_ITEMCALLBACK_LABELS;
	m_hCreationParameters.iStateCallBackMode = 0;

	m_hCreationParameters.iViewMode = WINGUI_TABLE_VIEW_DETAILED;
	m_hCreationParameters.bGroupMode = false;
	m_hCreationParameters.bHasHeadersInAllViews = false;

	m_hCreationParameters.bHasColumnHeaders = true;
	m_hCreationParameters.bHasStaticColumnHeaders = false;
	m_hCreationParameters.bHasDraggableColumnHeaders = true;
	m_hCreationParameters.bHasIconColumnOverflowButton = false;

	m_hCreationParameters.bHasCheckBoxes = true;
	m_hCreationParameters.bHasIconLabels = true;
	m_hCreationParameters.bHasEditableLabels = true;
	m_hCreationParameters.bHasSubItemImages = false;

	m_hCreationParameters.bSingleItemSelection = false;
	m_hCreationParameters.bIconSimpleSelection = false;

	m_hCreationParameters.bAutoSortAscending = false;
	m_hCreationParameters.bAutoSortDescending = false;

	m_hCreationParameters.bHasHotTrackingSingleClick = false;
	m_hCreationParameters.bHasHotTrackingDoubleClick = false;
	m_hCreationParameters.bHasHotTrackingSelection = false;

	m_hCreationParameters.bHasInfoTips = false;

	for( UInt i = 0; i < 4; ++i ) {
		StringFn->Format( m_arrColumn[i].strLabel, TEXT("Column_%d"), i );
		for( UInt j = 0; j < 4; ++j ) {
			StringFn->Format( m_arrItems[i].arrSubItems[j].strLabel, TEXT("Item_%d_%d"), i, j );
		}
	}
}
MyTableModel::~MyTableModel()
{
	// nothing to do
}

Void MyTableModel::Initialize()
{
	WinGUITable * pTable = (WinGUITable*)m_pController;

	pTable->AddColumn( 0, m_arrColumn[0].strLabel, 0, 0, 100 );
	pTable->AddColumn( 1, m_arrColumn[1].strLabel, 1, 1, 100 );
	pTable->AddColumn( 2, m_arrColumn[2].strLabel, 2, 2, 100 );
	pTable->AddColumn( 3, m_arrColumn[3].strLabel, 3, 3, 100 );

	pTable->AddItem( 0 );
	pTable->AddItem( 1 );
	pTable->AddItem( 2 );
	pTable->AddItem( 3 );
}

const WinGUILayout * MyTableModel::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	hLayout.UseScalingPosition = false;
	hLayout.FixedPosition.iX = 10;
	hLayout.FixedPosition.iY = 10;

	hLayout.UseScalingSize = false;
	hLayout.FixedSize.iX = 640;
	hLayout.FixedSize.iY = 480;

	return &hLayout;
}

GChar * MyTableModel::OnRequestItemLabel( UInt iItemIndex, UInt iSubItemIndex, Void * pItemData )
{
	DebugAssert( iItemIndex < 4 );
	DebugAssert( iSubItemIndex < 4 );
	return m_arrItems[iItemIndex].arrSubItems[iSubItemIndex].strLabel;
}
Void MyTableModel::OnUpdateItemLabel( UInt iItemIndex, UInt iSubItemIndex, Void * pItemData, const GChar * strItemLabel )
{
	DebugAssert( iItemIndex < 4 );
	DebugAssert( iSubItemIndex < 4 );
	StringFn->NCopy( m_arrItems[iItemIndex].arrSubItems[iSubItemIndex].strLabel, strItemLabel, 63 );
}

/////////////////////////////////////////////////////////////////////////////////
// MyContainerModelRight implementation
MyContainerModelRight::MyContainerModelRight( MyApplication * pApplication ):
	WinGUIContainerModel(RESID_CONTAINER_RIGHT_TEST)
{
	m_pApplication = pApplication;

	StringFn->Copy( m_hCreationParameters.strClassName, TEXT("RightContainer") );
	m_hCreationParameters.bAllowResizing = false;
	m_hCreationParameters.bClipChildren = true;
	m_hCreationParameters.bClipSibblings = true;
}
MyContainerModelRight::~MyContainerModelRight()
{
	// nothing to do
}

const WinGUILayout * MyContainerModelRight::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	WinGUITabs * pTabs = (WinGUITabs *)(m_pApplication->m_hTabsModel.GetController());

	WinGUIRectangle hRect;
	pTabs->GetDisplayArea( &hRect );

	hLayout.UseScalingPosition = false;
	hLayout.FixedPosition.iX = hRect.iLeft;
	hLayout.FixedPosition.iY = hRect.iTop;

	hLayout.UseScalingSize = false;
	hLayout.FixedSize.iX = hRect.iWidth;
	hLayout.FixedSize.iY = hRect.iHeight;

	return &hLayout;
}

/////////////////////////////////////////////////////////////////////////////////
// MyCheckBoxModel implementation
MyCheckBoxModel::MyCheckBoxModel( MyApplication * pApplication ):
	WinGUICheckBoxModel(RESID_CHECKBOX_TEST)
{
	m_pApplication = pApplication;

	StringFn->Copy( m_hCreationParameters.strLabel, TEXT("Enable Smthg ...") );
	m_hCreationParameters.bEnableTabStop = true;
}
MyCheckBoxModel::~MyCheckBoxModel()
{
	// nothing to do
}

const WinGUILayout * MyCheckBoxModel::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	hLayout.UseScalingPosition = false;
	hLayout.FixedPosition.iX = 10;
	hLayout.FixedPosition.iY = 10;

	hLayout.UseScalingSize = false;
	hLayout.FixedSize.iX = 100;
	hLayout.FixedSize.iY = 40;

	return &hLayout;
}

Bool MyCheckBoxModel::OnClick()
{
	WinGUICheckBox * pCheckBox = (WinGUICheckBox*)m_pController;
	WinGUITextEdit * pTextEdit = (WinGUITextEdit*)( m_pApplication->m_hTextEditModel.GetController() );

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

	StringFn->Copy( m_hCreationParameters.strInitialText, TEXT("Hmm ?!") );
	m_hCreationParameters.iAlign = WINGUI_TEXTEDIT_ALIGN_LEFT;
	m_hCreationParameters.iCase = WINGUI_TEXTEDIT_CASE_BOTH;
	m_hCreationParameters.iMode = WINGUI_TEXTEDIT_MODE_TEXT;
	m_hCreationParameters.bAllowHorizontalScroll = true;
	m_hCreationParameters.bDontHideSelection = false;
	m_hCreationParameters.bReadOnly = false;
	m_hCreationParameters.bEnableTabStop = true;
}
MyTextEditModel::~MyTextEditModel()
{
	// nothing to do
}

const WinGUILayout * MyTextEditModel::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	hLayout.UseScalingPosition = false;
	hLayout.FixedPosition.iX = 10;
	hLayout.FixedPosition.iY = 60;

	hLayout.UseScalingSize = false;
	hLayout.FixedSize.iX = 100;
	hLayout.FixedSize.iY = 20;

	return &hLayout;
}

/////////////////////////////////////////////////////////////////////////////////
// MyComboBoxModel implementation
MyComboBoxModel::MyComboBoxModel( MyApplication * pApplication ):
	WinGUIComboBoxModel(RESID_COMBOBOX_TEST)
{
	m_pApplication = pApplication;

	StringFn->Copy( m_arrLabels[0], TEXT("Some") );
	StringFn->Copy( m_arrData[0], TEXT("Some Data") );
	StringFn->Copy( m_arrLabels[1], TEXT("Another") );
	StringFn->Copy( m_arrData[1], TEXT("Another Data") );
	StringFn->Copy( m_arrLabels[2], TEXT("This") );
	StringFn->Copy( m_arrData[2], TEXT("This Data") );
	StringFn->Copy( m_arrLabels[3], TEXT("That") );
	StringFn->Copy( m_arrData[3], TEXT("That Data") );

	m_hCreationParameters.iItemCallBackMode = WINGUI_COMBOBOX_ITEMCALLBACK_LABELS;

	m_hCreationParameters.iType = WINGUI_COMBOBOX_BUTTON;
	m_hCreationParameters.iCase = WINGUI_COMBOBOX_CASE_BOTH;
	m_hCreationParameters.iInitialSelectedItem = 0;
	m_hCreationParameters.bAllowHorizontalScroll = false;
	m_hCreationParameters.bItemTextEllipsis = true;
	m_hCreationParameters.bCaseSensitiveSearch = false;
	m_hCreationParameters.bAutoSort = false;
	m_hCreationParameters.bEnableTabStop = true;
}
MyComboBoxModel::~MyComboBoxModel()
{
	// nothing to do
}

Void MyComboBoxModel::Initialize()
{
	WinGUIComboBox * pController = (WinGUIComboBox*)m_pController;

	pController->AddItem( 0 );
	pController->SetItemData( 0, m_arrData[0] );
	pController->AddItem( 1 );
	pController->SetItemData( 1, m_arrData[1] );
	pController->AddItem( 2 );
	pController->SetItemData( 2, m_arrData[2] );
	pController->AddItem( 3 );
	pController->SetItemData( 3, m_arrData[3] );

	pController->SelectItem( 0 );
}

const WinGUILayout * MyComboBoxModel::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	hLayout.UseScalingPosition = false;
	hLayout.FixedPosition.iX = 10;
	hLayout.FixedPosition.iY = 90;

	hLayout.UseScalingSize = false;
	hLayout.FixedSize.iX = 100;
	hLayout.FixedSize.iY = 100;

	return &hLayout;
}

Bool MyComboBoxModel::OnSelectionOK()
{
	UInt iSelectedItem = ((WinGUIComboBox*)m_pController)->GetSelectedItem();
	Void * pData = ((WinGUIComboBox*)m_pController)->GetItemData( iSelectedItem );
	((WinGUIStatic*)(m_pApplication->m_hStaticTextModel.GetController()))->SetText((const GChar *)pData);
	return false;
}

Void MyComboBoxModel::OnRequestItemLabel( GChar * outBuffer, UInt iMaxLength, UInt iItemIndex, Void * pItemData )
{
	DebugAssert( iItemIndex < 4 );

	StringFn->NCopy( outBuffer, m_arrLabels[iItemIndex], iMaxLength - 1 );
}

/////////////////////////////////////////////////////////////////////////////////
// MyStaticRectModel implementation
MyStaticRectModel::MyStaticRectModel( MyApplication * pApplication ):
	WinGUIStaticModel(RESID_STATIC_RECT_TEST)
{
	m_pApplication = pApplication;

	m_hCreationParameters.iType = WINGUI_STATIC_FRAME;
	m_hCreationParameters.bAddSunkenBorder = false;
	m_hCreationParameters.hFrame.iFrameType = WINGUI_STATIC_FRAME_ETCHED;
}
MyStaticRectModel::~MyStaticRectModel()
{
	// nothing to do
}

const WinGUILayout * MyStaticRectModel::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	hLayout.UseScalingPosition = false;
	hLayout.FixedPosition.iX = 10;
	hLayout.FixedPosition.iY = 200;

	hLayout.UseScalingSize = false;
	hLayout.FixedSize.iX = 100;
	hLayout.FixedSize.iY = 40;

	return &hLayout;
}

/////////////////////////////////////////////////////////////////////////////////
// MyProgressBarModel implementation
MyProgressBarModel::MyProgressBarModel( MyApplication * pApplication ):
	WinGUIProgressBarModel(RESID_PROGRESSBAR_TEST)
{
	m_pApplication = pApplication;

	m_hCreationParameters.bPendingMode = false;
	m_hCreationParameters.bSmoothWrap = true;
	m_hCreationParameters.bVertical = false;
}
MyProgressBarModel::~MyProgressBarModel()
{
	// nothing to do
}

const WinGUILayout * MyProgressBarModel::GetLayout() const
{
	static WinGUIManualLayout hLayout;

	hLayout.UseScalingPosition = false;
	hLayout.FixedPosition.iX = 10;
	hLayout.FixedPosition.iY = 300;

	hLayout.UseScalingSize = false;
	hLayout.FixedSize.iX = 200;
	hLayout.FixedSize.iY = 40;

	return &hLayout;
}

/////////////////////////////////////////////////////////////////////////////////
// MyStatusBarModel implementation
MyStatusBarModel::MyStatusBarModel( MyApplication * pApplication ):
	WinGUIStatusBarModel(RESID_STATUSBAR_TEST)
{
	m_pApplication = pApplication;

	m_hCreationParameters.bHasSizingGrip = false;
	m_hCreationParameters.bEnableToolTips = true;
}
MyStatusBarModel::~MyStatusBarModel()
{
	// nothing to do
}

const WinGUILayout * MyStatusBarModel::GetLayout() const
{
	// No layout
	return NULL;
}

/////////////////////////////////////////////////////////////////////////////////
// MyApplication implementation
MyApplication::MyApplication():
	m_hAppWindowModel(this),

	m_hTabsModel(this),

	m_hContainerModelLeft(this),

	m_hButtonModel(this),
	m_hGroupBoxModel(this),
	m_hRadioButtonModelA(this),
	m_hRadioButtonModelB(this),
	m_hRadioButtonGroup(),
	m_hStaticTextModel(this),

	m_hContainerModelCenter(this),

	m_hTableModel(this),
	m_hProgressBarModel(this),

	m_hContainerModelRight(this),

	m_hCheckBoxModel(this),
	m_hTextEditModel(this),
	m_hComboBoxModel(this),
	m_hStaticRectModel(this),

	m_hStatusBarModel(this)
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
	WinGUIFn->CreateGroupBox( pContainerLeft, &m_hGroupBoxModel );

	// A couple Radio Buttons
	m_hRadioButtonGroup.AddButton( &m_hRadioButtonModelA );
	m_hRadioButtonGroup.AddButton( &m_hRadioButtonModelB );
	m_hRadioButtonModelA.SetGroup( &m_hRadioButtonGroup );
	m_hRadioButtonModelB.SetGroup( &m_hRadioButtonGroup );
	WinGUIFn->CreateRadioButton( pContainerLeft, &m_hRadioButtonModelA );
	WinGUIFn->CreateRadioButton( pContainerLeft, &m_hRadioButtonModelB );
	( (WinGUIRadioButton*)(m_hRadioButtonModelA.GetController()) )->Check();

	// A Static Text
	WinGUIFn->CreateStatic( pContainerLeft, &m_hStaticTextModel );

	// Center Container
	WinGUIContainer * pContainerCenter = WinGUIFn->CreateContainer( pAppWindow, &m_hContainerModelCenter );

	// A Table
	WinGUITable * pTable = WinGUIFn->CreateTable( pContainerCenter, &m_hTableModel );
	m_hTableModel.Initialize();

	// Right Container
	WinGUIContainer * pContainerRight = WinGUIFn->CreateContainer( pAppWindow, &m_hContainerModelRight );

	// A CheckBox
	WinGUICheckBox * pCheckBox = WinGUIFn->CreateCheckBox( pContainerRight, &m_hCheckBoxModel );
	pCheckBox->Check();

	// A TextEdit
	WinGUITextEdit * pTextEdit = WinGUIFn->CreateTextEdit( pContainerRight, &m_hTextEditModel );
	pTextEdit->SetCueText( TEXT("Type Stuff"), false );
	pTextEdit->SetTextLimit( 32 );

	// A ComboBox
	WinGUIComboBox * pComboBox = WinGUIFn->CreateComboBox( pContainerRight, &m_hComboBoxModel );
	m_hComboBoxModel.Initialize();

	// A Static Rect
	WinGUIFn->CreateStatic( pContainerRight, &m_hStaticRectModel );

	// A ProgressBar
	WinGUIProgressBar * pProgressBar = WinGUIFn->CreateProgressBar( pContainerRight, &m_hProgressBarModel );
	pProgressBar->SetRange( 0, 100 );
	pProgressBar->SetState( WINGUI_PROGRESSBAR_INPROGRESS );

	// A StatusBar
	WinGUIStatusBar * pStatusBar = WinGUIFn->CreateStatusBar( pAppWindow, &m_hStatusBarModel );
	UInt arrEdges[2] = { 20, INVALID_OFFSET };
	pStatusBar->SetMinHeight( 32 );
	pStatusBar->SetParts( arrEdges, 2 );
	pStatusBar->SetPartText( 0, TEXT("Ggngngngn !"), WINGUI_STATUSBAR_DRAW_SINKBORDER );
	pStatusBar->SetPartTipText( 0, TEXT("Nothing Here !") );
	pStatusBar->SetPartText( 1, TEXT("AAhhhaahh !"), WINGUI_STATUSBAR_DRAW_SINKBORDER );
	pStatusBar->SetPartTipText( 1, TEXT("Not much more Here !") );

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
Void MyIdleTime( Void * pUserData )
{
	static TimeMeasure iLastTime = 0;
	static Int iDelta = +1;

	if ( iLastTime == 0 ) {
		iLastTime = SystemFn->TimeAbsolute( TIMEUNIT_MILLISECONDS );
		return;
	}

	MyApplication * pApp = (MyApplication*)pUserData;
	WinGUIProgressBar * pProgressBar = (WinGUIProgressBar*)( pApp->m_hProgressBarModel.GetController() );

	if ( pProgressBar->GetBarPosition() >= 100 )
		iDelta = -1;
	else if ( pProgressBar->GetBarPosition() <= 0 )
		iDelta = +1;

	TimeMeasure iCurrentTime = SystemFn->TimeAbsolute( TIMEUNIT_MILLISECONDS );
	TimeMeasure iElapsedTime = iCurrentTime - iLastTime;
	if ( iElapsedTime > 100.0 ) {
		pProgressBar->Progress( iDelta );
		iLastTime = iCurrentTime;
	}
}

int APIENTRY wWinMain( _In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow )
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    MyApplication hApplication;

    return WinGUIFn->MessageLoop( MyIdleTime, &hApplication );
}

