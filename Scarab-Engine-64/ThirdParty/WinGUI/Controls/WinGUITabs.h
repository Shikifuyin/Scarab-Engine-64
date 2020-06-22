/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUITabs.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : Tabs
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : IMPORTANT INFO !!!!
// The Parent Window MUST have ClipChildren DISABLED !
// The Parent Window SHOULD have ClipSibblings ENABLED !
// TabPane Containers SHOULD have ClipSibblings ENABLED !
// TabPane Containers CAN have ClipChildren ENABLED !
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITABS_H
#define SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITABS_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../WinGUIControl.h"
#include "../WinGUIContainer.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define WINGUI_TABS_MAX 64 // Should be more than enough !

// Creation Parameters
typedef struct _wingui_tabs_parameters {
	Bool bSingleLine;
	Bool bFixedWidth;
	UInt iTabCount;
	struct _tab {
		GChar strLabel[64];
		Void * pUserData;
	} arrTabs[WINGUI_TABS_MAX];
} WinGUITabsParameters;

// Prototypes
class WinGUITabsModel;
class WinGUITabs;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUITabsModel class
class WinGUITabsModel : public WinGUIControlModel
{
public:
	WinGUITabsModel( Int iResourceID );
	virtual ~WinGUITabsModel();

	// Creation Parameters
	inline WinGUITabsParameters * GetCreationParameters();

	// Events
	virtual Bool OnTabSelect() = 0; // Must-Implement, Use WinGUITabs::SwitchSelectedTabPane here

protected:
	WinGUITabsParameters m_hCreationParameters;
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUITabs class
class WinGUITabs : public WinGUIControl
{
public:
	WinGUITabs( WinGUIElement * pParent, WinGUITabsModel * pModel );
	virtual ~WinGUITabs();

	// Display Area
	Void GetDisplayArea( WinGUIRectangle * outDisplayArea ) const;

	// TabPane Switching
	inline WinGUIContainer * GetSelectedTabPane() const;
	Void SwitchSelectedTabPane( WinGUIContainer * pSelectedTabPane );

	// Tabs
	UInt GetTabsRowCount() const;

	Void SetMinTabWidth( UInt iWidth = INVALID_OFFSET );
	Void SetTabButtonPadding( UInt iHPadding, UInt iVPadding );

	Void AddTab( UInt iIndex, GChar * strLabel, Void * pUserData );
	Void RemoveTab( UInt iIndex );
	Void RemoveAllTabs();

	Void UpdateTab( UInt iIndex, GChar * strLabel, Void * pUserData );

	// Selection
	UInt GetSelectedTab() const;
	Void SelectTab( UInt iIndex );
	Void UnselectAll( Bool bKeepCurrent );

	// ToolTips
	//WinGUIToolTip * GetToolTip() const;
	//Void SetToolTip( WinGUIToolTip * pToolTip );

private:
	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode );

	// Currently Selected Tab Pane
	WinGUIContainer * m_pSelectedTabPane;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUITabs.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITABS_H

