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
// The Tab Containers SHOULD have ClipSibblings ENABLED !
// The Tab Containers CAN have ClipChildren ENABLED !
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

/////////////////////////////////////////////////////////////////////////////////
// The WinGUITabsModel class
class WinGUITabsModel : public WinGUIControlModel
{
public:
	WinGUITabsModel( Int iResourceID );
	virtual ~WinGUITabsModel();

	// Events
	virtual Bool OnTabSelect() = 0;

	// View
	virtual UInt GetTabCount() const = 0;
	virtual GChar * GetTabLabel( UInt iTabIndex ) const = 0;
	virtual Void * GetTabUserData( UInt iTabIndex ) const = 0;

protected:

};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUITabs class
class WinGUITabs : public WinGUIControl
{
public:
	WinGUITabs( WinGUIElement * pParent, WinGUITabsModel * pModel );
	virtual ~WinGUITabs();

	// Display Area access
	Void GetDisplayArea( WinGUIRectangle * outDisplayArea ) const;

	// TabPane access
	inline WinGUIContainer * GetSelectedTabPane() const;
	Void SwitchSelectedTabPane( WinGUIContainer * pSelectedTabPane );

	// Tabs access
	UInt GetTabsRowCount() const;

	Void SetMinTabWidth( UInt iWidth = INVALID_OFFSET );
	Void SetTabButtonPadding( UInt iHPadding, UInt iVPadding );

	Void AddTab( UInt iIndex, GChar * strLabel, Void * pUserData );
	Void RemoveTab( UInt iIndex );
	Void RemoveAllTabs();

	Void UpdateTab( UInt iIndex, GChar * strLabel, Void * pUserData );

	// Selection access
	UInt GetSelectedTab() const;
	Void SelectTab( UInt iIndex );
	Void UnselectAll( Bool bKeepCurrent );

	// ToolTips access
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

