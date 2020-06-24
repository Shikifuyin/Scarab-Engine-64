/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIComboBox.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : ComboBox
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUICOMBOBOX_H
#define SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUICOMBOBOX_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../WinGUIControl.h"

#include "WinGUIStatic.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// ComboBox Properties
enum WinGUIComboBoxType {
	WINGUI_COMBOBOX_BUTTON = 0, // Button, Drop-Down List
	WINGUI_COMBOBOX_EDIT,		// TextEdit, Drop-Down List
	WINGUI_COMBOBOX_LIST		// TextEdit, ListBox
};
enum WinGUIComboBoxCase {
	WINGUI_COMBOBOX_CASE_BOTH = 0,
	WINGUI_COMBOBOX_CASE_LOWER,
	WINGUI_COMBOBOX_CASE_UPPER
};

// Creation Parameters
typedef struct _wingui_combobox_parameters {
	WinGUIComboBoxType iType;
	WinGUIComboBoxCase iCase;
	UInt iInitialSelectedItem;
	Bool bAllowHorizontalScroll;
	Bool bAutoSort;
	Bool bEnableTabStop;
} WinGUIComboBoxParameters;

// Prototypes
class WinGUIComboBoxModel;
class WinGUIComboBox;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIComboBoxModel class
class WinGUIComboBoxModel : public WinGUIControlModel
{
public:
	WinGUIComboBoxModel( Int iResourceID );
	virtual ~WinGUIComboBoxModel();

	// Creation Parameters
	inline const WinGUIComboBoxParameters * GetCreationParameters() const;

	// Content Data (Must-Implement)
	virtual UInt GetItemCount() const = 0;
	virtual const GChar * GetItemString( UInt iIndex ) const = 0;
	virtual Void * GetItemData( UInt iIndex ) const = 0;

	// Events
	virtual Bool OnFocusGained() { return false; }
	virtual Bool OnFocusLost() { return false; }

	virtual Bool OnDblClick() { return false; }

	virtual Bool OnTextChange() { return false; }
	virtual Bool OnSelectionChange() { return false; }
	virtual Bool OnSelectionOK() { return false; }
	virtual Bool OnSelectionCancel() { return false; }

protected:
	WinGUIComboBoxParameters m_hCreationParameters;
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIComboBox class
class WinGUIComboBox : public WinGUIControl
{
public:
	WinGUIComboBox( WinGUIElement * pParent, WinGUIComboBoxModel * pModel );
	virtual ~WinGUIComboBox();

	// Enable / Disable
	Void Enable();
	Void Disable();

	// Display Properties
	UInt GetMinVisibleItems() const;
	Void SetMinVisibleItems( UInt iCount );

	UInt GetSelectionHeight() const;
	Void SetSelectionHeight( UInt iHeight );
	UInt GetListItemHeight() const;
	Void SetListItemHeight( UInt iHeight );

	Void SetTextLimit( UInt iMaxLength );

	// List
	UInt GetItemCount() const;
	UInt GetItemStringLength( UInt iIndex ) const;
	Void GetItemString( UInt iIndex, GChar * outBuffer ) const; // DANGER : Buffer must be large enough !

	UInt SearchItem( const GChar * strItem, UInt iStartIndex, Bool bExact ) const;

	UInt AddItem( const GChar * strItem );
	Void AddItem( UInt iIndex, const GChar * strItem );
	Void RemoveItem( UInt iIndex );
	Void RemoveAllItems();

	// Item Data
	Void * GetItemData( UInt iIndex ) const;
	Void SetItemData( UInt iIndex, Void * pUserData );

	// Selection
	UInt GetSelectedItem() const;
	UInt GetSelectedItemStringLength() const;
	Void GetSelectedItemString( GChar * outBuffer, UInt iMaxLength ) const;

	Void SelectItem( UInt iIndex );
	UInt SelectItem( const GChar * strItem, UInt iStartIndex );

	Void SetSelectionText( const GChar * strText ); // Invalid for WINGUI_COMBOBOX_BUTTON type

	// Text Cues (Invalid for WINGUI_COMBOBOX_BUTTON type)
	Void GetCueText( GChar * outText, UInt iMaxLength ) const;
	Void SetCueText( const GChar * strText );

	// Directory Listing, Paths can be a directory or a filename with wildcard chars
	UInt AddFiles( GChar * strPath, Bool bIncludeSubDirs );
	Void MakeDirectoryList( GChar * strPath, Bool bIncludeSubDirs, WinGUIStatic * pDisplay = NULL );

private:
	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode, Void * pParameters );
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIComboBox.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUICOMBOBOX_H

