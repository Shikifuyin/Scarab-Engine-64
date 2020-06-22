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

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIComboBoxModel class
class WinGUIComboBoxModel : public WinGUIControlModel
{
public:
	WinGUIComboBoxModel( Int iResourceID );
	virtual ~WinGUIComboBoxModel();

	// Events
	virtual Bool OnDblClick() = 0;

	virtual Bool OnTextChange() = 0;
	virtual Bool OnSelectionChange() = 0;
	virtual Bool OnSelectionOK() = 0;
	virtual Bool OnSelectionCancel() = 0;

	// View
	virtual const WinGUIRectangle * GetRectangle() const = 0;

	virtual WinGUIComboBoxType GetType() = 0;

	virtual Bool AllowHorizScroll() const = 0;
	virtual Bool AutoSort() const = 0;

	virtual WinGUIComboBoxCase GetTextCase() const = 0;

	virtual UInt GetItemCount() const = 0;
	virtual const GChar * GetItemString( UInt iIndex ) const = 0;
	virtual Void * GetItemData( UInt iIndex ) const = 0;

	virtual UInt GetInitialSelectedItem() const = 0;

protected:

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
	UInt GetListItemHeight() const;
	Void SetSelectionHeight( UInt iHeight );
	Void SetListItemHeight( UInt iHeight );

	Void SetTextLimit( UInt iMaxLength );

	// List access
	UInt GetItemCount() const;
	UInt GetItemStringLength( UInt iIndex ) const;
	Void GetItemString( UInt iIndex, GChar * outBuffer ) const; // DANGER : Buffer must be large enough !

	UInt SearchItem( const GChar * strItem, UInt iStartIndex, Bool bExact ) const;

	UInt AddItem( const GChar * strItem );
	Void AddItem( UInt iIndex, const GChar * strItem );
	Void RemoveItem( UInt iIndex );
	Void RemoveAllItems();

	// Item Data access
	Void * GetItemData( UInt iIndex ) const;
	Void SetItemData( UInt iIndex, Void * pUserData );

	// Selection access
	UInt GetSelectedItem() const;
	UInt GetSelectedItemStringLength() const;
	Void GetSelectedItemString( GChar * outBuffer, UInt iMaxLength ) const;

	Void SelectItem( UInt iIndex );
	UInt SelectItem( const GChar * strItem, UInt iStartIndex );

	Void SetSelectionText( const GChar * strText ); // Invalid for WINGUI_COMBOBOX_BUTTON type

	// Directory Listing, Paths can be a directory or a filename with wildcard chars
	UInt AddFiles( GChar * strPath, Bool bIncludeSubDirs );
	Void MakeDirectoryList( GChar * strPath, Bool bIncludeSubDirs, WinGUIStatic * pDisplay = NULL );

	// Text Cues (Invalid for WINGUI_COMBOBOX_BUTTON type)
	Void GetCueText( GChar * outText, UInt iMaxLength ) const;
	Void SetCueText( const GChar * strText );

private:
	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode );
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIComboBox.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUICOMBOBOX_H

