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

#include "../Tools/WinGUIImageList.h"

#include "WinGUIStatic.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// CallBack Modes
enum WinGUIComboBoxItemCallBackFlags {
	WINGUI_COMBOBOX_ITEMCALLBACK_LABELS      = 0x01,
	WINGUI_COMBOBOX_ITEMCALLBACK_IMAGES      = 0x02,
	WINGUI_COMBOBOX_ITEMCALLBACK_OVERLAY     = 0x04,
	WINGUI_COMBOBOX_ITEMCALLBACK_INDENTATION = 0x08,
	WINGUI_COMBOBOX_ITEMCALLBACK_ALL         = 0x0f
};
typedef UInt WinGUIComboBoxItemCallBackMode;

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
	WinGUIComboBoxItemCallBackMode iItemCallBackMode;

	WinGUIComboBoxType iType;
	WinGUIComboBoxCase iCase;
	UInt iInitialSelectedItem;
	Bool bAllowHorizontalScroll;
	Bool bItemTextEllipsis;
	Bool bCaseSensitiveSearch;
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

	// Events
	virtual Bool OnFocusGained() { return false; }
	virtual Bool OnFocusLost() { return false; }

	virtual Bool OnDblClick() { return false; }

	virtual Bool OnExpand() { return false; }
	virtual Bool OnCollapse() { return false; }

	virtual Bool OnSelectionChange() { return false; }
	virtual Bool OnSelectionOK() { return false; }
	virtual Bool OnSelectionCancel() { return false; }

	virtual Bool OnItemAdded( UInt iItemIndex, const GChar * strLabel, Void * pItemData ) { return false; } // Must return false
	virtual Bool OnItemRemoved( UInt iItemIndex, const GChar * strLabel, Void * pItemData ) { return false; } // Must return false

	virtual Bool OnEditStart()  { return false; } // Must return false
	virtual Bool OnEditEnd( const GChar * strEditedText, Bool bTextEditChanged, UInt iSelectedItem )    { return false; } // Return false to allow modification
	virtual Bool OnEditCancel( const GChar * strEditedText, Bool bTextEditChanged, UInt iSelectedItem ) { return false; } // Return false to allow modification

	// Item Callback Events (Must-Implement when using corresponding Callbacks)
	virtual GChar * OnRequestItemLabel( UInt iItemIndex, Void * pItemData ) { return NULL; }
	virtual UInt OnRequestItemImage( UInt iItemIndex, Void * pItemData ) { return INVALID_OFFSET; }
	virtual UInt OnRequestItemImageSelected( UInt iItemIndex, Void * pItemData ) { return INVALID_OFFSET; }
	virtual UInt OnRequestItemOverlayImage( UInt iItemIndex, Void * pItemData ) { return INVALID_OFFSET; }
	virtual UInt OnRequestItemIndentation( UInt iItemIndex, Void * pItemData ) { return 0; }

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

	// Text format
	Bool IsUnicode() const;
	Bool IsANSI() const;

	Void SetUnicode();
	Void SetANSI();

	// Enable / Disable
	Void Enable();
	Void Disable();

	// Expand / Collapse
	Void Expand();
	Void Collapse();

	// Visual Settings
	Void GetImageList( WinGUIImageList * outImageList ) const;
	Void SetImageList( const WinGUIImageList * pImageList );   // Caller retains ownership

	// Metrics
	UInt GetBoxItemHeight() const;
	Void SetBoxItemHeight( UInt iHeight );

	UInt GetListItemHeight() const;
	Void SetListItemHeight( UInt iHeight );

	UInt GetMinVisibleItems() const;
	Void SetMinVisibleItems( UInt iCount );

	Void SetTextLimit( UInt iMaxLength );

	// Item Operations
	UInt GetItemCount() const;

	Void AddItem( UInt iIndex );
	Void RemoveItem( UInt iIndex );
	Void RemoveAllItems();

	Void GetItemLabel( GChar * outLabelText, UInt iMaxLength, UInt iIndex ) const;
	Void SetItemLabel( UInt iIndex, GChar * strLabelText );

	UInt GetItemImage( UInt iIndex ) const;
	Void SetItemImage( UInt iIndex, UInt iImageIndex );

	UInt GetItemImageSelected( UInt iIndex ) const;
	Void SetItemImageSelected( UInt iIndex, UInt iImageIndex );

	Void * GetItemData( UInt iIndex ) const;
	Void SetItemData( UInt iIndex, Void * pData );

	UInt GetItemOverlayImage( UInt iIndex ) const;
	Void SetItemOverlayImage( UInt iIndex, UInt iOverlayImage );

	UInt GetItemIndentation( UInt iIndex ) const;
	Void SetItemIndentation( UInt iIndex, UInt iIndentation );

	// Selection
	UInt GetSelectedItem() const;
	Void SelectItem( UInt iIndex );

	// Search
	UInt SearchItem( const GChar * strItem, UInt iStartIndex, Bool bExact ) const;

	// Text Cues (Invalid for WINGUI_COMBOBOX_BUTTON type)
	Void GetCueText( GChar * outText, UInt iMaxLength ) const;
	Void SetCueText( const GChar * strText );

private:
	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode, Void * pParameters );

	// State
	WinGUIComboBoxItemCallBackMode m_iItemCallBackMode;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIComboBox.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUICOMBOBOX_H

