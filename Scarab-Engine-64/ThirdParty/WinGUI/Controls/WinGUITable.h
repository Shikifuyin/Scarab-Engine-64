/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUITable.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : Table (ListView)
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
#ifndef SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITABLE_H
#define SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITABLE_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../WinGUIControl.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// View Modes
enum WinGUITableViewMode {
	WINGUI_TABLE_VIEW_LIST = 0,
	WINGUI_TABLE_VIEW_ICONS,
	WINGUI_TABLE_VIEW_ICONS_SMALL,
	WINGUI_TABLE_VIEW_DETAILED,
	WINGUI_TABLE_VIEW_TILES
};

// Text Alignment
enum WinGUITableTextAlign {
	WINGUI_TABLE_TEXT_ALIGN_LEFT = 0,
	WINGUI_TABLE_TEXT_ALIGN_RIGHT,
	WINGUI_TABLE_TEXT_ALIGN_CENTER
};

// Icons Alignment
enum WinGUITableIconsAlign {
	WINGUI_TABLE_ICONS_ALIGN_TOP = 0,
	WINGUI_TABLE_ICONS_ALIGN_LEFT
};

// Creation Parameters
typedef struct _wingui_table_parameters {
	Bool bMakeVirtualTable; // When managing large amount of data
	Bool bHeadersInAllViews;
	WinGUITableViewMode iViewMode;
	
	Bool bStaticColumnHeaders; // Only when bHeadersInAllViews = true
	Bool bSnapColumnsWidth;    //
	Bool bAutoSizeColumns;     // Otherwise, Detailed View only

	Bool bEditableLabels;

	Bool bSingleItemSelection;
	Bool bAlwaysShowSelection;
	Bool bBorderSelection;

	Bool bSortAscending;  // Those cannot be used
	Bool bSortDescending; // with virtual tables

	Bool bAddCheckBoxes;
	Bool bAutoCheckOnSelect;

	Bool bHandleInfoTips;

	Bool bHotTrackingSingleClick;
	Bool bHotTrackingDoubleClick;
	Bool bHotTrackSelection; // Requires bHotTrackingSingleClick or bHotTrackingDoubleClick
	Bool bUnderlineHot;      // Requires bHotTrackingSingleClick or bHotTrackingDoubleClick
	Bool bUnderlineCold;     // Requires bHotTrackingDoubleClick

	Bool bSharedImageList;
	Bool bUseBackBuffer; // Reduces Flickering
	Bool bTransparentBackground;
	Bool bTransparentShadowText;

	union {
		// Nothing Specific to List Mode

		struct _iconsmode {
			WinGUITableIconsAlign iAlign;
			Bool bAutoArrange;
			Bool bHideLabels;
			Bool bNoLabelWrap;
			Bool bColumnOverflow; // Only when bHeadersInAllViews = true
			Bool bJustifiedColumns;
			Bool bSnapToGrid;
			Bool bSimpleSelection;
		} hIconsMode;

		struct _smalliconsmode {
			WinGUITableIconsAlign iAlign;
			Bool bAutoArrange;
			Bool bHideLabels;
			Bool bNoLabelWrap;
			Bool bColumnOverflow; // Only when bHeadersInAllViews = true
			Bool bJustifiedColumns;
			Bool bSnapToGrid;
			Bool bSimpleSelection;
		} hSmallIconsMode;

		struct _detailedmode {
			Bool bNoColumnHeaders;
			Bool bHeaderDragNDrop;
			Bool bFullRowSelection;
			Bool bShowGridLines;
			Bool bSubItemImages;
		} hDetailedMode;

		struct _tilesmode {
			Bool bColumnOverflow; // Only when bHeadersInAllViews = true
		} hTilesMode;
	};
} WinGUITableParameters;

// Column Infos
typedef struct _wingui_table_column_infos {
	UInt iOrderIndex;   // Left to Right Column Order
	UInt iSubItemIndex; // Assigned Sub Item Index

	GChar strHeaderText[64];
	WinGUITableTextAlign iRowsTextAlign;

	Bool bHeaderSplitButton;

	Bool bHeaderHasImage;
	Bool bRowsHaveImages;
	Bool bIsImageOnRight;
	UInt iImageListIndex;

	Bool bFixedWidth;
	Bool bFixedAspectRatio;
	UInt iWidth;
	UInt iMinWidth;
	UInt iDefaultWidth;
	UInt iIdealWidth;
} WinGUITableColumnInfos;

// Prototypes
class WinGUIButtonModel;
class WinGUIButton;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUITableModel class
class WinGUITableModel : public WinGUIControlModel
{
public:
	WinGUITableModel( Int iResourceID );
	virtual ~WinGUITableModel();

	// Creation Parameters
	inline const WinGUITableParameters * GetCreationParameters() const;

	// Content Data (Must-Implement)
	virtual UInt GetItemCount() const = 0;

	// Events

protected:
	WinGUITableParameters m_hCreationParameters;
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUITable class
class WinGUITable : public WinGUIControl
{
public:
	WinGUITable( WinGUIElement * pParent, WinGUITableModel * pModel );
	virtual ~WinGUITable();

	// Virtual Tables
	inline Bool IsVirtual() const;

	// View Modes
	inline WinGUITableViewMode GetViewMode() const;
	Void SwitchViewMode( WinGUITableViewMode iViewMode );

	// Display Properties
	Void AdjustRequiredDimensions( UInt * pWidth, UInt * pHeight, UInt iItemCount ) const; // Actually an approximation !

	UInt GetVisibleItemCount() const;

	Void GetEmptyText( GChar * outText, UInt iMaxLength ) const;

	//UInt GetBackgroundColor() const;
	// ListView_GetBkImage

	// View Operations
	Void MakeItemVisible( UInt iIndex, Bool bAllowPartial );

	// Columns Operations
	Void GetColumnInfos( UInt iIndex, WinGUITableColumnInfos * outInfos ) const;
	Void GetColumnOrder( UInt * outOrderedIndices, UInt iCount ) const;
	UInt GetColumnWidth( UInt iIndex ) const;

	Void RemoveColumn( UInt iIndex );

	// List Operations
	Void RemoveItem( UInt iIndex );
	Void RemoveAllItems();

	// Group Operations
	Void EnableGroups( Bool bEnable );

	UInt GetGroupCount() const;
	UInt GetFocusedGroup() const;

	// Item Operations
	Bool IsItemChecked( UInt iIndex ) const; // Only when using checkboxes

	Void * EditItemLabelStart( UInt iIndex ); // Table must have focus for this !
	//ListView_GetEditControl
	Void EditItemLabelCancel();

	// Footer Operations
	// Currently unsupported by Windows !

	// Search
	UInt SearchItem( const GChar * strLabel, Bool bExact, UInt iStartIndex, Bool bWrapAround ) const;
	UInt SearchItem( Void * pData, UInt iStartIndex, Bool bWrapAround ) const;
	UInt SearchItem( const WinGUIPoint * pPoint, KeyCode iDirection, UInt iStartIndex, Bool bWrapAround ) const; // Only in Icon views

	// Arrangement / Sorting
	//ListView_Arrange()

	//ListView_GetCallbackMask

private:
	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode, Void * pParameters );

	// State
	Bool m_bVirtualTable;
	WinGUITableViewMode m_iViewMode;
	Bool m_bHasCheckBoxes;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUITable.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITABLE_H

