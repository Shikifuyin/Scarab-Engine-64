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

#include "WinGUITextEdit.h"
#include "../Tools/WinGUIImageList.h"

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
	WINGUI_TABLE_ICONS_ALIGN_DEFAULT = 0,
	WINGUI_TABLE_ICONS_ALIGN_TOP,
	WINGUI_TABLE_ICONS_ALIGN_LEFT
};

// Tiles Sizing
enum WinGUITableTileSize {
	WINGUI_TABLE_TILES_AUTOSIZE = 0,
	WINGUI_TABLE_TILES_FIXED_WIDTH,
	WINGUI_TABLE_TILES_FIXED_HEIGHT,
	WINGUI_TABLE_TILES_FIXED_SIZE
};

// Creation Parameters
typedef struct _wingui_table_parameters {
	Bool bVirtualTable;

	Bool bHasBackBuffer;
	Bool bHasSharedImageLists;

	WinGUITableViewMode iViewMode;
	Bool bGroupMode;
	Bool bHasHeadersInAllViews;

	Bool bHasColumnHeaders;
	Bool bHasStaticColumnHeaders;
	Bool bHasDraggableColumnHeaders;
	Bool bHasIconColumnOverflowButton;

	Bool bHasCheckBoxes;
	Bool bHasIconLabels;
	Bool bHasEditableLabels;
	Bool bHasSubItemImages;

	Bool bSingleItemSelection;
	Bool bIconSimpleSelection;

	Bool bAutoSortAscending;
	Bool bAutoSortDescending;

	Bool bHasHotTrackingSingleClick;
	Bool bHasHotTrackingDoubleClick;
	Bool bHasHotTrackingSelection;

	Bool bHasInfoTips;
} WinGUITableParameters;

// Column Infos
typedef struct _wingui_table_column_infos {
	UInt iOrderIndex;   // Left to Right Column Order
	UInt iSubItemIndex; // Assigned Sub Item Index

	mutable GChar strHeaderText[64];
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

// Group Infos
typedef struct _wingui_table_group_infos {
	UInt iGroupID;

	UInt iFirstItemIndex;
	UInt iItemCount;

	Bool bHasSubSets;

	Bool bHasHeader;
	mutable GChar strHeaderText[64];
	WinGUITableTextAlign iHeaderTextAlign;

	mutable GChar strFooterText[64];
	WinGUITableTextAlign iFooterTextAlign;

	mutable GChar strSubTitleText[64];
	mutable GChar strTaskLinkText[64];
	mutable GChar strTopDescriptionText[64];
	mutable GChar strBottomDescriptionText[64];

	mutable GChar strSubSetTitleText[64]; // Only when group is a subset

	UInt iTitleImageIndex;
	UInt iExtendedImageIndex;

	Bool bCanCollapse;
	Bool bCollapsed; // Else expanded

	Bool bHasFocus;
	Bool bSubSetHasFocus;
	Bool bIsSelected;

	Bool bHidden;
} WinGUITableGroupInfos;

// Item Infos
typedef struct _wingui_table_item_infos {
	Bool bIsSubItem; // Item or SubItem
	UInt iItemIndex;
	UInt iSubItemIndex;

	UInt iParentGroupID;

	mutable GChar strLabelText[64];
	Void * pUserData;

	UInt iColumnCount;
	struct _column {
		UInt iIndex;
		Bool bLineBreak;
		Bool bFill;
		Bool bAllowWrap;
		Bool bNoTitle;
	} arrColumns[32]; // Ordered

	UInt iIndentDepth; // Invalid for sub items

	UInt iIconImage;
	UInt iOverlayImage; // Range = [0;14]
	UInt iStateImage;   // Range = [0;14]

	Bool bHasFocus;
	Bool bSelected;

	Bool bCutMarked;     // Item is being cut
	Bool bDropHighlight; // Item is hovered during drag & drop
} WinGUITableItemInfos;

// Tile Infos
typedef struct _wingui_table_tile_infos {
	UInt iItemIndex;

	UInt iColumnCount;
	struct _column {
		UInt iIndex;
		Bool bLineBreak;
		Bool bFill;
		Bool bAllowWrap;
		Bool bNoTitle;
	} arrColumns[32]; // Ordered
} WinGUITableTileInfos;

// Tile Metrics
typedef struct _wingui_table_tile_metrics {
	WinGUITableTileSize iSizeMode;
	UInt iWidth;
	UInt iHeight;
	UInt iMaxItemLabelLines;
	WinGUIRectangle hItemLabelMargin;
} WinGUITableTileMetrics;

// HitTest Results
typedef struct _wingui_table_hittest_results {
	WinGUIPoint hPoint;
	UInt iGroupIndex;
	UInt iItemIndex;
	UInt iSubItemIndex;

	Bool bOutsideAbove;
	Bool bOutsideBelow;
	Bool bOutsideLeft;
	Bool bOutsideRight;

	Bool bInsideNoWhere; // Inside client area but not on an item

	Bool bOnItem;
	Bool bOnItemIcon;
	Bool bOnItemLabel;
	Bool bOnItemStateIcon;

	Bool bOnGroup;
	Bool bOnGroupHeader;
	Bool bOnGroupFooter;
	Bool bOnGroupBackground;
	Bool bOnGroupExpandCollapse;
	Bool bOnGroupStateIcon;
	Bool bOnGroupSubSetLink;

	Bool bOnFooter;
} WinGUITableHitTestResult;

// Search Options
enum WinGUITableSearchSpatialDirection {
	WINGUI_TABLE_SEARCH_SPATIAL_UP = 0,
	WINGUI_TABLE_SEARCH_SPATIAL_DOWN,
	WINGUI_TABLE_SEARCH_SPATIAL_LEFT,
	WINGUI_TABLE_SEARCH_SPATIAL_RIGHT
};

typedef struct _wingui_table_search_options {
	Bool bSpatialSearch;

	Bool bReverseSearch;                                 // Regular Search : Toward decreasing indices
	WinGUITableSearchSpatialDirection iSpatialDirection; // Spatial Search : UP, DOWN, LEFT, RIGHT

	Bool bSameGroup;
	Bool bVisible;
	Bool bHasFocus;
	Bool bSelected;
	Bool bCutMarked;
	Bool bDropHighlight;
} WinGUITableSearchOptions;

// Comparators
// Beware f***in Win32 has reverse convention from us : -1 when A#B ordered and +1 when B#A ...
typedef Int (__stdcall * WinGUITableGroupComparator)( Int iGroupIDA, Int iGroupIDB, Void * pUserData );
typedef Int (__stdcall * WinGUITableItemComparator)( Void * pItemDataA, Void * pItemDataB, Void * pUserData );

// Prototypes
class WinGUITableModel;
class WinGUITable;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUITableModel class
class WinGUITableModel : public WinGUIControlModel
{
public:
	WinGUITableModel( Int iResourceID );
	virtual ~WinGUITableModel();

	// Creation Parameters
	inline const WinGUITableParameters * GetCreationParameters() const;

	// Events
	virtual Bool OnFocusGained() { return false; }
	virtual Bool OnFocusLost() { return false; }

	virtual Bool OnColumnHeaderClick( UInt iIndex ) { return false; }

	virtual Bool OnAddItem( UInt iItemIndex ) { return false; }
	virtual Bool OnRemoveItem( UInt iItemIndex, Void * pItemData ) { return false; } // Do NOT add/remove/rearrange items while handling this event !
	virtual Bool OnRemoveAllItems() { return true; }                                 // Return false to receive subsequent OnRemoveItem events

	virtual Bool OnScrollStart( const WinGUIPoint & hScrollPos ) { return false; }
	virtual Bool OnScrollEnd( const WinGUIPoint & hScrollPos ) { return false; }

	virtual Bool OnDragLeftStart( UInt iItemIndex ) { return false; }
	virtual Bool OnDragRightStart( UInt iItemIndex ) { return false; }

	virtual Bool OnLabelEditStart() { return false; }                          // Return true to cancel edition, Use GetEditItemLabel / ReleaseEditItemLabel here
	virtual Bool OnLabelEditEnd( const GChar * strNewLabel ) { return true; }  // Return true to allow the modification
	virtual Bool OnLabelEditCancel() { return false; }

	

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

	// General Settings /////////////////////////////////////////////////////////
	inline Bool IsVirtual() const;

	Bool IsUnicode() const;
	Bool IsANSI() const;

	Void SetUnicode();
	Void SetANSI();

	Void SetAllocatedItemCount( UInt iPreAllocatedItemCount );

	inline Bool HasBackBuffer() const;
	Void UseBackBuffer( Bool bEnable );

	inline Bool HasSharedImageLists() const;
	Void UseSharedImageLists( Bool bEnable );

	Void ForceRedraw( UInt iFirstItem, UInt iLastItem, Bool bImmediate );

	//ListView_GetCallbackMask
	//ListView_SetCallbackMask

	// View Modes ///////////////////////////////////////////////////////////////
	inline WinGUITableViewMode GetViewMode() const;
	Void SwitchViewMode( WinGUITableViewMode iViewMode );

	inline Bool IsGroupModeEnabled() const;
	Void ToggleGroupMode( Bool bEnable );

	inline Bool HasHeadersInAllViews() const;
	Void ToggleHeadersInAllViews( Bool bEnable );

	// Options //////////////////////////////////////////////////////////////////
	inline Bool HasColumnHeaders() const;
	Void ToggleColumnHeaders( Bool bEnable ); // Detailed View only

	inline Bool HasStaticColumnHeaders() const;
	Void ToggleStaticColumnHeaders( Bool bEnable );

	inline Bool HasDraggableColumnHeaders() const;
	Void ToggleDraggableColumnHeaders( Bool bEnable );

	inline Bool HasIconColumnOverflowButton() const;
	Void ToggleIconColumnOverflowButton( Bool bEnable ); // Only when HeadersInAllViews in Icon/Tile View

	inline Bool HasCheckBoxes() const;
	Void ToggleCheckBoxes( Bool bEnable );
	Void ToggleAutoCheckOnSelect( Bool bEnable );

	inline Bool HasIconLabels() const;
	Void ToggleIconLabels( Bool bEnable );
	Void PreventIconLabelWrap( Bool bEnable );

	inline Bool HasEditableLabels() const;
	Void ToggleEditableLabels( Bool bEnable );

	inline Bool HasSubItemImages() const;
	Void ToggleSubItemImages( Bool bEnable );

	inline Bool HasSingleItemSelection() const;
	Void ToggleSingleItemSelection( Bool bEnable );

	inline Bool HasIconSimpleSelection() const;
	Void ToggleIconSimpleSelection( Bool bEnable );

	inline Bool IsAutoSorted() const;
	inline Bool IsAutoSortedAscending() const;
	inline Bool IsAutoSortedDescending() const;
	Void ToggleAutoSorting( Bool bEnable, Bool bAscendingElseDescending ); // Not with virtual tables

	inline Bool HasHotTracking() const;
	inline Bool IsHotTrackingSingleClick() const;
	inline Bool IsHotTrackingDoubleClick() const;
	inline Bool HasHotTrackingSelection() const;
	Void ToggleHotTracking( Bool bEnable, Bool bUseSingleClick );
	Void ToggleHotTrackingSelection( Bool bEnable );
	Void ToggleHotTrackingUnderline( Bool bEnable, Bool bUnderlineHotElseCold );

	inline Bool HasInfoTips() const;
	Void ToggleInfoTips( Bool bEnable );

	// Visual Settings //////////////////////////////////////////////////////////
	Void ToggleTransparentBackground( Bool bEnable );
	Void ShowGridLines( Bool bEnable ); // Detailed View only

	UInt GetBackgroundColor() const;
	Void SetBackgroundColor( UInt iColor );

	Void GetBackgroundImage( WinGUIBitmap * outImage, WinGUIPointF * outRelativePos, Bool * outIsTiled ) const; // Caller doesn't take ownership of the background image (considered shared) !
	Void SetBackgroundImage( const WinGUIBitmap * pImage, const WinGUIPointF & hRelativePos, Bool bUseTiling ); // Caller retains ownership of the background image !
	Void RemoveBackgroundImage();

	Void ToggleTransparentShadowText( Bool bEnable ); // Requires Transparent Background

	UInt GetTextBackgroundColor() const;
	Void SetTextBackgroundColor( UInt iColor );

	UInt GetTextColor() const;
	Void SetTextColor( UInt iColor );

	Void AutoSizeColumns( Bool bEnable );
	Void SnapColumnWidths( Bool bEnable );
	Void JustifyIconColumns( Bool bEnable );

	Void SetIconAlignment( WinGUITableIconsAlign iAlign );
	Void SnapIconsToGrid( Bool bEnable );
	Void AutoArrangeIcons( Bool bEnable );

	Void ToggleAlwaysShowSelection( Bool bEnable );
	Void ToggleFullRowSelection( Bool bEnable );

	Bool HasBorderSelection() const;
	Void ToggleBorderSelection( Bool bEnable );
	UInt GetBorderSelectionColor() const;
	Void SetBorderSelectionColor( UInt iColor );

	UInt GetInsertionMarkColor() const;
	Void SetInsertionMarkColor( UInt iColor );

	Void GetEmptyText( GChar * outText, UInt iMaxLength ) const;

	Void GetImageListIcons( WinGUIImageList * outImageList ) const;
	Void SetImageListIcons( const WinGUIImageList * pImageList );

	Void GetImageListSmallIcons( WinGUIImageList * outImageList ) const;
	Void SetImageListSmallIcons( const WinGUIImageList * pImageList );

	Void GetImageListGroupHeaders( WinGUIImageList * outImageList ) const;
	Void SetImageListGroupHeaders( const WinGUIImageList * pImageList );

	Void GetImageListStates( WinGUIImageList * outImageList ) const;
	Void SetImageListStates( const WinGUIImageList * pImageList );

	// Metrics //////////////////////////////////////////////////////////////////
	Void GetViewOrigin( WinGUIPoint * outOrigin ) const;
	Void GetViewRect( WinGUIRectangle * outRectangle ) const; // Icon modes only

	Void GetRequiredDimensions( UInt * pWidth, UInt * pHeight, UInt iItemCount ) const; // Actually an approximation !
	UInt GetStringWidth( const GChar * strText ) const;

	Void GetIconSpacing( UInt * outSpacingH, UInt * outSpacingV, Bool bSmallIcons ) const; // Relative to icons' top-left corners
	Void SetIconSpacing( UInt iSpacingH, UInt iSpacingV );                                 // Values must include icon widths

	Void GetItemPosition( WinGUIPoint * outPosition, UInt iItemIndex ) const;
	Void GetItemRect( WinGUIRectangle * outRectangle, UInt iItemIndex, Bool bIconOnly, Bool bLabelOnly ) const;
	Void GetSubItemRect( WinGUIRectangle * outRectangle, UInt iItemIndex, UInt iSubItemIndex, Bool bIconOnly, Bool bLabelOnly ) const;
	Void GetSubItemRect( WinGUIRectangle * outRectangle, UInt iGroupIndex, UInt iItemIndex, UInt iSubItemIndex, Bool bIconOnly, Bool bLabelOnly ) const;

	Void GetGroupMetrics( WinGUIPoint * outBorderSizeLeftTop, WinGUIPoint * outBorderSizeRightBottom ) const;   // Only Border sizes
	Void SetGroupMetrics( const WinGUIPoint & hBorderSizeLeftTop, const WinGUIPoint & hBorderSizeRightBottom ); // as of now
	
	Void GetGroupRect( WinGUIRectangle * outRectangle, UInt iGroupID, Bool bCollapsed, Bool bLabelOnly ) const;

	Void GetTileMetrics( WinGUITableTileMetrics * outTileMetrics ) const;
	Void SetTileMetrics( const WinGUITableTileMetrics * pTileMetrics );

	Void GetInsertionMarkMetrics( WinGUIRectangle * outRectangle ) const;

	Void HitTest( WinGUITableHitTestResult * outResult, const WinGUIPoint & hPoint ) const; // In client-rect coords

	// Scroll Operations ////////////////////////////////////////////////////////
	Void Scroll( Int iScrollH, Int iScrollV );
	Void ScrollToItem( UInt iIndex, Bool bAllowPartial );

	// Column Operations ////////////////////////////////////////////////////////
	UInt GetColumnWidth( UInt iIndex ) const;
	Void SetColumnWidth( UInt iIndex, UInt iWidth );

	Void GetColumnOrder( UInt * outOrderedIndices, UInt iCount ) const;
	Void SetColumnOrder( const UInt * arrOrderedIndices, UInt iCount );

	// GetColumnCount ?
	Void GetColumn( WinGUITableColumnInfos * outInfos, UInt iIndex ) const;

	Void AddColumn( UInt iIndex, const WinGUITableColumnInfos * pColumnInfos );
	Void SetColumn( UInt iIndex, const WinGUITableColumnInfos * pColumnInfos );
	Void RemoveColumn( UInt iIndex );

	UInt GetSelectedColumn() const;
	Void SelectColumn( UInt iIndex );

	// Item Operations //////////////////////////////////////////////////////////
	Bool IsItemVisible( UInt iItemIndex ) const;
	UInt GetFirstVisibleItem() const;
	UInt GetVisibleItemCount() const;

	Void SetItemIconPosition( UInt iItemIndex, const WinGUIPoint & hPosition );

	UInt GetItemCount() const;
	Void GetItem( WinGUITableItemInfos * outItemInfos, UInt iItemIndex, UInt iSubItemIndex ) const;

	Void AddItem( UInt iItemIndex, const WinGUITableItemInfos * pItemInfos );
	Void SetItem( UInt iItemIndex, const WinGUITableItemInfos * pItemInfos );
	Void SetSubItem( UInt iItemIndex, UInt iSubItemIndex, const WinGUITableItemInfos * pItemInfos );
	Void RemoveItem( UInt iItemIndex );
	Void RemoveAllItems();

	Void GetItemLabelText( GChar * outLabelText, UInt iMaxLength, UInt iItemIndex, UInt iSubItemIndex );
	Void SetItemLabelText( UInt iItemIndex, UInt iSubItemIndex, GChar * strLabelText );

	UInt GetItemIcon( UInt iItemIndex, UInt iSubItemIndex );
	Void SetItemIcon( UInt iItemIndex, UInt iSubItemIndex, UInt iIconIndex );

	Void * GetItemData( UInt iItemIndex );
	Void SetItemData( UInt iItemIndex, Void * pData );

	UInt GetSelectedItemCount() const;
	UInt GetMultiSelectMark() const;
	Void SetMultiSelectMark( UInt iItemIndex );

	UInt GetInsertionMark( Bool * outInsertAfter ) const;
	UInt GetInsertionMark( Bool * outInsertAfter, const WinGUIPoint & hPoint ) const;
	Void SetInsertionMark( UInt iItemIndex, Bool bInsertAfter );

	Bool IsItemChecked( UInt iItemIndex ) const;      // Only when using
	Void CheckItem( UInt iItemIndex, Bool bChecked ); // Checkboxes

	UInt AssignItemID( UInt iItemIndex );
	UInt GetItemFromID( UInt iUniqueID );

	Void UpdateItem( UInt iItemIndex );

	// Item Label Edition ///////////////////////////////////////////////////////
	inline Bool IsEditingItemLabel( UInt * outItemIndex = NULL ) const;

	Void GetEditItemLabel( WinGUITextEdit * outTextEdit );
	Void ReleaseEditItemLabel( WinGUITextEdit * pTextEdit );

	Void EditItemLabelStart( WinGUITextEdit * outTextEdit, UInt iItemIndex );
	Void EditItemLabelEnd( WinGUITextEdit * pTextEdit );
	Void EditItemLabelCancel( WinGUITextEdit * pTextEdit );
	
	// Group Operations /////////////////////////////////////////////////////////
	Bool HasGroup( UInt iGroupID ) const;

	UInt GetGroupCount() const;
	Void GetGroupByID( WinGUITableGroupInfos * outGroupInfos, UInt iGroupID ) const;
	Void GetGroupByIndex( WinGUITableGroupInfos * outGroupInfos, UInt iGroupIndex ) const;

	Void AddGroup( UInt iGroupIndex, const WinGUITableGroupInfos * pGroupInfos );
	Void AddGroup( const WinGUITableGroupInfos * pGroupInfos, WinGUITableGroupComparator pfComparator, Void * pUserData );
	Void SetGroup( UInt iGroupID, const WinGUITableGroupInfos * pGroupInfos );
	Void RemoveGroup( UInt iGroupID );
	Void RemoveAllGroups();

	UInt GetFocusedGroup() const; // Returns a Group Index

	// Tile Operations //////////////////////////////////////////////////////////
	Void GetTile( WinGUITableTileInfos * outInfos, UInt iItemIndex ) const;
	Void SetTile( UInt iItemIndex, const WinGUITableTileInfos * pTileInfos );

	// Search Operations ////////////////////////////////////////////////////////
	UInt GetIncrementalSearchStringLength() const;
	Void GetIncrementalSearchString( GChar * outSearchString ) const; // DANGER ! Get the Length first !

	UInt SearchItem( const GChar * strLabel, Bool bExact, UInt iStartIndex, Bool bWrapAround ) const;
	UInt SearchItem( Void * pUserData, UInt iStartIndex, Bool bWrapAround ) const;
	UInt SearchItem( const WinGUIPoint * pPoint, WinGUITableSearchSpatialDirection iSpatialDirection, UInt iStartIndex, Bool bWrapAround ) const; // Only in Icon views

	UInt SearchNextItem( UInt iStartIndex, const WinGUITableSearchOptions & hSearchOptions ) const;
	UInt SearchNextItem( UInt iStartGroupIndex, UInt iStartIndex, const WinGUITableSearchOptions & hSearchOptions ) const;

	// Sorting Operations ///////////////////////////////////////////////////////
	Void SortGroups( WinGUITableGroupComparator pfComparator, Void * pUserData );
	Void SortItemsByIndex( WinGUITableItemComparator pfComparator, Void * pUserData );
	Void SortItemsByData( WinGUITableItemComparator pfComparator, Void * pUserData );

	Void ArrangeIcons( WinGUITableIconsAlign iAlign, Bool bSnapToGrid ); // WINGUI_TABLE_ICONS_ALIGN_LEFT when SnapToGrid = true

	// Hot Tracking /////////////////////////////////////////////////////////////
	UInt GetHotItem() const;
	Void SetHotItem( UInt iIndex );

	UInt GetHoverTime() const;         // in milliseconds
	Void SetHoverTime( UInt iTimeMS ); // in milliseconds

	Void GetHotCursor( WinGUICursor * outCursor ) const; // Always Shared
	Void SetHotCursor( const WinGUICursor * pCursor );   // Caller retains ownership of the cursor

	// Work Areas ///////////////////////////////////////////////////////////////
	UInt GetMaxWorkAreasCount() const;
	UInt GetWorkAreasCount() const;

	Void GetWorkAreas( WinGUIRectangle * outWorkAreas, UInt iMaxCount ) const;
	Void SetWorkAreas( const WinGUIRectangle * arrWorkAreas, UInt iCount );

	// Drag & Drop //////////////////////////////////////////////////////////////
	Void CreateDragImageList( WinGUIImageList * outDragImageList, WinGUIPoint * outInitialPosition, UInt iItemIndex ); // Caller takes ownership

	// Info/Tool Tips ///////////////////////////////////////////////////////////
	Void SetInfoTip( UInt iItemIndex, UInt iSubItemIndex, GChar * strInfoText );

	//ListView_GetToolTips
	//ListView_SetToolTips

private:
	// Helpers
	Void _Convert_ColumnInfos( WinGUITableColumnInfos * outColumnInfos, const Void * pColumnInfos ) const; // LVCOLUMN
	Void _Convert_ColumnInfos( Void * outColumnInfos, const WinGUITableColumnInfos * pColumnInfos ) const;

	Void _Convert_GroupInfos( WinGUITableGroupInfos * outGroupInfos, const Void * pGroupInfos ) const; // LVGROUP
	Void _Convert_GroupInfos( Void * outGroupInfos, const WinGUITableGroupInfos * pGroupInfos ) const;

	Void _Convert_ItemInfos( WinGUITableItemInfos * outItemInfos, const Void * pItemInfos ) const; // LVITEM
	Void _Convert_ItemInfos( Void * outItemInfos, const WinGUITableItemInfos * pItemInfos ) const;

	Void _Convert_TileInfos( WinGUITableTileInfos * outTileInfos, const Void * pTileInfos ) const; // LVTILEINFO
	Void _Convert_TileInfos( Void * outTileInfos, const WinGUITableTileInfos * pTileInfos ) const;

	Void _Convert_TileMetrics( WinGUITableTileMetrics * outTileMetrics, const Void * pTileMetrics ) const; // LVTILEVIEWINFO
	Void _Convert_TileMetrics( Void * outTileMetrics, const WinGUITableTileMetrics * pTileMetrics ) const;

	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode, Void * pParameters );

	// State (Track the most important ones)
	Bool m_bVirtualTable;

	Bool m_bHasBackBuffer;
	Bool m_bHasSharedImageLists;

	WinGUITableViewMode m_iViewMode;
	Bool m_bGroupMode;
	Bool m_bHasHeadersInAllViews;

	Bool m_bHasColumnHeaders;
	Bool m_bHasStaticColumnHeaders;
	Bool m_bHasDraggableColumnHeaders;
	Bool m_bHasIconColumnOverflowButton;

	Bool m_bHasCheckBoxes;
	Bool m_bHasIconLabels;
	Bool m_bHasEditableLabels;
	Bool m_bHasSubItemImages;

	Bool m_bSingleItemSelection;
	Bool m_bIconSimpleSelection;

	Bool m_bAutoSortAscending;
	Bool m_bAutoSortDescending;

	Bool m_bHasHotTrackingSingleClick;
	Bool m_bHasHotTrackingDoubleClick;
	Bool m_bHasHotTrackingSelection;

	Bool m_bHasInfoTips;

	// Edit Label Management
	Void * m_hEditLabelHandle; // HWND
	UInt m_iEditLabelItemIndex;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUITable.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITABLE_H

