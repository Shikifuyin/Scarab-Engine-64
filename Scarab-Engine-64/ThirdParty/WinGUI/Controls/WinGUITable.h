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
	WINGUI_TABLE_ICONS_ALIGN_TOP = 0,
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
	Bool bMakeVirtualTable; // When managing large amount of data (large = millions of items)
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

	// General Settings /////////////////////////////////////////////////////////
	inline Bool IsVirtual() const;

	Bool IsUnicode() const;
	Bool IsANSI() const;

	Void SetUnicode();
	Void SetANSI();

	Void SetAllocatedItemCount( UInt iPreAllocatedItemCount );

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

	inline Bool HasCheckBoxes() const;
	Void ToggleCheckBoxes( Bool bEnable );
	Void ToggleAutoCheckOnSelect( Bool bEnable );

	inline Bool HasEditableLabels() const;
	Void ToggleEditableLabels( Bool bEnable );

	// Visual Settings //////////////////////////////////////////////////////////
	UInt GetBackgroundColor() const;
	Void SetBackgroundColor( UInt iColor );

	//Bool GetBackgroundImage( WinGUIBitmap * outImage,  ) const; // ListView_GetBkImage
	Void SetBackgroundImage( const WinGUIBitmap * pImage, const WinGUIPointF & hRelativePos, Bool bUseTiling );
	Void RemoveBackgroundImage();

	UInt GetTextBackgroundColor() const;
	Void SetTextBackgroundColor( UInt iColor );

	UInt GetTextColor() const;
	Void SetTextColor( UInt iColor );

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

	Void SetItemLabelText( UInt iItemIndex, UInt iSubItemIndex, GChar * strLabelText );

	//Void * EditItemLabelStart( UInt iIndex ); // Table must have focus for this !
	Void EditItemLabelCancel();
	//ListView_GetEditControl

	Void UpdateItem( UInt iItemIndex );

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
	UInt SearchItem( const WinGUIPoint * pPoint, KeyCode iDirection, UInt iStartIndex, Bool bWrapAround ) const; // Only in Icon views

	UInt SearchNextItem( UInt iStartIndex, Bool bReverse, Bool bSameGroup, Bool bVisible, Bool bHasFocus, Bool bSelected, Bool bCutMarked, Bool bDropHighlight ) const;
	UInt SearchNextItem( UInt iStartIndex, KeyCode iDirection, Bool bSameGroup, Bool bVisible, Bool bHasFocus, Bool bSelected, Bool bCutMarked, Bool bDropHighlight ) const;

	UInt SearchNextItem( UInt iStartGroupIndex, UInt iStartIndex, Bool bReverse, Bool bSameGroup, Bool bVisible, Bool bHasFocus, Bool bSelected, Bool bCutMarked, Bool bDropHighlight ) const;
	UInt SearchNextItem( UInt iStartGroupIndex, UInt iStartIndex, KeyCode iDirection, Bool bSameGroup, Bool bVisible, Bool bHasFocus, Bool bSelected, Bool bCutMarked, Bool bDropHighlight ) const;

	// Sorting Operations ///////////////////////////////////////////////////////
	Void SortGroups( WinGUITableGroupComparator pfComparator, Void * pUserData );
	Void SortItemsByIndex( WinGUITableItemComparator pfComparator, Void * pUserData );
	Void SortItemsByData( WinGUITableItemComparator pfComparator, Void * pUserData );

	//ListView_Arrange()

	// Hot Tracking /////////////////////////////////////////////////////////////
	UInt GetHotItem() const;
	Void SetHotItem( UInt iIndex );

	UInt GetHoverTime() const;         // in milliseconds
	Void SetHoverTime( UInt iTimeMS ); // in milliseconds

	//ListView_GetHotCursor
	//ListView_SetHotCursor

	// Work Areas ///////////////////////////////////////////////////////////////
	//ListView_GetNumberOfWorkAreas
	//ListView_GetWorkAreas
	//ListView_SetWorkAreas

	// Drag & Drop //////////////////////////////////////////////////////////////
	//ListView_CreateDragImage

	// Tool/Info Tips ///////////////////////////////////////////////////////////
	Void SetInfoTip( UInt iItemIndex, UInt iSubItemIndex, GChar * strInfoText );

	//ListView_GetToolTips
	//ListView_SetToolTips
	


	

		// HeadersInAllViews or Detailed View
	Void ToggleStaticColumnHeaders( Bool bEnable );
	Void SnapColumnWidths( Bool bEnable );
	Void AutoSizeColumns( Bool bEnable );


	

	Void ToggleSingleItemSelection( Bool bEnable );

	Void AlwaysShowSelection( Bool bEnable );

	Void ToggleBorderSelection( Bool bEnable );
	
	Void ToggleSorting( Bool bEnable, Bool bAscendingElseDescending ); // Not with virtual tables

	

	Void ToggleInfoTips( Bool bEnable );

	Void ToggleHotTracking( Bool bEnable, Bool bUseSingleClick );
	Void ToggleHotTrackSelection( Bool bEnable );
	Void ToggleHotUnderline( Bool bEnable );
	Void ToggleColdUnderline( Bool bEnable ); // Requires DblClick HotTracking

	Void UseSharedImageLists( Bool bEnable );
	Void UseBackBuffering( Bool bEnable );
	Void UseTransparentBackground( Bool bEnable );
	Void UseTransparentShadowText( Bool bEnable );

		// Icon Modes
	Void SetIconAlignment( WinGUITableIconsAlign iAlign );
	Void ToggleAutoArrangeIcons( Bool bEnable );
	Void HideIconLabels( Bool bEnable );
	Void PreventIconLabelWrap( Bool bEnable );
	Void ToggleIconColumnOverflowButton( Bool bEnable ); // Only when HeadersInAllViews
	Void JustifyIconColumns( Bool bEnable );
	Void ToggleIconSnapToGrid( Bool bEnable );
	Void UseIconSimpleSelection( Bool bEnable );

		// Detailed Mode / HeadersInAllViews
	Void ToggleColumnHeaders( Bool bEnable );
	Void ToggleHeaderDragNDrop( Bool bEnable );
	Void ToggleFullRowSelection( Bool bEnable );
	Void ShowGridLines( Bool bEnable );
	Void ToggleSubItemImages( Bool bEnable );




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

	// State
	Bool m_bVirtualTable;
	WinGUITableViewMode m_iViewMode;
	Bool m_bGroupMode;

	Bool m_bHasHeadersInAllViews;
	Bool m_bHasCheckBoxes;
	Bool m_bHasEditableLabels;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUITable.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITABLE_H

