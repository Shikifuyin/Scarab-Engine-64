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


private:
	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode );

	// Flag for Virtual Tables
	Bool m_bVirtualTable;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUITable.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITABLE_H

