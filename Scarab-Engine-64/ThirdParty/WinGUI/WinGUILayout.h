/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUILayout.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Layouts
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
#ifndef SCARAB_THIRDPARTY_WINGUI_WINGUILAYOUT_H
#define SCARAB_THIRDPARTY_WINGUI_WINGUILAYOUT_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../System/System.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Layout Types
enum WinGUILayoutType {
	WINGUI_LAYOUT_MANUAL = 0, // Handy for very simple UIs
	WINGUI_LAYOUT_GRID	      // This should be your go-to choice
};

// Layout Anchor Modes
enum WinGUILayoutAnchor {
	// Anchor to Center
	WINGUI_LAYOUT_ANCHOR_CENTER = 0,

	// Anchor to Edges
	WINGUI_LAYOUT_ANCHOR_LEFT,
	WINGUI_LAYOUT_ANCHOR_TOP,
	WINGUI_LAYOUT_ANCHOR_RIGHT,
	WINGUI_LAYOUT_ANCHOR_BOTTOM,

	// Anchor to Corners
	WINGUI_LAYOUT_ANCHOR_TOPLEFT,
	WINGUI_LAYOUT_ANCHOR_TOPRIGHT,
	WINGUI_LAYOUT_ANCHOR_BOTTOMLEFT,
	WINGUI_LAYOUT_ANCHOR_BOTTOMRIGHT,

	// Anchor to Custom Point
	WINGUI_LAYOUT_ANCHOR_FIXED,
	WINGUI_LAYOUT_ANCHOR_SCALING
};

// Layout Fill Modes
enum WinGUILayoutFill {
	// Fill using Grid Cell
	WINGUI_LAYOUT_FILL_DEFAULT = 0,

	// Fill using Custom Dimensions
	WINGUI_LAYOUT_FILL_FIXED,
	WINGUI_LAYOUT_FILL_SCALING
};

// All-Purpose Point & Rectangle structures
typedef struct _wingui_pointf {
	Float fX;
	Float fY;
} WinGUIPointF;

typedef struct _wingui_point {
	UInt iX;
	UInt iY;
} WinGUIPoint;

typedef struct _wingui_rectangle {
	UInt iLeft;
	UInt iTop;
	UInt iWidth;
	UInt iHeight;
} WinGUIRectangle;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUILayout interface
class WinGUILayout
{
public:
	virtual ~WinGUILayout() {}

	// Layout Type
	virtual WinGUILayoutType GetType() const = 0;

	// A Layout computes a child's window area from its parent's client area.
	virtual Void ComputeLayout( WinGUIRectangle * outChildArea, const WinGUIRectangle & hParentClientArea ) const = 0;
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIManualLayout class
class WinGUIManualLayout : public WinGUILayout
{
public:
	WinGUIManualLayout();
	WinGUIManualLayout( const WinGUIManualLayout & rhs );
	virtual ~WinGUIManualLayout() {}

	// Layout Type
	inline virtual WinGUILayoutType GetType() const;

	// Layout Algorithm
	virtual Void ComputeLayout( WinGUIRectangle * outChildArea, const WinGUIRectangle & hParentClientArea ) const;

	// Manual Layout is very simple :
	// Coordinates are always relative to the parent area.
	// The user directly specifies the child's area, either in fixed or scaling coordinates.
	// Fixed (integer) coordinates will not change when resizing
	// Scaling (float) coordinates will adjust when resizing
	// Default behaviour is scaling position and fixed size
	// User is responsible for ensuring the child area is properly included in the parent area
	// when using fixed position / size !

	// Position
	Bool UseScalingPosition;
	WinGUIPoint FixedPosition;
	WinGUIPointF ScalingPosition; // in [0;1]

	// Size
	Bool UseScalingSize;
	WinGUIPoint FixedSize;
	WinGUIPointF ScalingSize; // in [0;1]
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIGridLayout class
class WinGUIGridLayout : public WinGUILayout
{
public:
	WinGUIGridLayout();
	WinGUIGridLayout( const WinGUIGridLayout & rhs );
	virtual ~WinGUIGridLayout() {}

	// Layout Type
	inline virtual WinGUILayoutType GetType() const;

	// Layout Algorithm
	virtual Void ComputeLayout( WinGUIRectangle * outChildArea, const WinGUIRectangle & hParentClientArea ) const;

	// Grid Layout is the standard method for flexible UI placement :
	// The user specifies a grid mesh to partition the parent area (GridColumns, GridRows).
	// Then a rectangular area of grid cells is selected (GridArea).
	// The user then specifies anchoring and filling policies inside the selected grid area (AnchorMode, FillModeH, FillModeV).
	// Anchoring and Filling can be specified to use custom position and size, using fixed or scaling coordinates.
	// Those coordinates are relative to the selected GridArea.

	// Grid Mesh
	UInt GridColumns; // X-Size of the grid, in cells
	UInt GridRows;	  // Y-Size of the grid, in cells

	// Grid Area Selection
	WinGUIRectangle GridArea; // Selected area on the grid, in cells

	// Anchoring Mode
	WinGUILayoutAnchor AnchorMode;
	WinGUIPoint FixedAnchor;    // WINGUI_LAYOUT_ANCHOR_FIXED Mode
	WinGUIPointF ScalingAnchor; // WINGUI_LAYOUT_ANCHOR_SCALING Mode

	// Filling Modes
	WinGUILayoutFill FillModeH;
	WinGUILayoutFill FillModeV;
	WinGUIPoint FixedFill;    // WINGUI_LAYOUT_FILL_FIXED Modes
	WinGUIPointF ScalingFill; // WINGUI_LAYOUT_FILL_SCALING Modes
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUILayout.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_WINGUILAYOUT_H

