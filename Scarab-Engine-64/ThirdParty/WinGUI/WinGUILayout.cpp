/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUILayout.cpp
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
// Includes
#include "WinGUILayout.h"

/////////////////////////////////////////////////////////////////////////////////
// WinGUIManualLayout implementation
WinGUIManualLayout::WinGUIManualLayout()
{
	UseScalingPosition = true;
	FixedPosition.iX = 0;
	FixedPosition.iY = 0;
	ScalingPosition.fX = 0.0f;
	ScalingPosition.fY = 0.0f;

	UseScalingSize = false;
	FixedSize.iX = 0;
	FixedSize.iY = 0;
	ScalingSize.fX = 0.0f;
	ScalingSize.fY = 0.0f;
}
WinGUIManualLayout::WinGUIManualLayout( const WinGUIManualLayout & rhs )
{
	UseScalingPosition = rhs.UseScalingPosition;
	FixedPosition.iX = rhs.FixedPosition.iX;
	FixedPosition.iY = rhs.FixedPosition.iY;
	ScalingPosition.fX = rhs.ScalingPosition.fX;
	ScalingPosition.fY = rhs.ScalingPosition.fY;

	UseScalingSize = rhs.UseScalingSize;
	FixedSize.iX = rhs.FixedSize.iX;
	FixedSize.iY = rhs.FixedSize.iY;
	ScalingSize.fX = rhs.ScalingSize.fX;
	ScalingSize.fY = rhs.ScalingSize.fY;
}

Void WinGUIManualLayout::ComputeLayout( WinGUIRectangle * outChildArea, const WinGUIRectangle & hParentClientArea ) const
{
	// Compute Position
	if ( UseScalingPosition ) {
		outChildArea->iLeft = (UInt)( ScalingPosition.fX * (Float)(hParentClientArea.iWidth) );
		outChildArea->iTop = (UInt)( ScalingPosition.fY * (Float)(hParentClientArea.iHeight) );
	} else {
		outChildArea->iLeft = FixedPosition.iX;
		outChildArea->iTop = FixedPosition.iY;
	}

	// Compute Size
	if ( UseScalingSize ) {
		outChildArea->iWidth = (UInt)( ScalingSize.fX * (Float)(hParentClientArea.iWidth) );
		outChildArea->iHeight = (UInt)( ScalingSize.fY * (Float)(hParentClientArea.iHeight) );
	} else {
		outChildArea->iWidth = FixedSize.iX;
		outChildArea->iHeight = FixedSize.iY;
	}
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUIGridLayout implementation
WinGUIGridLayout::WinGUIGridLayout()
{
	GridColumns = 1;
	GridRows = 1;

	GridArea.iLeft = 0;
	GridArea.iTop = 0;
	GridArea.iWidth = 1;
	GridArea.iHeight = 1;

	AnchorMode = WINGUI_LAYOUT_ANCHOR_CENTER;
	FixedAnchor.iX = 0;
	FixedAnchor.iY = 0;
	ScalingAnchor.fX = 0.0f;
	ScalingAnchor.fY = 0.0f;

	FillModeH = WINGUI_LAYOUT_FILL_DEFAULT;
	FillModeV = WINGUI_LAYOUT_FILL_DEFAULT;
	FixedFill.iX = 0;
	FixedFill.iY = 0;
	ScalingFill.fX = 0.0f;
	ScalingFill.fY = 0.0f;
}
WinGUIGridLayout::WinGUIGridLayout( const WinGUIGridLayout & rhs )
{
	GridColumns = rhs.GridColumns;
	GridRows = rhs.GridRows;

	GridArea.iLeft = rhs.GridArea.iLeft;
	GridArea.iTop = rhs.GridArea.iTop;
	GridArea.iWidth = rhs.GridArea.iWidth;
	GridArea.iHeight = rhs.GridArea.iHeight;

	AnchorMode = rhs.AnchorMode;
	FixedAnchor.iX = rhs.FixedAnchor.iX;
	FixedAnchor.iY = rhs.FixedAnchor.iY;
	ScalingAnchor.fX = rhs.ScalingAnchor.fX;
	ScalingAnchor.fY = rhs.ScalingAnchor.fY;

	FillModeH = rhs.FillModeH;
	FillModeV = rhs.FillModeV;
	FixedFill.iX = rhs.FixedFill.iX;
	FixedFill.iY = rhs.FixedFill.iY;
	ScalingFill.fX = rhs.ScalingFill.fX;
	ScalingFill.fY = rhs.ScalingFill.fY;
}

Void WinGUIGridLayout::ComputeLayout( WinGUIRectangle * outChildArea, const WinGUIRectangle & hParentClientArea ) const
{
	// Check grid dimensions are valid
    DebugAssert( GridColumns > 0 && GridRows > 0 );

	// Compute grid cell dimensions
	Float fGridCellWidth = ( (Float)(hParentClientArea.iWidth) / (Float)GridColumns );
	Float fGridCellHeight = ( (Float)(hParentClientArea.iHeight) / (Float)GridRows );

	// Deduce Selected Area
	Float fX = fGridCellWidth * (Float)( GridArea.iLeft );
	Float fY = fGridCellHeight * (Float)( GridArea.iTop );
	Float fWidth = fGridCellWidth * (Float)( GridArea.iWidth );
	Float fHeight = fGridCellHeight * (Float)( GridArea.iHeight );

	// Compute Child Rect Size
	UInt iChildWidth, iChildHeight;
	switch( FillModeH ) {
		case WINGUI_LAYOUT_FILL_DEFAULT:
			iChildWidth = (UInt)fWidth;
			break;
		case WINGUI_LAYOUT_FILL_FIXED:
			iChildWidth = FixedFill.iX;
			break;
		case WINGUI_LAYOUT_FILL_SCALING:
			iChildWidth = (UInt)( ScalingFill.fX * fWidth );
			break;
		default: DebugAssert(false); break;
	}
	switch( FillModeV ) {
		case WINGUI_LAYOUT_FILL_DEFAULT:
			iChildHeight = (UInt)fHeight;
			break;
		case WINGUI_LAYOUT_FILL_FIXED:
			iChildHeight = FixedFill.iY;
			break;
		case WINGUI_LAYOUT_FILL_SCALING:
			iChildHeight = (UInt)( ScalingFill.fY * fHeight );
			break;
		default: DebugAssert(false); break;
	}

	// Compute Child Rect Position
	UInt iChildX, iChildY;
	switch( AnchorMode ) {
		case WINGUI_LAYOUT_ANCHOR_CENTER:
			iChildX = (UInt)( (fWidth - (Float)iChildWidth) * 0.5f );
			iChildY = (UInt)( (fHeight - (Float)iChildHeight) * 0.5f );
			break;
		case WINGUI_LAYOUT_ANCHOR_LEFT:
			iChildX = (UInt)fX;
			iChildY = (UInt)( (fHeight - (Float)iChildHeight) * 0.5f );
			break;
		case WINGUI_LAYOUT_ANCHOR_TOP:
			iChildX = (UInt)( (fWidth - (Float)iChildWidth) * 0.5f );
			iChildY = (UInt)fY;
			break;
		case WINGUI_LAYOUT_ANCHOR_RIGHT:
			iChildX = (UInt)(fX + fWidth) - iChildWidth;
			iChildY = (UInt)( (fHeight - (Float)iChildHeight) * 0.5f );
			break;
		case WINGUI_LAYOUT_ANCHOR_BOTTOM:
			iChildX = (UInt)( (fWidth - (Float)iChildWidth) * 0.5f );
			iChildY = (UInt)(fY + fHeight) - iChildHeight;
			break;
		case WINGUI_LAYOUT_ANCHOR_TOPLEFT:
			iChildX = (UInt)fX;
			iChildY = (UInt)fY;
			break;
		case WINGUI_LAYOUT_ANCHOR_TOPRIGHT:
			iChildX = (UInt)(fX + fWidth) - iChildWidth;
			iChildY = (UInt)fY;
			break;
		case WINGUI_LAYOUT_ANCHOR_BOTTOMLEFT:
			iChildX = (UInt)fX;
			iChildY = (UInt)(fY + fHeight) - iChildHeight;
			break;
		case WINGUI_LAYOUT_ANCHOR_BOTTOMRIGHT:
			iChildX = (UInt)(fX + fWidth) - iChildWidth;
			iChildY = (UInt)(fY + fHeight) - iChildHeight;
			break;
		case WINGUI_LAYOUT_ANCHOR_FIXED:
			iChildX = FixedAnchor.iX;
			iChildY = FixedAnchor.iY;
			break;
		case WINGUI_LAYOUT_ANCHOR_SCALING:
			iChildX = (UInt)( ScalingAnchor.fX * fWidth );
			iChildY = (UInt)( ScalingAnchor.fY * fHeight );
			break;
		default: DebugAssert(false); break;
	}

    // Done
	outChildArea->iLeft = iChildX;
	outChildArea->iTop = iChildY;
	outChildArea->iWidth = iChildWidth;
	outChildArea->iHeight = iChildHeight;
}

