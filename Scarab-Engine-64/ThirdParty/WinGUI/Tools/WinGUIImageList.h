/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Tools/WinGUIImageList.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Image Lists
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : Note we only use DDB here. (Device-Dependant Bitmaps)
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_WINGUI_TOOLS_WINGUIIMAGELIST_H
#define SCARAB_THIRDPARTY_WINGUI_TOOLS_WINGUIIMAGELIST_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "WinGUIImage.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Drawing Options
typedef struct _wingui_imagelist_draw_options {
	Bool bStencilMode;       // Draw the mask instead of the image
	Bool bPreserveDestAlpha; // Operations don't affect dest alpha channel
	
	// Background
	Bool bUseBackgroundColor;
	Bool bUseDefaultBackgroundColor;
	UInt iBackgroundColor;

	// Blending
	Bool bUseBlending25; // Mutually
	Bool bUseBlending50; // Exclusive
	Bool bBlendWithDestination;
	Bool bUseDefaultBlendForegroundColor;
	UInt iBlendForegroundColor;

	// Alpha Blending
	Bool bUseAlphaBlending;
	UInt iAlphaValue;

	// Saturation
	Bool bUseSaturation;

	// Raster Operation
	Bool bUseRasterOp;
	WinGUIRasterOperation iRasterOp;

	// Overlay
	Bool bUseOverlay;
	Bool bOverlayRequiresMask;
	UInt iOverlayIndex;
	
	// Scaling
	Bool bUseScaling;
	Bool bUseDPIScaling;
} WinGUIImageListDrawOptions;

// Prototypes
class WinGUIWindow;

class WinGUITable;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIImageList class
class WinGUIImageList
{
public:
	WinGUIImageList();
	~WinGUIImageList();

	// Properties
	inline Bool IsCreated() const;
	inline Bool HasMasks() const;

	inline UInt GetWidth() const;
	inline UInt GetHeight() const;

	// Deferred Creation/Destruction
	Void Create( UInt iWidth, UInt iHeight, UInt iInitialImageCount, UInt iGrowCount );
	Void CreateMasked( UInt iWidth, UInt iHeight, UInt iInitialImageCount, UInt iGrowCount );

	Void CreateDuplicate( const WinGUIImageList * pList );

		// Create the list from 2 provided images, Image B drawn transparently over Image A.
		// Their Mask are ORed to generate result mask, both ImageLists A and B must have masks.
	Void MergeImages( WinGUIImageList * pListA, UInt iIndexA, WinGUIImageList * pListB, UInt iIndexB, Int iDX, Int iDY );

	Void Destroy();

	// Images Management
	UInt GetImageCount() const;
	Void GetImage( UInt iIndex, WinGUIRectangle * outBoundingRect, WinGUIBitmap * outImage, WinGUIBitmap * outMask = NULL ) const;

	Void MakeIcon( UInt iIndex, WinGUIIcon * outIcon, const WinGUIImageListDrawOptions & hOptions ) const;

	UInt AddImage( WinGUIBitmap * pImage, WinGUIBitmap * pMask = NULL );
	UInt AddImageMasked( WinGUIBitmap * pImage, UInt iKeyColor );
	UInt AddIcon( WinGUIIcon * pIcon );
	UInt AddCursor( WinGUICursor * pCursor );

	Void ReplaceImage( UInt iIndex, WinGUIBitmap * pImage, WinGUIBitmap * pMask = NULL );
	Void ReplaceIcon( UInt iIndex, WinGUIIcon * pIcon );
	Void ReplaceCursor( UInt iIndex, WinGUICursor * pCursor );

	Void SwapImages( UInt iIndexA, UInt iIndexB );    // Current Windows versions don't
	Void CopyImage( UInt iIndexDst, UInt iIndexSrc ); // allow to specify Dst/Src Lists

	Void RemoveImage( UInt iIndex );
	Void RemoveAll();

	// Settings
	UInt GetBackgroundColor() const;
	UInt SetBackgroundColor( UInt iColor );

		// Requires Masks
	Void SetOverlayImage( UInt iImageIndex, UInt iOverlayIndex );

	// Drawing
	Void Draw( WinGUIBitmap * pTarget, const WinGUIPoint & hDestOrigin, UInt iSrcImage, const WinGUIRectangle & hSrcRect, const WinGUIImageListDrawOptions & hOptions );
	Void Render( WinGUIElement * pTarget, const WinGUIPoint & hDestOrigin, UInt iSrcImage, const WinGUIRectangle & hSrcRect, const WinGUIImageListDrawOptions & hOptions );

	// Drag & Drop (Positions are given in window rect coords, not client rect !)
	Void DragBegin( UInt iImageIndex, const WinGUIPoint & hHotSpot );
	Void DragEnter( WinGUIWindow * pOwner, const WinGUIPoint & hPosition );
	Void DragEnd();
	Void DragLeave( WinGUIWindow * pOwner );

	Void DragShow( Bool bShow );
	Void DragMove( const WinGUIPoint & hPosition );

	//Void GetDragImage( WinGUIImageList * outImageList, WinGUIPoint * outDragPosition, WinGUIPoint * outHotSpot ) const;
	Void CombineDragImages( UInt iNewImageIndex, const WinGUIPoint & hNewHotSpot );

private:
	friend class WinGUITable;

	// Helpers
	Void _CreateFromHandle( Void * hHandle );

	// Members
	Void * m_hHandle; // HIMAGELIST

	Bool m_bHasMasks;
	UInt m_iWidth;
	UInt m_iHeight;

	Bool m_bIsDragging;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIImageList.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_TOOLS_WINGUIIMAGELIST_H

