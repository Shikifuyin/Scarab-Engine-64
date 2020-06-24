/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Tools/WinGUIImage.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Images (Bitmap or Icon)
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_THIRDPARTY_WINGUI_TOOLS_WINGUIIMAGE_H
#define SCARAB_THIRDPARTY_WINGUI_TOOLS_WINGUIIMAGE_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../WinGUILayout.h" // Rectangle, Point structures

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Image types
enum WinGUIImageType {
	WINGUI_IMAGE_BITMAP = 0,
	WINGUI_IMAGE_ICON,
	WINGUI_IMAGE_CURSOR
};

// Color Depth
enum WinGUIBitmapBPP {
	WINGUI_BITMAP_BPP_0 = 0, // Implied by file format (JPEG/PNG)
	WINGUI_BITMAP_BPP_1 = 1, // Monochrome, 2-colors palette
	//WINGUI_BITMAP_BPP_4 = 4, // 16-colors palette (obsolete, unsupported)
	//WINGUI_BITMAP_BPP_8 = 8, // 256-colors palette (obsolete, unsupported)
	WINGUI_BITMAP_BPP_16 = 16,
	WINGUI_BITMAP_BPP_24 = 24,
	WINGUI_BITMAP_BPP_32 = 32
};

// Compression
enum WinGUIBitmapCompression {
	WINGUI_BITMAP_RGB = 0,  // Uncompressed
	WINGUI_BITMAP_BITFIELD, // Uncompressed, Use Masks, 16/32 BPP only
	WINGUI_BITMAP_JPEG,     // JPEG Format (Requires Bottom-Up scanlines)
	WINGUI_BITMAP_PNG       // PNG Format (Requires Bottom-Up scanlines)
};

// Color Space
enum WinGUIBitmapColorSpace {
	WINGUI_BITMAP_SRGB = 0,   // System Default
	WINGUI_BITMAP_CALIBRATED  // Use provided Gamma & EndPoints
	// No support for Profiles yet ...
};

// Rendering Intent
enum WinGUIBitmapIntent {
	WINGUI_BITMAP_COLORIMETRIC_ABS = 0, // Maintain White, Match Nearest Color
	WINGUI_BITMAP_COLORIMETRIC_REL,     // Maintain Colorimetric Match
	WINGUI_BITMAP_SATURATION,           // Maintain Saturation
	WINGUI_BITMAP_PERCEPTUAL            // Maintain Contrast
};

// Bitmap Descriptor
typedef struct _wingui_bitmap_desc {
	Bool bBottomUpElseTopDown; // Scanlines ordering
	UInt iWidth;
	UInt iHeight;
	WinGUIBitmapBPP iBPP;
	WinGUIBitmapCompression iCompression;
	UInt iByteSize;
	UInt iPixelsPerMeterX;
	UInt iPixelsPerMeterY;
	UInt iMaskRed;
	UInt iMaskGreen;
	UInt iMaskBlue;
	UInt iMaskAlpha;
	WinGUIBitmapColorSpace iColorSpace;
	struct _endpoints {
		struct _coords {
			UInt iFixed2_30_X; // Fixed Point values
			UInt iFixed2_30_Y; // 2 digits for integer part
			UInt iFixed2_30_Z; // 30 digits for decimal part
		} Red, Green, Blue;
	} hEndPoints;
	UInt iGammaRed;
	UInt iGammaGreen;
	UInt iGammaBlue;
	WinGUIBitmapIntent iRenderingIntent;
} WinGUIBitmapDescriptor;

// Raster Operations
enum WinGUIRasterOperation {
	WINGUI_RASTER_BLACK = 0,        // Dest = Black
	WINGUI_RASTER_WHITE,            // Dest = White
	WINGUI_RASTER_COPY,             // Dest = Src
	WINGUI_RASTER_NOTDST,           // Dest = NOT(Dest)              -- (Invert Dest)
	WINGUI_RASTER_NOTSRC,           // Dest = NOT(Src)               -- (Invert Src & Copy)
	WINGUI_RASTER_AND,              // Dest = Dest AND Src           -- (Filter Dest using Src as a mask)
	WINGUI_RASTER_OR,               // Dest = Dest OR Src            -- (Fill Dest using Src as a mask, ie Paint Over)
	WINGUI_RASTER_XOR,              // Dest = Dest XOR Src           -- (Invert Dest using Src as a mask)
	WINGUI_RASTER_NOTDST_AND_SRC,   // Dest = Not(Dest) AND Src      -- (Erase Dest using Src as a mask)
	WINGUI_RASTER_DST_OR_NOTSRC,    // Dest = Dest OR NOT(Src)       -- (Fill Dest using Src as an inverted mask)
	WINGUI_RASTER_NOTDST_AND_NOTSRC // Dest = NOT(Dest) AND NOT(Src) -- (Erase Dest using Src as an inverted mask)
};

// Load Resize Mode
enum WinGUIImageResizeMode {
	WINGUI_IMAGE_RESIZE_KEEP = 0, // Keep resource dimension
	WINGUI_IMAGE_RESIZE_DEFAULT,  // Use system default (Icon/Cursor)
	WINGUI_IMAGE_RESIZE_USER      // Use specified values
};

// Load Parameters
typedef struct _wingui_image_load_params {
	WinGUIImageType iType;
	WinGUIImageResizeMode iResizeWidth;
	WinGUIImageResizeMode iResizeHeight;
	UInt iWidth;
	UInt iHeight;
	Bool bMakeDIB;
	Bool bMonochrome;
	Bool bTrueVGA;
	Bool bSharedResource; // Resources only, Required for system icon/cursor
} WinGUIImageLoadParameters;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIImage class
class WinGUIImage
{
public:
	WinGUIImage();
	~WinGUIImage();

	// State
	inline Bool IsCreated() const;
	inline Bool IsDeviceDependant() const;
	inline Bool IsShared() const;
	inline WinGUIImageType GetType() const;

	// Device-Dependant Bitmap (DDB)
	Void CreateDDBitmap( UInt iWidth, UInt iHeight );
	Void CreateDDBitmapMask( UInt iWidth, UInt iHeight );

	inline UInt GetDDWidth() const;
	inline UInt GetDDHeight() const;

	// Device-Independant Bitmap (DIB)
	Void CreateDIBitmap( const WinGUIBitmapDescriptor & hDescriptor );

	inline const WinGUIBitmapDescriptor * GetDIBDescriptor() const;

	Void LockDIB( Byte ** ppMemory );
	Void UnlockDIB( Byte ** ppMemory );

	// Load from File/Resource
	Void LoadFromFile( const GChar * strFilename, const WinGUIImageLoadParameters & hLoadParams );
	Void LoadFromResource( UInt iResourceID, const WinGUIImageLoadParameters & hLoadParams );

	// Resources Release
	Void Destroy();

	// Bit Block Transfer Operations
	Void BitBlit( const WinGUIRectangle & hDestRect, const WinGUIImage * pSrcImage, const WinGUIPoint & hSrcOrigin, WinGUIRasterOperation iOperation );
	Void StretchBlit( const WinGUIRectangle & hDestRect, const WinGUIImage * pSrcImage, const WinGUIRectangle & hSrcRect, WinGUIRasterOperation iOperation );
	Void MaskBlit( const WinGUIRectangle & hDestRect, const WinGUIImage * pSrcImage, const WinGUIPoint & hSrcOrigin,
				   const WinGUIImage * pMask, const WinGUIPoint & hMaskOrigin,
				   WinGUIRasterOperation iForegroundOP, WinGUIRasterOperation iBackgroundOP );
	Void TransparentBlit( const WinGUIRectangle & hDestRect, const WinGUIImage * pSrcImage, const WinGUIRectangle & hSrcRect, UInt iKeyColor );

	// TODO : Screen Capture ?
	// TODO : Save to File ?

private:
	// Helpers
	Void * _GetAppWindowHandle() const;

	// Members
	Bool m_bIsDeviceDependant;
	Bool m_bShared;

	Void * m_hHandle; // HBITMAP

	WinGUIImageType m_iType;
	UInt m_iDDWidth;
	UInt m_iDDHeight;

	WinGUIBitmapDescriptor m_hBitmapDesc;
	Byte * m_pBitmapData;
	Bool m_bLocked;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIImage.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_TOOLS_WINGUIIMAGE_H

