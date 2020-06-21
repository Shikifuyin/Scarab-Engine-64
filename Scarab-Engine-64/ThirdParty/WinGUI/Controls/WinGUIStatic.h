/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIStatic.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : Static Text/Graphics
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
#ifndef SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUISTATIC_H
#define SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUISTATIC_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../WinGUIControl.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Static Types
enum WinGUIStaticType {
	WINGUI_STATIC_FRAME = 0,
	WINGUI_STATIC_RECT,
	WINGUI_STATIC_TEXT,
	WINGUI_STATIC_BITMAP,
	WINGUI_STATIC_ICON
};

// Frame Properties
enum WinGUIStaticFrameType {
	WINGUI_STATIC_FRAME_ETCHED = 0,
	WINGUI_STATIC_FRAME_ETCHED_HORIZ,
	WINGUI_STATIC_FRAME_ETCHED_VERT
};

// Rectangle Properties
enum WinGUIStaticRectType {
	WINGUI_STATIC_RECT_HOLLOW_BLACK = 0,
	WINGUI_STATIC_RECT_HOLLOW_GRAY,
	WINGUI_STATIC_RECT_HOLLOW_WHITE,
	WINGUI_STATIC_RECT_FILLED_BLACK,
	WINGUI_STATIC_RECT_FILLED_GRAY,
	WINGUI_STATIC_RECT_FILLED_WHITE,
};

// Text Properties
enum WinGUIStaticTextAlign {
	WINGUI_STATIC_TEXT_ALIGN_LEFT = 0,
	WINGUI_STATIC_TEXT_ALIGN_RIGHT,
	WINGUI_STATIC_TEXT_ALIGN_CENTER
};
enum WinGUIStaticTextEllipsis {
	WINGUI_STATIC_TEXT_ELLIPSIS_NONE = 0,
	WINGUI_STATIC_TEXT_ELLIPSIS_END,
	WINGUI_STATIC_TEXT_ELLIPSIS_WORD,
	WINGUI_STATIC_TEXT_ELLIPSIS_PATH
};

// Image Properties
enum WinGUIStaticImageInfo {
	WINGUI_STATIC_IMAGE_DEFAULT = 0,
	WINGUI_STATIC_IMAGE_CENTERED,
	WINGUI_STATIC_IMAGE_FIT,
	WINGUI_STATIC_IMAGE_FIT_CENTERED,
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIStaticModel class
class WinGUIStaticModel : public WinGUIControlModel
{
public:
	WinGUIStaticModel( Int iResourceID );
	virtual ~WinGUIStaticModel();

	// View
	virtual const WinGUIRectangle * GetRectangle() const = 0;

	virtual WinGUIStaticType GetType() const = 0;
	virtual const GChar * GetText() const = 0; // Text or Bitmap/Icon Resource Name (not a filename)

	virtual Bool AddSunkenBorder() const = 0;

		// Frame Type
	virtual WinGUIStaticFrameType GetFrameType() const = 0;

		// Rect Type
	virtual WinGUIStaticRectType GetRectType() const = 0;

		// Text Type
	virtual WinGUIStaticTextAlign GetTextAlign() const = 0;
	virtual WinGUIStaticTextEllipsis GetTextEllipsis() const = 0;

		// Image Type (Bitmap/Icon)
	virtual WinGUIStaticImageInfo GetImageInfo() const = 0;

protected:

};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIStatic class
class WinGUIStatic : public WinGUIControl
{
public:
	WinGUIStatic( WinGUIElement * pParent, WinGUIStaticModel * pModel );
	virtual ~WinGUIStatic();

	// Enable / Disable
	Void Enable();
	Void Disable();

	// Text
	UInt GetTextLength() const;
	Void GetText( GChar * outText, UInt iMaxLength ) const;
	Void SetText( const GChar * strText );

	// Icon
	Void * GetIcon() const;       // Returns a HICON
	Void SetIcon( Void * hIcon ); // HICON Resource

private:
	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode );
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIStatic.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUISTATIC_H

