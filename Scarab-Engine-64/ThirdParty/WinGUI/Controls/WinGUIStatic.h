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

// Creation Parameters
typedef struct _wingui_static_parameters {
	WinGUIStaticType iType;
	Bool bAddSunkenBorder;
	union {
		struct _frame {
			WinGUIStaticFrameType iFrameType;
		} hFrame;
		struct _rect {
			WinGUIStaticRectType iRectType;
		} hRect;
		struct _text {
			GChar strLabel[64];
			WinGUIStaticTextAlign iAlign;
			WinGUIStaticTextEllipsis iEllipsis;
		} hText;
		struct _bitmap {
			GChar strResourceName[64]; // NOT a filename
			WinGUIStaticImageInfo iInfos;
		} hBitmap;
		struct _icon {
			GChar strResourceName[64]; // NOT a filename
			WinGUIStaticImageInfo iInfos;
		} hIcon;
	};
} WinGUIStaticParameters;

// Prototypes
class WinGUIStaticModel;
class WinGUIStatic;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIStaticModel class
class WinGUIStaticModel : public WinGUIControlModel
{
public:
	WinGUIStaticModel( Int iResourceID );
	virtual ~WinGUIStaticModel();

	// Creation Parameters
	inline const WinGUIStaticParameters * GetCreationParameters() const;

protected:
	WinGUIStaticParameters m_hCreationParameters;
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

	// Label Text
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

