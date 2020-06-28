/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIStatusBar.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : StatusBar
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
#ifndef SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUISTATUSBAR_H
#define SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUISTATUSBAR_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../WinGUIControl.h"

#include "../Tools/WinGUIImage.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Creation Parameters
typedef struct _wingui_statusbar_parameters {
	Bool bHasSizingGrip; // Only for resizable parent window
	Bool bEnableToolTips;
} WinGUIStatusBarParameters;

// Draw Modes
enum WinGUIStatusBarDrawMode {
	WINGUI_STATUSBAR_DRAW_NOBORDER = 0,
	WINGUI_STATUSBAR_DRAW_SINKBORDER,
	WINGUI_STATUSBAR_DRAW_RAISEBORDER
};

// Prototypes
class WinGUIStatusBarModel;
class WinGUIStatusBar;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIStatusBarModel class
class WinGUIStatusBarModel : public WinGUIControlModel
{
public:
	WinGUIStatusBarModel( Int iResourceID );
	virtual ~WinGUIStatusBarModel();

	// Creation Parameters
	inline const WinGUIStatusBarParameters * GetCreationParameters() const;

protected:
	WinGUIStatusBarParameters m_hCreationParameters;
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIStatusBar class
class WinGUIStatusBar : public WinGUIControl
{
public:
	WinGUIStatusBar( WinGUIElement * pParent, WinGUIStatusBarModel * pModel );
	virtual ~WinGUIStatusBar();

	// Text format
	Bool IsUnicode() const;
	Bool IsANSI() const;

	Void SetUnicode();
	Void SetANSI();

	// Visual Settings
	UInt SetBackgroundColor( UInt iColor ); // Returns previous color

	Void GetPartIcon( WinGUIIcon * outIcon,  UInt iPartIndex );
	Void SetPartIcon( UInt iPartIndex, const WinGUIIcon * pIcon );

	// Metrics
	Void GetBorders( UInt * outHorizontalBorderWidth, UInt * outVerticalBorderWidth, UInt * outSeparatorsWidth ) const;

	Void SetMinHeight( UInt iMinHeight );

	// Single-Part
	Bool IsSinglePart() const;
	Void SetSinglePart( Bool bSetSinglePart );

	// Multi-Parts
	UInt GetPartCount() const;
	Void GetParts( UInt * outPartEdges, UInt iMaxParts ) const;
	Void SetParts( const UInt * arrPartEdges, UInt iPartCount ); // Use INVALID_OFFSET to extend the last part to parent window border

	Void GetPartRect( WinGUIRectangle * outPartRect, UInt iPartIndex ) const;

	UInt GetPartTextLength( WinGUIStatusBarDrawMode * outDrawMode, UInt iPartIndex ) const;
	Void GetPartText( GChar * outPartText, UInt iPartIndex ) const; // DANGER ! Get the length first !
	Void SetPartText( UInt iPartIndex, const GChar * strPartText, WinGUIStatusBarDrawMode iDrawMode ) const;

	Void GetPartTipText( GChar * outPartTipText, UInt iMaxLength, UInt iPartIndex ) const;
	Void SetPartTipText( UInt iPartIndex, const GChar * strPartTipText ) const;

private:
	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode, Void * pParameters );
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIStatusBar.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUISTATUSBAR_H

