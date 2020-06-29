/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIToolTip.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : ToolTip
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
#ifndef SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITOOLTIP_H
#define SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITOOLTIP_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../WinGUIControl.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Creation Parameters
typedef struct _wingui_tooltip_parameters {
	Bool bAlwaysTip;
	Bool bBalloonTip;
	Bool bBalloonCloseButton; // Must have a title
	Bool bNoSlidingAnimation;
	Bool bNoFadingAnimation;
} WinGUIToolTipParameters;

// ToolTip Infos
enum WinGUIToolTipTrackingMode {
	WINGUI_TOOLTIP_TRACKING_DISABLED = 0,
	WINGUI_TOOLTIP_TRACKING_RELATIVE,
	WINGUI_TOOLTIP_TRACKING_ABSOLUTE
};

typedef struct _wingui_tooltip_infos {
	WinGUIElement * pToolElement;
	Void * pUserData;
	WinGUIToolTipTrackingMode iTrackingMode;
	Bool bCentered;
	Bool bForwardMouseEvents;
} WinGUIToolTipInfos;

// ToolTip Icons
enum WinGUIToolTipIcon {
	WINGUI_TOOLTIP_ICON_NONE = 0,
	WINGUI_TOOLTIP_ICON_INFO,
	WINGUI_TOOLTIP_ICON_WARNING,
	WINGUI_TOOLTIP_ICON_ERROR
};

// ToolTip Timers
typedef struct _wingui_tooltip_timers {
	UInt iDelayMS;    // Time (milliseconds) cursor must remain stationary before showing the tooltip
	UInt iDurationMS; // Time (milliseconds) the tooltip stays on screen once displayed
	UInt iIntervalMS; // Time (milliseconds) before another tooltip can be shown when moving cursor
} WinGUIToolTipTimers;

// Prototypes
class WinGUIToolTipModel;
class WinGUIToolTip;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIToolTipModel class
class WinGUIToolTipModel : public WinGUIControlModel
{
public:
	WinGUIToolTipModel( Int iResourceID );
	virtual ~WinGUIToolTipModel();

	// Creation Parameters
	inline const WinGUIToolTipParameters * GetCreationParameters() const;

	// Events
	virtual Bool OnShowTip() { return false; } // Return true if you want to reposition the tooltip window, Return false to use default position
	virtual Bool OnHideTip() { return false; }

	virtual Bool OnLinkClick() { return false; }

	// Callback Events (Must-Implement)
	virtual GChar * OnRequestToolTipText( WinGUIElement * pTool, Void * pUserData ) { return NULL; }

protected:
	WinGUIToolTipParameters m_hCreationParameters;
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIToolTip class
class WinGUIToolTip : public WinGUIControl
{
public:
	WinGUIToolTip( WinGUIElement * pParent, WinGUIToolTipModel * pModel );
	virtual ~WinGUIToolTip();

	// Enable / Disable
	Void Enable();
	Void Disable();

	// Show / Hide
	Void Show(); // At last mouse coords
	Void Hide();

	Void ForceRedraw();

	// Visual Settings (No effect when using visual themes)
	UInt GetBackgroundColor() const;
	Void SetBackgroundColor( UInt iColor );

	UInt GetTextColor() const;
	Void SetTextColor( UInt iColor );

	Void GetMargin( WinGUIPoint * outMarginLeftTop, WinGUIPoint * outMarginRightBottom ) const;
	Void SetMargin( const WinGUIPoint & hMarginLeftTop, const WinGUIPoint & hMarginRightBottom );

	// Metrics
	Void ToolTipRectToTextRect( WinGUIRectangle * pRect ) const;
	Void TextRectToToolTipRect( WinGUIRectangle * pRect ) const;

	UInt GetMaxWidth() const;
	Void SetMaxWidth( UInt iMaxWidth ); // Use INVALID_OFFSET for no max

	// Title
	Void GetTitle( GChar * outTitle, UInt iMaxLength, WinGUIToolTipIcon * outIcon ) const;
	Void SetTitle( const GChar * strTitle, WinGUIToolTipIcon iIcon );

	// Timing
	Void GetTimers( WinGUIToolTipTimers * outTimers ) const;
	Void SetTimers( const WinGUIToolTipTimers & hTimers );

	// Tools
	Void RegisterTool( const WinGUIToolTipInfos & hToolTipInfos );
	Void UnregisterTool( WinGUIElement * pToolElement );

	UInt GetToolCount() const;
	Bool GetTool( WinGUIToolTipInfos * outToolTipInfos, GChar * outToolTipText, UInt iMaxLength, UInt iToolIndex ) const;
	
	Void GetToolSize( WinGUIPoint * outSize, const WinGUIToolTipInfos & hToolTipInfos ) const;
	Void GetToolText( GChar * outToolTipText, UInt iMaxLength, WinGUIElement * pToolElement ) const;

	Bool HasCurrentTool() const;
	Void GetCurrentTool( WinGUIToolTipInfos * outToolTipInfos, GChar * outToolTipText, UInt iMaxLength ) const;

	// Tracking
	Void ToggleTracking( WinGUIElement * pToolElement, Bool bEnable );
	Void SetTrackPosition( const WinGUIPoint & hScreenPosition );

	Bool HitTest( WinGUIToolTipInfos * outToolTipInfos, const WinGUIPoint & hPosition, WinGUIElement * pToolElement ) const;

private:
	// Helpers
	Void _Convert_ToolTipInfos( WinGUIToolTipInfos * outToolTipInfos, const Void * pToolTipInfos ) const;
	Void _Convert_ToolTipInfos( Void * outToolTipInfos, const WinGUIToolTipInfos * pToolTipInfos ) const;

	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode, Void * pParameters );
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIToolTip.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUITOOLTIP_H

