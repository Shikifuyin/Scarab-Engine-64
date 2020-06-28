/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUISliderBar.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : SliderBar
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
#ifndef SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUISLIDERBAR_H
#define SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUISLIDERBAR_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../WinGUIControl.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Tick Marks Position
enum WinGUISliderBarTickMarks {
	WINGUI_SLIDERBAR_TICKMARKS_NONE = 0,
	WINGUI_SLIDERBAR_TICKMARKS_BELOWRIGHT,
	WINGUI_SLIDERBAR_TICKMARKS_ABOVELEFT,
	WINGUI_SLIDERBAR_TICKMARKS_BOTH
};

// Creation Parameters
typedef struct _wingui_sliderbar_parameters {
	GChar strLabel[64];
	Bool bTransparentBackground;
	WinGUISliderBarTickMarks iTickMarks;

	Bool bAllowSelectionRange;
	Bool bResizableSlider;
	Bool bNoSlider;

	Bool bReversedValues; // No visual effect, just a tag
	Bool bReverseEdgeMapping; // Down is Left / Up is Right

	Bool bVertical;

	Bool bEnableToolTips;
} WinGUISliderBarParameters;

// Prototypes
class WinGUISliderBarModel;
class WinGUISliderBar;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUISliderBarModel class
class WinGUISliderBarModel : public WinGUIControlModel
{
public:
	WinGUISliderBarModel( Int iResourceID );
	virtual ~WinGUISliderBarModel();

	// Creation Parameters
	inline const WinGUISliderBarParameters * GetCreationParameters() const;

	// Events
	virtual Bool OnSlideToMin( UInt iPosition ) { return false; }
	virtual Bool OnSlideToMax( UInt iPosition ) { return false; }

	virtual Bool OnSlideSmallAmount( UInt iPosition, Int iDirection ) { return false; }
	virtual Bool OnSlideLargeAmount( UInt iPosition, Int iDirection ) { return false; }

	virtual Bool OnSlide( UInt iPosition ) { return false; } // Progressive, Mouse Drag

	virtual Bool OnSlideEnd( UInt iPosition, Bool bWasProgressive ) { return false; }

protected:
	WinGUISliderBarParameters m_hCreationParameters;
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUITrackBar class
class WinGUISliderBar : public WinGUIControl
{
public:
	WinGUISliderBar( WinGUIElement * pParent, WinGUISliderBarModel * pModel );
	virtual ~WinGUISliderBar();

	// Text format
	Bool IsUnicode() const;
	Bool IsANSI() const;

	Void SetUnicode();
	Void SetANSI();

	// Metrics
	Void GetTrackRect( WinGUIRectangle * outTrackRect ) const;
	Void GetSliderRect( WinGUIRectangle * outSliderRect ) const;

	UInt GetSliderWidth() const;
	Void SetSliderWidth( UInt iWidth ); // Requires bResizableSlider

	// Range
	Void GetRange( UInt * outMinPosition, UInt * outMaxPosition ) const;
	Void SetRangeMin( UInt iMinPosition, Bool bRedraw );
	Void SetRangeMax( UInt iMaxPosition, Bool bRedraw );

	// Slide Amounts
	UInt GetSmallSlideAmount() const; // ArrowKeys
	Void SetSmallSlideAmount( UInt iDeltaPos ) const;

	UInt GetLargeSlideAmount() const; // Clicks / PageUp,PageDown / ...
	Void SetLargeSlideAmount( UInt iDeltaPos ) const;

	// Slider Position
	UInt GetSliderPosition() const;
	Void SetSliderPosition( UInt iPosition, Bool bRedraw ) const;

	// TickMarks (Inner Marks, First and Last are set on creation and cannot be accessed)

		// Requires iTickMarks != WINGUI_SLIDERBAR_TICKMARKS_NONE
	Void SetTickMarksFrequency( UInt iFrequency );

		// 0 if using WINGUI_SLIDERBAR_TICKMARKS_NONE
		// Else (RangeMax - RangeMin) / Frequency (Doesn't include end-points)
	UInt GetTickMarksCount() const;

	UInt GetTickMarkPosition( UInt iTickMark ) const;
	UInt * GetTickMarkPositions() const; // Valid pointer until tick marks are changed, Ordering is unspecified !

		// In physical units, x coord if horizontal or y coord if vertical, relative to client area
	UInt GetTickMarkClientPos( UInt iTickMark ) const; 

	Void AddTickMark( UInt iPosition );
	Void ClearTickMarks( Bool bRedraw );

	// Selection (bAllowSelectionRange only)
	Void GetSelectionRange( UInt * outSelectionStart, UInt * outSelectionEnd ) const;
	Void SetSelectionStart( UInt iStartPosition, Bool bRedraw );
	Void SetSelectionEnd( UInt iEndPosition, Bool bRedraw );

	Void ClearSelection( Bool bRedraw );

	// Buddy Windows
	//TBM_GETBUDDY
	//TBM_SETBUDDY

	// Tool Tips
	//TBM_GETTOOLTIPS
	//TBM_SETTOOLTIPS
	//TBM_SETTIPSIDE

private:
	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode, Void * pParameters );
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUISliderBar.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUISLIDERBAR_H

