/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUIProgressBar.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : ProgressBar
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
#ifndef SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUIPROGRESSBAR_H
#define SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUIPROGRESSBAR_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../WinGUIControl.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Creation Parameters
typedef struct _wingui_progressbar_parameters {
	Bool bPendingMode; // When estimation is not available
	Bool bSmoothWrap;
	Bool bVertical;
} WinGUIProgressBarParameters;

// State
enum WinGUIProgressBarState {
	WINGUI_PROGRESSBAR_INPROGRESS = 0,
	WINGUI_PROGRESSBAR_PAUSED,
	WINGUI_PROGRESSBAR_ERROR
};

// Prototypes
class WinGUIButtonModel;
class WinGUIButton;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIProgressBarModel class
class WinGUIProgressBarModel : public WinGUIControlModel
{
public:
	WinGUIProgressBarModel( Int iResourceID );
	virtual ~WinGUIProgressBarModel();

	// Creation Parameters
	inline const WinGUIProgressBarParameters * GetCreationParameters() const;

protected:
	WinGUIProgressBarParameters m_hCreationParameters;
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIProgressBar class
class WinGUIProgressBar : public WinGUIControl
{
public:
	WinGUIProgressBar( WinGUIElement * pParent, WinGUIProgressBarModel * pModel );
	virtual ~WinGUIProgressBar();

	// Pending Mode
	inline Bool IsPendingMode() const;
	Void TogglePendingMode( Bool bEnable );

	// Visual Settings (no effect when visual styles are enabled)
	UInt GetBackgroundColor() const;
	Void SetBackgroundColor( UInt iColor );

	UInt GetBarColor() const;
	Void SetBarColor( UInt iColor );

	// Range setup
	Void GetRange( Int * outLowerBound, Int * outUpperBound ) const;
	Void SetRange( Int iLowerBound, Int iUpperBound );

	// State
	WinGUIProgressBarState GetState() const;
	WinGUIProgressBarState SetState( WinGUIProgressBarState iState ); // Returns previous state

	// Progress manipulation
	Int GetBarPosition() const;
	Int SetBarPosition( Int iPosition ); // Returns previous position

	Int Progress( Int iDelta ); // Returns previous position

	Int GetStep() const;
	Int SetStep( Int iStep ); // Returns previous step

	Int Step(); // Returns previous position

	Void TogglePendingAnimation( Bool bEnable, UInt iUpdateIntervalMS = 30 ); // Pending mode only, obviously

private:
	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Dispatch
	virtual Bool _DispatchEvent( Int iNotificationCode, Void * pParameters );

	// State
	Bool m_bPendingMode;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIProgressBar.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_CONTROLS_WINGUIPROGRESSBAR_H

