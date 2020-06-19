/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Events/WinGUIEventProcessor.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Event Processor
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
#ifndef SCARAB_THIRDPARTY_WINGUI_EVENTS_WINGUIEVENTPROCESSOR_H
#define SCARAB_THIRDPARTY_WINGUI_EVENTS_WINGUIEVENTPROCESSOR_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../../System/System.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Prototypes
class WinGUIEventProcessorModel;
class WinGUIEventProcessor;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIEventProcessorModel interface
class WinGUIEventProcessorModel
{
public:
    WinGUIEventProcessorModel();
	virtual ~WinGUIEventProcessorModel();

    // Controller
    virtual Void OnTick( UInt iTimerID );

    virtual Bool OnKeyPress( KeyCode iKey, GUIEventFlag iFlags );
	virtual Bool OnKeyRelease( KeyCode iKey, GUIEventFlag iFlags );

    virtual Bool OnMousePress( const Point2 & ptLocalPos, KeyCode iKey, GUIEventFlag iFlags );
	virtual Bool OnMouseRelease( const Point2 & ptLocalPos, KeyCode iKey, GUIEventFlag iFlags );

	virtual Bool OnMouseMove( const Point2 & ptLocalPos, GUIEventFlag iFlags );
	virtual Bool OnMouseWheel( const Point2 & ptLocalPos, GUIEventFlag iFlags );

	virtual Bool OnMouseClick( const Point2 & ptLocalPos, KeyCode iKey, GUIEventFlag iFlags );
	virtual Bool OnMouseDblClick( const Point2 & ptLocalPos, KeyCode iKey, GUIEventFlag iFlags );

    virtual Void OnDraw();

    virtual Void OnEnterMoveSize();
    virtual Void OnExitMoveSize();
    virtual Void OnMove();
    virtual Void OnResize();

    virtual Void OnClose();

protected:
    friend class WinGUIEventProcessor;
    WinGUIEventProcessor * m_pEventProcessor;
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIEventProcessor class
class WinGUIEventProcessor
{
public:
    WinGUIEventProcessor( WinGUIEventProcessorModel * pModel );
	virtual ~WinGUIEventProcessor();

    // Getters
    inline WinGUIEventProcessorModel * GetModel() const;

protected:
    WinGUIEventProcessorModel * m_pModel;

private:
    // Event-Handling interface
    friend class WinGUIWindow;
    friend class WinGUI;

    static UIntPtr __stdcall _MessageCallback_Static( Void * hWnd, UInt message, UIntPtr wParam, UIntPtr lParam );
    virtual UIntPtr __stdcall _MessageCallback_Virtual( Void * hWnd, UInt message, UIntPtr wParam, UIntPtr lParam );
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIEventProcessor.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_EVENTS_WINGUIEVENTPROCESSOR_H

