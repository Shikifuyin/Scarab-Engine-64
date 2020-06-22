/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUIWindow.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Element : Windows
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
#ifndef SCARAB_THIRDPARTY_WINGUI_WINGUIWINDOW_H
#define SCARAB_THIRDPARTY_WINGUI_WINGUIWINDOW_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "WinGUIElement.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define WINGUI_WINDOW_MAX_CHILDREN 256 // Should be more than enough

// Creation Parameters
typedef struct _wingui_window_parameters {
	WinGUIRectangle hClientRect; // Windows have no parent, they don't use layouts
	GChar strClassName[64];
	GChar strTitle[256];
	Bool bHasSystemMenu;
	Bool bHasMinimizeButton;
	Bool bHasMaximizeButton;
	Bool bAllowResizing;
	Bool bClipChildren;
	Bool bClipSibblings;
} WinGUIWindowParameters;

// Prototypes
class WinGUIWindowModel;
class WinGUIWindow;

class WinGUI;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIContainerModel class
class WinGUIWindowModel : public WinGUIElementModel
{
public:
	WinGUIWindowModel( Int iResourceID );
	virtual ~WinGUIWindowModel();

	// Creation Parameters
	inline const WinGUIWindowParameters * GetCreationParameters() const;

	// Events
	virtual Bool OnClose() = 0; // Must-Implement, Destroy Application Window here

protected:
	WinGUIWindowParameters m_hCreationParameters;
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIWindow class
class WinGUIWindow : public WinGUIElement
{
public:
	WinGUIWindow( WinGUIWindowModel * pModel );
	virtual ~WinGUIWindow();

	// Type
	inline virtual WinGUIElementType GetElementType() const;

	// Children access
	inline UInt GetChildCount() const;
	inline WinGUIElement * GetChild( UInt iIndex ) const;

	WinGUIElement * GetChildByID( Int iResourceID ) const;

private:
	friend class WinGUI;

	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Handling
	static UIntPtr __stdcall _MessageCallback_Static( Void * hHandle, UInt iMessage, UIntPtr wParam, UIntPtr lParam );
    UIntPtr __stdcall _MessageCallback_Virtual( Void * hHandle, UInt iMessage, UIntPtr wParam, UIntPtr lParam );

	// Child Elements
	Void _AppendChild( WinGUIElement * pElement );
	Void _RemoveChild( WinGUIElement * pElement );

	UInt m_iChildCount;
	WinGUIElement * m_arrChildren[WINGUI_WINDOW_MAX_CHILDREN];
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIWindow.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_WINGUIWINDOW_H

