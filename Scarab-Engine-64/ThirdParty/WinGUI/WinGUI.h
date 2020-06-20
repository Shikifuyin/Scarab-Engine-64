/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUI.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Main Interface
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
#ifndef SCARAB_THIRDPARTY_WINGUI_WINGUI_H
#define SCARAB_THIRDPARTY_WINGUI_WINGUI_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../System/System.h"

#include "WinGUIWindow.h"
#include "WinGUIContainer.h"
#include "WinGUIControl.h"

#include "Controls/WinGUIButton.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define WinGUIFn WinGUI::GetInstance()

// Message Boxes
enum WinGUIMessageBoxType {
    WINGUI_MESSAGEBOX_OK = 0,
    WINGUI_MESSAGEBOX_OKCANCEL,
    WINGUI_MESSAGEBOX_YESNO,
    WINGUI_MESSAGEBOX_YESNOCANCEL,
    WINGUI_MESSAGEBOX_RETRYCANCEL
};
enum WinGUIMessageBoxIcon {
    WINGUI_MESSAGEBOX_ICON_NONE = 0,
    WINGUI_MESSAGEBOX_ICON_INFO,
    WINGUI_MESSAGEBOX_ICON_WARNING,
    WINGUI_MESSAGEBOX_ICON_ERROR
};
enum WinGUIMessageBoxResponse {
    WINGUI_MESSAGEBOX_RESPONSE_CANCEL = 0,
    WINGUI_MESSAGEBOX_RESPONSE_OK,
    WINGUI_MESSAGEBOX_RESPONSE_YES,
    WINGUI_MESSAGEBOX_RESPONSE_NO,
    WINGUI_MESSAGEBOX_RESPONSE_RETRY
};
typedef struct _wingui_message_box_options {
    WinGUIMessageBoxType iType;
    WinGUIMessageBoxIcon iIcon;
    WinGUIMessageBoxResponse iDefaultResponse;
    Bool bMustAnswer;
} WinGUIMessageBoxOptions;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUI class
class WinGUI
{
    // Discrete singleton interface
public:
    inline static WinGUI * GetInstance();

private:
    WinGUI();
    ~WinGUI();

public:

    // Application Window access
    inline WinGUIWindow * GetAppWindow() const;

    Void CreateAppWindow( WinGUIWindowModel * pModel );
    Void DestroyAppWindow();

    // Message Loop
    Int MessageLoop() const;

    // Message Boxes
    WinGUIMessageBoxResponse SpawnMessageBox( const GChar * strTitle, const GChar * strText, const WinGUIMessageBoxOptions & hOptions ) const;

    // Containers
    WinGUIContainer * CreateContainer( WinGUIElement * pParent, WinGUIContainerModel * pModel ) const;

    // Controls
    WinGUIButton * CreateButton( WinGUIElement * pParent, WinGUIButtonModel * pModel ) const;

    // Element Removal
    Void DestroyElement( WinGUIElement * pElement ) const;

private:
    // Application Window
    WinGUIWindow * m_pAppWindow;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUI.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_WINGUI_H

