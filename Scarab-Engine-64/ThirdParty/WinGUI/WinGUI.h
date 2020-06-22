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

#include "Controls/WinGUITabs.h"

#include "Controls/WinGUIButton.h"
#include "Controls/WinGUICheckBox.h"
#include "Controls/WinGUIRadioButton.h"
#include "Controls/WinGUIGroupBox.h"

#include "Controls/WinGUIStatic.h"
#include "Controls/WinGUITextEdit.h"

#include "Controls/WinGUIComboBox.h"

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
    WinGUITabs * CreateTabs( WinGUIElement * pParent, WinGUITabsModel * pModel ) const;

    WinGUIButton * CreateButton( WinGUIElement * pParent, WinGUIButtonModel * pModel ) const;
    WinGUICheckBox * CreateCheckBox( WinGUIElement * pParent, WinGUICheckBoxModel * pModel ) const;
    WinGUIRadioButton * CreateRadioButton( WinGUIElement * pParent, WinGUIRadioButtonModel * pModel ) const;
    WinGUIGroupBox * CreateGroupBox( WinGUIElement * pParent, WinGUIGroupBoxModel * pModel ) const;

    WinGUIStatic * CreateStatic( WinGUIElement * pParent, WinGUIStaticModel * pModel ) const;
    WinGUITextEdit * CreateTextEdit( WinGUIElement * pParent, WinGUITextEditModel * pModel ) const;

    WinGUIComboBox * CreateComboBox( WinGUIElement * pParent,WinGUIComboBoxModel * pModel ) const;

    // Element Removal
    Void DestroyElement( WinGUIElement * pElement ) const;

    // Font Infos
    inline UInt GetFontHeight() const;

private:
    // Application Window
    WinGUIWindow * m_pAppWindow;

    // Default Font
    UInt m_iFontWidth;
    UInt m_iFontHeight;
    Void * m_pDefaultFont; // HFONT
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUI.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_WINGUI_H

