/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUIContainer.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Container Base Interface
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
#ifndef SCARAB_THIRDPARTY_WINGUI_WINGUICONTAINER_H
#define SCARAB_THIRDPARTY_WINGUI_WINGUICONTAINER_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "WinGUIElement.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define WINGUI_CONTAINER_MAX_CHILDREN 256 // Should be more than enough

// Prototypes
class WinGUIContainerModel;
class WinGUIContainer;

class WinGUIControl;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIContainerModel class
class WinGUIContainerModel : public WinGUIElementModel
{
public:
	WinGUIContainerModel( Int iResourceID );
	virtual ~WinGUIContainerModel();

	// Events
	virtual Bool OnClose() = 0;

protected:

};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIContainer class
class WinGUIContainer : public WinGUIElement
{
public:
	WinGUIContainer( WinGUIContainerModel * pModel );
	virtual ~WinGUIContainer();

	// Type
	inline virtual WinGUIElementType GetElementType() const;

	// Parent access
	inline WinGUIContainer * GetParent() const;

	// Children access
	inline UInt GetChildlCount() const;
	inline WinGUIElement * GetChild( UInt iIndex ) const;

	WinGUIElement * GetChildByID( Int iResourceID ) const;

protected: 
	// Parent Container
	WinGUIContainer * m_pParent;

	// Child Elements
	UInt m_iChildCount;
	WinGUIElement * m_arrChildren[WINGUI_CONTAINER_MAX_CHILDREN];

private:
	// Create/Destroy Interface
	virtual Void _Create();
	virtual Void _Destroy();

	// Event Handling
	WinGUIControl * _SearchControl( Int iResourceID ) const;

	static UIntPtr __stdcall _MessageCallback_Static( Void * hHandle, UInt iMessage, UIntPtr wParam, UIntPtr lParam );
    UIntPtr __stdcall _MessageCallback_Virtual( Void * hHandle, UInt iMessage, UIntPtr wParam, UIntPtr lParam );
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIContainer.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_WINGUICONTAINER_H

