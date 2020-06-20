/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/WinGUIElement.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Element Base Interface
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
#ifndef SCARAB_THIRDPARTY_WINGUI_WINGUIELEMENT_H
#define SCARAB_THIRDPARTY_WINGUI_WINGUIELEMENT_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../System/System.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Element Types
enum WinGUIElementType {
	WINGUI_ELEMENT_WINDOW = 0,
	WINGUI_ELEMENT_CONTAINER,
	WINGUI_ELEMENT_CONTROL
};

// Prototypes
class WinGUIElementModel;
class WinGUIElement;

class WinGUI;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIContainerModel class
class WinGUIElementModel
{
public:
	WinGUIElementModel( Int iResourceID );
	virtual ~WinGUIElementModel();

	inline WinGUIElement * GetView() const;

protected:
	friend class WinGUIElement;

	// Model <-> View linkage
	WinGUIElement * m_pView;

private:
	Int m_iResourceID;
};

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIElement class
class WinGUIElement
{
public:
	WinGUIElement( WinGUIElement * pParent, WinGUIElementModel * pModel );
	virtual ~WinGUIElement();

	// Type
	virtual WinGUIElementType GetElementType() const = 0;

	// Model access
	inline WinGUIElementModel * GetModel() const;

	// Parent access
	inline WinGUIElement * GetParent() const;

protected:
	// Create/Destroy Interface
	friend class WinGUI;
	virtual Void _Create() = 0;
	virtual Void _Destroy() = 0;

	// Model <-> View linkage
	WinGUIElementModel * m_pModel;

	// Parent Element
	WinGUIElement * m_pParent;

	// Windows GUI Handles
	Void _SaveElementToHandle() const; // _Create must always call this !
	static WinGUIElement * _GetElementFromHandle( Void * hHandle );

	inline static Void * _GetHandle( const WinGUIElement * pElement );
	inline static Int _GetResourceID( const WinGUIElement * pElement );

	Void * m_hHandle; // HWND
	Int m_iResourceID;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "WinGUIElement.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_WINGUI_WINGUIELEMENT_H

