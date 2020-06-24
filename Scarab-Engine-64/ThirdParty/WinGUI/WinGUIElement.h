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
#include "WinGUILayout.h"

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

class WinGUIImage;

/////////////////////////////////////////////////////////////////////////////////
// The WinGUIContainerModel class
class WinGUIElementModel
{
public:
	WinGUIElementModel( Int iResourceID );
	virtual ~WinGUIElementModel();

	// Controller access
	inline WinGUIElement * GetController() const;

	// Layout access
	virtual const WinGUILayout * GetLayout() const = 0;

protected:
	friend class WinGUIElement;

	// Resource Identifier
	Int m_iResourceID;

	// Controller
	WinGUIElement * m_pController;
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

	// Visibility
	Bool IsVisible() const;
	Void SetVisible( Bool bVisible );

	// Area access
	Void GetWindowRect( WinGUIRectangle * outRectangle ) const;
	Void GetClientRect( WinGUIRectangle * outRectangle ) const;

protected:
	friend class WinGUI;
	friend class WinGUIImage;

	// Create/Destroy Interface
	virtual Void _Create() = 0;
	virtual Void _Destroy() = 0;

	virtual Void _ApplyDefaultFont( Void * hFont );

	// Model
	WinGUIElementModel * m_pModel;

	// Parent Element
	WinGUIElement * m_pParent;

	// Windows Handles
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

