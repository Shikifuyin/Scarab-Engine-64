/////////////////////////////////////////////////////////////////////////////////
// File : Lib/File/XML/XMLDocument.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : XML Document Entity
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
#ifndef SCARAB_LIB_FILE_XML_XMLDOCUMENT_H
#define SCARAB_LIB_FILE_XML_XMLDOCUMENT_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "XMLNode.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Prototypes
class XMLManager;

/////////////////////////////////////////////////////////////////////////////////
// The XMLDocument class
class XMLDocument : public XMLNode
{
protected:
    friend class XMLManager;
    XMLDocument( const GChar * strTagName, const GChar * strVersion, const GChar * strEncoding );
    virtual ~XMLDocument();

public:
    // Version & Encoding
    inline const GChar * GetVersion() const;
    inline const GChar * GetEncoding() const;

    inline Void SetVersion( const GChar * strVersion );
    inline Void SetEncoding( const GChar * strEncoding );

    // Includes & Imports
    //Void ResolveIncludes(); // Lookup all include nodes and replace them with their content
                              // This is recursive of course ...

    // Validation & normalization
    //Bool ValidateDTD() const;

    //Bool ValidateSchema( XMLReadCallback pfCallback, Void * pUserData );

    //Bool ValidateSchema( const GChar * strFile ) const;
    //Bool ValidateSchemaXML( const GChar * strXML ) const;

    //Void Normalize();

protected:
    // Cloning
	virtual XMLNode * _Clone( Bool bRecursive ) const;

    // Rendering
    virtual Bool _Render( _XMLWriteCallback pfCallback, Void * pUserData ) const;

    // Parsing
    Bool _Parse_Header( XMLToken * pOverflowToken, GChar * pOverflowChar, _XMLReadCallback pfCallback, Void * pUserData );

    virtual Bool _Parse( XMLToken * pOverflowToken, GChar * pOverflowChar, _XMLReadCallback pfCallback, Void * pUserData );

    // Version & Encoding
    GChar m_strVersion[64];
    GChar m_strEncoding[64];
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "XMLDocument.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_FILE_XML_XMLDOCUMENT_H

