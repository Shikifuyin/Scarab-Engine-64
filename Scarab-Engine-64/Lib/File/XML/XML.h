/////////////////////////////////////////////////////////////////////////////////
// File : Lib/File/XML/XML.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : XML documents parser & writer
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
#ifndef SCARAB_LIB_FILE_XML_XML_H
#define SCARAB_LIB_FILE_XML_XML_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "XMLAttribute.h"
#include "XMLNode.h"

#include "XMLText.h"
#include "XMLComment.h"

#include "XMLDocument.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions
#define XMLFn XMLManager::GetInstance()

// XML specifications
#define XML_VERSION          TEXT("1.0")

#define XML_ENCODING_ASCII   TEXT("us-ascii")
#define XML_ENCODING_UTF8    TEXT("utf-8")
#define XML_ENCODING_UTF16   TEXT("utf-16")
#define XML_ENCODING_ISO     TEXT("iso-8859-1")

#define XML_ENCODING_DEFAULT XML_ENCODING_UTF8

/////////////////////////////////////////////////////////////////////////////////
// The XMLManager class
class XMLManager
{
    // Discrete singleton interface
public:
    inline static XMLManager * GetInstance();

private:
    XMLManager();
    ~XMLManager();

public:
    // Document creation
    XMLDocument * CreateDocument( const GChar * strTagName, const GChar * strVersion = XML_VERSION, const GChar * strEncoding = XML_ENCODING_DEFAULT );
    XMLDocument * CreateDocument( const GChar * strFile );
    XMLDocument * CreateDocumentXML( const GChar * strXML );

    Void DestroyDocument( XMLDocument * pDocument );

    // Node management
    XMLNode * CreateNode( const GChar * strTagName, Bool bLeaf );
    XMLComment * CreateComment( const GChar * strComment = NULL );
    XMLText * CreateText( const GChar * strText = NULL );

    XMLNode * CreateNode( const GChar * strFile );
    XMLNode * CreateNodeXML( const GChar * strXML );

    Void DestroyNode( XMLNode * pNode );
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "XML.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_FILE_XML_XML_H

