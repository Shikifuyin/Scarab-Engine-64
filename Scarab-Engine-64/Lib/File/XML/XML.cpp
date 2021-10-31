/////////////////////////////////////////////////////////////////////////////////
// File : Lib/File/XML/XML.inl
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
// Includes
#include "XML.h"

/////////////////////////////////////////////////////////////////////////////////
// XMLManager implementation
XMLManager::XMLManager()
{
	// nothing to do
}
XMLManager::~XMLManager()
{
	// nothing to do
}

XMLDocument * XMLManager::CreateDocument( const GChar * strTagName, const GChar * strVersion, const GChar * strEncoding )
{
    XMLDocument * pDocument;
    New( XMLDocument, pDocument,  XMLDocument(strTagName, strVersion, strEncoding) );

    return pDocument;
}
XMLDocument * XMLManager::CreateDocument( const GChar * strFile )
{
    XMLDocument * pDocument;
    New( XMLDocument, pDocument,  XMLDocument(TEXT("_xmlnew"), XML_VERSION, XML_ENCODING_DEFAULT) );

    Bool bValid = pDocument->Parse( strFile );
    Assert( bValid );

    return pDocument;
}
XMLDocument * XMLManager::CreateDocumentXML( const GChar * strXML )
{
    XMLDocument * pDocument;
    New( XMLDocument, pDocument,  XMLDocument(TEXT("_xmlnew"), XML_VERSION, XML_ENCODING_DEFAULT) );

    Bool bValid = pDocument->ParseXML( strXML );
    Assert( bValid );

    return pDocument;
}

Void XMLManager::DestroyDocument( XMLDocument * pDocument )
{
    Delete( pDocument );
}

XMLNode * XMLManager::CreateNode( const GChar * strTagName, Bool bLeaf )
{
    XMLNode * pNode;
    New( XMLNode, pNode, XMLNode(XML_NODE, strTagName) );

    if ( bLeaf )
        pNode->_MakeLeaf();
    else
        pNode->_MakeNode();

    return pNode;
}
XMLComment * XMLManager::CreateComment( const GChar * strComment )
{
    XMLComment * pComment;
    New( XMLComment, pComment, XMLComment(strComment) );

    return pComment;
}
XMLText * XMLManager::CreateText( const GChar * strText )
{
    XMLText * pText;
    New( XMLText, pText, XMLText(strText) );

    return pText;
}

XMLNode * XMLManager::CreateNode( const GChar * strFile )
{
    XMLNode * pNode;
    New( XMLNode, pNode, XMLNode(XML_NODE, TEXT("_xmlnew")) );

    Bool bValid = pNode->Parse( strFile );
    Assert( bValid );

    return pNode;
}
XMLNode * XMLManager::CreateNodeXML( const GChar * strXML )
{
    XMLNode * pNode;
    New( XMLNode, pNode, XMLNode(XML_NODE, TEXT("_xmlnew")) );

    Bool bValid = pNode->ParseXML( strXML );
    Assert( bValid );

    return pNode;
}

Void XMLManager::DestroyNode( XMLNode * pNode )
{
    Assert( pNode->IsRoot() );

    Delete( pNode );
}

