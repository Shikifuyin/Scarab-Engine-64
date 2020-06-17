/////////////////////////////////////////////////////////////////////////////////
// File : Lib/File/XML/XMLDocument.cpp
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
// Includes
#include "XMLDocument.h"

/////////////////////////////////////////////////////////////////////////////////
// XMLDocument implementation
XMLDocument::XMLDocument( const GChar * strTagName, const GChar * strVersion, const GChar * strEncoding ):
    XMLNode( XML_DOCUMENT, strTagName )
{
    _MakeNode();

    // Initialize parent document propagation
    m_pParentDocument = this;

    // Version & Encoding
    StringFn->NCopy( m_strVersion, strVersion, 63 );
    StringFn->NCopy( m_strEncoding, strEncoding, 63 );
}
XMLDocument::~XMLDocument()
{
    // nothing to do
}

/////////////////////////////////////////////////////////////////////////////////

XMLNode * XMLDocument::_Clone( Bool bRecursive ) const
{
    // Create clone
    XMLDocument * pClone = New() XMLDocument( m_strTagName, m_strVersion, m_strEncoding );

    // Copy attributes
    _AttributeMap::Iterator itAttribute = m_mapAttributes.Begin();
    while( !(itAttribute.IsNull()) ) {
        pClone->CreateAttribute( itAttribute.GetKey(), itAttribute.GetItem()->GetValue() );
        ++itAttribute;
    }
    pClone->m_strIdentifierAttribute = m_strIdentifierAttribute;

    // Recurse
    if ( bRecursive ) {
        for( UInt i = 0; i < m_arrChildren.Count(); ++i ) {
            XMLNode * pChildClone = m_arrChildren[i]->Clone( true );
            pClone->AppendChild( pChildClone );
        }
    }

    // Done
    return pClone;
}

Bool XMLDocument::_Render( _XMLWriteCallback pfCallback, Void * pUserData ) const
{
    Assert( m_pParentDocument == this );
    Assert( m_pParent == NULL );
    Assert( m_iDepth == 0 );

    // Document header
    GChar strHeader[256];
    StringFn->Format( strHeader, TEXT("<?xml version=\"%s\" encoding=\"%s\"?>"), m_strVersion, m_strEncoding );
    Bool bContinue = pfCallback( strHeader, pUserData );
    if ( !bContinue )
        return false;

    // XML tree
    return XMLNode::_Render( pfCallback, pUserData );
}

Bool XMLDocument::_Parse_Header( XMLToken * pOverflowToken, GChar * pOverflowChar, _XMLReadCallback pfCallback, Void * pUserData )
{
    GChar strTokenData[XML_NAME_SIZE];

    // Header start
    XMLToken iToken;
    if ( *pOverflowToken == XMLTOKEN_UNDEFINED )
        iToken = _Parse_NextToken( strTokenData, pOverflowChar, pfCallback, pUserData );
    else {
        iToken = *pOverflowToken;
        *pOverflowToken = XMLTOKEN_UNDEFINED;
    }
    if ( iToken == XMLTOKEN_SEPARATOR )
        iToken = _Parse_NextToken( strTokenData, pOverflowChar, pfCallback, pUserData );
    if ( iToken != XMLTOKEN_HEADER_START )
        return false;

    // Attribute list
    Bool bContinue = _Parse_AttributeList( &iToken, pOverflowChar, pfCallback, pUserData );
    if ( !bContinue )
        return false;

    // Header end
    if ( iToken != XMLTOKEN_HEADER_END )
        return false;

    // Extract version & encoding from attributes
    XMLAttribute * pAttribute = GetAttribute( TEXT("version") );
    if ( pAttribute == NULL )
        return false;
    StringFn->NCopy( m_strVersion, pAttribute->GetValue(), 63 );
    DestroyAttribute( TEXT("version") );

    pAttribute = GetAttribute( TEXT("encoding") );
    if ( pAttribute == NULL )
        return false;
    StringFn->NCopy( m_strEncoding, pAttribute->GetValue(), 63 );
    DestroyAttribute( TEXT("encoding") );

    // Done
    Assert( *pOverflowToken == XMLTOKEN_UNDEFINED );
    Assert( *pOverflowChar == NULLBYTE );
    return true;
}

Bool XMLDocument::_Parse( XMLToken * pOverflowToken, GChar * pOverflowChar, _XMLReadCallback pfCallback, Void * pUserData )
{
    // Cleanup
    m_iDepth = 0;

    m_strTagName[0] = NULLBYTE;
    m_strPathName[0] = NULLBYTE;

    m_mapAttributes.Clear();
    m_strIdentifierAttribute = NULL;

    m_pParentDocument = this;
    m_pParent = NULL;
    m_iChildIndex = INVALID_OFFSET;

    m_pPrevSibbling = NULL;
    m_pNextSibbling = NULL;

    for( UInt i = 0; i < m_arrChildren.Count(); ++i )
        Delete( m_arrChildren[i] );
    m_arrChildren.Clear();
    m_mapChildrenID.Clear();

    m_strVersion[0] = NULLBYTE;
    m_strEncoding[0] = NULLBYTE;

    // Document header
    Bool bContinue = _Parse_Header( pOverflowToken, pOverflowChar, pfCallback, pUserData );
    if ( !bContinue )
        return false;

    // XML tree
    return XMLNode::_Parse( pOverflowToken, pOverflowChar, pfCallback, pUserData );
}

