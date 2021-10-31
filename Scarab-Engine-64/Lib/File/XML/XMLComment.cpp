/////////////////////////////////////////////////////////////////////////////////
// File : Lib/File/XML/XMLComment.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : XML Comment Entity
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
#include "XMLComment.h"

/////////////////////////////////////////////////////////////////////////////////
// XMLComment implementation
XMLComment::XMLComment( const GChar * strComment ):
    XMLNode( XML_COMMENT, TEXT("_xml_comment") )
{
    _MakeLeaf();

    m_strComment.Create();
    if ( strComment != NULL ) {
        while( *strComment != NULLBYTE )
            m_strComment.Push( *strComment++ );
    }
    m_strComment.Push( NULLBYTE );
}
XMLComment::~XMLComment()
{
    m_strComment.Destroy();
}

Void XMLComment::SetComment( const GChar * strComment )
{
    m_strComment.Clear();
    if ( strComment != NULL ) {
        while( *strComment != NULLBYTE )
            m_strComment.Push( *strComment++ );
    }
    m_strComment.Push( NULLBYTE );
}

Void XMLComment::AppendComment( const GChar * strComment )
{
    while( *strComment != NULLBYTE )
        m_strComment.Push( *strComment++ );
}
Void XMLComment::InsertComment( const GChar * strComment, UInt iIndex )
{
    Assert( iIndex < m_strComment.Count() );
    UInt iLength = StringFn->Length( strComment );
    m_strComment.Insert( iIndex, strComment, iLength );
}
Void XMLComment::ReplaceComment( const GChar * strComment, UInt iIndex, UInt iLength )
{
    Assert( (iIndex + iLength) < m_strComment.Count() );
    UInt iNewLength = StringFn->Length( strComment );
    m_strComment.Remove( iIndex, NULL, iLength );
    m_strComment.Insert( iIndex, strComment, iNewLength );
}
Void XMLComment::DeleteComment( UInt iIndex, UInt iLength )
{
    Assert( (iIndex + iLength) < m_strComment.Count() );
    m_strComment.Remove( iIndex, NULL, iLength );
}

/////////////////////////////////////////////////////////////////////////////////

XMLNode * XMLComment::_Clone( Bool /*bRecursive*/ ) const
{
    // Create clone
    XMLComment * pClone;
    New( XMLComment, pClone, XMLComment((const GChar *)m_strComment) );

    // Done
    return pClone;
}

Bool XMLComment::_Render( _XMLWriteCallback pfCallback, Void * pUserData ) const
{
    Bool bContinue;
    GChar strIndent[256];
    _GetIndentString( strIndent );

    // Indentation
    bContinue = pfCallback( strIndent, pUserData );
    if ( !bContinue )
        return false;

    // Start comment tag
    bContinue = pfCallback( TEXT("<!-- "), pUserData );
    if ( !bContinue )
        return false;

    // Comment text
    bContinue = pfCallback( (const GChar *)m_strComment, pUserData );
    if ( !bContinue )
        return false;
    
    // Indentation
    if ( m_strComment.Count() > 1 ) {
        bContinue = pfCallback( strIndent, pUserData );
        if ( !bContinue )
            return false;
    }

    // End comment tag
    bContinue = pfCallback( TEXT("-->"), pUserData );
    if ( !bContinue )
        return false;

    return true;
}

Bool XMLComment::_Parse( XMLToken * /*pOverflowToken*/, GChar * /*pOverflowChar*/, _XMLReadCallback /*pfCallback*/, Void * /*pUserData*/ )
{
    // nothing to do
    return true;
}

