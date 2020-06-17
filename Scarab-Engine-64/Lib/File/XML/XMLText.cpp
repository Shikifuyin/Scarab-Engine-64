/////////////////////////////////////////////////////////////////////////////////
// File : Lib/File/XML/XMLText.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : XML Text Entity
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
#include "XMLText.h"

/////////////////////////////////////////////////////////////////////////////////
// XMLText implementation
XMLText::XMLText( const GChar * strText ):
    XMLNode( XML_TEXT, TEXT("_xml_text") )
{
    _MakeLeaf();

    m_strText.Create();
    if ( strText != NULL ) {
        while( *strText != NULLBYTE )
            m_strText.Push( *strText++ );
    }
    m_strText.Push( NULLBYTE );
}
XMLText::~XMLText()
{
    m_strText.Destroy();
}

Void XMLText::SetText( const GChar * strText )
{
    m_strText.Clear();
    if ( strText != NULL ) {
        while( *strText != NULLBYTE )
            m_strText.Push( *strText++ );
    }
    m_strText.Push( NULLBYTE );
}

Void XMLText::AppendText( const GChar * strText )
{
    while( *strText != NULLBYTE )
        m_strText.Push( *strText++ );
}
Void XMLText::InsertText( const GChar * strText, UInt iIndex )
{
    Assert( iIndex < m_strText.Count() );
    UInt iLength = StringFn->Length( strText );
    m_strText.Insert( iIndex, strText, iLength );
}
Void XMLText::ReplaceText( const GChar * strText, UInt iIndex, UInt iLength )
{
    Assert( (iIndex + iLength) < m_strText.Count() );
    UInt iNewLength = StringFn->Length( strText );
    m_strText.Remove( iIndex, NULL, iLength );
    m_strText.Insert( iIndex, strText, iNewLength );
}
Void XMLText::DeleteText( UInt iIndex, UInt iLength )
{
    Assert( (iIndex + iLength) < m_strText.Count() );
    m_strText.Remove( iIndex, NULL, iLength );
}

/////////////////////////////////////////////////////////////////////////////////

XMLNode * XMLText::_Clone( Bool /*bRecursive*/ ) const
{
    // Create clone
    XMLText * pClone = New() XMLText( (const GChar *)m_strText );

    // Done
    return pClone;
}

Bool XMLText::_Render( _XMLWriteCallback pfCallback, Void * pUserData ) const
{
    Bool bContinue;
    GChar strIndent[256];
    _GetIndentString( strIndent );

    // Indentation
    bContinue = pfCallback( strIndent, pUserData );
    if ( !bContinue )
        return false;

    // Text
    bContinue = pfCallback( (const GChar *)m_strText, pUserData );
    if ( !bContinue )
        return false;
    
    return true;
}

Bool XMLText::_Parse( XMLToken * /*pOverflowToken*/, GChar * /*pOverflowChar*/, _XMLReadCallback /*pfCallback*/, Void * /*pUserData*/ )
{
    // nothing to do
    return true;
}


