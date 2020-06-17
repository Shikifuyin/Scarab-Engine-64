/////////////////////////////////////////////////////////////////////////////////
// File : Lib/File/XML/XMLComment.h
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
// Header prelude
#ifndef SCARAB_LIB_FILE_XML_XMLCOMMENT_H
#define SCARAB_LIB_FILE_XML_XMLCOMMENT_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "XMLNode.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Prototypes
class XMLManager;

/////////////////////////////////////////////////////////////////////////////////
// The XMLComment class
class XMLComment : public XMLNode
{
protected:
    friend class XMLNode;
    friend class XMLManager;
    XMLComment( const GChar * strComment = NULL );
public:
    virtual ~XMLComment();

    // Comment access
    inline const GChar * GetComment( UInt * outLength = NULL ) const;
    inline Array<GChar> * EditComment();
    Void SetComment( const GChar * strComment );

    Void AppendComment( const GChar * strComment );
    Void InsertComment( const GChar * strComment, UInt iIndex );
    Void ReplaceComment( const GChar * strComment, UInt iIndex, UInt iLength );
    Void DeleteComment( UInt iIndex, UInt iLength );

protected:
    // Cloning
	virtual XMLNode * _Clone( Bool bRecursive ) const;

    // Rendering
    virtual Bool _Render( _XMLWriteCallback pfCallback, Void * pUserData ) const;

    // Parsing
    virtual Bool _Parse( XMLToken * pOverflowToken, GChar * pOverflowChar, _XMLReadCallback pfCallback, Void * pUserData );

    // Comment
    Array<GChar> m_strComment;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "XMLComment.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_FILE_XML_XMLCOMMENT_H

