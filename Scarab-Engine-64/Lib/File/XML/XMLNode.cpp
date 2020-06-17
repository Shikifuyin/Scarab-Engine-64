/////////////////////////////////////////////////////////////////////////////////
// File : Lib/File/XML/XMLNode.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : XML Node Entity
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
#include "XMLNode.h"
#include "XMLText.h"
#include "XMLComment.h"

/////////////////////////////////////////////////////////////////////////////////
// XMLNode implementation
XMLNode::XMLNode( XMLNodeType iType, const GChar * strTagName ):
    m_mapAttributes(), m_itEnumerate(), m_arrChildren(), m_mapChildrenID()
{
    // Type & Depth
    m_iType = iType;
    m_iDepth = 0;

    // Path & Name
    StringFn->NCopy( m_strTagName, strTagName, XML_NAME_SIZE - 1 );
    m_strPathName[0] = TEXT('/');
    StringFn->NCopy( m_strPathName + 1, m_strTagName, XML_URI_SIZE - 2 );

    // Attributes
    m_mapAttributes.SetComparator( _Compare_Strings, NULL );
    m_mapAttributes.Create();

    m_strIdentifierAttribute = NULL;

    // Linkage
    m_pParentDocument = NULL;
    m_pParent = NULL;
    m_iChildIndex = INVALID_OFFSET;

    m_pPrevSibbling = NULL;
    m_pNextSibbling = NULL;

    m_bIsNode = false;
}
XMLNode::~XMLNode()
{
    if ( m_bIsNode ) {
        // Recurse
        for( UInt i = 0; i < m_arrChildren.Count(); ++i )
            Delete( m_arrChildren[i] );

        // Linkage
        m_mapChildrenID.Destroy();
        m_arrChildren.Destroy();
    }

    // Attributes
    m_mapAttributes.Destroy();
}

XMLAttribute * XMLNode::CreateAttribute( const GChar * strName, const GChar * strValue )
{
    Assert( m_iType <= XML_DOCUMENT );

    XMLAttribute * pAttribute = NULL;

    _AttributeMap::Iterator itAttribute = m_mapAttributes.Get( strName );
    if ( itAttribute.IsNull() ) {
        pAttribute = New() XMLAttribute( strName, strValue );

        Bool bInserted = m_mapAttributes.Insert( pAttribute->GetName(), pAttribute );
        Assert( bInserted );
    } else
        pAttribute = itAttribute.GetItem();

    pAttribute->SetValue( strValue );
    return pAttribute;
}
Void XMLNode::DestroyAttribute( const GChar * strName )
{
    _AttributeMap::Iterator itAttribute = m_mapAttributes.Get( strName );
    if ( itAttribute.IsNull() )
        return;
    Bool bRemoved = m_mapAttributes.Remove( strName );
    Assert( bRemoved );
}

Void XMLNode::SetIdentifierAttribute( const GChar * strName )
{
    _AttributeMap::Iterator itAttribute;
    
    if ( m_strIdentifierAttribute != NULL ) {
        itAttribute = m_mapAttributes.Get( m_strIdentifierAttribute );
        Assert( !(itAttribute.IsNull()) );
        itAttribute.GetItem()->m_bIdentifier = false;
    }
    
    if ( strName != NULL ) {
        itAttribute = m_mapAttributes.Get( strName );
        Assert( !(itAttribute.IsNull()) );
        itAttribute.GetItem()->m_bIdentifier = true;
    }

    m_strIdentifierAttribute = strName;
}

XMLNode * XMLNode::GetChildByTag( const GChar * strTagName, UInt iOffset, UInt * outIndex ) const
{
    Assert( m_bIsNode );
    Assert( iOffset < m_arrChildren.Count() );

    for( UInt i = iOffset; i < m_arrChildren.Count(); ++i ) {
        if ( _Compare_Strings(strTagName,m_arrChildren[i]->m_strTagName,NULL) == 0 ) {
            if ( outIndex != NULL )
                *outIndex = i;
            return m_arrChildren[i];
        }
    }

    if ( outIndex != NULL )
        *outIndex = INVALID_OFFSET;
    return NULL;
}
XMLNode * XMLNode::GetChildNByTag( const GChar * strTagName, UInt iOccurence ) const
{
    Assert( m_bIsNode );
    UInt iCount = ( iOccurence + 1 );
    UInt iOffset = 0, iIndex = 0;
    XMLNode * pNode = NULL;

    for( UInt i = 0; i < iCount; ++i ) {
        pNode = GetChildByTag( strTagName, iOffset, &iIndex );
        if ( pNode == NULL )
            break;
        iOffset = iIndex + 1;
    }
    return pNode;
}
XMLNode * XMLNode::GetChildByTagPath( const GChar * strTagPath ) const
{
    Assert( m_bIsNode );
    GChar strTag[XML_NAME_SIZE];
    GChar strOccurence[8];
    GChar * pStr;

    // Skip heading slash
    if ( *strTagPath == TEXT('/') )
        ++strTagPath;

    // Empty path case
    if ( *strTagPath == NULLBYTE )
        return NULL;

    // Extract tag
    strTag[0] = NULLBYTE;
    strOccurence[0] = NULLBYTE;
    pStr = strTag;

    while( true ) {
        if ( *strTagPath == TEXT('/') || *strTagPath == NULLBYTE ) {
            *pStr = NULLBYTE;
            break;
        }
        if ( *strTagPath == TEXT('#') ) {
            *pStr = NULLBYTE;
            pStr = strOccurence;
            ++strTagPath;
            continue;
        }
        *pStr++ = *strTagPath++;
    }

    // Get Child
    UInt iOccurence = (UInt)( StringFn->ToUInt(strOccurence) );
    XMLNode * pChild = GetChildNByTag( strTag, iOccurence );
        
    // Child not found case
    if ( pChild == NULL )
        return NULL;

    // Leaf child case
    if ( pChild->IsLeaf() )
        return pChild;

    // End of path case
    if ( *strTagPath == NULLBYTE )
        return pChild;

    // Trailing slash case
    if ( *(strTagPath+1) == NULLBYTE )
        return pChild;

    // Recurse
    return pChild->GetChildByTagPath( strTagPath );
}

XMLNode * XMLNode::GetChildByAttribute( const GChar * strName, const GChar * strValue, UInt iOffset, UInt * outIndex ) const
{
    Assert( m_bIsNode );
    Assert( iOffset < m_arrChildren.Count() );

    for( UInt i = iOffset; i < m_arrChildren.Count(); ++i ) {
        XMLAttribute * pAttribute = m_arrChildren[i]->GetAttribute( strName );
        if ( pAttribute != NULL ) {
            if ( _Compare_Strings(strValue,pAttribute->GetValue(),NULL) == 0 ) {
                if ( outIndex != NULL )
                    *outIndex = i;
                return m_arrChildren[i];
            }
        }
    }

    if ( outIndex != NULL )
        *outIndex = INVALID_OFFSET;
    return NULL;
}
XMLNode * XMLNode::GetChildNByAttribute( const GChar * strName, const GChar * strValue, UInt iOccurence ) const
{
    Assert( m_bIsNode );
    UInt iCount = ( iOccurence + 1 );
    UInt iOffset = 0, iIndex = 0;
    XMLNode * pNode = NULL;

    for( UInt i = 0; i < iCount; ++i ) {
        pNode = GetChildByAttribute( strName, strValue, iOffset, &iIndex );
        if ( pNode == NULL )
            break;
        iOffset = iIndex + 1;
    }
    return pNode;
}
XMLNode * XMLNode::GetChildByAttributePath( const GChar * strAttributePath ) const
{
    Assert( m_bIsNode );
    GChar strName[XML_NAME_SIZE];
    GChar strValue[XML_NAME_SIZE];
    GChar strOccurence[8];
    GChar * pStr;

    // Skip heading slash
    if ( *strAttributePath == TEXT('/') )
        ++strAttributePath;

    // Empty path case
    if ( *strAttributePath == NULLBYTE )
        return NULL;

    // Extract attribute
    strName[0] = NULLBYTE;
    strValue[0] = NULLBYTE;
    strOccurence[0] = NULLBYTE;
    pStr = strName;

    while( true ) {
        if ( *strAttributePath == TEXT('/') || *strAttributePath == NULLBYTE ) {
            *pStr = NULLBYTE;
            break;
        }
        if ( *strAttributePath == TEXT(':') ) {
            *pStr = NULLBYTE;
            pStr = strValue;
            ++strAttributePath;
            continue;
        }
        if ( *strAttributePath == TEXT('#') ) {
            *pStr = NULLBYTE;
            pStr = strOccurence;
            ++strAttributePath;
            continue;
        }
        *pStr++ = *strAttributePath++;
    }

    // Get Child
    UInt iOccurence = (UInt)( StringFn->ToUInt(strOccurence) );
    XMLNode * pChild = GetChildNByAttribute( strName, strValue, iOccurence );
        
    // Child not found case
    if ( pChild == NULL )
        return NULL;

    // Leaf child case
    if ( pChild->IsLeaf() )
        return pChild;

    // End of path case
    if ( *strAttributePath == NULLBYTE )
        return pChild;

    // Trailing slash case
    if ( *(strAttributePath+1) == NULLBYTE )
        return pChild;

    // Recurse
    return pChild->GetChildByAttributePath( strAttributePath );
}

XMLNode * XMLNode::GetChildByIDPath( const GChar * strIDPath ) const
{
    Assert( m_bIsNode );
    GChar strID[XML_NAME_SIZE];
    GChar * pStr;

    // Skip heading slash
    if ( *strIDPath == TEXT('/') )
        ++strIDPath;

    // Empty path case
    if ( *strIDPath == NULLBYTE )
        return NULL;

    // Extract ID
    strID[0] = NULLBYTE;
    pStr = strID;

    while( true ) {
        if ( *strIDPath == TEXT('/') || *strIDPath == NULLBYTE ) {
            *pStr = NULLBYTE;
            break;
        }
        *pStr++ = *strIDPath++;
    }

    // Get Child
    XMLNode * pChild = GetChildByID( strID );
        
    // Child not found case
    if ( pChild == NULL )
        return NULL;

    // Leaf child case
    if ( pChild->IsLeaf() )
        return pChild;

    // End of path case
    if ( *strIDPath == NULLBYTE )
        return pChild;

    // Trailing slash case
    if ( *(strIDPath+1) == NULLBYTE )
        return pChild;

    // Recurse
    return pChild->GetChildByIDPath( strIDPath );
}

Void XMLNode::GetChildText( Array<GChar> * outBuffer ) const
{
    Assert( m_bIsNode );

    outBuffer->Clear();
    for( UInt i = 0; i < m_arrChildren.Count(); ++i ) {
        if ( m_arrChildren[i]->m_iType == XML_TEXT ) {
            XMLText * pText = (XMLText*)( m_arrChildren[i] );
            const Array<GChar> * strText = pText->EditText();
            outBuffer->Push( *strText );
        }
    }
    outBuffer->Push( NULLBYTE );
}
Void XMLNode::GetChildText( GChar * outBuffer, UInt iSize ) const
{
    Assert( m_bIsNode );

    UInt iLength = 0;
    for( UInt i = 0; i < m_arrChildren.Count(); ++i ) {
        if ( m_arrChildren[i]->m_iType == XML_TEXT ) {
            XMLText * pText = (XMLText*)( m_arrChildren[i] );
            UInt iTextLength = 0;
            const GChar * strText = pText->GetText( &iTextLength );

            iLength += iTextLength;
            if ( (iLength + 1) > iSize )
                break;

            outBuffer = StringFn->NCopy( outBuffer, strText, iSize - (iLength + 1) );
        }
    }
    *outBuffer = NULLBYTE;
}

UInt XMLNode::AppendChild( XMLNode * pNode )
{
    Assert( m_bIsNode );
    Assert( pNode->m_iType != XML_DOCUMENT );
    UInt iIndex = m_arrChildren.Count();

    // Depth
    pNode->_rec_SetDepth( m_iDepth + 1 );

    // Path names
    StringFn->Format( pNode->m_strPathName, TEXT("%s/%s"), m_strPathName, pNode->m_strTagName );

    // Parent linkage
    pNode->_rec_SetParentDocument( m_pParentDocument );
    pNode->m_pParent = this;
    pNode->m_iChildIndex = iIndex;

    // Child linkage
    m_arrChildren.Push( pNode );

    if ( m_strIdentifierAttribute != NULL ) {
        XMLAttribute * pIDAttribute = pNode->GetAttribute( m_strIdentifierAttribute );
        if ( pIDAttribute != NULL ) {
            pNode->SetIdentifierAttribute( m_strIdentifierAttribute );
            Bool bInserted = m_mapChildrenID.Insert( pIDAttribute->GetValue(), pNode );
            Assert( bInserted );
        }
    }

    // Sibbling linkage
    pNode->m_pPrevSibbling = NULL;
    pNode->m_pNextSibbling = NULL;
    if ( iIndex > 0 ) {
        XMLNode * pPrevNode = m_arrChildren[iIndex - 1];
        pPrevNode->m_pNextSibbling = pNode;
        pNode->m_pPrevSibbling = pPrevNode;
    }

    return iIndex;
}
Void XMLNode::InsertChild( XMLNode * pNode, UInt iIndex )
{
    Assert( m_bIsNode );
    Assert( pNode->m_iType != XML_DOCUMENT );
    Assert( iIndex <= m_arrChildren.Count() );

    // Depth
    pNode->_rec_SetDepth( m_iDepth + 1 );

    // Path names
    StringFn->Format( pNode->m_strPathName, TEXT("%s/%s"), m_strPathName, pNode->m_strTagName );

    // Parent linkage
    pNode->_rec_SetParentDocument( m_pParentDocument );
    pNode->m_pParent = this;
    pNode->m_iChildIndex = iIndex;

    // Child linkage
    m_arrChildren.Insert( iIndex, pNode );
    for( UInt i = iIndex + 1; i < m_arrChildren.Count(); ++i )
        ++(m_arrChildren[i]->m_iChildIndex);

    if ( m_strIdentifierAttribute != NULL ) {
        XMLAttribute * pIDAttribute = pNode->GetAttribute( m_strIdentifierAttribute );
        if ( pIDAttribute != NULL ) {
            pNode->SetIdentifierAttribute( m_strIdentifierAttribute );
            Bool bInserted = m_mapChildrenID.Insert( pIDAttribute->GetValue(), pNode );
            Assert( bInserted );
        }
    }

    // Sibbling linkage
    pNode->m_pPrevSibbling = NULL;
    pNode->m_pNextSibbling = NULL;
    if ( iIndex > 0 ) {
        XMLNode * pPrevNode = m_arrChildren[iIndex - 1];
        pPrevNode->m_pNextSibbling = pNode;
        pNode->m_pPrevSibbling = pPrevNode;
    }
    if ( iIndex < m_arrChildren.Count() - 1 ) {
        XMLNode * pNextNode = m_arrChildren[iIndex + 1];
        pNextNode->m_pPrevSibbling = pNode;
        pNode->m_pNextSibbling = pNextNode;
    }
}
XMLNode * XMLNode::ReplaceChild( XMLNode * pNode, UInt iIndex )
{
    Assert( m_bIsNode );
    Assert( pNode->m_iType != XML_DOCUMENT );
    Assert( iIndex < m_arrChildren.Count() );
    XMLNode * pReplacedNode = m_arrChildren[iIndex];

    // Depth
    pNode->_rec_SetDepth( m_iDepth + 1 );

    pReplacedNode->_rec_SetDepth( 0 );

    // Path names
    StringFn->Format( pNode->m_strPathName, TEXT("%s/%s"), m_strPathName, pNode->m_strTagName );

    pReplacedNode->m_strPathName[0] = TEXT('/');
    StringFn->NCopy( pReplacedNode->m_strPathName + 1, pReplacedNode->m_strTagName, XML_URI_SIZE - 2 );

    // Parent linkage
    pNode->_rec_SetParentDocument( m_pParentDocument );
    pNode->m_pParent = this;
    pNode->m_iChildIndex = iIndex;

    pReplacedNode->_rec_SetParentDocument( NULL );
    pReplacedNode->m_pParent = NULL;
    pReplacedNode->m_iChildIndex = INVALID_OFFSET;

    // Child linkage
    m_arrChildren[iIndex] = pNode;

    if ( m_strIdentifierAttribute != NULL ) {
        XMLAttribute * pIDAttribute = pReplacedNode->GetAttribute( m_strIdentifierAttribute );
        if ( pIDAttribute != NULL ) {
            pReplacedNode->SetIdentifierAttribute( NULL );
            Bool bRemoved = m_mapChildrenID.Remove( pIDAttribute->GetValue() );
            Assert( bRemoved );
        }
        pIDAttribute = pNode->GetAttribute( m_strIdentifierAttribute );
        if ( pIDAttribute != NULL ) {
            pNode->SetIdentifierAttribute( m_strIdentifierAttribute );
            Bool bInserted = m_mapChildrenID.Insert( pIDAttribute->GetValue(), pNode );
            Assert( bInserted );
        }
    }

    // Sibbling linkage
    pNode->m_pPrevSibbling = pReplacedNode->m_pPrevSibbling;
    pNode->m_pNextSibbling = pReplacedNode->m_pNextSibbling;
    if ( pNode->m_pPrevSibbling != NULL )
        pNode->m_pPrevSibbling->m_pNextSibbling = pNode;
    if ( pNode->m_pNextSibbling != NULL )
        pNode->m_pNextSibbling->m_pPrevSibbling = pNode;

    pReplacedNode->m_pNextSibbling = NULL;
    pReplacedNode->m_pPrevSibbling = NULL;

    // Done
    return pReplacedNode;
}
XMLNode * XMLNode::RemoveChild( UInt iIndex )
{
    Assert( m_bIsNode );
    Assert( iIndex < m_arrChildren.Count() );
    XMLNode * pRemovedNode = m_arrChildren[iIndex];

    // Depth
    pRemovedNode->_rec_SetDepth( 0 );

    // Path names
    pRemovedNode->m_strPathName[0] = TEXT('/');
    StringFn->NCopy( pRemovedNode->m_strPathName + 1, pRemovedNode->m_strTagName, XML_URI_SIZE - 2 );

    // Parent linkage
    pRemovedNode->_rec_SetParentDocument( NULL );
    pRemovedNode->m_pParent = NULL;
    pRemovedNode->m_iChildIndex = INVALID_OFFSET;

    // Child linkage
    m_arrChildren.Remove( iIndex, NULL, 1 );
    for( UInt i = iIndex; i < m_arrChildren.Count(); ++i )
        --(m_arrChildren[i]->m_iChildIndex);
    
    if ( m_strIdentifierAttribute != NULL ) {
        XMLAttribute * pIDAttribute = pRemovedNode->GetAttribute( m_strIdentifierAttribute );
        if ( pIDAttribute != NULL ) {
            pRemovedNode->SetIdentifierAttribute( NULL );
            Bool bRemoved = m_mapChildrenID.Remove( pIDAttribute->GetValue() );
            Assert( bRemoved );
        }
    }

    // Sibbling linkage
    if ( pRemovedNode->m_pPrevSibbling != NULL )
        pRemovedNode->m_pPrevSibbling->m_pNextSibbling = pRemovedNode->m_pNextSibbling;
    if ( pRemovedNode->m_pNextSibbling != NULL )
        pRemovedNode->m_pNextSibbling->m_pPrevSibbling = pRemovedNode->m_pPrevSibbling;
    pRemovedNode->m_pPrevSibbling = NULL;
    pRemovedNode->m_pNextSibbling = NULL;

    // Done
    return pRemovedNode;
}

Void XMLNode::InsertBefore( XMLNode * pNode )
{
    Assert( m_iType != XML_DOCUMENT );
    Assert( pNode->m_iType != XML_DOCUMENT );

    // Depth
    pNode->_rec_SetDepth( m_iDepth );

    // Path names
    if ( m_pParent != NULL )
        StringFn->Format( pNode->m_strPathName, TEXT("%s/%s"), m_pParent->m_strPathName, pNode->m_strTagName );

    // Parent linkage
    pNode->_rec_SetParentDocument( m_pParentDocument );
    pNode->m_pParent = m_pParent;
    pNode->m_iChildIndex = INVALID_OFFSET;

    // Child linkage
    if ( m_pParent != NULL ) {
        pNode->m_iChildIndex = m_iChildIndex;
        m_pParent->m_arrChildren.Insert( pNode->m_iChildIndex, pNode );
        for( UInt i = pNode->m_iChildIndex + 1; i < m_pParent->m_arrChildren.Count(); ++i )
            ++(m_pParent->m_arrChildren[i]->m_iChildIndex);

        if ( m_pParent->m_strIdentifierAttribute != NULL ) {
            XMLAttribute * pIDAttribute = pNode->GetAttribute( m_pParent->m_strIdentifierAttribute );
            if ( pIDAttribute != NULL ) {
                pNode->SetIdentifierAttribute( m_pParent->m_strIdentifierAttribute );
                Bool bInserted = m_pParent->m_mapChildrenID.Insert( pIDAttribute->GetValue(), pNode );
                Assert( bInserted );
            }
        }
    }

    // Sibbling linkage
    pNode->m_pPrevSibbling = m_pPrevSibbling;
    pNode->m_pNextSibbling = this;
    if ( m_pPrevSibbling != NULL )
        m_pPrevSibbling->m_pNextSibbling = pNode;
    m_pPrevSibbling = pNode;
}
Void XMLNode::InsertAfter( XMLNode * pNode )
{
    Assert( m_iType != XML_DOCUMENT );
    Assert( pNode->m_iType != XML_DOCUMENT );

    // Depth
    pNode->_rec_SetDepth( m_iDepth );

    // Path names
    if ( m_pParent != NULL )
        StringFn->Format( pNode->m_strPathName, TEXT("%s/%s"), m_pParent->m_strPathName, pNode->m_strTagName );

    // Parent linkage
    pNode->_rec_SetParentDocument( m_pParentDocument );
    pNode->m_pParent = m_pParent;
    pNode->m_iChildIndex = INVALID_OFFSET;

    // Child linkage
    if ( m_pParent != NULL ) {
        pNode->m_iChildIndex = m_iChildIndex + 1;
        m_pParent->m_arrChildren.Insert( pNode->m_iChildIndex, pNode );
        for( UInt i = pNode->m_iChildIndex + 1; i < m_pParent->m_arrChildren.Count(); ++i )
            ++(m_pParent->m_arrChildren[i]->m_iChildIndex);

        if ( m_pParent->m_strIdentifierAttribute != NULL ) {
            XMLAttribute * pIDAttribute = pNode->GetAttribute( m_pParent->m_strIdentifierAttribute );
            if ( pIDAttribute != NULL ) {
                pNode->SetIdentifierAttribute( m_pParent->m_strIdentifierAttribute );
                Bool bInserted = m_pParent->m_mapChildrenID.Insert( pIDAttribute->GetValue(), pNode );
                Assert( bInserted );
            }
        }
    }

    // Sibbling linkage
    pNode->m_pPrevSibbling = this;
    pNode->m_pNextSibbling = m_pNextSibbling;
    if ( m_pNextSibbling != NULL )
        m_pNextSibbling->m_pPrevSibbling = pNode;
    m_pNextSibbling = pNode;
}

Bool XMLNode::Render( const GChar * strFile ) const
{
    HFile hFile = SystemFn->CreateFile( strFile, FILE_WRITE );
    if ( !(hFile.IsValid()) )
        return false;

    Bool bFinished = _Render( _WriteCallback_File, &hFile );

    hFile.Close();
    return bFinished;
}
Bool XMLNode::RenderXML( Array<GChar> * outXML ) const
{
    return _Render( _WriteCallback_XML, outXML );
}

Bool XMLNode::Parse( const GChar * strFile )
{
    HFile hFile = SystemFn->OpenFile( strFile, FILE_READ );
    if ( !(hFile.IsValid()) )
        return false;

    XMLToken iOverflowToken = XMLTOKEN_UNDEFINED;
    GChar chOverflowChar = NULLBYTE;
    Bool bValid = _Parse( &iOverflowToken, &chOverflowChar, _ReadCallback_File, &hFile );
    Assert( iOverflowToken == XMLTOKEN_UNDEFINED );
    Assert( chOverflowChar == NULLBYTE );

    hFile.Close();
    return bValid;
}
Bool XMLNode::ParseXML( const GChar * strXML )
{
    XMLToken iOverflowToken = XMLTOKEN_UNDEFINED;
    GChar chOverflowChar = NULLBYTE;
    Bool bValid = _Parse( &iOverflowToken, &chOverflowChar, _ReadCallback_XML, &strXML );
    Assert( iOverflowToken == XMLTOKEN_UNDEFINED );
    Assert( chOverflowChar == NULLBYTE );

    return bValid;
}

/////////////////////////////////////////////////////////////////////////////////

Void XMLNode::_MakeLeaf()
{
    Assert( !m_bIsNode );

    // nothing to do
}
Void XMLNode::_MakeNode()
{
    if ( m_bIsNode )
        return;

    // Create children storage
    m_arrChildren.Create();

    m_mapChildrenID.SetComparator( _Compare_Strings, NULL );
    m_mapChildrenID.Create();

    m_bIsNode = true;
}

Void XMLNode::_rec_SetDepth( UInt iDepth )
{
    // Set depth
    m_iDepth = iDepth;

    // Recurse
    if ( m_bIsNode ) {
        for( UInt i = 0; i < m_arrChildren.Count(); ++i )
            m_arrChildren[i]->_rec_SetDepth( iDepth + 1 );
    }
}
Void XMLNode::_rec_SetParentDocument( XMLDocument * pDocument )
{
    // Set parent document
    m_pParentDocument = pDocument;

    // Recurse
    if ( m_bIsNode ) {
        for( UInt i = 0; i < m_arrChildren.Count(); ++i )
            m_arrChildren[i]->_rec_SetParentDocument( pDocument );
    }
}

Bool XMLNode::_ReadCallback_File( GChar * outCh, Void * pUserData )
{
    HFile * pFile = (HFile*)pUserData;
    return pFile->ReadChar( outCh );
}
Bool XMLNode::_ReadCallback_XML( GChar * outCh, Void * pUserData )
{
    const GChar ** pXML = (const GChar **)pUserData;
    *outCh = **pXML;
    ++(*pXML);
    return true;
}

Bool XMLNode::_WriteCallback_File( const GChar * str, Void * pUserData )
{
    HFile * pFile = (HFile*)pUserData;
    return pFile->WriteString( str );
}
Bool XMLNode::_WriteCallback_XML( const GChar * str, Void * pUserData )
{
    Array<GChar> * pXML = (Array<GChar>*)pUserData;
    while( *str != NULLBYTE )
        pXML->Push( *str++ );
    return true;
}

XMLNode * XMLNode::_Clone( Bool bRecursive ) const
{
    // Create clone
    XMLNode * pClone = New() XMLNode( XML_NODE, m_strTagName );

    if ( m_bIsNode )
        pClone->_MakeNode();
    else
        pClone->_MakeLeaf();

    // Copy attributes
    _AttributeMap::Iterator itAttribute = m_mapAttributes.Begin();
    while( !(itAttribute.IsNull()) ) {
        pClone->CreateAttribute( itAttribute.GetKey(), itAttribute.GetItem()->GetValue() );
        ++itAttribute;
    }
    pClone->m_strIdentifierAttribute = m_strIdentifierAttribute;

    // Recurse
    if ( bRecursive && m_bIsNode ) {
        for( UInt i = 0; i < m_arrChildren.Count(); ++i ) {
            XMLNode * pChildClone = m_arrChildren[i]->Clone( true );
            pClone->AppendChild( pChildClone );
        }
    }

    // Done
    return pClone;
}

Void XMLNode::_GetIndentString( GChar * outIndentString ) const
{
    Assert( (m_iDepth+2) < 256 );
    outIndentString[0] = TEXT('\r');
    outIndentString[1] = TEXT('\n');
    for( UInt i = 0; i < m_iDepth; ++i )
        outIndentString[i+2] = TEXT('\t');
    outIndentString[m_iDepth+2] = NULLBYTE;
}

Bool XMLNode::_Render( _XMLWriteCallback pfCallback, Void * pUserData ) const
{
    Bool bContinue;
    GChar strIndent[256];
    _GetIndentString( strIndent );

    // Indentation
    bContinue = pfCallback( strIndent, pUserData );
    if ( !bContinue )
        return false;

    // Start opening tag
    bContinue = pfCallback( TEXT("<"), pUserData );
    if ( !bContinue )
        return false;

    // Tag Name
    bContinue = pfCallback( m_strTagName, pUserData );
    if ( !bContinue )
        return false;

    // Attributes
    _AttributeMap::Iterator itAttribute = m_mapAttributes.Begin();
    while( !(itAttribute.IsNull()) ) {
        XMLAttribute * pAttribute = itAttribute.GetItem();
        bContinue = pfCallback( TEXT(" "), pUserData );
        if ( !bContinue )
            return false;
        bContinue = pfCallback( pAttribute->GetName(), pUserData );
        if ( !bContinue )
            return false;
        bContinue = pfCallback( TEXT("=\""), pUserData );
        if ( !bContinue )
            return false;
        bContinue = pfCallback( pAttribute->GetValue(), pUserData );
        if ( !bContinue )
            return false;
        bContinue = pfCallback( TEXT("\""), pUserData );
        if ( !bContinue )
            return false;
        ++itAttribute;
    }

    // Leaf case
    if ( !m_bIsNode ) {
        // End opening tag
        bContinue = pfCallback( TEXT("/>"), pUserData );
        if ( !bContinue )
            return false;

        return true;
    }

    // End opening tag
    bContinue = pfCallback( TEXT(">"), pUserData );
    if ( !bContinue )
        return false;

    // Recurse
    for( UInt i = 0; i < m_arrChildren.Count(); ++i ) {
        bContinue = m_arrChildren[i]->_Render( pfCallback, pUserData );
        if ( !bContinue )
            return false;
    }

    // Indentation
    if ( m_arrChildren.Count() > 0 ) {
        bContinue = pfCallback( strIndent, pUserData );
        if ( !bContinue )
            return false;
    }

    // Start closing tag
    bContinue = pfCallback( TEXT("</"), pUserData );
    if ( !bContinue )
        return false;

    // Tag Name
    bContinue = pfCallback( m_strTagName, pUserData );
    if ( !bContinue )
        return false;

    // End closing tag
    bContinue = pfCallback( TEXT(">"), pUserData );
    if ( !bContinue )
        return false;

    return true;
}

XMLToken XMLNode::_Parse_NextToken( GChar * outTokenData, GChar * pOverflowChar, _XMLReadCallback pfCallback, Void * pUserData )
{
    Bool bContinue;
    GChar * strOut = outTokenData;

    // Perform lexical analysis
    GChar ch;
    UInt iState = 0;
    while( true ) {
        switch( iState ) {
            case 0: { // Initial state
                if ( *pOverflowChar == NULLBYTE )
                    bContinue = pfCallback( &ch, pUserData );
                else {
                    ch = *pOverflowChar;
                    *pOverflowChar = NULLBYTE;
                    bContinue = true;
                }
                if ( !bContinue ) {
                    *outTokenData = NULLBYTE;
                    return XMLTOKEN_UNDEFINED;
                }

                if ( _IsBlank(ch) ) {
                    *strOut++ = ch;
                    iState = 1;
                    break;
                }
                if ( ch == TEXT('<') ) {
                    *strOut++ = ch;
                    iState = 2;
                    break;
                }
                if ( ch == TEXT('>') ) {
                    *strOut++ = ch;
                    *strOut = NULLBYTE;
                    return XMLTOKEN_TAGEND;
                }
                if ( ch == TEXT('/') ) {
                    *strOut++ = ch;
                    iState = 3;
                    break;
                }
                if ( ch == TEXT('?') ) {
                    *strOut++ = ch;
                    iState = 4;
                    break;
                }
                if ( ch == TEXT('-') ) {
                    *strOut++ = ch;
                    iState = 5;
                    break;
                }
                if ( ch == TEXT('=') ) {
                    *strOut++ = ch;
                    *strOut = NULLBYTE;
                    return XMLTOKEN_ATTRIB_AFFECT;
                }
                if ( ch == TEXT('"') ) {
                    *strOut++ = ch;
                    *strOut = NULLBYTE;
                    return XMLTOKEN_ATTRIB_STRING;
                }
                if ( _IsNameChar(ch) ) {
                    *strOut++ = ch;
                    iState = 6;
                    break;
                }
                *strOut++ = ch;
                *strOut = NULLBYTE;
                return XMLTOKEN_TEXT;
            } break;
            case 1: { // Separator state
                bContinue = pfCallback( &ch, pUserData );
                if ( !bContinue ) {
                    *outTokenData = NULLBYTE;
                    return XMLTOKEN_UNDEFINED;
                }

                if ( !(_IsBlank(ch)) ) {
                    *strOut = NULLBYTE;
                    *pOverflowChar = ch;
                    return XMLTOKEN_SEPARATOR;
                }
                *strOut++ = ch;
            } break;
            case 2: { // Tag start state
                bContinue = pfCallback( &ch, pUserData );
                if ( !bContinue ) {
                    *outTokenData = NULLBYTE;
                    return XMLTOKEN_UNDEFINED;
                }

                if ( ch == TEXT('/') ) {
                    *strOut++ = ch;
                    *strOut = NULLBYTE;
                    return XMLTOKEN_TAGSTART_CLOSENODE;
                }
                if ( ch == TEXT('?') ) {
                    *strOut++ = ch;
                    iState = 7;
                    break;
                }
                if ( ch == TEXT('!') ) {
                    *strOut++ = ch;
                    iState = 8;
                    break;
                }
                if ( _IsNameChar(ch) ) {
                    *strOut = NULLBYTE;
                    *pOverflowChar = ch;
                    return XMLTOKEN_TAGSTART;
                }
                *strOut++ = ch;
                *strOut = NULLBYTE;
                return XMLTOKEN_TEXT;
            } break;
            case 3: { // Tag end (leaf) state
                bContinue = pfCallback( &ch, pUserData );
                if ( !bContinue ) {
                    *outTokenData = NULLBYTE;
                    return XMLTOKEN_UNDEFINED;
                }

                if ( ch == TEXT('>') ) {
                    *strOut++ = ch;
                    *strOut = NULLBYTE;
                    return XMLTOKEN_TAGEND_LEAF;
                }
                *strOut++ = ch;
                *strOut = NULLBYTE;
                return XMLTOKEN_TEXT;
            } break;
            case 4: { // Header end state
                bContinue = pfCallback( &ch, pUserData );
                if ( !bContinue ) {
                    *outTokenData = NULLBYTE;
                    return XMLTOKEN_UNDEFINED;
                }

                if ( ch == TEXT('>') ) {
                    *strOut++ = ch;
                    *strOut = NULLBYTE;
                    return XMLTOKEN_HEADER_END;
                }
                *strOut++ = ch;
                *strOut = NULLBYTE;
                return XMLTOKEN_TEXT;
            } break;
            case 5: { // Comment end state 1
                bContinue = pfCallback( &ch, pUserData );
                if ( !bContinue ) {
                    *outTokenData = NULLBYTE;
                    return XMLTOKEN_UNDEFINED;
                }

                if ( ch == TEXT('-') ) {
                    *strOut++ = ch;
                    iState = 9;
                    break;
                }
                *strOut++ = ch;
                *strOut = NULLBYTE;
                return XMLTOKEN_TEXT;
            } break;
            case 6: { // Name state
                bContinue = pfCallback( &ch, pUserData );
                if ( !bContinue ) {
                    *outTokenData = NULLBYTE;
                    return XMLTOKEN_UNDEFINED;
                }

                if ( !(_IsNameChar(ch)) ) {
                    *strOut = NULLBYTE;
                    *pOverflowChar = ch;
                    return XMLTOKEN_NAME;
                }
                *strOut++ = ch;
            } break;
            case 7: { // Header start state 1
                bContinue = pfCallback( &ch, pUserData );
                if ( !bContinue ) {
                    *outTokenData = NULLBYTE;
                    return XMLTOKEN_UNDEFINED;
                }

                if ( ch == TEXT('x') ) {
                    *strOut++ = ch;
                    iState = 10;
                    break;
                }
                *strOut++ = ch;
                *strOut = NULLBYTE;
                return XMLTOKEN_TEXT;
            } break;
            case 8: { // Comment start state 1
                bContinue = pfCallback( &ch, pUserData );
                if ( !bContinue ) {
                    *outTokenData = NULLBYTE;
                    return XMLTOKEN_UNDEFINED;
                }

                if ( ch == TEXT('-') ) {
                    *strOut++ = ch;
                    iState = 11;
                    break;
                }
                *strOut++ = ch;
                *strOut = NULLBYTE;
                return XMLTOKEN_TEXT;
            } break;
            case 9: { // Comment end state 2
                bContinue = pfCallback( &ch, pUserData );
                if ( !bContinue ) {
                    *outTokenData = NULLBYTE;
                    return XMLTOKEN_UNDEFINED;
                }

                if ( ch == TEXT('>') ) {
                    *strOut++ = ch;
                    *strOut = NULLBYTE;
                    return XMLTOKEN_COMMENT_END;
                }
                *strOut++ = ch;
                *strOut = NULLBYTE;
                return XMLTOKEN_TEXT;
            } break;
            case 10: { // Header start state 2
                bContinue = pfCallback( &ch, pUserData );
                if ( !bContinue ) {
                    *outTokenData = NULLBYTE;
                    return XMLTOKEN_UNDEFINED;
                }

                if ( ch == TEXT('m') ) {
                    *strOut++ = ch;
                    iState = 12;
                    break;
                }
                *strOut++ = ch;
                *strOut = NULLBYTE;
                return XMLTOKEN_TEXT;
            } break;
            case 11: { // Comment start state 2
                bContinue = pfCallback( &ch, pUserData );
                if ( !bContinue ) {
                    *outTokenData = NULLBYTE;
                    return XMLTOKEN_UNDEFINED;
                }

                if ( ch == TEXT('-') ) {
                    *strOut++ = ch;
                    *strOut = NULLBYTE;
                    return XMLTOKEN_COMMENT_START;
                }
                *strOut++ = ch;
                *strOut = NULLBYTE;
                return XMLTOKEN_TEXT;
            } break;
            case 12: { // Header start state 3
                bContinue = pfCallback( &ch, pUserData );
                if ( !bContinue ) {
                    *outTokenData = NULLBYTE;
                    return XMLTOKEN_UNDEFINED;
                }

                if ( ch == TEXT('l') ) {
                    *strOut++ = ch;
                    *strOut = NULLBYTE;
                    return XMLTOKEN_HEADER_START;
                }
                *strOut++ = ch;
                *strOut = NULLBYTE;
                return XMLTOKEN_TEXT;
            } break;
            default: Assert(false); *outTokenData = NULLBYTE; return XMLTOKEN_UNDEFINED;
        }
    }
}

Bool XMLNode::_Parse_AttributeList( XMLToken * pOverflowToken, GChar * pOverflowChar, _XMLReadCallback pfCallback, Void * pUserData )
{
    GChar strTokenData[XML_NAME_SIZE];
    GChar strName[XML_NAME_SIZE];

    // Separator 
    XMLToken iToken = _Parse_NextToken( strTokenData, pOverflowChar, pfCallback, pUserData );
    if ( iToken != XMLTOKEN_SEPARATOR ) {
        *pOverflowToken = iToken;
        return true;
    }

    // Name
    iToken = _Parse_NextToken( strName, pOverflowChar, pfCallback, pUserData );
    if ( iToken != XMLTOKEN_NAME ) {
        *pOverflowToken = iToken;
        return true;
    }

    // Affectation sign
    iToken = _Parse_NextToken( strTokenData, pOverflowChar, pfCallback, pUserData );
    if ( iToken == XMLTOKEN_SEPARATOR )
        iToken = _Parse_NextToken( strTokenData, pOverflowChar, pfCallback, pUserData );
    if ( iToken != XMLTOKEN_ATTRIB_AFFECT ) {
        *pOverflowToken = XMLTOKEN_UNDEFINED;
        return false;
    }

    // Value start
    iToken = _Parse_NextToken( strTokenData, pOverflowChar, pfCallback, pUserData );
    if ( iToken == XMLTOKEN_SEPARATOR )
        iToken = _Parse_NextToken( strTokenData, pOverflowChar, pfCallback, pUserData );
    if ( iToken != XMLTOKEN_ATTRIB_STRING ) {
        *pOverflowToken = XMLTOKEN_UNDEFINED;
        return false;
    }

    // Create attribute
    XMLAttribute * pAttribute = CreateAttribute( strName );
    Array<GChar> * pValue = pAttribute->EditValue();
    pValue->Clear();

    // Extract value
    Assert( *pOverflowChar == NULLBYTE );
    GChar ch;
    Bool bContinue = pfCallback( &ch, pUserData );
    if ( !bContinue ) {
        pValue->Push( NULLBYTE );
        *pOverflowToken = XMLTOKEN_UNDEFINED;
        return false;
    }
    while( ch != TEXT('"') ) {
        // Escape string
        if ( ch == TEXT('\\') ) {
            bContinue = pfCallback( &ch, pUserData );
            if ( !bContinue ) {
                pValue->Push( NULLBYTE );
                *pOverflowToken = XMLTOKEN_UNDEFINED;
                return false;
            }
            if ( ch == TEXT('\\') )
                pValue->Push( TEXT('\\') );
            else if ( ch == TEXT('n') )
                pValue->Push( TEXT('\n') );
            else if ( ch == TEXT('t') )
                pValue->Push( TEXT('\t') );
            else if ( ch == TEXT('"') )
                pValue->Push( TEXT('"') );
            else {
                pValue->Push( NULLBYTE );
                *pOverflowToken = XMLTOKEN_UNDEFINED;
                return false;
            }
        } else
            pValue->Push( ch );

        bContinue = pfCallback( &ch, pUserData );
        if ( !bContinue ) {
            pValue->Push( NULLBYTE );
            *pOverflowToken = XMLTOKEN_UNDEFINED;
            return false;
        }
    }
    pValue->Push( NULLBYTE );

    // Recurse
    return _Parse_AttributeList( pOverflowToken, pOverflowChar, pfCallback, pUserData );
}
Bool XMLNode::_Parse_NodeContent( XMLToken * pOverflowToken, GChar * pOverflowChar, _XMLReadCallback pfCallback, Void * pUserData )
{
    GChar strTokenData[XML_NAME_SIZE];
    Bool bContinue;

    // Parse next token
    XMLToken iToken;
    if ( *pOverflowToken == XMLTOKEN_UNDEFINED )
        iToken = _Parse_NextToken( strTokenData, pOverflowChar, pfCallback, pUserData );
    else {
        iToken = *pOverflowToken;
        *pOverflowToken = XMLTOKEN_UNDEFINED;
    }
    if ( iToken == XMLTOKEN_SEPARATOR )
        iToken = _Parse_NextToken( strTokenData, pOverflowChar, pfCallback, pUserData );

    // Closing case
    if ( iToken == XMLTOKEN_TAGSTART_CLOSENODE ) {
        *pOverflowToken = iToken;
        return true;
    }

    // Node / Leaf case
    if ( iToken == XMLTOKEN_TAGSTART ) {
        *pOverflowToken = iToken;

        // Create child
        XMLNode * pChild = New() XMLNode( XML_NODE, TEXT("_xmlnew") );

        // Recurse on child
        bContinue = pChild->_Parse( pOverflowToken, pOverflowChar, pfCallback, pUserData );
        if ( !bContinue )
            return false;

        // Add child
        AppendChild( pChild );

        // Recurse
        return _Parse_NodeContent( pOverflowToken, pOverflowChar, pfCallback, pUserData );
    }

    // Comment case
    if ( iToken == XMLTOKEN_COMMENT_START ) {
        // Create child
        XMLComment * pChild = New() XMLComment();

        // Extract Comment
        Array<GChar> * pComment = pChild->EditComment();
        pComment->Clear();

        Assert( *pOverflowChar == NULLBYTE );
        GChar ch;
        bContinue = pfCallback( &ch, pUserData );
        if ( !bContinue ) {
            pComment->Push( NULLBYTE );
            return false;
        }
        while( true ) {
            if ( ch == TEXT('-') ) {
                bContinue = pfCallback( &ch, pUserData );
                if ( !bContinue ) {
                    pComment->Push( NULLBYTE );
                    return false;
                }

                if ( ch == TEXT('-') ) {
                    bContinue = pfCallback( &ch, pUserData );
                    if ( !bContinue ) {
                        pComment->Push( NULLBYTE );
                        return false;
                    }

                    if ( ch == TEXT('>') )
                        break;
                    else {
                        pComment->Push( TEXT('-') );
                        pComment->Push( TEXT('-') );
                    }
                } else
                    pComment->Push( TEXT('-') );
            }
            pComment->Push( ch );

            bContinue = pfCallback( &ch, pUserData );
            if ( !bContinue ) {
                pComment->Push( NULLBYTE );
                return false;
            }
        }
        pComment->Push( NULLBYTE );

        // Add child
        AppendChild( pChild );

        // Recurse
        return _Parse_NodeContent( pOverflowToken, pOverflowChar, pfCallback, pUserData );
    }

    // Text case
        // Create child
    XMLText * pChild = New() XMLText();

        // Extract text
    Array<GChar> * pText = pChild->EditText();
    pText->Clear();
    pText->Push( strTokenData, StringFn->Length(strTokenData) );

    GChar ch;
    if ( *pOverflowChar == NULLBYTE ) {
        bContinue = pfCallback( &ch, pUserData );
        if ( !bContinue ) {
            pText->Push( NULLBYTE );
            return false;
        }
    } else {
        ch = *pOverflowChar;
        *pOverflowChar = NULLBYTE;
    }
    while( ch != TEXT('<') ) {
        pText->Push( ch );

        bContinue = pfCallback( &ch, pUserData );
        if ( !bContinue ) {
            pText->Push( NULLBYTE );
            return false;
        }
    }
    pText->Push( NULLBYTE );
    *pOverflowChar = ch;

        // Add child
    AppendChild( pChild );

        // Recurse
    return _Parse_NodeContent( pOverflowToken, pOverflowChar, pfCallback, pUserData );
}

Bool XMLNode::_Parse( XMLToken * pOverflowToken, GChar * pOverflowChar, _XMLReadCallback pfCallback, Void * pUserData )
{
    GChar strTokenData[XML_NAME_SIZE];

    // Parse open tag
    XMLToken iToken;
    if ( *pOverflowToken == XMLTOKEN_UNDEFINED )
        iToken = _Parse_NextToken( strTokenData, pOverflowChar, pfCallback, pUserData );
    else {
        iToken = *pOverflowToken;
        *pOverflowToken = XMLTOKEN_UNDEFINED;
    }
    if ( iToken == XMLTOKEN_SEPARATOR )
        iToken = _Parse_NextToken( strTokenData, pOverflowChar, pfCallback, pUserData );
    if ( iToken != XMLTOKEN_TAGSTART )
        return false;

        // Name & path name
    iToken = _Parse_NextToken( strTokenData, pOverflowChar, pfCallback, pUserData );
    if ( iToken != XMLTOKEN_NAME )
        return false;

    StringFn->NCopy( m_strTagName, strTokenData, XML_NAME_SIZE );
    if ( m_pParent != NULL )
        StringFn->Format( m_strPathName, TEXT("%s/%s"), m_pParent->m_strPathName, m_strTagName );
    else {
        m_strPathName[0] = TEXT('/');
        StringFn->NCopy( m_strPathName + 1, m_strTagName, XML_URI_SIZE - 2 );
    }

        // Attributes
    Bool bContinue = _Parse_AttributeList( &iToken, pOverflowChar, pfCallback, pUserData );
    if ( !bContinue )
        return false;

    // Leaf case
    if ( iToken == XMLTOKEN_TAGEND_LEAF ) {
        _MakeLeaf();

        Assert( *pOverflowToken == XMLTOKEN_UNDEFINED );
        Assert( *pOverflowChar == NULLBYTE );
        return true;
    }

    // Node case
    if ( iToken != XMLTOKEN_TAGEND )
        return false;
    _MakeNode();

    // Recurse
    bContinue = _Parse_NodeContent( pOverflowToken, pOverflowChar, pfCallback, pUserData );
    if ( !bContinue )
        return false;
    iToken = *pOverflowToken;
    *pOverflowToken = XMLTOKEN_UNDEFINED;

    // Parse close tag
    if ( iToken != XMLTOKEN_TAGSTART_CLOSENODE )
        return false;

    iToken = _Parse_NextToken( strTokenData, pOverflowChar, pfCallback, pUserData );
    if ( iToken != XMLTOKEN_NAME )
        return false;

    if ( _Compare_Strings(m_strTagName,strTokenData,NULL) != 0 )
        return false;

    iToken = _Parse_NextToken( strTokenData, pOverflowChar, pfCallback, pUserData );
    if ( iToken != XMLTOKEN_TAGEND )
        return false;

    // Done
    Assert( *pOverflowToken == XMLTOKEN_UNDEFINED );
    Assert( *pOverflowChar == NULLBYTE );
    return true;
}

