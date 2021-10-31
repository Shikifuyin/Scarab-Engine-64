/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Datastruct/List/List.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : List template implementation.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// List implementation
template<typename T>
typename const List<T>::Iterator List<T>::Iterator::Null = List<T>::Iterator();

template<typename T>
List<T>::List():
    Datastruct()
{
    m_pHead = NULL;
    m_pEnd = NULL;
    m_iLength = 0;
}
template<typename T>
List<T>::~List()
{
	Assert( !IsCreated() );
    // Call Destroy explicitly
}

template<typename T>
inline Bool List<T>::IsCreated() const {
    return ( m_pHead != NULL );
}
template<typename T>
Void List<T>::Create()
{
    Assert( m_iLength == 0 );
}
template<typename T>
Void List<T>::Destroy()
{
	Clear();
}

template<typename T>
inline SizeT List<T>::MemorySize() const {
    return ( m_iLength * sizeof(ListNode) );
}
template<typename T>
inline UInt List<T>::Count() const {
    return m_iLength;
}

template<typename T>
Void List<T>::Clear()
{
    while( m_iLength > 1 ) {
		m_pEnd = m_pEnd->pPrev;
		Delete( m_pEnd->pNext, m_iAllocatorID, m_iMemoryContextID );
		m_pEnd->pNext = NULL;
		--m_iLength;
	}
    if ( m_iLength > 0 )
        Delete( m_pEnd, m_iAllocatorID, m_iMemoryContextID );

	m_pHead = NULL;
	m_pEnd = NULL;
	m_iLength = 0;
}

template<typename T>
inline typename List<T>::Iterator List<T>::Begin() const {
    return Iterator(m_pHead);
}
template<typename T>
inline typename List<T>::Iterator List<T>::End() const {
    return Iterator(m_pEnd);
}

template<typename T>
Void List<T>::Push( const T & rItem )
{
	ListNode * pNewNode;
	New( ListNode, pNewNode, ListNode(), m_iAllocatorID, m_iMemoryContextID );

    pNewNode->pNext = NULL;
    pNewNode->pPrev = NULL;
    pNewNode->Item = rItem;
	if ( m_iLength == 0 )
		m_pHead = pNewNode;
	else {
		pNewNode->pPrev = m_pEnd;
		m_pEnd->pNext = pNewNode;
	}
	m_pEnd = pNewNode;
	++m_iLength;
}
template<typename T>
Void List<T>::Pop( T & outItem )
{
	Assert(m_iLength > 0);
	outItem = m_pEnd->Item;
	if ( m_iLength == 1 ) {
		Delete( m_pEnd, m_iAllocatorID, m_iMemoryContextID );

		m_pHead = NULL;
		m_pEnd = NULL;
	} else {
		m_pEnd = m_pEnd->pPrev;

		Delete( m_pEnd->pNext, m_iAllocatorID, m_iMemoryContextID );

		m_pEnd->pNext = NULL;
	}
	--m_iLength;
}
template<typename T>
Void List<T>::Unshift( const T & rItem )
{
	ListNode * pNewNode;
	New( ListNode, pNewNode, ListNode(), m_iAllocatorID, m_iMemoryContextID );

    pNewNode->pNext = NULL;
    pNewNode->pPrev = NULL;
    pNewNode->Item = rItem;
	if ( m_iLength == 0 )
		m_pEnd = pNewNode;
	else {
		pNewNode->pNext = m_pHead;
		m_pHead->pPrev = pNewNode;
	}
	m_pHead = pNewNode;
	++m_iLength;
}
template<typename T>
Void List<T>::Shift( T & outItem )
{
	Assert(m_iLength > 0);
	outItem = m_pHead->Item;
	if ( m_iLength == 1 ) {
		Delete( m_pHead, m_iAllocatorID, m_iMemoryContextID );

		m_pHead = NULL;
		m_pEnd = NULL;
	} else {
		m_pHead = m_pHead->pNext;

		Delete( m_pHead->pPrev, m_iAllocatorID, m_iMemoryContextID );

		m_pHead->pPrev = NULL;
	}
	--m_iLength;
}
template<typename T>
Void List<T>::Insert( Iterator & iAt, const T & rItem )
{
    Assert( !( iAt.IsNull() ) );

	ListNode * pNewNode;
	New( ListNode, pNewNode, ListNode(), m_iAllocatorID, m_iMemoryContextID );

    pNewNode->pNext = NULL;
    pNewNode->pPrev = NULL;
    pNewNode->Item = rItem;
	if ( iAt.IsBegin() || m_iLength < 2 ) {
		if ( m_iLength == 0 )
		    m_pEnd = pNewNode;
	    else {
		    pNewNode->pNext = m_pHead;
		    m_pHead->pPrev = pNewNode;
	    }
	    m_pHead = pNewNode;
	} else {
		pNewNode->pPrev = iAt.m_pNode->pPrev;
		pNewNode->pNext = iAt.m_pNode;
		iAt.m_pNode->pPrev->pNext = pNewNode;
		iAt.m_pNode->pPrev = pNewNode;
	}
    iAt.m_pNode = pNewNode;
    ++m_iLength;
}
template<typename T>
Void List<T>::Remove( Iterator & iAt, T & outItem )
{
    Assert(m_iLength > 0);
    Assert( !( iAt.IsNull() ) );

    outItem = iAt.m_pNode->Item;
    if ( m_iLength == 1 ) {
	    Delete( m_pHead, m_iAllocatorID, m_iMemoryContextID );

	    m_pHead = NULL;
	    m_pEnd = NULL;
		iAt.m_pNode = NULL;
    } else {
        if ( iAt.IsBegin() ) {
            m_pHead = m_pHead->pNext;

		    Delete( m_pHead->pPrev, m_iAllocatorID, m_iMemoryContextID );

		    m_pHead->pPrev = NULL;
            iAt.m_pNode = m_pHead;
        } else if ( iAt.IsEnd() ) {
            m_pEnd = m_pEnd->pPrev;

		    Delete( m_pEnd->pNext, m_iAllocatorID, m_iMemoryContextID );

		    m_pEnd->pNext = NULL;
            iAt.m_pNode = NULL;
        } else {
            ListNode * pNode = iAt.m_pNode->pNext;
		    iAt.m_pNode->pPrev->pNext = iAt.m_pNode->pNext;
		    iAt.m_pNode->pNext->pPrev = iAt.m_pNode->pPrev;

		    Delete( iAt.m_pNode, m_iAllocatorID, m_iMemoryContextID );

		    iAt.m_pNode = pNode;
        }
    }
    --m_iLength;
}

//template<typename T>
//Void List<T>::Push(const List<T> & tList)
//{
//	const ListNode<T> * pCurNode = tList.m_pHead;
//	while(pCurNode != NULL)
//	{
//		Push(pCurNode->Value);
//		pCurNode = pCurNode->pNext;
//	}
//}
//template<typename T>
//Void List<T>::Pop(List<T> & rOut, UInt length)
//{
//	if (length == 0)
//		return;
//	if (length > m_Length)
//		length = m_Length;
//	rOut.Clear();
//	rOut.m_pHead = m_pEnd;
//	rOut.m_pEnd = m_pEnd;
//	rOut.m_Length = 1;
//	while( rOut.m_Length < length )
//	{
//		rOut.m_pHead = rOut.m_pHead->pPrev;
//		++(rOut.m_Length);
//	}
//	m_Length -= rOut.m_Length;
//	if (m_Length == 0)
//	{
//		m_pHead = NULL;
//		m_pEnd = NULL;
//	}
//	else
//	{
//		m_pEnd = rOut.m_pHead->pPrev;
//		m_pEnd->pNext = NULL;
//	}
//	rOut.m_pHead->pPrev = NULL;
//	rOut.m_pEnd->pNext = NULL;
//}
//template<typename T>
//Void List<T>::Unshift(const List<T> & tList)
//{
//	const ListNode<T> * pCurNode = tList.m_pEnd;
//	while(pCurNode != NULL)
//	{
//		Unshift(pCurNode->Value);
//		pCurNode = pCurNode->pPrev;
//	}
//}
//template<typename T>
//Void List<T>::Shift(List<T> & rOut, UInt length)
//{
//	if (length == 0)
//		return;
//	if (length > m_Length)
//		length = m_Length;
//	rOut.Clear();
//	rOut.m_pHead = m_pHead;
//	rOut.m_pEnd = m_pHead;
//	rOut.m_Length = 1;
//	while( rOut.m_Length < length )
//	{
//		rOut.m_pEnd = rOut.m_pEnd->pNext;
//		++(rOut.m_Length);
//	}
//	m_Length -= rOut.m_Length;
//	if (m_Length == 0)
//	{
//		m_pHead = NULL;
//		m_pEnd = NULL;
//	}
//	else
//	{
//		m_pHead = rOut.m_pEnd->pNext;
//		m_pHead->pPrev = NULL;
//	}
//	rOut.m_pHead->pPrev = NULL;
//	rOut.m_pEnd->pNext = NULL;
//}
//template<typename T>
//Void List<T>::Insert(Iterator & iAt, const List<T> & tList)
//{
//	if (iAt.IsNull())
//		return;
//	const ListNode<T> * pCurNode = tList.m_pEnd;
//	while(pCurNode != NULL)
//	{
//		Insert(iAt, pCurNode->Value);
//		pCurNode = pCurNode->pPrev;
//	}
//}
//template<typename T>
//Void List<T>::Remove(List<T> & rOut, Iterator & iAt, UInt length)
//{
//	if (iAt.IsNull() || length == 0)
//		return;
//	rOut.Clear();
//	rOut.m_pHead = iAt.m_pNode;
//	rOut.m_pEnd = iAt.m_pNode;
//	rOut.m_Length = 1;
//	while((rOut.m_Length < length) && (rOut.m_pEnd->pNext != NULL))
//	{
//		rOut.m_pEnd = rOut.m_pEnd->pNext;
//		++rOut.m_Length;
//	}
//	m_Length -= rOut.m_Length;
//	iAt.m_pNode = rOut.m_pEnd->pNext;
//	if (m_Length == 0)
//	{
//		m_pHead = NULL;
//		m_pEnd = NULL;	
//	}
//	else
//	{
//		if (rOut.m_pEnd->pNext != NULL)
//			rOut.m_pEnd->pNext->pPrev = rOut.m_pHead->pPrev;
//		else
//			m_pEnd = rOut.m_pHead->pPrev;
//		if (rOut.m_pHead->pPrev != NULL)
//			rOut.m_pHead->pPrev->pNext = rOut.m_pEnd->pNext;
//		else
//			m_pHead = rOut.m_pEnd->pNext;
//	}
//	rOut.m_pHead->pPrev = NULL;
//	rOut.m_pEnd->pNext = NULL;
//}

template<typename T>
typename List<T>::Iterator List<T>::Search( const T & rItem, const Iterator & iAt ) const
{
    ListNode * pCurNode = iAt.m_pNode;
    if (pCurNode == NULL)
        pCurNode = m_pHead;
	while( pCurNode != NULL ) {
		if ( pCurNode->Item == rItem )
			return Iterator(pCurNode);
		pCurNode = pCurNode->pNext;
	}
    return Iterator::Null;
}
