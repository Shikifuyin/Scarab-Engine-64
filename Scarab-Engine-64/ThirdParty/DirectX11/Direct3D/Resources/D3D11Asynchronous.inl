/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/Resources/D3D11Asynchronous.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : GPU resources : Asynchronous queries.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// D3D11Asynchronous implementation
inline Bool D3D11Asynchronous::IsCreated() const {
    return ( m_pAsynchronous != NULL || m_bTemporaryDestroyed );
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11Query implementation
inline D3D11AsynchronousType D3D11Query::GetType() const {
    return D3D11ASYNCHRONOUS_QUERY;
}

inline D3D11QueryType D3D11Query::GetQueryType() const {
    return m_iQueryType;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11QueryCommandProcessing implementation

/////////////////////////////////////////////////////////////////////////////////
// D3D11QueryOcclusion implementation

/////////////////////////////////////////////////////////////////////////////////
// D3D11QueryTimeStampFrequency implementation

/////////////////////////////////////////////////////////////////////////////////
// D3D11QueryTimeStamp implementation

/////////////////////////////////////////////////////////////////////////////////
// D3D11QueryStatsPipeline implementation

/////////////////////////////////////////////////////////////////////////////////
// D3D11QueryStatsStreamOutput implementation

/////////////////////////////////////////////////////////////////////////////////
// D3D11Predicate implementation
inline D3D11AsynchronousType D3D11Predicate::GetType() const {
    return D3D11ASYNCHRONOUS_PREDICATE;
}

inline D3D11PredicateType D3D11Predicate::GetPredicateType() const {
    return m_iPredicateType;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11PredicateOcclusion implementation

/////////////////////////////////////////////////////////////////////////////////
// D3D11PredicateStreamOutputOverflow implementation

/////////////////////////////////////////////////////////////////////////////////
// D3D11Counter implementation
inline D3D11AsynchronousType D3D11Counter::GetType() const {
    return D3D11ASYNCHRONOUS_COUNTER;
}


