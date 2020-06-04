/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/2D/Resources/D2D1StrokeStyle.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : D2D1 Dev-Ind Resource : Stroke Styles.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// D2D1StrokeStyle implementation
inline Bool D2D1StrokeStyle::IsCreated() const {
    return ( m_pStrokeStyle != NULL );
}

inline D2D1StrokeCapStyle D2D1StrokeStyle::GetStartCap() const {
    DebugAssert( IsCreated() );
    return m_hDesc.iStartCap;
}
inline D2D1StrokeCapStyle D2D1StrokeStyle::GetEndCap() const {
    DebugAssert( IsCreated() );
    return m_hDesc.iEndCap;
}
inline D2D1StrokeCapStyle D2D1StrokeStyle::GetDashCap() const {
    DebugAssert( IsCreated() );
    return m_hDesc.iDashCap;
}

inline D2D1StrokeLineJoin D2D1StrokeStyle::GetLineJoin() const {
    DebugAssert( IsCreated() );
    return m_hDesc.iLineJoin;
}
inline Float D2D1StrokeStyle::GetMiterLimit() const {
    DebugAssert( IsCreated() );
    return m_hDesc.fMiterLimit;
}

inline D2D1StrokeDashStyle D2D1StrokeStyle::GetDashStyle() const {
    DebugAssert( IsCreated() );
    return m_hDesc.iDashStyle;
}
inline Float D2D1StrokeStyle::GetDashOffset() const {
    DebugAssert( IsCreated() );
    return m_hDesc.fDashOffset;
}

inline const D2D1StrokeStyleDesc * D2D1StrokeStyle::GetDesc() const {
    DebugAssert( IsCreated() );
    return &m_hDesc;
}




