/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/2D/D2D1RenderingContext.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Platform-dependant abstraction for 2D graphics.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// D2D1RenderingContext implementation
inline Bool D2D1RenderingContext::IsCreated() const {
    return ( m_pD2D1RenderingContext != NULL || m_bTemporaryDestroyed );
}
inline Bool D2D1RenderingContext::IsBoundToBackBuffer( UInt * outBackBuffer ) const {
    DebugAssert( IsCreated() );
    if ( outBackBuffer != NULL )
        *outBackBuffer = m_iBoundToBackBuffer;
    return ( m_iBoundToBackBuffer != INVALID_OFFSET );
}

inline Float D2D1RenderingContext::GetDpiX() const {
    return m_fDpiX;
}
inline Float D2D1RenderingContext::GetDpiY() const {
    return m_fDpiY;
}

inline D2D1AntialiasingMode D2D1RenderingContext::GetAntialiasingMode() const {
    return m_hRenderStateDesc.iAntialiasingMode;
}

inline const D2D1Matrix32 * D2D1RenderingContext::GetTransform() const {
    return &(m_hRenderStateDesc.matTransform);
}

inline D2D1Tag D2D1RenderingContext::GetTag1() const {
    return m_hRenderStateDesc.iTag1;
}
inline D2D1Tag D2D1RenderingContext::GetTag2() const {
    return m_hRenderStateDesc.iTag2;
}

inline D2D1TextAntialiasingMode D2D1RenderingContext::GetTextAntialiasingMode() const {
    return m_hRenderStateDesc.iTextAntialiasingMode;
}

inline Void D2D1RenderingContext::GetTextRenderState( D2D1TextRenderState * outTextRenderState ) const {
    DebugAssert( !(outTextRenderState->IsCreated()) );
    outTextRenderState->m_pTextRenderingParams = m_pTextRenderingParams;
}

inline Void D2D1RenderingContext::DrawCircle( const D2D1Point * pCenter, Float fRadius, const D2D1Brush * pBrush, Float fStrokeWidth, const D2D1StrokeStyle * pStrokeStyle ) {
    D2D1Ellipse hEllipse;
    hEllipse.hCenter.fX = pCenter->fX;
    hEllipse.hCenter.fY = pCenter->fY;
    hEllipse.fRadiusX = fRadius;
    hEllipse.fRadiusY = fRadius;
    DrawEllipse( &hEllipse, pBrush, fStrokeWidth, pStrokeStyle );
}
inline Void D2D1RenderingContext::FillCircle( const D2D1Point * pCenter, Float fRadius, const D2D1Brush * pBrush ) {
    D2D1Ellipse hEllipse;
    hEllipse.hCenter.fX = pCenter->fX;
    hEllipse.hCenter.fY = pCenter->fY;
    hEllipse.fRadiusX = fRadius;
    hEllipse.fRadiusY = fRadius;
    FillEllipse( &hEllipse, pBrush );
}



