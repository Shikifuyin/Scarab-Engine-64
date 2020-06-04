/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/2D/Resources/D2D1Brush.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : D2D1 Dev-Dep Resource : Brushes.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// D2D1Brush implementation
inline Bool D2D1Brush::IsCreated() const {
    return ( m_pBrush != NULL || m_bTemporaryDestroyed );
}

inline Float D2D1Brush::GetOpacity() const {
    return m_hBrushDesc.fOpacity;
}

inline Void D2D1Brush::GetTransform( D2D1Matrix32 * outTransform ) const {
    outTransform->f00 = m_hBrushDesc.matTransform.f00;
    outTransform->f01 = m_hBrushDesc.matTransform.f01;
    outTransform->f10 = m_hBrushDesc.matTransform.f10;
    outTransform->f11 = m_hBrushDesc.matTransform.f11;
    outTransform->f20 = m_hBrushDesc.matTransform.f20;
    outTransform->f21 = m_hBrushDesc.matTransform.f21;
}

inline const D2D1BrushDesc * D2D1Brush::GetDesc() const {
    return &m_hBrushDesc;
}

/////////////////////////////////////////////////////////////////////////////////
// D2D1SolidColorBrush implementation
inline D2D1BrushType D2D1SolidColorBrush::GetType() const {
    return D2D1BRUSH_SOLID_COLOR;
}

inline const D2D1Color * D2D1SolidColorBrush::GetColor() const {
    return &m_hBrushColor;
}
inline Void D2D1SolidColorBrush::GetColor( D2D1Color * outColor ) const {
    outColor->R = m_hBrushColor.R;
    outColor->G = m_hBrushColor.G;
    outColor->B = m_hBrushColor.B;
    outColor->A = m_hBrushColor.A;
}

/////////////////////////////////////////////////////////////////////////////////
// D2D1GradientBrush implementation
inline D2D1BrushGamma D2D1GradientBrush::GetGradientGammaMode() const {
    DebugAssert( IsCreated() );
    return m_hGradientDesc.iGamma;
}
inline D2D1BrushWrapMode D2D1GradientBrush::GetGradientWrapMode() const {
    DebugAssert( IsCreated() );
    return m_hGradientDesc.iWrapMode;
}

inline UInt D2D1GradientBrush::GetGradientStopCount() const {
    DebugAssert( IsCreated() );
    return m_hGradientDesc.iStopCount;
}
inline const D2D1GradientStop * D2D1GradientBrush::GetGradientStop( UInt iStop ) const {
    DebugAssert( IsCreated() );
    DebugAssert( iStop < m_hGradientDesc.iStopCount );
    return (const D2D1GradientStop *)( m_hGradientDesc.arrGradientStops + iStop );
}
inline Void D2D1GradientBrush::GetGradientStop( D2D1GradientStop * outGradientStop, UInt iStop ) const {
    DebugAssert( IsCreated() );
    DebugAssert( iStop < m_hGradientDesc.iStopCount );
    outGradientStop->fPosition = m_hGradientDesc.arrGradientStops[iStop].fPosition;
    outGradientStop->fColor.R = m_hGradientDesc.arrGradientStops[iStop].fColor.R;
    outGradientStop->fColor.G = m_hGradientDesc.arrGradientStops[iStop].fColor.G;
    outGradientStop->fColor.B = m_hGradientDesc.arrGradientStops[iStop].fColor.B;
    outGradientStop->fColor.A = m_hGradientDesc.arrGradientStops[iStop].fColor.A;
}

inline const D2D1GradientDesc * D2D1GradientBrush::GetGradientDesc() const {
    DebugAssert( IsCreated() );
    return &m_hGradientDesc;
}

/////////////////////////////////////////////////////////////////////////////////
// D2D1LinearGradientBrush implementation
inline D2D1BrushType D2D1LinearGradientBrush::GetType() const {
    return D2D1BRUSH_GRADIENT_LINEAR;
}

inline const D2D1Point * D2D1LinearGradientBrush::GetStartPoint() const {
    return &(m_hLinearGradientDesc.ptStart);
}
inline Void D2D1LinearGradientBrush::GetStartPoint( D2D1Point * outStart ) const {
    outStart->fX = m_hLinearGradientDesc.ptStart.fX;
    outStart->fY = m_hLinearGradientDesc.ptStart.fY;
}

inline const D2D1Point * D2D1LinearGradientBrush::GetEndPoint() const {
    return &(m_hLinearGradientDesc.ptEnd);
}
inline Void D2D1LinearGradientBrush::GetEndPoint( D2D1Point * outEnd ) const {
    outEnd->fX = m_hLinearGradientDesc.ptEnd.fX;
    outEnd->fY = m_hLinearGradientDesc.ptEnd.fY;
}

inline const D2D1BrushLinearGradientDesc * D2D1LinearGradientBrush::GetLinearGradientDesc() const {
    return &m_hLinearGradientDesc;
}

/////////////////////////////////////////////////////////////////////////////////
// D2D1RadialGradientBrush implementation
inline D2D1BrushType D2D1RadialGradientBrush::GetType() const {
    return D2D1BRUSH_GRADIENT_RADIAL;
}

inline const D2D1Point * D2D1RadialGradientBrush::GetCenter() const {
    return &(m_hRadialGradientDesc.ptCenter);
}

inline const D2D1Point * D2D1RadialGradientBrush::GetOffset() const {
    return &(m_hRadialGradientDesc.ptOffset);
}

inline Float D2D1RadialGradientBrush::GetRadiusX() const {
    return m_hRadialGradientDesc.fRadiusX;
}
inline Float D2D1RadialGradientBrush::GetRadiusY() const {
    return m_hRadialGradientDesc.fRadiusY;
}

inline const D2D1BrushRadialGradientDesc * D2D1RadialGradientBrush::GetRadialGradientDesc() const {
    return &m_hRadialGradientDesc;
}

/////////////////////////////////////////////////////////////////////////////////
// D2D1BitmapBrush implementation
inline D2D1BrushType D2D1BitmapBrush::GetType() const {
    return D2D1BRUSH_BITMAP;
}

inline D2D1Bitmap * D2D1BitmapBrush::GetBitmap() const {
    return m_pBitmap;
}

inline D2D1BitmapInterpolationMode D2D1BitmapBrush::GetInterpolationMode() const {
    return m_hBitmapDesc.iInterpolationMode;
}

inline D2D1BrushWrapMode D2D1BitmapBrush::GetWrapModeX() const {
    return m_hBitmapDesc.iWrapModeX;
}
inline D2D1BrushWrapMode D2D1BitmapBrush::GetWrapModeY() const {
    return m_hBitmapDesc.iWrapModeY;
}

inline const D2D1BrushBitmapDesc * D2D1BitmapBrush::GetBrushBitmapDesc() const {
    return &m_hBitmapDesc;
}

