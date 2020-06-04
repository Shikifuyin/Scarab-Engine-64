/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/2D/Resources/D2D1StrokeStyle.cpp
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
// Third-Party Includes
#pragma warning(disable:4005)

#define WIN32_LEAN_AND_MEAN
#include <d2d1.h>

#undef DebugAssert

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "D2D1StrokeStyle.h"

#include "../D2D1RenderingContext.h"

/////////////////////////////////////////////////////////////////////////////////
// D2D1StrokeStyle implementation
D2D1StrokeStyle::D2D1StrokeStyle()
{
    m_pStrokeStyle = NULL;

    m_hDesc.iStartCap = D2D1STROKE_CAPSTYLE_FLAT;
    m_hDesc.iEndCap = D2D1STROKE_CAPSTYLE_FLAT;
    m_hDesc.iDashCap = D2D1STROKE_CAPSTYLE_FLAT;
    m_hDesc.iLineJoin = D2D1STROKE_LINEJOIN_MITER;
    m_hDesc.fMiterLimit = 0.0f;
    m_hDesc.iDashStyle = D2D1STROKE_DASHSTYLE_SOLID;
    m_hDesc.fDashOffset = 0.0f;
}
D2D1StrokeStyle::~D2D1StrokeStyle()
{
    if ( IsCreated() )
        Destroy();
}

Void D2D1StrokeStyle::Create( const D2D1StrokeStyleDesc * pDesc, const Float * arrDashes, UInt iDashCount )
{
    DebugAssert( !(IsCreated()) );

    MemCopy( &m_hDesc, pDesc, sizeof(D2D1StrokeStyleDesc) );

    D2D1_STROKE_STYLE_PROPERTIES hStrokeStyleProperties;
    m_hDesc.ConvertTo( &hStrokeStyleProperties );

    if ( hStrokeStyleProperties.dashStyle == D2D1_DASH_STYLE_CUSTOM ) {
        DebugAssert( arrDashes != NULL && iDashCount > 0 );
    } else {
        DebugAssert( arrDashes == NULL && iDashCount == 0 );
    }

    m_pStrokeStyle = NULL;
    HRESULT hRes = ((ID2D1Factory*)(D2D1RenderingContext::sm_pD2D1Factory))->CreateStrokeStyle( &hStrokeStyleProperties, arrDashes, iDashCount, (ID2D1StrokeStyle**)&m_pStrokeStyle );
    DebugAssert( hRes == S_OK && m_pStrokeStyle != NULL );
}
Void D2D1StrokeStyle::Destroy()
{
    DebugAssert( IsCreated() );

    ((ID2D1StrokeStyle*)m_pStrokeStyle)->Release();
    m_pStrokeStyle = NULL;

    m_hDesc.iStartCap = D2D1STROKE_CAPSTYLE_FLAT;
    m_hDesc.iEndCap = D2D1STROKE_CAPSTYLE_FLAT;
    m_hDesc.iDashCap = D2D1STROKE_CAPSTYLE_FLAT;
    m_hDesc.iLineJoin = D2D1STROKE_LINEJOIN_MITER;
    m_hDesc.fMiterLimit = 0.0f;
    m_hDesc.iDashStyle = D2D1STROKE_DASHSTYLE_SOLID;
    m_hDesc.fDashOffset = 0.0f;
}

UInt D2D1StrokeStyle::GetDashCount() const
{
    DebugAssert( IsCreated() );
    DebugAssert( m_hDesc.iDashStyle == D2D1STROKE_DASHSTYLE_CUSTOM );

    return ((ID2D1StrokeStyle*)m_pStrokeStyle)->GetDashesCount();
}
Void D2D1StrokeStyle::GetDashes( Float * outDashes, UInt iMaxDashes ) const
{
    DebugAssert( IsCreated() );
    DebugAssert( m_hDesc.iDashStyle == D2D1STROKE_DASHSTYLE_CUSTOM );
    
    ((ID2D1StrokeStyle*)m_pStrokeStyle)->GetDashes( outDashes, iMaxDashes );
}

