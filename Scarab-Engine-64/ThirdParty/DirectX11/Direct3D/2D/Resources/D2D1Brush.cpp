/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/2D/Resources/D2D1Brush.cpp
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
// Third-Party Includes
#pragma warning(disable:4005)

#define WIN32_LEAN_AND_MEAN
#include <d2d1.h>

#undef DebugAssert

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "D2D1Brush.h"

#include "../D2D1RenderingContext.h"

/////////////////////////////////////////////////////////////////////////////////
// D2D1Brush implementation
D2D1Brush::D2D1Brush( D2D1RenderingContext * pContext2D )
{
    m_pContext2D = pContext2D;

    m_pBrush = NULL;

    m_hBrushDesc.fOpacity = 1.0f;
    m_hBrushDesc.matTransform.f00 = 1.0f;
    m_hBrushDesc.matTransform.f01 = 0.0f;
    m_hBrushDesc.matTransform.f10 = 0.0f;
    m_hBrushDesc.matTransform.f11 = 1.0f;
    m_hBrushDesc.matTransform.f20 = 0.0f;
    m_hBrushDesc.matTransform.f21 = 0.0f;

    m_bTemporaryDestroyed = false;
}
D2D1Brush::~D2D1Brush()
{
    // nothing to do
}

Void D2D1Brush::Destroy()
{
    DebugAssert( IsCreated() );

    if ( m_bTemporaryDestroyed )
        m_bTemporaryDestroyed = false;
    else
        _NakedDestroy();
}

Void D2D1Brush::OnDestroyDevice()
{
    DebugAssert( !m_bTemporaryDestroyed );

    if ( m_pBrush != NULL ) {
        _NakedDestroy();
        m_bTemporaryDestroyed = true;
    }
}
Void D2D1Brush::OnRestoreDevice()
{
    DebugAssert( m_pBrush == NULL );

    if ( m_bTemporaryDestroyed ) {
        _NakedCreate();
        m_bTemporaryDestroyed = false;
    }
}

Void D2D1Brush::SetOpacity( Float fOpacity )
{
    m_hBrushDesc.fOpacity = fOpacity;

    D2D1_BRUSH_PROPERTIES hD2D1Desc;
    m_hBrushDesc.ConvertTo( &hD2D1Desc );

    if ( IsCreated() )
        ((ID2D1Brush*)m_pBrush)->SetOpacity( hD2D1Desc.opacity );
}

Void D2D1Brush::SetTransform( const D2D1Matrix32 * pTransform )
{
    m_hBrushDesc.matTransform.f00 = pTransform->f00;
    m_hBrushDesc.matTransform.f01 = pTransform->f01;
    m_hBrushDesc.matTransform.f10 = pTransform->f10;
    m_hBrushDesc.matTransform.f11 = pTransform->f11;
    m_hBrushDesc.matTransform.f20 = pTransform->f20;
    m_hBrushDesc.matTransform.f21 = pTransform->f21;

    D2D1_BRUSH_PROPERTIES hD2D1Desc;
    m_hBrushDesc.ConvertTo( &hD2D1Desc );

    if ( IsCreated() )
        ((ID2D1Brush*)m_pBrush)->SetTransform( hD2D1Desc.transform );
}

Void D2D1Brush::SetDesc( const D2D1BrushDesc * pDesc )
{
    MemCopy( &m_hBrushDesc, pDesc, sizeof(D2D1BrushDesc) );

    D2D1_BRUSH_PROPERTIES hD2D1Desc;
    m_hBrushDesc.ConvertTo( &hD2D1Desc );

    if ( IsCreated() ) {
        ((ID2D1Brush*)m_pBrush)->SetOpacity( hD2D1Desc.opacity );
        ((ID2D1Brush*)m_pBrush)->SetTransform( hD2D1Desc.transform );
    }
}

/////////////////////////////////////////////////////////////////////////////////
// D2D1SolidColorBrush implementation
D2D1SolidColorBrush::D2D1SolidColorBrush( D2D1RenderingContext * pContext2D ):
    D2D1Brush( pContext2D )
{
    m_pSolidColorBrush = NULL;

    m_hBrushColor.R = 0.0f;
    m_hBrushColor.G = 0.0f;
    m_hBrushColor.B = 0.0f;
    m_hBrushColor.A = 1.0f;
}
D2D1SolidColorBrush::~D2D1SolidColorBrush()
{
    if ( IsCreated() )
        Destroy();
}

Void D2D1SolidColorBrush::Create()
{
    DebugAssert( !(IsCreated()) );

    _NakedCreate();
}

Void D2D1SolidColorBrush::SetColor( const D2D1Color * pColor )
{
    m_hBrushColor.R = pColor->R;
    m_hBrushColor.G = pColor->G;
    m_hBrushColor.B = pColor->B;
    m_hBrushColor.A = pColor->A;

    if ( IsCreated() )
        ((ID2D1SolidColorBrush*)m_pSolidColorBrush)->SetColor( (const D2D1_COLOR_F *)&m_hBrushColor );
}

/////////////////////////////////////////////////////////////////////////////////

Void D2D1SolidColorBrush::_NakedCreate()
{
    D2D1_BRUSH_PROPERTIES hD2D1Desc;
    m_hBrushDesc.ConvertTo( &hD2D1Desc );

    m_pSolidColorBrush = NULL;
    HRESULT hRes = ((ID2D1RenderTarget*)(m_pContext2D->m_pD2D1RenderingContext))->CreateSolidColorBrush( (const D2D1_COLOR_F *)&m_hBrushColor, &hD2D1Desc, (ID2D1SolidColorBrush**)&m_pSolidColorBrush );
    DebugAssert( hRes == S_OK && m_pSolidColorBrush != NULL );

    m_pBrush = NULL;
    hRes = ((ID2D1SolidColorBrush*)m_pSolidColorBrush)->QueryInterface( __uuidof(ID2D1Brush), &m_pBrush );
    DebugAssert( hRes == S_OK && m_pBrush != NULL );
}
Void D2D1SolidColorBrush::_NakedDestroy()
{
    ((ID2D1Brush*)m_pBrush)->Release();
    m_pBrush = NULL;

    ((ID2D1SolidColorBrush*)m_pSolidColorBrush)->Release();
    m_pSolidColorBrush = NULL;
}

/////////////////////////////////////////////////////////////////////////////////
// D2D1GradientBrush implementation
D2D1GradientBrush::D2D1GradientBrush( D2D1RenderingContext * pContext2D ):
    D2D1Brush( pContext2D )
{
    m_pGradientStopCollection = NULL;

    m_hGradientDesc.iGamma = D2D1BRUSH_GAMMA_2_2;
    m_hGradientDesc.iWrapMode = D2D1BRUSH_WRAPMODE_CLAMP;
    m_hGradientDesc.iStopCount = 2;

    m_hGradientDesc.arrGradientStops[0].fPosition = 0.0f;
    m_hGradientDesc.arrGradientStops[0].fColor.R = 0.0f;
    m_hGradientDesc.arrGradientStops[0].fColor.G = 0.0f;
    m_hGradientDesc.arrGradientStops[0].fColor.B = 0.0f;
    m_hGradientDesc.arrGradientStops[0].fColor.A = 1.0f;

    m_hGradientDesc.arrGradientStops[1].fPosition = 1.0f;
    m_hGradientDesc.arrGradientStops[1].fColor.R = 1.0f;
    m_hGradientDesc.arrGradientStops[1].fColor.G = 1.0f;
    m_hGradientDesc.arrGradientStops[1].fColor.B = 1.0f;
    m_hGradientDesc.arrGradientStops[1].fColor.A = 1.0f;
}
D2D1GradientBrush::~D2D1GradientBrush()
{
    // nothing to do
}

Void D2D1GradientBrush::GetGradientStops( D2D1GradientStop * outGradientStops, UInt iMaxStops ) const
{
    DebugAssert( IsCreated() );
    DebugAssert( iMaxStops <= m_hGradientDesc.iStopCount );

    for( UInt i = 0; i < iMaxStops; ++i ) {
        outGradientStops[i].fPosition = m_hGradientDesc.arrGradientStops[i].fPosition;
        outGradientStops[i].fColor.R = m_hGradientDesc.arrGradientStops[i].fColor.R;
        outGradientStops[i].fColor.G = m_hGradientDesc.arrGradientStops[i].fColor.G;
        outGradientStops[i].fColor.B = m_hGradientDesc.arrGradientStops[i].fColor.B;
        outGradientStops[i].fColor.A = m_hGradientDesc.arrGradientStops[i].fColor.A;
    }
}

/////////////////////////////////////////////////////////////////////////////////

Void D2D1GradientBrush::_CreateGradient( const D2D1GradientDesc * pGradientDesc )
{
    DebugAssert( m_pGradientStopCollection == NULL );
    DebugAssert( pGradientDesc->iStopCount <= D2D1GRADIENT_MAX_STOPS );

    MemCopy( &m_hGradientDesc, pGradientDesc, sizeof(D2D1GradientDesc) );
}
Void D2D1GradientBrush::_DestroyGradient()
{
    DebugAssert( m_pGradientStopCollection != NULL );

    m_hGradientDesc.iGamma = D2D1BRUSH_GAMMA_2_2;
    m_hGradientDesc.iWrapMode = D2D1BRUSH_WRAPMODE_CLAMP;
    m_hGradientDesc.iStopCount = 2;

    m_hGradientDesc.arrGradientStops[0].fPosition = 0.0f;
    m_hGradientDesc.arrGradientStops[0].fColor.R = 0.0f;
    m_hGradientDesc.arrGradientStops[0].fColor.G = 0.0f;
    m_hGradientDesc.arrGradientStops[0].fColor.B = 0.0f;
    m_hGradientDesc.arrGradientStops[0].fColor.A = 1.0f;

    m_hGradientDesc.arrGradientStops[1].fPosition = 1.0f;
    m_hGradientDesc.arrGradientStops[1].fColor.R = 1.0f;
    m_hGradientDesc.arrGradientStops[1].fColor.G = 1.0f;
    m_hGradientDesc.arrGradientStops[1].fColor.B = 1.0f;
    m_hGradientDesc.arrGradientStops[1].fColor.A = 1.0f;
}
Void D2D1GradientBrush::_NakedCreateGradient()
{
    D2D1_GAMMA iGamma = (D2D1_GAMMA)( D2D1BrushGammaToD2D1[m_hGradientDesc.iGamma] );
    D2D1_EXTEND_MODE iWrapMode = (D2D1_EXTEND_MODE)( D2D1BrushWrapModeToD2D1[m_hGradientDesc.iWrapMode] );

    m_pGradientStopCollection = NULL;
    HRESULT hRes = ((ID2D1RenderTarget*)(m_pContext2D->m_pD2D1RenderingContext))->CreateGradientStopCollection( (const D2D1_GRADIENT_STOP *)m_hGradientDesc.arrGradientStops, m_hGradientDesc.iStopCount,
                                                                                                                iGamma, iWrapMode, (ID2D1GradientStopCollection**)&m_pGradientStopCollection );
    DebugAssert( hRes == S_OK && m_pGradientStopCollection != NULL );
}
Void D2D1GradientBrush::_NakedDestroyGradient()
{
    ((ID2D1GradientStopCollection*)m_pGradientStopCollection)->Release();
    m_pGradientStopCollection = NULL;
}

/////////////////////////////////////////////////////////////////////////////////
// D2D1LinearGradientBrush implementation
D2D1LinearGradientBrush::D2D1LinearGradientBrush( D2D1RenderingContext * pContext2D ):
    D2D1GradientBrush( pContext2D )
{
    m_pLinearGradientBrush = NULL;

    m_hLinearGradientDesc.ptStart.fX = 0.0f;
    m_hLinearGradientDesc.ptStart.fY = 0.0f;
    m_hLinearGradientDesc.ptEnd.fX = 100.0f;
    m_hLinearGradientDesc.ptEnd.fY = 0.0f;
}
D2D1LinearGradientBrush::~D2D1LinearGradientBrush()
{
    if ( IsCreated() )
        Destroy();
}

Void D2D1LinearGradientBrush::Create( const D2D1GradientDesc * pGradientDesc )
{
    DebugAssert( !(IsCreated()) );

    _CreateGradient( pGradientDesc );
    _NakedCreate();
}
Void D2D1LinearGradientBrush::Destroy()
{
    D2D1Brush::Destroy();
    _DestroyGradient();
}

Void D2D1LinearGradientBrush::SetStartPoint( const D2D1Point * pStart )
{
    m_hLinearGradientDesc.ptStart.fX = pStart->fX;
    m_hLinearGradientDesc.ptStart.fY = pStart->fY;

    D2D1_LINEAR_GRADIENT_BRUSH_PROPERTIES hD2D1Desc;
    m_hLinearGradientDesc.ConvertTo( &hD2D1Desc );

    if ( IsCreated() )
        ((ID2D1LinearGradientBrush*)m_pLinearGradientBrush)->SetStartPoint( hD2D1Desc.startPoint );
}

Void D2D1LinearGradientBrush::SetEndPoint( const D2D1Point * pEnd )
{
    m_hLinearGradientDesc.ptEnd.fX = pEnd->fX;
    m_hLinearGradientDesc.ptEnd.fY = pEnd->fY;

    D2D1_LINEAR_GRADIENT_BRUSH_PROPERTIES hD2D1Desc;
    m_hLinearGradientDesc.ConvertTo( &hD2D1Desc );

    if ( IsCreated() )
        ((ID2D1LinearGradientBrush*)m_pLinearGradientBrush)->SetEndPoint( hD2D1Desc.endPoint );
}

Void D2D1LinearGradientBrush::SetLinearGradientDesc( const D2D1BrushLinearGradientDesc * pLinearGradientDesc )
{
    MemCopy( &m_hLinearGradientDesc, pLinearGradientDesc, sizeof(D2D1BrushLinearGradientDesc) );

    D2D1_LINEAR_GRADIENT_BRUSH_PROPERTIES hD2D1Desc;
    m_hLinearGradientDesc.ConvertTo( &hD2D1Desc );

    if ( IsCreated() ) {
        ((ID2D1LinearGradientBrush*)m_pLinearGradientBrush)->SetStartPoint( hD2D1Desc.startPoint );
        ((ID2D1LinearGradientBrush*)m_pLinearGradientBrush)->SetEndPoint( hD2D1Desc.endPoint );
    }
}

/////////////////////////////////////////////////////////////////////////////////

Void D2D1LinearGradientBrush::_NakedCreate()
{
    _NakedCreateGradient();

    D2D1_LINEAR_GRADIENT_BRUSH_PROPERTIES hD2D1DescLinear;
    m_hLinearGradientDesc.ConvertTo( &hD2D1DescLinear );

    D2D1_BRUSH_PROPERTIES hD2D1Desc;
    m_hBrushDesc.ConvertTo( &hD2D1Desc );

    m_pLinearGradientBrush = NULL;
    HRESULT hRes = ((ID2D1RenderTarget*)(m_pContext2D->m_pD2D1RenderingContext))->CreateLinearGradientBrush( &hD2D1DescLinear, &hD2D1Desc, (ID2D1GradientStopCollection*)m_pGradientStopCollection,
                                                                                                             (ID2D1LinearGradientBrush**)&m_pLinearGradientBrush );
    DebugAssert( hRes == S_OK && m_pLinearGradientBrush != NULL );

    m_pBrush = NULL;
    hRes = ((ID2D1LinearGradientBrush*)m_pLinearGradientBrush)->QueryInterface( __uuidof(ID2D1Brush), &m_pBrush );
    DebugAssert( hRes == S_OK && m_pBrush != NULL );
}
Void D2D1LinearGradientBrush::_NakedDestroy()
{
    ((ID2D1Brush*)m_pBrush)->Release();
    m_pBrush = NULL;

    ((ID2D1LinearGradientBrush*)m_pLinearGradientBrush)->Release();
    m_pLinearGradientBrush = NULL;

    _NakedDestroyGradient();
}

/////////////////////////////////////////////////////////////////////////////////
// D2D1RadialGradientBrush implementation
D2D1RadialGradientBrush::D2D1RadialGradientBrush( D2D1RenderingContext * pContext2D ):
    D2D1GradientBrush( pContext2D )
{
    m_pRadialGradientBrush = NULL;

    m_hRadialGradientDesc.ptCenter.fX = 0.0f;
    m_hRadialGradientDesc.ptCenter.fY = 0.0f;
    m_hRadialGradientDesc.ptOffset.fX = 0.0f;
    m_hRadialGradientDesc.ptOffset.fY = 0.0f;
    m_hRadialGradientDesc.fRadiusX = 1.0f;
    m_hRadialGradientDesc.fRadiusY = 1.0f;
}
D2D1RadialGradientBrush::~D2D1RadialGradientBrush()
{
    if ( IsCreated() )
        Destroy();
}

Void D2D1RadialGradientBrush::Create( const D2D1GradientDesc * pGradientDesc )
{
    DebugAssert( !(IsCreated()) );

    _CreateGradient( pGradientDesc );
    _NakedCreate();
}
Void D2D1RadialGradientBrush::Destroy()
{
    D2D1Brush::Destroy();
    _DestroyGradient();
}

Void D2D1RadialGradientBrush::SetCenter( const D2D1Point * pCenter )
{
    m_hRadialGradientDesc.ptCenter.fX = pCenter->fX;
    m_hRadialGradientDesc.ptCenter.fY = pCenter->fY;

    D2D1_RADIAL_GRADIENT_BRUSH_PROPERTIES hD2D1Desc;
    m_hRadialGradientDesc.ConvertTo( &hD2D1Desc );

    if ( IsCreated() )
        ((ID2D1RadialGradientBrush*)m_pRadialGradientBrush)->SetCenter( hD2D1Desc.center );
}

Void D2D1RadialGradientBrush::SetOffset( const D2D1Point * pOffset )
{
    m_hRadialGradientDesc.ptOffset.fX = pOffset->fX;
    m_hRadialGradientDesc.ptOffset.fY = pOffset->fY;

    D2D1_RADIAL_GRADIENT_BRUSH_PROPERTIES hD2D1Desc;
    m_hRadialGradientDesc.ConvertTo( &hD2D1Desc );

    if ( IsCreated() )
        ((ID2D1RadialGradientBrush*)m_pRadialGradientBrush)->SetGradientOriginOffset( hD2D1Desc.gradientOriginOffset );
}

Void D2D1RadialGradientBrush::SetRadiusX( Float fRadiusX )
{
    m_hRadialGradientDesc.fRadiusX = fRadiusX;

    if ( IsCreated() )
        ((ID2D1RadialGradientBrush*)m_pRadialGradientBrush)->SetRadiusX( m_hRadialGradientDesc.fRadiusX );
}
Void D2D1RadialGradientBrush::SetRadiusY( Float fRadiusY )
{
    m_hRadialGradientDesc.fRadiusY = fRadiusY;

    if ( IsCreated() )
        ((ID2D1RadialGradientBrush*)m_pRadialGradientBrush)->SetRadiusY( m_hRadialGradientDesc.fRadiusY );
}

Void D2D1RadialGradientBrush::SetRadialGradientDesc( const D2D1BrushRadialGradientDesc * pRadialGradientDesc )
{
    MemCopy( &m_hRadialGradientDesc, pRadialGradientDesc, sizeof(D2D1BrushRadialGradientDesc) );

    D2D1_RADIAL_GRADIENT_BRUSH_PROPERTIES hD2D1Desc;
    m_hRadialGradientDesc.ConvertTo( &hD2D1Desc );

    if ( IsCreated() ) {
        ((ID2D1RadialGradientBrush*)m_pRadialGradientBrush)->SetCenter( hD2D1Desc.center );
        ((ID2D1RadialGradientBrush*)m_pRadialGradientBrush)->SetGradientOriginOffset( hD2D1Desc.gradientOriginOffset );
        ((ID2D1RadialGradientBrush*)m_pRadialGradientBrush)->SetRadiusX( hD2D1Desc.radiusX );
        ((ID2D1RadialGradientBrush*)m_pRadialGradientBrush)->SetRadiusY( hD2D1Desc.radiusY );
    }
}

/////////////////////////////////////////////////////////////////////////////////

Void D2D1RadialGradientBrush::_NakedCreate()
{
    _NakedCreateGradient();

    D2D1_RADIAL_GRADIENT_BRUSH_PROPERTIES hD2D1DescRadial;
    m_hRadialGradientDesc.ConvertTo( &hD2D1DescRadial );

    D2D1_BRUSH_PROPERTIES hD2D1Desc;
    m_hBrushDesc.ConvertTo( &hD2D1Desc );

    m_pRadialGradientBrush = NULL;
    HRESULT hRes = ((ID2D1RenderTarget*)(m_pContext2D->m_pD2D1RenderingContext))->CreateRadialGradientBrush( &hD2D1DescRadial, &hD2D1Desc, (ID2D1GradientStopCollection*)m_pGradientStopCollection,
                                                                                                             (ID2D1RadialGradientBrush**)&m_pRadialGradientBrush );
    DebugAssert( hRes == S_OK && m_pRadialGradientBrush != NULL );

    m_pBrush = NULL;
    hRes = ((ID2D1RadialGradientBrush*)m_pRadialGradientBrush)->QueryInterface( __uuidof(ID2D1Brush), &m_pBrush );
    DebugAssert( hRes == S_OK && m_pBrush != NULL );
}
Void D2D1RadialGradientBrush::_NakedDestroy()
{
    ((ID2D1Brush*)m_pBrush)->Release();
    m_pBrush = NULL;

    ((ID2D1RadialGradientBrush*)m_pRadialGradientBrush)->Release();
    m_pRadialGradientBrush = NULL;

    _NakedDestroyGradient();
}

/////////////////////////////////////////////////////////////////////////////////
// D2D1BitmapBrush implementation
D2D1BitmapBrush::D2D1BitmapBrush( D2D1RenderingContext * pContext2D ):
    D2D1Brush( pContext2D )
{
    m_pBitmap = NULL;

    m_pBitmapBrush = NULL;

    m_hBitmapDesc.iInterpolationMode = D2D1BITMAP_INTERPOLATION_NEAREST;
    m_hBitmapDesc.iWrapModeX = D2D1BRUSH_WRAPMODE_CLAMP;
    m_hBitmapDesc.iWrapModeY = D2D1BRUSH_WRAPMODE_CLAMP;
}
D2D1BitmapBrush::~D2D1BitmapBrush()
{
    if ( IsCreated() )
        Destroy();
}

Void D2D1BitmapBrush::Create()
{
    DebugAssert( !(IsCreated()) );
    DebugAssert( m_pBitmap != NULL && m_pBitmap->IsCreated() );

    _NakedCreate();
}

Void D2D1BitmapBrush::SetBitmap( D2D1Bitmap * pBitmap )
{
    DebugAssert( m_pBitmap != NULL && m_pBitmap->IsCreated() );

    m_pBitmap = pBitmap;

    if ( IsCreated() )
        ((ID2D1BitmapBrush*)m_pBitmapBrush)->SetBitmap( (ID2D1Bitmap*)(m_pBitmap->m_pBitmap) );
}

Void D2D1BitmapBrush::SetInterpolationMode( D2D1BitmapInterpolationMode iInterpolationMode )
{
    m_hBitmapDesc.iInterpolationMode = iInterpolationMode;

    D2D1_BITMAP_BRUSH_PROPERTIES hD2D1Desc;
    m_hBitmapDesc.ConvertTo( &hD2D1Desc );

    if ( IsCreated() )
        ((ID2D1BitmapBrush*)m_pBitmapBrush)->SetInterpolationMode( hD2D1Desc.interpolationMode );
}

Void D2D1BitmapBrush::SetWrapModeX( D2D1BrushWrapMode iWrapModeX )
{
    m_hBitmapDesc.iWrapModeX = iWrapModeX;

    D2D1_BITMAP_BRUSH_PROPERTIES hD2D1Desc;
    m_hBitmapDesc.ConvertTo( &hD2D1Desc );

    if ( IsCreated() )
        ((ID2D1BitmapBrush*)m_pBitmapBrush)->SetExtendModeX( hD2D1Desc.extendModeX );
}
Void D2D1BitmapBrush::SetWrapModeY( D2D1BrushWrapMode iWrapModeY )
{
    m_hBitmapDesc.iWrapModeY = iWrapModeY;

    D2D1_BITMAP_BRUSH_PROPERTIES hD2D1Desc;
    m_hBitmapDesc.ConvertTo( &hD2D1Desc );

    if ( IsCreated() )
        ((ID2D1BitmapBrush*)m_pBitmapBrush)->SetExtendModeY( hD2D1Desc.extendModeY );
}

Void D2D1BitmapBrush::SetBrushBitmapDesc( const D2D1BrushBitmapDesc * pBitmapDesc )
{
    MemCopy( &m_hBitmapDesc, pBitmapDesc, sizeof(D2D1BrushBitmapDesc) );

    D2D1_BITMAP_BRUSH_PROPERTIES hD2D1Desc;
    m_hBitmapDesc.ConvertTo( &hD2D1Desc );

    if ( IsCreated() ) {
        ((ID2D1BitmapBrush*)m_pBitmapBrush)->SetInterpolationMode( hD2D1Desc.interpolationMode );
        ((ID2D1BitmapBrush*)m_pBitmapBrush)->SetExtendModeX( hD2D1Desc.extendModeX );
        ((ID2D1BitmapBrush*)m_pBitmapBrush)->SetExtendModeY( hD2D1Desc.extendModeY );
    }
}

/////////////////////////////////////////////////////////////////////////////////

Void D2D1BitmapBrush::_NakedCreate()
{
    D2D1_BITMAP_BRUSH_PROPERTIES hD2D1DescBitmap;
    m_hBitmapDesc.ConvertTo( &hD2D1DescBitmap );

    D2D1_BRUSH_PROPERTIES hD2D1Desc;
    m_hBrushDesc.ConvertTo( &hD2D1Desc );

    m_pBitmapBrush = NULL;
    HRESULT hRes = ((ID2D1RenderTarget*)(m_pContext2D->m_pD2D1RenderingContext))->CreateBitmapBrush( (ID2D1Bitmap*)(m_pBitmap->m_pBitmap), &hD2D1DescBitmap, &hD2D1Desc, (ID2D1BitmapBrush**)&m_pBitmapBrush );
    DebugAssert( hRes == S_OK && m_pBitmapBrush != NULL );

    m_pBrush = NULL;
    hRes = ((ID2D1BitmapBrush*)m_pBitmapBrush)->QueryInterface( __uuidof(ID2D1Brush), &m_pBrush );
    DebugAssert( hRes == S_OK && m_pBrush != NULL );
}
Void D2D1BitmapBrush::_NakedDestroy()
{
    ((ID2D1Brush*)m_pBrush)->Release();
    m_pBrush = NULL;

    ((ID2D1BitmapBrush*)m_pBitmapBrush)->Release();
    m_pBitmapBrush = NULL;
}

