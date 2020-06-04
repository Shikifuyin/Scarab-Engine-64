/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/2D/Resources/D2D1Brush.h
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
// Header prelude
#ifndef SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_2D_RESOURCES_D2D1BRUSH_H
#define SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_2D_RESOURCES_D2D1BRUSH_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "D2D1Bitmap.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

// Brush types
enum D2D1BrushType {
    D2D1BRUSH_SOLID_COLOR = 0,
    D2D1BRUSH_GRADIENT_LINEAR,
    D2D1BRUSH_GRADIENT_RADIAL,
    D2D1BRUSH_BITMAP
};

// Prototypes
class D2D1RenderingContext;
class D2D1TextLayout;

/////////////////////////////////////////////////////////////////////////////////
// The D2D1Brush class
class D2D1Brush
{
protected:
    D2D1Brush( D2D1RenderingContext * pContext2D );
public:
    virtual ~D2D1Brush();

    // Deferred construction
    inline Bool IsCreated() const;

    virtual Void Destroy();

    Void OnDestroyDevice();
    Void OnRestoreDevice();

    // Getters
    virtual D2D1BrushType GetType() const = 0;

    // Methods
    inline Float GetOpacity() const;
    Void SetOpacity( Float fOpacity );

    inline Void GetTransform( D2D1Matrix32 * outTransform ) const;
    Void SetTransform( const D2D1Matrix32 * pTransform );

    inline const D2D1BrushDesc * GetDesc() const;
    Void SetDesc( const D2D1BrushDesc * pDesc );

protected:
    friend class D2D1RenderingContext;
    D2D1RenderingContext * m_pContext2D;

    Void * m_pBrush;
    D2D1BrushDesc m_hBrushDesc;

    // Auto-Regen system
    virtual Void _NakedCreate() = 0;
    virtual Void _NakedDestroy() = 0;

    Bool m_bTemporaryDestroyed;
};

/////////////////////////////////////////////////////////////////////////////////
// The D2D1SolidColorBrush class
class D2D1SolidColorBrush : public D2D1Brush
{
public:
    D2D1SolidColorBrush( D2D1RenderingContext * pContext2D );
    virtual ~D2D1SolidColorBrush();

    // Deferred construction
    Void Create();

    // Getters
    inline virtual D2D1BrushType GetType() const;

    // Methods
    inline const D2D1Color * GetColor() const;
    inline Void GetColor( D2D1Color * outColor ) const;
    Void SetColor( const D2D1Color * pColor );

protected:
    friend class D2D1TextLayout;

    Void * m_pSolidColorBrush;
    D2D1Color m_hBrushColor;

    // Auto-Regen system
    virtual Void _NakedCreate();
    virtual Void _NakedDestroy();
};

/////////////////////////////////////////////////////////////////////////////////
// The D2D1GradientBrush class
class D2D1GradientBrush : public D2D1Brush
{
protected:
    D2D1GradientBrush( D2D1RenderingContext * pContext2D );
public:
    virtual ~D2D1GradientBrush();

    // Getters
    inline D2D1BrushGamma GetGradientGammaMode() const;
    inline D2D1BrushWrapMode GetGradientWrapMode() const;

    // Methods
    inline UInt GetGradientStopCount() const;
    inline const D2D1GradientStop * GetGradientStop( UInt iStop ) const;
    inline Void GetGradientStop( D2D1GradientStop * outGradientStop, UInt iStop ) const;
    Void GetGradientStops( D2D1GradientStop * outGradientStops, UInt iMaxStops ) const;

    inline const D2D1GradientDesc * GetGradientDesc() const;

protected:
    Void * m_pGradientStopCollection;
    D2D1GradientDesc m_hGradientDesc;

    // Auto-Regen system
    Void _CreateGradient( const D2D1GradientDesc * pGradientDesc );
    Void _DestroyGradient();
    Void _NakedCreateGradient();
    Void _NakedDestroyGradient();
};

/////////////////////////////////////////////////////////////////////////////////
// The D2D1LinearGradientBrush class
class D2D1LinearGradientBrush : public D2D1GradientBrush
{
public:
    D2D1LinearGradientBrush( D2D1RenderingContext * pContext2D );
    virtual ~D2D1LinearGradientBrush();

    // Deferred construction
    Void Create( const D2D1GradientDesc * pGradientDesc );
    virtual Void Destroy();

    // Getters
    inline virtual D2D1BrushType GetType() const;

    // Methods
    inline const D2D1Point * GetStartPoint() const;
    inline Void GetStartPoint( D2D1Point * outStart ) const;
    Void SetStartPoint( const D2D1Point * pStart );

    inline const D2D1Point * GetEndPoint() const;
    inline Void GetEndPoint( D2D1Point * outEnd ) const;
    Void SetEndPoint( const D2D1Point * pEnd );

    inline const D2D1BrushLinearGradientDesc * GetLinearGradientDesc() const;
    Void SetLinearGradientDesc( const D2D1BrushLinearGradientDesc * pLinearGradientDesc );

protected:
    Void * m_pLinearGradientBrush;
    D2D1BrushLinearGradientDesc m_hLinearGradientDesc;

    // Auto-Regen system
    virtual Void _NakedCreate();
    virtual Void _NakedDestroy();
};

/////////////////////////////////////////////////////////////////////////////////
// The D2D1RadialGradientBrush class
class D2D1RadialGradientBrush : public D2D1GradientBrush
{
public:
    D2D1RadialGradientBrush( D2D1RenderingContext * pContext2D );
    virtual ~D2D1RadialGradientBrush();

    // Deferred construction
    Void Create( const D2D1GradientDesc * pGradientDesc );
    virtual Void Destroy();

    // Getters
    inline virtual D2D1BrushType GetType() const;

    // Methods
    inline const D2D1Point * GetCenter() const;
    Void SetCenter( const D2D1Point * pCenter );

    inline const D2D1Point * GetOffset() const;
    Void SetOffset( const D2D1Point * pOffset );

    inline Float GetRadiusX() const;
    inline Float GetRadiusY() const;
    Void SetRadiusX( Float fRadiusX );
    Void SetRadiusY( Float fRadiusY );

    inline const D2D1BrushRadialGradientDesc * GetRadialGradientDesc() const;
    Void SetRadialGradientDesc( const D2D1BrushRadialGradientDesc * pRadialGradientDesc );

protected:
    Void * m_pRadialGradientBrush;
    D2D1BrushRadialGradientDesc m_hRadialGradientDesc;

    // Auto-Regen system
    virtual Void _NakedCreate();
    virtual Void _NakedDestroy();
};

/////////////////////////////////////////////////////////////////////////////////
// The D2D1BitmapBrush class
class D2D1BitmapBrush : public D2D1Brush
{
public:
    D2D1BitmapBrush( D2D1RenderingContext * pContext2D );
    virtual ~D2D1BitmapBrush();

    // Deferred construction
    Void Create();

    // Getters
    inline virtual D2D1BrushType GetType() const;

    // Methods
    inline D2D1Bitmap * GetBitmap() const;
    Void SetBitmap( D2D1Bitmap * pBitmap );

    inline D2D1BitmapInterpolationMode GetInterpolationMode() const;
    Void SetInterpolationMode( D2D1BitmapInterpolationMode iInterpolationMode );

    inline D2D1BrushWrapMode GetWrapModeX() const;
    inline D2D1BrushWrapMode GetWrapModeY() const;
    Void SetWrapModeX( D2D1BrushWrapMode iWrapModeX );
    Void SetWrapModeY( D2D1BrushWrapMode iWrapModeY );

    inline const D2D1BrushBitmapDesc * GetBrushBitmapDesc() const;
    Void SetBrushBitmapDesc( const D2D1BrushBitmapDesc * pBitmapDesc );

protected:
    D2D1Bitmap * m_pBitmap;

    Void * m_pBitmapBrush;
    D2D1BrushBitmapDesc m_hBitmapDesc;

    // Auto-Regen system
    virtual Void _NakedCreate();
    virtual Void _NakedDestroy();
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "D2D1Brush.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_2D_RESOURCES_D2D1BRUSH_H

