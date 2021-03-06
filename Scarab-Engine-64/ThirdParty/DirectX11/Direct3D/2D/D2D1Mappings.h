/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/2D/D2D1Mappings.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : API-dependant mappings for Direct2D and DirectWrite.
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
#ifndef SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_2D_D2D1MAPPINGS_H
#define SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_2D_D2D1MAPPINGS_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../D3D11Mappings.h"

/////////////////////////////////////////////////////////////////////////////////
// General Declarations
typedef struct _d2d1_point {
    Float fX;
    Float fY;
} D2D1Point;
typedef struct _d2d1_point_i {
    UInt iX;
    UInt iY;
} D2D1PointI;

typedef struct _d2d1_rectangle_i {
    UInt iLeft;
    UInt iTop;
    UInt iRight;
    UInt iBottom;
} D2D1RectangleI;

typedef struct _d2d1_color {
    Float R;
    Float G;
    Float B;
    Float A;
} D2D1Color;

/////////////////////////////////////////////////////////////////////////////////
// D2D1RenderingContext Declarations
enum D2D1RenderTargetType {
    D2D1_RENDERTARGET_DEFAULT = 0,
    D2D1_RENDERTARGET_SOFTWARE,
    D2D1_RENDERTARGET_HARDWARE,
    D2D1_RENDERTARGET_COUNT
};
extern D2D1RenderTargetType D2D1RenderTargetTypeFromD2D1[D2D1_RENDERTARGET_COUNT];
extern DWord D2D1RenderTargetTypeToD2D1[D2D1_RENDERTARGET_COUNT];

enum D2D1RenderTargetUsage {
    D2D1_RENDERTARGETUSAGE_NONE = 0,
    D2D1_RENDERTARGETUSAGE_FORCE_BITMAP_REMOTING,
    D2D1_RENDERTARGETUSAGE_GDI_COMPATIBLE,
    D2D1_RENDERTARGETUSAGE_COUNT
};
extern D2D1RenderTargetUsage D2D1RenderTargetUsageFromD2D1[D2D1_RENDERTARGETUSAGE_COUNT];
extern DWord D2D1RenderTargetUsageToD2D1[D2D1_RENDERTARGETUSAGE_COUNT];

enum D2D1BitmapAlphaMode {
    D2D1BITMAP_ALPHAMODE_UNKNOWN = 0,
    D2D1BITMAP_ALPHAMODE_PREMULTIPLIED,
    D2D1BITMAP_ALPHAMODE_STRAIGHT,
    D2D1BITMAP_ALPHAMODE_IGNORE,
    D2D1BITMAP_ALPHAMODE_COUNT
};
extern D2D1BitmapAlphaMode D2D1BitmapAlphaModeFromD2D1[D2D1BITMAP_ALPHAMODE_COUNT];
extern DWord D2D1BitmapAlphaModeToD2D1[D2D1BITMAP_ALPHAMODE_COUNT];

typedef struct _d2d1_rendertarget_desc {
    Void ConvertFrom( const Void * pD2D1Desc );
    Void ConvertTo( Void * outD2D1Desc ) const;

    D2D1RenderTargetType iType;
    D2D1RenderTargetUsage iUsage;
    PixelFormat iFormat;
    D2D1BitmapAlphaMode iAlphaMode;
    Float fDpiX;
    Float fDpiY;
} D2D1RenderTargetDesc;

enum D2D1TextDrawOptions {
    D2D1TEXT_DRAWOPTION_NONE = 0,
    D2D1TEXT_DRAWOPTION_NOSNAP,
    D2D1TEXT_DRAWOPTION_CLIP,
    D2D1TEXT_DRAWOPTION_COUNT
};
extern D2D1TextDrawOptions D2D1TextDrawOptionsFromD2D1[D2D1TEXT_DRAWOPTION_COUNT];
extern DWord D2D1TextDrawOptionsToD2D1[D2D1TEXT_DRAWOPTION_COUNT];

/////////////////////////////////////////////////////////////////////////////////
// D2D1RenderState Declarations
typedef struct _d2d1_matrix32 {
    Float f00; Float f01;
    Float f10; Float f11;
    Float f20; Float f21;
} D2D1Matrix32;

enum D2D1AntialiasingMode {
    D2D1_ANTIALIASING_PER_PRIMITIVE = 0,
    D2D1_ANTIALIASING_ALIASED,
    D2D1_ANTIALIASING_COUNT
};
extern D2D1AntialiasingMode D2D1AntialiasingModeFromD2D1[D2D1_ANTIALIASING_COUNT];
extern DWord D2D1AntialiasingModeToD2D1[D2D1_ANTIALIASING_COUNT];

enum D2D1TextAntialiasingMode {
    D2D1TEXT_ANTIALIASING_DEFAULT = 0,
    D2D1TEXT_ANTIALIASING_CLEARTYPE,
    D2D1TEXT_ANTIALIASING_GRAYSCALE,
    D2D1TEXT_ANTIALIASING_ALIASED,
    D2D1TEXT_ANTIALIASING_COUNT
};
extern D2D1TextAntialiasingMode D2D1TextAntialiasingModeFromD2D1[D2D1TEXT_ANTIALIASING_COUNT];
extern DWord D2D1TextAntialiasingModeToD2D1[D2D1TEXT_ANTIALIASING_COUNT];

typedef UInt64 D2D1Tag;

typedef struct _d2d1_renderstate_desc {
    Void ConvertFrom( const Void * pD2D1Desc );
    Void ConvertTo( Void * outD2D1Desc ) const;

    D2D1AntialiasingMode iAntialiasingMode;
    D2D1TextAntialiasingMode iTextAntialiasingMode;
    D2D1Tag iTag1;
    D2D1Tag iTag2;
    D2D1Matrix32 matTransform;
} D2D1RenderStateDesc;

enum D2D1TextPixelGeometry {
    D2D1TEXT_PIXELGEOMETRY_FLAT = 0,
    D2D1TEXT_PIXELGEOMETRY_RGB,
    D2D1TEXT_PIXELGEOMETRY_BGR,
    D2D1TEXT_PIXELGEOMETRY_COUNT
};
extern D2D1TextPixelGeometry D2D1TextPixelGeometryFromD2D1[D2D1TEXT_PIXELGEOMETRY_COUNT];
extern DWord D2D1TextPixelGeometryToD2D1[D2D1TEXT_PIXELGEOMETRY_COUNT];

enum D2D1TextRenderingMode {
    D2D1TEXT_RENDERINGMODE_DEFAULT = 0,
    D2D1TEXT_RENDERINGMODE_ALIASED,
    D2D1TEXT_RENDERINGMODE_OUTLINE,
    D2D1TEXT_RENDERINGMODE_CLEARTYPE_GDI_CLASSIC,
    D2D1TEXT_RENDERINGMODE_CLEARTYPE_GDI_NATURAL,
    D2D1TEXT_RENDERINGMODE_CLEARTYPE_NATURAL,
    D2D1TEXT_RENDERINGMODE_CLEARTYPE_NATURAL_SYMMETRIC,
    D2D1TEXT_RENDERINGMODE_COUNT
};
extern D2D1TextRenderingMode D2D1TextRenderingModeFromD2D1[D2D1TEXT_RENDERINGMODE_COUNT];
extern DWord D2D1TextRenderingModeToD2D1[D2D1TEXT_RENDERINGMODE_COUNT];

typedef struct _d2d1_text_renderstate_desc {
    Float fGamma;            // ]0;256]
    Float fEnhancedContrast; // [0;+inf]
    Float fClearTypeLevel;   // [0;1]
    D2D1TextPixelGeometry iPixelGeometry;
    D2D1TextRenderingMode iRenderingMode;
} D2D1TextRenderStateDesc;

/////////////////////////////////////////////////////////////////////////////////
// D2D1StrokeStyle Declarations
enum D2D1StrokeCapStyle {
    D2D1STROKE_CAPSTYLE_FLAT = 0,
    D2D1STROKE_CAPSTYLE_SQUARE,
    D2D1STROKE_CAPSTYLE_ROUND,
    D2D1STROKE_CAPSTYLE_TRIANGLE,
    D2D1STROKE_CAPSTYLE_COUNT
};
extern D2D1StrokeCapStyle D2D1StrokeCapStyleFromD2D1[D2D1STROKE_CAPSTYLE_COUNT];
extern DWord D2D1StrokeCapStyleToD2D1[D2D1STROKE_CAPSTYLE_COUNT];

enum D2D1StrokeLineJoin {
    D2D1STROKE_LINEJOIN_MITER = 0,
    D2D1STROKE_LINEJOIN_BEVEL,
    D2D1STROKE_LINEJOIN_ROUND,
    D2D1STROKE_LINEJOIN_MITER_OR_BEVEL,
    D2D1STROKE_LINEJOIN_COUNT
};
extern D2D1StrokeLineJoin D2D1StrokeLineJoinFromD2D1[D2D1STROKE_LINEJOIN_COUNT];
extern DWord D2D1StrokeLineJoinToD2D1[D2D1STROKE_LINEJOIN_COUNT];

enum D2D1StrokeDashStyle {
    D2D1STROKE_DASHSTYLE_SOLID = 0,
    D2D1STROKE_DASHSTYLE_DASH,
    D2D1STROKE_DASHSTYLE_DOT,
    D2D1STROKE_DASHSTYLE_DASHDOT,
    D2D1STROKE_DASHSTYLE_DASHDOTDOT,
    D2D1STROKE_DASHSTYLE_CUSTOM,
    D2D1STROKE_DASHSTYLE_COUNT
};
extern D2D1StrokeDashStyle D2D1StrokeDashStyleFromD2D1[D2D1STROKE_DASHSTYLE_COUNT];
extern DWord D2D1StrokeDashStyleToD2D1[D2D1STROKE_DASHSTYLE_COUNT];

typedef struct _d2d1_strokestyle_desc {
    Void ConvertFrom( const Void * pD2D1Desc );
    Void ConvertTo( Void * outD2D1Desc ) const;

    D2D1StrokeCapStyle iStartCap;
    D2D1StrokeCapStyle iEndCap;
    D2D1StrokeCapStyle iDashCap;
    D2D1StrokeLineJoin iLineJoin;
    Float fMiterLimit;
    D2D1StrokeDashStyle iDashStyle;
    Float fDashOffset;
} D2D1StrokeStyleDesc;

/////////////////////////////////////////////////////////////////////////////////
// D2D1Bitmap Declarations
enum D2D1BitmapInterpolationMode {
    D2D1BITMAP_INTERPOLATION_NEAREST = 0,
    D2D1BITMAP_INTERPOLATION_LINEAR,
    D2D1BITMAP_INTERPOLATION_COUNT
};
extern D2D1BitmapInterpolationMode D2D1BitmapInterpolationModeFromD2D1[D2D1BITMAP_INTERPOLATION_COUNT];
extern DWord D2D1BitmapInterpolationModeToD2D1[D2D1BITMAP_INTERPOLATION_COUNT];

enum D2D1BitmapOpacityMaskContent {
    D2D1BITMAP_OPACITYMASK_GRAPHICS = 0,
    D2D1BITMAP_OPACITYMASK_TEXT_NATURAL,
    D2D1BITMAP_OPACITYMASK_TEXT_GDICOMPATIBLE,
    D2D1BITMAP_OPACITYMASK_COUNT
};
extern D2D1BitmapOpacityMaskContent D2D1BitmapOpacityMaskContentFromD2D1[D2D1BITMAP_OPACITYMASK_COUNT];
extern DWord D2D1BitmapOpacityMaskContentToD2D1[D2D1BITMAP_OPACITYMASK_COUNT];

typedef struct _d2d1_bitmap_desc {
    PixelFormat iFormat;
    D2D1BitmapAlphaMode iAlphaMode;
    Float fDpiX;
    Float fDpiY;
    UInt iWidth;
    UInt iHeight;
    Float fDIPWidth;  // DIP = pixels / ( Dpi/96 )
    Float fDIPHeight; // pixels = DIP * ( Dpi/96 )
} D2D1BitmapDesc;

/////////////////////////////////////////////////////////////////////////////////
// D2D1Brush Declarations
typedef struct _d2d1_gradient_stop {
    Float fPosition;
    D2D1Color fColor;
} D2D1GradientStop;

enum D2D1BrushGamma {
    D2D1BRUSH_GAMMA_2_2 = 0,
    D2D1BRUSH_GAMMA_1_0,
    D2D1BRUSH_GAMMA_COUNT
};
extern D2D1BrushGamma D2D1BrushGammaFromD2D1[D2D1BRUSH_GAMMA_COUNT];
extern DWord D2D1BrushGammaToD2D1[D2D1BRUSH_GAMMA_COUNT];

enum D2D1BrushWrapMode {
    D2D1BRUSH_WRAPMODE_CLAMP = 0,
    D2D1BRUSH_WRAPMODE_REPEAT,
    D2D1BRUSH_WRAPMODE_MIRROR,
    D2D1BRUSH_WRAPMODE_COUNT
};
extern D2D1BrushWrapMode D2D1BrushWrapModeFromD2D1[D2D1BRUSH_WRAPMODE_COUNT];
extern DWord D2D1BrushWrapModeToD2D1[D2D1BRUSH_WRAPMODE_COUNT];

#define D2D1GRADIENT_MAX_STOPS 64

typedef struct _d2d1_gradient_desc {
    D2D1BrushGamma iGamma;
    D2D1BrushWrapMode iWrapMode;
    UInt iStopCount;
    D2D1GradientStop arrGradientStops[D2D1GRADIENT_MAX_STOPS];
} D2D1GradientDesc;

typedef struct _d2d1_brush_desc {
    Void ConvertFrom( const Void * pD2D1Desc );
    Void ConvertTo( Void * outD2D1Desc ) const;

    Float fOpacity;
    D2D1Matrix32 matTransform;
} D2D1BrushDesc;

typedef struct _d2d1_brush_lineargradient_desc {
    Void ConvertFrom( const Void * pD2D1Desc );
    Void ConvertTo( Void * outD2D1Desc ) const;

    D2D1Point ptStart;
    D2D1Point ptEnd;
} D2D1BrushLinearGradientDesc;

typedef struct _d2d1_brush_radialgradient_desc {
    Void ConvertFrom( const Void * pD2D1Desc );
    Void ConvertTo( Void * outD2D1Desc ) const;

    D2D1Point ptCenter;
    D2D1Point ptOffset;
    Float fRadiusX;
    Float fRadiusY;
} D2D1BrushRadialGradientDesc;

typedef struct _d2d1_brush_bitmap_desc {
    Void ConvertFrom( const Void * pD2D1Desc );
    Void ConvertTo( Void * outD2D1Desc ) const;

    D2D1BitmapInterpolationMode iInterpolationMode;
    D2D1BrushWrapMode iWrapModeX;
    D2D1BrushWrapMode iWrapModeY;
} D2D1BrushBitmapDesc;

/////////////////////////////////////////////////////////////////////////////////
// D2D1Mesh Declarations
typedef struct _d2d1_triangle {
    D2D1Point ptA;
    D2D1Point ptB;
    D2D1Point ptC;
} D2D1Triangle;

/////////////////////////////////////////////////////////////////////////////////
// D2D1Geometry Declarations
typedef struct _d2d1_rectangle {
    Float fLeft;
    Float fTop;
    Float fRight;
    Float fBottom;
} D2D1Rectangle;

typedef struct _d2d1_roundedrectangle {
    D2D1Rectangle hRect;
    Float fRadiusX;
    Float fRadiusY;
} D2D1RoundedRectangle;

typedef struct _d2d1_ellipse {
    D2D1Point hCenter;
    Float fRadiusX;
    Float fRadiusY;
} D2D1Ellipse;

typedef struct _d2d1_bezier_segment {
    D2D1Point ptControlPointA;
    D2D1Point ptControlPointB;
    D2D1Point ptEndPoint;
} D2D1BezierSegment;

typedef struct _d2d1_quadratic_bezier_segment {
    D2D1Point ptControlPoint;
    D2D1Point ptEndPoint;
} D2D1QuadraticBezierSegment;

enum D2D1ArcSweepDirection {
    D2D1ARC_SWEEPDIR_COUNTER_CLOCKWISE = 0, // D2D1_SWEEP_DIRECTION_COUNTER_CLOCKWISE
    D2D1ARC_SWEEPDIR_CLOCKWISE         = 1  // D2D1_SWEEP_DIRECTION_CLOCKWISE
};
enum D2D1ArcSize {
    D2D1ARC_SIZE_SMALL = 0, // D2D1_ARC_SIZE_SMALL
    D2D1ARC_SIZE_LARGE = 1  // D2D1_ARC_SIZE_LARGE
};
typedef struct _d2d1_arc_segment {
    D2D1Point ptEndPoint;
    D2D1Point ptRadius;
    Float fRotation;
    D2D1ArcSweepDirection iSweepDirection;
    D2D1ArcSize iArcSize;
} D2D1ArcSegment;

enum D2D1GeometryRelation {
    D2D1GEOMETRY_RELATION_UNKNOWN = 0,
    D2D1GEOMETRY_RELATION_DISJOINT,
    D2D1GEOMETRY_RELATION_ISCONTAINED,
    D2D1GEOMETRY_RELATION_CONTAINS,
    D2D1GEOMETRY_RELATION_OVERLAP,
    D2D1GEOMETRY_RELATION_COUNT
};
extern D2D1GeometryRelation D2D1GeometryRelationFromD2D1[D2D1GEOMETRY_RELATION_COUNT];
extern DWord D2D1GeometryRelationToD2D1[D2D1GEOMETRY_RELATION_COUNT];

enum D2D1GeometryCombineMode {
    D2D1GEOMETRY_COMBINE_UNION = 0,
    D2D1GEOMETRY_COMBINE_INTERSECT,
    D2D1GEOMETRY_COMBINE_EXCLUDE,
    D2D1GEOMETRY_COMBINE_XOR,
    D2D1GEOMETRY_COMBINE_COUNT
};
extern D2D1GeometryCombineMode D2D1GeometryCombineModeFromD2D1[D2D1GEOMETRY_COMBINE_COUNT];
extern DWord D2D1GeometryCombineModeToD2D1[D2D1GEOMETRY_COMBINE_COUNT];

enum D2D1GeometrySimplifyOption {
    D2D1GEOMETRY_SIMPLIFY_CUBICS_AND_LINES = 0,
    D2D1GEOMETRY_SIMPLIFY_LINES_ONLY,
    D2D1GEOMETRY_SIMPLIFY_COUNT
};
extern D2D1GeometrySimplifyOption D2D1GeometrySimplifyOptionFromD2D1[D2D1GEOMETRY_SIMPLIFY_COUNT];
extern DWord D2D1GeometrySimplifyOptionToD2D1[D2D1GEOMETRY_SIMPLIFY_COUNT];

enum D2D1GeometryFillMode {
    D2D1GEOMETRY_FILL_ALTERNATE = 0,
    D2D1GEOMETRY_FILL_WINDING,
    D2D1GEOMETRY_FILL_COUNT
};
extern D2D1GeometryFillMode D2D1GeometryFillModeFromD2D1[D2D1GEOMETRY_FILL_COUNT];
extern DWord D2D1GeometryFillModeToD2D1[D2D1GEOMETRY_FILL_COUNT];

enum D2D1GeometrySegmentFlag {
    D2D1GEOMETRY_SEGMENT_NONE = 0,
    D2D1GEOMETRY_SEGMENT_FORCE_UNSTROKED,
    D2D1GEOMETRY_SEGMENT_FORCE_ROUND_LINE_JOIN,
    D2D1GEOMETRY_SEGMENT_COUNT
};
extern D2D1GeometrySegmentFlag D2D1GeometrySegmentFlagFromD2D1[D2D1GEOMETRY_SEGMENT_COUNT];
extern DWord D2D1GeometrySegmentFlagToD2D1[D2D1GEOMETRY_SEGMENT_COUNT];

/////////////////////////////////////////////////////////////////////////////////
// D2D1Layout Declarations
enum D2D1LayerOptions {
    D2D1LAYER_OPTION_NONE = 0,
    D2D1LAYER_OPTION_CLEARTYPE_INIT,
    D2D1LAYER_OPTION_COUNT
};
extern D2D1LayerOptions D2D1LayerOptionsFromD2D1[D2D1LAYER_OPTION_COUNT];
extern DWord D2D1LayerOptionsToD2D1[D2D1LAYER_OPTION_COUNT];

/////////////////////////////////////////////////////////////////////////////////
// D2D1Text Declarations
typedef struct _d2d1_font_glyph_offset {
    Float fAdvanceOffset;
    Float fAscenderOffset;
} D2D1FontGlyphOffset;

typedef struct _d2d1_text_range {
    UInt iPosition;
    UInt iLength;
} D2D1TextRange;

typedef struct _d2d1_text_line_metrics {
    UInt iLength;
    UInt iTrailingWhitespaceLength;
    UInt iNewlineLength;
    Float fHeight;
    Float fBaseline;
    Int bIsTrimmed; // Bool
} D2D1TextLineMetrics;

typedef struct _d2d1_text_cluster_metrics {
    Float fWidth;
    UShort iLength;
    UShort bCanWrapLineAfter :1;
    UShort bIsWhitespace     :1;
    UShort bIsNewline        :1;
    UShort bIsSoftHyphen     :1;
    UShort bIsRightToLeft    :1;
    UShort _Padding          :11;    
} D2D1TextClusterMetrics;

typedef struct _d2d1_text_hittest_metrics {
    UInt iPosition; // First character position within the hit region
    UInt iLength;   // number of character positions within the hit region
    Float fLeft;   //
    Float fTop;    // Hit region
    Float fWidth;  // rectangle
    Float fHeight; //
    UInt iBidiLevel; // Unicode - BIDI algorithm for display ordering of bi-directional texts
                     // Gives the position of the character relative to display order rather than logical order
    Int bContainsText; // Bool - Hit region contains text
    Int bIsTrimmed;    // Bool - Text range is trimmed
} D2D1TextHitTestMetrics;

enum D2D1FontWeight {
    D2D1FONT_WEIGHT_THIN = 0,
    D2D1FONT_WEIGHT_EXTRALIGHT,
    D2D1FONT_WEIGHT_LIGHT,
    D2D1FONT_WEIGHT_NORMAL,
    D2D1FONT_WEIGHT_MEDIUM,
    D2D1FONT_WEIGHT_SEMIBOLD,
    D2D1FONT_WEIGHT_BOLD,
    D2D1FONT_WEIGHT_EXTRABOLD,
    D2D1FONT_WEIGHT_BLACK,
    D2D1FONT_WEIGHT_EXTRA_BLACK,
    D2D1FONT_WEIGHT_COUNT
};
D2D1FontWeight D2D1FontWeightFromD2D1( DWord iD2D1FontWeight );
extern DWord D2D1FontWeightToD2D1[D2D1FONT_WEIGHT_COUNT];

enum D2D1FontStyle {
    D2D1FONT_STYLE_NORMAL = 0,
    D2D1FONT_STYLE_OBLIQUE,
    D2D1FONT_STYLE_ITALIC,
    D2D1FONT_STYLE_COUNT
};
extern D2D1FontStyle D2D1FontStyleFromD2D1[D2D1FONT_STYLE_COUNT];
extern DWord D2D1FontStyleToD2D1[D2D1FONT_STYLE_COUNT];

enum D2D1FontStretch {
    D2D1FONT_STRETCH_UNDEFINED = 0,
    D2D1FONT_STRETCH_ULTRACONDENSED,
    D2D1FONT_STRETCH_EXTRACONDENSED,
    D2D1FONT_STRETCH_CONDENSED,
    D2D1FONT_STRETCH_SEMICONDENSED,
    D2D1FONT_STRETCH_NORMAL,
    D2D1FONT_STRETCH_SEMIEXPANDED,
    D2D1FONT_STRETCH_EXPANDED,
    D2D1FONT_STRETCH_EXTRAEXPANDED,
    D2D1FONT_STRETCH_ULTRAEXPANDED,
    D2D1FONT_STRETCH_COUNT
};
extern D2D1FontStretch D2D1FontStretchFromD2D1[D2D1FONT_STRETCH_COUNT];
extern DWord D2D1FontStretchToD2D1[D2D1FONT_STRETCH_COUNT];

enum D2D1FontInfoStringID {
    D2D1FONT_INFOSTRING_NONE = 0,
    D2D1FONT_INFOSTRING_COPYRIGHT,
    D2D1FONT_INFOSTRING_VERSION,
    D2D1FONT_INFOSTRING_TRADEMARK,
    D2D1FONT_INFOSTRING_MANUFACTURER,
    D2D1FONT_INFOSTRING_DESIGNER,
    D2D1FONT_INFOSTRING_DESIGNER_URL,
    D2D1FONT_INFOSTRING_DESCRIPTION,
    D2D1FONT_INFOSTRING_VENDOR_URL,
    D2D1FONT_INFOSTRING_LICENSE_DESCRIPTION,
    D2D1FONT_INFOSTRING_LICENSE_URL,
    D2D1FONT_INFOSTRING_WIN32_FAMILYNAMES,
    D2D1FONT_INFOSTRING_WIN32_SUBFAMILYNAMES,
    D2D1FONT_INFOSTRING_PREF_FAMILYNAMES,
    D2D1FONT_INFOSTRING_PREF_SUBFAMILYNAMES,
    D2D1FONT_INFOSTRING_SAMPLE_TEXT,
    D2D1FONT_INFOSTRING_COUNT
};
extern D2D1FontInfoStringID D2D1FontInfoStringIDFromD2D1[D2D1FONT_INFOSTRING_COUNT];
extern DWord D2D1FontInfoStringIDToD2D1[D2D1FONT_INFOSTRING_COUNT];

enum D2D1FontSimulation {
    D2D1FONT_SIMULATION_NONE = 0,
    D2D1FONT_SIMULATION_BOLD,
    D2D1FONT_SIMULATION_OBLIQUE,
    D2D1FONT_SIMULATION_COUNT
};
extern D2D1FontSimulation D2D1FontSimulationFromD2D1[D2D1FONT_SIMULATION_COUNT];
extern DWord D2D1FontSimulationToD2D1[D2D1FONT_SIMULATION_COUNT];

typedef struct _d2d1_font_metrics {
    Void ConvertFrom( const Void * pD2D1Desc );

    UShort iDesignUnitsPerEM;
    UShort iAscent;
    UShort iDescent;
    UShort iLineGap;
    UShort iCapHeight;
    UShort iXHeight;
    UShort iUnderlinePosition;
    UShort iUnderlineThickness;
    UShort iStrikethroughPosition;
    UShort iStrikethroughThickness;
} D2D1FontMetrics;

enum D2D1FontFaceType {
    D2D1FONT_FACE_CFF = 0,
    D2D1FONT_FACE_TRUETYPE,
    D2D1FONT_FACE_TRUETYPE_COLLECTION,
    D2D1FONT_FACE_TYPE1,
    D2D1FONT_FACE_VECTOR,
    D2D1FONT_FACE_BITMAP,
    D2D1FONT_FACE_UNKNOWN,
    D2D1FONT_FACE_COUNT
};
extern D2D1FontFaceType D2D1FontFaceTypeFromD2D1[D2D1FONT_FACE_COUNT];
extern DWord D2D1FontFaceTypeToD2D1[D2D1FONT_FACE_COUNT];

enum D2D1TextMeasuringMode {
    D2D1TEXT_MEASURING_NATURAL = 0,
    D2D1TEXT_MEASURING_GDI_CLASSIC,
    D2D1TEXT_MEASURING_GDI_NATURAL,
    D2D1TEXT_MEASURING_COUNT
};
extern D2D1TextMeasuringMode D2D1TextMeasuringModeFromD2D1[D2D1TEXT_MEASURING_COUNT];
extern DWord D2D1TextMeasuringModeToD2D1[D2D1TEXT_MEASURING_COUNT];

typedef struct _d2d1_font_glyph_metrics {
    Void ConvertFrom( const Void * pD2D1Desc );

    Int iVerticalOriginY;
    Int iLeftSideBearing;
    Int iTopSideBearing;
    Int iRightSideBearing;
    Int iBottomSideBearing;
    UInt iAdvanceWidth;
    UInt iAdvanceHeight;
} D2D1FontGlyphMetrics;

enum D2D1TextFlowDirection {
    D2D1TEXT_FLOWDIRECTION_TOP_BOTTOM = 0,
    D2D1TEXT_FLOWDIRECTION_COUNT
};
extern D2D1TextFlowDirection D2D1TextFlowDirectionFromD2D1[D2D1TEXT_FLOWDIRECTION_COUNT];
extern DWord D2D1TextFlowDirectionToD2D1[D2D1TEXT_FLOWDIRECTION_COUNT];

enum D2D1TextReadingDirection {
    D2D1TEXT_READINGDIRECTION_LEFT_RIGHT = 0,
    D2D1TEXT_READINGDIRECTION_RIGHT_LEFT,
    D2D1TEXT_READINGDIRECTION_COUNT
};
extern D2D1TextReadingDirection D2D1TextReadingDirectionFromD2D1[D2D1TEXT_READINGDIRECTION_COUNT];
extern DWord D2D1TextReadingDirectionToD2D1[D2D1TEXT_READINGDIRECTION_COUNT];

enum D2D1TextLineSpacingMethod {
    D2D1TEXT_LINESPACING_DEFAULT = 0,
    D2D1TEXT_LINESPACING_UNIFORM,
    D2D1TEXT_LINESPACING_COUNT
};
extern D2D1TextLineSpacingMethod D2D1TextLineSpacingMethodFromD2D1[D2D1TEXT_LINESPACING_COUNT];
extern DWord D2D1TextLineSpacingMethodToD2D1[D2D1TEXT_LINESPACING_COUNT];

enum D2D1TextAlignment {
    D2D1TEXT_ALIGNMENT_LEADING = 0,
    D2D1TEXT_ALIGNMENT_TRAILING,
    D2D1TEXT_ALIGNMENT_CENTER,
    D2D1TEXT_ALIGNMENT_COUNT
};
extern D2D1TextAlignment D2D1TextAlignmentFromD2D1[D2D1TEXT_ALIGNMENT_COUNT];
extern DWord D2D1TextAlignmentToD2D1[D2D1TEXT_ALIGNMENT_COUNT];

enum D2D1TextParagraphAlignment {
    D2D1TEXT_PARAGRAPHALIGNMENT_NEAR = 0,
    D2D1TEXT_PARAGRAPHALIGNMENT_FAR,
    D2D1TEXT_PARAGRAPHALIGNMENT_CENTER,
    D2D1TEXT_PARAGRAPHALIGNMENT_COUNT
};
extern D2D1TextParagraphAlignment D2D1TextParagraphAlignmentFromD2D1[D2D1TEXT_PARAGRAPHALIGNMENT_COUNT];
extern DWord D2D1TextParagraphAlignmentToD2D1[D2D1TEXT_PARAGRAPHALIGNMENT_COUNT];

enum D2D1TextTrimmingGranularity {
    D2D1TEXT_TRIMMINGGRANULARITY_NONE = 0,
    D2D1TEXT_TRIMMINGGRANULARITY_CHARACTER,
    D2D1TEXT_TRIMMINGGRANULARITY_WORD,
    D2D1TEXT_TRIMMINGGRANULARITY_COUNT
};
extern D2D1TextTrimmingGranularity D2D1TextTrimmingGranularityFromD2D1[D2D1TEXT_TRIMMINGGRANULARITY_COUNT];
extern DWord D2D1TextTrimmingGranularityToD2D1[D2D1TEXT_TRIMMINGGRANULARITY_COUNT];

enum D2D1TextWordWrapping {
    D2D1TEXT_WORDWRAPPING_NONE = 0,
    D2D1TEXT_WORDWRAPPING_WRAP,
    D2D1TEXT_WORDWRAPPING_COUNT
};
extern D2D1TextWordWrapping D2D1TextWordWrappingFromD2D1[D2D1TEXT_WORDWRAPPING_COUNT];
extern DWord D2D1TextWordWrappingToD2D1[D2D1TEXT_WORDWRAPPING_COUNT];

enum D2D1TextBreakCondition {
    D2D1TEXT_BREAKCONDITION_NEUTRAL,
    D2D1TEXT_BREAKCONDITION_CAN_BREAK,
    D2D1TEXT_BREAKCONDITION_MAY_NOT_BREAK,
    D2D1TEXT_BREAKCONDITION_MUST_BREAK,
    D2D1TEXT_BREAKCONDITION_COUNT
};
extern D2D1TextBreakCondition D2D1TextBreakConditionFromD2D1[D2D1TEXT_BREAKCONDITION_COUNT];
extern DWord D2D1TextBreakConditionToD2D1[D2D1TEXT_BREAKCONDITION_COUNT];

typedef struct _d2d1_text_inlineobject_metrics {
    Void ConvertFrom( const Void * pD2D1Desc );
    Void ConvertTo( Void * outD2D1Desc );

    Float fWidth;
    Float fHeight;
    Float fBaseline;
    Bool bSupportSideways;
} D2D1TextInlineObjectMetrics;

typedef struct _d2d1_text_overhang_metrics {
    Void ConvertFrom( const Void * pD2D1Desc );
    Void ConvertTo( Void * outD2D1Desc );

    Float fLeft;
    Float fTop;
    Float fRight;
    Float fBottom;
} D2D1TextOverhangMetrics;

typedef struct _d2d1_text_metrics {
    Void ConvertFrom( const Void * pD2D1Desc );

    Float fLeft;
    Float fTop;
    Float fWidth;
    Float fWidthIncludingTrailingWhitespaces;
    Float fHeight;
    Float fLayoutWidth;
    Float fLayoutHeight;
    UInt iMaxBidiReorderingDepth;
    UInt iLineCount;
} D2D1TextMetrics;

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "D2D1Mappings.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_THIRDPARTY_DIRECTX11_DIRECT3D_2D_D2D1MAPPINGS_H




