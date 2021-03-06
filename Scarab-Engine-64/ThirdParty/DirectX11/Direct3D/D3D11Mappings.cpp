/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/D3D11Mappings.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : API-dependant mappings for Win32, GDI, DXGI & Direct3D
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
#include <d3d11.h>
#include <d3dcompiler.h>

#undef DebugAssert

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "D3D11Mappings.h"

/////////////////////////////////////////////////////////////////////////////////
// General Definitions
DWord _D3D11ConvertFlags32( Byte * arrConvert, DWord iFlags )
{
    DWord iRes = 0, iLog2 = 0;
    while( iFlags != 0 ) {
        if ( iFlags & 1 )
            iRes |= ( 1 << arrConvert[iLog2] );
        iFlags >>= 1;
        ++iLog2;
    }
    return iRes;
}

PixelFormat PixelFormatFromDXGI[PIXEL_FMT_COUNT] = {
    // DXGI_FORMAT_UNKNOWN
    PIXEL_FMT_UNKNOWN,

    // DXGI_FORMAT_R32G32B32A32_TYPELESS, DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_UINT, DXGI_FORMAT_R32G32B32A32_SINT
    PIXEL_FMT_RGBA32, PIXEL_FMT_RGBA32F, PIXEL_FMT_RGBA32UI, PIXEL_FMT_RGBA32SI,

    // DXGI_FORMAT_R32G32B32_TYPELESS, DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_R32G32B32_UINT, DXGI_FORMAT_R32G32B32_SINT,
    PIXEL_FMT_RGB32, PIXEL_FMT_RGB32F, PIXEL_FMT_RGB32UI, PIXEL_FMT_RGB32SI,

    // DXGI_FORMAT_R16G16B16A16_TYPELESS, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_UNORM,
    // DXGI_FORMAT_R16G16B16A16_UINT, DXGI_FORMAT_R16G16B16A16_SNORM, DXGI_FORMAT_R16G16B16A16_SINT,
    PIXEL_FMT_RGBA16, PIXEL_FMT_RGBA16F, PIXEL_FMT_RGBA16UN, PIXEL_FMT_RGBA16UI, PIXEL_FMT_RGBA16SN, PIXEL_FMT_RGBA16SI,

    // DXGI_FORMAT_R32G32_TYPELESS, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32_SINT,
    PIXEL_FMT_RG32, PIXEL_FMT_RG32F, PIXEL_FMT_RG32UI, PIXEL_FMT_RG32SI,

    // DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, DXGI_FORMAT_X32_TYPELESS_G8X24_UINT,
    PIXEL_FMT_R32G8X24, PIXEL_FMT_D32F_S8X24UI, PIXEL_FMT_R32F_X8X24, PIXEL_FMT_X32_G8X24UI,

    // DXGI_FORMAT_R10G10B10A2_TYPELESS, DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R10G10B10A2_UINT, DXGI_FORMAT_R11G11B10_FLOAT,
    PIXEL_FMT_RGB10A2, PIXEL_FMT_RGB10A2UN, PIXEL_FMT_RGB10A2UI, PIXEL_FMT_RG11B10F,

    // DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
    // DXGI_FORMAT_R8G8B8A8_UINT, DXGI_FORMAT_R8G8B8A8_SNORM, DXGI_FORMAT_R8G8B8A8_SINT,
    PIXEL_FMT_RGBA8, PIXEL_FMT_RGBA8UN, PIXEL_FMT_RGBA8UN_SRGB, PIXEL_FMT_RGBA8UI, PIXEL_FMT_RGBA8SN, PIXEL_FMT_RGBA8SI,

    // DXGI_FORMAT_R16G16_TYPELESS, DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_UNORM,
    // DXGI_FORMAT_R16G16_UINT, DXGI_FORMAT_R16G16_SNORM, DXGI_FORMAT_R16G16_SINT,
    PIXEL_FMT_RG16, PIXEL_FMT_RG16F, PIXEL_FMT_RG16UN, PIXEL_FMT_RG16UI, PIXEL_FMT_RG16SN, PIXEL_FMT_RG16SI,

    // DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_SINT,
    PIXEL_FMT_R32, PIXEL_FMT_D32F, PIXEL_FMT_R32F, PIXEL_FMT_R32UI, PIXEL_FMT_R32SI,

    // DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_X24_TYPELESS_G8_UINT,
    PIXEL_FMT_R24G8, PIXEL_FMT_D24UN_S8UI, PIXEL_FMT_R24UN_X8, PIXEL_FMT_X24_G8UI,

    // DXGI_FORMAT_R8G8_TYPELESS, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8_UINT,
    // DXGI_FORMAT_R8G8_SNORM, DXGI_FORMAT_R8G8_SINT,
    PIXEL_FMT_RG8, PIXEL_FMT_RG8UN, PIXEL_FMT_RG8UI, PIXEL_FMT_RG8SN, PIXEL_FMT_RG8SI,

    // DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R16_UNORM,
    // DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16_SNORM, DXGI_FORMAT_R16_SINT,
    PIXEL_FMT_R16, PIXEL_FMT_R16F, PIXEL_FMT_D16UN, PIXEL_FMT_R16UN, PIXEL_FMT_R16UI, PIXEL_FMT_R16SN, PIXEL_FMT_R16SI,

    // DXGI_FORMAT_R8_TYPELESS, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8_UINT,
    // DXGI_FORMAT_R8_SNORM, DXGI_FORMAT_R8_SINT,
    PIXEL_FMT_R8, PIXEL_FMT_R8UN, PIXEL_FMT_R8UI, PIXEL_FMT_R8SN, PIXEL_FMT_R8SI,

    // DXGI_FORMAT_A8_UNORM, DXGI_FORMAT_R1_UNORM,
    PIXEL_FMT_A8UN, PIXEL_FMT_R1UN,

    // DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_FORMAT_R8G8_B8G8_UNORM, DXGI_FORMAT_G8R8_G8B8_UNORM,
    PIXEL_FMT_RGB9E5_SHAREDEXP, PIXEL_FMT_RG8UN_BG8UN, PIXEL_FMT_GR8UN_GB8UN,

    // DXGI_FORMAT_BC1_TYPELESS, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM_SRGB,
    PIXEL_FMT_BC1, PIXEL_FMT_BC1UN, PIXEL_FMT_BC1UN_SRGB,

    // DXGI_FORMAT_BC2_TYPELESS, DXGI_FORMAT_BC2_UNORM, DXGI_FORMAT_BC2_UNORM_SRGB,
    PIXEL_FMT_BC2, PIXEL_FMT_BC2UN, PIXEL_FMT_BC2UN_SRGB,

    // DXGI_FORMAT_BC3_TYPELESS, DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC3_UNORM_SRGB,
    PIXEL_FMT_BC3, PIXEL_FMT_BC3UN, PIXEL_FMT_BC3UN_SRGB,

    // DXGI_FORMAT_BC4_TYPELESS, DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC4_SNORM,
    PIXEL_FMT_BC4, PIXEL_FMT_BC4UN, PIXEL_FMT_BC4SN,

    // DXGI_FORMAT_BC5_TYPELESS, DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC5_SNORM,
    PIXEL_FMT_BC5, PIXEL_FMT_BC5UN, PIXEL_FMT_BC5SN,

    // DXGI_FORMAT_B5G6R5_UNORM, DXGI_FORMAT_B5G5R5A1_UNORM,
    PIXEL_FMT_B5G6R5UN, PIXEL_FMT_BGR5A1UN,

    // DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8X8_UNORM,
    PIXEL_FMT_BGRA8UN, PIXEL_FMT_BGRX8UN,

    // DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM
    PIXEL_FMT_RGB10XRBIAS_A2UN,

    // DXGI_FORMAT_B8G8R8A8_TYPELESS, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
    PIXEL_FMT_BGRA8, PIXEL_FMT_BGRA8UN_SRGB,

    // DXGI_FORMAT_B8G8R8X8_TYPELESS, DXGI_FORMAT_B8G8R8X8_UNORM_SRGB,
    PIXEL_FMT_BGRX8, PIXEL_FMT_BGRX8UN_SRGB,

    // DXGI_FORMAT_BC6H_TYPELESS, DXGI_FORMAT_BC6H_UF16, DXGI_FORMAT_BC6H_SF16,
    PIXEL_FMT_BC6H, PIXEL_FMT_BC6H_UF16, PIXEL_FMT_BC6H_SF16,

    // DXGI_FORMAT_BC7_TYPELESS, DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC7_UNORM_SRGB,
    PIXEL_FMT_BC7, PIXEL_FMT_BC7UN, PIXEL_FMT_BC7UN_SRGB,

    // DXGI_FORMAT_AYUV, DXGI_FORMAT_Y410, DXGI_FORMAT_Y416, DXGI_FORMAT_NV12, DXGI_FORMAT_P010, DXGI_FORMAT_P016,
    PIXEL_FMT_AYUV, PIXEL_FMT_Y410, PIXEL_FMT_Y416, PIXEL_FMT_NV12, PIXEL_FMT_P010, PIXEL_FMT_P016,

    // DXGI_FORMAT_420_OPAQUE,
    PIXEL_FMT_420_OPAQUE,

    // DXGI_FORMAT_YUY2, DXGI_FORMAT_Y210, DXGI_FORMAT_Y216, DXGI_FORMAT_NV11, DXGI_FORMAT_AI44, DXGI_FORMAT_IA44,
    PIXEL_FMT_YUY2, PIXEL_FMT_Y210, PIXEL_FMT_Y216, PIXEL_FMT_NV11, PIXEL_FMT_AI44, PIXEL_FMT_IA44,

    // DXGI_FORMAT_P8, DXGI_FORMAT_A8P8,
    PIXEL_FMT_P8, PIXEL_FMT_AP8,

    // DXGI_FORMAT_B4G4R4A4_UNORM,
    PIXEL_FMT_BGRA4UN
};
DWord PixelFormatToDXGI[PIXEL_FMT_COUNT] = {
    // PIXEL_FMT_UNKNOWN
    DXGI_FORMAT_UNKNOWN,

    // PIXEL_FMT_RGBA32, PIXEL_FMT_RGBA32F, PIXEL_FMT_RGBA32UI, PIXEL_FMT_RGBA32SI,
    DXGI_FORMAT_R32G32B32A32_TYPELESS, DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_UINT, DXGI_FORMAT_R32G32B32A32_SINT,

    // PIXEL_FMT_RGB32, PIXEL_FMT_RGB32F, PIXEL_FMT_RGB32UI, PIXEL_FMT_RGB32SI,
    DXGI_FORMAT_R32G32B32_TYPELESS, DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_R32G32B32_UINT, DXGI_FORMAT_R32G32B32_SINT,

    // PIXEL_FMT_RGBA16, PIXEL_FMT_RGBA16F, PIXEL_FMT_RGBA16UN, PIXEL_FMT_RGBA16UI, PIXEL_FMT_RGBA16SN, PIXEL_FMT_RGBA16SI,
    DXGI_FORMAT_R16G16B16A16_TYPELESS, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_UNORM,
    DXGI_FORMAT_R16G16B16A16_UINT, DXGI_FORMAT_R16G16B16A16_SNORM, DXGI_FORMAT_R16G16B16A16_SINT,

    // PIXEL_FMT_RG32, PIXEL_FMT_RG32F, PIXEL_FMT_RG32UI, PIXEL_FMT_RG32SI,
    DXGI_FORMAT_R32G32_TYPELESS, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32_SINT,

    // PIXEL_FMT_RGBA8, PIXEL_FMT_RGBA8UN, PIXEL_FMT_RGBA8UN_SRGB, PIXEL_FMT_RGBA8UI, PIXEL_FMT_RGBA8SN, PIXEL_FMT_RGBA8SI,
    DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
    DXGI_FORMAT_R8G8B8A8_UINT, DXGI_FORMAT_R8G8B8A8_SNORM, DXGI_FORMAT_R8G8B8A8_SINT,

    // PIXEL_FMT_BGRA8, PIXEL_FMT_BGRA8UN, PIXEL_FMT_BGRA8UN_SRGB,
    DXGI_FORMAT_B8G8R8A8_TYPELESS, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,

    // PIXEL_FMT_BGRX8, PIXEL_FMT_BGRX8UN, PIXEL_FMT_BGRX8UN_SRGB,
    DXGI_FORMAT_B8G8R8X8_TYPELESS, DXGI_FORMAT_B8G8R8X8_UNORM, DXGI_FORMAT_B8G8R8X8_UNORM_SRGB,

    // PIXEL_FMT_RGB10A2, PIXEL_FMT_RGB10A2UN, PIXEL_FMT_RGB10A2UI,
    DXGI_FORMAT_R10G10B10A2_TYPELESS, DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R10G10B10A2_UINT,

    // PIXEL_FMT_RG16, PIXEL_FMT_RG16F, PIXEL_FMT_RG16UN, PIXEL_FMT_RG16UI, PIXEL_FMT_RG16SN, PIXEL_FMT_RG16SI,
    DXGI_FORMAT_R16G16_TYPELESS, DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_UNORM,
    DXGI_FORMAT_R16G16_UINT, DXGI_FORMAT_R16G16_SNORM, DXGI_FORMAT_R16G16_SINT,

    // PIXEL_FMT_R32, PIXEL_FMT_R32F, PIXEL_FMT_R32UI, PIXEL_FMT_R32SI,
    DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_SINT,

    // PIXEL_FMT_BGR5A1UN, PIXEL_FMT_B5G6R5UN,
    DXGI_FORMAT_B5G5R5A1_UNORM, DXGI_FORMAT_B5G6R5_UNORM,

    // PIXEL_FMT_RG8, PIXEL_FMT_RG8UN, PIXEL_FMT_RG8UI, PIXEL_FMT_RG8SN, PIXEL_FMT_RG8SI,
    DXGI_FORMAT_R8G8_TYPELESS, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8_UINT,
    DXGI_FORMAT_R8G8_SNORM, DXGI_FORMAT_R8G8_SINT,

    // PIXEL_FMT_R16, PIXEL_FMT_R16F, PIXEL_FMT_R16UN, PIXEL_FMT_R16UI, PIXEL_FMT_R16SN, PIXEL_FMT_R16SI,
    DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_UNORM,
    DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16_SNORM, DXGI_FORMAT_R16_SINT,

    // PIXEL_FMT_R8, PIXEL_FMT_R8UN, PIXEL_FMT_R8UI, PIXEL_FMT_R8SN, PIXEL_FMT_R8SI,
    DXGI_FORMAT_R8_TYPELESS, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8_UINT,
    DXGI_FORMAT_R8_SNORM, DXGI_FORMAT_R8_SINT,

    // PIXEL_FMT_A8UN, PIXEL_FMT_R1UN,
    DXGI_FORMAT_A8_UNORM, DXGI_FORMAT_R1_UNORM,

    // PIXEL_FMT_D32F_S8X24UI, PIXEL_FMT_D24UN_S8UI, PIXEL_FMT_D32F, PIXEL_FMT_D16UN,
    DXGI_FORMAT_D32_FLOAT_S8X24_UINT, DXGI_FORMAT_D24_UNORM_S8_UINT,
    DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_D16_UNORM,

    // PIXEL_FMT_R32G8X24, PIXEL_FMT_R32F_X8X24, PIXEL_FMT_X32_G8X24UI,
    DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, DXGI_FORMAT_X32_TYPELESS_G8X24_UINT,

    // PIXEL_FMT_R24G8, PIXEL_FMT_R24UN_X8, PIXEL_FMT_X24_G8UI,
    DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_X24_TYPELESS_G8_UINT,

    // PIXEL_FMT_RG8UN_BG8UN, PIXEL_FMT_GR8UN_GB8UN,
    DXGI_FORMAT_R8G8_B8G8_UNORM, DXGI_FORMAT_G8R8_G8B8_UNORM,

    // PIXEL_FMT_RGB10XRBIAS_A2UN, PIXEL_FMT_RG11B10F, PIXEL_FMT_RGB9E5_SHAREDEXP,
    DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM, DXGI_FORMAT_R11G11B10_FLOAT, DXGI_FORMAT_R9G9B9E5_SHAREDEXP,

    // PIXEL_FMT_P8, PIXEL_FMT_AP8,
    DXGI_FORMAT_P8, DXGI_FORMAT_A8P8,

    // PIXEL_FMT_BGRA4UN,
    DXGI_FORMAT_B4G4R4A4_UNORM,

    // PIXEL_FMT_BC1, PIXEL_FMT_BC1UN, PIXEL_FMT_BC1UN_SRGB,
    DXGI_FORMAT_BC1_TYPELESS, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM_SRGB,

    // PIXEL_FMT_BC2, PIXEL_FMT_BC2UN, PIXEL_FMT_BC2UN_SRGB,
    DXGI_FORMAT_BC2_TYPELESS, DXGI_FORMAT_BC2_UNORM, DXGI_FORMAT_BC2_UNORM_SRGB,

    // PIXEL_FMT_BC3, PIXEL_FMT_BC3UN, PIXEL_FMT_BC3UN_SRGB,
    DXGI_FORMAT_BC3_TYPELESS, DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC3_UNORM_SRGB,

    // PIXEL_FMT_BC4, PIXEL_FMT_BC4UN, PIXEL_FMT_BC4SN,
    DXGI_FORMAT_BC4_TYPELESS, DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC4_SNORM,

    // PIXEL_FMT_BC5, PIXEL_FMT_BC5UN, PIXEL_FMT_BC5SN,
    DXGI_FORMAT_BC5_TYPELESS, DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC5_SNORM,

    // PIXEL_FMT_BC6H, PIXEL_FMT_BC6H_UF16, PIXEL_FMT_BC6H_SF16,
    DXGI_FORMAT_BC6H_TYPELESS, DXGI_FORMAT_BC6H_UF16, DXGI_FORMAT_BC6H_SF16,

    // PIXEL_FMT_BC7, PIXEL_FMT_BC7UN, PIXEL_FMT_BC7UN_SRGB,
    DXGI_FORMAT_BC7_TYPELESS, DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC7_UNORM_SRGB,

    // PIXEL_FMT_AYUV, PIXEL_FMT_Y410, PIXEL_FMT_Y416, PIXEL_FMT_NV12, PIXEL_FMT_P010, PIXEL_FMT_P016,
    DXGI_FORMAT_AYUV, DXGI_FORMAT_Y410, DXGI_FORMAT_Y416, DXGI_FORMAT_NV12, DXGI_FORMAT_P010, DXGI_FORMAT_P016,

    // PIXEL_FMT_YUY2, PIXEL_FMT_Y210, PIXEL_FMT_Y216, PIXEL_FMT_NV11, PIXEL_FMT_AI44, PIXEL_FMT_IA44,
    DXGI_FORMAT_YUY2, DXGI_FORMAT_Y210, DXGI_FORMAT_Y216, DXGI_FORMAT_NV11, DXGI_FORMAT_AI44, DXGI_FORMAT_IA44,

    // PIXEL_FMT_420_OPAQUE,
    DXGI_FORMAT_420_OPAQUE
};

/////////////////////////////////////////////////////////////////////////////////
// D3D11Window Definitions
D3D11OutputRotation D3D11OutputRotationFromDXGI[D3D11OUTPUT_ROTATION_COUNT] = {
    D3D11OUTPUT_ROTATION_UNDEFINED,
    D3D11OUTPUT_ROTATION_IDENTITY,
    D3D11OUTPUT_ROTATION_90,
    D3D11OUTPUT_ROTATION_180,
    D3D11OUTPUT_ROTATION_270
};
DWord D3D11OutputRotationToDXGI[D3D11OUTPUT_ROTATION_COUNT] = {
    DXGI_MODE_ROTATION_UNSPECIFIED,
    DXGI_MODE_ROTATION_IDENTITY,
    DXGI_MODE_ROTATION_ROTATE90,
    DXGI_MODE_ROTATION_ROTATE180,
    DXGI_MODE_ROTATION_ROTATE270
};

D3D11DisplayModeScanlineOrdering D3D11DisplayModeScanlineOrderingFromDXGI[D3D11DISPLAYMODE_SCANLINE_COUNT] = {
    D3D11DISPLAYMODE_SCANLINE_UNDEFINED,
    D3D11DISPLAYMODE_SCANLINE_PROGRESSIVE,
    D3D11DISPLAYMODE_SCANLINE_UPPER_FIELD_FIRST,
    D3D11DISPLAYMODE_SCANLINE_LOWER_FIELD_FIRST
};
DWord D3D11DisplayModeScanlineOrderingToDXGI[D3D11DISPLAYMODE_SCANLINE_COUNT] = {
    DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED,
    DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE,
    DXGI_MODE_SCANLINE_ORDER_UPPER_FIELD_FIRST,
    DXGI_MODE_SCANLINE_ORDER_LOWER_FIELD_FIRST
};

D3D11DisplayModeScaling D3D11DisplayModeScalingFromDXGI[D3D11DISPLAYMODE_SCALING_COUNT] = {
    D3D11DISPLAYMODE_SCALING_UNDEFINED,
    D3D11DISPLAYMODE_SCALING_CENTERED,
    D3D11DISPLAYMODE_SCALING_STRETCHED
};
DWord D3D11DisplayModeScalingToDXGI[D3D11DISPLAYMODE_SCALING_COUNT] = {
    DXGI_MODE_SCALING_UNSPECIFIED,
    DXGI_MODE_SCALING_CENTERED,
    DXGI_MODE_SCALING_STRETCHED
};

Void D3D11AdapterDesc::ConvertFrom( const Void * pDXGIDesc, UInt iAdapter )
{
    const DXGI_ADAPTER_DESC * pDesc = (const DXGI_ADAPTER_DESC *)pDXGIDesc;

    iIndex = iAdapter;

    iAdapterUID = ((((Int64)(pDesc->AdapterLuid.HighPart)) << 32) | ((Int64)(pDesc->AdapterLuid.LowPart)));
    StringFn->NCopy( strDescription, pDesc->Description, 127 );
    iVendorId = pDesc->VendorId;
    iDeviceId = pDesc->DeviceId;
    iSubSysId = pDesc->SubSysId;
    iRevision = pDesc->Revision;

    iDedicatedSystemMemory = pDesc->DedicatedSystemMemory;
    iDedicatedVideoMemory = pDesc->DedicatedVideoMemory;
    iSharedSystemMemory = pDesc->SharedSystemMemory;
}
Void D3D11AdapterDesc::ConvertTo( Void * outDXGIDesc, UInt * outAdapter ) const
{
    DXGI_ADAPTER_DESC * outDesc = (DXGI_ADAPTER_DESC *)outDXGIDesc;

    *outAdapter = iIndex;

    outDesc->AdapterLuid.HighPart = (LONG)(iAdapterUID >> 32);
    outDesc->AdapterLuid.LowPart = (DWORD)(iAdapterUID & 0x00000000ffffffffi64);
    StringFn->NCopy( outDesc->Description, strDescription, 127 );
    outDesc->VendorId = iVendorId;
    outDesc->DeviceId = iDeviceId;
    outDesc->SubSysId = iSubSysId;
    outDesc->Revision = iRevision;

    outDesc->DedicatedSystemMemory = iDedicatedSystemMemory;
    outDesc->DedicatedVideoMemory = iDedicatedVideoMemory;
    outDesc->SharedSystemMemory = iSharedSystemMemory;
}

Void D3D11OutputDesc::ConvertFrom( const Void * pDXGIDesc, UInt iOutput )
{
    const DXGI_OUTPUT_DESC * pDesc = (const DXGI_OUTPUT_DESC *)pDXGIDesc;

    iIndex = iOutput;

    StringFn->NCopy( strDeviceName, pDesc->DeviceName, 31 );
    pMonitor = (Void*)pDesc->Monitor;

    iRotation = D3D11OutputRotationFromDXGI[pDesc->Rotation];

    bAttachedToDesktop = ( pDesc->AttachedToDesktop != FALSE );
    iDesktopLeft = pDesc->DesktopCoordinates.left;
    iDesktopRight = pDesc->DesktopCoordinates.right;
    iDesktopTop = pDesc->DesktopCoordinates.top;
    iDesktopBottom = pDesc->DesktopCoordinates.bottom;
}
Void D3D11OutputDesc::ConvertTo( Void * outDXGIDesc, UInt * outOutput ) const
{
    DXGI_OUTPUT_DESC * outDesc = (DXGI_OUTPUT_DESC *)outDXGIDesc;

    *outOutput = iIndex;

    StringFn->NCopy( outDesc->DeviceName, strDeviceName, 31 );
    outDesc->Monitor = (HMONITOR)pMonitor;

    outDesc->Rotation = (DXGI_MODE_ROTATION)( D3D11OutputRotationToDXGI[iRotation] );

    outDesc->AttachedToDesktop = (bAttachedToDesktop) ? TRUE : FALSE;
    outDesc->DesktopCoordinates.left = iDesktopLeft;
    outDesc->DesktopCoordinates.right = iDesktopRight;
    outDesc->DesktopCoordinates.top = iDesktopTop;
    outDesc->DesktopCoordinates.bottom = iDesktopBottom;
}

Void D3D11DisplayModeDesc::ConvertFrom( const Void * pDXGIDesc, UInt iDisplayMode )
{
    const DXGI_MODE_DESC * pDesc = (const DXGI_MODE_DESC *)pDXGIDesc;

    iIndex = iDisplayMode;

    iWidth = pDesc->Width;
    iHeight = pDesc->Height;
    iFormat = PixelFormatFromDXGI[pDesc->Format];
    iRefreshRateNumerator = pDesc->RefreshRate.Numerator;
    iRefreshRateDenominator = pDesc->RefreshRate.Denominator;

    iScanlineOrdering = D3D11DisplayModeScanlineOrderingFromDXGI[pDesc->ScanlineOrdering];
    iScaling = D3D11DisplayModeScalingFromDXGI[pDesc->Scaling];
}
Void D3D11DisplayModeDesc::ConvertTo( Void * outDXGIDesc, UInt * outDisplayMode ) const
{
    DXGI_MODE_DESC * outDesc = (DXGI_MODE_DESC *)outDXGIDesc;

    *outDisplayMode = iIndex;

    outDesc->Width = iWidth;
    outDesc->Height = iHeight;
    outDesc->Format = (DXGI_FORMAT)( PixelFormatToDXGI[iFormat] );
    outDesc->RefreshRate.Numerator = iRefreshRateNumerator;
    outDesc->RefreshRate.Denominator = iRefreshRateDenominator;

    outDesc->ScanlineOrdering = (DXGI_MODE_SCANLINE_ORDER)( D3D11DisplayModeScanlineOrderingToDXGI[iScanlineOrdering] );
    outDesc->Scaling = (DXGI_MODE_SCALING)( D3D11DisplayModeScalingToDXGI[iScaling] );
}

Void D3D11GammaCaps::ConvertFrom( const Void * pDXGIDesc )
{
    const DXGI_GAMMA_CONTROL_CAPABILITIES * pDesc = (const DXGI_GAMMA_CONTROL_CAPABILITIES *)pDXGIDesc;

    bScaleAndOffsetSupported = ( pDesc->ScaleAndOffsetSupported != FALSE );
    fMaxConvertedValue = pDesc->MaxConvertedValue;
    fMinConvertedValue = pDesc->MinConvertedValue;

    iControlPointCount = pDesc->NumGammaControlPoints;
    for( UInt i = 0; i < iControlPointCount; ++i )
        arrControlPoints[i] = pDesc->ControlPointPositions[i];
}
Void D3D11GammaCaps::ConvertTo( Void * outDXGIDesc ) const
{
    DXGI_GAMMA_CONTROL_CAPABILITIES * outDesc = (DXGI_GAMMA_CONTROL_CAPABILITIES *)outDXGIDesc;

    outDesc->ScaleAndOffsetSupported = (bScaleAndOffsetSupported) ? TRUE : FALSE;
    outDesc->MaxConvertedValue = fMaxConvertedValue;
    outDesc->MinConvertedValue = fMinConvertedValue;

    outDesc->NumGammaControlPoints = iControlPointCount;
    for( UInt i = 0; i < iControlPointCount; ++i )
        outDesc->ControlPointPositions[i] = arrControlPoints[i];
}

Void D3D11GammaControl::ConvertFrom( const Void * pDXGIDesc )
{
    const DXGI_GAMMA_CONTROL * pDesc = (const DXGI_GAMMA_CONTROL *)pDXGIDesc;

    vScale.R = pDesc->Scale.Red;
    vScale.G = pDesc->Scale.Green;
    vScale.B = pDesc->Scale.Blue;
    vOffset.R = pDesc->Offset.Red;
    vOffset.G = pDesc->Offset.Green;
    vOffset.B = pDesc->Offset.Blue;

    for( UInt i = 0; i < 1025; ++i ) {
        arrGammaCurve[i].R = pDesc->GammaCurve[i].Red;
        arrGammaCurve[i].G = pDesc->GammaCurve[i].Green;
        arrGammaCurve[i].B = pDesc->GammaCurve[i].Blue;
    }
}
Void D3D11GammaControl::ConvertTo( Void * outDXGIDesc ) const
{
    DXGI_GAMMA_CONTROL * outDesc = (DXGI_GAMMA_CONTROL *)outDXGIDesc;

    outDesc->Scale.Red = vScale.R;
    outDesc->Scale.Green = vScale.G;
    outDesc->Scale.Blue = vScale.B;
    outDesc->Offset.Red = vOffset.R;
    outDesc->Offset.Green = vOffset.G;
    outDesc->Offset.Blue = vOffset.B;

    for( UInt i = 0; i < 1025; ++i ) {
        outDesc->GammaCurve[i].Red = arrGammaCurve[i].R;
        outDesc->GammaCurve[i].Green = arrGammaCurve[i].G;
        outDesc->GammaCurve[i].Blue = arrGammaCurve[i].B;
    }
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11Renderer Definitions
Byte D3D11PixelFormatSupportFlags1FromD3D11[D3D11PIXELFORMAT_SUPPORT1_COUNT] = {
    0,  // D3D11PIXELFORMAT_SUPPORT1_BUFFER
    1,  // D3D11PIXELFORMAT_SUPPORT1_IA_VERTEX_BUFFER
    2,  // D3D11PIXELFORMAT_SUPPORT1_IA_INDEX_BUFFER
    3,  // D3D11PIXELFORMAT_SUPPORT1_SO_BUFFER
    4,  // D3D11PIXELFORMAT_SUPPORT1_TEXTURE1D
    5,  // D3D11PIXELFORMAT_SUPPORT1_TEXTURE2D
    6,  // D3D11PIXELFORMAT_SUPPORT1_TEXTURE3D
    7,  // D3D11PIXELFORMAT_SUPPORT1_TEXTURECUBE
    8,  // D3D11PIXELFORMAT_SUPPORT1_SHADER_LOAD
    9,  // D3D11PIXELFORMAT_SUPPORT1_SHADER_SAMPLE
    10, // D3D11PIXELFORMAT_SUPPORT1_SHADER_SAMPLE_COMPARISON
    11, // D3D11PIXELFORMAT_SUPPORT1_SHADER_SAMPLE_MONO_TEXT
    12, // D3D11PIXELFORMAT_SUPPORT1_MIP
    13, // D3D11PIXELFORMAT_SUPPORT1_MIP_AUTOGEN
    14, // D3D11PIXELFORMAT_SUPPORT1_RENDER_TARGET
    15, // D3D11PIXELFORMAT_SUPPORT1_BLENDABLE
    16, // D3D11PIXELFORMAT_SUPPORT1_DEPTH_STENCIL
    17, // D3D11PIXELFORMAT_SUPPORT1_CPU_LOCKABLE
    18, // D3D11PIXELFORMAT_SUPPORT1_MULTISAMPLE_RESOLVE
    19, // D3D11PIXELFORMAT_SUPPORT1_DISPLAY
    20, // D3D11PIXELFORMAT_SUPPORT1_CAST_WITHIN_BIT_LAYOUT
    21, // D3D11PIXELFORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET
    22, // D3D11PIXELFORMAT_SUPPORT1_MULTISAMPLE_LOAD
    23, // D3D11PIXELFORMAT_SUPPORT1_SHADER_GATHER
    24, // D3D11PIXELFORMAT_SUPPORT1_BACK_BUFFER_CAST
    25, // D3D11PIXELFORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW
    26, // D3D11PIXELFORMAT_SUPPORT1_SHADER_GATHER_COMPARISON
    27, // D3D11PIXELFORMAT_SUPPORT1_DECODER_OUTPUT
    28, // D3D11PIXELFORMAT_SUPPORT1_VIDEO_PROCESSOR_OUTPUT
    29, // D3D11PIXELFORMAT_SUPPORT1_VIDEO_PROCESSOR_INPUT
    30  // D3D11PIXELFORMAT_SUPPORT1_VIDEO_ENCODER
};
Byte D3D11PixelFormatSupportFlags1ToD3D11[D3D11PIXELFORMAT_SUPPORT1_COUNT] = {
    0,  // D3D11_FORMAT_SUPPORT_BUFFER
    1,  // D3D11_FORMAT_SUPPORT_IA_VERTEX_BUFFER
    2,  // D3D11_FORMAT_SUPPORT_IA_INDEX_BUFFER
    3,  // D3D11_FORMAT_SUPPORT_SO_BUFFER
    4,  // D3D11_FORMAT_SUPPORT_TEXTURE1D
    5,  // D3D11_FORMAT_SUPPORT_TEXTURE2D
    6,  // D3D11_FORMAT_SUPPORT_TEXTURE3D
    7,  // D3D11_FORMAT_SUPPORT_TEXTURECUBE
    8,  // D3D11_FORMAT_SUPPORT_SHADER_LOAD
    9,  // D3D11_FORMAT_SUPPORT_SHADER_SAMPLE
    10, // D3D11_FORMAT_SUPPORT_SHADER_SAMPLE_COMPARISON
    11, // D3D11_FORMAT_SUPPORT_SHADER_SAMPLE_MONO_TEXT
    12, // D3D11_FORMAT_SUPPORT_MIP
    13, // D3D11_FORMAT_SUPPORT_MIP_AUTOGEN
    14, // D3D11_FORMAT_SUPPORT_RENDER_TARGET
    15, // D3D11_FORMAT_SUPPORT_BLENDABLE
    16, // D3D11_FORMAT_SUPPORT_DEPTH_STENCIL
    17, // D3D11_FORMAT_SUPPORT_CPU_LOCKABLE
    18, // D3D11_FORMAT_SUPPORT_MULTISAMPLE_RESOLVE
    19, // D3D11_FORMAT_SUPPORT_DISPLAY
    20, // D3D11_FORMAT_SUPPORT_CAST_WITHIN_BIT_LAYOUT
    21, // D3D11_FORMAT_SUPPORT_MULTISAMPLE_RENDERTARGET
    22, // D3D11_FORMAT_SUPPORT_MULTISAMPLE_LOAD
    23, // D3D11_FORMAT_SUPPORT_SHADER_GATHER
    24, // D3D11_FORMAT_SUPPORT_BACK_BUFFER_CAST
    25, // D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW
    26, // D3D11_FORMAT_SUPPORT_SHADER_GATHER_COMPARISON
    27, // D3D11_FORMAT_SUPPORT_DECODER_OUTPUT
    28, // D3D11_FORMAT_SUPPORT_VIDEO_PROCESSOR_OUTPUT
    29, // D3D11_FORMAT_SUPPORT_VIDEO_PROCESSOR_INPUT
    30  // D3D11_FORMAT_SUPPORT_VIDEO_ENCODER
};

Byte D3D11PixelFormatSupportFlags2FromD3D11[D3D11PIXELFORMAT_SUPPORT2_COUNT] = {
    0, // D3D11PIXELFORMAT_SUPPORT2_UAV_ATOMIC_ADD
    1, // D3D11PIXELFORMAT_SUPPORT2_UAV_ATOMIC_BITWISE_OPS
    2, // D3D11PIXELFORMAT_SUPPORT2_UAV_ATOMIC_COMPARE_STORE_OR_COMPARE_EXCHANGE
    3, // D3D11PIXELFORMAT_SUPPORT2_UAV_ATOMIC_EXCHANGE
    4, // D3D11PIXELFORMAT_SUPPORT2_UAV_ATOMIC_SIGNED_MIN_OR_MAX
    5, // D3D11PIXELFORMAT_SUPPORT2_UAV_ATOMIC_UNSIGNED_MIN_OR_MAX
    6, // D3D11PIXELFORMAT_SUPPORT2_UAV_TYPED_LOAD
    7, // D3D11PIXELFORMAT_SUPPORT2_UAV_TYPED_STORE
    8  // D3D11PIXELFORMAT_SUPPORT2_OUTPUT_MERGER_LOGIC_OP
};
Byte D3D11PixelFormatSupportFlags2ToD3D11[D3D11PIXELFORMAT_SUPPORT2_COUNT] = {
    0, // D3D11_FORMAT_SUPPORT2_UAV_ATOMIC_ADD
    1, // D3D11_FORMAT_SUPPORT2_UAV_ATOMIC_BITWISE_OPS
    2, // D3D11_FORMAT_SUPPORT2_UAV_ATOMIC_COMPARE_STORE_OR_COMPARE_EXCHANGE
    3, // D3D11_FORMAT_SUPPORT2_UAV_ATOMIC_EXCHANGE
    4, // D3D11_FORMAT_SUPPORT2_UAV_ATOMIC_SIGNED_MIN_OR_MAX
    5, // D3D11_FORMAT_SUPPORT2_UAV_ATOMIC_UNSIGNED_MIN_OR_MAX
    6, // D3D11_FORMAT_SUPPORT2_UAV_TYPED_LOAD
    7, // D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE
    8  // D3D11_FORMAT_SUPPORT2_OUTPUT_MERGER_LOGIC_OP
};

Void D3D11PixelFormatSupport::ConvertFrom( UInt iD3D11Flags1, UInt iD3D11Flags2 )
{
    iFlags1 = _D3D11ConvertFlags32( D3D11PixelFormatSupportFlags1FromD3D11, iD3D11Flags1 );
    iFlags2 = _D3D11ConvertFlags32( D3D11PixelFormatSupportFlags2FromD3D11, iD3D11Flags2 );
}

Void D3D11CounterSupport::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_COUNTER_INFO * pDesc = (const D3D11_COUNTER_INFO *)pD3D11Desc;

    if ( pDesc->LastDeviceDependentCounter == 0 )
        iMaxCounters = 0;
    else
        iMaxCounters = ( pDesc->LastDeviceDependentCounter - D3D11_COUNTER_DEVICE_DEPENDENT_0 );
    iMaxParallelCounters = pDesc->NumSimultaneousCounters;
    iMaxParallelUnitsDetection = pDesc->NumDetectableParallelUnits;
}

Void D3D11DeviceFeatures::ConvertFrom( const Void * pD3D11DescThreading, const Void * pD3D11DescArch,
                                       const Void * pD3D11DescDoubles, const Void * pD3D11DescMinPrecision,
                                       const Void * pD3D11DescD3D9, const Void * pD3D11DescD3D9Shadows,
                                       const Void * pD3D11DescD3D10, const Void * pD3D11DescD3D11 )
{
    const D3D11_FEATURE_DATA_THREADING * pDescThreading                         = (const D3D11_FEATURE_DATA_THREADING *)pD3D11DescThreading;
    const D3D11_FEATURE_DATA_ARCHITECTURE_INFO * pDescArch                      = (const D3D11_FEATURE_DATA_ARCHITECTURE_INFO *)pD3D11DescArch;
    const D3D11_FEATURE_DATA_DOUBLES * pDescDoubles                             = (const D3D11_FEATURE_DATA_DOUBLES *)pD3D11DescDoubles;
    const D3D11_FEATURE_DATA_SHADER_MIN_PRECISION_SUPPORT * pDescMinPrecision   = (const D3D11_FEATURE_DATA_SHADER_MIN_PRECISION_SUPPORT *)pD3D11DescMinPrecision;
    const D3D11_FEATURE_DATA_D3D9_OPTIONS * pDescD3D9                           = (const D3D11_FEATURE_DATA_D3D9_OPTIONS *)pD3D11DescD3D9;
    const D3D11_FEATURE_DATA_D3D9_SHADOW_SUPPORT * pDescD3D9Shadows             = (const D3D11_FEATURE_DATA_D3D9_SHADOW_SUPPORT *)pD3D11DescD3D9Shadows;
    const D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS * pDescD3D10              = (const D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS *)pD3D11DescD3D10;
    const D3D11_FEATURE_DATA_D3D11_OPTIONS * pDescD3D11                         = (const D3D11_FEATURE_DATA_D3D11_OPTIONS *)pD3D11DescD3D11;

    bDriverConcurrentCreates = ( pDescThreading->DriverConcurrentCreates != FALSE );
    bDriverCommandLists = ( pDescThreading->DriverCommandLists != FALSE );

    bTileBasedDeferredRenderer = ( pDescArch->TileBasedDeferredRenderer != FALSE );

    bDoublePrecisionShaderOps = ( pDescDoubles->DoublePrecisionFloatShaderOps != FALSE );

    bPixelShaderMinPrecision10Bits = ( ( pDescMinPrecision->PixelShaderMinPrecision & D3D11_SHADER_MIN_PRECISION_10_BIT ) != 0 );
    bPixelShaderMinPrecision16Bits = ( ( pDescMinPrecision->PixelShaderMinPrecision & D3D11_SHADER_MIN_PRECISION_16_BIT ) != 0 );
    bAllOtherShaderStagesMinPrecision10Bits = ( ( pDescMinPrecision->AllOtherShaderStagesMinPrecision & D3D11_SHADER_MIN_PRECISION_10_BIT ) != 0 );
    bAllOtherShaderStagesMinPrecision16Bits = ( ( pDescMinPrecision->AllOtherShaderStagesMinPrecision & D3D11_SHADER_MIN_PRECISION_16_BIT ) != 0 );

    bFullNonPow2TextureSupport = ( pDescD3D9->FullNonPow2TextureSupport != FALSE );
    bSupportsDepthAsTextureWithLessEqualComparisonFilter = ( pDescD3D9Shadows->SupportsDepthAsTextureWithLessEqualComparisonFilter != FALSE );

    bComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x = ( pDescD3D10->ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x != FALSE );

    bOutputMergerLogicOp = ( pDescD3D11->OutputMergerLogicOp != FALSE );
    bUAVOnlyRenderingForcedSampleCount = ( pDescD3D11->UAVOnlyRenderingForcedSampleCount != FALSE );
    bDiscardAPIsSeenByDriver = ( pDescD3D11->DiscardAPIsSeenByDriver != FALSE );
    bFlagsForUpdateAndCopySeenByDriver = ( pDescD3D11->FlagsForUpdateAndCopySeenByDriver != FALSE );
    bClearView = ( pDescD3D11->ClearView != FALSE );
    bCopyWithOverlap = ( pDescD3D11->CopyWithOverlap != FALSE );
    bConstantBufferPartialUpdate = ( pDescD3D11->ConstantBufferPartialUpdate != FALSE );
    bConstantBufferOffsetting = ( pDescD3D11->ConstantBufferOffsetting != FALSE );
    bMapNoOverwriteOnDynamicConstantBuffer = ( pDescD3D11->MapNoOverwriteOnDynamicConstantBuffer != FALSE );
    bMapNoOverwriteOnDynamicBufferSRV = ( pDescD3D11->MapNoOverwriteOnDynamicBufferSRV != FALSE );
    bMultisampleRTVWithForcedSampleCountOne = ( pDescD3D11->MultisampleRTVWithForcedSampleCountOne != FALSE );
    bSAD4ShaderInstructions = ( pDescD3D11->SAD4ShaderInstructions != FALSE );
    bExtendedDoublesShaderInstructions = ( pDescD3D11->ExtendedDoublesShaderInstructions != FALSE );
    bExtendedResourceSharing = ( pDescD3D11->ExtendedResourceSharing != FALSE );
}

D3D11SwapChainSwapEffect D3D11SwapChainSwapEffectFromDXGI[D3D11SWAPCHAIN_SWAPEFFECT_COUNT] = {
    D3D11SWAPCHAIN_SWAPEFFECT_DISCARD,
    D3D11SWAPCHAIN_SWAPEFFECT_SEQUENTIAL,
    D3D11SWAPCHAIN_SWAPEFFECT_FLIP,
    D3D11SWAPCHAIN_SWAPEFFECT_FLIP_SEQUENTIAL
};
DWord D3D11SwapChainSwapEffectToDXGI[D3D11SWAPCHAIN_SWAPEFFECT_COUNT] = {
    DXGI_SWAP_EFFECT_DISCARD,
    DXGI_SWAP_EFFECT_SEQUENTIAL,
    DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
    DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL
};

Byte D3D11SwapChainBufferUsageFlagsFromDXGI[D3D11SWAPCHAIN_BUFFERUSAGE_COUNT + 4] = {
    0, 0, 0, 0,
    0, // D3D11SWAPCHAIN_BUFFERUSAGE_SHADER_INPUT
    1, // D3D11SWAPCHAIN_BUFFERUSAGE_RENDER_TARGET_OUTPUT
    2, // D3D11SWAPCHAIN_BUFFERUSAGE_BACK_BUFFER
    3, // D3D11SWAPCHAIN_BUFFERUSAGE_SHARED
    4, // D3D11SWAPCHAIN_BUFFERUSAGE_READ_ONLY
    5, // D3D11SWAPCHAIN_BUFFERUSAGE_DISCARD_ON_PRESENT
    6  // D3D11SWAPCHAIN_BUFFERUSAGE_UNORDERED_ACCESS
};
Byte D3D11SwapChainBufferUsageFlagsToDXGI[D3D11SWAPCHAIN_BUFFERUSAGE_COUNT] = {
    4, // DXGI_USAGE_SHADER_INPUT
    5, // DXGI_USAGE_RENDER_TARGET_OUTPUT
    6, // DXGI_USAGE_BACK_BUFFER
    7, // DXGI_USAGE_SHARED
    8, // DXGI_USAGE_READ_ONLY
    9, // DXGI_USAGE_DISCARD_ON_PRESENT
    10 // DXGI_USAGE_UNORDERED_ACCESS
};

Byte D3D11SwapChainFlagsFromDXGI[D3D11SWAPCHAIN_FLAG_COUNT] = {
    0, // D3D11SWAPCHAIN_FLAG_NONPREROTATED
    1, // D3D11SWAPCHAIN_FLAG_ALLOW_MODE_SWITCH
    2, // D3D11SWAPCHAIN_FLAG_GDI_COMPATIBLE
    3, // D3D11SWAPCHAIN_FLAG_RESTRICTED_CONTENT
    4, // D3D11SWAPCHAIN_FLAG_RESTRICT_SHARED_RESOURCE_DRIVER
    5  // D3D11SWAPCHAIN_FLAG_DISPLAY_ONLY
};
Byte D3D11SwapChainFlagsToDXGI[D3D11SWAPCHAIN_FLAG_COUNT] = {
    0, // DXGI_SWAP_CHAIN_FLAG_NONPREROTATED
    1, // DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH
    2, // DXGI_SWAP_CHAIN_FLAG_GDI_COMPATIBLE
    3, // DXGI_SWAP_CHAIN_FLAG_RESTRICTED_CONTENT
    4, // DXGI_SWAP_CHAIN_FLAG_RESTRICT_SHARED_RESOURCE_DRIVER
    5  // DXGI_SWAP_CHAIN_FLAG_DISPLAY_ONLY
};

Byte D3D11PresentFlagsFromDXGI[D3D11PRESENT_FLAG_COUNT] = {
    0, // D3D11PRESENT_FLAG_TEST
    1, // D3D11PRESENT_FLAG_DONT_SEQUENCE
    2, // D3D11PRESENT_FLAG_RESTART
    3, // D3D11PRESENT_FLAG_DONT_WAIT
    4, // D3D11PRESENT_FLAG_STEREO_PREFER_RIGHT
    5, // D3D11PRESENT_FLAG_STEREO_TEMP_MONO
    6  // D3D11PRESENT_FLAG_RESTRICT_TO_OUTPUT
};
Byte D3D11PresentFlagsToDXGI[D3D11PRESENT_FLAG_COUNT] = {
    0, // DXGI_PRESENT_TEST
    1, // DXGI_PRESENT_DO_NOT_SEQUENCE
    2, // DXGI_PRESENT_RESTART
    3, // DXGI_PRESENT_DO_NOT_WAIT
    4, // DXGI_PRESENT_STEREO_PREFER_RIGHT
    5, // DXGI_PRESENT_STEREO_TEMPORARY_MONO
    6  // DXGI_PRESENT_RESTRICT_TO_OUTPUT
};

Void D3D11SwapChainDesc::ConvertFrom( const Void * pD3D11Desc )
{
    const DXGI_SWAP_CHAIN_DESC * pDesc = (const DXGI_SWAP_CHAIN_DESC *)pD3D11Desc;

    pOutputWindow = (Void*)( pDesc->OutputWindow );
    bWindowed = ( pDesc->Windowed != FALSE );

    iRefreshRateNumerator = pDesc->BufferDesc.RefreshRate.Numerator;
    iRefreshRateDenominator = pDesc->BufferDesc.RefreshRate.Denominator;

    iBufferCount = pDesc->BufferCount;
    iBufferUsageFlags = _D3D11ConvertFlags32( D3D11SwapChainBufferUsageFlagsFromDXGI, pDesc->BufferUsage );

    iWidth = pDesc->BufferDesc.Width;
    iHeight = pDesc->BufferDesc.Height;

    iFormat = PixelFormatFromDXGI[pDesc->BufferDesc.Format];
    iSampleCount = pDesc->SampleDesc.Count;
    iSampleQuality = pDesc->SampleDesc.Quality;

    iScanlineOrdering = D3D11DisplayModeScanlineOrderingFromDXGI[pDesc->BufferDesc.ScanlineOrdering];
    iScaling = D3D11DisplayModeScalingFromDXGI[pDesc->BufferDesc.Scaling];    

    iSwapEffect = D3D11SwapChainSwapEffectFromDXGI[pDesc->SwapEffect];
    iFlags = _D3D11ConvertFlags32( D3D11SwapChainFlagsFromDXGI, pDesc->Flags );
}
Void D3D11SwapChainDesc::ConvertTo( Void * outD3D11Desc ) const
{
    DXGI_SWAP_CHAIN_DESC * outDesc = (DXGI_SWAP_CHAIN_DESC*)outD3D11Desc;

    outDesc->OutputWindow = (HWND)pOutputWindow;
    outDesc->Windowed = (bWindowed) ? TRUE : FALSE;

    outDesc->BufferDesc.RefreshRate.Numerator = iRefreshRateNumerator;
    outDesc->BufferDesc.RefreshRate.Denominator = iRefreshRateDenominator;

    outDesc->BufferCount = iBufferCount;
    outDesc->BufferUsage = (DXGI_USAGE)( _D3D11ConvertFlags32(D3D11SwapChainBufferUsageFlagsToDXGI, iBufferUsageFlags) );

    outDesc->BufferDesc.Width = iWidth;
    outDesc->BufferDesc.Height = iHeight;

    outDesc->BufferDesc.Format = (DXGI_FORMAT)( PixelFormatToDXGI[iFormat] );
    outDesc->SampleDesc.Count = iSampleCount;
    outDesc->SampleDesc.Quality = iSampleQuality;

    outDesc->BufferDesc.ScanlineOrdering = (DXGI_MODE_SCANLINE_ORDER)( D3D11DisplayModeScanlineOrderingToDXGI[iScanlineOrdering] );
    outDesc->BufferDesc.Scaling = (DXGI_MODE_SCALING)( D3D11DisplayModeScalingToDXGI[iScaling] );

    outDesc->SwapEffect = (DXGI_SWAP_EFFECT)( D3D11SwapChainSwapEffectToDXGI[iSwapEffect] );
    outDesc->Flags = _D3D11ConvertFlags32( D3D11SwapChainFlagsToDXGI, iFlags );
}

Void D3D11FrameStats::ConvertFrom( const Void * pD3D11Desc, UInt iD3D11LastPresentCount )
{
    const DXGI_FRAME_STATISTICS * pDesc = (const DXGI_FRAME_STATISTICS *)pD3D11Desc;

    iLastPresentCount = iD3D11LastPresentCount;

    iPresentCount = pDesc->PresentCount;
    iPresentRefreshCount = pDesc->PresentRefreshCount;
    iSyncRefreshCount = pDesc->SyncRefreshCount;
    iSyncQPCTime = pDesc->SyncQPCTime.QuadPart;
    iSyncGPUTime = pDesc->SyncGPUTime.QuadPart;
}
Void D3D11FrameStats::ConvertTo( Void * outD3D11Desc, UInt * outLastPresentCount ) const
{
    DXGI_FRAME_STATISTICS * outDesc = (DXGI_FRAME_STATISTICS*)outD3D11Desc;

    *outLastPresentCount = iLastPresentCount;

    outDesc->PresentCount = iPresentCount;
    outDesc->PresentRefreshCount = iPresentRefreshCount;
    outDesc->SyncRefreshCount = iSyncRefreshCount;
    outDesc->SyncQPCTime.QuadPart = iSyncQPCTime;
    outDesc->SyncGPUTime.QuadPart = iSyncGPUTime;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11DeferredContext Definitions

/////////////////////////////////////////////////////////////////////////////////
// D3D11InputLayout Definitions
D3D11InputFieldSlotClass D3D11InputFieldSlotClassFromD3D11[D3D11INPUTFIELD_SLOTCLASS_COUNT] = {
    D3D11INPUTFIELD_SLOTCLASS_PER_VERTEX,
    D3D11INPUTFIELD_SLOTCLASS_PER_INSTANCE
};
DWord D3D11InputFieldSlotClassToD3D11[D3D11INPUTFIELD_SLOTCLASS_COUNT] = {
    D3D11_INPUT_PER_VERTEX_DATA,
    D3D11_INPUT_PER_INSTANCE_DATA
};

Void D3D11InputFieldDesc::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_INPUT_ELEMENT_DESC * pDesc = (const D3D11_INPUT_ELEMENT_DESC *)pD3D11Desc;

    iType = D3D11InputFieldTypeFromD3D11( pDesc->Format );
    strSemantic = pDesc->SemanticName;
    iSemanticIndex = pDesc->SemanticIndex;

    iSlot = pDesc->InputSlot;
    iSlotClass = D3D11InputFieldSlotClassFromD3D11[pDesc->InputSlotClass];

    iInstanceDataStepRate = (iSlotClass == D3D11INPUTFIELD_SLOTCLASS_PER_INSTANCE) ? pDesc->InstanceDataStepRate : 0;
}
Void D3D11InputFieldDesc::ConvertTo( Void * outD3D11Desc ) const
{
    D3D11_INPUT_ELEMENT_DESC * outDesc = (D3D11_INPUT_ELEMENT_DESC*)outD3D11Desc;

    outDesc->Format = (DXGI_FORMAT)( D3D11InputFieldTypeToD3D11(iType) );
    outDesc->SemanticName = strSemantic;
    outDesc->SemanticIndex = iSemanticIndex;

    outDesc->InputSlot = iSlot;
    outDesc->InputSlotClass = (D3D11_INPUT_CLASSIFICATION)( D3D11InputFieldSlotClassToD3D11[iSlotClass] );

    outDesc->InstanceDataStepRate = (iSlotClass == D3D11INPUTFIELD_SLOTCLASS_PER_INSTANCE) ? iInstanceDataStepRate : 0;

    outDesc->AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11RenderState Definitions
Byte D3D11SamplerFilterFlagsFromD3D11[D3D11SAMPLER_FILTERFLAG_COUNT + 3] = {
    0, // D3D11SAMPLER_FILTERFLAG_MIP_LINEAR
    0,
    1, // D3D11SAMPLER_FILTERFLAG_MAG_LINEAR
    0,
    2, // D3D11SAMPLER_FILTERFLAG_MIN_LINEAR
    0,
    3, // D3D11SAMPLER_FILTERFLAG_ANISOTROPY
    4  // D3D11SAMPLER_FILTERFLAG_CMP
};
Byte D3D11SamplerFilterFlagsToD3D11[D3D11SAMPLER_FILTERFLAG_COUNT] = {
    0, // D3D11_SAMPLERFILTERFLAG_MIP_LINEAR
    2, // D3D11_SAMPLERFILTERFLAG_MAG_LINEAR
    4, // D3D11_SAMPLERFILTERFLAG_MIN_LINEAR
    6, // D3D11_SAMPLERFILTERFLAG_ANISOTROPY
    7  // D3D11_SAMPLERFILTERFLAG_CMP
};

D3D11SamplerWrapMode D3D11SamplerWrapModeFromD3D11[D3D11SAMPLER_WRAP_COUNT + 1] = {
    (D3D11SamplerWrapMode)0, // INVALID !!!
    D3D11SAMPLER_WRAP_REPEAT,
    D3D11SAMPLER_WRAP_MIRROR_REPEAT,
    D3D11SAMPLER_WRAP_CLAMP,
    D3D11SAMPLER_WRAP_BORDER,
    D3D11SAMPLER_WRAP_MIRROR_ONCE,
};
DWord D3D11SamplerWrapModeToD3D11[D3D11SAMPLER_WRAP_COUNT] = {
    D3D11_TEXTURE_ADDRESS_CLAMP,
    D3D11_TEXTURE_ADDRESS_BORDER,
    D3D11_TEXTURE_ADDRESS_WRAP,
    D3D11_TEXTURE_ADDRESS_MIRROR_ONCE,
    D3D11_TEXTURE_ADDRESS_MIRROR
};

D3D11SamplerCompareFunction D3D11SamplerCompareFunctionFromD3D11[D3D11SAMPLER_COMPARE_COUNT + 1] = {
    (D3D11SamplerCompareFunction)0, // INVALID !!!
    D3D11SAMPLER_COMPARE_NEVER,
    D3D11SAMPLER_COMPARE_LESSER,
    D3D11SAMPLER_COMPARE_EQUAL,
    D3D11SAMPLER_COMPARE_LESSER_EQUAL,
    D3D11SAMPLER_COMPARE_GREATER,
    D3D11SAMPLER_COMPARE_NOT_EQUAL,
    D3D11SAMPLER_COMPARE_GREATER_EQUAL,
    D3D11SAMPLER_COMPARE_ALLWAYS
};
DWord D3D11SamplerCompareFunctionToD3D11[D3D11SAMPLER_COMPARE_COUNT] = {
    D3D11_COMPARISON_NEVER,
    D3D11_COMPARISON_ALWAYS,
    D3D11_COMPARISON_EQUAL,
    D3D11_COMPARISON_NOT_EQUAL,
    D3D11_COMPARISON_LESS,
    D3D11_COMPARISON_LESS_EQUAL,
    D3D11_COMPARISON_GREATER,
    D3D11_COMPARISON_GREATER_EQUAL
};

Void D3D11SamplerStateDesc::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_SAMPLER_DESC * pDesc = (const D3D11_SAMPLER_DESC *)pD3D11Desc;

    iFilterMode = D3D11SamplerFilterModeFromD3D11( pDesc->Filter );

    iWrapModeU = D3D11SamplerWrapModeFromD3D11[pDesc->AddressU];
    iWrapModeV = D3D11SamplerWrapModeFromD3D11[pDesc->AddressV];
    iWrapModeW = D3D11SamplerWrapModeFromD3D11[pDesc->AddressW];

    arrBorderColor[0] = pDesc->BorderColor[0];
    arrBorderColor[1] = pDesc->BorderColor[1];
    arrBorderColor[2] = pDesc->BorderColor[2];
    arrBorderColor[3] = pDesc->BorderColor[3];

    fMinLOD = pDesc->MinLOD;
    fMaxLOD = pDesc->MaxLOD;
    fLODBias = pDesc->MipLODBias;

    iMaxAnisotropy = pDesc->MaxAnisotropy;

    iCompareFunction = D3D11SamplerCompareFunctionFromD3D11[pDesc->ComparisonFunc];
}
Void D3D11SamplerStateDesc::ConvertTo( Void * outD3D11Desc ) const
{
    D3D11_SAMPLER_DESC * outDesc = (D3D11_SAMPLER_DESC*)outD3D11Desc;

    outDesc->Filter = (D3D11_FILTER)( D3D11SamplerFilterModeToD3D11(iFilterMode) );

    outDesc->AddressU = (D3D11_TEXTURE_ADDRESS_MODE)( D3D11SamplerWrapModeToD3D11[iWrapModeU] );
    outDesc->AddressV = (D3D11_TEXTURE_ADDRESS_MODE)( D3D11SamplerWrapModeToD3D11[iWrapModeV] );
    outDesc->AddressW = (D3D11_TEXTURE_ADDRESS_MODE)( D3D11SamplerWrapModeToD3D11[iWrapModeW] );

    outDesc->BorderColor[0] = arrBorderColor[0];
    outDesc->BorderColor[1] = arrBorderColor[1];
    outDesc->BorderColor[2] = arrBorderColor[2];
    outDesc->BorderColor[3] = arrBorderColor[3];

    outDesc->MinLOD = fMinLOD;
    outDesc->MaxLOD = fMaxLOD;
    outDesc->MipLODBias = fLODBias;

    outDesc->MaxAnisotropy = iMaxAnisotropy;

    outDesc->ComparisonFunc = (D3D11_COMPARISON_FUNC)( D3D11SamplerCompareFunctionToD3D11[iCompareFunction] );
}

D3D11RasterizerFillMode D3D11RasterizerFillModeFromD3D11[D3D11RASTERIZER_FILL_COUNT + 2] = {
    (D3D11RasterizerFillMode)0, // INVALID !!!
    (D3D11RasterizerFillMode)0, // INVALID !!!
    D3D11RASTERIZER_FILL_WIREFRAME,
    D3D11RASTERIZER_FILL_SOLID
};
DWord D3D11RasterizerFillModeToD3D11[D3D11RASTERIZER_FILL_COUNT] = {
    D3D11_FILL_WIREFRAME,
    D3D11_FILL_SOLID
};

D3D11RasterizerCullMode D3D11RasterizerCullModeFromD3D11[D3D11RASTERIZER_CULL_COUNT + 1] = {
    (D3D11RasterizerCullMode)0, // INVALID !!!
    D3D11RASTERIZER_CULL_NONE,
    D3D11RASTERIZER_CULL_FRONT,
    D3D11RASTERIZER_CULL_BACK
};
DWord D3D11RasterizerCullModeToD3D11[D3D11RASTERIZER_CULL_COUNT] = {
    D3D11_CULL_NONE,
    D3D11_CULL_FRONT,
    D3D11_CULL_BACK
};

Void D3D11RasterizerStateDesc::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_RASTERIZER_DESC * pDesc = (const D3D11_RASTERIZER_DESC *)pD3D11Desc;

    iFillMode = D3D11RasterizerFillModeFromD3D11[pDesc->FillMode];
    iCullMode = D3D11RasterizerCullModeFromD3D11[pDesc->CullMode];
    bFrontCounterClockwise = ( pDesc->FrontCounterClockwise != FALSE );

    iDepthBias = pDesc->DepthBias;
    fDepthBiasClamp = pDesc->DepthBiasClamp;
    fSlopeScaledDepthBias = pDesc->SlopeScaledDepthBias;

    bDepthClipEnabled = ( pDesc->DepthClipEnable != FALSE );

    bScissorEnabled = ( pDesc->ScissorEnable != FALSE );

    bMultisampleEnabled = ( pDesc->MultisampleEnable != FALSE );
    bAntialiasedLineEnabled = ( pDesc->AntialiasedLineEnable != FALSE );
}
Void D3D11RasterizerStateDesc::ConvertTo( Void * outD3D11Desc ) const
{
    D3D11_RASTERIZER_DESC * outDesc = (D3D11_RASTERIZER_DESC*)outD3D11Desc;

    outDesc->FillMode = (D3D11_FILL_MODE)( D3D11RasterizerFillModeToD3D11[iFillMode] );
    outDesc->CullMode = (D3D11_CULL_MODE)( D3D11RasterizerCullModeToD3D11[iCullMode] );
    outDesc->FrontCounterClockwise = (bFrontCounterClockwise) ? TRUE : FALSE;

    outDesc->DepthBias = iDepthBias;
    outDesc->DepthBiasClamp = fDepthBiasClamp;
    outDesc->SlopeScaledDepthBias = fSlopeScaledDepthBias;

    outDesc->DepthClipEnable = (bDepthClipEnabled) ? TRUE : FALSE;

    outDesc->ScissorEnable = (bScissorEnabled) ? TRUE : FALSE;

    outDesc->MultisampleEnable = (bMultisampleEnabled) ? TRUE : FALSE;
    outDesc->AntialiasedLineEnable = (bAntialiasedLineEnabled) ? TRUE : FALSE;
}

D3D11DepthWriteMask D3D11DepthWriteMaskFromD3D11[D3D11DEPTH_WRITEMASK_COUNT] = {
    D3D11DEPTH_WRITEMASK_ZERO,
    D3D11DEPTH_WRITEMASK_ALL
};
DWord D3D11DepthWriteMaskToD3D11[D3D11DEPTH_WRITEMASK_COUNT] = {
    D3D11_DEPTH_WRITE_MASK_ZERO,
    D3D11_DEPTH_WRITE_MASK_ALL
};

D3D11StencilOperation D3D11StencilOperationFromD3D11[D3D11STENCIL_OP_COUNT + 1] = {
    (D3D11StencilOperation)0, // INVALID !!!
    D3D11STENCIL_OP_KEEP,
    D3D11STENCIL_OP_ZERO,
    D3D11STENCIL_OP_REPLACE,
    D3D11STENCIL_OP_INCREMENT_SAT,
    D3D11STENCIL_OP_DECREMENT_SAT,
    D3D11STENCIL_OP_INVERT,
    D3D11STENCIL_OP_INCREMENT,
    D3D11STENCIL_OP_DECREMENT
};
DWord D3D11StencilOperationToD3D11[D3D11STENCIL_OP_COUNT] = {
    D3D11_STENCIL_OP_KEEP,
    D3D11_STENCIL_OP_ZERO,
    D3D11_STENCIL_OP_INVERT,
    D3D11_STENCIL_OP_REPLACE,
    D3D11_STENCIL_OP_INCR,
    D3D11_STENCIL_OP_INCR_SAT,
    D3D11_STENCIL_OP_DECR,
    D3D11_STENCIL_OP_DECR_SAT
};

D3D11DepthStencilCompareFunction D3D11DepthStencilCompareFunctionFromD3D11[D3D11DEPTHSTENCIL_COMPARE_COUNT + 1] = {
    (D3D11DepthStencilCompareFunction)0, // INVALID !!!
    D3D11DEPTHSTENCIL_COMPARE_NEVER,
    D3D11DEPTHSTENCIL_COMPARE_LESSER,
    D3D11DEPTHSTENCIL_COMPARE_EQUAL,
    D3D11DEPTHSTENCIL_COMPARE_LESSER_EQUAL,
    D3D11DEPTHSTENCIL_COMPARE_GREATER,
    D3D11DEPTHSTENCIL_COMPARE_NOT_EQUAL,
    D3D11DEPTHSTENCIL_COMPARE_GREATER_EQUAL,
    D3D11DEPTHSTENCIL_COMPARE_ALLWAYS
};
DWord D3D11DepthStencilCompareFunctionToD3D11[D3D11DEPTHSTENCIL_COMPARE_COUNT] = {
    D3D11_COMPARISON_NEVER,
    D3D11_COMPARISON_ALWAYS,
    D3D11_COMPARISON_EQUAL,
    D3D11_COMPARISON_NOT_EQUAL,
    D3D11_COMPARISON_LESS,
    D3D11_COMPARISON_LESS_EQUAL,
    D3D11_COMPARISON_GREATER,
    D3D11_COMPARISON_GREATER_EQUAL
};

Void D3D11DepthStencilStateDesc::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_DEPTH_STENCIL_DESC * pDesc = (const D3D11_DEPTH_STENCIL_DESC *)pD3D11Desc;

    bDepthEnabled = ( pDesc->DepthEnable != FALSE );
    iDepthWriteMask = D3D11DepthWriteMaskFromD3D11[pDesc->DepthWriteMask];
    iDepthFunction = D3D11DepthStencilCompareFunctionFromD3D11[pDesc->DepthFunc];

    bStencilEnabled = ( pDesc->StencilEnable != FALSE );
    iStencilReadMask = pDesc->StencilReadMask;
    iStencilWriteMask = pDesc->StencilWriteMask;

    hFrontFace.iOnStencilFail = D3D11StencilOperationFromD3D11[pDesc->FrontFace.StencilFailOp];
    hFrontFace.iOnStencilDepthFail = D3D11StencilOperationFromD3D11[pDesc->FrontFace.StencilDepthFailOp];
    hFrontFace.iOnStencilPass = D3D11StencilOperationFromD3D11[pDesc->FrontFace.StencilPassOp];
    hFrontFace.iStencilFunction = D3D11DepthStencilCompareFunctionFromD3D11[pDesc->FrontFace.StencilFunc];

    hBackFace.iOnStencilFail = D3D11StencilOperationFromD3D11[pDesc->BackFace.StencilFailOp];
    hBackFace.iOnStencilDepthFail = D3D11StencilOperationFromD3D11[pDesc->BackFace.StencilDepthFailOp];
    hBackFace.iOnStencilPass = D3D11StencilOperationFromD3D11[pDesc->BackFace.StencilPassOp];
    hBackFace.iStencilFunction = D3D11DepthStencilCompareFunctionFromD3D11[pDesc->BackFace.StencilFunc];
}
Void D3D11DepthStencilStateDesc::ConvertTo( Void * outD3D11Desc ) const
{
    D3D11_DEPTH_STENCIL_DESC * outDesc = (D3D11_DEPTH_STENCIL_DESC*)outD3D11Desc;

    outDesc->DepthEnable = (bDepthEnabled) ? TRUE : FALSE;
    outDesc->DepthWriteMask = (D3D11_DEPTH_WRITE_MASK)( D3D11DepthWriteMaskToD3D11[iDepthWriteMask] );
    outDesc->DepthFunc = (D3D11_COMPARISON_FUNC)( D3D11DepthStencilCompareFunctionToD3D11[iDepthFunction] );

    outDesc->StencilEnable = (bStencilEnabled) ? TRUE : FALSE;
    outDesc->StencilReadMask = iStencilReadMask;
    outDesc->StencilWriteMask = iStencilWriteMask;

    outDesc->FrontFace.StencilFailOp = (D3D11_STENCIL_OP)( D3D11StencilOperationToD3D11[hFrontFace.iOnStencilFail] );
    outDesc->FrontFace.StencilDepthFailOp = (D3D11_STENCIL_OP)( D3D11StencilOperationToD3D11[hFrontFace.iOnStencilDepthFail] );
    outDesc->FrontFace.StencilPassOp = (D3D11_STENCIL_OP)( D3D11StencilOperationToD3D11[hFrontFace.iOnStencilPass] );
    outDesc->FrontFace.StencilFunc = (D3D11_COMPARISON_FUNC)( D3D11DepthStencilCompareFunctionToD3D11[hFrontFace.iStencilFunction] );

    outDesc->BackFace.StencilFailOp = (D3D11_STENCIL_OP)( D3D11StencilOperationToD3D11[hBackFace.iOnStencilFail] );
    outDesc->BackFace.StencilDepthFailOp = (D3D11_STENCIL_OP)( D3D11StencilOperationToD3D11[hBackFace.iOnStencilDepthFail] );
    outDesc->BackFace.StencilPassOp = (D3D11_STENCIL_OP)( D3D11StencilOperationToD3D11[hBackFace.iOnStencilPass] );
    outDesc->BackFace.StencilFunc = (D3D11_COMPARISON_FUNC)( D3D11DepthStencilCompareFunctionToD3D11[hBackFace.iStencilFunction] );
}

D3D11BlendParameter D3D11BlendParameterFromD3D11[D3D11BLEND_PARAM_COUNT + 3] = {
    (D3D11BlendParameter)0, // INVALID !!!
    D3D11BLEND_PARAM_ZERO,
    D3D11BLEND_PARAM_ONE,
    D3D11BLEND_PARAM_SRC_COLOR,
    D3D11BLEND_PARAM_SRC_COLOR_INV,
    D3D11BLEND_PARAM_SRC_ALPHA,
    D3D11BLEND_PARAM_SRC_ALPHA_INV,
    D3D11BLEND_PARAM_DST_ALPHA,
    D3D11BLEND_PARAM_DST_ALPHA_INV,
    D3D11BLEND_PARAM_DST_COLOR,
    D3D11BLEND_PARAM_DST_COLOR_INV,
    D3D11BLEND_PARAM_SRC_ALPHA_SAT,
    (D3D11BlendParameter)0, // INVALID !!!
    (D3D11BlendParameter)0, // INVALID !!!
    D3D11BLEND_PARAM_BLENDFACTOR,
    D3D11BLEND_PARAM_BLENDFACTOR_INV,
    D3D11BLEND_PARAM_SRC1_COLOR,
    D3D11BLEND_PARAM_SRC1_COLOR_INV,
    D3D11BLEND_PARAM_SRC1_ALPHA,
    D3D11BLEND_PARAM_SRC1_ALPHA_INV
};
DWord D3D11BlendParameterToD3D11[D3D11BLEND_PARAM_COUNT] = {
    D3D11_BLEND_ZERO,
    D3D11_BLEND_ONE,
    D3D11_BLEND_SRC_COLOR,
    D3D11_BLEND_INV_SRC_COLOR,
    D3D11_BLEND_SRC_ALPHA,
    D3D11_BLEND_INV_SRC_ALPHA,
    D3D11_BLEND_SRC_ALPHA_SAT,
    D3D11_BLEND_DEST_COLOR,
    D3D11_BLEND_INV_DEST_COLOR,
    D3D11_BLEND_DEST_ALPHA,
    D3D11_BLEND_INV_DEST_ALPHA,
    D3D11_BLEND_BLEND_FACTOR,
    D3D11_BLEND_INV_BLEND_FACTOR,
    D3D11_BLEND_SRC1_COLOR,
    D3D11_BLEND_INV_SRC1_COLOR,
    D3D11_BLEND_SRC1_ALPHA,
    D3D11_BLEND_INV_SRC1_ALPHA
};

D3D11BlendOperation D3D11BlendOperationFromD3D11[D3D11BLEND_OP_COUNT + 1] = {
    (D3D11BlendOperation)0, // INVALID !!!
    D3D11BLEND_OP_ADD,
    D3D11BLEND_OP_SUB,
    D3D11BLEND_OP_SUB_REV,
    D3D11BLEND_OP_MIN,
    D3D11BLEND_OP_MAX
};
DWord D3D11BlendOperationToD3D11[D3D11BLEND_OP_COUNT] = {
    D3D11_BLEND_OP_ADD,
    D3D11_BLEND_OP_SUBTRACT,
    D3D11_BLEND_OP_REV_SUBTRACT,
    D3D11_BLEND_OP_MIN,
    D3D11_BLEND_OP_MAX
};

Byte D3D11BlendColorWriteMaskFromD3D11[D3D11BLEND_COLORWRITEMASK_COUNT] = {
    0, // D3D11BLEND_COLORWRITEMASK_RED
    1, // D3D11BLEND_COLORWRITEMASK_GREEN
    2, // D3D11BLEND_COLORWRITEMASK_BLUE
    3  // D3D11BLEND_COLORWRITEMASK_ALPHA
};
Byte D3D11BlendColorWriteMaskToD3D11[D3D11BLEND_COLORWRITEMASK_COUNT] = {
    0, // D3D11_COLOR_WRITE_ENABLE_RED
    1, // D3D11_COLOR_WRITE_ENABLE_GREEN
    2, // D3D11_COLOR_WRITE_ENABLE_BLUE
    3  // D3D11_COLOR_WRITE_ENABLE_ALPHA
};

Void D3D11BlendStateDesc::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_BLEND_DESC * pDesc = (const D3D11_BLEND_DESC *)pD3D11Desc;

    bAlphaToCoverageEnabled = ( pDesc->AlphaToCoverageEnable != FALSE );
    bIndependentBlendEnabled = ( pDesc->IndependentBlendEnable != FALSE );

    for( UInt i = 0; i < D3D11RENDERER_MAX_RENDERTARGET_SLOTS; ++i ) {
        arrRenderTargets[i].bBlendEnabled = ( pDesc->RenderTarget[i].BlendEnable != FALSE );

        arrRenderTargets[i].iBlendSrc = D3D11BlendParameterFromD3D11[pDesc->RenderTarget[i].SrcBlend];
        arrRenderTargets[i].iBlendSrcAlpha = D3D11BlendParameterFromD3D11[pDesc->RenderTarget[i].SrcBlendAlpha];
        arrRenderTargets[i].iBlendDst = D3D11BlendParameterFromD3D11[pDesc->RenderTarget[i].DestBlend];
        arrRenderTargets[i].iBlendDstAlpha = D3D11BlendParameterFromD3D11[pDesc->RenderTarget[i].DestBlendAlpha];
        arrRenderTargets[i].iBlendOp = D3D11BlendOperationFromD3D11[pDesc->RenderTarget[i].BlendOp];
        arrRenderTargets[i].iBlendOpAlpha = D3D11BlendOperationFromD3D11[pDesc->RenderTarget[i].BlendOpAlpha];

        arrRenderTargets[i].iColorWriteMask = (Byte)( _D3D11ConvertFlags32(D3D11BlendColorWriteMaskFromD3D11, pDesc->RenderTarget[i].RenderTargetWriteMask) );
    }
}
Void D3D11BlendStateDesc::ConvertTo( Void * outD3D11Desc ) const
{
    D3D11_BLEND_DESC * outDesc = (D3D11_BLEND_DESC*)outD3D11Desc;

    outDesc->AlphaToCoverageEnable = (bAlphaToCoverageEnabled) ? TRUE : FALSE;
    outDesc->IndependentBlendEnable = (bIndependentBlendEnabled) ? TRUE : FALSE;

    for( UInt i = 0; i < D3D11RENDERER_MAX_RENDERTARGET_SLOTS; ++i ) {
        outDesc->RenderTarget[i].BlendEnable = (arrRenderTargets[i].bBlendEnabled) ? TRUE : FALSE;

        outDesc->RenderTarget[i].SrcBlend = (D3D11_BLEND)( D3D11BlendParameterToD3D11[arrRenderTargets[i].iBlendSrc] );
        outDesc->RenderTarget[i].SrcBlendAlpha = (D3D11_BLEND)( D3D11BlendParameterToD3D11[arrRenderTargets[i].iBlendSrcAlpha] );
        outDesc->RenderTarget[i].DestBlend = (D3D11_BLEND)( D3D11BlendParameterToD3D11[arrRenderTargets[i].iBlendDst] );
        outDesc->RenderTarget[i].DestBlendAlpha = (D3D11_BLEND)( D3D11BlendParameterToD3D11[arrRenderTargets[i].iBlendDstAlpha] );
        outDesc->RenderTarget[i].BlendOp = (D3D11_BLEND_OP)( D3D11BlendOperationToD3D11[arrRenderTargets[i].iBlendOp] );
        outDesc->RenderTarget[i].BlendOpAlpha = (D3D11_BLEND_OP)( D3D11BlendOperationToD3D11[arrRenderTargets[i].iBlendOpAlpha] );

        outDesc->RenderTarget[i].RenderTargetWriteMask = (UINT8)( _D3D11ConvertFlags32(D3D11BlendColorWriteMaskToD3D11, arrRenderTargets[i].iColorWriteMask) );
    }
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11Asynchronous Definitions
D3D11QueryType D3D11QueryTypeFromD3D11[D3D11QUERY_COUNT + 5] = {
    D3D11QUERY_EVENT,
    D3D11QUERY_OCCLUSION,
    D3D11QUERY_TIMESTAMP,
    D3D11QUERY_TIMESTAMP_FREQUENCY,
    D3D11QUERY_STATS_PIPELINE,
    (D3D11QueryType)0, // INVALID !!!
    D3D11QUERY_STATS_STREAMOUTPUT,
    (D3D11QueryType)0, // INVALID !!!
    D3D11QUERY_STATS_STREAMOUTPUT_0,
    (D3D11QueryType)0, // INVALID !!!
    D3D11QUERY_STATS_STREAMOUTPUT_1,
    (D3D11QueryType)0, // INVALID !!!
    D3D11QUERY_STATS_STREAMOUTPUT_2,
    (D3D11QueryType)0, // INVALID !!!
    D3D11QUERY_STATS_STREAMOUTPUT_3
};
DWord D3D11QueryTypeToD3D11[D3D11QUERY_COUNT] = {
    D3D11_QUERY_EVENT,
    D3D11_QUERY_OCCLUSION,
    D3D11_QUERY_TIMESTAMP_DISJOINT,
    D3D11_QUERY_TIMESTAMP,
    D3D11_QUERY_PIPELINE_STATISTICS,
    D3D11_QUERY_SO_STATISTICS,
    D3D11_QUERY_SO_STATISTICS_STREAM0,
    D3D11_QUERY_SO_STATISTICS_STREAM1,
    D3D11_QUERY_SO_STATISTICS_STREAM2,
    D3D11_QUERY_SO_STATISTICS_STREAM3
};

D3D11PredicateType D3D11PredicateTypeFromD3D11[D3D11PREDICATE_COUNT + 10] = {
    (D3D11PredicateType)0, // INVALID !!!
    (D3D11PredicateType)0, // INVALID !!!
    (D3D11PredicateType)0, // INVALID !!!
    (D3D11PredicateType)0, // INVALID !!!
    (D3D11PredicateType)0, // INVALID !!!
    D3D11PREDICATE_OCCLUSION,
    (D3D11PredicateType)0, // INVALID !!!
    D3D11PREDICATE_OVERFLOW_STREAMOUTPUT,
    (D3D11PredicateType)0, // INVALID !!!
    D3D11PREDICATE_OVERFLOW_STREAMOUTPUT_0,
    (D3D11PredicateType)0, // INVALID !!!
    D3D11PREDICATE_OVERFLOW_STREAMOUTPUT_1,
    (D3D11PredicateType)0, // INVALID !!!
    D3D11PREDICATE_OVERFLOW_STREAMOUTPUT_2,
    (D3D11PredicateType)0, // INVALID !!!
    D3D11PREDICATE_OVERFLOW_STREAMOUTPUT_3
};
DWord D3D11PredicateTypeToD3D11[D3D11PREDICATE_COUNT] = {
    D3D11_QUERY_OCCLUSION_PREDICATE,
    D3D11_QUERY_SO_OVERFLOW_PREDICATE,
    D3D11_QUERY_SO_OVERFLOW_PREDICATE_STREAM0,
    D3D11_QUERY_SO_OVERFLOW_PREDICATE_STREAM1,
    D3D11_QUERY_SO_OVERFLOW_PREDICATE_STREAM2,
    D3D11_QUERY_SO_OVERFLOW_PREDICATE_STREAM3
};

D3D11CounterType D3D11CounterTypeFromD3D11[D3D11COUNTER_COUNT] = {
    D3D11COUNTER_FLOAT,
    D3D11COUNTER_WORD,
    D3D11COUNTER_DWORD,
    D3D11COUNTER_QWORD
};
DWord D3D11CounterTypeToD3D11[D3D11COUNTER_COUNT] = {
    D3D11_COUNTER_TYPE_FLOAT32,
    D3D11_COUNTER_TYPE_UINT16,
    D3D11_COUNTER_TYPE_UINT32,
    D3D11_COUNTER_TYPE_UINT64
};

/////////////////////////////////////////////////////////////////////////////////
// D3D11Resource Definitions
DWord D3D11ResourcePriorityToD3D11[D3D11RESOURCE_PRIORITY_COUNT] = {
    DXGI_RESOURCE_PRIORITY_MINIMUM,
    DXGI_RESOURCE_PRIORITY_LOW,
    DXGI_RESOURCE_PRIORITY_NORMAL,
    DXGI_RESOURCE_PRIORITY_HIGH,
    DXGI_RESOURCE_PRIORITY_MAXIMUM
};

D3D11ResourceUsage D3D11ResourceUsageFromD3D11[D3D11RESOURCE_USAGE_COUNT] = {
    D3D11RESOURCE_USAGE_DEFAULT,
    D3D11RESOURCE_USAGE_CONST,
    D3D11RESOURCE_USAGE_DYNAMIC,
    D3D11RESOURCE_USAGE_STAGING
};
DWord D3D11ResourceUsageToD3D11[D3D11RESOURCE_USAGE_COUNT] = {
    D3D11_USAGE_DEFAULT,
    D3D11_USAGE_IMMUTABLE,
    D3D11_USAGE_DYNAMIC,
    D3D11_USAGE_STAGING
};

Byte D3D11ResourceBindFromD3D11[D3D11RESOURCE_BIND_COUNT] = {
    0, // D3D11RESOURCE_BIND_VERTEX_BUFFER
    1, // D3D11RESOURCE_BIND_INDEX_BUFFER
    2, // D3D11RESOURCE_BIND_CONSTANT_BUFFER
    3, // D3D11RESOURCE_BIND_SHADER_INPUT
    4, // D3D11RESOURCE_BIND_STREAM_OUTPUT
    5, // D3D11RESOURCE_BIND_RENDER_TARGET
    6, // D3D11RESOURCE_BIND_DEPTH_STENCIL
    7  // D3D11RESOURCE_BIND_UNORDERED_ACCESS
};
Byte D3D11ResourceBindToD3D11[D3D11RESOURCE_BIND_COUNT] = {
    0, // D3D11_BIND_VERTEX_BUFFER
    1, // D3D11_BIND_INDEX_BUFFER
    2, // D3D11_BIND_CONSTANT_BUFFER
    3, // D3D11_BIND_SHADER_RESOURCE
    4, // D3D11_BIND_STREAM_OUTPUT
    5, // D3D11_BIND_RENDER_TARGET
    6, // D3D11_BIND_DEPTH_STENCIL
    7  // D3D11_BIND_UNORDERED_ACCESS
};

D3D11ResourceDimension D3D11ResourceDimensionFromD3D11[D3D11RESOURCE_DIMENSION_COUNT] = {
    D3D11RESOURCE_DIMENSION_UNKNOWN,
    D3D11RESOURCE_DIMENSION_BUFFER,
    D3D11RESOURCE_DIMENSION_1D,
    D3D11RESOURCE_DIMENSION_2D,
    D3D11RESOURCE_DIMENSION_3D
};
DWord D3D11ResourceDimensionToD3D11[D3D11RESOURCE_DIMENSION_COUNT] = {
    D3D11_RESOURCE_DIMENSION_UNKNOWN,
    D3D11_RESOURCE_DIMENSION_BUFFER,
    D3D11_RESOURCE_DIMENSION_TEXTURE1D,
    D3D11_RESOURCE_DIMENSION_TEXTURE2D,
    D3D11_RESOURCE_DIMENSION_TEXTURE3D
};

D3D11ResourceLock D3D11ResourceLockFromD3D11[D3D11RESOURCE_LOCK_COUNT] = {
    D3D11RESOURCE_LOCK_NONE,
    D3D11RESOURCE_LOCK_READ,
    D3D11RESOURCE_LOCK_WRITE,
    D3D11RESOURCE_LOCK_READ_WRITE,
    D3D11RESOURCE_LOCK_WRITE_DISCARD,
    D3D11RESOURCE_LOCK_WRITE_NO_OVERWRITE
};
DWord D3D11ResourceLockToD3D11[D3D11RESOURCE_LOCK_COUNT] = {
    (D3D11_MAP)0, // INVALID !!!
    D3D11_MAP_READ,
    D3D11_MAP_WRITE,
    D3D11_MAP_READ_WRITE,
    D3D11_MAP_WRITE_DISCARD,
    D3D11_MAP_WRITE_NO_OVERWRITE
};

Byte D3D11ResourceLockFlagsFromD3D11[D3D11RESOURCE_LOCKFLAG_COUNT + 20] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0 // D3D11RESOURCE_LOCKFLAG_DONT_WAIT
};
Byte D3D11ResourceLockFlagsToD3D11[D3D11RESOURCE_LOCKFLAG_COUNT] = {
    20 // D3D11_MAP_FLAG_DO_NOT_WAIT
};

/////////////////////////////////////////////////////////////////////////////////
// D3D11Buffer Definitions

/////////////////////////////////////////////////////////////////////////////////
// D3D11Texture Definitions
D3D11TextureCubeFace D3D11TextureCubeFaceFromD3D11[D3D11TEXTURE_CUBEFACE_COUNT] = {
    D3D11TEXTURE_CUBEFACE_X_POS,
    D3D11TEXTURE_CUBEFACE_X_NEG,
    D3D11TEXTURE_CUBEFACE_Y_POS,
    D3D11TEXTURE_CUBEFACE_Y_NEG,
    D3D11TEXTURE_CUBEFACE_Z_POS,
    D3D11TEXTURE_CUBEFACE_Z_NEG
};
DWord D3D11TextureCubeFaceToD3D11[D3D11TEXTURE_CUBEFACE_COUNT] = {
    D3D11_TEXTURECUBE_FACE_POSITIVE_X,
    D3D11_TEXTURECUBE_FACE_NEGATIVE_X,
    D3D11_TEXTURECUBE_FACE_POSITIVE_Y,
    D3D11_TEXTURECUBE_FACE_NEGATIVE_Y,
    D3D11_TEXTURECUBE_FACE_POSITIVE_Z,
    D3D11_TEXTURECUBE_FACE_NEGATIVE_Z
};

/////////////////////////////////////////////////////////////////////////////////
// D3D11ResourceView Definitions
D3D11RenderTargetViewDimension D3D11RenderTargetViewDimensionFromD3D11[D3D11RENDERTARGETVIEW_DIM_COUNT] = {
    D3D11RENDERTARGETVIEW_DIM_UNKNOWN,
    D3D11RENDERTARGETVIEW_DIM_BUFFER,
    D3D11RENDERTARGETVIEW_DIM_TEXTURE1D,
    D3D11RENDERTARGETVIEW_DIM_TEXTURE1DARRAY,
    D3D11RENDERTARGETVIEW_DIM_TEXTURE2D,
    D3D11RENDERTARGETVIEW_DIM_TEXTURE2DARRAY,
    D3D11RENDERTARGETVIEW_DIM_TEXTURE2DMS,
    D3D11RENDERTARGETVIEW_DIM_TEXTURE2DMSARRAY,
    D3D11RENDERTARGETVIEW_DIM_TEXTURE3D
};
DWord D3D11RenderTargetViewDimensionToD3D11[D3D11RENDERTARGETVIEW_DIM_COUNT] = {
    D3D11_RTV_DIMENSION_UNKNOWN,
    D3D11_RTV_DIMENSION_BUFFER,
    D3D11_RTV_DIMENSION_TEXTURE1D,
    D3D11_RTV_DIMENSION_TEXTURE1DARRAY,
    D3D11_RTV_DIMENSION_TEXTURE2D,
    D3D11_RTV_DIMENSION_TEXTURE2DARRAY,
    D3D11_RTV_DIMENSION_TEXTURE2DMS,
    D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY,
    D3D11_RTV_DIMENSION_TEXTURE3D
};

Void D3D11RenderTargetViewDesc::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_RENDER_TARGET_VIEW_DESC * pDesc = (const D3D11_RENDER_TARGET_VIEW_DESC *)pD3D11Desc;

    iFormat = PixelFormatFromDXGI[pDesc->Format];

    iViewDimension = D3D11RenderTargetViewDimensionFromD3D11[pDesc->ViewDimension];
    switch( iViewDimension ) {
        case D3D11RENDERTARGETVIEW_DIM_BUFFER:
            hBuffer.iOffset = pDesc->Buffer.ElementOffset;
            hBuffer.iSize = pDesc->Buffer.ElementWidth;
            break;
        case D3D11RENDERTARGETVIEW_DIM_TEXTURE1D:
            hTexture1D.iMipSlice = pDesc->Texture1D.MipSlice;
            break;
        case D3D11RENDERTARGETVIEW_DIM_TEXTURE1DARRAY:
            hTexture1DArray.iMipSlice = pDesc->Texture1DArray.MipSlice;
            hTexture1DArray.iArraySlice = pDesc->Texture1DArray.FirstArraySlice;
            hTexture1DArray.iArraySliceCount = pDesc->Texture1DArray.ArraySize;
            break;
        case D3D11RENDERTARGETVIEW_DIM_TEXTURE2D:
            hTexture2D.iMipSlice = pDesc->Texture2D.MipSlice;
            break;
        case D3D11RENDERTARGETVIEW_DIM_TEXTURE2DARRAY:
            hTexture2DArray.iMipSlice = pDesc->Texture2DArray.MipSlice;
            hTexture2DArray.iArraySlice = pDesc->Texture2DArray.FirstArraySlice;
            hTexture2DArray.iArraySliceCount = pDesc->Texture2DArray.ArraySize;
            break;
        case D3D11RENDERTARGETVIEW_DIM_TEXTURE2DMS:
            hTexture2DMS._reserved = 0;
            break;
        case D3D11RENDERTARGETVIEW_DIM_TEXTURE2DMSARRAY:
            hTexture2DMSArray.iArraySlice = pDesc->Texture2DMSArray.FirstArraySlice;
            hTexture2DMSArray.iArraySliceCount = pDesc->Texture2DMSArray.ArraySize;
            break;
        case D3D11RENDERTARGETVIEW_DIM_TEXTURE3D:
            hTexture3D.iMipSlice = pDesc->Texture3D.MipSlice;
            hTexture3D.iDepthSlice = pDesc->Texture3D.FirstWSlice;
            hTexture3D.iDepthSliceCount = pDesc->Texture3D.WSize;
            break;
        default:
            DebugAssert( false );
            break;
    }
}
Void D3D11RenderTargetViewDesc::ConvertTo( Void * outD3D11Desc ) const
{
    D3D11_RENDER_TARGET_VIEW_DESC * outDesc = (D3D11_RENDER_TARGET_VIEW_DESC*)outD3D11Desc;

    outDesc->Format = (DXGI_FORMAT)( PixelFormatToDXGI[iFormat] );

    outDesc->ViewDimension = (D3D11_RTV_DIMENSION)( D3D11RenderTargetViewDimensionToD3D11[iViewDimension] );
    switch( iViewDimension ) {
        case D3D11RENDERTARGETVIEW_DIM_BUFFER:
            outDesc->Buffer.ElementOffset = hBuffer.iOffset;
            outDesc->Buffer.ElementWidth = hBuffer.iSize;
            break;
        case D3D11RENDERTARGETVIEW_DIM_TEXTURE1D:
            outDesc->Texture1D.MipSlice = hTexture1D.iMipSlice;
            break;
        case D3D11RENDERTARGETVIEW_DIM_TEXTURE1DARRAY:
            outDesc->Texture1DArray.MipSlice = hTexture1DArray.iMipSlice;
            outDesc->Texture1DArray.FirstArraySlice = hTexture1DArray.iArraySlice;
            outDesc->Texture1DArray.ArraySize = hTexture1DArray.iArraySliceCount;
            break;
        case D3D11RENDERTARGETVIEW_DIM_TEXTURE2D:
            outDesc->Texture2D.MipSlice = hTexture2D.iMipSlice;
            break;
        case D3D11RENDERTARGETVIEW_DIM_TEXTURE2DARRAY:
            outDesc->Texture2DArray.MipSlice = hTexture2DArray.iMipSlice;
            outDesc->Texture2DArray.FirstArraySlice = hTexture2DArray.iArraySlice;
            outDesc->Texture2DArray.ArraySize = hTexture2DArray.iArraySliceCount;
            break;
        case D3D11RENDERTARGETVIEW_DIM_TEXTURE2DMS:
            outDesc->Texture2DMS.UnusedField_NothingToDefine = 0;
            break;
        case D3D11RENDERTARGETVIEW_DIM_TEXTURE2DMSARRAY:
            outDesc->Texture2DMSArray.FirstArraySlice = hTexture2DMSArray.iArraySlice;
            outDesc->Texture2DMSArray.ArraySize = hTexture2DMSArray.iArraySliceCount;
            break;
        case D3D11RENDERTARGETVIEW_DIM_TEXTURE3D:
            outDesc->Texture3D.MipSlice = hTexture3D.iMipSlice;
            outDesc->Texture3D.FirstWSlice = hTexture3D.iDepthSlice;
            outDesc->Texture3D.WSize = hTexture3D.iDepthSliceCount;
            break;
        default:
            DebugAssert( false );
            break;
    }
}

D3D11DepthStencilViewDimension D3D11DepthStencilViewDimensionFromD3D11[D3D11DEPTHSTENCILVIEW_DIM_COUNT] = {
    D3D11DEPTHSTENCILVIEW_DIM_UNKNOWN,
    D3D11DEPTHSTENCILVIEW_DIM_TEXTURE1D,
    D3D11DEPTHSTENCILVIEW_DIM_TEXTURE1DARRAY,
    D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2D,
    D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2DARRAY,
    D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2DMS,
    D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2DMSARRAY
};
DWord D3D11DepthStencilViewDimensionToD3D11[D3D11DEPTHSTENCILVIEW_DIM_COUNT] = {
    D3D11_DSV_DIMENSION_UNKNOWN,
    D3D11_DSV_DIMENSION_TEXTURE1D,
    D3D11_DSV_DIMENSION_TEXTURE1DARRAY,
    D3D11_DSV_DIMENSION_TEXTURE2D,
    D3D11_DSV_DIMENSION_TEXTURE2DARRAY,
    D3D11_DSV_DIMENSION_TEXTURE2DMS,
    D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY
};

Byte D3D11DepthStencilViewFlagsFromD3D11[D3D11DEPTHSTENCILVIEW_FLAG_COUNT] = {
    0, // D3D11DEPTHSTENCILVIEW_FLAG_READONLY_DEPTH
    1  // D3D11DEPTHSTENCILVIEW_FLAG_READONLY_STENCIL
};
Byte D3D11DepthStencilViewFlagsToD3D11[D3D11DEPTHSTENCILVIEW_FLAG_COUNT] = {
    0, // D3D11_DSV_READ_ONLY_DEPTH
    1  // D3D11_DSV_READ_ONLY_STENCIL
};

Void D3D11DepthStencilViewDesc::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_DEPTH_STENCIL_VIEW_DESC * pDesc = (const D3D11_DEPTH_STENCIL_VIEW_DESC *)pD3D11Desc;

    iFormat = PixelFormatFromDXGI[pDesc->Format];
    iFlags = _D3D11ConvertFlags32( D3D11DepthStencilViewFlagsFromD3D11, pDesc->Flags );

    iViewDimension = D3D11DepthStencilViewDimensionFromD3D11[pDesc->ViewDimension];
    switch( iViewDimension ) {
        case D3D11DEPTHSTENCILVIEW_DIM_TEXTURE1D:
            hTexture1D.iMipSlice = pDesc->Texture1D.MipSlice;
            break;
        case D3D11DEPTHSTENCILVIEW_DIM_TEXTURE1DARRAY:
            hTexture1DArray.iMipSlice = pDesc->Texture1DArray.MipSlice;
            hTexture1DArray.iArraySlice = pDesc->Texture1DArray.FirstArraySlice;
            hTexture1DArray.iArraySliceCount = pDesc->Texture1DArray.ArraySize;
            break;
        case D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2D:
            hTexture2D.iMipSlice = pDesc->Texture2D.MipSlice;
            break;
        case D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2DARRAY:
            hTexture2DArray.iMipSlice = pDesc->Texture2DArray.MipSlice;
            hTexture2DArray.iArraySlice = pDesc->Texture2DArray.FirstArraySlice;
            hTexture2DArray.iArraySliceCount = pDesc->Texture2DArray.ArraySize;
            break;
        case D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2DMS:
            hTexture2DMS._reserved = 0;
            break;
        case D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2DMSARRAY:
            hTexture2DMSArray.iArraySlice = pDesc->Texture2DMSArray.FirstArraySlice;
            hTexture2DMSArray.iArraySliceCount = pDesc->Texture2DMSArray.ArraySize;
            break;
        default:
            DebugAssert( false );
            break;
    }
}
Void D3D11DepthStencilViewDesc::ConvertTo( Void * outD3D11Desc ) const
{
    D3D11_DEPTH_STENCIL_VIEW_DESC * outDesc = (D3D11_DEPTH_STENCIL_VIEW_DESC*)outD3D11Desc;

    outDesc->Format = (DXGI_FORMAT)( PixelFormatToDXGI[iFormat] );
    outDesc->Flags = _D3D11ConvertFlags32( D3D11DepthStencilViewFlagsToD3D11, iFlags );

    outDesc->ViewDimension = (D3D11_DSV_DIMENSION)( D3D11DepthStencilViewDimensionToD3D11[iViewDimension] );
    switch( iViewDimension ) {
        case D3D11DEPTHSTENCILVIEW_DIM_TEXTURE1D:
            outDesc->Texture1D.MipSlice = hTexture1D.iMipSlice;
            break;
        case D3D11DEPTHSTENCILVIEW_DIM_TEXTURE1DARRAY:
            outDesc->Texture1DArray.MipSlice = hTexture1DArray.iMipSlice;
            outDesc->Texture1DArray.FirstArraySlice = hTexture1DArray.iArraySlice;
            outDesc->Texture1DArray.ArraySize = hTexture1DArray.iArraySliceCount;
            break;
        case D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2D:
            outDesc->Texture2D.MipSlice = hTexture2D.iMipSlice;
            break;
        case D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2DARRAY:
            outDesc->Texture2DArray.MipSlice = hTexture2DArray.iMipSlice;
            outDesc->Texture2DArray.FirstArraySlice = hTexture2DArray.iArraySlice;
            outDesc->Texture2DArray.ArraySize = hTexture2DArray.iArraySliceCount;
            break;
        case D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2DMS:
            outDesc->Texture2DMS.UnusedField_NothingToDefine = 0;
            break;
        case D3D11DEPTHSTENCILVIEW_DIM_TEXTURE2DMSARRAY:
            outDesc->Texture2DMSArray.FirstArraySlice = hTexture2DMSArray.iArraySlice;
            outDesc->Texture2DMSArray.ArraySize = hTexture2DMSArray.iArraySliceCount;
            break;
        default:
            DebugAssert( false );
            break;
    }
}

D3D11ShaderViewDimension D3D11ShaderViewDimensionFromD3D11[D3D11SHADERVIEW_DIM_COUNT] = {
    D3D11SHADERVIEW_DIM_UNKNOWN,
    D3D11SHADERVIEW_DIM_BUFFER,
    D3D11SHADERVIEW_DIM_TEXTURE1D,
    D3D11SHADERVIEW_DIM_TEXTURE1DARRAY,
    D3D11SHADERVIEW_DIM_TEXTURE2D,
    D3D11SHADERVIEW_DIM_TEXTURE2DARRAY,
    D3D11SHADERVIEW_DIM_TEXTURE2DMS,
    D3D11SHADERVIEW_DIM_TEXTURE2DMSARRAY,
    D3D11SHADERVIEW_DIM_TEXTURE3D,
    D3D11SHADERVIEW_DIM_TEXTURECUBE,
    D3D11SHADERVIEW_DIM_TEXTURECUBEARRAY,
    D3D11SHADERVIEW_DIM_BUFFEREX,
};
DWord D3D11ShaderViewDimensionToD3D11[D3D11SHADERVIEW_DIM_COUNT] = {
    D3D11_SRV_DIMENSION_UNKNOWN,
    D3D11_SRV_DIMENSION_BUFFER,
    D3D11_SRV_DIMENSION_BUFFEREX,
    D3D11_SRV_DIMENSION_TEXTURE1D,
    D3D11_SRV_DIMENSION_TEXTURE1DARRAY,
    D3D11_SRV_DIMENSION_TEXTURE2D,
    D3D11_SRV_DIMENSION_TEXTURE2DARRAY,
    D3D11_SRV_DIMENSION_TEXTURE2DMS,
    D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY,
    D3D11_SRV_DIMENSION_TEXTURE3D,
    D3D11_SRV_DIMENSION_TEXTURECUBE,
    D3D11_SRV_DIMENSION_TEXTURECUBEARRAY
};

Byte D3D11ShaderViewBufferExFlagsFromD3D11[D3D11SHADERVIEW_BUFFEREXFLAG_COUNT] = {
    0 // D3D11SHADERVIEW_BUFFEREXFLAG_RAW
};
Byte D3D11ShaderViewBufferExFlagsToD3D11[D3D11SHADERVIEW_BUFFEREXFLAG_COUNT] = {
    0 // D3D11_BUFFEREX_SRV_FLAG_RAW
};

Void D3D11ShaderViewDesc::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_SHADER_RESOURCE_VIEW_DESC * pDesc = (const D3D11_SHADER_RESOURCE_VIEW_DESC *)pD3D11Desc;

    iFormat = PixelFormatFromDXGI[pDesc->Format];

    iViewDimension = D3D11ShaderViewDimensionFromD3D11[pDesc->ViewDimension];
    switch( iViewDimension ) {
        case D3D11SHADERVIEW_DIM_BUFFER:
            hBuffer.iOffset = pDesc->Buffer.ElementOffset;
            hBuffer.iSize = pDesc->Buffer.ElementWidth;
            break;
        case D3D11SHADERVIEW_DIM_BUFFEREX:
            hBufferEx.iOffset = pDesc->BufferEx.FirstElement;
            hBufferEx.iSize = pDesc->BufferEx.NumElements;
            hBufferEx.iFlags = _D3D11ConvertFlags32( D3D11ShaderViewBufferExFlagsFromD3D11, pDesc->BufferEx.Flags );
            break;
        case D3D11SHADERVIEW_DIM_TEXTURE1D:
            hTexture1D.iMostDetailedMip = pDesc->Texture1D.MostDetailedMip;
            hTexture1D.iMipLevels = pDesc->Texture1D.MipLevels;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURE1DARRAY:
            hTexture1DArray.iMostDetailedMip = pDesc->Texture1DArray.MostDetailedMip;
            hTexture1DArray.iMipLevels = pDesc->Texture1DArray.MipLevels;
            hTexture1DArray.iArraySlice = pDesc->Texture1DArray.FirstArraySlice;
            hTexture1DArray.iArraySliceCount = pDesc->Texture1DArray.ArraySize;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURE2D:
            hTexture2D.iMostDetailedMip = pDesc->Texture2D.MostDetailedMip;
            hTexture2D.iMipLevels = pDesc->Texture2D.MipLevels;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURE2DARRAY:
            hTexture2DArray.iMostDetailedMip = pDesc->Texture2DArray.MostDetailedMip;
            hTexture2DArray.iMipLevels = pDesc->Texture2DArray.MipLevels;
            hTexture2DArray.iArraySlice = pDesc->Texture2DArray.FirstArraySlice;
            hTexture2DArray.iArraySliceCount = pDesc->Texture2DArray.ArraySize;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURE2DMS:
            hTexture2DMS._reserved = 0;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURE2DMSARRAY:
            hTexture2DMSArray.iArraySlice = pDesc->Texture2DMSArray.FirstArraySlice;
            hTexture2DMSArray.iArraySliceCount = pDesc->Texture2DMSArray.ArraySize;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURE3D:
            hTexture3D.iMostDetailedMip = pDesc->Texture3D.MostDetailedMip;
            hTexture3D.iMipLevels = pDesc->Texture3D.MipLevels;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURECUBE:
            hTextureCube.iMostDetailedMip = pDesc->TextureCube.MostDetailedMip;
            hTextureCube.iMipLevels = pDesc->TextureCube.MipLevels;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURECUBEARRAY:
            hTextureCubeArray.iMostDetailedMip = pDesc->TextureCubeArray.MostDetailedMip;
            hTextureCubeArray.iMipLevels = pDesc->TextureCubeArray.MipLevels;
            hTextureCubeArray.iFirstFaceIndex = pDesc->TextureCubeArray.First2DArrayFace;
            hTextureCubeArray.iCubeCount = pDesc->TextureCubeArray.NumCubes;
            break;
        default:
            DebugAssert( false );
            break;
    }
}
Void D3D11ShaderViewDesc::ConvertTo( Void * outD3D11Desc ) const
{
    D3D11_SHADER_RESOURCE_VIEW_DESC * outDesc = (D3D11_SHADER_RESOURCE_VIEW_DESC*)outD3D11Desc;

    outDesc->Format = (DXGI_FORMAT)( PixelFormatToDXGI[iFormat] );

    outDesc->ViewDimension = (D3D11_SRV_DIMENSION)( D3D11ShaderViewDimensionToD3D11[iViewDimension] );
    switch( iViewDimension ) {
        case D3D11SHADERVIEW_DIM_BUFFER:
            outDesc->Buffer.ElementOffset = hBuffer.iOffset;
            outDesc->Buffer.ElementWidth = hBuffer.iSize;
            break;
        case D3D11SHADERVIEW_DIM_BUFFEREX:
            outDesc->BufferEx.FirstElement = hBufferEx.iOffset;
            outDesc->BufferEx.NumElements = hBufferEx.iSize;
            outDesc->BufferEx.Flags = _D3D11ConvertFlags32( D3D11ShaderViewBufferExFlagsToD3D11, hBufferEx.iFlags );
            break;
        case D3D11SHADERVIEW_DIM_TEXTURE1D:
            outDesc->Texture1D.MostDetailedMip = hTexture1D.iMostDetailedMip;
            outDesc->Texture1D.MipLevels = hTexture1D.iMipLevels;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURE1DARRAY:
            outDesc->Texture1DArray.MostDetailedMip = hTexture1DArray.iMostDetailedMip;
            outDesc->Texture1DArray.MipLevels = hTexture1DArray.iMipLevels;
            outDesc->Texture1DArray.FirstArraySlice = hTexture1DArray.iArraySlice;
            outDesc->Texture1DArray.ArraySize = hTexture1DArray.iArraySliceCount;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURE2D:
            outDesc->Texture2D.MostDetailedMip = hTexture2D.iMostDetailedMip;
            outDesc->Texture2D.MipLevels = hTexture2D.iMipLevels;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURE2DARRAY:
            outDesc->Texture2DArray.MostDetailedMip = hTexture2DArray.iMostDetailedMip;
            outDesc->Texture2DArray.MipLevels = hTexture2DArray.iMipLevels;
            outDesc->Texture2DArray.FirstArraySlice = hTexture2DArray.iArraySlice;
            outDesc->Texture2DArray.ArraySize = hTexture2DArray.iArraySliceCount;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURE2DMS:
            outDesc->Texture2DMS.UnusedField_NothingToDefine = 0;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURE2DMSARRAY:
            outDesc->Texture2DMSArray.FirstArraySlice = hTexture2DMSArray.iArraySlice;
            outDesc->Texture2DMSArray.ArraySize = hTexture2DMSArray.iArraySliceCount;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURE3D:
            outDesc->Texture3D.MostDetailedMip = hTexture3D.iMostDetailedMip;
            outDesc->Texture3D.MipLevels = hTexture3D.iMipLevels;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURECUBE:
            outDesc->TextureCube.MostDetailedMip = hTextureCube.iMostDetailedMip;
            outDesc->TextureCube.MipLevels = hTextureCube.iMipLevels;
            break;
        case D3D11SHADERVIEW_DIM_TEXTURECUBEARRAY:
            outDesc->TextureCubeArray.MostDetailedMip = hTextureCubeArray.iMostDetailedMip;
            outDesc->TextureCubeArray.MipLevels = hTextureCubeArray.iMipLevels;
            outDesc->TextureCubeArray.First2DArrayFace = hTextureCubeArray.iFirstFaceIndex;
            outDesc->TextureCubeArray.NumCubes = hTextureCubeArray.iCubeCount;
            break;
        default:
            DebugAssert( false );
            break;
    }
}

D3D11UnorderedAccessViewDimension D3D11UnorderedAccessViewDimensionFromD3D11[D3D11UNORDEREDACCESSVIEW_DIM_COUNT + 2] = {
    D3D11UNORDEREDACCESSVIEW_DIM_UNKNOWN,
    D3D11UNORDEREDACCESSVIEW_DIM_BUFFER,
    D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE1D,
    D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE1DARRAY,
    D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE2D,
    D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE2DARRAY,
    (D3D11UnorderedAccessViewDimension)0, // INVALID !!!
    (D3D11UnorderedAccessViewDimension)0, // INVALID !!!
    D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE3D
};
DWord D3D11UnorderedAccessViewDimensionToD3D11[D3D11UNORDEREDACCESSVIEW_DIM_COUNT] = {
    D3D11_UAV_DIMENSION_UNKNOWN,
    D3D11_UAV_DIMENSION_BUFFER,
    D3D11_UAV_DIMENSION_TEXTURE1D,
    D3D11_UAV_DIMENSION_TEXTURE1DARRAY,
    D3D11_UAV_DIMENSION_TEXTURE2D,
    D3D11_UAV_DIMENSION_TEXTURE2DARRAY,
    D3D11_UAV_DIMENSION_TEXTURE3D
};

Byte D3D11UnorderedAccessViewBufferFlagsFromD3D11[D3D11UNORDEREDACCESSVIEW_BUFFERFLAG_COUNT] = {
    0, // D3D11UNORDEREDACCESSVIEW_BUFFERFLAG_RAW
    1, // D3D11UNORDEREDACCESSVIEW_BUFFERFLAG_APPEND
    2  // D3D11UNORDEREDACCESSVIEW_BUFFERFLAG_COUNTER
};
Byte D3D11UnorderedAccessViewBufferFlagsToD3D11[D3D11UNORDEREDACCESSVIEW_BUFFERFLAG_COUNT] = {
    0, // D3D11_BUFFER_UAV_FLAG_RAW
    1, // D3D11_BUFFER_UAV_FLAG_APPEND
    2  // D3D11_BUFFER_UAV_FLAG_COUNTER
};

Void D3D11UnorderedAccessViewDesc::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_UNORDERED_ACCESS_VIEW_DESC * pDesc = (const D3D11_UNORDERED_ACCESS_VIEW_DESC *)pD3D11Desc;

    iFormat = PixelFormatFromDXGI[pDesc->Format];

    iViewDimension = D3D11UnorderedAccessViewDimensionFromD3D11[pDesc->ViewDimension];
    switch( iViewDimension ) {
        case D3D11UNORDEREDACCESSVIEW_DIM_BUFFER:
            hBuffer.iOffset = pDesc->Buffer.FirstElement;
            hBuffer.iSize = pDesc->Buffer.NumElements;
            hBuffer.iFlags = _D3D11ConvertFlags32( D3D11UnorderedAccessViewBufferFlagsFromD3D11, pDesc->Buffer.Flags );
            break;
        case D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE1D:
            hTexture1D.iMipSlice = pDesc->Texture1D.MipSlice;
            break;
        case D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE1DARRAY:
            hTexture1DArray.iMipSlice = pDesc->Texture1DArray.MipSlice;
            hTexture1DArray.iArraySlice = pDesc->Texture1DArray.FirstArraySlice;
            hTexture1DArray.iArraySliceCount = pDesc->Texture1DArray.ArraySize;
            break;
        case D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE2D:
            hTexture2D.iMipSlice = pDesc->Texture2D.MipSlice;
            break;
        case D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE2DARRAY:
            hTexture2DArray.iMipSlice = pDesc->Texture2DArray.MipSlice;
            hTexture2DArray.iArraySlice = pDesc->Texture2DArray.FirstArraySlice;
            hTexture2DArray.iArraySliceCount = pDesc->Texture2DArray.ArraySize;
            break;
        case D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE3D:
            hTexture3D.iMipSlice = pDesc->Texture3D.MipSlice;
            hTexture3D.iDepthSlice = pDesc->Texture3D.FirstWSlice;
            hTexture3D.iDepthSliceCount = pDesc->Texture3D.WSize;
            break;
        default:
            DebugAssert( false );
            break;
    }
}
Void D3D11UnorderedAccessViewDesc::ConvertTo( Void * outD3D11Desc ) const
{
    D3D11_UNORDERED_ACCESS_VIEW_DESC * outDesc = (D3D11_UNORDERED_ACCESS_VIEW_DESC*)outD3D11Desc;

    outDesc->Format = (DXGI_FORMAT)( PixelFormatToDXGI[iFormat] );
    
    outDesc->ViewDimension = (D3D11_UAV_DIMENSION)( D3D11UnorderedAccessViewDimensionToD3D11[iViewDimension] );
    switch( iViewDimension ) {
        case D3D11UNORDEREDACCESSVIEW_DIM_BUFFER:
            outDesc->Buffer.FirstElement = hBuffer.iOffset;
            outDesc->Buffer.NumElements = hBuffer.iSize;
            outDesc->Buffer.Flags = _D3D11ConvertFlags32( D3D11UnorderedAccessViewBufferFlagsToD3D11, hBuffer.iFlags );
            break;
        case D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE1D:
            outDesc->Texture1D.MipSlice = hTexture1D.iMipSlice;
            break;
        case D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE1DARRAY:
            outDesc->Texture1DArray.MipSlice = hTexture1DArray.iMipSlice;
            outDesc->Texture1DArray.FirstArraySlice = hTexture1DArray.iArraySlice;
            outDesc->Texture1DArray.ArraySize = hTexture1DArray.iArraySliceCount;
            break;
        case D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE2D:
            outDesc->Texture2D.MipSlice = hTexture2D.iMipSlice;
            break;
        case D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE2DARRAY:
            outDesc->Texture2DArray.MipSlice = hTexture2DArray.iMipSlice;
            outDesc->Texture2DArray.FirstArraySlice = hTexture2DArray.iArraySlice;
            outDesc->Texture2DArray.ArraySize = hTexture2DArray.iArraySliceCount;
            break;
        case D3D11UNORDEREDACCESSVIEW_DIM_TEXTURE3D:
            outDesc->Texture3D.MipSlice = hTexture3D.iMipSlice;
            outDesc->Texture3D.FirstWSlice = hTexture3D.iDepthSlice;
            outDesc->Texture3D.WSize = hTexture3D.iDepthSliceCount;
            break;
        default:
            DebugAssert( false );
            break;
    }
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11Shader Definitions
Void D3D11StreamOutputDeclaration::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_SO_DECLARATION_ENTRY * pDesc = (const D3D11_SO_DECLARATION_ENTRY *)pD3D11Desc;

    iStream = pDesc->Stream;

    strSemanticName = pDesc->SemanticName;
    iSemanticIndex = pDesc->SemanticIndex;

    iStartComponent = pDesc->StartComponent;
    iComponentCount = pDesc->ComponentCount;
    iOutputSlot = pDesc->OutputSlot;
}
Void D3D11StreamOutputDeclaration::ConvertTo( Void * outD3D11Desc ) const
{
    D3D11_SO_DECLARATION_ENTRY * outDesc = (D3D11_SO_DECLARATION_ENTRY*)outD3D11Desc;

    outDesc->Stream = iStream;

    outDesc->SemanticName = strSemanticName;
    outDesc->SemanticIndex = iSemanticIndex;

    outDesc->StartComponent = iStartComponent;
    outDesc->ComponentCount = iComponentCount;
    outDesc->OutputSlot = iOutputSlot;
}

Byte D3D11ShaderCompilationFlagsFromD3D11[D3D11SHADER_COMPILE_COUNT + 2] = {
    0,  // D3D11SHADER_COMPILE_DEBUG
    6,  // D3D11SHADER_COMPILE_SKIP_VALIDATION
    5,  // D3D11SHADER_COMPILE_SKIP_OPTIMIZATION
    11, // D3D11SHADER_COMPILE_PACK_MATRIX_ROW_MAJOR
    12, // D3D11SHADER_COMPILE_PACK_MATRIX_COLUMN_MAJOR
    10, // D3D11SHADER_COMPILE_PARTIAL_PRECISION
    7,  // D3D11SHADER_COMPILE_VS_SOFTWARE_NO_OPT
    8,  // D3D11SHADER_COMPILE_PS_SOFTWARE_NO_OPT
    9,  // D3D11SHADER_COMPILE_NO_PRESHADER
    13, // D3D11SHADER_COMPILE_AVOID_FLOW_CONTROL
    14, // D3D11SHADER_COMPILE_PREFER_FLOW_CONTROL
    2,  // D3D11SHADER_COMPILE_STRICT
    4,  // D3D11SHADER_COMPILE_BACKWARDS_COMPATIBILITY
    3,  // D3D11SHADER_COMPILE_STRICT_IEEE
    15, // D3D11SHADER_COMPILE_OPTIMIZATION_COMPILESPEED
    16, // D3D11SHADER_COMPILE_OPTIMIZATION_EXECUTESPEED
    0, 0,
    1   // D3D11SHADER_COMPILE_WARNINGS_AS_ERRORS
};
Byte D3D11ShaderCompilationFlagsToD3D11[D3D11SHADER_COMPILE_COUNT] = {
    0,  // D3DCOMPILE_DEBUG
    18, // D3DCOMPILE_WARNINGS_ARE_ERRORS
    11, // D3DCOMPILE_ENABLE_STRICTNESS
    13, // D3DCOMPILE_IEEE_STRICTNESS
    12, // D3DCOMPILE_ENABLE_BACKWARDS_COMPATIBILITY
    2,  // D3DCOMPILE_SKIP_OPTIMIZATION
    1,  // D3DCOMPILE_SKIP_VALIDATION
    6,  // D3DCOMPILE_FORCE_VS_SOFTWARE_NO_OPT
    7,  // D3DCOMPILE_FORCE_PS_SOFTWARE_NO_OPT
    8,  // D3DCOMPILE_NO_PRESHADER
    5,  // D3DCOMPILE_PARTIAL_PRECISION
    3,  // D3DCOMPILE_PACK_MATRIX_ROW_MAJOR
    4,  // D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR
    9,  // D3DCOMPILE_AVOID_FLOW_CONTROL
    10, // D3DCOMPILE_PREFER_FLOW_CONTROL
    14, // D3DCOMPILE_OPTIMIZATION_LEVEL0
    15  // D3DCOMPILE_OPTIMIZATION_LEVEL3
};

Byte D3D11ShaderRequirementFlagsFromD3D11[D3D11SHADER_REQUIRES_COUNT] = {
    3, // D3D11SHADER_REQUIRES_DOUBLES
    5, // D3D11SHADER_REQUIRES_EARLY_DEPTH_STENCIL
    7, // D3D11SHADER_REQUIRES_UAVS_AT_EVERY_STAGE
    6, // D3D11SHADER_REQUIRES_UAVS_64
    4, // D3D11SHADER_REQUIRES_MINIMUM_PRECISION
    1, // D3D11SHADER_REQUIRES_11_1_DOUBLE_EXTENSIONS
    2, // D3D11SHADER_REQUIRES_11_1_SHADER_EXTENSIONS
    0  // D3D11SHADER_REQUIRES_9_X_SHADOWS
};
Byte D3D11ShaderRequirementFlagsToD3D11[D3D11SHADER_REQUIRES_COUNT] = {
    7, // D3D_SHADER_REQUIRES_LEVEL_9_COMPARISON_FILTERING
    5, // D3D_SHADER_REQUIRES_11_1_DOUBLE_EXTENSIONS
    6, // D3D_SHADER_REQUIRES_11_1_SHADER_EXTENSIONS
    0, // D3D_SHADER_REQUIRES_DOUBLES
    4, // D3D_SHADER_REQUIRES_MINIMUM_PRECISION
    1, // D3D_SHADER_REQUIRES_EARLY_DEPTH_STENCIL
    3, // D3D_SHADER_REQUIRES_64_UAVS
    2  // D3D_SHADER_REQUIRES_UAVS_AT_EVERY_STAGE
};

D3D11ShaderPrimitive D3D11ShaderPrimitiveFromD3D11[D3D11SHADER_PRIMITIVE_COUNT + 3] = {
    D3D11SHADER_PRIMITIVE_UNDEFINED,
    D3D11SHADER_PRIMITIVE_POINT,
    D3D11SHADER_PRIMITIVE_LINE,
    D3D11SHADER_PRIMITIVE_TRIANGLE,
    (D3D11ShaderPrimitive)0, // INVALID !!!
    (D3D11ShaderPrimitive)0, // INVALID !!!
    D3D11SHADER_PRIMITIVE_LINE_ADJ,
    D3D11SHADER_PRIMITIVE_TRIANGLE_ADJ,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_1,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_2,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_3,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_4,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_5,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_6,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_7,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_8,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_9,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_10,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_11,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_12,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_13,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_14,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_15,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_16,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_17,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_18,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_19,
    (D3D11ShaderPrimitive)0, // INVALID !!!
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_20,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_21,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_22,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_23,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_24,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_25,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_26,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_27,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_28,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_29,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_30,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_31,
    D3D11SHADER_PRIMITIVE_CONTROL_POINT_PATCH_32
};
DWord D3D11ShaderPrimitiveToD3D11[D3D11SHADER_PRIMITIVE_COUNT] = {
    D3D_PRIMITIVE_UNDEFINED,
    D3D_PRIMITIVE_POINT,
    D3D_PRIMITIVE_LINE,
    D3D_PRIMITIVE_TRIANGLE,
    D3D_PRIMITIVE_LINE_ADJ,
    D3D_PRIMITIVE_TRIANGLE_ADJ,
    D3D_PRIMITIVE_1_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_2_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_3_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_4_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_5_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_6_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_7_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_8_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_9_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_10_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_11_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_12_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_13_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_14_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_15_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_16_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_17_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_18_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_19_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_20_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_21_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_22_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_23_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_24_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_25_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_26_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_27_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_28_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_29_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_30_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_31_CONTROL_POINT_PATCH,
    D3D_PRIMITIVE_32_CONTROL_POINT_PATCH
};

D3D11ShaderPrimitiveTopology _D3D11ShaderPrimitiveTopologyFromD3D11[D3D11SHADER_PRIMITIVETOPOLOGY_COUNT + 5] = {
    D3D11SHADER_PRIMITIVETOPOLOGY_UNDEFINED,
    D3D11SHADER_PRIMITIVETOPOLOGY_POINTLIST,
    D3D11SHADER_PRIMITIVETOPOLOGY_LINELIST,
    D3D11SHADER_PRIMITIVETOPOLOGY_LINESTRIP,
    D3D11SHADER_PRIMITIVETOPOLOGY_TRIANGLELIST,
    D3D11SHADER_PRIMITIVETOPOLOGY_TRIANGLESTRIP,
    (D3D11ShaderPrimitiveTopology)0, // INVALID !!!
    (D3D11ShaderPrimitiveTopology)0, // INVALID !!!
    (D3D11ShaderPrimitiveTopology)0, // INVALID !!!
    (D3D11ShaderPrimitiveTopology)0, // INVALID !!!
    D3D11SHADER_PRIMITIVETOPOLOGY_LINELIST_ADJ,
    D3D11SHADER_PRIMITIVETOPOLOGY_LINESTRIP_ADJ,
    D3D11SHADER_PRIMITIVETOPOLOGY_TRIANGLELIST_ADJ,
    D3D11SHADER_PRIMITIVETOPOLOGY_TRIANGLESTRIP_ADJ,
    (D3D11ShaderPrimitiveTopology)0, // INVALID !!!
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_1,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_2,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_3,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_4,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_5,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_6,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_7,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_8,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_9,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_10,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_11,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_12,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_13,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_14,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_15,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_16,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_17,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_18,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_19,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_20,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_21,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_22,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_23,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_24,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_25,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_26,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_27,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_28,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_29,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_30,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_31,
    D3D11SHADER_PRIMITIVETOPOLOGY_CONTROL_POINT_PATCHLIST_32
};
D3D11ShaderPrimitiveTopology D3D11ShaderPrimitiveTopologyFromD3D11( DWord iD3DPrimitiveTopology )
{
    if ( iD3DPrimitiveTopology >= D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST )
        iD3DPrimitiveTopology -= ( D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST - 15 );
    return _D3D11ShaderPrimitiveTopologyFromD3D11[iD3DPrimitiveTopology];
}
DWord D3D11ShaderPrimitiveTopologyToD3D11[D3D11SHADER_PRIMITIVETOPOLOGY_COUNT] = {
    D3D_PRIMITIVE_TOPOLOGY_UNDEFINED,
    D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
    D3D_PRIMITIVE_TOPOLOGY_LINELIST,
    D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
    D3D_PRIMITIVE_TOPOLOGY_LINELIST_ADJ,
    D3D_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ,
    D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_2_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_5_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_6_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_7_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_8_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_9_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_10_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_11_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_12_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_13_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_14_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_15_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_16_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_17_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_18_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_19_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_20_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_21_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_22_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_23_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_24_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_25_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_26_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_27_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_28_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_29_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_30_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_31_CONTROL_POINT_PATCHLIST,
    D3D_PRIMITIVE_TOPOLOGY_32_CONTROL_POINT_PATCHLIST
};

D3D11ShaderTesselatorOutput D3D11ShaderTesselatorOutputFromD3D11[D3D11SHADER_TESSELLATOROUTPUT_COUNT] = {
    D3D11SHADER_TESSELLATOROUTPUT_UNDEFINED,
    D3D11SHADER_TESSELLATOROUTPUT_POINT,
    D3D11SHADER_TESSELLATOROUTPUT_LINE,
    D3D11SHADER_TESSELLATOROUTPUT_TRIANGLE_CW,
    D3D11SHADER_TESSELLATOROUTPUT_TRIANGLE_CCW
};
DWord D3D11ShaderTesselatorOutputToD3D11[D3D11SHADER_TESSELLATOROUTPUT_COUNT] = {
    D3D_TESSELLATOR_OUTPUT_UNDEFINED,
    D3D_TESSELLATOR_OUTPUT_POINT,
    D3D_TESSELLATOR_OUTPUT_LINE,
    D3D_TESSELLATOR_OUTPUT_TRIANGLE_CW,
    D3D_TESSELLATOR_OUTPUT_TRIANGLE_CCW
};

D3D11ShaderTesselatorPartitioning D3D11ShaderTesselatorPartitioningFromD3D11[D3D11SHADER_TESSELLATORPARTITIONING_COUNT] = {
    D3D11SHADER_TESSELLATORPARTITIONING_UNDEFINED,
    D3D11SHADER_TESSELLATORPARTITIONING_INTEGER,
    D3D11SHADER_TESSELLATORPARTITIONING_POW2,
    D3D11SHADER_TESSELLATORPARTITIONING_FRACTIONAL_ODD,
    D3D11SHADER_TESSELLATORPARTITIONING_FRACTIONAL_EVEN
};
DWord D3D11ShaderTesselatorPartitioningToD3D11[D3D11SHADER_TESSELLATORPARTITIONING_COUNT] = {
    D3D_TESSELLATOR_PARTITIONING_UNDEFINED,
    D3D_TESSELLATOR_PARTITIONING_INTEGER,
    D3D_TESSELLATOR_PARTITIONING_POW2,
    D3D_TESSELLATOR_PARTITIONING_FRACTIONAL_ODD,
    D3D_TESSELLATOR_PARTITIONING_FRACTIONAL_EVEN
};

D3D11ShaderTesselatorDomain D3D11ShaderTesselatorDomainFromD3D11[D3D11SHADER_TESSELLATORDOMAIN_COUNT] = {
    D3D11SHADER_TESSELLATORDOMAIN_UNDEFINED,
    D3D11SHADER_TESSELLATORDOMAIN_ISOLINE,
    D3D11SHADER_TESSELLATORDOMAIN_TRIANGLE,
    D3D11SHADER_TESSELLATORDOMAIN_QUAD
};
DWord D3D11ShaderTesselatorDomainToD3D11[D3D11SHADER_TESSELLATORDOMAIN_COUNT] = {
    D3D_TESSELLATOR_DOMAIN_UNDEFINED,
    D3D_TESSELLATOR_DOMAIN_ISOLINE,
    D3D_TESSELLATOR_DOMAIN_TRI,
    D3D_TESSELLATOR_DOMAIN_QUAD
};

Void D3D11ShaderDesc::ConvertFrom( const Void * pD3D11Desc, Bool bD3D11IsSampleFrequencyShader, UInt iD3D11RequirementFlags,
                                   UInt iD3D11NumInterfaceSlots, UInt iD3D11BitwiseInstructionCount, UInt iD3D11ConversionInstructionCount,
                                   UInt iD3D11MOVInstructionCount, UInt iD3D11MOVCInstructionCount )
{
    const D3D11_SHADER_DESC * pDesc = (const D3D11_SHADER_DESC *)pD3D11Desc;

    iVersion = pDesc->Version;
    strCreator = pDesc->Creator;
    
    bIsSampleFrequencyShader = bD3D11IsSampleFrequencyShader;

    iCompilationFlags = _D3D11ConvertFlags32( D3D11ShaderCompilationFlagsFromD3D11, pDesc->Flags );
    iRequirementFlags = _D3D11ConvertFlags32( D3D11ShaderRequirementFlagsFromD3D11, iD3D11RequirementFlags );

    iInputParameterCount = pDesc->InputParameters;
    iOutputParameterCount = pDesc->OutputParameters;
    iPatchConstantParameterCount = pDesc->PatchConstantParameters;
    iBindingCount = pDesc->BoundResources;
    iConstantBufferCount = pDesc->ConstantBuffers;

    iInterfaceSlotCount = iD3D11NumInterfaceSlots;

    iTempRegisterCount = pDesc->TempRegisterCount;
    iTempArrayCount = pDesc->TempArrayCount;

    iInstructionCount = pDesc->InstructionCount;

    iIntInstructionCount = pDesc->IntInstructionCount;
    iUIntInstructionCount = pDesc->UintInstructionCount;
    iFloatInstructionCount = pDesc->FloatInstructionCount;
    iArrayInstructionCount = pDesc->ArrayInstructionCount;

    iBitwiseInstructionCount = iD3D11BitwiseInstructionCount;
    iConversionInstructionCount = iD3D11ConversionInstructionCount;
    iMOVInstructionCount = iD3D11MOVInstructionCount;
    iMOVCInstructionCount = iD3D11MOVCInstructionCount;

    iDefInstructionCount = pDesc->DefCount;
    iDclInstructionCount = pDesc->DclCount;
    iMacroInstructionCount = pDesc->MacroInstructionCount;

    iTextureNormalInstructionCount = pDesc->TextureNormalInstructions;
    iTextureLoadInstructionCount = pDesc->TextureLoadInstructions;
    iTextureCompInstructionCount = pDesc->TextureCompInstructions;
    iTextureBiasInstructionCount = pDesc->TextureBiasInstructions;
    iTextureGradientInstructionCount = pDesc->TextureGradientInstructions;

    iCutInstructionCount = pDesc->CutInstructionCount;
    iEmitInstructionCount = pDesc->EmitInstructionCount;

    iStaticFlowControlCount = pDesc->StaticFlowControlCount;
    iDynamicFlowControlCount = pDesc->DynamicFlowControlCount;

    iGSHSInputPrimitive = D3D11ShaderPrimitiveFromD3D11[pDesc->InputPrimitive];
    // iGSInputPrimitive = D3D11ShaderPrimitiveFromD3D11[m_pReflector->GetGSInputPrimitive()];

    iGSOutputTopology = D3D11ShaderPrimitiveTopologyFromD3D11( pDesc->GSOutputTopology );
    iGSMaxOutputVertexCount = pDesc->GSMaxOutputVertexCount;
    iGSInstanceCount = pDesc->cGSInstanceCount;

    iTesselatorControlPointCount = pDesc->cControlPoints;
    iTesselatorDomain = D3D11ShaderTesselatorDomainFromD3D11[pDesc->TessellatorDomain];
    iHSOutputPrimitive = D3D11ShaderTesselatorOutputFromD3D11[pDesc->HSOutputPrimitive];
    iHSPartitioningMode = D3D11ShaderTesselatorPartitioningFromD3D11[pDesc->HSPartitioning];

    iCSBarrierInstructionCount = pDesc->cBarrierInstructions;
    iCSInterlockedInstructionCount = pDesc->cInterlockedInstructions;
    iCSTextureStoreInstructionCount = pDesc->cTextureStoreInstructions;
}

D3D11ShaderParameterType _D3D11ShaderParameterTypeFromD3D11[D3D11SHADER_PARAMETER_COUNT + 1] = {
    D3D11SHADER_PARAMETER_UNDEFINED,
    D3D11SHADER_PARAMETER_POSITION,
    D3D11SHADER_PARAMETER_CLIP_DISTANCE,
    D3D11SHADER_PARAMETER_CULL_DISTANCE,
    D3D11SHADER_PARAMETER_RENDER_TARGET_ARRAY_INDEX,
    D3D11SHADER_PARAMETER_VIEWPORT_ARRAY_INDEX,
    D3D11SHADER_PARAMETER_VERTEX_ID,
    D3D11SHADER_PARAMETER_PRIMITIVE_ID,
    D3D11SHADER_PARAMETER_INSTANCE_ID,
    D3D11SHADER_PARAMETER_IS_FRONT_FACE,
    D3D11SHADER_PARAMETER_SAMPLE_INDEX,
    D3D11SHADER_PARAMETER_TESSFACTOR_FINAL_QUAD_EDGE,
    D3D11SHADER_PARAMETER_TESSFACTOR_FINAL_QUAD_INSIDE,
    D3D11SHADER_PARAMETER_TESSFACTOR_FINAL_TRI_EDGE,
    D3D11SHADER_PARAMETER_TESSFACTOR_FINAL_TRI_INSIDE,
    D3D11SHADER_PARAMETER_TESSFACTOR_FINAL_LINE_DETAIL,
    D3D11SHADER_PARAMETER_TESSFACTOR_FINAL_LINE_DENSITY,
    (D3D11ShaderParameterType)0, // INVALID !!!
    D3D11SHADER_PARAMETER_TARGET,
    D3D11SHADER_PARAMETER_DEPTH,
    D3D11SHADER_PARAMETER_COVERAGE,
    D3D11SHADER_PARAMETER_DEPTH_GREATER_EQUAL,
    D3D11SHADER_PARAMETER_DEPTH_LESS_EQUAL
};
D3D11ShaderParameterType D3D11ShaderParameterTypeFromD3D11( DWord iD3DName )
{
    if ( iD3DName >= D3D_NAME_TARGET )
        iD3DName -= ( D3D_NAME_TARGET - 18 );
    return _D3D11ShaderParameterTypeFromD3D11[iD3DName];
}
DWord D3D11ShaderParameterTypeToD3D11[D3D11SHADER_PARAMETER_COUNT] = {
    D3D_NAME_UNDEFINED,
    D3D_NAME_VERTEX_ID,
    D3D_NAME_PRIMITIVE_ID,
    D3D_NAME_INSTANCE_ID,
    D3D_NAME_POSITION,
    D3D_NAME_CLIP_DISTANCE,
    D3D_NAME_CULL_DISTANCE,
    D3D_NAME_IS_FRONT_FACE,
    D3D_NAME_SAMPLE_INDEX,
    D3D_NAME_TARGET,
    D3D_NAME_RENDER_TARGET_ARRAY_INDEX,
    D3D_NAME_VIEWPORT_ARRAY_INDEX,
    D3D_NAME_DEPTH,
    D3D_NAME_DEPTH_GREATER_EQUAL,
    D3D_NAME_DEPTH_LESS_EQUAL,
    D3D_NAME_COVERAGE,
    D3D_NAME_FINAL_LINE_DETAIL_TESSFACTOR,
    D3D_NAME_FINAL_LINE_DENSITY_TESSFACTOR,
    D3D_NAME_FINAL_TRI_EDGE_TESSFACTOR,
    D3D_NAME_FINAL_TRI_INSIDE_TESSFACTOR,
    D3D_NAME_FINAL_QUAD_EDGE_TESSFACTOR,
    D3D_NAME_FINAL_QUAD_INSIDE_TESSFACTOR
};

D3D11ShaderRegisterComponentType D3D11ShaderRegisterComponentTypeFromD3D11[D3D11SHADER_REGISTERCOMPONENT_COUNT] = {
    D3D11SHADER_REGISTERCOMPONENT_UNKNOWN,
    D3D11SHADER_REGISTERCOMPONENT_UINT32,
    D3D11SHADER_REGISTERCOMPONENT_SINT32,
    D3D11SHADER_REGISTERCOMPONENT_FLOAT32
};
DWord D3D11ShaderRegisterComponentTypeToD3D11[D3D11SHADER_REGISTERCOMPONENT_COUNT] = {
    D3D_REGISTER_COMPONENT_UNKNOWN,
    D3D_REGISTER_COMPONENT_FLOAT32,
    D3D_REGISTER_COMPONENT_UINT32,
    D3D_REGISTER_COMPONENT_SINT32
};

D3D11ShaderMinPrecision _D3D11ShaderMinPrecisionFromD3D11[D3D11SHADER_MINPRECISION_COUNT + 2] = {
    D3D11SHADER_MINPRECISION_FLOAT_32,
    D3D11SHADER_MINPRECISION_FLOAT_16,
    D3D11SHADER_MINPRECISION_FLOAT_2_8,
    (D3D11ShaderMinPrecision)0, // INVALID !!!
    D3D11SHADER_MINPRECISION_SINT_16,
    D3D11SHADER_MINPRECISION_UINT_16,
    (D3D11ShaderMinPrecision)0, // INVALID !!!
    D3D11SHADER_MINPRECISION_ANY_16,
    D3D11SHADER_MINPRECISION_ANY_10
};
D3D11ShaderMinPrecision D3D11ShaderMinPrecisionFromD3D11( DWord iD3DMinPrecision )
{
    if ( iD3DMinPrecision >= D3D_MIN_PRECISION_ANY_16 )
        iD3DMinPrecision -= ( D3D_MIN_PRECISION_ANY_16 - 7 );
    return _D3D11ShaderMinPrecisionFromD3D11[iD3DMinPrecision];
}
DWord D3D11ShaderMinPrecisionToD3D11[D3D11SHADER_MINPRECISION_COUNT] = {
    D3D_MIN_PRECISION_DEFAULT,
    D3D_MIN_PRECISION_FLOAT_16,
    D3D_MIN_PRECISION_FLOAT_2_8,
    D3D_MIN_PRECISION_UINT_16,
    D3D_MIN_PRECISION_SINT_16,
    D3D_MIN_PRECISION_ANY_16,
    D3D_MIN_PRECISION_ANY_10
};

Void D3D11ShaderParameterDesc::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_SIGNATURE_PARAMETER_DESC * pDesc = (const D3D11_SIGNATURE_PARAMETER_DESC *)pD3D11Desc;

    iStreamIndex = pDesc->Stream;

    strSemanticName = pDesc->SemanticName;
    iSemanticIndex = pDesc->SemanticIndex;

    iRegister = pDesc->Register;

    iSystemValueType = D3D11ShaderParameterTypeFromD3D11( pDesc->SystemValueType );
    iComponentType = D3D11ShaderRegisterComponentTypeFromD3D11[pDesc->ComponentType];

    iComponentMask = pDesc->Mask;
    iReadWriteMask = pDesc->ReadWriteMask;

    iMinPrecision = D3D11ShaderMinPrecisionFromD3D11( pDesc->MinPrecision );
}

D3D11ShaderInputType D3D11ShaderInputTypeFromD3D11[D3D11SHADER_INPUT_COUNT] = {
    D3D11SHADER_INPUT_CBUFFER,
    D3D11SHADER_INPUT_TBUFFER,
    D3D11SHADER_INPUT_TEXTURE,
    D3D11SHADER_INPUT_SAMPLER,
    D3D11SHADER_INPUT_UAV_RWTYPED,
    D3D11SHADER_INPUT_STRUCTURED,
    D3D11SHADER_INPUT_UAV_RWSTRUCTURED,
    D3D11SHADER_INPUT_BYTEADDRESS,
    D3D11SHADER_INPUT_UAV_RWBYTEADDRESS,
    D3D11SHADER_INPUT_UAV_RWSTRUCTURED_APPEND,
    D3D11SHADER_INPUT_UAV_RWSTRUCTURED_CONSUME,
    D3D11SHADER_INPUT_UAV_RWSTRUCTURED_WITH_COUNTER,
};
DWord D3D11ShaderInputTypeToD3D11[D3D11SHADER_INPUT_COUNT] = {
    D3D_SIT_CBUFFER,
    D3D_SIT_TBUFFER,
    D3D_SIT_TEXTURE,
    D3D_SIT_SAMPLER,
    D3D_SIT_BYTEADDRESS,
    D3D_SIT_STRUCTURED,
    D3D_SIT_UAV_RWBYTEADDRESS,
    D3D_SIT_UAV_RWSTRUCTURED,
    D3D_SIT_UAV_APPEND_STRUCTURED,
    D3D_SIT_UAV_CONSUME_STRUCTURED,
    D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER,
    D3D_SIT_UAV_RWTYPED
};

Byte D3D11ShaderInputFlagsFromD3D11[D3D11SHADER_INPUTFLAG_COUNT] = {
    0, // D3D11SHADER_INPUTFLAG_USERPACKED
    1, // D3D11SHADER_INPUTFLAG_COMPARISON_SAMPLER
    2, // D3D11SHADER_INPUTFLAG_TEXTURE_COMPONENT_0
    3, // D3D11SHADER_INPUTFLAG_TEXTURE_COMPONENT_1
    4  // D3D11SHADER_INPUTFLAG_UNUSED
};
Byte D3D11ShaderInputFlagsToD3D11[D3D11SHADER_INPUTFLAG_COUNT] = {
    0, // D3D_SIF_USERPACKED
    1, // D3D_SIF_COMPARISON_SAMPLER
    2, // D3D_SIF_TEXTURE_COMPONENT_0
    3, // D3D_SIF_TEXTURE_COMPONENT_1
    4  // D3D_SIF_UNUSED
};

D3D11ShaderReturnType D3D11ShaderReturnTypeFromD3D11[D3D11SHADER_RETURN_COUNT + 1] = {
    (D3D11ShaderReturnType)0, // INVALID !!!
    D3D11SHADER_RETURN_UNORM,
    D3D11SHADER_RETURN_SNORM,
    D3D11SHADER_RETURN_SINT,
    D3D11SHADER_RETURN_UINT,
    D3D11SHADER_RETURN_FLOAT,
    D3D11SHADER_RETURN_MIXED,
    D3D11SHADER_RETURN_DOUBLE,
    D3D11SHADER_RETURN_CONTINUED
};
DWord D3D11ShaderReturnTypeToD3D11[D3D11SHADER_RETURN_COUNT] = {
    D3D_RETURN_TYPE_UNORM,
    D3D_RETURN_TYPE_SNORM,
    D3D_RETURN_TYPE_UINT,
    D3D_RETURN_TYPE_SINT,
    D3D_RETURN_TYPE_FLOAT,
    D3D_RETURN_TYPE_DOUBLE,
    D3D_RETURN_TYPE_MIXED,
    D3D_RETURN_TYPE_CONTINUED
};

Void D3D11ShaderBindingDesc::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_SHADER_INPUT_BIND_DESC * pDesc = (const D3D11_SHADER_INPUT_BIND_DESC *)pD3D11Desc;

    strName = pDesc->Name;

    iType = D3D11ShaderInputTypeFromD3D11[pDesc->Type];
    iFlags = _D3D11ConvertFlags32( D3D11ShaderInputFlagsFromD3D11, pDesc->uFlags );

    iBindPoint = pDesc->BindPoint;
    iBindCount = pDesc->BindCount;

    iReturnType = D3D11ShaderReturnTypeFromD3D11[pDesc->ReturnType];
    iViewDimension = D3D11ShaderViewDimensionFromD3D11[pDesc->Dimension];
    iSampleCount = pDesc->NumSamples;
}

D3D11ShaderConstantBufferType D3D11ShaderConstantBufferTypeFromD3D11[D3D11SHADER_CONSTANTBUFFER_COUNT] = {
    D3D11SHADER_CONSTANTBUFFER_CBUFFER,
    D3D11SHADER_CONSTANTBUFFER_TBUFFER,
    D3D11SHADER_CONSTANTBUFFER_INTERFACE_POINTERS,
    D3D11SHADER_CONSTANTBUFFER_RESOURCE_BIND_INFO
};
DWord D3D11ShaderConstantBufferTypeToD3D11[D3D11SHADER_CONSTANTBUFFER_COUNT] = {
    D3D_CT_CBUFFER,
    D3D_CT_TBUFFER,
    D3D_CT_INTERFACE_POINTERS,
    D3D_CT_RESOURCE_BIND_INFO
};

Byte D3D11ShaderConstantBufferFlagsFromD3D11[D3D11SHADER_CONSTANTBUFFERFLAG_COUNT] = {
    0, // D3D11SHADER_CONSTANTBUFFERFLAG_USERPACKED
};
Byte D3D11ShaderConstantBufferFlagsToD3D11[D3D11SHADER_CONSTANTBUFFERFLAG_COUNT] = {
    0, // D3D_CBF_USERPACKED
};

Void D3D11ShaderConstantBufferDesc::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_SHADER_BUFFER_DESC * pDesc = (const D3D11_SHADER_BUFFER_DESC *)pD3D11Desc;

    strName = pDesc->Name;
    
    iType = D3D11ShaderConstantBufferTypeFromD3D11[pDesc->Type];

    iByteSize = pDesc->Size;
    iVariableCount = pDesc->Variables;

    iFlags = _D3D11ConvertFlags32( D3D11ShaderConstantBufferFlagsFromD3D11, pDesc->uFlags );
}

D3D11ShaderVariableClass D3D11ShaderVariableClassFromD3D11[D3D11SHADER_VARIABLECLASS_COUNT] = {
    D3D11SHADER_VARIABLECLASS_SCALAR,
    D3D11SHADER_VARIABLECLASS_VECTOR,
    D3D11SHADER_VARIABLECLASS_MATRIX_ROWS,
    D3D11SHADER_VARIABLECLASS_MATRIX_COLUMNS,
    D3D11SHADER_VARIABLECLASS_OBJECT,
    D3D11SHADER_VARIABLECLASS_STRUCT,
    D3D11SHADER_VARIABLECLASS_INTERFACE_CLASS,
    D3D11SHADER_VARIABLECLASS_INTERFACE_POINTER
};
DWord D3D11ShaderVariableClassToD3D11[D3D11SHADER_VARIABLECLASS_COUNT] = {
    D3D_SVC_SCALAR,
    D3D_SVC_VECTOR,
    D3D_SVC_MATRIX_ROWS,
    D3D_SVC_MATRIX_COLUMNS,
    D3D_SVC_STRUCT,
    D3D_SVC_OBJECT,
    D3D_SVC_INTERFACE_CLASS,
    D3D_SVC_INTERFACE_POINTER
};

D3D11ShaderVariableType D3D11ShaderVariableTypeFromD3D11[D3D11SHADER_VARIABLE_COUNT] = {
    D3D11SHADER_VARIABLE_VOID,
    D3D11SHADER_VARIABLE_BOOL,
    D3D11SHADER_VARIABLE_INT,
    D3D11SHADER_VARIABLE_FLOAT,
    D3D11SHADER_VARIABLE_STRING,
    D3D11SHADER_VARIABLE_TEXTURE,
    D3D11SHADER_VARIABLE_TEXTURE1D,
    D3D11SHADER_VARIABLE_TEXTURE2D,
    D3D11SHADER_VARIABLE_TEXTURE3D,
    D3D11SHADER_VARIABLE_TEXTURECUBE,
    D3D11SHADER_VARIABLE_SAMPLER,
    D3D11SHADER_VARIABLE_SAMPLER1D,
    D3D11SHADER_VARIABLE_SAMPLER2D,
    D3D11SHADER_VARIABLE_SAMPLER3D,
    D3D11SHADER_VARIABLE_SAMPLERCUBE,
    D3D11SHADER_VARIABLE_PIXELSHADER,
    D3D11SHADER_VARIABLE_VERTEXSHADER,
    D3D11SHADER_VARIABLE_PIXELFRAGMENT,
    D3D11SHADER_VARIABLE_VERTEXFRAGMENT,
    D3D11SHADER_VARIABLE_UINT,
    D3D11SHADER_VARIABLE_BYTE,
    D3D11SHADER_VARIABLE_GEOMETRYSHADER,
    D3D11SHADER_VARIABLE_RASTERIZER,
    D3D11SHADER_VARIABLE_DEPTHSTENCIL,
    D3D11SHADER_VARIABLE_BLEND,
    D3D11SHADER_VARIABLE_BUFFER,
    D3D11SHADER_VARIABLE_CBUFFER,
    D3D11SHADER_VARIABLE_TBUFFER,
    D3D11SHADER_VARIABLE_TEXTURE1DARRAY,
    D3D11SHADER_VARIABLE_TEXTURE2DARRAY,
    D3D11SHADER_VARIABLE_RENDERTARGETVIEW,
    D3D11SHADER_VARIABLE_DEPTHSTENCILVIEW,
    D3D11SHADER_VARIABLE_TEXTURE2DMS,
    D3D11SHADER_VARIABLE_TEXTURE2DMSARRAY,
    D3D11SHADER_VARIABLE_TEXTURECUBEARRAY,
    D3D11SHADER_VARIABLE_HULLSHADER,
    D3D11SHADER_VARIABLE_DOMAINSHADER,
    D3D11SHADER_VARIABLE_INTERFACEPOINTER,
    D3D11SHADER_VARIABLE_COMPUTESHADER,
    D3D11SHADER_VARIABLE_DOUBLE,
    D3D11SHADER_VARIABLE_RWTEXTURE1D,
    D3D11SHADER_VARIABLE_RWTEXTURE1DARRAY,
    D3D11SHADER_VARIABLE_RWTEXTURE2D,
    D3D11SHADER_VARIABLE_RWTEXTURE2DARRAY,
    D3D11SHADER_VARIABLE_RWTEXTURE3D,
    D3D11SHADER_VARIABLE_RWBUFFER,
    D3D11SHADER_VARIABLE_RAWBUFFER,
    D3D11SHADER_VARIABLE_RWRAWBUFFER,
    D3D11SHADER_VARIABLE_STRUCTUREDBUFFER,
    D3D11SHADER_VARIABLE_RWSTRUCTUREDBUFFER,
    D3D11SHADER_VARIABLE_STRUCTUREDBUFFER_APPEND,
    D3D11SHADER_VARIABLE_STRUCTUREDBUFFER_CONSUME,
    D3D11SHADER_VARIABLE_FLOAT8,
    D3D11SHADER_VARIABLE_FLOAT10,
    D3D11SHADER_VARIABLE_FLOAT16,
    D3D11SHADER_VARIABLE_INT12,
    D3D11SHADER_VARIABLE_SHORT,
    D3D11SHADER_VARIABLE_USHORT
};
DWord D3D11ShaderVariableTypeToD3D11[D3D11SHADER_VARIABLE_COUNT] = {
    D3D_SVT_VOID,
    D3D_SVT_BOOL,
    D3D_SVT_UINT8,
    D3D_SVT_MIN16INT,
    D3D_SVT_MIN16UINT,
    D3D_SVT_INT,
    D3D_SVT_UINT,
    D3D_SVT_FLOAT,
    D3D_SVT_DOUBLE,
    D3D_SVT_MIN12INT,
    D3D_SVT_MIN8FLOAT,
    D3D_SVT_MIN10FLOAT,
    D3D_SVT_MIN16FLOAT,
    D3D_SVT_INTERFACE_POINTER,
    D3D_SVT_STRING,
    D3D_SVT_BUFFER,
    D3D_SVT_RWBUFFER,
    D3D_SVT_CBUFFER,
    D3D_SVT_TBUFFER,
    D3D_SVT_BYTEADDRESS_BUFFER,
    D3D_SVT_RWBYTEADDRESS_BUFFER,
    D3D_SVT_STRUCTURED_BUFFER,
    D3D_SVT_RWSTRUCTURED_BUFFER,
    D3D_SVT_APPEND_STRUCTURED_BUFFER,
    D3D_SVT_CONSUME_STRUCTURED_BUFFER,
    D3D_SVT_TEXTURE,
    D3D_SVT_TEXTURE1D,
    D3D_SVT_TEXTURE1DARRAY,
    D3D_SVT_RWTEXTURE1D,
    D3D_SVT_RWTEXTURE1DARRAY,
    D3D_SVT_TEXTURE2D,
    D3D_SVT_TEXTURE2DARRAY,
    D3D_SVT_RWTEXTURE2D,
    D3D_SVT_RWTEXTURE2DARRAY,
    D3D_SVT_TEXTURE2DMS,
    D3D_SVT_TEXTURE2DMSARRAY,
    D3D_SVT_TEXTURE3D,
    D3D_SVT_RWTEXTURE3D,
    D3D_SVT_TEXTURECUBE,
    D3D_SVT_TEXTURECUBEARRAY,
    D3D_SVT_SAMPLER,
    D3D_SVT_SAMPLER1D,
    D3D_SVT_SAMPLER2D,
    D3D_SVT_SAMPLER3D,
    D3D_SVT_SAMPLERCUBE,
    D3D_SVT_VERTEXSHADER,
    D3D_SVT_GEOMETRYSHADER,
    D3D_SVT_PIXELSHADER,
    D3D_SVT_HULLSHADER,
    D3D_SVT_DOMAINSHADER,
    D3D_SVT_COMPUTESHADER,
    D3D_SVT_VERTEXFRAGMENT,
    D3D_SVT_PIXELFRAGMENT,
    D3D_SVT_RASTERIZER,
    D3D_SVT_DEPTHSTENCIL,
    D3D_SVT_BLEND,
    D3D_SVT_RENDERTARGETVIEW,
    D3D_SVT_DEPTHSTENCILVIEW
};

Void D3D11ShaderTypeDesc::ConvertFrom( const Void * pD3D11Desc, UInt iD3D11InterfaceCount )
{
    const D3D11_SHADER_TYPE_DESC * pDesc = (const D3D11_SHADER_TYPE_DESC *)pD3D11Desc;

    strName = pDesc->Name;

    iClass = D3D11ShaderVariableClassFromD3D11[pDesc->Class];
    iType = D3D11ShaderVariableTypeFromD3D11[pDesc->Type];

    iInterfaceCount = iD3D11InterfaceCount;
    iMemberCount = pDesc->Members;
    iElementCount = pDesc->Elements;

    iColumnCount = pDesc->Columns;
    iRowCount = pDesc->Rows;

    iOffset = pDesc->Offset;
}

Byte D3D11ShaderVariableFlagsFromD3D11[D3D11SHADER_VARIABLEFLAG_COUNT] = {
    0, // D3D11SHADER_VARIABLEFLAG_USERPACKED
    1, // D3D11SHADER_VARIABLEFLAG_USED
    2, // D3D11SHADER_VARIABLEFLAG_INTERFACE_POINTER
    3  // D3D11SHADER_VARIABLEFLAG_INTERFACE_PARAMETER
};
Byte D3D11ShaderVariableFlagsToD3D11[D3D11SHADER_VARIABLEFLAG_COUNT] = {
    0, // D3D_SVF_USERPACKED
    1, // D3D_SVF_USED
    2, // D3D_SVF_INTERFACE_POINTER
    3  // D3D_SVF_INTERFACE_PARAMETER
};

Void D3D11ShaderVariableDesc::ConvertFrom( const Void * pD3D11Desc )
{
    const D3D11_SHADER_VARIABLE_DESC * pDesc = (const D3D11_SHADER_VARIABLE_DESC *)pD3D11Desc;

    strName = pDesc->Name;

    iStartOffset = pDesc->StartOffset;
    iByteSize = pDesc->Size;

    iStartTextureSlot = pDesc->StartTexture;
    iTextureSlotCount = pDesc->TextureSize;

    iStartSamplerSlot = pDesc->StartSampler;
    iSamplerSlotCount = pDesc->SamplerSize;

    iFlags = _D3D11ConvertFlags32( D3D11ShaderVariableFlagsFromD3D11, pDesc->uFlags );

    pDefaultValue = pDesc->DefaultValue;
}

