/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/D3D11Mappings.inl
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
// General Definitions

/////////////////////////////////////////////////////////////////////////////////
// D3D11Window Definitions

/////////////////////////////////////////////////////////////////////////////////
// D3D11Renderer Definitions

/////////////////////////////////////////////////////////////////////////////////
// D3D11DeferredContext Definitions

/////////////////////////////////////////////////////////////////////////////////
// D3D11InputLayout Definitions
inline D3D11InputFieldType D3D11InputFieldTypeFromD3D11( DWord iDXGIFormat ) {
    return (D3D11InputFieldType)( PixelFormatFromDXGI[iDXGIFormat] );
}
inline DWord D3D11InputFieldTypeToD3D11( D3D11InputFieldType iInputFieldType ) {
    return PixelFormatToDXGI[(PixelFormat)iInputFieldType];
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11RenderState Definitions
inline D3D11SamplerFilterMode D3D11SamplerFilterModeFromD3D11( DWord iD3D11Filter ) {
    return (D3D11SamplerFilterMode)( _D3D11ConvertFlags32(D3D11SamplerFilterFlagsFromD3D11, iD3D11Filter) );
}
inline DWord D3D11SamplerFilterModeToD3D11( D3D11SamplerFilterMode iFilterMode ) {
    return _D3D11ConvertFlags32( D3D11SamplerFilterFlagsToD3D11, iFilterMode );
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11Asynchronous Definitions

/////////////////////////////////////////////////////////////////////////////////
// D3D11Resource Definitions
inline D3D11ResourcePriority D3D11ResourcePriorityFromD3D11( DWord iD3D11ResourcePriority ) {
    DWord iRes = 0;
    iD3D11ResourcePriority = ( iD3D11ResourcePriority >> 24 );
    if ( iD3D11ResourcePriority & 0x28 ) {
        iD3D11ResourcePriority &= 0xd7; // ~(0x28)
        ++iRes;
    }
    iRes += ( iD3D11ResourcePriority / 0x25 ) - 1;
    return (D3D11ResourcePriority)iRes;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11Buffer Definitions

/////////////////////////////////////////////////////////////////////////////////
// D3D11Texture Definitions

/////////////////////////////////////////////////////////////////////////////////
// D3D11ResourceView Definitions

/////////////////////////////////////////////////////////////////////////////////
// D3D11Shader Definitions



