/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/Resources/D3D11Resource.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Platform-dependant abstraction for GPU resources.
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

#undef DebugAssert

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "D3D11Resource.h"

/////////////////////////////////////////////////////////////////////////////////
// D3D11Resource implementation
D3D11Resource::D3D11Resource()
{
    m_pResource = NULL;

    m_iUsage = D3D11RESOURCE_USAGE_DEFAULT;
    m_iBinds = D3D11RESOURCE_BIND_NONE;
    m_iMiscFlags = 0;

    m_bLocked = false;

}
D3D11Resource::~D3D11Resource()
{
    // nothing to do
}

Void D3D11Resource::SetUsage( D3D11ResourceUsage iUsage )
{
    DebugAssert( !(IsCreated()) );

    m_iUsage = iUsage;
}

D3D11ResourcePriority D3D11Resource::GetEvictionPriority() const
{
    DebugAssert( IsCreated() );

    return D3D11ResourcePriorityFromD3D11( ((ID3D11Resource*)m_pResource)->GetEvictionPriority() );
}
Void D3D11Resource::SetEvictionPriority( D3D11ResourcePriority iPriority )
{
    DebugAssert( IsCreated() );

    ((ID3D11Resource*)m_pResource)->SetEvictionPriority( D3D11ResourcePriorityToD3D11[iPriority] );
}

/////////////////////////////////////////////////////////////////////////////////

UInt D3D11Resource::_GetCPUAccessFlags() const
{
    if ( m_iUsage == D3D11RESOURCE_USAGE_DYNAMIC )
        return D3D11_CPU_ACCESS_WRITE;

    if ( m_iUsage == D3D11RESOURCE_USAGE_STAGING )
        return ( D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE );

    return 0;
}

