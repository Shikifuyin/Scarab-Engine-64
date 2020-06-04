/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/DirectX11/Direct3D/D3D11Window.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Platform-dependant window implementation.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// D3D11WindowCallbacks implementation
inline Void D3D11WindowCallbacks::SetUserData( Void * pUserData ) {
    m_pUserData = pUserData;
}

/////////////////////////////////////////////////////////////////////////////////
// D3D11Window implementation
inline Bool D3D11Window::IsActive() const {
    return m_bActive;
}
inline Bool D3D11Window::IsMinimized() const {
    return m_bMinimized;
}
inline Bool D3D11Window::IsMaximized() const {
    return m_bMaximized;
}
inline Bool D3D11Window::IsInSizeMove() const {
    return m_bIsInSizeMove;
}

inline const GChar * D3D11Window::GetTitle() const {
    return m_strTitle;
}

inline UInt D3D11Window::GetAdapterCount() const {
    return m_iAdapterCount;
}
inline const D3D11AdapterDesc * D3D11Window::GetAdapterDesc( UInt iAdapter ) const {
    DebugAssert( iAdapter < m_iAdapterCount );
    return ( m_arrAdapters + iAdapter );
}

inline UInt D3D11Window::GetCurrentAdapter() const {
    return m_iAdapter;
}
inline const D3D11AdapterDesc * D3D11Window::GetCurrentAdapterDesc() const {
    return &m_hAdapterDesc;
}

inline UInt D3D11Window::GetOutputCount( UInt iAdapter ) const {
    if ( iAdapter == INVALID_OFFSET )
        iAdapter = m_iAdapter;
    DebugAssert( iAdapter < m_iAdapterCount );
    return m_arrOutputCounts[iAdapter];
}
inline const D3D11OutputDesc * D3D11Window::GetOutputDesc( UInt iOutput, UInt iAdapter ) const {
    if ( iAdapter == INVALID_OFFSET )
        iAdapter = m_iAdapter;
    DebugAssert( iAdapter < m_iAdapterCount );
    UInt iOutputIndex = ( iAdapter * D3D11WINDOW_MAX_OUTPUTS ) + iOutput;
    DebugAssert( iOutputIndex < m_arrOutputCounts[iAdapterIndex] );
    return ( m_arrOutputs + iOutputIndex );
}

inline UInt D3D11Window::GetCurrentOutput() const {
    return m_iOutput;
}
inline const D3D11OutputDesc * D3D11Window::GetCurrentOutputDesc() const {
    return &m_hOutputDesc;
}

inline UInt D3D11Window::GetDisplayModeCount( UInt iOutput, UInt iAdapter ) const {
    if ( iAdapter == INVALID_OFFSET )
        iAdapter = m_iAdapter; 
    if ( iOutput == INVALID_OFFSET )
        iOutput = m_iOutput;
    DebugAssert( iAdapter < m_iAdapterCount );
    UInt iOutputIndex = ( iAdapter * D3D11WINDOW_MAX_OUTPUTS ) + iOutput;
    DebugAssert( iOutputIndex < m_arrOutputCounts[iAdapterIndex] );
    return m_arrDisplayModeCounts[iOutputIndex];
}
inline const D3D11DisplayModeDesc * D3D11Window::GetDisplayModeDesc( UInt iDisplayMode, UInt iOutput, UInt iAdapter ) const {
    if ( iAdapter == INVALID_OFFSET )
        iAdapter = m_iAdapter;
    if ( iOutput == INVALID_OFFSET )
        iOutput = m_iOutput;
    DebugAssert( iAdapter < m_iAdapterCount );
    UInt iOutputIndex = ( iAdapter * D3D11WINDOW_MAX_OUTPUTS ) + iOutput;
    DebugAssert( iOutputIndex < m_arrOutputCounts[iAdapterIndex] );
    UInt iDisplayModeIndex = ( iOutputIndex * D3D11WINDOW_MAX_DISPLAYMODES ) + iDisplayMode;
    DebugAssert( iDisplayModeIndex < m_arrDisplayModeCounts[iOutputIndex] );
    return ( m_arrDisplayModes + iDisplayModeIndex );
}

inline UInt D3D11Window::GetCurrentDisplayMode() const {
    return m_iDisplayMode;
}
inline const D3D11DisplayModeDesc * D3D11Window::GetCurrentDisplayModeDesc() const {
    return &m_hDisplayModeDesc;
}

inline Bool D3D11Window::IsFullScreenWindowed() const {
    return m_bFullScreenWindowed;
}

