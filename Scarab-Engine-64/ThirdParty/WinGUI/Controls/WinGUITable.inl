/////////////////////////////////////////////////////////////////////////////////
// File : ThirdParty/WinGUI/Controls/WinGUITable.inl
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Windows GUI Control : Table (ListView)
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None.
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// WinGUITableModel implementation
inline const WinGUITableParameters * WinGUITableModel::GetCreationParameters() const {
	return &m_hCreationParameters;
}

/////////////////////////////////////////////////////////////////////////////////
// WinGUITable implementation
inline Bool WinGUITable::IsVirtual() const {
	return m_bVirtualTable;
}

inline Bool WinGUITable::HasBackBuffer() const {
	return m_bHasBackBuffer;
}

inline Bool WinGUITable::HasSharedImageLists() const {
	return m_bHasSharedImageLists;
}

inline WinGUITableViewMode WinGUITable::GetViewMode() const {
	return m_iViewMode;
}

inline Bool WinGUITable::IsGroupModeEnabled() const {
	return m_bGroupMode;
}

inline Bool WinGUITable::HasHeadersInAllViews() const {
	return m_bHasHeadersInAllViews;
}

inline Bool WinGUITable::HasColumnHeaders() const {
	return m_bHasColumnHeaders;
}

inline Bool WinGUITable::HasStaticColumnHeaders() const {
	return m_bHasStaticColumnHeaders;
}

inline Bool WinGUITable::HasDraggableColumnHeaders() const {
	return m_bHasDraggableColumnHeaders;
}

inline Bool WinGUITable::HasIconColumnOverflowButton() const {
	return m_bHasIconColumnOverflowButton;
}

inline Bool WinGUITable::HasCheckBoxes() const {
	return m_bHasCheckBoxes;
}

inline Bool WinGUITable::HasIconLabels() const {
	return m_bHasIconLabels;
}

inline Bool WinGUITable::HasEditableLabels() const {
	return m_bHasEditableLabels;
}

inline Bool WinGUITable::HasSubItemImages() const {
	return m_bHasSubItemImages;
}

inline Bool WinGUITable::HasSingleItemSelection() const {
	return m_bSingleItemSelection;
}

inline Bool WinGUITable::HasIconSimpleSelection() const {
	return m_bIconSimpleSelection;
}

inline Bool WinGUITable::IsAutoSorted() const {
	return ( m_bAutoSortAscending || m_bAutoSortDescending );
}
inline Bool WinGUITable::IsAutoSortedAscending() const {
	return m_bAutoSortAscending;
}
inline Bool WinGUITable::IsAutoSortedDescending() const {
	return m_bAutoSortDescending;
}

inline Bool WinGUITable::HasHotTracking() const {
	return ( m_bHasHotTrackingSingleClick || m_bHasHotTrackingDoubleClick );
}
inline Bool WinGUITable::IsHotTrackingSingleClick() const {
	return m_bHasHotTrackingSingleClick;
}
inline Bool WinGUITable::IsHotTrackingDoubleClick() const {
	return m_bHasHotTrackingDoubleClick;
}
inline Bool WinGUITable::HasHotTrackingSelection() const {
	return m_bHasHotTrackingSelection;
}

inline Bool WinGUITable::HasInfoTips() const {
	return m_bHasInfoTips;
}

inline Bool WinGUITable::IsEditingItemLabel( UInt * outItemIndex ) const {
	if ( outItemIndex != NULL )
		*outItemIndex = m_iEditLabelItemIndex;
	return ( m_hEditLabelHandle != NULL );
}
