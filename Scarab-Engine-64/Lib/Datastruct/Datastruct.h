/////////////////////////////////////////////////////////////////////////////////
// File : Lib/Datastruct/Datastruct.h
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : An abstraction layer for data-containers management.
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Header prelude
#ifndef SCARAB_LIB_DATASTRUCT_DATASTRUCT_H
#define SCARAB_LIB_DATASTRUCT_DATASTRUCT_H

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "../Memory/MemoryManager.h"

/////////////////////////////////////////////////////////////////////////////////
// Constants definitions

///////////////////////////////////////////////////////////////////////////////
// The Datastruct class
class Datastruct
{
public:
    Datastruct();
    virtual ~Datastruct();

    // Fixed memory context
    inline virtual Void UseMemory( MemoryContextID iMemoryContextID, MemoryAllocatorID iAllocatorID );

    // Common minimal methods
    virtual Bool IsCreated() const = 0;
    virtual Void Create() = 0;
    virtual Void Destroy() = 0;

    inline Bool IsEmpty() const;
    virtual SizeT MemorySize() const = 0;
    virtual UInt Count() const = 0;

    virtual Void Clear() = 0;

protected:
    // Fixed memory context
    MemoryContextID m_iMemoryContextID;
    MemoryAllocatorID m_iAllocatorID;
};

/////////////////////////////////////////////////////////////////////////////////
// Backward Includes (Inlines & Templates)
#include "Datastruct.inl"

/////////////////////////////////////////////////////////////////////////////////
// Header end
#endif // SCARAB_LIB_DATASTRUCT_DATASTRUCT_H
