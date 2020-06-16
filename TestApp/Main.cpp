/////////////////////////////////////////////////////////////////////////////////
// File : Main.cpp
/////////////////////////////////////////////////////////////////////////////////
// Version : 0.1
// Status : Alpha
/////////////////////////////////////////////////////////////////////////////////
// Description : Test Entry Point
/////////////////////////////////////////////////////////////////////////////////
// Part of Scarab-Engine, licensed under the
// Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
//   http://creativecommons.org/licenses/by-nc-nd/3.0/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Known Bugs : None
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Includes
#include "LIB/Math/Types/Matrix/Matrix4.h"

/////////////////////////////////////////////////////////////////////////////////
// Entry Point
int main()
{
	Matrix4 matL(
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16
	);

	Matrix4 matR(
		1, 1, 1, 1,
		10, 10, 10, 10,
		100, 100, 100, 100,
		1000, 1000, 1000, 1000
	);

	Matrix4 matProduct = matL * matR;

	return 0;
}

