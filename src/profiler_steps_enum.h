#pragma once

enum Solver_Execution_Steps : unsigned {
	Cells_Init, Sort, Cols_Init, Cols_Resolve, Move,
	ALL_
};

inline Solver_Execution_Steps& operator++(Solver_Execution_Steps& orig)
{
	orig = static_cast<Solver_Execution_Steps>(orig + 1); // static_cast required because enum + int -> int
	return orig;
}

inline std::ostream& operator<<(std::ostream& os, const Solver_Execution_Steps& e)
{
	switch (e)
	{
	case Cells_Init:	os << "Cells_Init"; break;
	case Sort:			os << "Sort"; break;
	case Cols_Init:		os << "Cols_Init"; break;
	case Cols_Resolve:	os << "Cols_Resolve"; break;
	case Move:			os << "Move"; break;
	case ALL_: break;
	}
	return os;
}