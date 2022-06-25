#include <iostream>
#include <fstream>
#include <sstream>

namespace pim {
std::string ToBinary(int n);
std::string FillZero(std::string s, size_t n);

void GWrite(std::ostream& OS, char* buf, int r);
void GAct(std::ostream& OS, char* buf, int k, int r, int j, int num_act);
void Comp(std::ostream& OS, char* buf, int k, int r, int j, int h);

void ReadRes(std::ostream& OS);

void OutputNewtonTrace(std::ostream& OS, std::string kernel_name, int64_t row, int64_t col);
void OutputNewtonTraceV2(std::ostream& OS, std::string kernel_name, int64_t row, int64_t col, int64_t stride);
}
