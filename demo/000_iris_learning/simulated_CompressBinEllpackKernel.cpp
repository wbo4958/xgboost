#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <cstring>

namespace detail
{
static const int kPadding = 4;
// The number of bits required to represent a given unsigned range
static size_t SymbolBits(size_t num_symbols)
{
  auto bits = std::ceil(std::log2(num_symbols));
  return std::max(static_cast<size_t>(bits), size_t(1));
}
} // namespace detail

using CompressedByteT = unsigned char;

void AtomicOrByte(unsigned int *buffer, size_t ibyte, unsigned char b)
{
  size_t index = ibyte / sizeof(unsigned int);
  int v = (unsigned int)b << (ibyte % (sizeof(unsigned int)) * 8);
  int pre = buffer[index];
  buffer[index] = buffer[index] | v;
  printf("%d prev:%d now:%d\n", index, pre, buffer[index]);
  //atomicOr(&buffer[ibyte / sizeof(unsigned int)], (unsigned int)b << (ibyte % (sizeof(unsigned int)) * 8));
}

void AtomicWriteSymbol(CompressedByteT *buffer, uint64_t symbol, size_t offset)
{
  int symbol_bits_ = 3;
  size_t ibit_start = offset * symbol_bits_;
  size_t ibit_end = (offset + 1) * symbol_bits_ - 1;
  size_t ibyte_start = ibit_start / 8, ibyte_end = ibit_end / 8;

  symbol <<= 7 - ibit_end % 8;

  for (ptrdiff_t ibyte = ibyte_end; ibyte >= (ptrdiff_t)ibyte_start; --ibyte)
  {
    AtomicOrByte(reinterpret_cast<unsigned int *>(buffer + detail::kPadding),
                 ibyte, symbol & 0xff);
    symbol >>= 8;
  }
}

int upper_boundSQ(const float* s, int size, float val)
{
  for (int i = 0; i < size; i++) {
    if (val <= *(s + i)) {
      return i;
    }
  }
  return size-1;
}

static size_t CalculateBufferSize(size_t num_elements, size_t num_symbols)
{
  const int bits_per_byte = 8;
  size_t compressed_size = static_cast<size_t>(std::ceil(
      static_cast<double>(detail::SymbolBits(num_symbols) * num_elements) /
      bits_per_byte));
  return compressed_size + detail::kPadding;
}

// Bin each input data entry, store the bin indices in compressed form.
void CompressBinEllpackKernel(
    CompressedByteT * buffer,  // gidx_buffer
    const size_t * row_ptrs,   // row offset of input data
    const float * entries,     // One batch of input data
    const float * cuts,        // HistogramCuts::cut
    const uint32_t * cut_rows, // HistogramCuts::row_ptrs
    size_t base_row,                       // batch_row_begin
    size_t n_rows,
    size_t row_stride,
    unsigned int null_gidx_value)
{

  int ifeature = 0;
  for (size_t irow = 0; irow < n_rows; irow++)
  {
    int row_length = static_cast<int>(row_ptrs[irow + 1] - row_ptrs[irow]);
    unsigned int bin = null_gidx_value;
    if (ifeature < row_length)
    {
      float entry = entries[row_ptrs[irow] - row_ptrs[0] + ifeature];
      int feature = 0;
      float fvalue = entry;
      // {feature_cuts, ncuts} forms the array of cuts of `feature'.
      const float *feature_cuts = &cuts[cut_rows[feature]];
      int ncuts = cut_rows[feature + 1] - cut_rows[feature];
      // Assigning the bin in current entry.
      // S.t.: fvalue < feature_cuts[bin]
      bin = upper_boundSQ(feature_cuts, ncuts, fvalue);
      if (bin >= ncuts)
      {
        bin = ncuts - 1;
      }
      // Add the number of bins in previous features.
      bin += cut_rows[feature];
    }
    // Write to gidx buffer.
    int offset = (irow + base_row) * row_stride + ifeature;
    //printf("symbol:%d irow:%d ifeature:%d offset:%d\n", bin, irow, ifeature, offset);
    AtomicWriteSymbol(buffer, bin, (irow + base_row) * row_stride + ifeature);
  }
}

int main(void)
{
  int row_stride = 1;
  int num_rows = 150;
  size_t num_symbols = 7;

  size_t *offset = new size_t[151];
  for (size_t i = 0; i < 151; i++)
  {
    offset[i] = i;
  }

  float *entry = new float[150]{1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5, 1.7, 1.5, 1, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.5, 1.3, 1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7, 4.5, 4.9, 4, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4, 4.9, 4.7, 4.3, 4.4, 4.8, 5, 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4, 4.4, 4.6, 4, 3.3, 4.2, 4.2, 4.2, 4.3, 3, 4.1, 6, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5, 5, 5.1, 5.3, 5.5, 6.7, 6.9, 5, 5.7, 4.9, 6.7, 4.9, 5.7, 6, 4.8, 4.9, 5.6, 5.8, 6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.1, 5.9, 5.7, 5.2, 5, 5.2, 5.4, 5.1};

  float *cuts = new float[7]{1.25, 2.25, 3.7, 4.65, 5.2, 6.2, 13.8};

  //int x = upper_boundSQ(cuts, 7, 14.65);
  //printf("xx ----- %d\n", x);

  uint32_t *cut_rows = new uint32_t[2]{0, 7};

  size_t compressed_size_bytes = CalculateBufferSize(row_stride * num_rows, num_symbols);

  std::cout << "compressed_size_bytes:" << compressed_size_bytes << std::endl;

  CompressedByteT *gidx_buffer = new unsigned char[compressed_size_bytes];
  memset(gidx_buffer, 0, compressed_size_bytes);

  CompressBinEllpackKernel(gidx_buffer, offset, entry, cuts, cut_rows, 0, 150, 1, 7);
  for (int i = 0; i < compressed_size_bytes; i++) {
    printf("i:%d v:%d\n", i, gidx_buffer[i]);
  }
}
