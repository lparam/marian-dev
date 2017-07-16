#pragma once

namespace marian {

typedef uint8_t byte;

class MemoryPiece {
  private:
    byte* data_;
    size_t size_;

  public:
    MemoryPiece(byte* data, size_t size)
      : data_(data), size_(size) {}

    byte* data() const { return data_; }
    byte* data() { return data_; }
    size_t size() const { return size_; }

    void set(byte* data, size_t size) {
      data_ = data;
      size_ = size;
    }

    void setPtr(byte* data) {
      data_ = data;
    }

    friend std::ostream& operator<<(std::ostream& out, const MemoryPiece mp) {
      out << "MemoryPiece - ptr: " << std::hex << (size_t)mp.data()
        << std::dec << " size: " << mp.size();
      return out;
    }
};

}