// Alpacca — minimal SHA-256 (FIPS 180-4) for download integrity checks.
// MIT License. See LICENSE.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace alpacca {

class SHA256 {
  public:
    SHA256();
    void update(const void * data, size_t len);
    // Finalize and return lowercase hex digest. The object must not be reused.
    std::string hex_digest();

    // Hash an entire file; returns lowercase hex digest or "" on I/O error.
    static std::string file_hex(const std::string & path);

  private:
    void process_block(const uint8_t * block);

    uint32_t state_[8];
    uint64_t total_len_;
    uint8_t  buffer_[64];
    size_t   buffer_len_;
};

} // namespace alpacca
