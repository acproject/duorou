#ifndef DUOROU_MEDIA_VIDEO_FRAME_H
#define DUOROU_MEDIA_VIDEO_FRAME_H

#ifdef __cplusplus
#include <vector>
#include <cstdint>

namespace duorou {
namespace media {

struct VideoFrame {
    int width;
    int height;
    int channels;
    double timestamp;
    std::vector<uint8_t> data;
};

} // namespace media
} // namespace duorou

#endif // __cplusplus
#endif // DUOROU_MEDIA_VIDEO_FRAME_H