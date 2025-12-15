#ifndef DUOROU_MEDIA_MEDIA_FILE_DECODER_H
#define DUOROU_MEDIA_MEDIA_FILE_DECODER_H

#ifdef __cplusplus

#include <string>
#include <vector>

#include "audio_capture.h"
#include "video_frame.h"

namespace duorou {
namespace media {

struct AudioFileDecodeOptions {
  int target_sample_rate;
  int target_channels;
};

struct VideoFileDecodeOptions {
  double frame_interval_seconds;
  int target_width;
  int target_height;
};

bool decode_audio_file(const std::string &path, AudioFrame &out_frame,
                       const AudioFileDecodeOptions &options);

bool decode_video_file(const std::string &path,
                       std::vector<VideoFrame> &out_frames,
                       const VideoFileDecodeOptions &options);

} // namespace media
} // namespace duorou

#endif // __cplusplus

#endif // DUOROU_MEDIA_MEDIA_FILE_DECODER_H

