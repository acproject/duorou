#ifndef DUOROU_MEDIA_MACOS_SPEECH_RECOGNITION_H
#define DUOROU_MEDIA_MACOS_SPEECH_RECOGNITION_H

#include <string>

namespace duorou {
namespace media {

std::string macos_transcribe_wav(const std::string &wav_path,
                                 std::string *error);

} // namespace media
} // namespace duorou

#endif // DUOROU_MEDIA_MACOS_SPEECH_RECOGNITION_H
