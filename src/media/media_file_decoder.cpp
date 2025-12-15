#include "media_file_decoder.h"

#include <cstring>
#include <iostream>

#ifdef HAVE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/imgutils.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
#endif

namespace duorou {
namespace media {

namespace {

struct FFmpegInitializer {
  FFmpegInitializer() {
#ifdef HAVE_FFMPEG
    av_log_set_level(AV_LOG_ERROR);
#endif
  }
};

FFmpegInitializer g_ffmpeg_initializer;

#ifdef HAVE_FFMPEG

bool open_input(const std::string &path, AVFormatContext *&fmt_ctx) {
  fmt_ctx = nullptr;
  if (avformat_open_input(&fmt_ctx, path.c_str(), nullptr, nullptr) < 0) {
    std::cout << "Failed to open media file: " << path << std::endl;
    return false;
  }
  if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
    std::cout << "Failed to find stream info for: " << path << std::endl;
    avformat_close_input(&fmt_ctx);
    return false;
  }
  return true;
}

int find_stream_index(AVFormatContext *fmt_ctx, AVMediaType type) {
  int best_index = av_find_best_stream(fmt_ctx, type, -1, -1, nullptr, 0);
  if (best_index < 0) {
    std::cout << "No suitable stream found for media type "
              << static_cast<int>(type) << std::endl;
  }
  return best_index;
}

bool open_codec_context(AVCodecContext *&codec_ctx, AVFormatContext *fmt_ctx,
                        int stream_index) {
  codec_ctx = nullptr;
  AVStream *stream = fmt_ctx->streams[stream_index];
  const AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
  if (!codec) {
    std::cout << "Failed to find decoder" << std::endl;
    return false;
  }
  codec_ctx = avcodec_alloc_context3(codec);
  if (!codec_ctx) {
    std::cout << "Failed to allocate codec context" << std::endl;
    return false;
  }
  if (avcodec_parameters_to_context(codec_ctx, stream->codecpar) < 0) {
    std::cout << "Failed to copy codec parameters to context" << std::endl;
    avcodec_free_context(&codec_ctx);
    return false;
  }
  if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
    std::cout << "Failed to open codec" << std::endl;
    avcodec_free_context(&codec_ctx);
    return false;
  }
  return true;
}

#endif

}  // namespace

bool decode_audio_file(const std::string &path, AudioFrame &out_frame,
                       const AudioFileDecodeOptions &options) {
#ifndef HAVE_FFMPEG
  std::cout << "FFmpeg is not available, audio file decoding disabled"
            << std::endl;
  return false;
#else
  AVFormatContext *fmt_ctx = nullptr;
  if (!open_input(path, fmt_ctx)) {
    return false;
  }

  int audio_stream_index = find_stream_index(fmt_ctx, AVMEDIA_TYPE_AUDIO);
  if (audio_stream_index < 0) {
    avformat_close_input(&fmt_ctx);
    return false;
  }

  AVCodecContext *codec_ctx = nullptr;
  if (!open_codec_context(codec_ctx, fmt_ctx, audio_stream_index)) {
    avformat_close_input(&fmt_ctx);
    return false;
  }

  int src_sample_rate = codec_ctx->sample_rate;
  if (src_sample_rate <= 0) {
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    return false;
  }

  int src_channels = codec_ctx->ch_layout.nb_channels;
  if (src_channels <= 0) {
    src_channels = 1;
  }

  int target_sample_rate =
      options.target_sample_rate > 0 ? options.target_sample_rate
                                     : src_sample_rate;
  int target_channels =
      options.target_channels > 0 ? options.target_channels : src_channels;

  AVChannelLayout in_layout;
  if (av_channel_layout_copy(&in_layout, &codec_ctx->ch_layout) < 0) {
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    return false;
  }

  AVChannelLayout out_layout;
  av_channel_layout_default(&out_layout, target_channels);

  SwrContext *swr_ctx = nullptr;
  if (swr_alloc_set_opts2(&swr_ctx,
                          &out_layout, AV_SAMPLE_FMT_FLT, target_sample_rate,
                          &in_layout, codec_ctx->sample_fmt, src_sample_rate,
                          0, nullptr) < 0 ||
      !swr_ctx) {
    std::cout << "Failed to allocate audio resampler" << std::endl;
    av_channel_layout_uninit(&out_layout);
    av_channel_layout_uninit(&in_layout);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    return false;
  }

  if (swr_init(swr_ctx) < 0) {
    std::cout << "Failed to initialize audio resampler" << std::endl;
    swr_free(&swr_ctx);
    av_channel_layout_uninit(&out_layout);
    av_channel_layout_uninit(&in_layout);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    return false;
  }

  AVPacket *packet = av_packet_alloc();
  AVFrame *frame = av_frame_alloc();
  if (!packet || !frame) {
    if (packet) {
      av_packet_free(&packet);
    }
    if (frame) {
      av_frame_free(&frame);
    }
    swr_free(&swr_ctx);
    av_channel_layout_uninit(&out_layout);
    av_channel_layout_uninit(&in_layout);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    return false;
  }

  std::vector<float> samples;

  while (av_read_frame(fmt_ctx, packet) >= 0) {
    if (packet->stream_index != audio_stream_index) {
      av_packet_unref(packet);
      continue;
    }

    if (avcodec_send_packet(codec_ctx, packet) < 0) {
      av_packet_unref(packet);
      break;
    }

    av_packet_unref(packet);

    while (true) {
      int ret = avcodec_receive_frame(codec_ctx, frame);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        break;
      }
      if (ret < 0) {
        break;
      }

      int dst_nb_channels = target_channels;
      int dst_nb_samples =
          av_rescale_rnd(swr_get_delay(swr_ctx, src_sample_rate) +
                             frame->nb_samples,
                         target_sample_rate, src_sample_rate,
                         AV_ROUND_UP);

      if (dst_nb_samples <= 0) {
        continue;
      }

      std::vector<float> out_buffer(
          static_cast<size_t>(dst_nb_samples) *
          static_cast<size_t>(dst_nb_channels));
      uint8_t *out_data =
          reinterpret_cast<uint8_t *>(out_buffer.data());

      int converted = swr_convert(
          swr_ctx, &out_data, dst_nb_samples,
          const_cast<const uint8_t **>(frame->data), frame->nb_samples);

      if (converted < 0) {
        std::cout << "Error during audio resampling" << std::endl;
        break;
      }

      size_t valid_samples =
          static_cast<size_t>(converted) *
          static_cast<size_t>(dst_nb_channels);

      samples.insert(samples.end(),
                     out_buffer.begin(),
                     out_buffer.begin() + valid_samples);
    }
  }

  av_frame_free(&frame);
  av_packet_free(&packet);
  swr_free(&swr_ctx);
  av_channel_layout_uninit(&out_layout);
  av_channel_layout_uninit(&in_layout);
  avcodec_free_context(&codec_ctx);
  avformat_close_input(&fmt_ctx);

  if (samples.empty()) {
    std::cout << "No audio data decoded from file: " << path << std::endl;
    return false;
  }

  out_frame.data = std::move(samples);
  out_frame.sample_rate = target_sample_rate;
  out_frame.channels = target_channels;
  out_frame.frame_count =
      static_cast<int>(out_frame.data.size() /
                       static_cast<size_t>(target_channels));
  out_frame.timestamp = 0.0;

  return true;
#endif
}

bool decode_video_file(const std::string &path,
                       std::vector<VideoFrame> &out_frames,
                       const VideoFileDecodeOptions &options) {
#ifndef HAVE_FFMPEG
  std::cout << "FFmpeg is not available, video file decoding disabled"
            << std::endl;
  return false;
#else
  AVFormatContext *fmt_ctx = nullptr;
  if (!open_input(path, fmt_ctx)) {
    return false;
  }

  int video_stream_index =
      find_stream_index(fmt_ctx, AVMEDIA_TYPE_VIDEO);
  if (video_stream_index < 0) {
    avformat_close_input(&fmt_ctx);
    return false;
  }

  AVCodecContext *codec_ctx = nullptr;
  if (!open_codec_context(codec_ctx, fmt_ctx, video_stream_index)) {
    avformat_close_input(&fmt_ctx);
    return false;
  }

  AVStream *video_stream = fmt_ctx->streams[video_stream_index];

  int target_width =
      options.target_width > 0 ? options.target_width : codec_ctx->width;
  int target_height =
      options.target_height > 0 ? options.target_height : codec_ctx->height;

  SwsContext *sws_ctx = sws_getContext(
      codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
      target_width, target_height, AV_PIX_FMT_RGB24,
      SWS_BILINEAR, nullptr, nullptr, nullptr);

  if (!sws_ctx) {
    std::cout << "Failed to create video scaler" << std::endl;
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    return false;
  }

  AVPacket *packet = av_packet_alloc();
  AVFrame *frame = av_frame_alloc();
  AVFrame *rgb_frame = av_frame_alloc();

  if (!packet || !frame || !rgb_frame) {
    if (packet) {
      av_packet_free(&packet);
    }
    if (frame) {
      av_frame_free(&frame);
    }
    if (rgb_frame) {
      av_frame_free(&rgb_frame);
    }
    sws_freeContext(sws_ctx);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    return false;
  }

  rgb_frame->format = AV_PIX_FMT_RGB24;
  rgb_frame->width = target_width;
  rgb_frame->height = target_height;

  int rgb_buffer_size =
      av_image_get_buffer_size(AV_PIX_FMT_RGB24, target_width,
                               target_height, 1);
  std::vector<uint8_t> rgb_buffer(
      static_cast<size_t>(rgb_buffer_size));

  av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize,
                       rgb_buffer.data(), AV_PIX_FMT_RGB24,
                       target_width, target_height, 1);

  double next_capture_time = 0.0;
  double interval =
      options.frame_interval_seconds > 0.0
          ? options.frame_interval_seconds
          : 1.0;

  while (av_read_frame(fmt_ctx, packet) >= 0) {
    if (packet->stream_index != video_stream_index) {
      av_packet_unref(packet);
      continue;
    }

    if (avcodec_send_packet(codec_ctx, packet) < 0) {
      av_packet_unref(packet);
      break;
    }

    av_packet_unref(packet);

    while (true) {
      int ret = avcodec_receive_frame(codec_ctx, frame);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        break;
      }
      if (ret < 0) {
        break;
      }

      double pts = 0.0;
      if (frame->best_effort_timestamp != AV_NOPTS_VALUE) {
        pts = frame->best_effort_timestamp *
              av_q2d(video_stream->time_base);
      }

      if (pts < next_capture_time) {
        continue;
      }

      sws_scale(sws_ctx, frame->data, frame->linesize, 0,
                codec_ctx->height, rgb_frame->data, rgb_frame->linesize);

      VideoFrame out;
      out.width = target_width;
      out.height = target_height;
      out.channels = 3;
      out.timestamp = pts;

      size_t data_size =
          static_cast<size_t>(target_width) *
          static_cast<size_t>(target_height) * 3;
      out.data.resize(data_size);

      for (int y = 0; y < target_height; ++y) {
        uint8_t *src_row =
            rgb_frame->data[0] +
            static_cast<size_t>(y) *
                static_cast<size_t>(rgb_frame->linesize[0]);
        uint8_t *dst_row =
            out.data.data() +
            static_cast<size_t>(y) *
                static_cast<size_t>(target_width) * 3;
        std::memcpy(dst_row, src_row,
                    static_cast<size_t>(target_width) * 3);
      }

      out_frames.push_back(std::move(out));
      next_capture_time += interval;
    }
  }

  av_frame_free(&rgb_frame);
  av_frame_free(&frame);
  av_packet_free(&packet);
  sws_freeContext(sws_ctx);
  avcodec_free_context(&codec_ctx);
  avformat_close_input(&fmt_ctx);

  if (out_frames.empty()) {
    std::cout << "No video frames decoded from file: " << path
              << std::endl;
    return false;
  }

  return true;
#endif
}

}  // namespace media
}  // namespace duorou

