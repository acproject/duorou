#include "macos_speech_recognition.h"

#import <Foundation/Foundation.h>
#import <Speech/Speech.h>

namespace duorou {
namespace media {

std::string macos_transcribe_wav(const std::string &wav_path,
                                 std::string *error) {
  @autoreleasepool {
    if (error) {
      error->clear();
    }

    if (wav_path.empty()) {
      if (error) {
        *error = "Empty wav path";
      }
      return {};
    }

    __block SFSpeechRecognizerAuthorizationStatus auth =
        SFSpeechRecognizerAuthorizationStatusNotDetermined;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [SFSpeechRecognizer requestAuthorization:^(SFSpeechRecognizerAuthorizationStatus status) {
      auth = status;
      dispatch_semaphore_signal(sem);
    }];
    dispatch_time_t wait_time =
        dispatch_time(DISPATCH_TIME_NOW, (int64_t)(5 * NSEC_PER_SEC));
    dispatch_semaphore_wait(sem, wait_time);

    if (auth != SFSpeechRecognizerAuthorizationStatusAuthorized) {
      if (error) {
        *error = "Speech recognition permission not granted";
      }
      return {};
    }

    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:wav_path.c_str()]];
    if (!url) {
      if (error) {
        *error = "Invalid wav url";
      }
      return {};
    }

    auto transcribe_with_locale_identifier =
        [&](NSString *locale_id, std::string *out_error) -> std::string {
      if (out_error) {
        out_error->clear();
      }

      NSLocale *locale = nil;
      if (locale_id && locale_id.length > 0) {
        locale = [NSLocale localeWithLocaleIdentifier:locale_id];
      }

      SFSpeechRecognizer *recognizer =
          locale ? [[SFSpeechRecognizer alloc] initWithLocale:locale]
                 : [[SFSpeechRecognizer alloc] init];
      if (!recognizer || !recognizer.isAvailable) {
        if (out_error) {
          *out_error = "Speech recognizer not available";
        }
        return {};
      }

      SFSpeechURLRecognitionRequest *request =
          [[SFSpeechURLRecognitionRequest alloc] initWithURL:url];
      request.shouldReportPartialResults = NO;
      request.taskHint = SFSpeechRecognitionTaskHintDictation;

      __block std::string out;
      __block std::string out_err;
      __block bool signaled = false;
      __block SFSpeechRecognitionTask *task = nil;
      dispatch_semaphore_t done = dispatch_semaphore_create(0);

      task = [recognizer recognitionTaskWithRequest:request
                                     resultHandler:^(
                                         SFSpeechRecognitionResult *_Nullable result,
                                         NSError *_Nullable err) {
                                       if (signaled) {
                                         return;
                                       }
                                       if (err) {
                                         NSString *desc = [err localizedDescription];
                                         if (desc) {
                                           out_err = std::string([desc UTF8String]);
                                         } else {
                                           out_err = "Speech recognition error";
                                         }
                                         signaled = true;
                                         dispatch_semaphore_signal(done);
                                         return;
                                       }
                                       if (result && result.isFinal) {
                                         NSString *s =
                                             result.bestTranscription.formattedString;
                                         if (s) {
                                           out = std::string([s UTF8String]);
                                         }
                                         signaled = true;
                                         dispatch_semaphore_signal(done);
                                       }
                                     }];

      dispatch_time_t done_wait =
          dispatch_time(DISPATCH_TIME_NOW, (int64_t)(30 * NSEC_PER_SEC));
      long rc = dispatch_semaphore_wait(done, done_wait);
      if (rc != 0) {
        if (task) {
          [task cancel];
        }
        if (out_error) {
          *out_error = "Speech recognition timeout";
        }
        return {};
      }

      if (!out_err.empty()) {
        if (out_error) {
          *out_error = out_err;
        }
        return {};
      }

      return out;
    };

    NSString *current_locale_id = [NSLocale currentLocale].localeIdentifier;
    NSArray<NSString *> *try_locale_ids = @[
      @"zh-CN",
      current_locale_id ? current_locale_id : @"",
      @"en-US",
    ];

    NSMutableSet<NSString *> *seen = [NSMutableSet set];
    std::string last_err;
    for (NSString *locale_id in try_locale_ids) {
      if (!locale_id || locale_id.length == 0) {
        continue;
      }
      if ([seen containsObject:locale_id]) {
        continue;
      }
      [seen addObject:locale_id];

      std::string per_err;
      std::string text = transcribe_with_locale_identifier(locale_id, &per_err);
      if (!text.empty()) {
        return text;
      }
      if (!per_err.empty()) {
        last_err = per_err;
      }
    }

    std::string fallback_err;
    std::string text = transcribe_with_locale_identifier(nil, &fallback_err);
    if (!text.empty()) {
      return text;
    }

    if (error) {
      if (!fallback_err.empty()) {
        *error = fallback_err;
      } else if (!last_err.empty()) {
        *error = last_err;
      } else {
        *error = "Speech recognition failed";
      }
    }
    return {};
  }
}

} // namespace media
} // namespace duorou
