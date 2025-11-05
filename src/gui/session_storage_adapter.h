#ifndef SESSION_STORAGE_ADAPTER_H
#define SESSION_STORAGE_ADAPTER_H

#ifdef __cplusplus

#include "chat_session.h"
#include <memory>
#include <string>
#include <vector>

namespace duorou {
namespace gui {

/**
 * Session storage adapter class
 * Encapsulates MiniMemory's DataStore interface, specifically for persistent storage of chat sessions
 */
class SessionStorageAdapter {
public:
  SessionStorageAdapter();
  ~SessionStorageAdapter();

  // Initialize storage connection
  bool initialize(const std::string &server_host = "localhost",
                  int server_port = 6379);

  // Authenticate with server if requirepass is set
  bool authenticate(const std::string &password);

  // Save single session
  bool saveSession(const duorou::gui::ChatSession &session);

  // Load single session
  std::unique_ptr<duorou::gui::ChatSession>
  loadSession(const std::string &session_id);

  // Delete session
  bool deleteSession(const std::string &session_id);

  // Get all session ID list
  std::vector<std::string> getAllSessionIds();

  // Check if session exists
  bool sessionExists(const std::string &session_id);

  // Save all session data to disk
  bool saveToFile();

  // Load all session data from disk
  bool loadFromFile();

  // Clear all session data
  bool clearAllSessions();

  // Get session count
  size_t getSessionCount();

private:
  std::string server_host_;
  int server_port_;
  int socket_fd_;
  bool connected_;

  // Serialize session to JSON string
  std::string serializeSession(const duorou::gui::ChatSession &session);

  // Deserialize session from JSON string
  std::unique_ptr<duorou::gui::ChatSession>
  deserializeSession(const std::string &json_data);

  // Generate session key name in storage
  std::string getSessionKey(const std::string &session_id);

  // Session list key name
  static const std::string SESSION_LIST_KEY;

  // Session data key prefix
  static const std::string SESSION_DATA_PREFIX;

  // Network communication methods
  bool connectToServer();
  void disconnectFromServer();
  bool sendCommand(const std::string &command);
  std::string receiveResponse();

  // Redis protocol command building
  std::string buildSetCommand(const std::string &key, const std::string &value);
  std::string buildGetCommand(const std::string &key);
  std::string buildDelCommand(const std::string &key);
  std::string buildExistsCommand(const std::string &key);
  std::string buildAuthCommand(const std::string &password);
};

} // namespace gui
} // namespace duorou

#endif // __cplusplus

#endif // SESSION_STORAGE_ADAPTER_H