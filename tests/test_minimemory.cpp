#include <iostream>
#include <string>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

class MiniMemoryTester {
public:
    MiniMemoryTester() : socket_fd_(-1), connected_(false) {}
    
    ~MiniMemoryTester() {
        disconnect();
    }
    
    bool connect(const std::string& host = "127.0.0.1", int port = 6379) {
        socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd_ < 0) {
            std::cerr << "Error creating socket" << std::endl;
            return false;
        }
        
        struct sockaddr_in server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);
        
        if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
            std::cerr << "Invalid address: " << host << std::endl;
            close(socket_fd_);
            socket_fd_ = -1;
            return false;
        }
        
        if (::connect(socket_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Connection failed to " << host << ":" << port << std::endl;
            close(socket_fd_);
            socket_fd_ = -1;
            return false;
        }
        
        connected_ = true;
        std::cout << "Connected to MiniMemory server at " << host << ":" << port << std::endl;
        return true;
    }
    
    void disconnect() {
        if (socket_fd_ >= 0) {
            close(socket_fd_);
            socket_fd_ = -1;
        }
        connected_ = false;
    }
    
    bool sendCommand(const std::string& command) {
        if (!connected_ || socket_fd_ < 0) {
            return false;
        }
        
        ssize_t bytes_sent = send(socket_fd_, command.c_str(), command.length(), 0);
        return bytes_sent == static_cast<ssize_t>(command.length());
    }
    
    std::string receiveResponse() {
        if (!connected_ || socket_fd_ < 0) {
            return "";
        }
        
        char buffer[4096];
        ssize_t bytes_received = recv(socket_fd_, buffer, sizeof(buffer) - 1, 0);
        
        if (bytes_received <= 0) {
            disconnect();
            return "";
        }
        
        buffer[bytes_received] = '\0';
        return std::string(buffer);
    }
    
    std::string buildSetCommand(const std::string& key, const std::string& value) {
        return "*3\r\n$3\r\nSET\r\n$" + std::to_string(key.length()) + "\r\n" + key +
               "\r\n$" + std::to_string(value.length()) + "\r\n" + value + "\r\n";
    }
    
    std::string buildGetCommand(const std::string& key) {
        return "*2\r\n$3\r\nGET\r\n$" + std::to_string(key.length()) + "\r\n" + key + "\r\n";
    }
    
    bool testSetGet() {
        std::cout << "\n=== Testing SET/GET operations ===" << std::endl;
        
        // Test SET command
        std::string test_key = "test_session_123";
        std::string test_value = "{\"id\":\"test_session_123\",\"title\":\"Test Chat\",\"messages\":[]}";
        
        std::string set_cmd = buildSetCommand(test_key, test_value);
        std::cout << "Sending SET command..." << std::endl;
        
        if (!sendCommand(set_cmd)) {
            std::cerr << "Failed to send SET command" << std::endl;
            return false;
        }
        
        std::string set_response = receiveResponse();
        std::cout << "SET response: " << set_response << std::endl;
        
        if (set_response.find("+OK") != 0) {
            std::cerr << "SET command failed" << std::endl;
            return false;
        }
        
        // Test GET command
        std::string get_cmd = buildGetCommand(test_key);
        std::cout << "Sending GET command..." << std::endl;
        
        if (!sendCommand(get_cmd)) {
            std::cerr << "Failed to send GET command" << std::endl;
            return false;
        }
        
        std::string get_response = receiveResponse();
        std::cout << "GET response: " << get_response << std::endl;
        
        // Check if the response contains our test value
        if (get_response.find(test_value) == std::string::npos) {
            std::cerr << "GET command did not return expected value" << std::endl;
            return false;
        }
        
        std::cout << "SET/GET test passed!" << std::endl;
        return true;
    }
    
private:
    int socket_fd_;
    bool connected_;
};

int main() {
    std::cout << "MiniMemory Connection Test" << std::endl;
    std::cout << "==========================" << std::endl;
    
    MiniMemoryTester tester;
    
    // Test connection
    if (!tester.connect()) {
        std::cerr << "Failed to connect to MiniMemory server" << std::endl;
        std::cerr << "Make sure MiniMemory server is running on localhost:6379" << std::endl;
        return 1;
    }
    
    // Test basic operations
    if (!tester.testSetGet()) {
        std::cerr << "MiniMemory operations test failed" << std::endl;
        return 1;
    }
    
    std::cout << "\nAll tests passed! MiniMemory server is working correctly." << std::endl;
    return 0;
}