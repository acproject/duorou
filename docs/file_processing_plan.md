# File Processing Enhancement Plan

## 1. Overview
The goal is to enhance the `duorou` application with file processing capabilities, allowing the AI model to "read" and understand content from uploaded files.
Supported formats will initially include:
- **PDF**: Portable Document Format (Text extraction).
- **Markdown**: `.md` files.
- **CSV**: Comma-Separated Values.
- **TXT**: Plain text files.

## 2. Architecture Design

### 2.1. Dependency Management
- **PDF Library**: Use `pdf-text-extraction` (based on `PDFHummus`) for robust PDF text extraction.
  - Location: `third_party/pdf-text-extraction`.
  - Integration: via CMake `add_subdirectory`.

### 2.2. Core Logic (`src/core`)
- **`FileParser` Interface**: A common interface for all file parsers.
  ```cpp
  class FileParser {
  public:
      virtual ~FileParser() = default;
      virtual std::string parse(const std::string& file_path) = 0;
      virtual bool supports(const std::string& extension) = 0;
  };
  ```
- **`PdfParser`**: Implementation using `pdf-text-extraction`.
- **`TextFileParser`**: Implementation for text-based formats (MD, CSV, TXT).
- **`FileParserFactory`** or **`ResourceManager` extension**: To manage and retrieve the correct parser.

### 2.3. UI Integration (`src/gui`)
- **`ChatView`**:
  - Intercept file uploads/sends.
  - When a file is selected and sent:
    1.  Detect file type.
    2.  Invoke the appropriate parser to extract text.
    3.  Append the extracted text to the message context (hidden or visible).
    4.  Send the combined context to the LLM.

## 3. Implementation Plan

### Phase 1: Dependency Setup (Completed)
- [x] Clone `pdf-text-extraction` to `third_party/pdf-text-extraction`.
- [ ] Verify build configuration.

### Phase 2: Core Implementation
- [ ] Create `src/core/file_parser.h` (Interface).
- [ ] Create `src/core/pdf_parser.h` and `src/core/pdf_parser.cpp`.
- [ ] Create `src/core/text_file_parser.h` and `src/core/text_file_parser.cpp`.
- [ ] Update `CMakeLists.txt` to include new files and link dependencies.

### Phase 3: Integration
- [ ] Modify `src/gui/chat_view.cpp` to use the parsers.
- [ ] Handle large file content (truncation or chunking - initially simple truncation).

### Phase 4: Verification
- [ ] Test with sample PDF, MD, CSV files.
- [ ] Verify CMake build passes.
