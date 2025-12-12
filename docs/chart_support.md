# Chart Support in Duorou

Duorou supports rendering Mermaid.js charts (e.g., flowcharts, sequence diagrams) in the chat interface.

## Requirements

1.  **WebKitGTK**: The application must be built with WebKitGTK support enabled.
    *   **macOS**: This is currently not standard on macOS builds unless `webkitgtk` is installed via Homebrew and linked properly.
    *   **Linux**: Install `libwebkit2gtk-4.0-dev` or `libwebkitgtk-6.0-dev`.

2.  **Mermaid.js**: The file `mermaid.min.js` must be present in the application's runtime directory (usually copied automatically by CMake).

## Usage

Ask the AI to generate a chart using standard Markdown Mermaid syntax:

    ```mermaid
    graph TD;
        A-->B;
        A-->C;
        B-->D;
        C-->D;
    ```

## Troubleshooting

*   **Charts show as code blocks**: This means either WebKitGTK is missing (falling back to native text rendering) or `mermaid.min.js` failed to load.
*   **"WebKitGTK not found" during build**: Install the development packages for your OS.
