# Change Log

## Version 9.1 - 2026-02-05

### Added
- **Release Notes Preview**: Added a release note preview in the version update interface
- **Markdown Support**: Introduced `markdown` and `tkhtmlview` modules for formatted release note rendering

### Changed
- **Documentation Update**: Updated README with the latest information
- **Dependency Management**: Added `google-crc32c` for Python 3.12+ to silence `numcodecs` deprecation warnings
- **Normalized Diagram Rendering**: Reworked the normalized diagram implementation to significantly improve display performance
- **Shared Library Refactor**: Refactored common utility libraries for better maintainability

### Fixed
- **Module Import Handling**: Fixed the module import error handling workflow
- **MDC Fitter Save Prompt**: Fixed the save prompt logic in MDC Fitter to correctly detect unsaved changes