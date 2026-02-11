# Change Log

## Version 9.1.4 - 2026-02-11

### Fixed
- **Icon Manager**: Resolved image file issues in IconManager.

# Change Log

## Version 9.1.3 - 2026-02-08

### Changed
- **EDC Fitter Reopening**: Restored the EDC Fitter feature with partial functionality compared to MDC Fitter.
- **Menu Button Visual Enhancement**: Optimized the visual effects of menu button icons for improved user interface consistency.
- **MenuIcon Extraction Refinement**: Fine-tuned the MenuIcon extraction methodology for improved reliability and performance.

# Change Log

## Version 9.1.2 - 2026-02-07

### Added
- **MDC Fitter Keyboard Shortcuts**: Added keyboard shortcut definitions for quick index switching in MDC Fitter.
- **System Tray Icon for Qt Applications**: Introduced a system tray icon feature for better application accessibility.

### Changed
- **Unit Tests Update**: Enhanced test coverage and refined test execution rules for improved code validation.
- **UI Window Refinements**: Minor adjustments to AboutWindow and VersionUpdateWindow for better user experience.
- **Code Refactoring**: Refactored portions of the codebase for improved maintainability.

### Fixed
- **macOS Application Icon Display Issue**: Resolved the issue with application icon not displaying correctly on macOS.

# Change Log

## Version 9.1.1 - 2026-02-05

### Changed
- **Unit Tests Update**: Enhanced test coverage and refined test execution rules for improved code validation and reliability.
- **Test Configuration Optimization**: Fine-tuned pytest coverage exclusion rules in setup.cfg to better align with project requirements.

### Fixed
- **Spectrogram Window Size Adjustment on macOS**: Optimized window rendering and sizing behavior for improved display consistency on macOS systems.

# Change Log

## Version 9.1 - 2026-02-05

### Added
- **Release Notes Preview**: Added a release note preview in the version update interface.
- **Markdown Support**: Introduced `markdown` and `tkhtmlview` modules for formatted release note rendering.

### Changed
- **Documentation Update**: Updated README with the latest information.
- **Dependency Management**: Added `google-crc32c` for Python 3.12+ to silence `numcodecs` deprecation warnings.
- **Normalized Diagram Rendering**: Reworked the normalized diagram implementation to significantly improve display performance.
- **Shared Library Refactor**: Refactored common utility libraries for better maintainability.

### Fixed
- **Module Import Handling**: Fixed the module import error handling workflow.
- **MDC Fitter Save Prompt**: Fixed the save prompt logic in MDC Fitter to correctly detect unsaved changes.