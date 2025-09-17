# Changelog

## [0.4.3] - 2025-09-17

### Performance
- Reduced package size by 25.6%
- Rust compiler optimizations and dependency feature reduction
- Removed unused code methods from removed implementations

## [0.4.2] - 2025-09-16

### Documentation
- Updated README with performance benchmarks from finite alphabet sequence analysis
- Added evidence-based recommendations (F1: 0.71 for 16-state protocols, +6.2% vs traditional ML)
- Added some API examples to the root documentation
- Clear niche positioning for finite alphabet sequences (≤20 symbols) while highlighting the opportunity to batch process the states to reduce the alphabets even more

## [0.4.1] - 2025-01-27

### Changed
- Improved scoring system performance with better weight distribution
- Enhanced adaptive calibration for more accurate anomaly detection
- Optimized likelihood calculation for better separation

### Fixed
- Better anomaly separation between normal and anomalous sequences
- Improved numerical stability in score calculations

## [0.4.0] - 2025-09-13

### Added
- Parameter validation for max_order
- Improved error messages
- Clean API with essential utilities

### Changed
- Simplified codebase structure
- Updated documentation for accuracy

### Fixed
- Parameter validation now occurs at instantiation
- Removed false claims from documentation

## [0.3.1] - 2025-09-12

### Added
- Enhanced anomaly scoring algorithm
- Comprehensive test suite
- Performance optimizations

### Fixed
- Scoring calibration improvements
- Better handling of edge cases

## [0.3.0] - 2025-09-11

- Updated Rust library bindings

## [0.1.0] - 2025-09-10

### Added
- Initial release
- AnomalyDetector class
- Python 3.8+ support
- Basic test suite
