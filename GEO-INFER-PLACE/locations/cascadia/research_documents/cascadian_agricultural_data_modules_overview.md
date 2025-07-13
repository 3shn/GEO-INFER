# Cascadian Agricultural Data Modules: Implementation Specification

**Northern California + Oregon Agricultural Land Analysis Framework**
*Integrated H3-OSC Backend for Joint Analysis and Visualization*

---

## 1. Executive Summary

This document is the **authoritative technical reference** for the implementation, integration, and operation of all Cascadian agricultural data modules. It details:
- **Module-specific data acquisition, processing, and analysis workflows**
- **H3/OSC (Open Spatial Computing) integration patterns** via the GEO-INFER-SPACE module
- **Separation of concerns**: module-specific logic, place-based orchestration, and fundamental geospatial methods
- **Robust error handling, caching, and documentation standards**
- **Explicit use of the OSC repo path** for all OSC-related operations

The framework is architected so that **all geospatial data is standardized into H3 hexagonal grids** using the OSC loader tools, with all real data downloads, processing, and analysis performed in a modular, reproducible, and scalable manner.

---

## 2. Core Architecture & Configuration

- **Orchestration**: The `cascadia_main.py` script manages the full analysis pipeline, reading configuration from `config/analysis_config.json` and supporting command-line overrides.
- **Module Structure**: Each module (e.g., zoning, current_use, ownership, etc.) is implemented as a subfolder with:
  - `data_sources.py`: Data acquisition and caching logic
  - `geo_infer_<module>.py`: H3/OSC integration and analysis logic
  - `__init__.py`: Module API
- **Separation of Concerns**:
  - **Module-specific logic**: In each module's subfolder
  - **Generic place-based orchestration**: In `PLACE/src/geo_infer_place/core/`
  - **Fundamental geospatial/H3/OSC methods**: In `SPACE/src/geo_infer_space/`
- **OSC Repo Path**: All OSC-related operations (e.g., H3 loader, OSC data integration) must use the repo path from `@/repo` (never hardcoded). This is passed via config or backend initialization.

---

## 3. Standardized Module Workflow

All modules must follow this workflow:

### Step 1: Data Acquisition and Caching
- **Download real data** from authoritative sources (API, direct download, or scraping as needed)
- **Cache** all raw data in `data/<module_name>/` to avoid redundant downloads
- **Robust error handling**: Log and handle all download, network, and file errors
- **Document** all data sources, update frequencies, and access methods

### Step 2: H3/OSC Integration and Processing
- **Process raw data** into a standard geospatial format (GeoDataFrame, raster, etc.)
- **Convert to H3** using the OSC loader via GEO-INFER-SPACE utilities, always referencing the correct OSC repo path
- **Store H3-indexed data** in the module's cache directory
- **Document** all processing steps, including CRS handling, attribute mapping, and spatial joins

### Step 3: Analysis and Exposure
- **Load H3 data** for analysis
- **Perform module-specific analysis** (e.g., zoning classification, crop use, ownership concentration)
- **Return H3-indexed results** to the backend for aggregation
- **Document** all analysis methods, scoring, and output formats

---

## 4. Module-by-Module Technical Specifications

### Zoning (`GeoInferZoning`)
- **Purpose**: Land use/zoning classification for regulatory and suitability analysis
- **Data**: FMMP (CA), ORMAP (OR), Regrid, county sources
- **Acquisition**: Download or API, with cache check and robust error handling
- **Processing**: Convert polygons to H3 using OSC loader (repo path from config)
- **Analysis**: Zoning class aggregation, regulatory status, development pressure
- **Documentation**: All methods must have docstrings, type hints, and usage examples

### Current Use (`GeoInferCurrentUse`)
- **Purpose**: Real-time crop/land use classification
- **Data**: NASS CDL, Land IQ, Oregon EFU
- **Acquisition**: API or direct download, with SSL and error handling
- **Processing**: Raster-to-H3 conversion using OSC loader
- **Analysis**: Crop type, diversity, intensity, water/economic value
- **Documentation**: As above

### Ownership (`GeoInferOwnership`)
- **Purpose**: Ownership patterns, concentration, institutional share
- **Data**: ParcelQuest, Regrid, ORMAP
- **Acquisition**: API or bulk download, with cache and error handling
- **Processing**: Parcel-to-H3 mapping using OSC loader
- **Analysis**: Herfindahl index, largest owner, diversity
- **Documentation**: As above

### Mortgage Debt (`GeoInferMortgageDebt`)
- **Purpose**: Debt/financial risk estimation
- **Data**: HMDA, USDA ERS, county records
- **Acquisition**: Bulk download, API, or scraping as needed
- **Processing**: Join to census tracts, then to H3 using OSC loader
- **Analysis**: Debt-to-asset ratio, risk classification
- **Documentation**: As above

### Improvements (`GeoInferImprovements`)
- **Purpose**: Infrastructure/building/irrigation analysis
- **Data**: Microsoft/Google building footprints, USDA, DWR
- **Acquisition**: Download, with cache and error handling
- **Processing**: Footprint-to-H3 mapping using OSC loader
- **Analysis**: Building area, density, modernization score
- **Documentation**: As above

### Surface Water (`GeoInferSurfaceWater`)
- **Purpose**: Surface water rights and allocation
- **Data**: eWRIMS, CalWATRS, Oregon WRD, NHD
- **Acquisition**: API or download, with cache and error handling
- **Processing**: Points/diversions to H3 using OSC loader
- **Analysis**: Allocation, seniority, seasonal patterns
- **Documentation**: As above

### Ground Water (`GeoInferGroundWater`)
- **Purpose**: Groundwater rights, well density, availability
- **Data**: USGS NWIS, CASGEM, Oregon GWIC
- **Acquisition**: API, with cache and error handling
- **Processing**: Well points to H3 using OSC loader
- **Analysis**: Well density, depth, yield, sustainability
- **Documentation**: As above

### Power Source (`GeoInferPowerSource`)
- **Purpose**: Utility/energy infrastructure and consumption
- **Data**: HIFLD, EIA, utility companies, NREL
- **Acquisition**: API or download, with cache and error handling
- **Processing**: Infrastructure to H3 using OSC loader
- **Analysis**: Utility provider, renewable capacity, consumption
- **Documentation**: As above

---

## 5. Unified H3-OSC Backend and Integration

- **Backend**: `CascadianAgriculturalH3Backend` orchestrates all modules, passing the OSC repo path to all H3/OSC operations
- **Target Region**: Defined by county boundaries or bioregion, polyfilled to H3 using SPACE utilities
- **Module Initialization**: Each module is initialized with the backend, config, and OSC repo path
- **Analysis Pipeline**: For each module: acquire/cached data → process to H3 (OSC loader) → analyze → aggregate results
- **Error Handling**: All errors are logged and surfaced to the backend; critical errors halt the pipeline
- **Documentation**: All backend methods must be fully documented

---

## 6. Documentation and Quality Standards

- **Docstrings**: Every public function/method must have a comprehensive docstring (purpose, params, return, exceptions, usage)
- **Type Hints**: All functions must use type hints
- **Error Handling**: All data acquisition and processing must have robust error handling and logging
- **OSC Repo Path**: All OSC/H3 operations must use the repo path from config/backend, never hardcoded
- **Testing**: Each module must be testable standalone, with real data and H3/OSC integration
- **Separation of Concerns**: Module-specific logic in subfolders, generic orchestration in PLACE/src, geospatial/H3/OSC in SPACE/src

---

## 7. Implementation and Validation Plan

1. **Audit and update all module data acquisition and processing methods** for:
   - Use of SPACE H3/OSC utilities
   - Use of the correct OSC repo path
   - Robust error handling and logging
   - Complete, accurate docstrings and type hints
2. **Refactor as needed** to:
   - Move any generic logic to PLACE/src or SPACE/src
   - Ensure all module-specific logic is in the module’s subfolder
   - Remove any hardcoded paths or mock methods
3. **Update documentation** for:
   - Each method (docstrings)
   - Each module (README, research docs)
   - Data and output directory structure
4. **Test each module standalone** to ensure:
   - Real data can be downloaded and processed
   - H3/OSC integration works as expected
   - Outputs are generated in the correct format and location

---

## 8. Quality Assurance and Active Inference Integration

- **Data Quality**: Source validation, temporal consistency, spatial accuracy, cross-module consistency
- **Active Inference**: Bayesian uncertainty quantification, predictive modeling, adaptive learning
- **Continuous Improvement**: All modules and documentation are living, version-controlled, and subject to ongoing review

---

## 9. Conclusion

This document is the single source of truth for the technical implementation of the Cascadian agricultural data modules. All code, documentation, and integration must adhere to the standards and patterns described herein. The result is a robust, scalable, and extensible geospatial inference framework for agricultural analysis in the Cascadian bioregion. 