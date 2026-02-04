[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
<!-- [![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/CubicsYang/Vision2Slope) -->

<div align="center">
    <a href="" title="Vision2Slope — Street-view-based urban slope estimation framework">
        <img src="images/logo.png" alt="Vision2Slope — Street-view-based urban slope estimation framework" style="max-width:480px; height:auto;">
    </a>
</div>

# Vision2Slope

An integrated pipeline for two-level road slope analysis from single panoramic street view images

## Overview

**Vision2Slope** is a comprehensive pipeline designed to analyze road slopes using single panoramic street view images. The pipeline leverages advanced computer vision techniques to extract and compute slope information along the road, including both **point-level and segment-level** analyses, which can be useful for various applications such as urban planning, navigation, and infrastructure development.

## Key Features

- **Panorama Support**: Automatically converts panoramic street view images to perspective views (left and right) for accurate slope analysis.

- **Side-view Deskewing**: Transforms panoramic images into side-view perspectives and corrects vertical distortions to ensure accurate analysis using semantic and geometric prompts.

- **Point-level Slope Estimation**: Computes the slope of the road surface at specific points using the segmented road areas and their 3D geometry.

- **Segment-level Slope Estimation**: Analyzes the slope over larger road segments to provide a comprehensive understanding of road gradients and relief.


## Usage and Installation

### Example Code Snippet and Tutorial
Please refer to the [Tutorial](examples/Vision2Slope_Tutorial.ipynb) or [Script](examples/example.py) for detailed instructions on how to use Vision2Slope for slope estimation from street view images.

------
### Quick prompts based on the tutorial
1. Install from source:
```bash
python -m pip install -e ./src/Vision2Slope
```
2. Import the pipeline in Python:
```python
from vision2slope import (
    PipelineConfig,
    VisualizationConfig,
    ProcessingConfig,
    Vision2SlopePipeline
)
```
3. Optional: download Mapillary panoramas (requires an API key):
```python
from zensvi.download import MLYDownloader

mly_api_key = "YOUR_MAPILLARY_API_KEY_HERE"
downloader = MLYDownloader(mly_api_key=mly_api_key)
downloader.download_svi(
    "sf",
    lat=37.797423890238,
    lon=-122.44403351517,
    buffer=5,
    resolution=2048,
    image_type="pano"
)
```
4. Configure and run panoramic processing:
```python
config_panorama = PipelineConfig(
    input_dir="sf/mly_svi/batch_1",
    output_dir="sf/mly_svi/output_pano",
    processing_config=ProcessingConfig(
        is_panorama=True,
        panorama_fov=90,
        panorama_phi=0.0,
        panorama_aspects=(10, 10),
        panorama_show_size=100,
        log_level="INFO"
    ),
    viz_config=VisualizationConfig(
        save_visualizations=True,
        save_corrected_images=True,
        save_intermediate_results=True
    )
)

pipeline_panorama = Vision2SlopePipeline(config_panorama)
results_panorama = pipeline_panorama.process_batch()
```
5. Optional: Configure and run perspective images (non-panoramic):
```python
config_perspective = PipelineConfig(
    input_dir="input",
    output_dir="output"
)

pipeline_perspective = Vision2SlopePipeline(config_perspective)
results_perspective = pipeline_perspective.process_batch()
```

## Results

The pipeline generates detailed slope analysis results, including visualizations of slope distributions and numerical slope values for both point-level and segment-level analyses.

![Vision2Slope — Street-view-based urban slope estimation framework](images/SF-map.png)

*Figure: Two-level road slope maps using Vision2Slope.*

## TODO list

- ✅ Release the codebase
- ✅ Add installation instructions
- ✅ Provide usage examples
- ✅ Support panoramic image input
- ✅ Integrate more SVI platforms into Vision2Slope
- [ ] Expand study to diverse geographic locations


## Acknowledgements

This project is inspired and supported by the Google Street View, OpenStreetMap, and Opentopography communities for providing open access to their valuable data resources. Additional thanks to the developers of the open-source libraries utilized in this project, including OpenCV, NumPy, Pandas, Matplotlib, PIL, [ZenSVI](https://github.com/koito19960406/ZenSVI) and [streetlevel](https://github.com/sk-zk/streetlevel).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.