# BEV Semantic Annotation Tool

A GUI tool for annotating Bird's Eye View (BEV) images with semantic polygon regions.

## Dependencies

```bash
pip install pillow openpyxl
```

`tkinter` is part of the Python standard library (on Ubuntu: `sudo apt install python3-tk` if missing).

## Inputs

The tool expects `frames/` (one BEV image per session) and `BEV Tags.xlsx` (the tags
to draw per session) next to `tag_annotator.py`. These inputs are not part of this
repository; the annotations and masks produced with the tool are released with the
dataset under `BEV/BEV_annotations/`. Use `acme_dataset.bev` (see
`notebooks/03_bev_trajectories.ipynb`) to read them.

## Usage

```bash
cd bev_semantic_annotation_tool
python tag_annotator.py
```

## Workflow

1. The tool loads the first BEV image from `frames/` that has a matching entry in `BEV Tags.xlsx`.
2. The tags associated with the current image are shown as buttons in the right panel, along with a permanent **Ignore** tag.
3. **Select a tag** by clicking its button (it will highlight).
4. **Draw a polygon** on the image by left-clicking to place vertices. A rubber-band line follows the cursor.
5. **Close the polygon** by clicking near the first vertex (a dashed circle appears as a guide once ≥ 3 points are placed), or press `Enter` / `c`.
6. The polygon is added to the **Regions** list on the right. A tag can have multiple polygons.
7. **Delete** a polygon by selecting it in the Regions list and clicking **Delete Selected**.
8. Navigate between images with **◄ Prev** / **Next ►** (or `←` / `→` arrow keys).
9. Press `Escape` to cancel a polygon in progress.
10. Click **Exit** to close the application.

The canvas is scrollable. Pan by dragging with the **middle mouse button**.

## Output

Outputs are saved automatically on every polygon add/delete — no manual save step needed.

```
bev_semantic_annotation_tool/
├── annotations/<stem>.json       # polygon vertex data (used to restore state on relaunch)
└── masks/<stem>/
    ├── tag_1_Open_Area.jpg       # binary mask (255 = covered, 0 = not)
    ├── tag_4_Intersection.jpg
    └── ...                       # one file per tag that has at least one polygon
```

Each mask is an 8-bit grayscale JPEG at the original image resolution (threshold at 128 when reading). Tags can overlap freely since each lives in its own file.

## Tag IDs

| ID | Tag |
|----|-----|
| 1  | Open Area |
| 2  | Wide Corridor |
| 3  | Narrow Corridor |
| 4  | Intersection |
| 5  | Entry/Exit |
| 6  | Blind Corner |
| 7  | Ignore |
