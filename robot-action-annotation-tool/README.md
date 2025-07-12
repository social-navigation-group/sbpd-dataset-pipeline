# Annotation of Robot Actions as temporal events + Scenario Tagging
- Step 1: Read the instructions: https://docs.google.com/document/d/1FMzO5qanRmVTddC7rCmlcdaZCVK-AfdjpvnB9VY_fV8/
- Step 2: Check out sample videos for each category from NUS's dataset: https://drive.google.com/drive/u/1/folders/1Vfw-t9hTKwqGI_vOij0T2iOcBA7jAEai
- Step 3: Generate Videos from your bag files: Refer to [annotation_video_generation](annotation_video_generation) for instructions
- Step 3: Download our [modified annotator tool](via-3.x.y)  
- Step 4: Annotate Videos based on the instructions. **Remember to save the annotations using the Floppy Disk icon in the tool!**

# Robot action annotation tool

Please refer to [original README](./README_original.md) for details on using the annotation tool.

___

- The tool has been modified to include robot action categories, and scenario tags.
- The [video annotator](./via-3.x.y/src/html/_via_video_annotator.html) can be run in a browser, and video files for annotation (generated from rosbags) can be batch uploaded.
- To modify the robot action categories, modify the following section in [_via_temporal_segmenter.js](./via-3.x.y/src/js/_via_temporal_segmenter.js) and reload the html(via-3.x.y/src/html/_via_video_annotator.html). Ensure the categories and scenario tags are consistent across all annotations.\\

```js
var temporal_segment_options = {
      'Lane Keeping': 'Lane Keeping',
      'Lane Switching': 'Lane Switching',
      'Avoid Group': 'Avoid Group',
      'Avoid Pedestrian': 'Avoid Pedestrian',
      'Stop':  'Stop',
      'Slow down': 'Slow down',
      'Dense Crowd': 'Dense Crowd'
    };
var scenario_tag_options = {
    'With Traffic': 'With Traffic',
    'Against Traffic': 'Against Traffic',
    'Overtaking Pedestrian(s)': 'Overtaking Pedestrian(s)',
    'Passing Conversational Groups': 'Passing Conversational Groups',
    'Blind corner': 'Blind corner',
    'Entry/Exit': 'Entry/Exit',
    'Line Formation (like queues)': 'Line Formation (like queues)',
    'Navigating through large crowds': 'Navigating through large crowds',
    'Open Area': 'Open Area',
    'Intersections': 'Intersections',
    'Wide Corridors': 'Wide Corridors',
    'Narrow Corridors': 'Narrow Corridors'
};
```

### Points to note:

1. The annotations will be saved in a .json file with the same name as the corresponding video file. This saves some redundant information in the .json file, but will be handled by our parsing script.
2. Loading too many (> 20) or large video files at once can cause the tool to hang, resulting in unsaved progress being lost.

> **WARNING: all unsaved changes are lost on refreshing/reloading the browser window.**
