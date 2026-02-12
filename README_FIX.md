AstroGraphAnomaly: fix Matplotlib canvas capture (GIF)

What this fixes
- Newer Matplotlib versions: FigureCanvasAgg does not expose tostring_rgb().
- The A to H viz script failed when exporting 07_proper_motion_trails.gif.

What changed
- In tools/viz_a_to_h_suite.py, frame capture now uses canvas.buffer_rgba() and falls back to
  tostring_rgb() or tostring_argb() for older Matplotlib.

How to apply
1) Unzip this archive at the repo root.
2) Overwrite: tools/viz_a_to_h_suite.py
3) Commit and push, then rerun the manual_viz_a_to_h workflow.
