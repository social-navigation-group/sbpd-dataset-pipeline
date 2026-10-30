"""
BEV Semantic Annotation Tool
Draws polygon regions on BEV images, saves integer masks and JSON annotations.
"""

import json
import math
import re
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl
from PIL import Image, ImageDraw, ImageTk

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
FRAMES_DIR      = BASE_DIR / "frames"
TRAJ_FRAMES_DIR = BASE_DIR / "traj_frames"
MASKS_DIR       = BASE_DIR / "masks"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
XLSX_PATH       = BASE_DIR / "BEV Tags.xlsx"

# ─── Tag registry ─────────────────────────────────────────────────────────────
TAG_IDS: dict[str, int] = {
    "Open Area":      1,
    "Wide Corridor":  2,
    "Narrow Corridor":3,
    "Intersection":   4,
    "Entry/Exit":     5,
    "Blind Corner":   6,
    "Ignore":         7,
}

# RGBA fill colors (semi-transparent overlays)
TAG_COLORS: dict[int, tuple] = {
    1: (255, 200,   0,  90),
    2: (  0, 180, 255,  90),
    3: (255,  80,  80,  90),
    4: (  0, 220,  80,  90),
    5: (220,   0, 220,  90),
    6: (255, 140,   0,  90),
    7: (160, 160, 160,  90),
}

# Tkinter outline colors (fully opaque)
TAG_OUTLINE: dict[int, str] = {
    1: "#FFD700",
    2: "#00B4FF",
    3: "#FF5050",
    4: "#00DC50",
    5: "#DC00DC",
    6: "#FF8C00",
    7: "#A0A0A0",
}

# Button highlight background per tag
TAG_BG: dict[int, str] = {
    1: "#665500",
    2: "#004466",
    3: "#660000",
    4: "#005500",
    5: "#440044",
    6: "#663300",
    7: "#444444",
}

CLOSE_THRESHOLD_PX = 15


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_xlsx(path: Path) -> dict[str, list[str]]:
    """Return {stem: [tag_name, ...]} from the BEV Tags xlsx."""
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    result: dict[str, list[str]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        filename, tags_cell = row[0], row[1]
        if not filename or not tags_cell:
            continue
        stem = Path(str(filename)).stem
        tags = [t.strip() for t in str(tags_cell).split(",") if t.strip() in TAG_IDS]
        if tags:
            result[stem] = tags
    return result


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class RegionRecord:
    tag_id:   int
    tag_name: str
    points:   list[tuple[int, int]] = field(default_factory=list)


# ─── Drawing state machine ────────────────────────────────────────────────────

class DrawingStateMachine:
    def __init__(self):
        self.state: str = "idle"         # "idle" | "drawing"
        self.active_tag_id: int | None = None
        self.active_tag_name: str = ""
        self.points_img: list[tuple[float, float]] = []  # image-space floats
        self.scale: float = 1.0

    def start(self, tag_id: int, tag_name: str):
        self.state = "drawing"
        self.active_tag_id = tag_id
        self.active_tag_name = tag_name
        self.points_img = []

    def add_point(self, cx: float, cy: float):
        """Convert canvas coords to image coords and store."""
        self.points_img.append((cx / self.scale, cy / self.scale))

    def should_close(self, cx: float, cy: float) -> bool:
        if len(self.points_img) < 3:
            return False
        p0x = self.points_img[0][0] * self.scale
        p0y = self.points_img[0][1] * self.scale
        dist = math.hypot(cx - p0x, cy - p0y)
        return dist < CLOSE_THRESHOLD_PX

    def complete(self) -> RegionRecord:
        pts = [(round(x), round(y)) for x, y in self.points_img]
        record = RegionRecord(
            tag_id=self.active_tag_id,
            tag_name=self.active_tag_name,
            points=pts,
        )
        self.state = "idle"
        self.points_img = []
        return record

    def cancel(self):
        self.state = "idle"
        self.points_img = []

    @property
    def canvas_points(self) -> list[tuple[float, float]]:
        return [(x * self.scale, y * self.scale) for x, y in self.points_img]


# ─── Main application ─────────────────────────────────────────────────────────

class AnnotatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BEV Semantic Annotation Tool")
        self.configure(bg="#2b2b2b")

        # Load xlsx
        self.stem_to_tags = load_xlsx(XLSX_PATH)

        # Build natural-sorted image list (only jpgs whose stem is in xlsx)
        def _natural_key(p: Path) -> list:
            return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.stem)]

        self.image_list: list[Path] = sorted(
            (p for p in FRAMES_DIR.glob("*.jpg") if p.stem in self.stem_to_tags),
            key=_natural_key,
        )
        if not self.image_list:
            tk.messagebox.showerror(
                "No images",
                f"No matching jpg files found in {FRAMES_DIR}"
            )
            self.destroy()
            return

        self.current_index: int = 0
        self.regions: list[RegionRecord] = []
        self.active_tag_id: int | None = None
        self.drawing = DrawingStateMachine()
        self.cursor_pos: tuple[float, float] = (0.0, 0.0)
        self._photo_base: ImageTk.PhotoImage | None = None
        self._photo_overlay: ImageTk.PhotoImage | None = None
        self._orig_img: Image.Image | None = None  # original PIL image (full res)
        self.scale: float = 1.0
        self._canvas_size: tuple[int, int] = (0, 0)
        self._tag_buttons: dict[int, tk.Button] = {}

        MASKS_DIR.mkdir(exist_ok=True)
        ANNOTATIONS_DIR.mkdir(exist_ok=True)

        self._build_ui()
        self._load_image(0)
        self._fit_window_to_image()

        # Keyboard bindings
        self.bind("<Return>", self._on_key_close)
        self.bind("c",        self._on_key_close)
        self.bind("<Escape>", self._on_key_cancel)
        self.bind("<Left>",   lambda _e: self._navigate(-1))
        self.bind("<Right>",  lambda _e: self._navigate(1))

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Main horizontal split: canvas_frame | right_panel
        self._main_frame = tk.Frame(self, bg="#2b2b2b")
        self._main_frame.pack(fill=tk.BOTH, expand=True)

        # ── File list panel ────────────────────────────────────────────────
        file_panel = tk.Frame(self._main_frame, bg="#2a2a2a", width=180)
        file_panel.pack(side=tk.LEFT, fill=tk.Y)
        file_panel.pack_propagate(False)

        tk.Label(file_panel, text="Files", bg="#2a2a2a", fg="white",
                 font=("Helvetica", 12, "bold")).pack(anchor=tk.W, padx=8, pady=(8, 4))

        file_list_frame = tk.Frame(file_panel, bg="#2a2a2a")
        file_list_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 8))

        file_scrollbar = tk.Scrollbar(file_list_frame)
        file_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._file_listbox = tk.Listbox(
            file_list_frame, yscrollcommand=file_scrollbar.set,
            bg="#1e1e1e", fg="#cccccc", selectbackground="#4a6fa5",
            font=("Helvetica", 11), relief=tk.FLAT, borderwidth=0,
            activestyle="none",
        )
        self._file_listbox.pack(fill=tk.BOTH, expand=True)
        file_scrollbar.config(command=self._file_listbox.yview)

        for p in self.image_list:
            self._file_listbox.insert(tk.END, p.stem)

        self._file_listbox.bind("<<ListboxSelect>>", self._on_file_select)

        # ── Left: canvas + bottom bar ──────────────────────────────────────
        left_col = tk.Frame(self._main_frame, bg="#2b2b2b")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        canvas_frame = tk.Frame(left_col, bg="#1e1e1e")
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        hbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(canvas_frame, bg="#1e1e1e", cursor="crosshair",
                                highlightthickness=0,
                                xscrollcommand=hbar.set,
                                yscrollcommand=vbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hbar.config(command=self.canvas.xview)
        vbar.config(command=self.canvas.yview)

        self.canvas.bind("<Button-1>",  self._on_canvas_click)
        self.canvas.bind("<Motion>",    self._on_canvas_motion)
        # Middle-click + drag to pan
        self.canvas.bind("<ButtonPress-2>",   self._on_pan_start)
        self.canvas.bind("<B2-Motion>",       self._on_pan_move)
        # Mouse wheel scroll
        self.canvas.bind("<MouseWheel>",        self._on_mousewheel)    # Windows/macOS vertical
        self.canvas.bind("<Button-4>",          self._on_mousewheel)    # Linux scroll up
        self.canvas.bind("<Button-5>",          self._on_mousewheel)    # Linux scroll down
        self.canvas.bind("<Control-MouseWheel>",  self._on_mousewheel_h)  # Windows/macOS horizontal
        self.canvas.bind("<Control-Button-4>",    self._on_mousewheel_h)  # Linux ctrl+scroll up
        self.canvas.bind("<Control-Button-5>",    self._on_mousewheel_h)  # Linux ctrl+scroll down

        # Bottom navigation bar
        bottom = tk.Frame(left_col, bg="#2b2b2b", pady=6)
        bottom.pack(fill=tk.X)

        nav_frame = tk.Frame(bottom, bg="#2b2b2b")
        nav_frame.pack()

        btn_style = dict(bg="#444", fg="white", relief=tk.FLAT,
                         padx=14, pady=6, font=("Helvetica", 13))
        tk.Button(nav_frame, text="◄ Prev", command=lambda: self._navigate(-1),
                  **btn_style).pack(side=tk.LEFT, padx=8)
        tk.Button(nav_frame, text="Next ►", command=lambda: self._navigate(1),
                  **btn_style).pack(side=tk.LEFT, padx=8)
        tk.Button(nav_frame, text="Exit", command=self.destroy,
                  bg="#883333", fg="white", relief=tk.FLAT,
                  padx=14, pady=6, font=("Helvetica", 13)).pack(side=tk.LEFT, padx=8)

        self._status_label = tk.Label(bottom, text="", bg="#2b2b2b", fg="#aaaaaa",
                                      font=("Helvetica", 11))
        self._status_label.pack(pady=(4, 0))

        # ── Right: tags + regions ──────────────────────────────────────────
        right_panel = tk.Frame(self._main_frame, bg="#333333", width=260,
                               padx=10, pady=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        tk.Label(right_panel, text="Tags", bg="#333333", fg="white",
                 font=("Helvetica", 13, "bold")).pack(anchor=tk.W, pady=(0, 6))

        self._tags_frame = tk.Frame(right_panel, bg="#333333")
        self._tags_frame.pack(fill=tk.X)

        tk.Frame(right_panel, bg="#555555", height=1).pack(fill=tk.X, pady=10)

        tk.Label(right_panel, text="Regions", bg="#333333", fg="white",
                 font=("Helvetica", 13, "bold")).pack(anchor=tk.W, pady=(0, 6))

        list_frame = tk.Frame(right_panel, bg="#333333")
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._region_listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set,
            bg="#222222", fg="white", selectbackground="#555555",
            font=("Helvetica", 12), relief=tk.FLAT, borderwidth=0,
            activestyle="none",
        )
        self._region_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._region_listbox.yview)

        tk.Button(right_panel, text="Delete Selected",
                  command=self._delete_selected,
                  bg="#883333", fg="white", relief=tk.FLAT,
                  padx=8, pady=6, font=("Helvetica", 12)
                  ).pack(fill=tk.X, pady=(10, 0))

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_image(self, index: int):
        self.current_index = index
        path = self.image_list[index]
        stem = path.stem

        # Load PIL image, composite RGBA over white
        img = Image.open(path)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Overlay traj_frame at 50% opacity if it exists
        traj_path = TRAJ_FRAMES_DIR / path.name
        if traj_path.exists():
            traj = Image.open(traj_path).convert("RGB")
            if traj.size != img.size:
                traj = traj.resize(img.size, Image.LANCZOS)
            img = Image.blend(img, traj, alpha=0.3)

        self._orig_img = img

        # Load saved annotations
        self._load_annotations(stem)

        # Cancel any in-progress drawing
        self.drawing.cancel()
        self.active_tag_id = None

        # Rebuild right panel
        self._rebuild_tag_buttons(stem)
        self._rebuild_region_list()

        # Trigger canvas sizing + display
        self._update_canvas_image()

        total = len(self.image_list)
        self._status_label.config(
            text=f"[{index+1}/{total}] {path.name}  —  Click a tag, then draw on the image  |  Scroll: vertical  Ctrl+Scroll: horizontal"
        )
        self.title(f"BEV Annotator — {path.name}")

    def _update_canvas_image(self):
        """Display image at 1:1 scale with scrollable canvas."""
        if self._orig_img is None:
            return
        iw, ih = self._orig_img.size
        self.scale = 1.0
        self.drawing.scale = 1.0
        self._canvas_size = (iw, ih)
        self.canvas.config(scrollregion=(0, 0, iw, ih))
        self._redraw_canvas()

    def _fit_window_to_image(self):
        """Resize the window so the canvas matches the image size, capped to screen."""
        if self._orig_img is None:
            return
        iw, ih = self._orig_img.size
        right_panel_w = 260   # matches the right panel width
        nav_bar_h     = 60    # approx height of bottom nav + status label
        scrollbar_w   = 16    # vertical scrollbar width
        scrollbar_h   = 16    # horizontal scrollbar height

        desired_w = iw + scrollbar_w + right_panel_w
        desired_h = ih + scrollbar_h + nav_bar_h

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        win_w = min(desired_w, screen_w - 40)
        win_h = min(desired_h, screen_h - 80)

        self.geometry(f"{win_w}x{win_h}")

    # ── Tag buttons ───────────────────────────────────────────────────────────

    def _rebuild_tag_buttons(self, stem: str):
        for w in self._tags_frame.winfo_children():
            w.destroy()
        self._tag_buttons.clear()

        tags = self.stem_to_tags.get(stem, []) + ["Ignore"]
        for tag_name in tags:
            tag_id = TAG_IDS[tag_name]
            btn = tk.Button(
                self._tags_frame,
                text=tag_name,
                command=lambda tid=tag_id, tn=tag_name: self._select_tag(tid, tn),
                bg="#444444", fg="white", relief=tk.FLAT,
                padx=6, pady=6, font=("Helvetica", 12),
                anchor=tk.W,
            )
            btn.pack(fill=tk.X, pady=2)
            self._tag_buttons[tag_id] = btn

    def _select_tag(self, tag_id: int, tag_name: str):
        # Cancel any in-progress polygon
        if self.drawing.state == "drawing":
            self.drawing.cancel()

        self.active_tag_id = tag_id

        # Update button highlights
        for tid, btn in self._tag_buttons.items():
            if tid == tag_id:
                btn.config(bg=TAG_BG[tid], relief=tk.SUNKEN)
            else:
                btn.config(bg="#444444", relief=tk.FLAT)

        self.drawing.start(tag_id, tag_name)
        self._redraw_canvas()
        self._status_label.config(
            text=f"Drawing: {tag_name}  —  Click to add points, click near start (or Enter) to close"
        )

    # ── Canvas interaction ────────────────────────────────────────────────────

    def _on_pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _on_pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_mousewheel(self, event):
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_h(self, event):
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self.canvas.xview_scroll(delta, "units")

    def _on_canvas_click(self, event):
        if self.drawing.state != "drawing":
            return

        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        if self.drawing.should_close(cx, cy):
            self._finish_polygon()
        else:
            self.drawing.add_point(cx, cy)
            self._redraw_canvas()

    def _on_canvas_motion(self, event):
        self.cursor_pos = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        if self.drawing.state == "drawing":
            self._redraw_canvas()

    def _on_key_close(self, event):
        if self.drawing.state == "drawing" and len(self.drawing.points_img) >= 3:
            self._finish_polygon()

    def _on_key_cancel(self, event):
        if self.drawing.state == "drawing":
            self.drawing.cancel()
            self._redraw_canvas()
            self._status_label.config(text="Drawing cancelled. Select a tag to start again.")

    # ── Polygon completion ────────────────────────────────────────────────────

    def _finish_polygon(self):
        record = self.drawing.complete()
        self.regions.append(record)
        self._save_annotations()
        self._render_and_save_mask()
        self._rebuild_region_list()
        self._redraw_canvas()

        # Re-start drawing for the same tag so user can add more regions immediately
        self.drawing.start(record.tag_id, record.tag_name)
        self._status_label.config(
            text=f"Region added ({record.tag_name}). Keep drawing or select another tag."
        )

    # ── Canvas rendering ──────────────────────────────────────────────────────

    def _redraw_canvas(self):
        self.canvas.delete("all")

        cw, ch = self._canvas_size

        # Build display image: base + overlay composite
        if self._orig_img:
            base_rgb = self._orig_img.resize((cw, ch), Image.LANCZOS)
            base_rgba = base_rgb.convert("RGBA")

            if self.regions:
                overlay = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
                odraw = ImageDraw.Draw(overlay)
                for r in self.regions:
                    pts = [(x * self.scale, y * self.scale) for x, y in r.points]
                    if len(pts) >= 3:
                        odraw.polygon(pts, fill=TAG_COLORS[r.tag_id])
                composite = Image.alpha_composite(base_rgba, overlay).convert("RGB")
            else:
                composite = base_rgb

            self._photo_base = ImageTk.PhotoImage(composite)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo_base)

        # Outline completed polygons
        for r in self.regions:
            pts = [(x * self.scale, y * self.scale) for x, y in r.points]
            if len(pts) >= 3:
                flat = [coord for pt in pts for coord in pt]
                self.canvas.create_polygon(
                    flat,
                    outline=TAG_OUTLINE[r.tag_id],
                    fill="",
                    width=2,
                )

        # In-progress polygon
        if self.drawing.state == "drawing":
            cpts = self.drawing.canvas_points
            color = TAG_OUTLINE.get(self.drawing.active_tag_id, "#ffffff")
            dot_r = 4

            # Edges so far
            for i in range(1, len(cpts)):
                self.canvas.create_line(
                    cpts[i-1][0], cpts[i-1][1],
                    cpts[i][0],   cpts[i][1],
                    fill=color, width=2,
                )

            # Vertex dots
            for i, (px, py) in enumerate(cpts):
                r = dot_r + 3 if i == 0 else dot_r
                self.canvas.create_oval(
                    px - r, py - r, px + r, py + r,
                    fill=color, outline="white", width=1,
                )

            # Rubber-band line to cursor
            if cpts:
                self.canvas.create_line(
                    cpts[-1][0], cpts[-1][1],
                    self.cursor_pos[0], self.cursor_pos[1],
                    fill=color, width=1, dash=(4, 4),
                )

            # Close-zone indicator on first point (shown when ≥3 pts)
            if len(cpts) >= 3:
                p0 = cpts[0]
                self.canvas.create_oval(
                    p0[0] - CLOSE_THRESHOLD_PX, p0[1] - CLOSE_THRESHOLD_PX,
                    p0[0] + CLOSE_THRESHOLD_PX, p0[1] + CLOSE_THRESHOLD_PX,
                    outline=color, fill="", width=1, dash=(3, 3),
                )

    # ── Mask & annotation persistence ─────────────────────────────────────────

    def _render_and_save_mask(self):
        if self._orig_img is None:
            return
        iw, ih = self._orig_img.size
        stem = self.image_list[self.current_index].stem
        out_dir = MASKS_DIR / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect which tag IDs are present in current regions
        present_tag_ids = {r.tag_id for r in self.regions if len(r.points) >= 3}

        # Remove stale per-tag mask files for tags no longer present
        for existing in out_dir.glob("tag_*.jpg"):
            existing.unlink()

        # Save one binary mask per tag
        for tag_id in present_tag_ids:
            tag_name = next(r.tag_name for r in self.regions if r.tag_id == tag_id)
            mask = Image.new("L", (iw, ih), 0)
            mdraw = ImageDraw.Draw(mask)
            for r in self.regions:
                if r.tag_id == tag_id and len(r.points) >= 3:
                    mdraw.polygon(r.points, fill=255)
            safe_name = tag_name.replace("/", "_").replace(" ", "_")
            mask.save(out_dir / f"tag_{tag_id}_{safe_name}.jpg")

    def _save_annotations(self):
        stem = self.image_list[self.current_index].stem
        data = [
            {"tag_id": r.tag_id, "tag_name": r.tag_name, "points": r.points}
            for r in self.regions
        ]
        with open(ANNOTATIONS_DIR / f"{stem}.json", "w") as f:
            json.dump(data, f, indent=2)

    def _load_annotations(self, stem: str):
        ann_path = ANNOTATIONS_DIR / f"{stem}.json"
        if ann_path.exists():
            with open(ann_path) as f:
                data = json.load(f)
            self.regions = [
                RegionRecord(
                    tag_id=d["tag_id"],
                    tag_name=d["tag_name"],
                    points=[tuple(p) for p in d["points"]],
                )
                for d in data
            ]
        else:
            self.regions = []

    # ── Region list ───────────────────────────────────────────────────────────

    def _rebuild_region_list(self):
        self._region_listbox.delete(0, tk.END)
        for i, r in enumerate(self.regions):
            self._region_listbox.insert(tk.END, f"[{r.tag_name}] polygon {i+1}")
            hex_color = TAG_OUTLINE.get(r.tag_id, "#ffffff")
            self._region_listbox.itemconfig(i, fg=hex_color)

    def _delete_selected(self):
        sel = self._region_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.regions):
            self.regions.pop(idx)
            self._save_annotations()
            self._render_and_save_mask()
            self._rebuild_region_list()
            self._redraw_canvas()

    # ── Navigation ────────────────────────────────────────────────────────────

    def _navigate(self, delta: int):
        if self.drawing.state == "drawing":
            self.drawing.cancel()
        new_idx = (self.current_index + delta) % len(self.image_list)
        self._load_image(new_idx)

    def _on_file_select(self, _event):
        sel = self._file_listbox.curselection()
        if sel and sel[0] != self.current_index:
            if self.drawing.state == "drawing":
                self.drawing.cancel()
            self._load_image(sel[0])


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    MASKS_DIR.mkdir(exist_ok=True)
    ANNOTATIONS_DIR.mkdir(exist_ok=True)
    app = AnnotatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
