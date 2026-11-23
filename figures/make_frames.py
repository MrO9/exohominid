"""Cut the frame strips used in the paper from the screen recordings in ../docs/assets/vid (needs ffmpeg and Pillow)."""
import subprocess
from PIL import Image, ImageDraw
V = "../docs/assets/vid/"

def strip(video, times, box, out, label):
    ims = []
    for t in times:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", V + video, "-frames:v", "1", "/tmp/_f.png"], check=True)
        ims.append(Image.open("/tmp/_f.png").convert("RGB").crop(box).resize((540, 418)))
    m = Image.new("RGB", (550 * len(ims) - 10, 452), "white"); d = ImageDraw.Draw(m)
    for i, (im, t) in enumerate(zip(ims, times)):
        m.paste(im, (i * 550, 0)); d.text((i * 550 + 8, 424), f"{label}, t = {t:.1f} s", fill="black")
    m.save(out)

strip("baseline.mp4", [0.5, 1.5, 2.5, 3.5], (40, 30, 400, 310), "../paper/figs/fig_frames.png", "recording")
strip("training_h01.mp4", [0.2, 1.2, 2.2, 3.2], (150, 20, 550, 330), "../paper/figs/fig_trained_frames.png", "training recording")
