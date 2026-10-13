"""
自适应图标预览：复刻 res/drawable/ic_launcher_*.xml 的几何，渲染成 PNG 供肉眼确认。

不是构建产物，只是设计校验工具（不进 APK）。
  python tools/icon_preview.py [输出目录]
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

S = 512                      # 输出边长
SS = 4                       # 超采样倍数（抗锯齿）
BG_A = (124, 140, 255)       # #7C8CFF
BG_B = (94, 107, 224)        # #5E6BE0
BG_C = (62, 59, 168)         # #3E3BA8

# 108 画布坐标 -> 输出像素
def to_px(x: float, y: float, size: int) -> tuple[float, float]:
    return x / 108 * size, y / 108 * size


def gradient(size: int) -> Image.Image:
    """对角线渐变背景（对应 android:angle=315）"""
    img = Image.new("RGB", (size, size))
    px = img.load()
    for j in range(size):
        for i in range(size):
            t = (i + j) / (2 * size)
            if t < 0.5:
                k = t / 0.5
                a, b = BG_A, BG_B
            else:
                k = (t - 0.5) / 0.5
                a, b = BG_B, BG_C
            px[i, j] = tuple(int(a[c] + (b[c] - a[c]) * k) for c in range(3))
    return img


def bezier(p0, p1, p2, p3, steps=90):
    """三次贝塞尔采样（眼形轮廓的上下弧）"""
    out = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        out.append((
            u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
            u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
        ))
    return out


def stroke_path(d: ImageDraw.ImageDraw, points, width: float, fill) -> None:
    """沿线盖圆点模拟圆头描边：比 line(joint=curve) 干净，无毛刺"""
    r = width / 2
    for x, y in points:
        d.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def draw_mark(img: Image.Image, size: int, fg=(255, 255, 255, 255), stroke_w=5.5) -> None:
    """眼睛轮廓 + 播放三角（前景层内容）"""
    layer = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    s = size * 2

    def P(x, y):
        return to_px(x, y, s)

    # 眼形：两段三次贝塞尔（与 vector pathData 对应）
    top = bezier(P(22, 54), P(31, 38.5), P(42.5, 31), P(54, 31))
    top += bezier(P(54, 31), P(65.5, 31), P(77, 38.5), P(86, 54))[1:]
    bottom = bezier(P(86, 54), P(77, 69.5), P(65.5, 77), P(54, 77))
    bottom += bezier(P(54, 77), P(42.5, 77), P(31, 69.5), P(22, 54))[1:]
    stroke_path(d, top + bottom, stroke_w / 108 * s, fg)

    # 播放三角
    d.polygon([P(47.5, 42.2), P(67.5, 54), P(47.5, 65.8)], fill=fg)

    img.alpha_composite(layer.resize((size, size), Image.LANCZOS))


def mask_circle(img: Image.Image) -> Image.Image:
    circle = Image.new("L", img.size, 0)
    ImageDraw.Draw(circle).ellipse((0, 0, img.size[0] - 1, img.size[1] - 1), fill=255)
    out = img.copy()
    out.putalpha(circle)
    return out


def squircle(img: Image.Image) -> Image.Image:
    """自适应图标常见的圆角方形遮罩（系统实际裁切形状的近似）"""
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, img.size[0] - 1, img.size[1] - 1), radius=int(img.size[0] * 0.235), fill=255
    )
    out = img.copy()
    out.putalpha(mask)
    return out


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    full = S * SS
    base = gradient(full).convert("RGBA")
    draw_mark(base, full)

    icon_squircle = squircle(base.resize((S, S), Image.LANCZOS))
    icon_circle = mask_circle(base.resize((S, S), Image.LANCZOS))

    # 无背景前景层（monochrome / Play 商店用）
    fg_only = Image.new("RGBA", (full, full), (0, 0, 0, 0))
    draw_mark(fg_only, full, fg=(255, 255, 255, 255))
    fg_only = fg_only.resize((S, S), Image.LANCZOS)

    # 排版预览：浅色/深色底 + 小尺寸效果
    sheet = Image.new("RGB", (S * 3 + 80, S + 260), (245, 246, 250))
    sheet.paste(icon_squircle, (20, 20), icon_squircle)
    sheet.paste(icon_circle, (S + 40, 20), icon_circle)
    dark = Image.new("RGB", (S, S), (18, 20, 26))
    dark.paste(icon_squircle, (0, 0), icon_squircle)
    sheet.paste(dark, (S * 2 + 60, 20))

    # 尺寸阶梯：验证小尺寸下是否还认得出
    x = 20
    for size in (96, 72, 56, 40, 28):
        small = icon_squircle.resize((size, size), Image.LANCZOS)
        sheet.paste(small, (x, S + 30), small)
        x += size + 16

    # 无背景前景层（monochrome 主题图标效果）
    mono = Image.new("RGB", (S, S), (32, 34, 42))
    mono.paste(fg_only.convert("RGB"), (0, 0), fg_only.getchannel("A"))
    sheet.paste(mono, (x + 10, S + 20))

    icon_squircle.save(out_dir / "icon_preview_squircle.png")
    icon_circle.save(out_dir / "icon_preview_circle.png")
    fg_only.save(out_dir / "icon_preview_foreground.png")
    sheet.save(out_dir / "icon_preview_sheet.png")
    print(f"已生成预览: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
