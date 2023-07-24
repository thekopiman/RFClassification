from PIL import Image, ImageDraw


class Drawing:
    def __init__(self) -> None:
        pass

    @classmethod
    def show_box(self, image_path, label_path):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        with open(label_path, "r") as f:
            for line in f.readlines():
                label, x, y, w, h = line.split(" ")

                x = float(x)
                y = float(y)
                w = float(w)
                h = float(h)

                W, H = image.size
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H

                draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 1), width=3)

        return image
