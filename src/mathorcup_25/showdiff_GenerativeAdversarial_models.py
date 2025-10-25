import os

import matplotlib.pyplot as plt

PICTURE_ROUTES = [
    r"data-bin\ori-data\images\test",
    r"output-bin\pic_realesr-animevideov3",
    r"output-bin\pic_realesrgan-x4plus",
    r"output-bin\pic_realesrgan-x4plus-anime",
]
ROUTE_TITLES = [
    "Original",
    "RealESR-AnimeVideoV3",
    "RealESRGAN-x4plus",
    "RealESRGAN-x4plus-anime",
]
output_dir = r"output-bin\pic_show_diff"


def main():
    # 获取每个路径下的图片列表
    pic_lists = []
    for route in PICTURE_ROUTES:
        pics = sorted([f for f in os.listdir(route) if f.endswith((".jpg", ".png"))])
        pic_lists.append([os.path.join(route, p) for p in pics])

    pic_num = min(len(p) for p in pic_lists)
    os.makedirs(output_dir, exist_ok=True)

    for j in range(pic_num):
        plt.figure(figsize=(15, 10))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(plt.imread(pic_lists[i][j]))
            plt.title(ROUTE_TITLES[i])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{j}.png"))
        plt.close()


if __name__ == "__main__":
    main()
