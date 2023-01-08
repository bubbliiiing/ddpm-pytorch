#-------------------------------------#
#   运行predict.py可以Diffusion图片
#   Diffusion1x1的图片和5x5的图片
#-------------------------------------#
from ddpm import Diffusion

if __name__ == "__main__":
    save_path_5x5 = "results/predict_out/predict_5x5_results.png"
    save_path_1x1 = "results/predict_out/predict_1x1_results.png"

    ddpm = Diffusion()
    while True:
        img = input('Just Click Enter~')
        ddpm.generate_1x1_image(save_path_1x1)
        ddpm.generate_5x5_image(save_path_5x5)