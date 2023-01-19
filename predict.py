#-------------------------------------#
#   运行predict.py可以生成图片
#   生成1x1的图片和5x5的图片
#-------------------------------------#
from ddpm import Diffusion

if __name__ == "__main__":
    save_path_5x5 = "results/predict_out/predict_5x5_results.png"
    save_path_1x1 = "results/predict_out/predict_1x1_results.png"

    ddpm = Diffusion()
    while True:
        img = input('Just Click Enter~')
        print("Generate_1x1_image")
        ddpm.generate_1x1_image(save_path_1x1)
        print("Generate_1x1_image Done")
        
        print("Generate_5x5_image")
        ddpm.generate_5x5_image(save_path_5x5)
        print("Generate_5x5_image Done")