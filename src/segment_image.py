from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import os
import cv2



def generate_segmentated_img(img_path):
    config_dir = 'config'
    config_path = os.path.join(config_dir,'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py')
    checkpoint_path = os.path.join(config_dir,'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth')

    # build the model from a config file and a checkpoint file
    model = init_model(config_path, checkpoint_path, device='cuda:0')

    # test a single image and show the results
    result = inference_model(model, img_path)
    vis_image = show_result_pyplot(model, img_path, result,opacity=0.5,show=False)
    print(vis_image.shape)
    return vis_image

