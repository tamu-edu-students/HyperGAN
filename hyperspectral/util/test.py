import processor
import eval_metrics
#from cuvis.Export import EnviExporter

if __name__== '__main__':
    sys_root = './Dataset/'
    im_name = 'indian_pines'

    img_path = sys_root + im_name + '.tif'
    print(img_path)
    #e = EnviExporter()
    p = processor.Processor()
    img = p.prepare_data(img_path)



    #p.display_band(img, 40)
    # p.display_band(img, 56)
    print(eval_metrics.correlation(img[10], img[200]))
    # print(eval_metrics.PSNR(img[10], img[11]))
    
    # p.genFalseRGB(36, 18, 10, img)

    # for i in range(4, 210):
    #     print(eval_metrics.SSIM(img[4], img[i]))
