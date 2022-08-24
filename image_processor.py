import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def image_processor(img, final_dim=128):
    img = Image.open(img)

    img_size = img.size
    max_dim = max(img.size)
    sf = final_dim/max_dim

    new_img_size = (int(img_size[0]*sf), int(img_size[1]*sf))
    new_img = img.resize(new_img_size)

    final_img = Image.new(mode='RGB', size=(final_dim, final_dim))
    final_img.paste(new_img, ((final_dim-new_img_size[0])//2, (final_dim-new_img_size[1])//2))

    output = transforms.ToTensor()(final_img)
    
    output = torch.unsqueeze(output,dim=0)

    return output

if __name__ == "__main__":
    output = image_processor('images/test/baby-kids-stuff/fe27910d-b6c8-45fe-a2cc-ad1dad4de8b5.jpg')
    print(output.size())
    plt.imshow(transforms.ToPILImage()(torch.squeeze(output)), interpolation="bicubic")
    plt.show()
