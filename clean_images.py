# %%
# IMPORT LIBRARIES
from PIL import Image
import glob

print('Done')

# %%
# LIST IMAGE FILES

list_img = glob.glob('images/*.jpg')

print('Done')

# %%
# RESIZE IMAGE

final_dim = 128
for img in list_img:
    img_name = img.split('/')[-1]

    img = Image.open(img)
    img_size = img.size
    max_dim = max(img.size)
    sf = final_dim/max_dim

    new_img_size = (int(img_size[0]*sf), int(img_size[1]*sf))
    new_img = img.resize(new_img_size)

    final_img = Image.new(mode='RGB', size=(final_dim, final_dim))
    final_img.paste(new_img, ((final_dim-new_img_size[0])//2, (final_dim-new_img_size[1])//2))
    final_img.save(f'clean_images/{img_name}')

print('Done')

# %%
